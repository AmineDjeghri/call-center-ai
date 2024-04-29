# First imports, to make sure the following logs are first
from helpers.logging import build_logger
from helpers.config import CONFIG


_logger = build_logger(__name__)
_logger.info(f"call-center-ai v{CONFIG.version}")


# General imports
from typing import Any, Optional, Union, Tuple
from azure.core.messaging import CloudEvent
from azure.eventgrid import EventGridEvent, SystemEventNames
from fastapi import FastAPI, status, Request, HTTPException, BackgroundTasks, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from helpers.pydantic_types.phone_numbers import PhoneNumber
from jinja2 import Environment, FileSystemLoader
from models.call import CallStateModel, CallInitiateModel, CallGetModel
from models.next import ActionEnum as NextActionEnum
from urllib.parse import quote_plus, urljoin
from uuid import UUID
import asyncio
import mistune
from helpers.call_events import (
    on_call_connected,
    on_call_disconnected,
    on_ivr_recognized,
    on_incoming_call,
    on_play_completed,
    on_play_error,
    on_speech_recognized,
    on_speech_timeout_error,
    on_speech_unknown_error,
    on_transfer_completed,
    on_transfer_error,
)
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from models.readiness import ReadinessModel, ReadinessCheckModel, ReadinessStatus


# Jinja configuration
_jinja = Environment(
    autoescape=True,
    enable_async=True,
    loader=FileSystemLoader("public_website"),
)
# Jinja custom functions
_jinja.filters["quote_plus"] = lambda x: quote_plus(str(x)) if x else ""
_jinja.filters["markdown"] = lambda x: mistune.create_markdown(escape=False, plugins=["abbr", "speedup", "url"])(x) if x else ""  # type: ignore

# Persistences
_cache = CONFIG.cache.instance()
_db = CONFIG.database.instance()
_search = CONFIG.ai_search.instance()
_sms = CONFIG.sms.instance()
_voice = CONFIG.voice.instance()

# FastAPI
_logger.info(f'Using root path "{CONFIG.api.root_path}"')
api = FastAPI(
    contact={
        "url": "https://github.com/clemlesne/call-center-ai",
    },
    description="AI-powered call center solution with Azure and OpenAI GPT.",
    license_info={
        "name": "Apache-2.0",
        "url": "https://github.com/clemlesne/call-center-ai/blob/master/LICENCE",
    },
    root_path=CONFIG.api.root_path,
    title="call-center-ai",
    version=CONFIG.version,
)

# OpenTelemetry
FastAPIInstrumentor.instrument_app(api)


assert CONFIG.api.events_domain, "api.events_domain config is not set"
_CALL_EVENT_URL = urljoin(
    str(CONFIG.api.events_domain),
    "/call/communication-services/{call_id}/{callback_secret}",
)
_logger.info(f"Using call event URL {_CALL_EVENT_URL}")


@api.get(
    "/health/liveness",
    status_code=status.HTTP_204_NO_CONTENT,
    description="Liveness healthckeck, always returns 204, used to check if the API is up.",
    name="Get liveness",
)
async def health_liveness_get() -> None:
    pass


@api.get(
    "/health/readiness",
    description="Readiness healthckeck, returns the status of all components, and fails if one of them is not ready. If all components are ready, returns 200, otherwise 503.",
    name="Get readiness",
)
async def health_readiness_get() -> JSONResponse:
    # Check all components in parallel
    cache_check, db_check, search_check, sms_check, voice_check = await asyncio.gather(
        _cache.areadiness(),
        _db.areadiness(),
        _search.areadiness(),
        _sms.areadiness(),
        _voice.areadiness(),
    )
    readiness = ReadinessModel(
        status=ReadinessStatus.OK,
        checks=[
            ReadinessCheckModel(id="cache", status=cache_check),
            ReadinessCheckModel(id="index", status=db_check),
            ReadinessCheckModel(id="startup", status=ReadinessStatus.OK),
            ReadinessCheckModel(id="store", status=search_check),
            ReadinessCheckModel(id="stream", status=sms_check),
            ReadinessCheckModel(id="voice", status=voice_check),
        ],
    )
    # If one of the checks fails, the whole readiness fails
    status_code = status.HTTP_200_OK
    for check in readiness.checks:
        if check.status != ReadinessStatus.OK:
            readiness.status = ReadinessStatus.FAIL
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            break
    return JSONResponse(
        content=jsonable_encoder(readiness),
        status_code=status_code,
    )


# Serve static files
api.mount("/static", StaticFiles(directory="public_website/static"))


@api.get(
    "/report/{phone_number}",
    description="Display the history of calls in a web page.",
    name="Get call history",
)
async def report_history_get(phone_number: PhoneNumber) -> HTMLResponse:
    calls = await _db.call_asearch_all(phone_number) or []

    template = _jinja.get_template("history.html.jinja")
    render = await template.render_async(
        calls=calls,
        phone_number=phone_number,
        version=CONFIG.version,
    )
    return HTMLResponse(content=render)


@api.get(
    "/report/{phone_number}/{call_id}",
    description="Display the call report in a web page.",
    name="Get call report",
)
async def report_call_get(phone_number: PhoneNumber, call_id: UUID) -> HTMLResponse:
    call = await _db.call_aget(call_id)
    if not call or call.initiate.phone_number != phone_number:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Call {call_id} for phone number {phone_number} not found",
        )

    template = _jinja.get_template("report.html.jinja")
    render = await template.render_async(
        bot_company=call.initiate.bot_company,
        bot_name=call.initiate.bot_name,
        call=call,
        next_actions=[action for action in NextActionEnum],
        version=CONFIG.version,
    )
    return HTMLResponse(content=render)


@api.get(
    "/call",
    description="Search all calls by phone number.",
    name="Search calls",
)
async def call_search_get(phone_number: PhoneNumber) -> list[CallGetModel]:
    calls = await _db.call_asearch_all(phone_number) or []
    output = [CallGetModel.model_validate(call) for call in calls]
    return output


@api.get(
    "/call/{call_id}",
    description="Get a call by its ID.",
    name="Get call",
)
async def call_get(call_id: UUID) -> CallGetModel:
    call = await _db.call_aget(call_id)
    if not call:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Call {call_id} not found",
        )
    return CallGetModel.model_validate(call)


@api.post(
    "/call",
    description="Initiate an outbound call to a phone number.",
    name="Create call",
)
async def call_post(
    initiate: CallInitiateModel,
    background_tasks: BackgroundTasks,
) -> CallGetModel:
    _logger.info(f"Initiating outbound call to {initiate.phone_number}")
    url, call = await _callback_url(initiate.phone_number, initiate)
    await _voice.acreate(
        background_tasks=background_tasks,
        call=call,
        callback_url=url,
        phone_number=initiate.phone_number,
    )
    return CallGetModel.model_validate(call)


@api.post(
    "/call/eventgrid",
    description="Handle incoming call from a Azure Event Grid event originating from Azure Communication Services.",
    name="Create Event Grid event",
)
async def call_eventgrid_post(
    request: Request,
    background_tasks: BackgroundTasks,
) -> Response:
    responses = await asyncio.gather(
        *[
            _call_eventgrid_worker(event_dict, background_tasks)
            for event_dict in await request.json()
        ]
    )
    for response in responses:
        if response:
            return response
    return Response(status_code=status.HTTP_204_NO_CONTENT)


async def _call_eventgrid_worker(
    event_dict: dict[str, Any],
    background_tasks: BackgroundTasks,
) -> Optional[Union[JSONResponse, Response]]:
    event = EventGridEvent.from_dict(event_dict)
    event_type = event.event_type

    _logger.debug(f"Call inbound event {event_type} with data {event.data}")

    if event_type == SystemEventNames.EventGridSubscriptionValidationEventName:
        validation_code = event.data["validationCode"]
        _logger.info(f"Validating Event Grid subscription ({validation_code})")
        return JSONResponse(
            content={"validationResponse": event.data["validationCode"]},
            status_code=status.HTTP_200_OK,
        )

    elif event_type == SystemEventNames.AcsIncomingCallEventName:
        if event.data["from"]["kind"] == "phoneNumber":
            phone_number = event.data["from"]["phoneNumber"]["value"]
        else:
            phone_number = event.data["from"]["rawId"]
        call_context = event.data["incomingCallContext"]

        url, call = await _callback_url(phone_number)
        event_status = await on_incoming_call(
            background_tasks=background_tasks,
            call=call,
            callback_url=url,
            incoming_context=call_context,
            phone_number=phone_number,
        )

        if not event_status:
            return Response(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        return None


@api.post(
    "/call/communication-services/{call_id}/{secret}",
    description="Handle callbacks from Azure Communication Services.",
    name="Create Communication Services event",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def call_communicationservices_post(
    request: Request,
    background_tasks: BackgroundTasks,
    call_id: UUID,
    secret: str,
) -> None:
    await asyncio.gather(
        *[
            _call_communicationservices_worker(
                background_tasks, event_dict, call_id, secret
            )
            for event_dict in await request.json()
        ]
    )


async def _call_communicationservices_worker(
    background_tasks: BackgroundTasks,
    event_dict: dict,
    call_id: UUID,
    secret: str,
) -> None:
    call = await _db.call_aget(call_id)
    if not call:
        _logger.warning(f"Call {call_id} not found")
        return
    if call.callback_secret != secret:
        _logger.warning(f"Secret for call {call_id} does not match")
        return

    event = CloudEvent.from_dict(event_dict)
    assert isinstance(event.data, dict)

    operation_context: Optional[str] = event.data.get("operationContext", None)
    event_type = event.type

    _logger.debug(f"Call event received {event_type} for call {call}")
    _logger.debug(event.data)

    if event_type == "Microsoft.Communication.CallConnected":  # Call answered
        background_tasks.add_task(
            on_call_connected,
            background_tasks=background_tasks,
            call=call,
        )

    elif event_type == "Microsoft.Communication.CallDisconnected":  # Call hung up
        background_tasks.add_task(
            on_call_disconnected,
            background_tasks=background_tasks,
            call=call,
        )

    elif (
        event_type == "Microsoft.Communication.RecognizeCompleted"
    ):  # Speech recognized
        recognition_result = event.data["recognitionType"]

        if recognition_result == "speech":  # Handle voice
            speech_text = event.data["speechResult"]["speech"]
            if (
                speech_text != None and len(speech_text) > 0
            ):  # TODO: Is this check necessary?
                background_tasks.add_task(
                    on_speech_recognized,
                    background_tasks=background_tasks,
                    call=call,
                    text=speech_text,
                )

        elif recognition_result == "choices":  # Handle IVR
            label_detected = event.data["choiceResult"]["label"]
            background_tasks.add_task(
                on_ivr_recognized,
                background_tasks=background_tasks,
                call=call,
                label=label_detected,
            )

    elif (
        event_type == "Microsoft.Communication.RecognizeFailed"
    ):  # Speech recognition failed
        result_information = event.data["resultInformation"]
        error_code = result_information["subCode"]

        # Error codes:
        # 8510 = Action failed, initial silence timeout reached
        # 8532 = Action failed, inter-digit silence timeout reached
        # See: https://github.com/MicrosoftDocs/azure-docs/blob/main/articles/communication-services/how-tos/call-automation/recognize-action.md#event-codes
        if error_code in (8510, 8532):  # Timeout retry
            background_tasks.add_task(
                on_speech_timeout_error,
                background_tasks=background_tasks,
                call=call,
            )
        else:  # Unknown error
            background_tasks.add_task(
                on_speech_unknown_error,
                background_tasks=background_tasks,
                call=call,
                error_code=error_code,
            )

    elif event_type == "Microsoft.Communication.PlayCompleted":  # Media played
        background_tasks.add_task(
            on_play_completed,
            background_tasks=background_tasks,
            call=call,
            context=operation_context,
        )

    elif event_type == "Microsoft.Communication.PlayFailed":  # Media play failed
        result_information = event.data["resultInformation"]
        error_code = result_information["subCode"]
        background_tasks.add_task(
            on_play_error,
            error_code=error_code,
        )

    elif (
        event_type == "Microsoft.Communication.CallTransferAccepted"
    ):  # Call transfer accepted
        background_tasks.add_task(
            on_transfer_completed,
            background_tasks=background_tasks,
            call=call,
        )

    elif (
        event_type == "Microsoft.Communication.CallTransferFailed"
    ):  # Call transfer failed
        result_information = event.data["resultInformation"]
        sub_code = result_information["subCode"]
        background_tasks.add_task(
            on_transfer_error,
            background_tasks=background_tasks,
            call=call,
            error_code=sub_code,
        )

    await _db.call_aset(
        call
    )  # TODO: Do not persist on every event, this is simpler but not efficient


async def _callback_url(
    phone_number: PhoneNumber, initiate: Optional[CallInitiateModel] = None
) -> Tuple[str, CallStateModel]:
    """
    Generate the callback URL for a call.

    If the caller has already called, use the same call ID, to keep the conversation history. Otherwise, create a new call ID.
    """
    call = await _db.call_asearch_one(phone_number)
    if not call:
        initiate = initiate or CallInitiateModel(
            **CONFIG.workflow.default_initiate.model_dump(),
            phone_number=phone_number,  # type: ignore
        )
        call = CallStateModel(initiate=initiate)
        await _db.call_aset(call)  # Create for the first time
    url = _CALL_EVENT_URL.format(
        callback_secret=call.callback_secret,
        call_id=str(call.call_id),
    )
    return url, call
