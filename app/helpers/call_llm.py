import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from functools import wraps

from aiojobs import Job, Scheduler
from azure.cognitiveservices.speech import (
    SpeechSynthesizer,
)
from azure.cognitiveservices.speech.audio import PushAudioInputStream
from azure.communication.callautomation.aio import CallAutomationClient
from openai import APIError
from pydub import AudioSegment
from pydub.effects import (
    high_pass_filter,
    low_pass_filter,
)
from webrtcvad import Vad

from app.helpers.call_utils import (
    handle_media,
    handle_realtime_tts,
    tts_sentence_split,
    use_stt_client,
    use_tts_client,
)
from app.helpers.config import CONFIG
from app.helpers.features import (
    answer_hard_timeout_sec,
    answer_soft_timeout_sec,
    phone_silence_timeout_sec,
    vad_cutoff_timeout_ms,
    vad_silence_timeout_ms,
)
from app.helpers.llm_tools import DefaultPlugin
from app.helpers.llm_worker import (
    MaximumTokensReachedError,
    SafetyCheckError,
    completion_stream,
)
from app.helpers.logging import logger
from app.helpers.monitoring import CallAttributes, span_attribute, tracer
from app.models.call import CallStateModel
from app.models.message import (
    ActionEnum as MessageAction,
    MessageModel,
    PersonaEnum as MessagePersonaEnum,
    StyleEnum as MessageStyleEnum,
    ToolModel as MessageToolModel,
    extract_message_style,
    remove_message_action,
)

_db = CONFIG.database.instance()


# TODO: Refacto, this function is too long
@tracer.start_as_current_span("call_load_llm_chat")
async def load_llm_chat(  # noqa: PLR0913
    audio_bits_per_sample: int,
    audio_channels: int,
    audio_in: asyncio.Queue[bytes],
    audio_out: asyncio.Queue[bytes | bool],
    audio_sample_rate: int,
    automation_client: CallAutomationClient,
    call: CallStateModel,
    post_callback: Callable[[CallStateModel], Awaitable[None]],
    scheduler: Scheduler,
    training_callback: Callable[[CallStateModel], Awaitable[None]],
) -> None:
    # Init language recognition
    stt_buffer: list[str] = []
    stt_complete_gate = asyncio.Event()

    def _stt_callback(text: str) -> None:
        # Skip if no text
        if not text:
            return

        stt_buffer.append(text)
        logger.debug("Complete recognition: %s", stt_buffer)

        # Open the recognition gate
        stt_complete_gate.set()

    async with (
        use_stt_client(
            audio_bits_per_sample=audio_bits_per_sample,
            audio_channels=audio_channels,
            audio_sample_rate=audio_sample_rate,
            call=call,
            callback=_stt_callback,
        ) as stt_stream,
        use_tts_client(
            audio=audio_out,
            call=call,
        ) as tts_client,
    ):
        # Build scheduler
        last_response: Job | None = None

        async def _timeout_callback() -> None:
            from app.helpers.call_events import on_realtime_recognize_error

            logger.info("Phone silence timeout triggered")

            # Execute business logic
            await scheduler.spawn(
                on_realtime_recognize_error(
                    call=call,
                    client=automation_client,
                    post_callback=post_callback,
                    scheduler=scheduler,
                    tts_client=tts_client,
                )
            )

        async def _clear_audio_callback() -> None:
            # Stop TTS, clear the buffer and send a stop signal
            tts_client.stop_speaking_async()
            while not audio_out.empty():
                audio_out.get_nowait()
                audio_out.task_done()
            await audio_out.put(False)

            # Close the recognition gate
            stt_complete_gate.clear()

            # Close previous response if any
            if last_response:
                await scheduler.spawn(last_response.close(timeout=0))

            # Clear the recognition buffer
            stt_buffer.clear()

        async def _response_callback() -> None:
            # Wait for the complete recognition
            await stt_complete_gate.wait()

            stt_text = " ".join(stt_buffer).strip()

            # Skip if no partial recognition
            if not stt_text:
                return

            # Add it to the call history and update last interaction
            logger.info("Voice stored: %s", stt_buffer)
            async with _db.call_transac(
                call=call,
                scheduler=scheduler,
            ):
                call.last_interaction_at = datetime.now(UTC)
                call.messages.append(
                    MessageModel(
                        content=stt_text,
                        persona=MessagePersonaEnum.HUMAN,
                    )
                )

            # Clear the recognition buffer
            stt_buffer.clear()

            # Store recognitio task
            nonlocal last_response
            last_response = await scheduler.spawn(
                _out_answer(
                    call=call,
                    client=automation_client,
                    post_callback=post_callback,
                    scheduler=scheduler,
                    training_callback=training_callback,
                    tts_client=tts_client,
                )
            )

            # Wait for the response to be processed
            await last_response.wait()

        await _in_audio(
            bits_per_sample=audio_bits_per_sample,
            call=call,
            channels=audio_channels,
            clear_audio_callback=_clear_audio_callback,
            in_stream=audio_in,
            out_stream=stt_stream,
            response_callback=_response_callback,
            sample_rate=audio_sample_rate,
            timeout_callback=_timeout_callback,
        )


# TODO: Refacto, this function is too long (and remove PLR0912/PLR0915 ignore)
@tracer.start_as_current_span("call_load_out_answer")
async def _out_answer(  # noqa: PLR0915, PLR0913
    call: CallStateModel,
    client: CallAutomationClient,
    post_callback: Callable[[CallStateModel], Awaitable[None]],
    scheduler: Scheduler,
    training_callback: Callable[[CallStateModel], Awaitable[None]],
    tts_client: SpeechSynthesizer,
    _iterations_remaining: int = 3,
) -> CallStateModel:
    """
    Handle the intelligence of the call, including: LLM chat, TTS, and media play.

    Play the loading sound while waiting for the intelligence to be processed. If the intelligence is not processed after few secs, play the timeout sound. If the intelligence is not processed after more secs, stop the intelligence processing and play the error sound.

    Returns the updated call model.
    """
    # Add span attributes
    span_attribute(CallAttributes.CALL_CHANNEL, "voice")
    span_attribute(CallAttributes.CALL_MESSAGE, call.messages[-1].content)

    # Reset recognition retry counter
    async with _db.call_transac(
        call=call,
        scheduler=scheduler,
    ):
        call.recognition_retry = 0

    # By default, play the loading sound
    play_loading_sound = True

    async def _tts_callback(text: str, style: MessageStyleEnum) -> None:
        """
        Send back the TTS to the user.
        """
        nonlocal play_loading_sound
        # For first TTS, interrupt loading sound and disable loading it
        if play_loading_sound:
            play_loading_sound = False
        # Play the TTS
        await scheduler.spawn(
            handle_realtime_tts(
                call=call,
                scheduler=scheduler,
                style=style,
                text=text,
                tts_client=tts_client,
            )
        )

    # Chat
    chat_task = asyncio.create_task(
        _execute_llm_chat(
            call=call,
            client=client,
            post_callback=post_callback,
            scheduler=scheduler,
            tts_callback=_tts_callback,
            tts_client=tts_client,
            use_tools=_iterations_remaining > 0,
        )
    )

    # Loading
    def _loading_task() -> asyncio.Task:
        return asyncio.create_task(asyncio.sleep(loading_timer))

    loading_timer = 5  # Play loading sound every 5 secs
    loading_task = _loading_task()

    # Timeouts
    soft_timeout_triggered = False
    soft_timeout_task = asyncio.create_task(
        asyncio.sleep(await answer_soft_timeout_sec())
    )
    hard_timeout_task = asyncio.create_task(
        asyncio.sleep(await answer_hard_timeout_sec())
    )

    def _clear_tasks() -> None:
        chat_task.cancel()
        hard_timeout_task.cancel()
        loading_task.cancel()
        soft_timeout_task.cancel()

    is_error = True
    continue_chat = True
    try:
        while True:
            # logger.debug("Chat task status: %s", chat_task.done())

            # Break when chat coroutine is done
            if chat_task.done():
                # Clean up
                _clear_tasks()
                # Get result
                is_error, continue_chat, call = (
                    chat_task.result()
                )  # Store updated chat model
                await training_callback(call)  # Trigger trainings generation
                break

            # Break when hard timeout is reached
            if hard_timeout_task.done():
                logger.warning(
                    "Hard timeout of %ss reached",
                    await answer_hard_timeout_sec(),
                )
                # Clean up
                _clear_tasks()
                break

            # Catch timeout if async loading is not started
            if play_loading_sound:
                # Speak when soft timeout is reached
                if soft_timeout_task.done() and not soft_timeout_triggered:
                    logger.warning(
                        "Soft timeout of %ss reached",
                        await answer_soft_timeout_sec(),
                    )
                    soft_timeout_triggered = True
                    # Never store the error message in the call history, it has caused hallucinations in the LLM
                    await scheduler.spawn(
                        handle_realtime_tts(
                            call=call,
                            scheduler=scheduler,
                            store=False,
                            text=await CONFIG.prompts.tts.timeout_loading(call),
                            tts_client=tts_client,
                        )
                    )

                # Do not play timeout prompt plus loading, it can be frustrating for the user
                elif loading_task.done():
                    loading_task = _loading_task()
                    await scheduler.spawn(
                        handle_media(
                            call=call,
                            client=client,
                            sound_url=CONFIG.prompts.sounds.loading(),
                        )
                    )

            # Wait to not block the event loop for other requests
            await asyncio.sleep(1)

    except Exception:
        # TODO: Remove last message
        logger.exception("Error loading intelligence")

    # Error during chat
    if is_error:
        # Maximum retries reached
        if not continue_chat or _iterations_remaining < 1:
            logger.warning("Maximum retries reached, stopping chat")
            content = await CONFIG.prompts.tts.error(call)
            # Speak the error
            await _tts_callback(content, MessageStyleEnum.NONE)
            # Never store the error message in the call history, it has caused hallucinations in the LLM

        # Retry chat after an error
        else:
            logger.info("Retrying chat, %s remaining", _iterations_remaining - 1)
            return await _out_answer(
                call=call,
                client=client,
                post_callback=post_callback,
                scheduler=scheduler,
                training_callback=training_callback,
                tts_client=tts_client,
                _iterations_remaining=_iterations_remaining - 1,
            )

    # Contiue chat
    elif continue_chat and _iterations_remaining > 0:
        logger.info("Continuing chat, %s remaining", _iterations_remaining - 1)
        return await _out_answer(
            call=call,
            client=client,
            post_callback=post_callback,
            scheduler=scheduler,
            training_callback=training_callback,
            tts_client=tts_client,
            _iterations_remaining=_iterations_remaining - 1,
        )  # Recursive chat (like for for retry or tools)

        # End chat
        # TODO: Re-implement

    return call


# TODO: Refacto, this function is too long
@tracer.start_as_current_span("call_execute_llm_chat")
async def _execute_llm_chat(  # noqa: PLR0913, PLR0911, PLR0912, PLR0915
    call: CallStateModel,
    client: CallAutomationClient,
    post_callback: Callable[[CallStateModel], Awaitable[None]],
    scheduler: Scheduler,
    tts_callback: Callable[[str, MessageStyleEnum], Awaitable[None]],
    tts_client: SpeechSynthesizer,
    use_tools: bool,
) -> tuple[bool, bool, CallStateModel]:
    """
    Perform the chat with the LLM model.

    This function will handle:

    - The chat with the LLM model (incl system prompts, tools, and user callback)
    - Retry as possible if the LLM model fails to return a response

    Returns a tuple with:

    1. `bool`, notify error
    2. `bool`, should retry chat
    3. `CallStateModel`, the updated model
    """
    logger.debug("Running LLM chat")
    content_full = ""

    async def _plugin_tts_callback(text: str) -> None:
        nonlocal content_full
        content_full += f" {text}"
        await tts_callback(text, MessageStyleEnum.NONE)

    async def _content_callback(buffer: str) -> None:
        # Remove tool calls from buffer content and detect style
        style, local_content = extract_message_style(remove_message_action(buffer))
        await tts_callback(local_content, style)

    # Build RAG
    trainings = await call.trainings()
    logger.info("Enhancing LLM chat with %s trainings", len(trainings))
    # logger.debug("Trainings: %s", trainings)

    # System prompts
    system = CONFIG.prompts.llm.chat_system(
        call=call,
        trainings=trainings,
    )

    # Build plugins
    plugins = DefaultPlugin(
        call=call,
        client=client,
        post_callback=post_callback,
        scheduler=scheduler,
        tts_callback=_plugin_tts_callback,
        tts_client=tts_client,
    )

    tools = []
    if not use_tools:
        logger.warning("Tools disabled for this chat")
    else:
        tools = await plugins.to_openai()
        # logger.debug("Tools: %s", tools)

    # Execute LLM inference
    maximum_tokens_reached = False
    content_buffer_pointer = 0
    tool_calls_buffer: dict[int, MessageToolModel] = {}
    try:
        async for delta in completion_stream(
            max_tokens=160,  # Lowest possible value for 90% of the cases, if not sufficient, retry will be triggered, 100 tokens ~= 75 words, 20 words ~= 1 sentence, 6 sentences ~= 160 tokens
            messages=call.messages,
            system=system,
            tools=tools,
        ):
            if not delta.content:
                for piece in delta.tool_calls or []:
                    tool_calls_buffer[piece.index] = tool_calls_buffer.get(
                        piece.index, MessageToolModel()
                    )
                    tool_calls_buffer[piece.index] += piece
            else:
                # Store whole content
                content_full += delta.content
                for sentence, length in tts_sentence_split(
                    content_full[content_buffer_pointer:], False
                ):
                    content_buffer_pointer += length
                    await _content_callback(sentence)

    # Retry on maximum tokens reached
    except MaximumTokensReachedError:
        logger.warning("Maximum tokens reached for this completion, retry asked")
        maximum_tokens_reached = True
    # Retry on API error
    except APIError as e:
        logger.warning("OpenAI API call error: %s", e)
        return True, True, call  # Error, retry
    # Last user message is trash, remove it
    except SafetyCheckError as e:
        logger.warning("Safety Check error: %s", e)
        # Remove last user message
        if last_message := next(
            (
                call
                for call in reversed(call.messages)
                if call.persona == MessagePersonaEnum.HUMAN
                and call.action in [MessageAction.SMS, MessageAction.TALK]
            ),
            None,
        ):
            call.messages.remove(last_message)
        return True, False, call  # Error, no retry

    # Flush the remaining buffer
    if content_buffer_pointer < len(content_full):
        await _content_callback(content_full[content_buffer_pointer:])

    # Convert tool calls buffer
    tool_calls = [tool_call for _, tool_call in tool_calls_buffer.items()]

    # Delete action and style from the message as they are in the history and LLM hallucinates them
    last_style, content_full = extract_message_style(
        remove_message_action(content_full)
    )

    logger.debug("Chat response: %s", content_full)
    logger.debug("Tool calls: %s", tool_calls)

    # OpenAI GPT-4 Turbo sometimes return wrong tools schema, in that case, retry within limits
    # TODO: Tries to detect this error earlier
    # See: https://community.openai.com/t/model-tries-to-call-unknown-function-multi-tool-use-parallel/490653
    if any(
        tool_call.function_name == "multi_tool_use.parallel" for tool_call in tool_calls
    ):
        logger.warning('LLM send back invalid tool schema "multi_tool_use.parallel"')
        return True, True, call  # Error, retry

    # OpenAI GPT-4 Turbo tends to return empty content, in that case, retry within limits
    if not content_full and not tool_calls:
        logger.warning("Empty content, retrying")
        return True, True, call  # Error, retry

    # Execute tools
    tool_tasks = [tool_call.execute_function(plugins) for tool_call in tool_calls]
    await asyncio.gather(*tool_tasks)
    call = plugins.call  # Update call model if object reference changed

    # Store message
    async with _db.call_transac(
        call=call,
        scheduler=scheduler,
    ):
        call.messages.append(
            MessageModel(
                content="",  # Content has already been stored within the TTS callback
                persona=MessagePersonaEnum.ASSISTANT,
                style=last_style,
                tool_calls=tool_calls,
            )
        )

    # Recusive call if needed
    if tool_calls:
        return False, True, call
    # Retry if maximum tokens reached
    if maximum_tokens_reached:
        return False, True, call  # TODO: Should we notify an error?
    # No error, no retry
    return False, False, call


# TODO: Refacto and simplify
async def _in_audio(  # noqa: PLR0913
    bits_per_sample: int,
    call: CallStateModel,
    channels: int,
    clear_audio_callback: Callable[[], Awaitable[None]],
    in_stream: asyncio.Queue[bytes],
    out_stream: PushAudioInputStream,
    response_callback: Callable[[], Awaitable[None]],
    sample_rate: int,
    timeout_callback: Callable[[], Awaitable[None]],
) -> None:
    clear_tts_task: asyncio.Task | None = None
    silence_task: asyncio.Task | None = None
    vad = Vad(
        # Aggressiveness mode (0, 1, 2, or 3)
        # Sets the VAD operating mode. A more aggressive (higher mode) VAD is more restrictive in reporting speech. Put in other words the probability of being speech when the VAD returns 1 is increased with increasing mode. As a consequence also the missed detection rate goes up.
        mode=3,
    )

    async def _silence_callback() -> None:
        """
        Flush the audio buffer if no audio is detected for a while and trigger the timeout if required.
        """
        # Wait before flushing
        nonlocal clear_tts_task
        timeout_ms = await vad_silence_timeout_ms()
        await asyncio.sleep(timeout_ms / 1000)

        # Cancel the clear TTS task if any
        if clear_tts_task:
            clear_tts_task.cancel()
            clear_tts_task = None

        # Flush the audio buffer
        logger.debug("Flushing audio buffer after %i ms", timeout_ms)
        await response_callback()

        # Wait for silence and trigger timeout
        timeout_sec = await phone_silence_timeout_sec()
        while True:
            # Stop this time if the call played a message
            timeout_start = datetime.now(UTC)
            await asyncio.sleep(timeout_sec)

            # Stop if the call ended
            if not call.in_progress:
                break

            # Cancel if an interaction happened in the meantime
            if (
                call.last_interaction_at
                and call.last_interaction_at + timedelta(seconds=timeout_sec)
                > timeout_start
            ):
                logger.debug(
                    "Message sent in the meantime, canceling this silence timeout"
                )
                continue

            # Trigger the timeout
            logger.info("Silence triggered after %i sec", timeout_sec)
            await timeout_callback()

    async def _clear_tts_callback() -> None:
        """
        Clear the TTS queue.

        Start is the index of the buffer where the TTS was triggered.
        """
        timeout_ms = await vad_cutoff_timeout_ms()

        # Wait before clearing the TTS queue
        await asyncio.sleep(timeout_ms / 1000)

        logger.debug("Canceling TTS after %i ms", timeout_ms)

        # Clear the queue
        await clear_audio_callback()

    # Consumes audio stream
    while True:
        # Wait for the next audio packet
        in_chunck = await in_stream.get()

        # Load audio
        in_audio: AudioSegment = AudioSegment(
            channels=channels,
            data=in_chunck,
            frame_rate=sample_rate,
            sample_width=bits_per_sample // 8,
        )

        # Apply high-pass and low-pass filters in a simple attempt to reduce noise
        in_audio = high_pass_filter(seg=in_audio, cutoff=85)
        in_audio = low_pass_filter(seg=in_audio, cutoff=3000)

        # Always add the audio to the buffer
        assert isinstance(in_audio.raw_data, bytes)
        out_stream.write(in_audio.raw_data)

        # Confirm ASAP that the event is processed
        in_stream.task_done()

        # Use WebRTC VAD algorithm to detect voice
        in_empty = False
        if not vad.is_speech(
            buf=in_audio.raw_data,
            sample_rate=in_audio.frame_rate,
        ):
            in_empty = True
            # Start timeout if not already started
            if not silence_task:
                silence_task = asyncio.create_task(_silence_callback())

        if in_empty:
            # Continue to the next audio packet
            continue

        # Voice detected, cancel the timeout if any
        if silence_task:
            silence_task.cancel()
            silence_task = None

        # Start the TTS clear task
        if not clear_tts_task:
            clear_tts_task = asyncio.create_task(_clear_tts_callback())


def _tts_callback(
    call: CallStateModel,
    scheduler: Scheduler,
    tts_client: SpeechSynthesizer,
) -> Callable[[str, MessageStyleEnum], Awaitable[None]]:
    """
    Send back the TTS to the user.
    """

    @wraps(_tts_callback)
    async def wrapper(
        text: str,
        style: MessageStyleEnum,
    ) -> None:
        # Skip if no text
        if not text:
            return

        # Play the TTS
        await scheduler.spawn(
            handle_realtime_tts(
                call=call,
                scheduler=scheduler,
                style=style,
                text=text,
                tts_client=tts_client,
            )
        )

    return wrapper
