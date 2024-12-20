import json
import re
from datetime import UTC, datetime
from enum import Enum
from inspect import getmembers, isfunction
from typing import Any

from json_repair import repair_json
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from pydantic import BaseModel, Field, field_validator

from app.helpers.config import CONFIG
from app.helpers.llm_utils import AbstractPlugin
from app.helpers.monitoring import tracer

_FUNC_NAME_SANITIZER_R = r"[^a-zA-Z0-9_-]"
_MESSAGE_ACTION_R = r"(?:action=*([a-z_]*))? *(.*)"
_MESSAGE_STYLE_R = r"(?:style=*([a-z_]*))? *(.*)"

_db = CONFIG.database.instance()


class StyleEnum(str, Enum):
    """
    Voice styles the Azure AI Speech Service supports.

    Doc:
    - Speaking styles: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-synthesis-markup-voice#use-speaking-styles-and-roles
    - Support by language: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=tts#voice-styles-and-roles
    """

    CHEERFUL = "cheerful"
    NONE = "none"
    """This is not a valid style, but we use it in the code to indicate no style."""
    SAD = "sad"


class ActionEnum(str, Enum):
    CALL = "call"
    """User called the assistant."""
    HANGUP = "hangup"
    """User hung up the call."""
    SMS = "sms"
    """User sent an SMS."""
    TALK = "talk"
    """User sent a message."""


class PersonaEnum(str, Enum):
    ASSISTANT = "assistant"
    """Represents an AI assistant."""
    HUMAN = "human"
    """Represents a human user."""
    TOOL = "tool"
    """Not used but deprecated, kept for backward compatibility."""


class ToolModel(BaseModel):
    content: str = ""
    function_arguments: str = ""
    function_name: str = ""
    tool_id: str = ""

    def __add__(self, other: object) -> "ToolModel":
        if not isinstance(other, ChoiceDeltaToolCall):
            return NotImplemented
        if other.id:
            self.tool_id = other.id
        if other.function:
            if other.function.name:
                self.function_name = other.function.name
            if other.function.arguments:
                self.function_arguments += other.function.arguments
        return self

    def __hash__(self) -> int:
        return self.tool_id.__hash__()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ToolModel):
            return False
        return self.tool_id == other.tool_id

    def to_openai(self) -> ChatCompletionMessageToolCallParam:
        return ChatCompletionMessageToolCallParam(
            id=self.tool_id,
            type="function",
            function={
                "arguments": self.function_arguments,
                "name": "-".join(
                    re.sub(
                        _FUNC_NAME_SANITIZER_R,
                        "-",
                        self.function_name,
                    ).split("-")
                ),  # Sanitize with dashes then deduplicate dashes, backward compatibility with old models
            },
        )

    async def execute_function(self, plugin: AbstractPlugin) -> None:
        from app.helpers.logging import logger

        json_str = self.function_arguments
        name = self.function_name

        # Confirm the function name exists, this is a security measure to prevent arbitrary code execution, plus, Pydantic validator is not used on purpose to comply with older tools plugins
        if name not in ToolModel._available_function_names():
            res = f"Invalid function names {name}, available are {ToolModel._available_function_names()}."
            logger.warning(res)
            self.content = res
            return

        # Try to fix JSON args to catch LLM hallucinations
        # See: https://community.openai.com/t/gpt-4-1106-preview-messes-up-function-call-parameters-encoding/478500
        args: dict[str, Any] | Any = repair_json(
            json_str=json_str,
            return_objects=True,
        )  # pyright: ignore

        if not isinstance(args, dict):
            logger.warning(
                "Error decoding JSON args for function %s: %s...%s",
                name,
                self.function_arguments[:20],
                self.function_arguments[-20:],
            )
            self.content = f"Bad arguments, available are {ToolModel._available_function_names()}. Please try again."
            return

        with tracer.start_as_current_span(
            name="message_execute_function",
            attributes={
                "args": json.dumps(args),
                "name": name,
            },
        ) as span:
            try:
                # Persist the call if updating it
                async with _db.call_transac(
                    call=plugin.call,
                    scheduler=plugin.scheduler,
                ):
                    res = await getattr(plugin, name)(**args)
                # Format pretty logs
                res_log = f"{res[:20]}...{res[-20:]}"
                logger.info("Executing function %s (%s): %s", name, args, res_log)
            except TypeError as e:
                logger.warning(
                    "Wrong arguments for function %s: %s. Error: %s",
                    name,
                    args,
                    e,
                )
                res = "Wrong arguments, please fix them and try again."
                res_log = res
            except Exception as e:
                logger.exception(
                    "Error executing function %s with args %s",
                    self.function_name,
                    args,
                )
                res = f"Error: {e}."
                res_log = res
            span.set_attribute("result", res_log)
            self.content = res

    @staticmethod
    def _available_function_names() -> list[str]:
        from app.helpers.llm_tools import (
            DefaultPlugin,
        )

        return [name for name, _ in getmembers(DefaultPlugin, isfunction)]


class MessageModel(BaseModel):
    # Immutable fields
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC), frozen=True)
    # Editable fields
    action: ActionEnum = ActionEnum.TALK
    content: str
    persona: PersonaEnum
    style: StyleEnum = StyleEnum.NONE
    tool_calls: list[ToolModel] = []

    @field_validator("created_at")
    @classmethod
    def _validate_created_at(cls, created_at: datetime) -> datetime:
        """
        Ensure the created_at field is timezone-aware.

        Backward compatibility with models created before the timezone was added. All dates require the same timezone to be compared.
        """
        if not created_at.tzinfo:
            return created_at.replace(tzinfo=UTC)
        return created_at

    def to_openai(
        self,
    ) -> list[
        ChatCompletionAssistantMessageParam
        | ChatCompletionToolMessageParam
        | ChatCompletionUserMessageParam
    ]:
        # Removing newlines from the content to avoid hallucinations issues with GPT-4 Turbo
        content = " ".join([line.strip() for line in self.content.splitlines()])

        if self.persona == PersonaEnum.HUMAN:
            return [
                ChatCompletionUserMessageParam(
                    content=f"action={self.action.value} {content}",
                    role="user",
                )
            ]

        if self.persona == PersonaEnum.ASSISTANT:
            if not self.tool_calls:
                return [
                    ChatCompletionAssistantMessageParam(
                        content=f"action={self.action.value} style={self.style.value} {content}",
                        role="assistant",
                    )
                ]

        res = []
        res.append(
            ChatCompletionAssistantMessageParam(
                content=f"action={self.action.value} style={self.style.value} {content}",
                role="assistant",
                tool_calls=[tool_call.to_openai() for tool_call in self.tool_calls],
            )
        )
        res.extend(
            ChatCompletionToolMessageParam(
                content=tool_call.content,
                role="tool",
                tool_call_id=tool_call.tool_id,
            )
            for tool_call in self.tool_calls
            if tool_call.content
        )
        return res


def remove_message_action(text: str) -> str:
    """
    Remove action from content. AI often adds it by mistake event if explicitly asked not to.

    Example:
    - Input: "action=talk Hello!"
    - Output: "Hello!"
    """
    # TODO: Use JSON as LLM response instead of using a regex to parse the text
    res = re.match(_MESSAGE_ACTION_R, text)
    if not res:
        return text
    try:
        return res.group(2) or ""
    # Regex failed, return original text
    except ValueError:
        return text


def extract_message_style(text: str) -> tuple[StyleEnum, str]:
    """
    Detect the style of a message and extract it from the text.

    Example:
    - Input: "style=cheerful Hello!"
    - Output: (StyleEnum.CHEERFUL, "Hello!")
    """
    default_style = StyleEnum.NONE
    res = re.match(_MESSAGE_STYLE_R, text)
    if not res:
        return default_style, text
    try:
        return (
            StyleEnum(res.group(1)),  # style
            (res.group(2) or ""),  # content
        )
    # Regex failed, return original text
    except ValueError:
        return default_style, text
