"""Conversation domain models."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    """Single message in a conversation.

    Frozen: immutable value object.
    """

    role: Literal["user", "assistant", "tool"]
    content: str
    tool_use_id: str | None = None
    tool_name: str | None = None
    tool_use_input: dict[str, Any] | None = None

    model_config = ConfigDict(frozen=True)


class ToolSchema(BaseModel):
    """Tool definition with input schema.

    Frozen: immutable value object.
    """

    name: str
    description: str
    input_schema: dict[str, Any]

    model_config = ConfigDict(frozen=True)


class LLMChunk(BaseModel):
    """Chunk of streaming response from LLM.

    Tagged union using type discriminator field.
    Frozen: immutable value object.
    """

    type: Literal[
        "text_delta",
        "tool_use_start",
        "tool_use_id",
        "tool_use_input",
        "tool_use_complete",
        "stop",
    ]

    # type="text_delta" fields
    text: str | None = None

    # type="tool_use_start" fields
    tool_name: str | None = None

    # type="tool_use_id" and "tool_use_start" fields
    tool_use_id: str | None = None

    # type="tool_use_input" fields (partial JSON)
    tool_use_input_delta: str | None = None
    # Backwards compatibility with old name
    input_chunk: str | None = None

    # type="tool_use_complete" fields
    tool_use_input_complete: dict[str, Any] | None = None

    # type="stop" fields
    stop_reason: str | None = None

    model_config = ConfigDict(frozen=True)

    @classmethod
    def text_delta(cls, text: str) -> "LLMChunk":
        """Create text_delta chunk."""
        return cls(type="text_delta", text=text)

    @classmethod
    def tool_use_start(cls, tool_name: str, tool_use_id: str) -> "LLMChunk":
        """Create tool_use_start chunk."""
        return cls(type="tool_use_start", tool_name=tool_name, tool_use_id=tool_use_id)

    @classmethod
    def tool_use_id_chunk(cls, tool_use_id: str) -> "LLMChunk":
        """Create tool_use_id chunk."""
        return cls(type="tool_use_id", tool_use_id=tool_use_id)

    @classmethod
    def tool_use_input_delta_chunk(cls, input_delta: str) -> "LLMChunk":
        """Create tool_use_input chunk (partial JSON)."""
        return cls(type="tool_use_input", tool_use_input_delta=input_delta)

    @classmethod
    def tool_use_input(cls, input_chunk: str) -> "LLMChunk":
        """Create tool_use_input chunk (backwards compatibility)."""
        return cls(type="tool_use_input", input_chunk=input_chunk)

    @classmethod
    def tool_use_complete(
        cls, tool_name: str, tool_use_id: str, tool_use_input: dict[str, Any]
    ) -> "LLMChunk":
        """Create tool_use_complete chunk."""
        return cls(
            type="tool_use_complete",
            tool_name=tool_name,
            tool_use_id=tool_use_id,
            tool_use_input_complete=tool_use_input,
        )

    @classmethod
    def stop(cls, stop_reason: str) -> "LLMChunk":
        """Create stop chunk."""
        return cls(type="stop", stop_reason=stop_reason)


class Conversation(BaseModel):
    """Immutable conversation history.

    Frozen: immutable value object.
    """

    messages: tuple[Message, ...] = Field(default_factory=tuple)

    model_config = ConfigDict(frozen=True)

    def __init__(self, messages: list[Message] | None = None, **kwargs: Any) -> None:
        """Initialize conversation with optional messages."""
        if messages is None:
            messages = []
        super().__init__(messages=tuple(messages), **kwargs)
