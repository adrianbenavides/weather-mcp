"""LLM port - driven port for language model streaming."""

from typing import AsyncIterator, Protocol

from mcp_chat.domain.conversation import Conversation, LLMChunk, ToolSchema


class LLMPort(Protocol):
    """Port for streaming responses from a language model."""

    def stream_response(
        self,
        conversation: Conversation,
        tools: list[ToolSchema],
    ) -> AsyncIterator[LLMChunk]:
        """Stream response from LLM with optional tool definitions.

        Args:
            conversation: Current conversation history.
            tools: Available tools for the LLM to invoke.

        Returns:
            AsyncIterator yielding LLMChunk objects.
        """
        ...
