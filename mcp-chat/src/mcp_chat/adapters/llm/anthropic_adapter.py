"""Anthropic LLM adapter."""

import json
from typing import Any, AsyncIterator

import anthropic

from mcp_chat.adapters.llm.message_converter import (
    conversation_to_anthropic_messages,
    tools_to_definitions,
)
from mcp_chat.application.config import AppConfig
from mcp_chat.domain.conversation import Conversation, LLMChunk, ToolSchema
from mcp_chat.ports.llm import LLMPort


class AnthropicAdapter(LLMPort):
    """LLM adapter for Anthropic Claude models."""

    def __init__(self, config: AppConfig) -> None:
        """Initialize Anthropic adapter.

        Args:
            config: Application configuration with API key and model settings.
        """
        if not config.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY required for AnthropicAdapter")

        self._client = anthropic.AsyncAnthropic(api_key=config.anthropic_api_key)
        self._model = config.llm_model or "claude-haiku-4-5-20251001"

    async def stream_response(
        self,
        conversation: Conversation,
        tools: list[ToolSchema],
    ) -> AsyncIterator[LLMChunk]:
        """Implementation of stream_response.

        Args:
            conversation: Current conversation history.
            tools: Available tools for the LLM to invoke.

        Yields:
            LLMChunk objects as streamed from the API.
        """
        messages = conversation_to_anthropic_messages(conversation)
        tool_defs = tools_to_definitions(tools) if tools else []

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": 1024,
            "messages": messages,
        }
        if tool_defs:
            kwargs["tools"] = tool_defs

        async with self._client.messages.stream(**kwargs) as stream:  # type: ignore[arg-type]
            async for chunk in self._process_stream_events(stream):
                yield chunk

    async def _process_stream_events(self, stream: Any) -> AsyncIterator[LLMChunk]:
        """Process streaming events from Anthropic API.

        Args:
            stream: Event stream from SDK.

        Yields:
            LLMChunk objects.
        """
        tool_use_buffer: dict[str, Any] = {}
        async for event in stream:
            if event.type == "content_block_start":
                if event.content_block.type == "tool_use":
                    tool_use_buffer = {
                        "id": event.content_block.id,
                        "name": event.content_block.name,
                        "input_parts": [],
                    }
            elif event.type == "content_block_delta":
                if event.delta.type == "text_delta":
                    yield LLMChunk.text_delta(event.delta.text)
                elif event.delta.type == "input_json_delta":
                    yield LLMChunk.tool_use_input_delta_chunk(event.delta.partial_json)
                    tool_use_buffer["input_parts"].append(event.delta.partial_json)
            elif event.type == "content_block_stop":
                if tool_use_buffer:
                    for chunk in self._emit_tool_use_chunks(tool_use_buffer):
                        yield chunk
                    tool_use_buffer = {}
            elif event.type == "message_stop":
                yield LLMChunk.stop("end_turn")

    def _emit_tool_use_chunks(self, tool_use_buffer: dict[str, Any]) -> list[LLMChunk]:
        """Emit tool_use chunks from accumulated buffer.

        Args:
            tool_use_buffer: Accumulated tool use data.

        Returns:
            List of LLMChunk objects for tool use.
        """
        input_json_str = "".join(tool_use_buffer["input_parts"])
        try:
            input_dict = json.loads(input_json_str)
        except json.JSONDecodeError, ValueError:
            input_dict = {}
        return [
            LLMChunk.tool_use_start(
                tool_use_buffer["name"],
                tool_use_buffer["id"],
            ),
            LLMChunk.tool_use_complete(
                tool_use_buffer["name"],
                tool_use_buffer["id"],
                input_dict,
            ),
        ]
