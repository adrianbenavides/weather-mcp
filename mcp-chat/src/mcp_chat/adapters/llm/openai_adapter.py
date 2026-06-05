"""OpenAI LLM adapter."""

import json
from beartype.typing import Any, AsyncIterator, cast

import openai
from beartype import beartype

from mcp_chat.adapters.llm.message_converter import conversation_to_openai_messages
from mcp_chat.application.config import AppConfig
from mcp_chat.domain.conversation import Conversation, LLMChunk, ToolSchema
from mcp_chat.ports.llm import LLMPort


@beartype
class OpenAIAdapter(LLMPort):
    """LLM adapter for OpenAI GPT models."""

    def __init__(self, config: AppConfig) -> None:
        """Initialize OpenAI adapter.

        Args:
            config: Application configuration with API key and model settings.
        """
        if not config.openai_api_key:
            raise ValueError("OPENAI_API_KEY required for OpenAIAdapter")

        self._client = openai.AsyncOpenAI(api_key=config.openai_api_key)
        self._model = config.llm_model or "gpt-4o-mini"

    async def stream_response(
        self,
        conversation: Conversation,
        tools: list[ToolSchema],
    ) -> AsyncIterator[LLMChunk]:
        """Stream response from OpenAI GPT.

        Args:
            conversation: Current conversation history.
            tools: Available tools for the LLM to invoke.

        Yields:
            LLMChunk objects as streamed from the API.
        """
        messages = conversation_to_openai_messages(conversation)
        tool_defs = self._tools_to_openai_format(tools) if tools else []

        stream_obj = self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            tools=tool_defs if tool_defs else None,  # type: ignore[arg-type]
            stream=True,
        )
        async with cast(Any, stream_obj) as stream:
            async for chunk in self._process_stream_chunks(stream):
                yield chunk

    def _tools_to_openai_format(self, tools: list[ToolSchema]) -> list[dict[str, Any]]:
        """Convert tools to OpenAI function format.

        Args:
            tools: Domain tool schemas.

        Returns:
            List of tool definitions in OpenAI format.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
            for tool in tools
        ]

    async def _process_stream_chunks(self, stream: Any) -> AsyncIterator[LLMChunk]:
        """Process streaming chunks from OpenAI API.

        Args:
            stream: Chunk stream from SDK.

        Yields:
            LLMChunk objects.
        """
        tool_call_buffer: dict[int, dict[str, Any]] = {}
        async for chunk in stream:
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            if not choice.delta:
                continue

            delta = choice.delta

            if delta.content:
                yield LLMChunk.text_delta(delta.content)

            if delta.tool_calls:
                self._accumulate_tool_calls(delta.tool_calls, tool_call_buffer)
                for chunk_out in self._yield_tool_call_chunks(delta.tool_calls, tool_call_buffer):
                    yield chunk_out

            if choice.finish_reason:
                for chunk_out in self._emit_complete_tool_calls(tool_call_buffer):
                    yield chunk_out
                tool_call_buffer.clear()

                if choice.finish_reason == "stop":
                    yield LLMChunk.stop("end_turn")

    def _accumulate_tool_calls(
        self,
        tool_calls: list[Any],
        buffer: dict[int, dict[str, Any]],
    ) -> None:
        """Accumulate tool call data from delta.

        Args:
            tool_calls: Tool calls from delta.
            buffer: Buffer to accumulate into.
        """
        for tool_call in tool_calls:
            idx = tool_call.index
            if idx not in buffer:
                buffer[idx] = {
                    "id": None,
                    "name": None,
                    "args_parts": [],
                }

            if tool_call.id:
                buffer[idx]["id"] = tool_call.id

            if tool_call.function and tool_call.function.name:
                buffer[idx]["name"] = tool_call.function.name

            if tool_call.function and tool_call.function.arguments:
                buffer[idx]["args_parts"].append(tool_call.function.arguments)

    def _yield_tool_call_chunks(
        self,
        tool_calls: list[Any],
        buffer: dict[int, dict[str, Any]],
    ) -> list[LLMChunk]:
        """Build chunks for tool call deltas.

        Args:
            tool_calls: Tool calls from delta.
            buffer: Accumulated buffer.

        Returns:
            List of LLMChunk objects.
        """
        chunks = []
        for tool_call in tool_calls:
            if tool_call.id:
                chunks.append(LLMChunk.tool_use_id_chunk(tool_call.id))

            if tool_call.function and tool_call.function.arguments:
                chunks.append(LLMChunk.tool_use_input_delta_chunk(tool_call.function.arguments))
        return chunks

    def _emit_complete_tool_calls(self, buffer: dict[int, dict[str, Any]]) -> list[LLMChunk]:
        """Build tool_use_complete chunks for all buffered tool calls.

        Args:
            buffer: Accumulated tool calls.

        Returns:
            List of LLMChunk objects.
        """
        chunks = []
        for buf in buffer.values():
            if buf["name"] and buf["id"]:
                args_str = "".join(buf["args_parts"])
                try:
                    args_dict = json.loads(args_str)
                except json.JSONDecodeError, ValueError:
                    args_dict = {}
                chunks.append(
                    LLMChunk.tool_use_complete(
                        buf["name"],
                        buf["id"],
                        args_dict,
                    )
                )
        return chunks
