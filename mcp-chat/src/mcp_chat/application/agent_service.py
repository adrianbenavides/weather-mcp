"""AgentService - application orchestrator for LLM-MCP agent loop."""

import time
from uuid import uuid4

import structlog
from beartype import beartype
from beartype.typing import Any, AsyncIterator

from mcp_chat.application.tool_buffer import ToolUseBuffer
from mcp_chat.domain.conversation import Conversation, LLMChunk, Message
from mcp_chat.ports.llm import LLMPort
from mcp_chat.ports.mcp_client import MCPClientPort


@beartype
class AgentService:
    """Orchestrates LLM streaming with tool-call loop and conversation history."""

    def __init__(self, llm: LLMPort, mcp_client: MCPClientPort) -> None:
        """Initialize with LLM and MCP client adapters."""
        self.llm = llm
        self.mcp_client = mcp_client
        self._messages: list[Message] = []

    async def run_turn(self, user_query: str) -> AsyncIterator[LLMChunk]:
        """Stream LLM response with tool calls executed and results injected."""
        turn_id = str(uuid4())
        structlog.contextvars.bind_contextvars(turn_id=turn_id)
        log = structlog.get_logger()

        try:
            tools = await self.mcp_client.list_tools()
            self._messages.append(Message(role="user", content=user_query))
            conversation = Conversation(messages=self._messages)

            while True:
                stream = self.llm.stream_response(conversation, tools)
                tool_buffer = ToolUseBuffer()

                async for chunk in stream:
                    self._process_chunk(chunk, tool_buffer)
                    yield chunk

                    if chunk.type == "stop" and tool_buffer.is_active:
                        break

                if not tool_buffer.is_active:
                    break

                await self._execute_tool_and_update_conversation(tool_buffer, log)
                conversation = Conversation(messages=self._messages)
                tool_buffer.reset()
        finally:
            structlog.contextvars.clear_contextvars()

    def _process_chunk(self, chunk: LLMChunk, tool_buffer: ToolUseBuffer) -> None:
        """Buffer tool use data from chunk."""
        if chunk.type == "tool_use_start":
            tool_buffer.start(chunk.tool_name or "", chunk.tool_use_id or "")
        elif chunk.type == "tool_use_input":
            input_part = chunk.input_chunk or chunk.tool_use_input_delta or ""
            tool_buffer.add_input_part(input_part)
        elif chunk.type == "tool_use_complete":
            if chunk.tool_use_input_complete:
                tool_buffer.set_complete_input(chunk.tool_use_input_complete)

    async def _execute_tool_and_update_conversation(
        self,
        tool_buffer: ToolUseBuffer,
        log: Any,
    ) -> None:
        """Execute buffered tool and inject result into conversation."""
        tool_name = tool_buffer.get_tool_name()
        if not tool_name:
            return

        tool_input = tool_buffer.get_tool_input()

        start_time = time.time()
        try:
            tool_result = await self.mcp_client.call_tool(tool_name, tool_input)
        except Exception as e:
            tool_result = f"Error calling tool {tool_name}: {str(e)}"
            log.error(
                "tool_call_failed",
                tool_name=tool_name,
                error=str(e),
            )
        finally:
            latency_ms = (time.time() - start_time) * 1000

        log.info(
            "tool_call",
            tool_name=tool_name,
            latency_ms=latency_ms,
        )

        tool_use_id = tool_buffer.get_tool_use_id() or ""
        self._messages.append(
            Message(
                role="assistant",
                content="",
                tool_use_id=tool_use_id,
                tool_name=tool_name,
                tool_use_input=tool_input,
            )
        )
        self._messages.append(Message(role="tool", content=tool_result, tool_use_id=tool_use_id))

    async def compact(self) -> None:
        """Summarize messages (no-op if <=1 message)."""
        if len(self._messages) <= 1:
            return

        log = structlog.get_logger()

        summary_request = list(self._messages) + [Message(role="user", content=self._summarization_prompt())]
        conversation = Conversation(messages=summary_request)

        summary_text = ""
        stream = self.llm.stream_response(conversation, tools=[])

        try:
            async for chunk in stream:
                if chunk.type == "text_delta" and chunk.text:
                    summary_text += chunk.text
        except Exception as e:
            log.error("compact_summarization_failed", error=str(e))
            return

        self._messages = [Message(role="user", content=summary_text)]

    @staticmethod
    def _summarization_prompt() -> str:
        """Prompt for LLM to summarize conversation context."""
        return (
            "Please summarize the conversation so far in a concise paragraph "
            "that preserves the key context and decisions."
        )
