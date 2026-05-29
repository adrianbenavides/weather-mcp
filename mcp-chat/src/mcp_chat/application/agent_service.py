"""AgentService - application orchestrator for LLM-MCP agent loop."""

import time
from typing import Any, AsyncIterator
from uuid import uuid4

import structlog

from mcp_chat.application.tool_buffer import ToolUseBuffer
from mcp_chat.domain.conversation import Conversation, LLMChunk, Message
from mcp_chat.ports.llm import LLMPort
from mcp_chat.ports.mcp_client import MCPClientPort


class AgentService:
    """Orchestrates LLM streaming with tool-call loop.

    Responsible for:
    - Managing conversation history
    - Detecting tool_use chunks and pausing stream
    - Calling MCPClientPort to execute tools
    - Injecting tool results back into conversation
    - Resuming LLM stream for continuation
    """

    def __init__(self, llm: LLMPort, mcp_client: MCPClientPort) -> None:
        """Initialize agent service.

        Args:
            llm: LLMPort for streaming responses.
            mcp_client: MCPClientPort for tool execution.
        """
        self.llm = llm
        self.mcp_client = mcp_client

    async def run(self, user_query: str) -> AsyncIterator[LLMChunk]:
        """Run agent with user query, handle tool calls, yield all chunks.

        Args:
            user_query: User's question or request.

        Yields:
            LLMChunk objects from LLM stream.
        """
        # Generate unique turn ID and bind to context
        turn_id = str(uuid4())
        structlog.contextvars.bind_contextvars(turn_id=turn_id)
        log = structlog.get_logger()

        try:
            # Get available tools from MCP server
            tools = await self.mcp_client.list_tools()

            # Initialize conversation with user query
            messages = [Message(role="user", content=user_query)]
            conversation = Conversation(messages=messages)

            # Main loop: stream LLM response and handle tool calls
            while True:
                # Stream response from LLM
                stream = self.llm.stream_response(conversation, tools)
                tool_buffer = ToolUseBuffer()

                async for chunk in stream:
                    # Track and buffer tool use data
                    if chunk.type == "tool_use_start":
                        tool_buffer.start(chunk.tool_name or "", chunk.tool_use_id or "")
                    elif chunk.type == "tool_use_input":
                        input_part = chunk.input_chunk or chunk.tool_use_input_delta or ""
                        tool_buffer.add_input_part(input_part)
                    elif chunk.type == "tool_use_complete":
                        if chunk.tool_use_input_complete:
                            tool_buffer.set_complete_input(chunk.tool_use_input_complete)

                    # Always yield chunks to client
                    yield chunk

                    # If stream ends with stop and we have a pending tool call, break
                    if chunk.type == "stop" and tool_buffer.is_active:
                        break

                # If no tool calls, we're done
                if not tool_buffer.is_active:
                    break

                # Execute buffered tool
                await self._execute_tool_and_update_conversation(tool_buffer, conversation, messages, log)

                # Rebuild frozen Conversation from updated messages list
                conversation = Conversation(messages=messages)

                # Reset buffer for next iteration
                tool_buffer.reset()
        finally:
            # Clear context vars after turn completes
            structlog.contextvars.clear_contextvars()

    async def _execute_tool_and_update_conversation(
        self,
        tool_buffer: ToolUseBuffer,
        conversation: Conversation,
        messages: list[Message],
        log: Any,
    ) -> None:
        """Execute a tool and inject result back into conversation.

        Args:
            tool_buffer: Accumulated tool use buffer.
            conversation: Current conversation (mutated by reference for messages).
            messages: List of messages to append to.
            log: Logger instance.
        """
        tool_name = tool_buffer.get_tool_name()
        if not tool_name:
            return

        tool_input = tool_buffer.get_tool_input()

        # Execute tool with timing and error handling
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

        # Log tool call
        log.info(
            "tool_call",
            tool_name=tool_name,
            latency_ms=latency_ms,
        )

        # Inject tool use and result into conversation with metadata for proper API formatting
        tool_use_id = tool_buffer.get_tool_use_id() or ""
        messages.append(
            Message(
                role="assistant",
                content="",
                tool_use_id=tool_use_id,
                tool_name=tool_name,
                tool_use_input=tool_input,
            )
        )
        messages.append(Message(role="tool", content=tool_result, tool_use_id=tool_use_id))
