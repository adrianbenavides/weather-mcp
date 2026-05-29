"""CLI renderer - async consumer of LLMChunk stream.

Renders streamed LLM responses to stdout/stderr with streaming feel.
"""

import sys
from typing import AsyncIterator

from mcp_chat.domain.conversation import LLMChunk


async def render_to_cli(chunks: AsyncIterator[LLMChunk]) -> None:
    """Render LLMChunk stream to CLI output.

    Args:
        chunks: AsyncIterator of LLMChunk objects from LLM stream.

    Output:
        - Text chunks printed to stdout (no newline, flushed immediately)
        - Tool calls printed to stderr or stdout as indicators
        - Final newline printed after stream ends
    """
    async for chunk in chunks:
        if chunk.type == "text_delta" and chunk.text:
            # Print text immediately for streaming feel
            sys.stdout.write(chunk.text)
            sys.stdout.flush()

        elif chunk.type == "tool_use_start":
            # Print tool call indicator to stderr
            tool_name = chunk.tool_name or "unknown"
            sys.stderr.write(f"\n[Calling tool: {tool_name}]\n")
            sys.stderr.flush()

        elif chunk.type == "tool_use_complete":
            # Tool completed - already indicated in tool_use_start
            pass

        elif chunk.type == "stop":
            # Stream ended
            pass

    # Print final newline after stream completes
    sys.stdout.write("\n")
    sys.stdout.flush()
