"""Unit tests for CLI renderer."""

import pytest
from mcp_chat.domain.conversation import LLMChunk
from mcp_chat.transport.cli import render_to_cli


async def _chunks(*chunks: LLMChunk):  # type: ignore
    """Helper: async generator for chunks."""
    for c in chunks:
        yield c


class TestRenderToCliShould:
    """CLI renderer behavior."""

    @pytest.mark.asyncio
    async def test_writes_text_delta_to_stdout(self, capsys) -> None:  # type: ignore
        """Given: text_delta chunk with text
        When: render_to_cli processes it
        Then: text written to stdout
        """
        await render_to_cli(_chunks(LLMChunk.text_delta("Hello")))
        captured = capsys.readouterr()
        assert "Hello" in captured.out

    @pytest.mark.asyncio
    async def test_skips_text_delta_with_none_text(self, capsys) -> None:  # type: ignore
        """Given: text_delta chunk with text=None
        When: render_to_cli processes it
        Then: no text written (only final newline)
        """
        chunk = LLMChunk(type="text_delta", text=None)
        await render_to_cli(_chunks(chunk))
        captured = capsys.readouterr()
        assert captured.out == "\n"

    @pytest.mark.asyncio
    async def test_writes_tool_use_start_to_stderr(self, capsys) -> None:  # type: ignore
        """Given: tool_use_start chunk with tool_name
        When: render_to_cli processes it
        Then: tool name written to stderr
        """
        await render_to_cli(_chunks(LLMChunk.tool_use_start("get_weather", "id1")))
        captured = capsys.readouterr()
        assert "get_weather" in captured.err

    @pytest.mark.asyncio
    async def test_tool_use_start_with_none_tool_name_writes_unknown(self, capsys) -> None:  # type: ignore
        """Given: tool_use_start chunk with tool_name=None
        When: render_to_cli processes it
        Then: 'unknown' written to stderr
        """
        chunk = LLMChunk(type="tool_use_start", tool_name=None, tool_use_id="id1")
        await render_to_cli(_chunks(chunk))
        captured = capsys.readouterr()
        assert "unknown" in captured.err

    @pytest.mark.asyncio
    async def test_tool_use_complete_produces_no_output(self, capsys) -> None:  # type: ignore
        """Given: tool_use_complete chunk
        When: render_to_cli processes it
        Then: no output to stderr, only final newline to stdout
        """
        await render_to_cli(_chunks(LLMChunk.tool_use_complete("t", "id1", {})))
        captured = capsys.readouterr()
        assert captured.err == ""
        assert captured.out == "\n"

    @pytest.mark.asyncio
    async def test_stop_produces_no_output_except_final_newline(self, capsys) -> None:  # type: ignore
        """Given: stop chunk
        When: render_to_cli processes it
        Then: no output except final newline
        """
        await render_to_cli(_chunks(LLMChunk.stop("end_turn")))
        captured = capsys.readouterr()
        assert captured.err == ""
        assert captured.out == "\n"

    @pytest.mark.asyncio
    async def test_always_writes_final_newline(self, capsys) -> None:  # type: ignore
        """Given: empty stream
        When: render_to_cli processes it
        Then: final newline always written
        """
        await render_to_cli(_chunks())
        captured = capsys.readouterr()
        assert captured.out.endswith("\n")

    @pytest.mark.asyncio
    async def test_multiple_text_chunks_concatenated(self, capsys) -> None:  # type: ignore
        """Given: multiple text_delta chunks
        When: render_to_cli processes them
        Then: all text concatenated in stdout
        """
        await render_to_cli(
            _chunks(
                LLMChunk.text_delta("Hello "),
                LLMChunk.text_delta("world"),
                LLMChunk.stop("end_turn"),
            )
        )
        captured = capsys.readouterr()
        assert "Hello world" in captured.out
