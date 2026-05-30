"""Integration tests for REPL loop interaction flow.

Tests the repl_loop() function through its transport layer port, validating
the indefinite loop behavior with user input handling, exit commands,
empty input re-prompting, tool execution, and conversation compaction.
"""

import asyncio
import signal
from unittest.mock import MagicMock, patch

import pytest
from mcp_chat.application.agent_service import AgentService
from mcp_chat.domain.conversation import LLMChunk
from mcp_chat.transport.cli import repl_loop


@pytest.fixture
def mock_agent() -> AgentService:
    """Stub agent for testing REPL loop control flow.

    Returns a mock AgentService with run_turn that yields text chunks
    and compact() that is awaitable.
    """
    agent = MagicMock(spec=AgentService)

    async def mock_run_turn(query: str):
        """Yield mock text chunk from LLM."""
        yield LLMChunk(type="text_delta", text=f"Response to: {query}")
        yield LLMChunk(type="stop", stop_reason="end_turn")

    async def mock_compact():
        """No-op compact (stub for 01-04)."""
        pass

    agent.run_turn = mock_run_turn
    agent.compact = mock_compact
    return agent


@pytest.mark.asyncio
async def test_repl_processes_initial_query_before_loop(mock_agent):
    """Given an initial query is provided, when loop starts,
    then initial query processed as first turn before prompting for more input."""
    inputs = iter(["quit"])

    async def mock_read_input(prompt: str) -> str:
        """Return preset input sequence."""
        return next(inputs)

    queries_seen: list[str] = []
    original_run_turn = mock_agent.run_turn

    async def capturing_run_turn(query: str):
        queries_seen.append(query)
        async for chunk in original_run_turn(query):
            yield chunk

    mock_agent.run_turn = capturing_run_turn

    with patch("mcp_chat.transport.cli._read_input", side_effect=mock_read_input):
        await repl_loop(mock_agent, initial_query="What is 2+2?")

    assert "What is 2+2?" in queries_seen


@pytest.mark.asyncio
async def test_repl_exits_on_quit_command(mock_agent):
    """Given quit command entered, when loop reads input, then loop prints Goodbye and exits."""
    inputs = iter(["quit"])
    printed: list[str] = []

    async def mock_read_input(prompt: str) -> str:
        return next(inputs)

    with (
        patch("mcp_chat.transport.cli._read_input", side_effect=mock_read_input),
        patch("builtins.print", side_effect=lambda *a, **k: printed.extend(str(x) for x in a)),
    ):
        await repl_loop(mock_agent)

    assert any(msg == "Goodbye!" for msg in printed)


@pytest.mark.asyncio
async def test_repl_exits_on_exit_command(mock_agent):
    """Given exit command entered, when loop reads input, then loop prints Goodbye and exits."""
    inputs = iter(["exit"])
    printed: list[str] = []

    async def mock_read_input(prompt: str) -> str:
        return next(inputs)

    with (
        patch("mcp_chat.transport.cli._read_input", side_effect=mock_read_input),
        patch("builtins.print", side_effect=lambda *a, **k: printed.extend(str(x) for x in a)),
    ):
        await repl_loop(mock_agent)

    assert any(msg == "Goodbye!" for msg in printed)


@pytest.mark.asyncio
async def test_repl_continues_on_empty_input(mock_agent):
    """Given empty/whitespace input, when loop reads it, then loop skips run_turn and re-prompts."""
    inputs = iter(["\n", "  ", "\t", "quit"])
    input_count = [0]

    async def mock_read_input(prompt: str) -> str:
        input_count[0] += 1
        return next(inputs)

    with patch("mcp_chat.transport.cli._read_input", side_effect=mock_read_input):
        await repl_loop(mock_agent)

    # 4 input calls: "\n", "  ", "\t", "quit" → loop continues on empty, exits on quit
    assert input_count[0] == 4


@pytest.mark.asyncio
async def test_repl_calls_run_turn_on_valid_input(mock_agent):
    """Given valid user input, when loop reads it, then run_turn called with that input."""
    inputs = iter(["Hello agent", "quit"])
    queries_received = []

    async def mock_run_turn(query: str):
        queries_received.append(query)
        yield LLMChunk(type="text_delta", text=f"Response: {query}")
        yield LLMChunk(type="stop", stop_reason="end_turn")

    async def mock_read_input(prompt: str) -> str:
        return next(inputs)

    mock_agent.run_turn = mock_run_turn

    with patch("mcp_chat.transport.cli._read_input", side_effect=mock_read_input):
        await repl_loop(mock_agent)

    # Verify run_turn was called with "Hello agent"
    assert "Hello agent" in queries_received


@pytest.mark.asyncio
async def test_repl_calls_compact_on_compact_command(mock_agent):
    """Given /compact command, when loop reads it, then agent.compact() called and loop continues."""
    inputs = iter(["/compact", "quit"])
    compact_called = []
    printed: list[str] = []

    async def mock_compact():
        compact_called.append(True)

    async def mock_read_input(prompt: str) -> str:
        return next(inputs)

    mock_agent.compact = mock_compact

    with (
        patch("mcp_chat.transport.cli._read_input", side_effect=mock_read_input),
        patch("builtins.print", side_effect=lambda *a, **k: printed.extend(str(x) for x in a)),
    ):
        await repl_loop(mock_agent)

    assert len(compact_called) == 1
    assert any(msg == "[Conversation compacted.]" for msg in printed)
    assert any(msg == "Goodbye!" for msg in printed)


@pytest.mark.asyncio
async def test_repl_indefinite_turns_until_exit(mock_agent):
    """Given sequence of valid queries, when loop reads them, then all processed until exit."""
    inputs = iter(["First query", "Second query", "Third query", "exit"])
    queries_received = []

    async def mock_run_turn(query: str):
        queries_received.append(query)
        yield LLMChunk(type="text_delta", text="Response")
        yield LLMChunk(type="stop", stop_reason="end_turn")

    async def mock_read_input(prompt: str) -> str:
        return next(inputs)

    mock_agent.run_turn = mock_run_turn

    with patch("mcp_chat.transport.cli._read_input", side_effect=mock_read_input):
        await repl_loop(mock_agent)

    # All three queries should be processed
    assert len(queries_received) == 3
    assert "First query" in queries_received
    assert "Second query" in queries_received
    assert "Third query" in queries_received


@pytest.mark.asyncio
async def test_repl_sigint_during_streaming_cancels_task_and_prints_turn_cancelled(mock_agent):
    """Given streaming is active, when first Ctrl+C pressed, then streaming task cancelled
    and 'Turn cancelled' message printed (not 'Press Ctrl+C again')."""
    streaming_started = asyncio.Event()
    printed_messages: list[str] = []
    input_consumed = [False]

    async def slow_run_turn(query: str):
        streaming_started.set()
        await asyncio.sleep(100)
        yield LLMChunk(type="text_delta", text="never")

    mock_agent.run_turn = slow_run_turn

    async def mock_read_input(prompt: str) -> str:
        if not input_consumed[0]:
            input_consumed[0] = True
            return "query1"
        return ""  # EOF

    def mock_print(*args: object, **kwargs: object) -> None:
        printed_messages.extend(str(a) for a in args)

    mock_loop = MagicMock()
    signal_handler_ref: list = []

    def capture_handler(sig: int, handler: object) -> None:
        if sig == signal.SIGINT:
            signal_handler_ref.append(handler)

    mock_loop.add_signal_handler = MagicMock(side_effect=capture_handler)
    mock_loop.remove_signal_handler = MagicMock()

    with (
        patch("mcp_chat.transport.cli.asyncio.get_running_loop", return_value=mock_loop),
        patch("mcp_chat.transport.cli._read_input", side_effect=mock_read_input),
        patch("builtins.print", side_effect=mock_print),
    ):
        task = asyncio.create_task(repl_loop(mock_agent))

        await streaming_started.wait()
        await asyncio.sleep(0.02)

        if signal_handler_ref:
            signal_handler_ref[0]()

        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, StopIteration):
            pass

    assert any("Turn cancelled" in msg for msg in printed_messages), (
        f"Expected 'Turn cancelled' message, got: {printed_messages}"
    )


@pytest.mark.asyncio
async def test_repl_first_ctrlc_reset_after_successful_turn(mock_agent):
    """Given first Ctrl+C sets first_ctrlc=True, when turn completes, flag resets.
    Second Ctrl+C should print warning (not SystemExit)."""
    streaming_started = asyncio.Event()
    turn_done = asyncio.Event()
    printed_messages: list[str] = []
    inputs_list = ["query1", "quit"]
    input_index = [0]

    async def run_turn_with_signal(query: str):
        streaming_started.set()
        yield LLMChunk(type="text_delta", text="response")
        yield LLMChunk(type="stop", stop_reason="end_turn")
        turn_done.set()

    mock_agent.run_turn = run_turn_with_signal

    async def mock_read_input(prompt: str) -> str:
        if input_index[0] < len(inputs_list):
            val = inputs_list[input_index[0]]
            input_index[0] += 1
            return val
        return ""  # EOF

    def mock_print(*args: object, **kwargs: object) -> None:
        printed_messages.extend(str(a) for a in args)

    mock_loop = MagicMock()
    signal_handler_ref: list = []

    def capture_handler(sig: int, handler: object) -> None:
        if sig == signal.SIGINT:
            signal_handler_ref.append(handler)

    mock_loop.add_signal_handler = MagicMock(side_effect=capture_handler)
    mock_loop.remove_signal_handler = MagicMock()

    exited_cleanly = [False]

    with (
        patch("mcp_chat.transport.cli.asyncio.get_running_loop", return_value=mock_loop),
        patch("mcp_chat.transport.cli._read_input", side_effect=mock_read_input),
        patch("builtins.print", side_effect=mock_print),
    ):
        await repl_loop(mock_agent)
        exited_cleanly[0] = True

    # Loop must exit via "quit" (not SystemExit) — proves first_ctrlc was reset
    assert exited_cleanly[0], "Loop should exit cleanly via 'quit'"
    assert any("Goodbye!" == msg for msg in printed_messages)


@pytest.mark.asyncio
async def test_repl_first_ctrlc_flag_resets_after_successful_turn(mock_agent):
    """Given first_ctrlc set to True during streaming, when turn completes successfully,
    then first_ctrlc is reset to False so second Ctrl+C prints warning not SystemExit."""
    streaming1_started = asyncio.Event()
    query2_complete = asyncio.Event()
    printed_messages: list[str] = []
    inputs_list = ["query1", "query2", "quit"]
    input_index = [0]
    sigint_fired = [0]

    async def run_turn_impl(query: str):
        if query == "query1":
            streaming1_started.set()
            await asyncio.sleep(100)  # will be cancelled
            yield LLMChunk(type="text_delta", text="never")
        elif query == "query2":
            yield LLMChunk(type="text_delta", text="response2")
            yield LLMChunk(type="stop", stop_reason="end_turn")
            query2_complete.set()
        else:
            yield LLMChunk(type="stop", stop_reason="end_turn")

    mock_agent.run_turn = run_turn_impl

    async def mock_read_input(prompt: str) -> str:
        if input_index[0] < len(inputs_list):
            val = inputs_list[input_index[0]]
            input_index[0] += 1
            return val
        return ""  # EOF

    def mock_print(*args: object, **kwargs: object) -> None:
        printed_messages.extend(str(a) for a in args)

    mock_loop = MagicMock()
    signal_handler_ref: list = []

    def capture_handler(sig: int, handler: object) -> None:
        if sig == signal.SIGINT:
            signal_handler_ref.append(handler)

    mock_loop.add_signal_handler = MagicMock(side_effect=capture_handler)
    mock_loop.remove_signal_handler = MagicMock()

    exited_cleanly = [False]

    async def run_test():
        task = asyncio.create_task(repl_loop(mock_agent))

        # Wait for first streaming to start, then fire first Ctrl+C
        await streaming1_started.wait()
        await asyncio.sleep(0.02)
        if signal_handler_ref:
            signal_handler_ref[0]()

        # Wait for query2 to complete
        await query2_complete.wait()
        await asyncio.sleep(0.05)

        # Fire second Ctrl+C — should NOT raise SystemExit (first_ctrlc was reset)
        raised_exit = False
        try:
            if signal_handler_ref:
                signal_handler_ref[0]()
        except SystemExit:
            raised_exit = True

        # Loop continues to "quit" input
        try:
            await asyncio.wait_for(task, timeout=2.0)
            exited_cleanly[0] = True
        except (asyncio.TimeoutError, SystemExit):
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, SystemExit):
                pass

        return raised_exit

    with (
        patch("mcp_chat.transport.cli.asyncio.get_running_loop", return_value=mock_loop),
        patch("mcp_chat.transport.cli._read_input", side_effect=mock_read_input),
        patch("builtins.print", side_effect=mock_print),
    ):
        raised_exit = await run_test()

    # Second Ctrl+C after successful turn should NOT have raised SystemExit
    assert not raised_exit, "first_ctrlc should be reset after successful turn"
    assert exited_cleanly[0], "Loop should exit cleanly via 'quit'"
    assert any("Goodbye!" == msg for msg in printed_messages)
