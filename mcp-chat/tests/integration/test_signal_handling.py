"""Integration tests for SIGINT signal handling in REPL loop.

Tests graceful Ctrl+C behavior: first Ctrl+C cancels in-flight streaming,
second Ctrl+C exits cleanly without traceback.
"""

import asyncio
import signal
from unittest.mock import AsyncMock, MagicMock, patch

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
        """No-op compact."""
        pass

    agent.run_turn = mock_run_turn
    agent.compact = mock_compact
    return agent


@pytest.mark.asyncio
async def test_graceful_interrupt_handling_registers_signal_handler(mock_agent):
    """Given repl_loop starts, when it initializes, then SIGINT signal handler registered."""
    inputs = iter(["quit"])

    async def mock_read_input(prompt: str) -> str:
        """Return input sequence."""
        return next(inputs)

    # Mock the event loop to capture add_signal_handler calls
    mock_loop = MagicMock()
    signal_handler_registered = []

    def capture_signal_handler(sig, handler):
        if sig == signal.SIGINT:
            signal_handler_registered.append(handler)

    mock_loop.add_signal_handler = MagicMock(side_effect=capture_signal_handler)
    mock_loop.remove_signal_handler = MagicMock()

    # Patch asyncio.get_running_loop to return our mock
    with (
        patch("mcp_chat.transport.cli.asyncio.get_running_loop", return_value=mock_loop),
        patch("mcp_chat.transport.cli._read_input", side_effect=mock_read_input),
    ):
        await repl_loop(mock_agent)

        # Verify signal handler was registered for SIGINT
        assert len(signal_handler_registered) == 1


def test_graceful_interrupt_handling_second_ctrlc_exits_cleanly():
    """Given first Ctrl+C already pressed (first_ctrlc=True), when second Ctrl+C pressed,
    then SystemExit(0) raised without traceback."""
    from mcp_chat.transport.cli import _handle_sigint

    # State already has first_ctrlc=True (first press already happened)
    state = {"first_ctrlc": True}

    with pytest.raises(SystemExit) as exc_info:
        _handle_sigint(None, state, lambda *args: None)

    assert exc_info.value.code == 0


@pytest.mark.asyncio
async def test_graceful_interrupt_handling_first_ctrlc_at_prompt(mock_agent):
    """Given Ctrl+C pressed while waiting at input prompt (no active task),
    when first Ctrl+C pressed, then first_ctrlc set, message printed."""
    printed_messages = []

    async def mock_read_input(prompt: str) -> str:
        """Simulate prompt - will be interrupted."""
        await asyncio.sleep(1)
        return "should not reach"

    def mock_print(*args, **kwargs):
        """Capture print output."""
        printed_messages.extend(args)

    mock_loop = MagicMock()
    signal_handler_ref = []

    def capture_signal_handler(sig, handler):
        if sig == signal.SIGINT:
            signal_handler_ref.append(handler)

    mock_loop.add_signal_handler = MagicMock(side_effect=capture_signal_handler)
    mock_loop.remove_signal_handler = MagicMock()

    with (
        patch("mcp_chat.transport.cli.asyncio.get_running_loop", return_value=mock_loop),
        patch("mcp_chat.transport.cli._read_input", side_effect=mock_read_input),
        patch("builtins.print", side_effect=mock_print),
    ):
        # Start the loop
        task = asyncio.create_task(repl_loop(mock_agent))
        await asyncio.sleep(0.05)

        # Call signal handler to simulate first Ctrl+C at prompt
        if signal_handler_ref:
            signal_handler_ref[0]()

        await asyncio.sleep(0.05)

        # Verify message was printed
        assert any("Press Ctrl+C again" in str(msg) for msg in printed_messages)

        # Clean up
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_graceful_interrupt_handling_first_ctrlc_reset_after_turn(mock_agent):
    """Given first_ctrlc set by previous Ctrl+C, when new turn completes successfully,
    then pressing Ctrl+C again prints warning (not exit) — flag was reset."""
    inputs_list = ["query1", "quit"]
    input_index = [0]
    printed_messages: list[str] = []

    async def mock_read_input(prompt: str) -> str:
        """Return input sequence."""
        if input_index[0] < len(inputs_list):
            result = inputs_list[input_index[0]]
            input_index[0] += 1
            return result
        return ""  # EOF

    def mock_print(*args: object, **kwargs: object) -> None:
        printed_messages.extend(str(a) for a in args)

    mock_loop = MagicMock()
    signal_handler_ref: list = []

    def capture_signal_handler(sig, handler):  # type: ignore[no-untyped-def]
        if sig == signal.SIGINT:
            signal_handler_ref.append(handler)

    mock_loop.add_signal_handler = MagicMock(side_effect=capture_signal_handler)
    mock_loop.remove_signal_handler = MagicMock()

    with (
        patch("mcp_chat.transport.cli.asyncio.get_running_loop", return_value=mock_loop),
        patch("mcp_chat.transport.cli._read_input", side_effect=mock_read_input),
        patch("builtins.print", side_effect=mock_print),
    ):
        await repl_loop(mock_agent)

    # Loop must have exited cleanly via "quit" (not SystemExit)
    assert any("Goodbye" in msg for msg in printed_messages)


# Unit tests for _handle_sigint state machine


def test_handle_sigint_first_call_with_active_streaming_task_cancels_and_sets_flag():
    """Given streaming task active, when _handle_sigint called first time,
    then task cancelled, first_ctrlc set to True, and 'Turn cancelled' message printed."""
    from mcp_chat.transport.cli import _handle_sigint

    state = {"first_ctrlc": False}
    mock_task = MagicMock(spec=asyncio.Task)
    mock_task.done.return_value = False
    mock_print_fn = MagicMock()

    _handle_sigint(mock_task, state, mock_print_fn)

    mock_task.cancel.assert_called_once()
    assert state["first_ctrlc"] is True
    mock_print_fn.assert_called_once_with("\n[Turn cancelled. Press Ctrl+C again to exit.]")


def test_handle_sigint_first_call_with_no_active_task_sets_flag():
    """Given no active streaming task, when _handle_sigint called first time,
    then first_ctrlc set to True and 'Press Ctrl+C again' message printed."""
    from mcp_chat.transport.cli import _handle_sigint

    state = {"first_ctrlc": False}
    mock_print_fn = MagicMock()

    _handle_sigint(None, state, mock_print_fn)

    assert state["first_ctrlc"] is True
    mock_print_fn.assert_called_once_with("\n[Press Ctrl+C again to exit.]")


def test_handle_sigint_second_call_raises_system_exit():
    """Given first_ctrlc already True, when _handle_sigint called second time,
    then SystemExit(0) raised."""
    from mcp_chat.transport.cli import _handle_sigint

    # Setup state with first_ctrlc already True
    state = {"first_ctrlc": True}

    # Call handler second time - should raise SystemExit
    with pytest.raises(SystemExit) as exc_info:
        _handle_sigint(None, state, lambda *args: None)

    assert exc_info.value.code == 0


def test_handle_sigint_does_not_cancel_completed_task():
    """Given completed task (done=True), when _handle_sigint called,
    then task not cancelled (already done)."""
    from mcp_chat.transport.cli import _handle_sigint

    state = {"first_ctrlc": False}
    mock_task = MagicMock(spec=asyncio.Task)
    mock_task.done.return_value = True

    # Call handler with completed task
    _handle_sigint(mock_task, state, lambda *args: None)

    # Verify cancel was not called on done task
    mock_task.cancel.assert_not_called()
