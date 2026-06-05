"""CLI renderer - async consumer of LLMChunk stream.

Renders streamed LLM responses to stdout/stderr with streaming feel.
"""

import asyncio
import signal
import sys

from beartype import beartype
from beartype.typing import AsyncIterator, Callable

from mcp_chat.application.agent_service import AgentService
from mcp_chat.domain.conversation import LLMChunk


@beartype
def _handle_sigint(
    streaming_task: asyncio.Task[None] | None,
    state: dict[str, bool],
    print_fn: Callable[..., None],
) -> None:
    """Handle SIGINT - exit on second press, cancel task on first."""
    if state["first_ctrlc"]:
        raise SystemExit(0)

    state["first_ctrlc"] = True

    if streaming_task is not None and not streaming_task.done():
        streaming_task.cancel()
        print_fn("\n[Turn cancelled. Press Ctrl+C again to exit.]")
    else:
        print_fn("\n[Press Ctrl+C again to exit.]")


@beartype
async def _read_input(prompt: str) -> str:
    """Read one line from stdin via fd monitoring — cancellable, no blocking thread."""
    loop = asyncio.get_running_loop()
    future: asyncio.Future[str] = loop.create_future()

    sys.stdout.write(prompt)
    sys.stdout.flush()

    def _stdin_readable() -> None:
        if future.done():
            return
        loop.remove_reader(sys.stdin.fileno())
        try:
            line = sys.stdin.readline()
            future.set_result(line)
        except Exception as exc:
            if not future.done():
                future.set_exception(exc)

    loop.add_reader(sys.stdin.fileno(), _stdin_readable)
    try:
        return await future
    except asyncio.CancelledError:
        loop.remove_reader(sys.stdin.fileno())
        raise


@beartype
async def render_to_cli(chunks: AsyncIterator[LLMChunk]) -> None:
    """Render streamed LLM chunks to CLI (text to stdout, tools to stderr)."""
    async for chunk in chunks:
        if chunk.type == "text_delta" and chunk.text:
            sys.stdout.write(chunk.text)
            sys.stdout.flush()
        elif chunk.type == "tool_use_start":
            tool_name = chunk.tool_name or "unknown"
            sys.stderr.write(f"\n[Calling tool: {tool_name}]\n")
            sys.stderr.flush()

    sys.stdout.write("\n")
    sys.stdout.flush()


@beartype
async def repl_loop(
    agent: AgentService,
    initial_query: str | None = None,
) -> None:
    """Read user input, run agent turns, handle /compact and Ctrl+C gracefully."""
    state = {"first_ctrlc": False}
    loop = asyncio.get_running_loop()
    current_task: asyncio.Task[None] | None = None

    def sigint_handler() -> None:
        """Handle SIGINT by delegating to _handle_sigint with current task."""
        _handle_sigint(current_task, state, print)

    loop.add_signal_handler(signal.SIGINT, sigint_handler)

    try:
        if initial_query:
            try:
                await render_to_cli(agent.run_turn(initial_query))
            except asyncio.CancelledError:
                pass

        while True:
            line = await _read_input("> ")

            if not line:  # EOF: readline returns '' at end of stream
                break

            user_input = line.rstrip("\n").strip()
            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            if user_input == "/compact":
                await agent.compact()
                print("[Conversation compacted.]")
                continue

            try:
                current_task = asyncio.create_task(render_to_cli(agent.run_turn(user_input)))
                await current_task
                state["first_ctrlc"] = False
            except asyncio.CancelledError:
                pass
    finally:
        loop.remove_signal_handler(signal.SIGINT)
