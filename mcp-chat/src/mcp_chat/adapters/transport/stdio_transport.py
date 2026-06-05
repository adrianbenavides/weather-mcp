"""MCP stdio transport adapter."""

import asyncio
import sys
from datetime import timedelta
from pathlib import Path

from beartype import beartype
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from mcp_chat.adapters.transport.session_adapter import MCPSessionAdapter
from mcp_chat.ports.mcp_transport import MCPTransportPort


@beartype
class StdioMCPTransport(MCPTransportPort):
    """Transport implementation using stdio for MCP server subprocess communication.

    Spawns mcp-server subprocess and manages JSON-RPC 2.0 communication over stdin/stdout.

    Uses a background asyncio Task to own the stdio_client context manager, because
    anyio cancel scopes must be entered and exited within the same Task. Calling
    __aenter__/__aexit__ from different tasks triggers "Attempted to exit cancel scope
    in a different task" errors.
    """

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self._session: ClientSession | None = None
        self._session_adapter: MCPSessionAdapter | None = None
        self._connected = False
        self._background_task: asyncio.Task[None] | None = None
        self._ready_event: asyncio.Event = asyncio.Event()
        self._disconnect_event: asyncio.Event = asyncio.Event()
        self._connect_error: BaseException | None = None

    async def connect(self) -> None:
        """Establish connection to MCP server subprocess."""
        self._ready_event = asyncio.Event()
        self._disconnect_event = asyncio.Event()
        self._connect_error = None
        self._background_task = asyncio.create_task(self._run_connection())
        await self._ready_event.wait()
        if self._connect_error is not None:
            raise self._connect_error
        self._connected = True

    async def _run_connection(self) -> None:
        """Run the MCP connection lifecycle in a single Task to keep cancel scopes intact."""
        mcp_server_dir = self.project_root / "mcp-server"
        server_params = StdioServerParameters(
            command="uv",
            args=["run", "--directory", str(mcp_server_dir), "python", "-m", "mcp_server"],
        )
        try:
            async with stdio_client(server_params, errlog=sys.stderr) as (read_stream, write_stream):
                async with ClientSession(
                    read_stream, write_stream, read_timeout_seconds=timedelta(seconds=30)
                ) as session:
                    await session.initialize()
                    self._session = session
                    self._session_adapter = MCPSessionAdapter(session)
                    self._ready_event.set()
                    await self._disconnect_event.wait()
        except Exception as exc:
            self._connect_error = exc
            self._ready_event.set()
        finally:
            self._session = None
            self._session_adapter = None

    async def disconnect(self) -> None:
        """Close connection to MCP server."""
        self._disconnect_event.set()
        if self._background_task is not None:
            try:
                await self._background_task
            except Exception:
                pass
            self._background_task = None
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    @property
    def session(self) -> ClientSession | None:
        """Get the raw MCP ClientSession.

        Returns:
            ClientSession instance if connected, None otherwise.
        """
        return self._session
