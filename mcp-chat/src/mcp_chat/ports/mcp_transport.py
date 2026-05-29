"""MCP transport port - driven port for MCP connection management."""

from typing import Any, Protocol


class MCPTransportPort(Protocol):
    """Port for managing transport connection to MCP server."""

    async def connect(self) -> None:
        """Establish connection to MCP server."""
        ...

    async def disconnect(self) -> None:
        """Close connection to MCP server."""
        ...

    def is_connected(self) -> bool:
        """Check if connected to MCP server.

        Returns:
            True if connected, False otherwise.
        """
        ...

    @property
    def session(self) -> Any:
        """Get the session for tool operations.

        Returns:
            Session object or None if not connected.
        """
        ...
