"""Base HTTP adapter for common httpx client lifecycle management."""

import httpx


class HTTPAdapter:
    """Base adapter managing httpx AsyncClient lifecycle.

    Handles client creation, ownership, and cleanup.
    """

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        """Initialize HTTP adapter.

        Args:
            client: Optional AsyncClient for DI. If None, creates a new one.
        """
        self._client = client
        self._own_client = client is None

    async def _cleanup_client(self, client: httpx.AsyncClient | None) -> None:
        """Clean up HTTP client if we own it.

        Args:
            client: AsyncClient to clean up.
        """
        if self._own_client and client is not None:
            await client.aclose()

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client.

        Returns:
            AsyncClient instance.
        """
        return self._client or httpx.AsyncClient()
