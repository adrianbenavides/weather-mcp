"""Geocoding port - driven port for location-to-coordinates conversion."""

from typing import Protocol, runtime_checkable

from mcp_server.domain.weather import Coordinates


@runtime_checkable
class GeocodingPort(Protocol):
    """Port for converting location names to geographic coordinates."""

    async def geocode(self, location: str) -> Coordinates | None:
        """Convert location name to coordinates.

        Args:
            location: Location name (e.g., "San Francisco", "Paris").

        Returns:
            Coordinates with latitude and longitude, or None if not found.
        """
        ...
