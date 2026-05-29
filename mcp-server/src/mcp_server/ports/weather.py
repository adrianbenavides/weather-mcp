"""Weather port - driven port for weather data retrieval."""

from typing import Protocol

from mcp_server.domain.weather import Coordinates, WeatherData


class WeatherPort(Protocol):
    """Port for retrieving current weather data at coordinates."""

    async def get_weather(self, coordinates: Coordinates) -> WeatherData:
        """Get current weather at coordinates.

        Args:
            coordinates: Geographic coordinates (latitude, longitude).

        Returns:
            Current weather data including temperature, wind speed, conditions.
        """
        ...
