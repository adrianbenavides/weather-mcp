"""WeatherService application layer - orchestrates geocoding and weather queries.

This is the driving port for weather functionality.
"""

from mcp_server.domain.errors import WeatherError
from mcp_server.domain.weather import WeatherData
from mcp_server.ports.geocoding import GeocodingPort
from mcp_server.ports.weather import WeatherPort


class WeatherService:
    """Orchestrates location geocoding and weather data retrieval.

    Depends on two driven ports:
    - GeocodingPort: converts location names to coordinates
    - WeatherPort: retrieves weather data at coordinates
    """

    def __init__(self, geocoding: GeocodingPort, weather: WeatherPort) -> None:
        """Initialize service with geocoding and weather ports.

        Args:
            geocoding: Port for location-to-coordinates conversion.
            weather: Port for weather data retrieval at coordinates.
        """
        self.geocoding = geocoding
        self.weather = weather

    async def run_weather_query(self, location: str) -> WeatherData:
        """Query current weather for a location.

        Args:
            location: Location name (e.g., "London", "Paris").

        Returns:
            Current weather data for the location.

        Raises:
            WeatherError: If location not found or weather data unavailable.
        """
        coordinates = await self.geocoding.geocode(location)
        if coordinates is None:
            raise WeatherError(
                f"Location '{location}' not found",
                location=location,
            )

        weather_data = await self.weather.get_weather(coordinates)
        return weather_data
