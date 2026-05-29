"""Open-Meteo weather adapter implementing WeatherPort."""

from typing import Any

import httpx

from mcp_server.adapters.http_adapter import HTTPAdapter
from mcp_server.domain.weather import Coordinates, WeatherData

# WMO weather code to human-readable condition mapping
_WMO_CODE_MAPPING = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Foggy",
    51: "Drizzle",
    53: "Drizzle",
    55: "Drizzle",
    61: "Rain",
    63: "Rain",
    65: "Rain",
    71: "Snow",
    73: "Snow",
    75: "Snow",
    80: "Rain showers",
    81: "Rain showers",
    82: "Rain showers",
    95: "Thunderstorm",
}

_UNKNOWN_CONDITION = "Unknown conditions"
_DEFAULT_WMO_CODE = 0
_TEMP_MISSING_DEFAULT = 0.0
_WIND_MISSING_DEFAULT = 0.0


def wmo_code_to_condition(code: int) -> str:
    """Convert WMO weather code to human-readable condition string.

    Args:
        code: WMO weather code.

    Returns:
        Human-readable condition string, or unknown condition if code not in mapping.
    """
    return _WMO_CODE_MAPPING.get(code, _UNKNOWN_CONDITION)


class OpenMeteoWeatherAdapter(HTTPAdapter):
    """Weather adapter using Open-Meteo Weather API.

    Implements WeatherPort: retrieves current weather data at coordinates.
    """

    async def get_weather(self, coordinates: Coordinates) -> WeatherData:
        """Get current weather at coordinates via Open-Meteo API.

        Args:
            coordinates: Geographic coordinates (latitude, longitude).

        Returns:
            Current weather data including temperature, wind speed, conditions, location.

        Raises:
            WeatherError: If API call fails.
        """
        from mcp_server.domain.errors import WeatherError

        client = self._get_client()
        try:
            response = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": coordinates.latitude,
                    "longitude": coordinates.longitude,
                    "current": "temperature_2m,wind_speed_10m,weathercode",
                },
            )
            response.raise_for_status()
            data = response.json()

            return self._parse_weather_response(data, coordinates)
        except httpx.TimeoutException as e:
            raise WeatherError(f"Request timed out while fetching weather: {str(e)}") from e
        except httpx.ConnectError as e:
            raise WeatherError(f"Could not connect to weather service: {str(e)}") from e
        except httpx.RequestError as e:
            raise WeatherError(f"Network error during weather fetch: {str(e)}") from e
        finally:
            await self._cleanup_client(client)

    def _parse_weather_response(self, data: dict[str, Any], coordinates: Coordinates) -> WeatherData:
        """Parse weather response from API.

        Args:
            data: JSON response from Open-Meteo API.
            coordinates: Coordinates for location name formatting.

        Returns:
            WeatherData domain model.
        """
        current = data.get("current", {})
        wmo_code = current.get("weathercode", _DEFAULT_WMO_CODE)
        conditions = wmo_code_to_condition(wmo_code)

        location_name = f"({coordinates.latitude:.2f}, {coordinates.longitude:.2f})"

        return WeatherData(
            temperature_c=current.get("temperature_2m", _TEMP_MISSING_DEFAULT),
            wind_speed_kmh=current.get("wind_speed_10m", _WIND_MISSING_DEFAULT),
            conditions=conditions,
            location_name=location_name,
        )
