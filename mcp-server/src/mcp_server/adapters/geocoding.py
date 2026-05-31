"""Open-Meteo geocoding adapter implementing GeocodingPort."""

import httpx

from mcp_server.adapters.http_adapter import HTTPAdapter
from mcp_server.domain.weather import Coordinates


class OpenMeteoGeocodingAdapter(HTTPAdapter):
    """Geocoding adapter using Open-Meteo Geocoding API.

    Implements GeocodingPort: converts location names to coordinates.
    """

    async def geocode(self, location: str) -> Coordinates | None:
        """Convert location name to coordinates via Open-Meteo API.

        Args:
            location: Location name (e.g., "London", "San Francisco").

        Returns:
            Coordinates with latitude and longitude, or None if not found.

        Raises:
            WeatherError: If API call fails or location not found.
        """
        from mcp_server.domain.errors import WeatherError

        client = self._get_client()
        try:
            response = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={
                    "name": location,
                    "count": 1,
                    "language": "en",
                    "format": "json",
                },
            )
            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            if not results:
                return None

            first_result = results[0]
            return Coordinates(
                latitude=first_result["latitude"],
                longitude=first_result["longitude"],
            )
        except httpx.TimeoutException as e:
            raise WeatherError(f"Request timed out while geocoding '{location}': {str(e)}") from e
        except httpx.ConnectError as e:
            raise WeatherError(f"Could not connect to geocoding service: {str(e)}") from e
        except httpx.HTTPStatusError as e:
            raise WeatherError(f"API error (HTTP {e.response.status_code}) during geocoding: {str(e)}") from e
        except httpx.RequestError as e:
            raise WeatherError(f"Network error during geocoding: {str(e)}") from e
        finally:
            await self._cleanup_client(client)
