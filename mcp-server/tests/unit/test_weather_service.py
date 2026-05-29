"""Unit tests for WeatherService application layer.

Tests the orchestration of geocoding and weather queries through the
driving port (WeatherService.run_weather_query).

Uses test doubles (InMemory) for GeocodingPort and WeatherPort.
"""

import pytest
from mcp_server.application.weather_service import WeatherService
from mcp_server.domain.errors import WeatherError
from mcp_server.domain.weather import Coordinates, WeatherData


class InMemoryGeocodingAdapter:
    """Test double for GeocodingPort - provides controlled responses."""

    def __init__(self, known_locations: dict[str, Coordinates] | None = None) -> None:
        """Initialize with known location mappings.

        Args:
            known_locations: Map of location names to coordinates.
                             If None, uses default test data.
        """
        self.known_locations = known_locations or {
            "London": Coordinates(latitude=51.5074, longitude=-0.1278),
            "Paris": Coordinates(latitude=48.8566, longitude=2.3522),
        }

    async def geocode(self, location: str) -> Coordinates | None:
        """Return coordinates for known locations, None for unknown.

        Args:
            location: Location name to geocode.

        Returns:
            Coordinates if found, None if not found.
        """
        assert location is not None, "location must not be None"
        assert isinstance(location, str), f"location must be string, got {type(location)}"
        assert location, "location must not be empty string"

        return self.known_locations.get(location)


class InMemoryWeatherAdapter:
    """Test double for WeatherPort - returns fixed weather data."""

    def __init__(self, weather_data: WeatherData | None = None) -> None:
        """Initialize with fixed weather response.

        Args:
            weather_data: WeatherData to return for any coordinates.
                         If None, uses default test data.
        """
        self.weather_data = weather_data or WeatherData(
            temperature_c=15.0,
            wind_speed_kmh=10.0,
            conditions="Partly cloudy",
            location_name="Test Location",
        )

    async def get_weather(self, coordinates: Coordinates) -> WeatherData:
        """Return fixed weather data.

        Args:
            coordinates: Geographic coordinates (unused in test double).

        Returns:
            Pre-configured WeatherData.
        """
        assert coordinates is not None, "coordinates must not be None"
        return self.weather_data


@pytest.mark.asyncio
async def test_returns_weather_data_for_valid_location():
    """WeatherService returns WeatherData when location is found by geocoder.

    Behavior: valid location -> coordinates -> weather data returned
    """
    # Arrange
    london_coords = Coordinates(latitude=51.5074, longitude=-0.1278)
    geocoding = InMemoryGeocodingAdapter({"London": london_coords})
    weather = InMemoryWeatherAdapter(
        WeatherData(
            temperature_c=15.0,
            wind_speed_kmh=10.0,
            conditions="Partly cloudy",
            location_name="London",
        )
    )
    service = WeatherService(geocoding=geocoding, weather=weather)

    # Act
    result = await service.run_weather_query("London")

    # Assert
    assert isinstance(result, WeatherData)
    assert result.temperature_c == 15.0
    assert result.wind_speed_kmh == 10.0
    assert result.conditions == "Partly cloudy"
    assert result.location_name == "London"


@pytest.mark.asyncio
async def test_raises_weather_error_for_unknown_location():
    """WeatherService raises WeatherError when geocoder returns None.

    Behavior: unknown location -> geocoder returns None -> WeatherError raised
    """
    # Arrange
    geocoding = InMemoryGeocodingAdapter({})  # No known locations
    weather = InMemoryWeatherAdapter()
    service = WeatherService(geocoding=geocoding, weather=weather)

    # Act & Assert
    with pytest.raises(WeatherError) as exc_info:
        await service.run_weather_query("UnknownCity")

    error = exc_info.value
    assert "not found" in error.message.lower()
    assert error.location == "UnknownCity"
