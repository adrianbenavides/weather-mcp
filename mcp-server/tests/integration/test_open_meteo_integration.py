"""Integration tests for Open-Meteo adapters.

These tests exercise the adapters through their port interfaces
with real HTTP calls to the Open-Meteo API.
"""

import pytest
from mcp_server.adapters.geocoding import OpenMeteoGeocodingAdapter
from mcp_server.adapters.weather import OpenMeteoWeatherAdapter
from mcp_server.domain.weather import Coordinates


@pytest.mark.integration
async def test_geocoding_adapter_finds_london():
    """GeocodingPort.geocode('London') returns coordinates within expected range via real API."""
    adapter = OpenMeteoGeocodingAdapter()
    result = await adapter.geocode("London")

    assert result is not None
    assert isinstance(result, Coordinates)
    # London is approximately at 51.5°N, -0.1°W
    assert 50.5 <= result.latitude <= 52.0
    assert -1.0 <= result.longitude <= 0.5


@pytest.mark.integration
async def test_geocoding_adapter_returns_none_for_invalid_location():
    """GeocodingPort.geocode(invalid_name) returns None for unknown locations via real API."""
    adapter = OpenMeteoGeocodingAdapter()
    result = await adapter.geocode("xyzzy_invalid_999")

    assert result is None


@pytest.mark.integration
async def test_weather_adapter_returns_weather_data_for_london():
    """WeatherPort.get_weather(coordinates) returns WeatherData with all fields populated."""
    geocoding = OpenMeteoGeocodingAdapter()
    weather = OpenMeteoWeatherAdapter()

    # First geocode London to get coordinates
    coords = await geocoding.geocode("London")
    assert coords is not None

    # Then get weather for those coordinates
    weather_data = await weather.get_weather(coords)

    assert weather_data is not None
    assert weather_data.temperature_c is not None
    assert isinstance(weather_data.temperature_c, float)
    assert weather_data.wind_speed_kmh is not None
    assert isinstance(weather_data.wind_speed_kmh, float)
    assert weather_data.conditions is not None
    assert isinstance(weather_data.conditions, str)
    assert weather_data.location_name is not None
    assert isinstance(weather_data.location_name, str)
