"""Acceptance tests for mcp-server domain models and ports.

This test verifies:
- WeatherData domain model (frozen, all fields required)
- Coordinates domain model (frozen, valid lat/lon)
- WeatherError exception class
- GeocodingPort protocol interface
- WeatherPort protocol interface
"""

import pytest
from mcp_server.domain.errors import WeatherError
from mcp_server.domain.weather import Coordinates, WeatherData
from mcp_server.ports.geocoding import GeocodingPort
from mcp_server.ports.weather import WeatherPort


class TestWeatherDataModel:
    """WeatherData domain model behavior."""

    def test_creates_weather_data_with_all_required_fields(self) -> None:
        """WeatherData requires all fields: temperature_c, wind_speed_kmh, conditions, location_name."""
        data = WeatherData(
            temperature_c=22.5,
            wind_speed_kmh=15.0,
            conditions="Partly cloudy",
            location_name="San Francisco",
        )
        assert data.temperature_c == 22.5
        assert data.wind_speed_kmh == 15.0
        assert data.conditions == "Partly cloudy"
        assert data.location_name == "San Francisco"

    def test_weather_data_is_frozen(self) -> None:
        """WeatherData is immutable - assignment should raise error."""
        data = WeatherData(
            temperature_c=22.5,
            wind_speed_kmh=15.0,
            conditions="Sunny",
            location_name="SF",
        )
        with pytest.raises(Exception):  # ValidationError or AttributeError
            data.temperature_c = 25.0  # type: ignore

    def test_weather_data_rejects_missing_fields(self) -> None:
        """WeatherData requires all fields - missing field raises ValidationError."""
        with pytest.raises(Exception):  # ValidationError
            WeatherData(
                temperature_c=22.5,
                wind_speed_kmh=15.0,
                conditions="Sunny",
                # Missing location_name
            )  # type: ignore


class TestCoordinatesModel:
    """Coordinates domain model behavior."""

    def test_creates_coordinates_with_valid_values(self) -> None:
        """Coordinates requires latitude and longitude as float."""
        coords = Coordinates(latitude=37.7749, longitude=-122.4194)
        assert coords.latitude == 37.7749
        assert coords.longitude == -122.4194

    def test_coordinates_is_frozen(self) -> None:
        """Coordinates is immutable."""
        coords = Coordinates(latitude=37.7749, longitude=-122.4194)
        with pytest.raises(Exception):  # ValidationError or AttributeError
            coords.latitude = 40.0  # type: ignore

    def test_coordinates_rejects_missing_fields(self) -> None:
        """Coordinates requires both latitude and longitude."""
        with pytest.raises(Exception):  # ValidationError
            Coordinates(latitude=37.7749)  # type: ignore


class TestWeatherError:
    """WeatherError exception class."""

    def test_weather_error_is_exception_subclass(self) -> None:
        """WeatherError is an Exception."""
        err = WeatherError("Test error", location="SF")
        assert isinstance(err, Exception)

    def test_weather_error_with_message_and_location(self) -> None:
        """WeatherError stores message and optional location."""
        err = WeatherError("API unavailable", location="London")
        assert err.args
        assert "API unavailable" in str(err) or hasattr(err, "message")


class TestGeocodingPort:
    """GeocodingPort interface structure."""

    def test_geocoding_port_is_protocol(self) -> None:
        """GeocodingPort defines geocode method signature."""
        # GeocodingPort should be a Protocol with geocode(location: str) -> Coordinates | None
        assert hasattr(GeocodingPort, "__mro__")
        # Verify it's a Protocol by checking for runtime_checkable or Protocol in bases
        assert "Protocol" in str(type(GeocodingPort))

    def test_geocoding_port_defines_geocode_method(self) -> None:
        """GeocodingPort.geocode(location: str) -> Coordinates | None."""
        # The Protocol should define a geocode method
        # We can't instantiate Protocol directly, but we can verify structure exists
        assert hasattr(GeocodingPort, "geocode")


class TestWeatherPort:
    """WeatherPort interface structure."""

    def test_weather_port_is_protocol(self) -> None:
        """WeatherPort defines get_weather method signature."""
        assert hasattr(WeatherPort, "__mro__")
        assert "Protocol" in str(type(WeatherPort))

    def test_weather_port_defines_get_weather_method(self) -> None:
        """WeatherPort.get_weather(coordinates: Coordinates) -> WeatherData."""
        assert hasattr(WeatherPort, "get_weather")
