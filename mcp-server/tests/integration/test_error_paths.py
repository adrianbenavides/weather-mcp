"""Integration tests for error handling in weather adapters.

Tests network error scenarios and error propagation through the stack.
"""

from unittest.mock import AsyncMock

import pytest
from mcp_server.adapters.geocoding import OpenMeteoGeocodingAdapter
from mcp_server.adapters.weather import OpenMeteoWeatherAdapter
from mcp_server.application.weather_service import WeatherService
from mcp_server.domain.errors import WeatherError
from mcp_server.domain.weather import Coordinates
from mcp_server.ports.geocoding import GeocodingPort
from mcp_server.ports.weather import WeatherPort


class TestGeocodingAdapterErrorHandling:
    """Tests for GeocodingAdapter error handling."""

    @pytest.mark.asyncio
    async def test_handles_httpx_timeout_gracefully(self):
        """GeocodingAdapter catches httpx.TimeoutException and raises WeatherError.

        Behavior: httpx timeout -> WeatherError with descriptive message
        """
        import httpx

        # Mock client that raises TimeoutException
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.side_effect = httpx.TimeoutException("Request timed out")
        mock_client.aclose = AsyncMock()

        adapter = OpenMeteoGeocodingAdapter(client=mock_client)

        # Act & Assert
        with pytest.raises(WeatherError) as exc_info:
            await adapter.geocode("London")

        error = exc_info.value
        assert "timed out" in error.message.lower() or "timeout" in error.message.lower()
        mock_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_httpx_connect_error_gracefully(self):
        """GeocodingAdapter catches httpx.ConnectError and raises WeatherError.

        Behavior: connection refused -> WeatherError with descriptive message
        """
        import httpx

        # Mock client that raises ConnectError
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.side_effect = httpx.ConnectError("Could not connect")
        mock_client.aclose = AsyncMock()

        adapter = OpenMeteoGeocodingAdapter(client=mock_client)

        # Act & Assert
        with pytest.raises(WeatherError) as exc_info:
            await adapter.geocode("London")

        error = exc_info.value
        assert (
            "connect" in error.message.lower()
            or "network" in error.message.lower()
            or "unavailable" in error.message.lower()
        )
        mock_client.get.assert_called_once()


class TestWeatherAdapterErrorHandling:
    """Tests for WeatherAdapter error handling."""

    @pytest.mark.asyncio
    async def test_handles_httpx_timeout_gracefully(self):
        """WeatherAdapter catches httpx.TimeoutException and raises WeatherError.

        Behavior: httpx timeout -> WeatherError with descriptive message
        """
        import httpx

        # Mock client that raises TimeoutException
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.side_effect = httpx.TimeoutException("Request timed out")
        mock_client.aclose = AsyncMock()

        adapter = OpenMeteoWeatherAdapter(client=mock_client)
        coords = Coordinates(latitude=51.5074, longitude=-0.1278)

        # Act & Assert
        with pytest.raises(WeatherError) as exc_info:
            await adapter.get_weather(coords)

        error = exc_info.value
        assert "timed out" in error.message.lower() or "timeout" in error.message.lower()
        mock_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_httpx_connect_error_gracefully(self):
        """WeatherAdapter catches httpx.ConnectError and raises WeatherError.

        Behavior: connection refused -> WeatherError with descriptive message
        """
        import httpx

        # Mock client that raises ConnectError
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.side_effect = httpx.ConnectError("Could not connect")
        mock_client.aclose = AsyncMock()

        adapter = OpenMeteoWeatherAdapter(client=mock_client)
        coords = Coordinates(latitude=51.5074, longitude=-0.1278)

        # Act & Assert
        with pytest.raises(WeatherError) as exc_info:
            await adapter.get_weather(coords)

        error = exc_info.value
        assert (
            "connect" in error.message.lower()
            or "network" in error.message.lower()
            or "unavailable" in error.message.lower()
        )
        mock_client.get.assert_called_once()


class TestWeatherServiceErrorPropagation:
    """Tests for error propagation through WeatherService."""

    @pytest.mark.asyncio
    async def test_propagates_adapter_errors_as_weather_error(self):
        """WeatherService propagates adapter WeatherError without modification.

        Behavior: adapter raises WeatherError -> propagated to caller
        """
        # Mock geocoding that raises error
        geocoding = AsyncMock(spec=GeocodingPort)
        geocoding.geocode.side_effect = WeatherError("Network error: connection timed out")

        weather = AsyncMock(spec=WeatherPort)
        service = WeatherService(geocoding=geocoding, weather=weather)

        # Act & Assert
        with pytest.raises(WeatherError) as exc_info:
            await service.run_weather_query("London")

        error = exc_info.value
        assert "network error" in error.message.lower() or "timed out" in error.message.lower()

    @pytest.mark.asyncio
    async def test_raises_error_for_invalid_location(self):
        """WeatherService raises WeatherError when location not found.

        Behavior: geocoding returns None -> WeatherError with location context
        """
        # Mock geocoding that returns None
        geocoding = AsyncMock(spec=GeocodingPort)
        geocoding.geocode.return_value = None

        weather = AsyncMock(spec=WeatherPort)
        service = WeatherService(geocoding=geocoding, weather=weather)

        # Act & Assert
        with pytest.raises(WeatherError) as exc_info:
            await service.run_weather_query("XYZ123InvalidCity")

        error = exc_info.value
        assert "not found" in error.message.lower()
        assert error.location == "XYZ123InvalidCity"
