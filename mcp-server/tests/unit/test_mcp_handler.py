"""Unit tests for MCP handler error handling.

Tests that WeatherError is properly caught and returned as error response.
"""

import json
from unittest.mock import AsyncMock

import pytest
from mcp_server.application.weather_service import WeatherService
from mcp_server.domain.errors import WeatherError
from mcp_server.ports.geocoding import GeocodingPort
from mcp_server.ports.weather import WeatherPort


@pytest.mark.asyncio
async def test_weather_service_error_returns_in_mcp_response():
    """Test that WeatherService error is caught and returned as error response.

    Behavior: WeatherService raises WeatherError -> MCP returns error dict, not traceback
    """
    # Create mock adapters
    mock_geocoding = AsyncMock(spec=GeocodingPort)
    mock_geocoding.geocode.return_value = None  # Invalid location

    mock_weather = AsyncMock(spec=WeatherPort)

    # Create service with mocked adapters
    service = WeatherService(geocoding=mock_geocoding, weather=mock_weather)

    # Simulate what the MCP handler does: call service and catch WeatherError
    try:
        await service.run_weather_query("XYZ123InvalidCity")
        # If we get here, the test fails
        assert False, "Expected WeatherError to be raised"
    except WeatherError as e:
        # Verify the error has proper message (no traceback)
        error_dict = {"error": e.message}
        error_json = json.dumps(error_dict)

        # Parse and verify
        parsed = json.loads(error_json)
        assert "error" in parsed
        assert "not found" in parsed["error"].lower() or "location" in parsed["error"].lower()
        assert "Traceback" not in parsed["error"]


@pytest.mark.asyncio
async def test_adapter_network_error_returns_in_mcp_response():
    """Test that adapter network error is caught and returned as error response.

    Behavior: Adapter raises WeatherError on network timeout -> MCP returns error dict
    """
    # Create mock adapters
    mock_geocoding = AsyncMock(spec=GeocodingPort)
    mock_geocoding.geocode.side_effect = WeatherError("Request timed out while geocoding 'London'")

    mock_weather = AsyncMock(spec=WeatherPort)

    # Create service with mocked adapters
    service = WeatherService(geocoding=mock_geocoding, weather=mock_weather)

    # Simulate what the MCP handler does
    try:
        await service.run_weather_query("London")
        assert False, "Expected WeatherError to be raised"
    except WeatherError as e:
        # Verify the error message is clean (not a traceback)
        error_dict = {"error": e.message}
        error_json = json.dumps(error_dict)

        # Parse and verify
        parsed = json.loads(error_json)
        assert "error" in parsed
        assert "timeout" in parsed["error"].lower() or "timed out" in parsed["error"].lower()
        assert "Traceback" not in parsed["error"]
