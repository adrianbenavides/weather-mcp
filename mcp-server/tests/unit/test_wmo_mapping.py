"""Unit tests for WMO weather code mapping.

Tests the pure domain function that maps WMO codes to condition strings.
"""

import pytest
from mcp_server.adapters.weather import wmo_code_to_condition


class TestWmoCodeMapping:
    """WMO code to condition string mapping."""

    @pytest.mark.parametrize(
        "code,expected_condition",
        [
            (0, "Clear sky"),
            (1, "Mainly clear"),
            (2, "Partly cloudy"),
            (3, "Overcast"),
            (45, "Foggy"),
            (48, "Foggy"),
            (51, "Drizzle"),
            (53, "Drizzle"),
            (55, "Drizzle"),
            (61, "Rain"),
            (63, "Rain"),
            (65, "Rain"),
            (71, "Snow"),
            (73, "Snow"),
            (75, "Snow"),
            (80, "Rain showers"),
            (81, "Rain showers"),
            (82, "Rain showers"),
            (95, "Thunderstorm"),
        ],
    )
    def test_maps_known_wmo_codes_to_conditions(self, code: int, expected_condition: str) -> None:
        """Known WMO codes map to human-readable condition strings."""
        result = wmo_code_to_condition(code)
        assert result == expected_condition

    def test_unknown_wmo_code_returns_fallback(self) -> None:
        """Unknown WMO code returns generic fallback."""
        result = wmo_code_to_condition(999)
        assert result == "Unknown conditions"
