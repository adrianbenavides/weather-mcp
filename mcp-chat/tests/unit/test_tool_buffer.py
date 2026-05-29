"""Unit tests for ToolUseBuffer."""

import pytest
from mcp_chat.application.tool_buffer import ToolUseBuffer


class TestToolUseBufferShould:
    """ToolUseBuffer behavior."""

    def test_initial_state_is_inactive(self) -> None:
        """New buffer is inactive with empty data."""
        buf = ToolUseBuffer()
        assert buf.is_active is False
        assert buf.data == {}

    def test_start_sets_active_and_stores_name_id(self) -> None:
        """start() activates buffer and stores name and id."""
        buf = ToolUseBuffer()
        buf.start("get_weather", "id1")
        assert buf.is_active is True
        assert buf.get_tool_name() == "get_weather"
        assert buf.data["id"] == "id1"

    def test_add_input_part_appends_when_active(self) -> None:
        """add_input_part() appends to input_parts when active."""
        buf = ToolUseBuffer()
        buf.start("tool", "id1")
        buf.add_input_part('{"loc')
        buf.add_input_part('ation": "London"}')
        assert buf.data["input_parts"] == ['{"loc', 'ation": "London"}']

    def test_add_input_part_noop_when_inactive(self) -> None:
        """add_input_part() does nothing when buffer is inactive."""
        buf = ToolUseBuffer()
        buf.add_input_part("should be ignored")
        assert buf.data == {}

    def test_set_complete_input_stores_dict_when_active(self) -> None:
        """set_complete_input() stores dict when active."""
        buf = ToolUseBuffer()
        buf.start("tool", "id1")
        buf.set_complete_input({"location": "London"})
        assert buf.data["input"] == {"location": "London"}

    def test_set_complete_input_noop_when_inactive(self) -> None:
        """set_complete_input() does nothing when inactive."""
        buf = ToolUseBuffer()
        buf.set_complete_input({"should": "be ignored"})
        assert buf.data == {}

    def test_get_tool_name_returns_none_when_empty(self) -> None:
        """get_tool_name() returns None if name not set."""
        buf = ToolUseBuffer()
        assert buf.get_tool_name() is None

    def test_get_tool_input_returns_empty_dict_when_no_input(self) -> None:
        """get_tool_input() returns {} if input not set."""
        buf = ToolUseBuffer()
        buf.start("tool", "id1")
        assert buf.get_tool_input() == {}

    def test_get_tool_input_returns_empty_dict_when_non_dict(self) -> None:
        """get_tool_input() returns {} if input is not a dict."""
        buf = ToolUseBuffer()
        buf.start("tool", "id1")
        buf.data["input"] = "not a dict"
        assert buf.get_tool_input() == {}

    def test_get_tool_input_returns_stored_dict(self) -> None:
        """get_tool_input() returns stored input dict."""
        buf = ToolUseBuffer()
        buf.start("tool", "id1")
        buf.set_complete_input({"location": "London"})
        assert buf.get_tool_input() == {"location": "London"}

    def test_reset_clears_data_and_deactivates(self) -> None:
        """reset() clears data and deactivates buffer."""
        buf = ToolUseBuffer()
        buf.start("tool", "id1")
        buf.add_input_part("part")
        buf.reset()
        assert buf.is_active is False
        assert buf.data == {}
