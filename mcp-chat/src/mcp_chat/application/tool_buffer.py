"""Tool use buffer for accumulating streaming tool data."""

from typing import Any

from beartype import beartype


@beartype
class ToolUseBuffer:
    """Accumulates tool use data from streaming chunks."""

    def __init__(self) -> None:
        """Initialize empty buffer."""
        self.data: dict[str, Any] = {}
        self.is_active = False

    def start(self, tool_name: str, tool_id: str) -> None:
        """Start buffering a new tool use.

        Args:
            tool_name: Name of the tool being used.
            tool_id: ID of the tool use.
        """
        self.data = {
            "name": tool_name,
            "id": tool_id,
            "input_parts": [],
        }
        self.is_active = True

    def add_input_part(self, part: str) -> None:
        """Add a partial JSON input to the buffer.

        Args:
            part: Partial JSON string.
        """
        if self.is_active:
            self.data["input_parts"].append(part)

    def set_complete_input(self, input_dict: dict[str, Any]) -> None:
        """Set the complete parsed input.

        Args:
            input_dict: Complete input dictionary.
        """
        if self.is_active:
            self.data["input"] = input_dict

    def get_tool_use_id(self) -> str | None:
        """Get buffered tool use ID.

        Returns:
            Tool use ID or None if not set.
        """
        return self.data.get("id")

    def get_tool_name(self) -> str | None:
        """Get buffered tool name.

        Returns:
            Tool name or None if not set.
        """
        return self.data.get("name")

    def get_tool_input(self) -> dict[str, Any]:
        """Get buffered tool input.

        Returns:
            Tool input dictionary or empty dict.
        """
        input_data = self.data.get("input", {})
        if not isinstance(input_data, dict):
            return {}
        return input_data

    def reset(self) -> None:
        """Clear the buffer."""
        self.data = {}
        self.is_active = False
