"""Unit tests for structured logging configuration."""

import json
from io import StringIO

import pytest
import structlog


def test_configure_logging_json_format_produces_valid_json() -> None:
    """Test: configure_logging with format='json' produces JSON-parseable output."""
    log_output = StringIO()

    # Import and call configure_logging with JSON format
    from mcp_chat.application.observability import configure_logging

    configure_logging(log_format="json", log_output=log_output)

    # Get the logger and log a test message
    log = structlog.get_logger()
    log.info("test_event", key="value", number=42)

    # Verify output is valid JSON
    log_text = log_output.getvalue()
    assert len(log_text) > 0, "No log output generated"

    # Parse as JSON
    log_lines = [line for line in log_text.strip().split("\n") if line]
    for line in log_lines:
        try:
            event = json.loads(line)
            assert isinstance(event, dict), "Log line should be a dict"
            assert "event" in event or "message" in event, "Log should have event or message key"
        except json.JSONDecodeError as e:
            pytest.fail(f"Failed to parse log line as JSON: {line}\n{e}")


def test_configure_logging_console_format_produces_non_json() -> None:
    """Test: configure_logging with format='console' produces non-JSON output."""
    log_output = StringIO()

    # Import and call configure_logging with console format
    from mcp_chat.application.observability import configure_logging

    configure_logging(log_format="console", log_output=log_output)

    # Get the logger and log a test message
    log = structlog.get_logger()
    log.info("test_event", key="value")

    # Verify output is NOT JSON (console renderer outputs text)
    log_text = log_output.getvalue()
    assert len(log_text) > 0, "No log output generated"

    # Verify at least one line is NOT valid JSON
    log_lines = [line for line in log_text.strip().split("\n") if line]
    non_json_found = False
    for line in log_lines:
        try:
            json.loads(line)
        except json.JSONDecodeError:
            non_json_found = True
            break

    assert non_json_found, "Console format should produce non-JSON output"


def test_configure_logging_default_format_is_json() -> None:
    """Test: configure_logging default format is 'json'."""
    log_output = StringIO()

    from mcp_chat.application.observability import configure_logging

    # Call with no format argument (should default to json)
    configure_logging(log_output=log_output)

    # Get the logger and log a test message
    log = structlog.get_logger()
    log.info("test_event")

    # Verify output is valid JSON
    log_text = log_output.getvalue()
    assert len(log_text) > 0, "No log output generated"

    # Parse as JSON
    log_lines = [line for line in log_text.strip().split("\n") if line]
    for line in log_lines:
        try:
            json.loads(line)
        except json.JSONDecodeError:
            pytest.fail(f"Default format should be JSON: {line}")
