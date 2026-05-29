"""Structured logging configuration using structlog."""

import logging
import sys
from typing import Any, Optional

import structlog


def configure_logging(
    log_format: str = "json",
    log_output: Optional[Any] = None,
) -> None:
    """Configure structlog for JSON or console output.

    Args:
        log_format: Either "json" (default, production) or "console" (dev-friendly)
        log_output: Optional file-like object for testing (defaults to stdout)
    """
    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if log_format == "console":
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        processors.append(structlog.processors.JSONRenderer())

    logger_factory: Any
    if log_output is not None:
        logger_factory = structlog.PrintLoggerFactory(file=log_output)
    else:
        logger_factory = structlog.PrintLoggerFactory(file=sys.stdout)

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=logger_factory,
    )
