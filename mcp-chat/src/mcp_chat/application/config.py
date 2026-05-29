"""Application configuration."""

import os
from typing import Literal

from pydantic import BaseModel, ConfigDict


class AppConfig(BaseModel):
    """Application configuration from environment variables."""

    llm_provider: Literal["anthropic", "openai"] = "anthropic"
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    llm_model: str | None = None

    model_config = ConfigDict(validate_assignment=True)

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load configuration from environment variables.

        Raises:
            ValueError: If required API key is missing for the configured provider.
        """
        provider = os.getenv("LLM_PROVIDER", "anthropic")
        if provider not in ("anthropic", "openai"):
            provider = "anthropic"

        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        # Validate API key for configured provider
        if provider == "anthropic" and not anthropic_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable required for anthropic provider")
        if provider == "openai" and not openai_key:
            raise ValueError("OPENAI_API_KEY environment variable required for openai provider")

        return cls(
            llm_provider=provider,  # type: ignore[arg-type]
            anthropic_api_key=anthropic_key,
            openai_api_key=openai_key,
            llm_model=os.getenv("LLM_MODEL"),
        )
