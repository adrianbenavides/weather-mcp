"""Application configuration - environment validation."""

from pydantic import Field
from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    """Application configuration from environment variables.

    Currently no required environment variables.
    Open-Meteo API is free and requires no authentication.
    """

    log_level: str = Field(default="INFO", description="Logging level")

    class Config:
        """Pydantic settings configuration."""

        env_prefix = "MCP_SERVER_"
        case_sensitive = False
