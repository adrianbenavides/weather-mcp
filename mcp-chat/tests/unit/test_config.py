"""Unit tests for AppConfig."""

import os
from unittest.mock import patch

import pytest
from mcp_chat.application.config import AppConfig


class TestAppConfigShould:
    """AppConfig behavior."""

    def test_default_provider_is_anthropic(self) -> None:
        """Default provider is 'anthropic'."""
        config = AppConfig(anthropic_api_key="key")
        assert config.llm_provider == "anthropic"

    def test_from_env_raises_if_anthropic_key_missing(self) -> None:
        """from_env() raises ValueError if ANTHROPIC_API_KEY missing and provider is anthropic."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "anthropic"}, clear=True):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                AppConfig.from_env()

    def test_from_env_raises_if_openai_key_missing(self) -> None:
        """from_env() raises ValueError if OPENAI_API_KEY missing and provider is openai."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "openai"}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                AppConfig.from_env()

    def test_from_env_invalid_provider_defaults_to_anthropic(self) -> None:
        """from_env() defaults to anthropic if provider is invalid."""
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "invalid_provider",
                "ANTHROPIC_API_KEY": "test-key",
            },
            clear=True,
        ):
            config = AppConfig.from_env()
            assert config.llm_provider == "anthropic"

    def test_from_env_loads_anthropic_key(self) -> None:
        """from_env() loads ANTHROPIC_API_KEY into config."""
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "anthropic",
                "ANTHROPIC_API_KEY": "my-key",
            },
            clear=True,
        ):
            config = AppConfig.from_env()
            assert config.anthropic_api_key == "my-key"

    def test_from_env_loads_llm_model(self) -> None:
        """from_env() loads LLM_MODEL from environment."""
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "anthropic",
                "ANTHROPIC_API_KEY": "my-key",
                "LLM_MODEL": "claude-opus-4-7",
            },
            clear=True,
        ):
            config = AppConfig.from_env()
            assert config.llm_model == "claude-opus-4-7"

    def test_from_env_openai_success(self) -> None:
        """from_env() loads OpenAI config successfully."""
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "openai",
                "OPENAI_API_KEY": "openai-key",
            },
            clear=True,
        ):
            config = AppConfig.from_env()
            assert config.llm_provider == "openai"
            assert config.openai_api_key == "openai-key"

    def test_llm_model_defaults_to_none(self) -> None:
        """llm_model defaults to None if not set."""
        config = AppConfig(anthropic_api_key="key")
        assert config.llm_model is None

    def test_anthropic_api_key_defaults_to_none(self) -> None:
        """anthropic_api_key can be None."""
        config = AppConfig(openai_api_key="key")
        assert config.anthropic_api_key is None

    def test_openai_api_key_defaults_to_none(self) -> None:
        """openai_api_key can be None."""
        config = AppConfig(anthropic_api_key="key")
        assert config.openai_api_key is None
