from pathlib import Path

import pytest

from atlas_agent_platform.core.config import Settings


def test_settings_use_defaults_without_env_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    for variable_name in (
        "ATLAS_APP_NAME",
        "ATLAS_APP_VERSION",
        "ATLAS_ENVIRONMENT",
        "ATLAS_DEBUG",
    ):
        monkeypatch.delenv(variable_name, raising=False)

    settings = Settings()

    assert settings.app_name == "Atlas Agent Platform"
    assert settings.environment == "local"
    assert settings.debug is False


def test_environment_variables_override_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ATLAS_APP_NAME", "Atlas Test Platform")
    monkeypatch.setenv("ATLAS_DEBUG", "true")

    settings = Settings()

    assert settings.app_name == "Atlas Test Platform"
    assert settings.debug is True

def test_llm_settings_use_safe_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    for variable_name in (
        "ATLAS_LLM_PROVIDER",
        "ATLAS_LLM_MODEL",
        "ATLAS_LLM_BASE_URL",
        "ATLAS_LLM_API_KEY",
        "ATLAS_LLM_TIMEOUT_SECONDS",
        "ATLAS_LLM_MAX_RETRIES",
    ):
        monkeypatch.delenv(variable_name, raising=False)

    settings = Settings()

    assert settings.llm_provider == "mock"
    assert settings.llm_model == "mock-model"
    assert settings.llm_base_url is None
    assert settings.llm_api_key is None
    assert settings.llm_timeout_seconds == 30.0
    assert settings.llm_max_retries == 2


def test_llm_api_key_is_redacted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ATLAS_LLM_API_KEY", "secret-test-key")

    settings = Settings()
    api_key = settings.llm_api_key

    assert api_key is not None
    assert api_key.get_secret_value() == "secret-test-key"
    assert "secret-test-key" not in repr(settings)