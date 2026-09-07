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