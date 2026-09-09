from functools import lru_cache
from typing import Literal

from pydantic import AnyHttpUrl, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["local", "test", "staging", "production"]
ModelProvider = Literal["mock", "openai", "qwen", "anthropic"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="ATLAS_",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Atlas Agent Platform"
    app_version: str = "0.1.0"
    environment: Environment = "local"
    debug: bool = False

    llm_provider: ModelProvider = "mock"
    llm_model: str = "mock-model"
    llm_base_url: AnyHttpUrl | None = None
    llm_api_key: SecretStr | None = None
    llm_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    llm_max_retries: int = Field(default=2, ge=0, le=10)


@lru_cache
def get_settings() -> Settings:
    return Settings()