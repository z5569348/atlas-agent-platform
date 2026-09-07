from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["local", "test", "staging", "production"]


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


@lru_cache
def get_settings() -> Settings:
    return Settings()