from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    database_url_test: str = ""
    hltv_api_base_url: str = "http://localhost:8000"
    pipeline_interval_hours: int = 6
    min_matches_to_retrain: int = 10


@lru_cache
def get_settings() -> Settings:
    return Settings()
