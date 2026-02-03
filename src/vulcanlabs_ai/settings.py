from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_file_encoding="utf-8")

    OPENAI_API_KEY: str

    TEXT_MODEL_NAME: str = "openai/gpt-5-nano"
    SUMMARIZATION_MODEL_NAME: str = "openai/gpt-5-nano"

    SUMMARY_TRIGGER_TOKENS: int = 10
    RECENT_N: int = 1

settings = Settings()