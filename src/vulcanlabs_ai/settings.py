from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_file_encoding="utf-8")

    OPENAI_API_KEY: str

    LLM_MODEL_NAME: str = "gpt-5-nano"

    SUMMARY_TRIGGER_TOKENS: int = 400
    RECENT_N: int = 3

settings = Settings()