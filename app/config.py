from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""
    WEBHOOK_SECRET: str = "change-me"
    DEFAULT_RISK_PCT: float = 1.0
    ACCOUNT_BALANCE: float = 10000.0
    SIGNAL_ENGINE_ENABLED: bool = True
    SIGNAL_ENGINE_INTERVAL_MINUTES: int = 15
    PORT: int = 8000

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
