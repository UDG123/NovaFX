from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""
    WEBHOOK_SECRET: str = "change-me"
    DEFAULT_RISK_PCT: float = 1.0
    ACCOUNT_BALANCE: float = 10000.0
    SIGNAL_ENGINE_ENABLED: bool = True
    SIGNAL_ENGINE_INTERVAL_MINUTES: int = 15
    TWELVEDATA_API_KEY: str = ""
    PORT: int = 8000

    # Telegram desk channels (per asset class)
    TG_DESK1: str = ""  # Forex Majors
    TG_DESK2: str = ""  # Forex Crosses
    TG_DESK3: str = ""  # Crypto
    TG_DESK4: str = ""  # Stocks
    TG_DESK5: str = ""  # Commodities
    TG_DESK6: str = ""  # Indices
    TG_PORTFOLIO: str = ""  # Mirror all signals
    TG_SYSTEM: str = ""  # System alerts

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
