# config.py — central place for paths and app settings

import os
from datetime import date
from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    MODEL_DIR: Path = BASE_DIR / "model"
    FRONTEND_DIR: Path = BASE_DIR / "frontend"
    OUTPUT_DIR: Path = BASE_DIR / "outputs"

    MODEL_WEIGHTS_PATH: Path = MODEL_DIR / "model_gru_weights.weights.h5"
    SCALER_PATH: Path = MODEL_DIR / "scaler.pkl"
    METADATA_PATH: Path = MODEL_DIR / "metadata.json"

    # Application settings
    DEFAULT_TICKER: str = "AAPL"
    DEFAULT_START: str = "2000-01-01"
    # None means "fetch up to today" — resolved at call time in utils.py
    DEFAULT_END: Optional[str] = None

    DEFAULT_WINDOW_SIZE: int = 90
    WINDOW_SIZES: List[int] = [30, 60, 90]
    FEATURE_COLUMNS: List[str] = ["Close", "Open", "Volume", "RSI_14", "SMA_20"]

    # Sensitive/environment-specific data — set these in a .env file
    STOCK_API_KEY: str = os.getenv("STOCK_API_KEY", "default_secret_key")
    DEBUG: bool = os.getenv("DEBUG", "True") == "True"

    def ensure_directories(self) -> None:
        """Create required directories if they don't exist yet."""
        for directory in (self.MODEL_DIR, self.FRONTEND_DIR, self.OUTPUT_DIR):
            directory.mkdir(parents=True, exist_ok=True)


# Initialize settings (singleton-style, import this where you need config)
settings = Settings()
settings.ensure_directories()

BASE_DIR = BASE_DIR
MODEL_DIR = settings.MODEL_DIR
FRONTEND_DIR = settings.FRONTEND_DIR
OUTPUT_DIR = settings.OUTPUT_DIR

MODEL_WEIGHTS_PATH = settings.MODEL_WEIGHTS_PATH
SCALER_PATH = settings.SCALER_PATH
METADATA_PATH = settings.METADATA_PATH

DEFAULT_TICKER = settings.DEFAULT_TICKER
DEFAULT_START = settings.DEFAULT_START
DEFAULT_END = settings.DEFAULT_END

DEFAULT_WINDOW_SIZE = settings.DEFAULT_WINDOW_SIZE
WINDOW_SIZES = settings.WINDOW_SIZES
FEATURE_COLUMNS = settings.FEATURE_COLUMNS

STOCK_API_KEY = settings.STOCK_API_KEY
DEBUG = settings.DEBUG