

# utils.py — data fetching, feature engineering, and array helpers

from __future__ import annotations

import re
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf

from .config import FEATURE_COLUMNS
from .exceptions import DataFetchError

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def normalize_ticker(ticker: str) -> str:
    """Uppercase + strip whitespace; reject empty input."""
    if ticker is None:
        raise DataFetchError("Ticker symbol is required.")
    ticker = ticker.strip().upper()
    if not ticker:
        raise DataFetchError("Ticker symbol is required.")
    return ticker


def _validate_date(label: str, value: str | None) -> None:
    """Basic sanity check on YYYY-MM-DD date strings before hitting the API."""
    if value is None:
        return
    if not _DATE_RE.match(value):
        raise DataFetchError(f"Invalid {label} date '{value}'. Expected format YYYY-MM-DD.")


def fetch_stock_data(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    ticker = normalize_ticker(ticker)
    _validate_date("start", start)
    _validate_date("end", end)

    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,
            threads=False,
        )
    except Exception as exc:  # noqa: BLE001 — yfinance can raise many exception types
        raise DataFetchError(f"Failed to download data: {exc}", ticker=ticker) from exc

    if df is None or df.empty:
        raise DataFetchError(f"No stock data found for {ticker}.", ticker=ticker)

    # yfinance sometimes returns a MultiIndex column structure (e.g. for
    # multi-ticker downloads); flatten it to plain column names.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = {"Close", "Open", "Volume"}
    if not required.issubset(set(df.columns)):
        missing = required - set(df.columns)
        raise DataFetchError(
            f"Downloaded data for {ticker} is missing required columns: {missing}.",
            ticker=ticker,
        )

    return add_technical_indicators(df)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Attach SMA_20 and RSI_14 columns, dropping rows with insufficient history."""
    df = df.copy()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()

    change = df["Close"].diff()
    gain = change.mask(change < 0, 0.0)
    loss = -change.mask(change > 0, 0.0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()

    # Avoid division by zero when there's no losing streak yet
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))
    df["RSI_14"] = df["RSI_14"].fillna(100)  # no losses == maximally overbought

    df.dropna(inplace=True)
    if df.empty:
        raise DataFetchError("Not enough rows to calculate SMA and RSI features.")

    return df

def frame_to_features(df: pd.DataFrame) -> np.ndarray:
    """Select and order the model's expected feature columns as a numpy array."""
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise DataFetchError(f"DataFrame is missing expected feature columns: {missing}.")
    return df[FEATURE_COLUMNS].to_numpy()

def create_sequences(scaled_data: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    if len(scaled_data) <= window_size:
        raise DataFetchError(
            f"Not enough data ({len(scaled_data)} rows) to build a window of size {window_size}."
        )

    x, y = [], []
    for i in range(window_size, len(scaled_data)):
        x.append(scaled_data[i - window_size : i, :])
        y.append(scaled_data[i, 0])
    return np.array(x), np.array(y)

def inverse_close_prices(scaler, predicted_scaled: np.ndarray) -> np.ndarray:
    predicted_scaled = np.asarray(predicted_scaled).reshape(-1, 1)
    repeated = np.repeat(predicted_scaled, len(FEATURE_COLUMNS), axis=-1)
    return scaler.inverse_transform(repeated)[:, 0]