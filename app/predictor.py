

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Optional
import torch
import joblib
import numpy as np

from .config import DEFAULT_START, METADATA_PATH, MODEL_WEIGHTS_PATH, SCALER_PATH
from .exceptions import DataFetchError, ModelNotFoundError, PredictionError
from .model import StockModel
from .utils import (
    fetch_stock_data,
    frame_to_features,
    inverse_close_prices,
    normalize_ticker,
)

logger = logging.getLogger(__name__)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
    
device = get_device()
logger.info(f"System is using device: {device}")

@lru_cache(maxsize=1)
def load_prediction_assets():
    missing = [
        str(path.name)
        for path in [MODEL_WEIGHTS_PATH, SCALER_PATH, METADATA_PATH]
        if not path.exists()
    ]
    if missing:
        raise ModelNotFoundError(
            "Missing trained files: "
            + ", ".join(missing)
            + ". Run `python train.py --ticker AAPL` first."
        )

    try:
        with METADATA_PATH.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        window_size = int(metadata["window_size"])
        stock_model = StockModel()
        stock_model.load_weights(window_size)
        model = stock_model.model
        scaler = joblib.load(SCALER_PATH)
    except (json.JSONDecodeError, KeyError, OSError, ValueError) as exc:
        raise ModelNotFoundError(f"Failed to load trained assets: {exc}") from exc

    logger.info("Loaded model assets (window_size=%d)", window_size)
    logger.info(f"Model successfully loaded and moved to {device}")
    return model, scaler, metadata


def reload_prediction_assets():
    """Force a fresh reload of model/scaler/metadata on the next call."""
    load_prediction_assets.cache_clear()

def _build_latest_window(df, scaler, window_size: int) -> np.ndarray:
    """Scale features and slice out the most recent window for inference."""
    feature_values = frame_to_features(df)

    if len(feature_values) <= window_size:
        raise DataFetchError(
            f"Need more than {window_size} prepared rows for prediction, "
            f"got {len(feature_values)}."
        )

    scaled_values = scaler.transform(feature_values)
    latest_window = scaled_values[-window_size:].reshape(
        1, window_size, scaled_values.shape[1]
    )
    return latest_window


def predict_next_close(ticker: str, start: str = DEFAULT_START) -> dict:
    """
    Predict the next closing price for `ticker`.

    Returns a dict ready to be serialized as JSON / consumed by a UI.
    Raises DataFetchError, ModelNotFoundError, or PredictionError on failure.
    """
    ticker = normalize_ticker(ticker)
    logger.info("Predicting next close for %s (start=%s)", ticker, start)

    model, scaler, metadata = load_prediction_assets()
    window_size = int(metadata["window_size"])

    df = fetch_stock_data(ticker, start=start, end=None)
    if df is None or df.empty:
        raise DataFetchError(f"No market data returned for ticker '{ticker}'.")

    latest_window = _build_latest_window(df, scaler, window_size)

    try:
        predicted_scaled = model.predict(latest_window, verbose=0)
        predicted_close = float(inverse_close_prices(scaler, predicted_scaled)[0])
    except Exception as exc:  # noqa: BLE001 — wrap any inference failure cleanly
        raise PredictionError(f"Model inference failed for '{ticker}': {exc}") from exc

    current_close = float(df["Close"].iloc[-1])
    change = predicted_close - current_close
    change_percent = (change / current_close) * 100 if current_close else 0.0

    history = [
        {"date": str(index.date()), "close": round(float(row["Close"]), 2)}
        for index, row in df.tail(60).iterrows()
    ]

    result = {
        "ticker": ticker,
        "window_size": window_size,
        "mape": metadata.get("mape"),
        "current_price": round(current_close, 2),
        "predicted_price": round(predicted_close, 2),
        "change": round(change, 2),
        "change_percent": round(change_percent, 2),
        "history": history,
    }

    logger.info(
        "%s prediction: current=%.2f predicted=%.2f (%.2f%%)",
        ticker, current_close, predicted_close, change_percent,
    )
    return result


def predict_multiple(tickers: list[str], start: str = DEFAULT_START) -> dict:
    results = {}
    for ticker in tickers:
        try:
            results[ticker] = predict_next_close(ticker, start=start)
        except (DataFetchError, ModelNotFoundError, PredictionError) as exc:
            logger.warning("Prediction failed for %s: %s", ticker, exc)
            results[ticker] = {"ticker": ticker, "error": str(exc)}
    return results