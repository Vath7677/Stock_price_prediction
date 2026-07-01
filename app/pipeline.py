# pipeline.py — Orchestration layer sitting above predictor.py

from __future__ import annotations

import pytesseract
from PIL import Image
import pandas as pd
import io
import logging
import time
from typing import Optional

# Import your custom exceptions and predictors
from .config import DEFAULT_START, DEFAULT_TICKER
from .exceptions import DataFetchError, InvalidInputError, OCREngineException
from .predictor import predict_multiple, predict_next_close, load_prediction_assets, _build_latest_window
from .utils import normalize_ticker, inverse_close_prices, add_technical_indicators

logger = logging.getLogger(__name__)


class Pipeline:
    """Orchestration layer that wraps predictor calls with validation + caching."""

    CACHE_TTL_SECONDS = 60 * 5  # 5 minutes

    def __init__(
        self,
        ticker_default: str = DEFAULT_TICKER,
        start_default: str = DEFAULT_START,
        cache_ttl_seconds: int = CACHE_TTL_SECONDS,
    ) -> None:
        self.ticker_default = ticker_default
        self.start_default = start_default
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cache: dict[str, tuple[float, dict]] = {}

    def _cache_key(self, ticker: str, start: str) -> str:
        return f"{ticker}:{start}"

    def _get_cached(self, key: str) -> Optional[dict]:
        entry = self._cache.get(key)
        if entry is None:
            return None
        cached_at, value = entry
        if (time.time() - cached_at) > self.cache_ttl_seconds:
            self._cache.pop(key, None)
            return None
        return value

    def _set_cached(self, key: str, value: dict) -> None:
        self._cache[key] = (time.time(), value)

    def clear_cache(self) -> None:
        """Manually invalidate all cached predictions (e.g. after retraining)."""
        self._cache.clear()

    def _validate_ticker_input(self, ticker: str) -> str:
        if not ticker or not ticker.strip():
            raise InvalidInputError("Ticker symbol cannot be empty.")

        ticker = normalize_ticker(ticker)

        if not ticker.replace(".", "").replace("-", "").isalnum():
            raise InvalidInputError(f"Ticker '{ticker}' contains invalid characters.")

        if len(ticker) > 10:
            raise InvalidInputError(f"Ticker '{ticker}' looks too long to be valid.")

        return ticker
    
    def run(
        self,
        ticker: str = None,
        start: str = None,
        use_cache: bool = True,
    ) -> dict:
        ticker = ticker if ticker is not None else self.ticker_default
        start = start if start is not None else self.start_default

        ticker = self._validate_ticker_input(ticker)
        key = self._cache_key(ticker, start)

        if use_cache:
            cached = self._get_cached(key)
            if cached is not None:
                logger.info("Cache hit for %s (start=%s)", ticker, start)
                return {**cached, "cached": True}

        result = predict_next_close(ticker=ticker, start=start)
        result["cached"] = False

        if use_cache:
            self._set_cached(key, result)

        return result

    def run_batch(
        self,
        tickers: list[str],
        start: str = None,
        use_cache: bool = True,
    ) -> dict:
        if not tickers:
            raise InvalidInputError("At least one ticker is required.")

        start = start if start is not None else self.start_default

        results: dict[str, dict] = {}
        uncached_tickers: list[str] = []

        for raw_ticker in tickers:
            try:
                ticker = self._validate_ticker_input(raw_ticker)
            except InvalidInputError as exc:
                results[raw_ticker] = {"ticker": raw_ticker, "error": str(exc)}
                continue

            key = self._cache_key(ticker, start)
            cached = self._get_cached(key) if use_cache else None
            if cached is not None:
                results[ticker] = {**cached, "cached": True}
            else:
                uncached_tickers.append(ticker)

        if uncached_tickers:
            fresh_results = predict_multiple(uncached_tickers, start=start)
            for ticker, result in fresh_results.items():
                if "error" not in result:
                    result["cached"] = False
                    if use_cache:
                        self._set_cached(self._cache_key(ticker, start), result)
                results[ticker] = result

        return results
    
    def execute_prediction_from_file(self, file_bytes: bytes, filename: str) -> dict:
        try:
            if filename.lower().endswith(".csv"):
                df = pd.read_csv(io.BytesIO(file_bytes))
            elif filename.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(io.BytesIO(file_bytes))
            else:
                raise InvalidInputError(f"Unsupported file type: {filename}")

            df = add_technical_indicators(df)
        except Exception as e:
            raise InvalidInputError(f"Failed to read file: {str(e)}")

        try:
            model, scaler, metadata = load_prediction_assets()
            window_size = int(metadata["window_size"])
            input_data = _build_latest_window(df, scaler, window_size)
            
            predicted_scaled = model.predict(input_data, verbose=0)
            
            prediction = float(inverse_close_prices(scaler, predicted_scaled)[0])

            return {
                "status": "success",
                "predicted_price": round(prediction, 2),
                "filename": filename
            }
        except Exception as e:
            logger.error(f"Prediction logic failed: {e}")
            raise Exception(f"Model prediction error: {str(e)}")

pipeline = Pipeline()