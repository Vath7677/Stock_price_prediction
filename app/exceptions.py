# exceptions.py — Custom error handling for the stock prediction app

# exceptions.py — Custom error handling for the stock prediction app

from __future__ import annotations

import traceback
from enum import IntEnum
from typing import Optional


class ErrorCode(IntEnum):
    MODEL_NOT_FOUND = 1001
    DATA_FETCH_FAILED = 1002
    PREDICTION_FAILED = 1003
    INVALID_INPUT = 1004
    OCR_ENGINE_FAILED = 1005 


class StockPredictionError(Exception):
    def __init__(self, message: str, error_code: ErrorCode):
        self.error_code = error_code
        self.message = message

        stack = traceback.extract_stack()[:-1]
        if stack:
            frame = stack[-1]
            self.file_name = frame.filename
            self.line_number = frame.lineno
        else:
            self.file_name = "Unknown"
            self.line_number = "Unknown"

        super().__init__(
            f"[{self.error_code.name}={self.error_code.value}] {self.message} "
            f"(File: {self.file_name}, Line: {self.line_number})"
        )

    def to_dict(self) -> dict:
        return {
            "error_code": int(self.error_code),
            "error_name": self.error_code.name,
            "message": self.message,
        }


class ModelNotFoundError(StockPredictionError):
    def __init__(self, message: str):
        super().__init__(message=message, error_code=ErrorCode.MODEL_NOT_FOUND)


class DataFetchError(StockPredictionError):
    def __init__(self, message: str, ticker: Optional[str] = None):
        self.ticker = ticker
        full_message = f"[{ticker}] {message}" if ticker else message
        super().__init__(message=full_message, error_code=ErrorCode.DATA_FETCH_FAILED)


class PredictionError(StockPredictionError):
    def __init__(self, message: str):
        super().__init__(message=message, error_code=ErrorCode.PREDICTION_FAILED)


class InvalidInputError(StockPredictionError):
    def __init__(self, message: str):
        super().__init__(message=message, error_code=ErrorCode.INVALID_INPUT)

class OCREngineException(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(self.message)