

# Defines the Architecture , Loads Trained Weights


from keras.models import Sequential, load_model
from keras.layers import GRU, Dense, Dropout, Input
from keras.optimizers import Adam
from .config import settings
from .exceptions import ModelNotFoundError

class StockModel:
    def __init__(self):
        self.model = None

    def build_model(self, window_size: int) -> Sequential:
        """Constructs the GRU architecture."""
        model = Sequential([
            Input(shape=(window_size, len(settings.FEATURE_COLUMNS))),
            GRU(units=64, return_sequences=True),
            Dropout(0.2),
            GRU(units=32, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.0001), loss="mse")
        return model

    def load_weights(self, window_size: int):
        """Loads trained weights with production error checking."""
        if not settings.MODEL_WEIGHTS_PATH.exists():
            raise ModelNotFoundError(str(settings.MODEL_WEIGHTS_PATH))
            
        try:
            self.model = self.build_model(window_size)
            self.model.load_weights(settings.MODEL_WEIGHTS_PATH)
            print("Model weights loaded successfully.")
        except Exception as e:
            raise Exception(f"Critical error loading model weights: {e}")

    def predict(self, data):
        """Standardized prediction method for real-world inference."""
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load_weights() first.")
        return self.model.predict(data)
