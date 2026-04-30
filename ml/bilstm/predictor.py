import numpy as np
import tensorflow as tf
from app.core.config import config
from app.core.logger import logger


class BiLSTMPredictor:
    """Generates predictions from a trained Bi-LSTM model."""

    def __init__(self, model: tf.keras.Model):
        self.model = model
        self._W = config.SEQUENCE_WINDOW

    def predict_series(self, triples: np.ndarray) -> np.ndarray:
        """
        Generate one-step-ahead predicted fluctuations
        for all valid windows in the input triples.

        Parameters
        ----------
        triples : np.ndarray  shape (N, 3)  neutrosophic triples

        Returns
        -------
        np.ndarray  shape (N - W,)  predicted fluctuations
        """
        W = self._W
        n = len(triples)

        if n <= W:
            raise ValueError(
                f"Input length {n} must exceed window {W}."
            )

        # Batch all windows at once for efficiency
        X = np.array(
            [triples[i - W:i] for i in range(W, n)],
            dtype=np.float32
        )

        predictions = self.model.predict(
            X, batch_size=config.BATCH_SIZE, verbose=0
        ).flatten()

        logger.info(
            f"Bi-LSTM predictions generated. "
            f"Shape: {predictions.shape}"
        )
        return predictions.astype(np.float32)
