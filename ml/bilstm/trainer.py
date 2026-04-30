import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from ml.bilstm.architecture import build_bilstm_model
from app.core.config import config
from app.core.logger import logger


class BiLSTMTrainer:
    """Handles sequence construction, training, and model persistence."""

    def __init__(self):
        self.model: tf.keras.Model = None
        self.history: dict = None

    def _build_sequences(
        self,
        triples: np.ndarray,
        targets: np.ndarray,
        window: int
    ) -> tuple:
        """
        Create (X, y) sliding-window sequences.

        Parameters
        ----------
        triples : (N, 3)  neutrosophic triples
        targets : (N,)    next-day fluctuation values
        window  : int     sequence length W

        Returns
        -------
        X : (N-W, W, 3)
        y : (N-W,)
        """
        if len(triples) != len(targets):
            raise ValueError(
                f"triples length ({len(triples)}) must equal "
                f"targets length ({len(targets)})."
            )
        if len(triples) <= window:
            raise ValueError(
                f"Not enough data: {len(triples)} samples, "
                f"need > {window}."
            )

        X = np.array(
            [triples[i - window:i] for i in range(window, len(triples))],
            dtype=np.float32
        )
        y = np.array(
            [targets[i] for i in range(window, len(targets))],
            dtype=np.float32
        )
        return X, y

    def train(
        self,
        train_triples: np.ndarray,
        train_targets: np.ndarray,
        val_triples: np.ndarray,
        val_targets: np.ndarray,
        ticker: str = "model"
    ) -> dict:
        """
        Train Bi-LSTM on neutrosophic triple sequences.

        Parameters
        ----------
        train_triples : (N_train, 3)
        train_targets : (N_train,)  next-day fluctuations
        val_triples   : (N_val, 3)
        val_targets   : (N_val,)
        ticker        : str  used for model save filename

        Returns
        -------
        dict  training history (loss, val_loss per epoch)
        """
        W = config.SEQUENCE_WINDOW

        X_train, y_train = self._build_sequences(
            train_triples, train_targets, W
        )
        X_val, y_val = self._build_sequences(
            val_triples, val_targets, W
        )

        logger.info(
            f"Training shapes — "
            f"X_train: {X_train.shape}, y_train: {y_train.shape}, "
            f"X_val: {X_val.shape}, y_val: {y_val.shape}"
        )

        self.model = build_bilstm_model()
        self.model.summary(print_fn=logger.info)

        save_path = os.path.join(
            config.BILSTM_MODEL_DIR, f"{ticker}.keras"
        )
        os.makedirs(config.BILSTM_MODEL_DIR, exist_ok=True)

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=config.LR_REDUCE_FACTOR,
                patience=config.LR_REDUCE_PATIENCE,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=save_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
        ]

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config.MAX_EPOCHS,
            batch_size=config.BATCH_SIZE,
            callbacks=callbacks,
            shuffle=False,              # Preserve temporal order
            verbose=1
        )

        self.history = history.history
        best_val = min(self.history['val_loss'])
        logger.info(
            f"Bi-LSTM training complete. "
            f"Best val_loss: {best_val:.6f}. "
            f"Saved → {save_path}"
        )
        return self.history

    def load(self, ticker: str) -> "BiLSTMTrainer":
        path = os.path.join(
            config.BILSTM_MODEL_DIR, f"{ticker}.keras"
        )
        self.model = tf.keras.models.load_model(path)
        logger.info(f"Bi-LSTM loaded from {path}")
        return self
