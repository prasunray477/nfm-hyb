import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Bidirectional, LSTM, Dense, Dropout, Input
)
from app.core.config import config


def build_bilstm_model(
    sequence_length: int = config.SEQUENCE_WINDOW,
    n_features: int = 3
) -> tf.keras.Model:
    """
    Construct the Bi-directional LSTM model.

    Input  shape: (batch, sequence_length, 3)  — neutrosophic triples
    Output shape: (batch, 1)                   — predicted fluctuation

    Architecture rationale:
    - Two Bi-LSTM layers: first extracts local patterns (return_sequences=True),
      second distills into a single sequence representation.
    - Dropout after each Bi-LSTM prevents co-adaptation overfitting.
    - Huber loss handles stock price outliers better than MSE.
    """
    model = Sequential([
        Input(shape=(sequence_length, n_features)),

        Bidirectional(
            LSTM(
                config.BILSTM_UNITS_L1,
                return_sequences=True,
                activation='tanh',
                recurrent_activation='sigmoid',
                kernel_regularizer=tf.keras.regularizers.l2(1e-5)
            ),
            name="bilstm_1"
        ),
        Dropout(config.DROPOUT_L1, name="dropout_1"),

        Bidirectional(
            LSTM(
                config.BILSTM_UNITS_L2,
                return_sequences=False,
                activation='tanh',
                recurrent_activation='sigmoid',
                kernel_regularizer=tf.keras.regularizers.l2(1e-5)
            ),
            name="bilstm_2"
        ),
        Dropout(config.DROPOUT_L2, name="dropout_2"),

        Dense(
            config.DENSE_UNITS,
            activation='relu',
            name="dense_1"
        ),
        Dense(1, activation='linear', name="output")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config.LEARNING_RATE,
            clipnorm=1.0              # Gradient clipping for stability
        ),
        loss=tf.keras.losses.Huber(delta=config.HUBER_DELTA),
        metrics=['mae']
    )

    return model
