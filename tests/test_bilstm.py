import numpy as np
import pytest
import tensorflow as tf
from ml.bilstm.architecture import build_bilstm_model
from ml.bilstm.predictor import BiLSTMPredictor
from app.core.config import config


class TestBiLSTMArchitecture:

    def test_model_builds_without_error(self):
        model = build_bilstm_model()
        assert model is not None

    def test_output_shape(self):
        model = build_bilstm_model()
        batch = np.random.randn(4, config.SEQUENCE_WINDOW, 3).astype(np.float32)
        out = model.predict(batch, verbose=0)
        assert out.shape == (4, 1)

    def test_model_has_correct_layers(self):
        model = build_bilstm_model()
        layer_names = [l.name for l in model.layers]
        assert "bilstm_1" in layer_names
        assert "bilstm_2" in layer_names
        assert "output" in layer_names

    def test_trainable_parameters_positive(self):
        model = build_bilstm_model()
        assert model.count_params() > 0


class TestBiLSTMPredictor:

    def setup_method(self):
        self.model = build_bilstm_model()
        # Initialize weights with random data
        dummy = np.random.randn(
            2, config.SEQUENCE_WINDOW, 3
        ).astype(np.float32)
        self.model.predict(dummy, verbose=0)
        self.predictor = BiLSTMPredictor(self.model)

    def test_predict_series_output_length(self):
        W = config.SEQUENCE_WINDOW
        n = W + 50
        triples = np.random.rand(n, 3).astype(np.float32)
        preds = self.predictor.predict_series(triples)
        assert len(preds) == 50

    def test_predictions_are_finite(self):
        W = config.SEQUENCE_WINDOW
        triples = np.random.rand(W + 20, 3).astype(np.float32)
        preds = self.predictor.predict_series(triples)
        assert np.all(np.isfinite(preds))

    def test_short_input_raises(self):
        with pytest.raises(ValueError):
            self.predictor.predict_series(
                np.random.rand(5, 3).astype(np.float32)
            )
