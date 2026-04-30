import numpy as np
import pytest
from ml.neutrosophic.normalizer import NeutrosophicNormalizer
from ml.neutrosophic.entropy import compute_indeterminacy


class TestNeutrosophicNormalizer:

    def setup_method(self):
        np.random.seed(42)
        self.flucts = np.random.randn(200).astype(np.float32)
        self.normalizer = NeutrosophicNormalizer(entropy_window=10)

    def test_fit_sets_benchmark(self):
        self.normalizer.fit(self.flucts)
        assert self.normalizer.len_benchmark > 0
        assert self.normalizer._fitted is True

    def test_transform_output_shape(self):
        triples = self.normalizer.fit_transform(self.flucts)
        expected_len = len(self.flucts) - 10
        assert triples.shape == (expected_len, 3)

    def test_values_in_unit_interval(self):
        triples = self.normalizer.fit_transform(self.flucts)
        assert triples.min() >= 0.0
        assert triples.max() <= 1.0

    def test_transform_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            self.normalizer.transform(self.flucts)

    def test_entropy_range(self):
        window = np.array([1.0, -1.0, 0.5, -0.5, 0.2])
        L = 0.5
        I = compute_indeterminacy(window, L)
        assert 0.0 <= I <= 1.0

    def test_high_entropy_for_chaotic_market(self):
        # Alternating strong up/down → high entropy
        chaotic = np.array([2.0, -2.0, 2.0, -2.0, 2.0,
                            -2.0, 2.0, -2.0, 2.0, -2.0])
        I = compute_indeterminacy(chaotic, L=1.0)
        assert I > 0.5

    def test_low_entropy_for_trending_market(self):
        # All strong ups → very low entropy
        trending = np.ones(10) * 3.0
        I = compute_indeterminacy(trending, L=1.0)
        assert I < 0.2
