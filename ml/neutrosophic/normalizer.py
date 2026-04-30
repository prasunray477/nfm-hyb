import numpy as np
from app.core.config import config
from app.core.logger import logger
from ml.neutrosophic.entropy import compute_indeterminacy


class NeutrosophicNormalizer:
    """
    Converts raw fluctuation series into neutrosophic triples.

    Each trading day t becomes X_t = (T(U_t), I(U_t), F(U_t))
    where T=bullish strength, I=market entropy, F=bearish strength.

    Usage
    -----
    normalizer = NeutrosophicNormalizer()
    triples_train = normalizer.fit_transform(fluct_train)
    triples_val   = normalizer.transform(fluct_val)
    triples_test  = normalizer.transform(fluct_test)
    """

    def __init__(self, entropy_window: int = config.ENTROPY_WINDOW):
        self.entropy_window = entropy_window
        self.len_benchmark: float = None
        self._fitted: bool = False

    # ── Public API ────────────────────────────────────────────────

    def fit(self, fluctuations: np.ndarray) -> "NeutrosophicNormalizer":
        """
        Compute benchmark length from training fluctuations.
        Must only be called on training data.
        """
        self.len_benchmark = float(np.mean(np.abs(fluctuations)))

        if self.len_benchmark < 1e-8:
            raise ValueError(
                "Benchmark length near zero — "
                "check that fluctuations are non-trivial."
            )

        self._fitted = True
        logger.info(
            f"NeutrosophicNormalizer fitted. "
            f"len_benchmark = {self.len_benchmark:.6f}"
        )
        return self

    def transform(self, fluctuations: np.ndarray) -> np.ndarray:
        """
        Transform fluctuation series into neutrosophic triples.

        Parameters
        ----------
        fluctuations : np.ndarray  shape (N,)

        Returns
        -------
        np.ndarray  shape (N - entropy_window, 3)
                    columns: [T, I, F]
        """
        if not self._fitted:
            raise RuntimeError(
                "Call fit() on training data before transform()."
            )

        m = self.entropy_window
        n = len(fluctuations)

        if n <= m:
            raise ValueError(
                f"Fluctuation series length ({n}) must exceed "
                f"entropy window ({m})."
            )

        triples = np.zeros((n - m, 3), dtype=np.float32)

        for t in range(m, n):
            T_val = self._truth(fluctuations[t])
            F_val = self._falsity(fluctuations[t])
            I_val = compute_indeterminacy(
                fluctuations[t - m:t],
                self.len_benchmark
            )
            triples[t - m] = [T_val, I_val, F_val]

        logger.info(
            f"Neutrosophic transform complete. "
            f"Input: {n}, Output shape: {triples.shape}. "
            f"T mean: {triples[:,0].mean():.3f}, "
            f"I mean: {triples[:,1].mean():.3f}, "
            f"F mean: {triples[:,2].mean():.3f}"
        )
        return triples

    def fit_transform(self, fluctuations: np.ndarray) -> np.ndarray:
        return self.fit(fluctuations).transform(fluctuations)

    # ── Private membership functions ──────────────────────────────

    def _truth(self, u: float) -> float:
        """T(U_t): bullish momentum strength ∈ [0,1]"""
        L = self.len_benchmark
        if u <= -0.5 * L:
            return 0.0
        if u >= L:
            return 1.0
        return float(u / (1.5 * L) + 1.0 / 3.0)

    def _falsity(self, u: float) -> float:
        """F(U_t): bearish pressure strength ∈ [0,1]"""
        L = self.len_benchmark
        if u >= 0.5 * L:
            return 0.0
        if u <= -L:
            return 1.0
        return float(-u / (1.5 * L) + 1.0 / 3.0)
