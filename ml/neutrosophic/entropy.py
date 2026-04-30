import numpy as np


def compute_indeterminacy(
    window: np.ndarray,
    len_benchmark: float
) -> float:
    """
    Compute normalized Shannon entropy over a rolling window
    of fluctuation values. Maps to indeterminacy membership I(U_t).

    Parameters
    ----------
    window        : np.ndarray  shape (m,) last m fluctuations
    len_benchmark : float       mean absolute fluctuation (from fit)

    Returns
    -------
    float   normalized entropy in [0, 1]
            0 = perfectly consistent trending market
            1 = maximally chaotic/uncertain market
    """
    L = len_benchmark
    boundaries = [-np.inf, -1.5*L, -0.5*L, 0.5*L, 1.5*L, np.inf]

    counts = np.zeros(5, dtype=np.float64)
    for val in window:
        for i in range(5):
            if boundaries[i] <= val < boundaries[i + 1]:
                counts[i] += 1.0
                break

    m = float(len(window))
    probabilities = counts / m

    # Shannon entropy: 0·log₂(0) ≡ 0 by convention
    entropy = 0.0
    for p in probabilities:
        if p > 1e-12:
            entropy -= p * np.log2(p)

    # Normalize: max entropy with 5 labels = log₂(5) ≈ 2.3219
    return float(entropy / np.log2(5))
