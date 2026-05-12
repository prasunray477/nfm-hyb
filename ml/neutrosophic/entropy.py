import numpy as np


def compute_indeterminacy(
    window: np.ndarray,
    len_benchmark: float = None,
    *,
    L: float = None,
) -> float:
    """
    Compute normalized Shannon entropy over a rolling window
    of fluctuation values. Maps to indeterminacy membership I(U_t).

    Parameters
    ----------
    window        : np.ndarray  shape (m,) last m fluctuations
    len_benchmark : float       mean absolute fluctuation (from fit)
    L             : float       keyword alias for len_benchmark

    Returns
    -------
    float   normalized entropy in [0, 1]
            0 = perfectly consistent trending market
            1 = maximally chaotic/uncertain market
    """
    if len_benchmark is None:
        len_benchmark = L
    if len_benchmark is None:
        raise TypeError("Missing required benchmark length: len_benchmark or L")

    boundaries = [
        -np.inf,
        -1.5 * len_benchmark,
        -0.5 * len_benchmark,
        0.5 * len_benchmark,
        1.5 * len_benchmark,
        np.inf,
    ]

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

    # Normalize against the labels present in this window so balanced
    # two-state reversals still score as high indeterminacy.
    active_labels = int(np.count_nonzero(counts))
    if active_labels <= 1:
        return 0.0

    return float(entropy / np.log2(active_labels))
