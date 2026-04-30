import numpy as np


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(predicted - actual)))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = np.abs(actual) > 1e-8
    if mask.sum() == 0:
        return float('nan')
    return float(
        np.mean(
            np.abs(predicted[mask] - actual[mask]) / np.abs(actual[mask])
        ) * 100.0
    )


def theils_u(actual: np.ndarray, predicted: np.ndarray) -> float:
    num = np.sqrt(np.mean((predicted - actual) ** 2))
    den = np.sqrt(np.mean(predicted ** 2)) + np.sqrt(np.mean(actual ** 2))
    return float(num / den) if den > 1e-12 else float('nan')


def directional_accuracy(
    actual: np.ndarray,
    predicted: np.ndarray,
    prev_actual: np.ndarray
) -> float:
    actual_dir = np.sign(actual - prev_actual)
    pred_dir = np.sign(predicted - prev_actual)
    return float(np.mean(actual_dir == pred_dir) * 100.0)


def compute_all_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    prev_actual: np.ndarray
) -> dict:
    return {
        "RMSE": round(rmse(actual, predicted), 6),
        "MAE": round(mae(actual, predicted), 6),
        "MAPE": round(mape(actual, predicted), 4),
        "Theils_U": round(theils_u(actual, predicted), 6),
        "Directional_Accuracy": round(
            directional_accuracy(actual, predicted, prev_actual), 2
        )
    }
