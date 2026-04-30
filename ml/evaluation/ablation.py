import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List
from app.core.logger import logger
from ml.evaluation.metrics import compute_all_metrics
from sklearn.preprocessing import MinMaxScaler


@dataclass
class AblationResult:
    config_id: str
    description: str
    metrics: Dict[str, float] = field(default_factory=dict)


class AblationStudy:
    """
    Runs 6 ablation configurations on the same test data
    to isolate contribution of each pipeline component.

    Configurations:
    ───────────────
    C1: ARIMA only              (standard input)
    C2: Bi-LSTM only            (standard MinMax normalization)
    C3: Bi-LSTM only            (neutrosophic normalization)
    C4: ARIMA + Bi-LSTM hybrid  (standard norm, equal weights α=0.5)
    C5: ARIMA + Bi-LSTM hybrid  (neutrosophic norm, equal weights α=0.5)
    C6: ARIMA + Bi-LSTM hybrid  (neutrosophic norm, optimized α*)  ← FULL MODEL
    """

    CONFIGS = [
        ("C1", "ARIMA only (linear baseline)"),
        ("C2", "Bi-LSTM only (standard MinMax norm)"),
        ("C3", "Bi-LSTM only (neutrosophic norm)"),
        ("C4", "Hybrid ARIMA+Bi-LSTM (standard norm, α=0.50)"),
        ("C5", "Hybrid ARIMA+Bi-LSTM (neutrosophic norm, α=0.50)"),
        ("C6", "Hybrid ARIMA+Bi-LSTM (neutrosophic norm, α=α*) [FULL MODEL]"),
    ]

    def run(
        self,
        actual_test: np.ndarray,
        prev_actual: np.ndarray,
        arima_test: np.ndarray,
        bilstm_test_standard: np.ndarray,
        bilstm_test_neutro: np.ndarray,
        alpha_star: float
    ) -> List[AblationResult]:
        """
        Compute metrics for all 6 configurations.

        Parameters
        ----------
        actual_test           : actual prices on test set
        prev_actual           : prices one step before test set (for DA)
        arima_test            : ARIMA-only price forecasts
        bilstm_test_standard  : Bi-LSTM forecasts with standard MinMax input
        bilstm_test_neutro    : Bi-LSTM forecasts with neutrosophic input
        alpha_star            : optimized weight for Bi-LSTM

        Returns
        -------
        List[AblationResult]  metrics for each configuration
        """
        results = []

        predictions = {
            "C1": arima_test,
            "C2": bilstm_test_standard,
            "C3": bilstm_test_neutro,
            "C4": 0.5 * bilstm_test_standard + 0.5 * arima_test,
            "C5": 0.5 * bilstm_test_neutro   + 0.5 * arima_test,
            "C6": alpha_star * bilstm_test_neutro
                  + (1 - alpha_star) * arima_test,
        }

        for config_id, description in self.CONFIGS:
            pred = predictions[config_id]

            # Align lengths
            n = min(len(actual_test), len(pred), len(prev_actual))
            metrics = compute_all_metrics(
                actual=actual_test[:n],
                predicted=pred[:n],
                prev_actual=prev_actual[:n]
            )

            result = AblationResult(
                config_id=config_id,
                description=description,
                metrics=metrics
            )
            results.append(result)

            logger.info(
                f"Ablation {config_id}: "
                f"RMSE={metrics['RMSE']:.4f} | "
                f"MAPE={metrics['MAPE']:.2f}% | "
                f"DA={metrics['Directional_Accuracy']:.1f}%"
            )

        # Log improvement summary
        c1_rmse = results[0].metrics["RMSE"]
        c6_rmse = results[5].metrics["RMSE"]
        improvement = (c1_rmse - c6_rmse) / c1_rmse * 100
        logger.info(
            f"Full model improvement over ARIMA baseline: "
            f"{improvement:.1f}% RMSE reduction"
        )

        return results

    def to_dict(self, results: List[AblationResult]) -> List[dict]:
        return [
            {
                "config_id": r.config_id,
                "description": r.description,
                **r.metrics
            }
            for r in results
        ]

    def friedman_ranking(
        self, results: List[AblationResult]
    ) -> Dict[str, float]:
        """
        Compute average RMSE ranks across configurations.
        Lower rank = better model. Used for Friedman significance test.
        """
        rmse_values = [r.metrics["RMSE"] for r in results]
        sorted_ids = sorted(
            range(len(rmse_values)), key=lambda i: rmse_values[i]
        )
        ranks = {
            results[sorted_ids[i]].config_id: i + 1
            for i in range(len(results))
        }
        logger.info(f"Friedman ranks: {ranks}")
        return ranks
