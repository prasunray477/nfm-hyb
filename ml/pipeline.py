import numpy as np
import os
from app.core.config import config
from app.core.logger import logger
from app.services.data_service import DataService
from app.utils.stationarity import enforce_stationarity
from ml.neutrosophic.normalizer import NeutrosophicNormalizer
from ml.arima.model import ARIMAForecaster
from ml.bilstm.trainer import BiLSTMTrainer
from ml.bilstm.predictor import BiLSTMPredictor
from ml.aggregator.static_weights import StaticWeightOptimizer
from ml.aggregator.adaptive_weights import AdaptiveWeightAggregator
from ml.evaluation.metrics import compute_all_metrics


class ForecastPipeline:
    """
    Orchestrates all 7 phases of the forecasting system.
    Single entry point for the FastAPI backend.
    """

    def __init__(
        self,
        ticker: str,
        use_adaptive: bool = False,
        period: str = config.DEFAULT_PERIOD
    ):
        self.ticker = ticker.upper()
        self.use_adaptive = use_adaptive
        self.period = period

        self.normalizer = NeutrosophicNormalizer(
            entropy_window=config.ENTROPY_WINDOW
        )
        self.arima = ARIMAForecaster()
        self.bilstm_trainer = BiLSTMTrainer()
        self.static_optimizer = StaticWeightOptimizer()
        self.adaptive_aggregator = AdaptiveWeightAggregator(
            window=config.ADAPTIVE_WINDOW
        )

    def run(self) -> dict:
        logger.info(
            f"Pipeline START: {self.ticker}"
        )

        # ── Phase 1: Data ──────────────────────────────────────────
        logger.info("Phase 1: Data collection & stationarity...")
        prices = DataService.fetch(self.ticker, self.period)
        fluctuations = enforce_stationarity(prices)
        n = len(fluctuations)

        # ── Phase 3: Temporal Split ────────────────────────────────
        train_end = int(n * config.TRAIN_RATIO)
        val_end = int(n * (config.TRAIN_RATIO + config.VAL_RATIO))

        fluct_train = fluctuations[:train_end]
        fluct_val = fluctuations[train_end:val_end]
        fluct_test = fluctuations[val_end:]

        prices_train = prices[:train_end]
        prices_val = prices[train_end:val_end]
        prices_test = prices[val_end:]

        logger.info(
            f"Split: train={len(fluct_train)}, "
            f"val={len(fluct_val)}, test={len(fluct_test)}"
        )

        # ── Phase 2: Neutrosophic Normalization ────────────────────
        logger.info("Phase 2: Neutrosophic normalization...")
        triples_train = self.normalizer.fit_transform(fluct_train)
        triples_val = self.normalizer.transform(fluct_val)
        triples_test = self.normalizer.transform(fluct_test)

        m = config.ENTROPY_WINDOW

        # ── Phase 4: ARIMA ─────────────────────────────────────────
        logger.info("Phase 4: Fitting ARIMA model...")
        self.arima.fit(fluct_train)

        # Validation ARIMA forecasts (expanding window)
        arima_val_flucts = []
        for i in range(len(fluct_val)):
            hist = np.concatenate([fluct_train, fluct_val[:i]])
            arima_val_flucts.append(
                self.arima.forecast_one_step(hist)
            )
        arima_val_flucts = np.array(arima_val_flucts, dtype=np.float32)

        # Test ARIMA forecasts
        arima_test_flucts = []
        full_hist_base = np.concatenate([fluct_train, fluct_val])
        for i in range(len(fluct_test)):
            hist = np.concatenate([full_hist_base, fluct_test[:i]])
            arima_test_flucts.append(
                self.arima.forecast_one_step(hist)
            )
        arima_test_flucts = np.array(arima_test_flucts, dtype=np.float32)

        # ── Phase 5: Bi-LSTM ───────────────────────────────────────
        logger.info("Phase 5: Training Bi-LSTM model...")

        # Targets = next-day fluctuations aligned to triples
        # triples_train covers indices m..train_end
        # corresponding next-day fluctuation: fluct_train[m+1..train_end]
        train_targets = fluct_train[m + 1:][:len(triples_train)]
        val_targets = fluct_val[1:][:len(triples_val)]

        # Align targets length with triples
        min_tr = min(len(triples_train), len(train_targets))
        min_vl = min(len(triples_val), len(val_targets))

        self.bilstm_trainer.train(
            train_triples=triples_train[:min_tr],
            train_targets=train_targets[:min_tr],
            val_triples=triples_val[:min_vl],
            val_targets=val_targets[:min_vl],
            ticker=self.ticker
        )

        predictor = BiLSTMPredictor(self.bilstm_trainer.model)
        bilstm_val_flucts = predictor.predict_series(triples_val)
        bilstm_test_flucts = predictor.predict_series(triples_test)

        # Convert fluctuation predictions → price predictions
        # Align length: use prices shifted by 1 as base
        def to_price(base_prices, fluct_preds, offset=0):
            base = base_prices[offset:offset + len(fluct_preds)]
            return (base + fluct_preds).astype(np.float32)

        # Validation price forecasts
        bilstm_val_prices = to_price(prices_val, bilstm_val_flucts)
        arima_val_prices = to_price(prices_val, arima_val_flucts)

        # Test price forecasts
        bilstm_test_prices = to_price(prices_test, bilstm_test_flucts)
        arima_test_prices = to_price(prices_test, arima_test_flucts)

        # Align actuals
        min_val = min(
            len(bilstm_val_prices), len(arima_val_prices),
            len(prices_val) - 1
        )
        min_test = min(
            len(bilstm_test_prices), len(arima_test_prices),
            len(prices_test) - 1
        )

        actual_val = prices_val[1:min_val + 1]
        actual_test = prices_test[1:min_test + 1]

        bilstm_val_prices = bilstm_val_prices[:min_val]
        arima_val_prices = arima_val_prices[:min_val]
        bilstm_test_prices = bilstm_test_prices[:min_test]
        arima_test_prices = arima_test_prices[:min_test]

        # ── Phase 6: Weighted Aggregation ─────────────────────────
        logger.info("Phase 6: Optimizing ensemble weights...")
        self.static_optimizer.optimize(
            bilstm_val_prices, arima_val_prices, actual_val
        )

        if self.use_adaptive:
            logger.info("Using adaptive inverse-error weighting...")
            final_test = self.adaptive_aggregator.combine(
                bilstm_test_prices,
                arima_test_prices,
                actual_test
            )
        else:
            final_test = self.static_optimizer.combine(
                bilstm_test_prices,
                arima_test_prices
            )

        # ── Phase 7: Evaluation ────────────────────────────────────
        logger.info("Phase 7: Computing evaluation metrics...")
        metrics = compute_all_metrics(
            actual=actual_test,
            predicted=final_test,
            prev_actual=prices_test[:min_test]
        )

        logger.info(
            f"Pipeline COMPLETE: {self.ticker} "
            f"RMSE={metrics['RMSE']:.4f} | "
            f"DA={metrics['Directional_Accuracy']:.1f}%"
        )

        return {
            "ticker": self.ticker,
            "metrics": metrics,
            "alpha": self.static_optimizer.alpha_star,
            "beta": self.static_optimizer.beta_star,
            "actual": actual_test.tolist(),
            "predicted": final_test.tolist(),
            "bilstm_only": bilstm_test_prices.tolist(),
            "arima_only": arima_test_prices.tolist(),
            "n_test_days": int(min_test),
            "arima_order": list(self.arima.order)
            if self.arima.order else None
        }
