import numpy as np
import os
import pandas as pd
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
            f"═══ Pipeline START: {self.ticker} ═══"
        )

        # ── Phase 1: Data ──────────────────────────────────────────
        logger.info("Phase 1: Data collection & stationarity...")
        prices = DataService.fetch(self.ticker, self.period)
        dates = DataService.last_dates
        if len(dates) != len(prices):
            dates = pd.bdate_range(
                end=pd.Timestamp.today(), periods=len(prices)
            ).strftime("%Y-%m-%d").tolist()
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
        train_triple_dates = dates[m + 1:train_end + 1]
        val_triple_dates = dates[train_end + m + 1:val_end + 1]
        test_triple_dates = dates[val_end + m + 1:len(fluctuations) + 1]

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
            f"═══ Pipeline COMPLETE: {self.ticker} ═══ "
            f"RMSE={metrics['RMSE']:.4f} | "
            f"DA={metrics['Directional_Accuracy']:.1f}%"
        )

        # ── Store intermediate state for forecast_future() ─────────
        self._fluct_train = fluct_train
        self._fluct_val   = fluct_val
        self._fluct_test  = fluct_test
        self._prices_full = prices
        self._dates_full = dates
        self._triples_full = np.vstack([triples_train, triples_val, triples_test])
        self._triple_dates_full = (
            train_triple_dates + val_triple_dates + test_triple_dates
        )
        self._actual_test    = actual_test.tolist()
        self._predicted_test = final_test.tolist()
        self._metrics = metrics

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
            if self.arima.order else None,
            "historical_dates": dates,
            "historical_prices": prices.tolist(),
            "daily_movements": fluctuations.tolist(),
            "movement_dates": dates[1:len(fluctuations) + 1],
            "train_end_idx": int(train_end),
            "triples": self._triples_full.tolist(),
            "triple_dates": self._triple_dates_full,
        }

    # ── 30-Day Future Projection ───────────────────────────────────

    def forecast_future(self, n_days: int = 30) -> dict:
        """
        Generate n_days forward predictions beyond the last known price.

        Strategy:
        - Use the last SEQUENCE_WINDOW neutrosophic triples from the full
          dataset as the seed window for Bi-LSTM rolling prediction.
        - Use the fitted ARIMA model to forecast n_days ahead directly.
        - Combine with stored alpha_star weights.
        - Reconstruct prices from predicted fluctuations.
        - Generate upper/lower confidence bands using ±1.5 × std(test_errors).

        Returns dict with keys:
          future_dates, bilstm_future, arima_future, combined_future,
          upper_band, lower_band, last_known_price, trend_label,
          trend_strength, recommendation, recommendation_reason,
          projected_change_pct, confidence_margin
        """
        import pandas as pd
        from datetime import timedelta
        from statsmodels.tsa.arima.model import ARIMA

        # Generate future business dates after the final market data point.
        last_date = pd.Timestamp(self._dates_full[-1])
        future_dates = []
        d = last_date
        while len(future_dates) < n_days:
            d += timedelta(days=1)
            if d.weekday() < 5:  # Monday=0 ... Friday=4
                future_dates.append(d.strftime('%Y-%m-%d'))

        # ARIMA multi-step forecast on full history
        full_series = np.concatenate([
            self._fluct_train, self._fluct_val, self._fluct_test
        ])
        arima_model = ARIMA(
            full_series.astype(np.float64), order=self.arima.order
        ).fit()
        arima_forecast_raw = arima_model.forecast(steps=n_days)
        # Handle both Series and ndarray returns
        if hasattr(arima_forecast_raw, 'values'):
            arima_raw = arima_forecast_raw.values
        else:
            arima_raw = np.asarray(arima_forecast_raw)

        # Bi-LSTM rolling forecast using last window
        last_window = self._triples_full[-config.SEQUENCE_WINDOW:]
        bilstm_raw = []
        current_window = last_window.copy()

        for step in range(n_days):
            x = current_window.reshape(1, config.SEQUENCE_WINDOW, 3)
            pred_fluct = float(
                self.bilstm_trainer.model.predict(x, verbose=0)[0][0]
            )
            bilstm_raw.append(pred_fluct)
            # Synthesize next triple from predicted fluctuation
            next_T = self.normalizer._truth(pred_fluct)
            next_F = self.normalizer._falsity(pred_fluct)
            next_I = float(np.mean([current_window[-1, 1]]))
            next_triple = np.array(
                [[next_T, next_I, next_F]], dtype=np.float32
            )
            current_window = np.vstack([current_window[1:], next_triple])

        bilstm_raw = np.array(bilstm_raw, dtype=np.float32)

        # Reconstruct prices from fluctuations
        last_price = float(self._prices_full[-1])
        bilstm_prices, arima_prices, combined_prices = [], [], []
        current_price = last_price

        for i in range(n_days):
            bl = current_price + float(bilstm_raw[i])
            ar = current_price + float(arima_raw[i])
            cb = (self.static_optimizer.alpha_star * bl
                  + self.static_optimizer.beta_star * ar)
            bilstm_prices.append(bl)
            arima_prices.append(ar)
            combined_prices.append(cb)
            current_price = cb

        combined_prices = np.array(combined_prices)

        # Confidence bands from test set error
        test_errors = np.abs(
            np.array(self._actual_test) - np.array(self._predicted_test)
        )
        margin = float(np.std(test_errors) * 1.5)
        upper_band = (combined_prices + margin).tolist()
        lower_band = (combined_prices - margin).tolist()

        # Trend detection
        price_change_pct = (
            (combined_prices[-1] - last_price) / last_price * 100
        )
        if price_change_pct > 3.0:
            trend_label = "Bullish"
            trend_strength = min(100, price_change_pct * 8)
        elif price_change_pct < -3.0:
            trend_label = "Bearish"
            trend_strength = min(100, abs(price_change_pct) * 8)
        else:
            trend_label = "Neutral"
            trend_strength = max(0, 50 - abs(price_change_pct) * 5)

        # Investment recommendation
        da = self._metrics.get('Directional_Accuracy', 50)
        rmse = self._metrics.get('RMSE', 999)

        if (trend_label == "Bullish" and da > 60
                and price_change_pct > 5):
            rec = "BUY"
            reason = (
                f"The model projects a {price_change_pct:.1f}% price "
                f"increase over 30 days with {da:.1f}% trend prediction "
                f"accuracy. Strong bullish signal with high model confidence."
            )
        elif (trend_label == "Bearish" and da > 60
              and price_change_pct < -5):
            rec = "SELL"
            reason = (
                f"The model projects a {abs(price_change_pct):.1f}% price "
                f"decline over 30 days with {da:.1f}% trend prediction "
                f"accuracy. Strong bearish signal — consider reducing "
                f"exposure."
            )
        else:
            rec = "HOLD"
            reason = (
                f"The model projects a {price_change_pct:.1f}% price "
                f"change over 30 days. The signal is not strong enough "
                f"({da:.1f}% accuracy) to recommend directional action. "
                f"Monitor for clearer trend confirmation."
            )

        return {
            "future_dates": future_dates,
            "bilstm_future": bilstm_prices,
            "arima_future": arima_prices,
            "combined_future": combined_prices.tolist(),
            "upper_band": upper_band,
            "lower_band": lower_band,
            "last_known_price": last_price,
            "trend_label": trend_label,
            "trend_strength": round(float(trend_strength), 1),
            "recommendation": rec,
            "recommendation_reason": reason,
            "projected_change_pct": round(float(price_change_pct), 2),
            "confidence_margin": round(margin, 4),
        }
