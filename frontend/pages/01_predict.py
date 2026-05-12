import streamlit as st
import httpx
import numpy as np
import os
import time
from components.charts import (
    plot_price_history,
    plot_daily_movements,
    plot_future_projection,
    plot_forecast_vs_actual,
    plot_error_distribution,
)
from components.metrics_display import show_metrics_v2
from components.neutrosophic_viz import plot_neutrosophic_mapping

API_URL = st.secrets.get("API_URL", os.getenv("API_URL", "http://localhost:8000"))

st.title("\U0001f52e Stock Price Forecast")
st.markdown(
    "Run the full Neutrosophic Bi-LSTM + ARIMA analysis engine on any stock symbol."
)

with st.form("predict_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        ticker = st.text_input(
            "Stock Symbol",
            value="AAPL",
            help="Examples: AAPL, MSFT, GOOGL, RELIANCE.NS, TCS.NS",
        )
    with col2:
        period = st.selectbox(
            "Historical Period",
            options=["3y", "5y", "7y", "10y"],
            index=1,
        )
    with col3:
        use_adaptive = st.checkbox(
            "Adaptive Weights",
            value=False,
            help="Dynamically adjusts model weights based on rolling accuracy",
        )
    submitted = st.form_submit_button("\u25b6 Run Forecast", type="primary")

if submitted:
    # ── Staged progress bar ────────────────────────────────────────
    progress = st.progress(0, text="Connecting to market data source\u2026")
    time.sleep(0.3)
    progress.progress(10, text="Downloading historical price data\u2026")
    time.sleep(0.3)
    progress.progress(20, text="Running stationarity checks\u2026")
    time.sleep(0.3)
    progress.progress(30, text="Encoding market signals (T, I, F)\u2026")
    time.sleep(0.3)
    progress.progress(40, text="Sending request to analysis engine\u2026")

    try:
        response = httpx.post(
            f"{API_URL}/api/predict",
            json={
                "ticker": ticker.upper(),
                "use_adaptive": use_adaptive,
                "period": period,
            },
            timeout=900.0,
        )
        progress.progress(50, text="Training Deep Learning model\u2026")
        response.raise_for_status()
        data = response.json()

        progress.progress(85, text="Fitting Statistical model\u2026")
        time.sleep(0.2)
        progress.progress(95, text="Optimizing ensemble weights\u2026")
        time.sleep(0.2)
        progress.progress(100, text="Analysis complete.")
        time.sleep(0.3)
        progress.empty()

        st.success(f"\u2705 Forecast complete for **{ticker.upper()}**")

        # ── Four-tab layout ────────────────────────────────────────
        tab_market, tab_signal, tab_outlook, tab_perf = st.tabs([
            "\U0001f4ca  Market Data",
            "\U0001f52c  Market Signal Encoding",
            "\U0001f52e  30-Day Outlook",
            "\U0001f4c8  Model Performance",
        ])

        # ────────────── Tab 1: Market Data ──────────────────────────
        with tab_market:
            st.subheader("Historical Price & Daily Movement")

            if all(k in data for k in (
                "historical_prices", "historical_dates", "train_end_idx"
            )):
                plot_price_history(
                    prices=data["historical_prices"],
                    dates=data["historical_dates"],
                    train_end_idx=data["train_end_idx"],
                    ticker=ticker.upper(),
                )

            if all(k in data for k in ("daily_movements", "movement_dates")):
                plot_daily_movements(
                    fluctuations=data["daily_movements"],
                    dates=data["movement_dates"],
                    ticker=ticker.upper(),
                )

        # ────────────── Tab 2: Market Signal Encoding ───────────────
        with tab_signal:
            st.subheader("Neutrosophic Market Signal Encoding")
            if all(k in data for k in ("triples", "triple_dates")):
                plot_neutrosophic_mapping(
                    triples=np.array(data["triples"], dtype=np.float32),
                    dates_trimmed=data["triple_dates"],
                    ticker=ticker.upper(),
                )

        # ────────────── Tab 3: 30-Day Outlook ───────────────────────
        with tab_outlook:
            # Recommendation banner
            rec = data.get("recommendation", "HOLD")
            trend = data.get("trend_label", "Neutral")
            change = data.get("projected_change_pct", 0.0)
            strength = data.get("trend_strength", 0.0)
            reason = data.get("recommendation_reason", "")

            colors = {"BUY": "#0D9488", "HOLD": "#F59E0B", "SELL": "#F43F5E"}
            icons = {"BUY": "\U0001f4c8", "HOLD": "\u23f8", "SELL": "\U0001f4c9"}
            bg = {"BUY": "#0A2018", "HOLD": "#1C1500", "SELL": "#1C0A0A"}

            st.markdown(f"""
<div style="background:{bg[rec]};border:1.5px solid {colors[rec]};
border-radius:8px;padding:18px 24px;margin-bottom:16px;">
  <div style="display:flex;align-items:center;gap:16px;">
    <span style="font-size:2.2rem">{icons[rec]}</span>
    <div>
      <div style="color:{colors[rec]};font-size:1.5rem;font-weight:700;
      letter-spacing:2px">{rec} \u2014 {trend} Signal</div>
      <div style="color:#94A3B8;font-size:0.92rem;margin-top:4px">{reason}</div>
    </div>
    <div style="margin-left:auto;text-align:right">
      <div style="color:{colors[rec]};font-size:1.8rem;font-weight:700">
        {'\u2191' if change > 0 else '\u2193'}{abs(change):.1f}%
      </div>
      <div style="color:#64748B;font-size:0.82rem">Projected 30-day change</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

            # Trend strength gauge
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                st.markdown(f"**Trend Strength:** {strength:.0f} / 100")
                st.progress(min(int(strength), 100))
                da_val = data.get("metrics", {}).get("Directional_Accuracy", 0)
                margin_val = data.get("confidence_margin", 0)
                st.caption(
                    f"Confidence Margin: \u00b1{margin_val:.2f} per forecast point | "
                    f"Model Trend Prediction Accuracy: {da_val:.1f}%"
                )

            # Future projection chart
            if all(k in data for k in (
                "future_dates", "combined_future", "upper_band",
                "lower_band", "bilstm_future", "arima_future",
                "trend_label", "last_known_price",
            )):
                plot_future_projection(data, ticker.upper())

            # Disclaimer
            st.caption(
                "\u26a0\ufe0f  This projection is generated by a machine learning model "
                "trained on historical price patterns. It is not financial advice. "
                "Past performance does not guarantee future results. Always conduct "
                "independent research before making investment decisions."
            )

        # ────────────── Tab 4: Model Performance ────────────────────
        with tab_perf:
            st.subheader("Performance Metrics & Forecast Accuracy")

            # Metrics cards
            show_metrics_v2(data.get("metrics", {}))

            # Ensemble weight display
            st.markdown("#### Ensemble Model Weights (Optimized on Validation Data)")
            colA, colB = st.columns(2)
            colA.metric(
                "Deep Learning Model Weight",
                f"{data.get('alpha', 0):.0%}",
                help="Proportion of the final forecast contributed by the Bi-directional LSTM model.",
            )
            colB.metric(
                "Statistical Model Weight",
                f"{data.get('beta', 0):.0%}",
                help="Proportion of the final forecast contributed by the ARIMA statistical model.",
            )

            # Forecast vs actual chart
            if all(k in data for k in ("actual", "predicted", "bilstm_only", "arima_only")):
                plot_forecast_vs_actual(data, ticker.upper())

            # Error distribution chart
            if "actual" in data and "predicted" in data:
                plot_error_distribution(
                    data["actual"], data["predicted"], ticker.upper()
                )

    except httpx.HTTPStatusError as e:
        progress.empty()
        st.error(f"API Error {e.response.status_code}: {e.response.text}")
    except httpx.TimeoutException:
        progress.empty()
        st.error(
            "Request timed out. The model may still be training. "
            "Try again in a few minutes."
        )
    except Exception as e:
        progress.empty()
        st.error(f"Unexpected error: {e}")
