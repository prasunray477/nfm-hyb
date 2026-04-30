import streamlit as st
import httpx
import os
from components.charts import plot_forecast
from components.metrics_display import show_metrics

API_URL = st.secrets.get("API_URL", os.getenv("API_URL", "http://localhost:8000"))

st.title("🔮 Stock Price Forecast")
st.markdown("Run the full Neutrosophic Bi-LSTM + ARIMA pipeline on any stock ticker.")

with st.form("predict_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        ticker = st.text_input(
            "Stock Ticker",
            value="AAPL",
            help="Examples: AAPL, MSFT, GOOGL, RELIANCE.NS, TCS.NS"
        )
    with col2:
        period = st.selectbox(
            "Historical Period",
            options=["3y", "5y", "7y", "10y"],
            index=1
        )
    with col3:
        use_adaptive = st.checkbox(
            "Adaptive Weights",
            value=False,
            help="Dynamically adjusts α/β based on rolling accuracy"
        )
    submitted = st.form_submit_button("▶ Run Forecast", type="primary")

if submitted:
    with st.spinner(
        f"Running pipeline for {ticker}... "
        "This takes 3–8 minutes (training Bi-LSTM)."
    ):
        try:
            response = httpx.post(
                f"{API_URL}/api/predict",
                json={
                    "ticker": ticker.upper(),
                    "use_adaptive": use_adaptive,
                    "period": period
                },
                timeout=900.0
            )
            response.raise_for_status()
            data = response.json()

            st.success(f"✅ Forecast complete for **{ticker.upper()}**")

            show_metrics(data["metrics"])

            col_a, col_b, col_c = st.columns(3)
            col_a.metric(
                "BiLSTM Weight (α)", f"{data['alpha']:.2f}"
            )
            col_b.metric(
                "ARIMA Weight (β)", f"{data['beta']:.2f}"
            )
            col_c.metric(
                "Test Days", str(data["n_test_days"])
            )

            plot_forecast(
                actual=data["actual"],
                predicted=data["predicted"],
                bilstm=data["bilstm_only"],
                arima=data["arima_only"],
                ticker=ticker.upper()
            )

        except httpx.HTTPStatusError as e:
            st.error(f"API Error {e.response.status_code}: {e.response.text}")
        except httpx.TimeoutException:
            st.error(
                "Request timed out. The model may still be training. "
                "Try again in a few minutes."
            )
        except Exception as e:
            st.error(f"Unexpected error: {e}")
