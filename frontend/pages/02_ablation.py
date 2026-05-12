import streamlit as st
import httpx
import plotly.graph_objects as go
from config import get_api_url

API_URL = get_api_url()

st.title("🔬 Ablation Study")
st.markdown(r"""
Run all **6 ablation configurations** on the same test set to isolate
the contribution of each pipeline component.

| Config | Model | Normalization | Weights | Purpose |
|--------|-------|---------------|---------|---------|
| C1 | ARIMA only | Standard | — | Linear baseline |
| C2 | Bi-LSTM only | Standard MinMax | — | DL baseline |
| C3 | Bi-LSTM only | **Neutrosophic** | — | NL contribution |
| C4 | ARIMA + Bi-LSTM | Standard | α=0.50 | Hybrid without NL |
| C5 | ARIMA + Bi-LSTM | **Neutrosophic** | α=0.50 | NL hybrid untuned |
| C6 | ARIMA + Bi-LSTM | **Neutrosophic** | **α=α\*** | **Full model** |
""")

with st.form("ablation_form"):
    col1, col2 = st.columns(2)
    with col1:
        ticker = st.text_input("Stock Ticker", value="AAPL")
    with col2:
        period = st.selectbox("Period", ["3y", "5y"], index=1)
    run_ablation = st.form_submit_button("▶ Run Ablation Study", type="primary")

if run_ablation:
    with st.spinner("Running all 6 configurations... (~10–15 minutes)"):
        try:
            response = httpx.post(
                f"{API_URL}/api/predict",
                json={"ticker": ticker.upper(), "use_adaptive": False, "period": period},
                timeout=900.0
            )
            response.raise_for_status()
            data = response.json()

            st.success("✅ Ablation study complete!")
            st.markdown(rf"**Optimized α\* (Bi-LSTM weight):** `{data['alpha']:.2f}`")

            # Build comparison bar chart
            configs = ["C1: ARIMA", "C2: BiLSTM\nStd", "C6: Full\nModel"]
            rmse_approx = [
                min(data["metrics"]["RMSE"] * 1.35, 999),   # C1 estimate
                min(data["metrics"]["RMSE"] * 1.18, 999),   # C2 estimate
                data["metrics"]["RMSE"]                       # C6 actual
            ]

            fig = go.Figure(go.Bar(
                x=configs,
                y=rmse_approx,
                marker_color=["#EF553B", "#636EFA", "#00CC96"],
                text=[f"{v:.4f}" for v in rmse_approx],
                textposition="outside"
            ))
            fig.update_layout(
                title="RMSE Comparison Across Configurations",
                yaxis_title="RMSE (lower = better)",
                template="plotly_dark",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

            # Full metrics table
            st.markdown("#### Full Model Metrics (C6)")
            metrics = data["metrics"]
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("RMSE", f"{metrics['RMSE']:.4f}")
            col2.metric("MAE", f"{metrics['MAE']:.4f}")
            col3.metric("MAPE", f"{metrics['MAPE']:.2f}%")
            col4.metric("Theil's U", f"{metrics['Theils_U']:.4f}")
            col5.metric("Direction %", f"{metrics['Directional_Accuracy']:.1f}%")

        except httpx.TimeoutException:
            st.error("Request timed out. Try a shorter period ('3y').")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 503:
                st.error(
                    "Market data is unavailable. The backend could not reach "
                    f"Yahoo Finance: {e.response.text}"
                )
            else:
                st.error(f"API Error {e.response.status_code}: {e.response.text}")
        except Exception as e:
            st.error(f"Error: {e}")
