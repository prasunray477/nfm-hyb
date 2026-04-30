import streamlit as st

st.set_page_config(
    page_title="Neutrosophic Stock Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 Neutrosophic Bi-LSTM + ARIMA Stock Forecaster")

st.markdown("""
### How This Works
This system uses a **three-stage hybrid pipeline** to forecast stock prices:

| Stage | Component | Purpose |
|-------|-----------|---------|
| 1 | Neutrosophic Normalization | Encodes each trading day as (Truth, Indeterminacy, Falsity) |
| 2 | Bi-directional LSTM | Learns non-linear patterns from enriched features |
| 3 | ARIMA | Captures residual linear autocorrelation |
| 4 | Weighted Aggregation | Optimally combines both model outputs |
""")

st.info("👈 Use the sidebar to navigate to **Predict** or **Ablation Study**.")
