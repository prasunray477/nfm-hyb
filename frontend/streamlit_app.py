import streamlit as st

st.set_page_config(
    page_title="Neutrosophic Stock Forecaster",
    page_icon="\U0001f4c8",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styled sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style='text-align:center;padding:12px 0 20px 0'>
  <div style='font-size:2rem'>📈</div>
  <div style='font-size:1.1rem;font-weight:700;color:#0D9488'>
    Stock Forecast System
  </div>
  <div style='font-size:0.78rem;color:#64748B;margin-top:4px'>
    Powered by Neutrosophic AI
  </div>
</div>
""", unsafe_allow_html=True)
    st.divider()
    st.caption("Navigate using the pages below")

# ── Home page ──────────────────────────────────────────────────────────────────
st.title("\U0001f4c8 Neutrosophic AI Stock Forecasting")
st.markdown(
    "An intelligent stock analysis system combining market uncertainty modeling "
    "with deep learning and statistical forecasting."
)
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
<div style='background:#0A2018;border:1px solid #0D9488;border-radius:8px;padding:18px'>
<div style='font-size:1.5rem'>🔬</div>
<div style='font-weight:700;color:#0D9488;margin:8px 0 4px'>Market Signal Encoding</div>
<div style='color:#94A3B8;font-size:0.88rem'>Transforms raw price movements into rich
three-dimensional signals capturing bullish momentum, market uncertainty, and bearish
pressure.</div>
</div>
""", unsafe_allow_html=True)

with col2:
    st.markdown("""
<div style='background:#0A0A1F;border:1px solid #6D28D9;border-radius:8px;padding:18px'>
<div style='font-size:1.5rem'>🧠</div>
<div style='font-weight:700;color:#7C3AED;margin:8px 0 4px'>Deep Learning Engine</div>
<div style='color:#94A3B8;font-size:0.88rem'>Bi-directional LSTM reads 60-day market
patterns forward and backward to learn which signals reliably predict future price
movement.</div>
</div>
""", unsafe_allow_html=True)

with col3:
    st.markdown("""
<div style='background:#1A1200;border:1px solid #F59E0B;border-radius:8px;padding:18px'>
<div style='font-size:1.5rem'>📊</div>
<div style='font-weight:700;color:#F59E0B;margin:8px 0 4px'>30-Day Outlook</div>
<div style='color:#94A3B8;font-size:0.88rem'>Projects the next 30 trading days with
confidence bands, trend classification, and a clear BUY / HOLD / SELL
recommendation.</div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── How it works table ─────────────────────────────────────────────────────────
st.markdown("### How This Works")
st.markdown("""
| Stage | Component | Purpose |
|-------|-----------|---------|
| 1 | Market Signal Encoding | Encodes each trading day as (Truth, Indeterminacy, Falsity) |
| 2 | Deep Learning Engine (Bi-LSTM) | Learns non-linear patterns from enriched features |
| 3 | Statistical Model (ARIMA) | Captures residual linear autocorrelation |
| 4 | Ensemble Optimiser | Optimally combines both model outputs |
| 5 | 30-Day Projection | Rolls the trained models forward to forecast future prices |
""")

st.info(
    "\U0001f448 Use the sidebar to navigate to **Stock Price Forecast** or "
    "**Comparative Model Analysis**."
)
