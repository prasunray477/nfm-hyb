import plotly.graph_objects as go
import streamlit as st
from typing import List


def plot_forecast(
    actual: List[float],
    predicted: List[float],
    bilstm: List[float],
    arima: List[float],
    ticker: str
) -> None:
    """Render interactive forecast comparison chart."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=actual, name="Actual Price",
        line=dict(color="#00CC96", width=2.5)
    ))
    fig.add_trace(go.Scatter(
        y=predicted, name="Hybrid Forecast (Final)",
        line=dict(color="#EF553B", width=2.5, dash="dash")
    ))
    fig.add_trace(go.Scatter(
        y=bilstm, name="Bi-LSTM Only",
        line=dict(color="#636EFA", width=1.2, dash="dot"),
        opacity=0.7
    ))
    fig.add_trace(go.Scatter(
        y=arima, name="ARIMA Only",
        line=dict(color="#FFA15A", width=1.2, dash="dot"),
        opacity=0.7
    ))

    fig.update_layout(
        title=dict(
            text=f"{ticker} — Actual vs Forecast (Test Set)",
            font=dict(size=18)
        ),
        xaxis_title="Trading Day (Test Period)",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1
        ),
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)
