import plotly.graph_objects as go
import numpy as np
import pandas as pd
import streamlit as st
from typing import List


# ── Tab 1: Market Data charts ──────────────────────────────────────────────────

def plot_price_history(prices, dates, train_end_idx, ticker):
    """Historical closing price with training/evaluation period shading."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates, y=prices,
        mode='lines',
        name='Closing Price',
        line=dict(color='#0D9488', width=1.8),
        hovertemplate='%{x|%d %b %Y}<br>Price: %{y:.2f}<extra></extra>'
    ))

    # Training region shading
    fig.add_vrect(
        x0=dates[0], x1=dates[train_end_idx],
        fillcolor='#0EA5E9', opacity=0.07,
        layer='below', line_width=0,
        annotation_text='Training Period',
        annotation_position='top left',
        annotation_font_size=11,
        annotation_font_color='#0EA5E9'
    )

    # Evaluation region shading
    fig.add_vrect(
        x0=dates[train_end_idx], x1=dates[-1],
        fillcolor='#F59E0B', opacity=0.07,
        layer='below', line_width=0,
        annotation_text='Evaluation Period',
        annotation_position='top right',
        annotation_font_size=11,
        annotation_font_color='#F59E0B'
    )

    fig.update_layout(
        title=f'{ticker} — Historical Closing Price',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_dark',
        height=340,
        margin=dict(l=60, r=20, t=50, b=40),
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_daily_movements(fluctuations, dates, ticker):
    """Daily price movement bar chart with 20-day rolling volatility line."""
    colors = ['#0D9488' if v >= 0 else '#F43F5E' for v in fluctuations]
    roll_vol = pd.Series(np.abs(fluctuations)).rolling(20).mean().values
    x_dates = dates if len(dates) == len(fluctuations) else dates[1:]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x_dates, y=fluctuations,
        marker_color=colors,
        name='Daily Movement',
        hovertemplate='%{x|%d %b %Y}<br>Change: %{y:.2f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=x_dates, y=roll_vol,
        mode='lines',
        name='20-Day Avg Volatility',
        line=dict(color='#F59E0B', width=1.5),
    ))
    fig.add_hline(y=0, line_color='white', line_width=0.5, opacity=0.4)
    fig.update_layout(
        title=f'{ticker} — Daily Price Movement (Gain / Loss)',
        xaxis_title='Date',
        yaxis_title='Price Change',
        template='plotly_dark',
        height=280,
        margin=dict(l=60, r=20, t=50, b=40),
        hovermode='x unified',
        barmode='relative'
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Tab 4: Model Performance charts ───────────────────────────────────────────

def plot_forecast_vs_actual(data, ticker):
    """Enhanced forecast vs actual chart for the Evaluation Period tab."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=data['actual'], name='Actual Price',
        mode='lines',
        line=dict(color='#0D9488', width=2.2),
        hovertemplate='Day %{x}<br>Actual: %{y:.2f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        y=data['predicted'], name='Combined Forecast',
        mode='lines',
        line=dict(color='#F59E0B', width=2.2, dash='dash'),
        hovertemplate='Day %{x}<br>Forecast: %{y:.2f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        y=data['bilstm_only'], name='Deep Learning Only',
        mode='lines', opacity=0.55,
        line=dict(color='#6D28D9', width=1, dash='dot')
    ))
    fig.add_trace(go.Scatter(
        y=data['arima_only'], name='Statistical Only',
        mode='lines', opacity=0.55,
        line=dict(color='#0EA5E9', width=1, dash='dot')
    ))

    fig.update_layout(
        title=f'{ticker} — Forecast vs Actual Price (Evaluation Period)',
        xaxis_title='Trading Day',
        yaxis_title='Price',
        template='plotly_dark',
        height=380,
        legend=dict(orientation='h', y=1.06),
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_error_distribution(actual, predicted, ticker):
    """Histogram of forecast errors over the evaluation period."""
    errors = np.array(actual) - np.array(predicted)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=errors, nbinsx=30,
        marker_color='#0D9488', opacity=0.75,
        name='Forecast Error'
    ))
    fig.add_vline(x=0, line_dash='dash', line_color='white', opacity=0.5)
    fig.add_vline(
        x=np.mean(errors), line_dash='dot', line_color='#F59E0B',
        annotation_text=f'Mean: {np.mean(errors):.2f}',
        annotation_font_color='#F59E0B'
    )

    fig.update_layout(
        title=f'{ticker} — Forecast Error Distribution (Evaluation Period)',
        xaxis_title='Error (Actual − Forecast)',
        yaxis_title='Frequency',
        template='plotly_dark',
        height=280,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "A distribution centered near zero with low spread "
        "indicates a well-calibrated model."
    )


# ── Tab 3: 30-Day Outlook chart ────────────────────────────────────────────────

def plot_future_projection(data, ticker):
    """30-day forward price projection with confidence bands."""
    last_price = data["last_known_price"]
    dates      = data["future_dates"]
    combined   = data["combined_future"]
    upper      = data["upper_band"]
    lower      = data["lower_band"]
    bilstm     = data["bilstm_future"]
    arima      = data["arima_future"]
    trend      = data["trend_label"]

    trend_color = {
        "Bullish": "#0D9488",
        "Bearish": "#F43F5E",
        "Neutral": "#F59E0B"
    }[trend]

    fig = go.Figure()

    # Confidence band
    fig.add_trace(go.Scatter(
        x=dates + dates[::-1],
        y=upper + lower[::-1],
        fill='toself',
        fillcolor='rgba(13,148,136,0.10)',
        line=dict(color='rgba(0,0,0,0)'),
        name='Confidence Band',
        hoverinfo='skip'
    ))

    # Individual model lines
    fig.add_trace(go.Scatter(
        x=dates, y=bilstm,
        mode='lines', name='Deep Learning Model',
        line=dict(color='#6D28D9', width=1.2, dash='dot'),
        opacity=0.65
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=arima,
        mode='lines', name='Statistical Model',
        line=dict(color='#0EA5E9', width=1.2, dash='dot'),
        opacity=0.65
    ))

    # Main projection line
    fig.add_trace(go.Scatter(
        x=dates, y=combined,
        mode='lines+markers', name='Combined Forecast',
        line=dict(color=trend_color, width=2.5),
        marker=dict(size=4),
        hovertemplate='%{x}<br>Projected Price: %{y:.2f}<extra></extra>'
    ))

    # Last known price reference line
    fig.add_hline(
        y=last_price, line_dash='dash', line_color='white',
        opacity=0.4, annotation_text=f'Last Known: {last_price:.2f}',
        annotation_font_size=10
    )

    fig.update_layout(
        title=f'{ticker} — 30-Day Price Projection ({trend} Outlook)',
        xaxis_title='Date',
        yaxis_title='Projected Price',
        template='plotly_dark',
        height=420,
        legend=dict(orientation='h', y=1.06),
        hovermode='x unified',
        margin=dict(l=60, r=20, t=60, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Legacy alias kept for any internal callers ─────────────────────────────────

def plot_forecast(
    actual: List[float],
    predicted: List[float],
    bilstm: List[float],
    arima: List[float],
    ticker: str
) -> None:
    """Kept for backward compatibility — delegates to plot_forecast_vs_actual."""
    data = {
        'actual':     actual,
        'predicted':  predicted,
        'bilstm_only': bilstm,
        'arima_only':  arima,
    }
    plot_forecast_vs_actual(data, ticker)
