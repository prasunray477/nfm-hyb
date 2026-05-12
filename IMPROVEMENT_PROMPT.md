Here is the precise improvement prompt you can feed directly to an AI code generator as pre-context:

---

# IMPROVEMENT PROMPT: Neutrosophic Bi-LSTM + ARIMA Stock Forecasting System — v2 Enhancements

## Context

This prompt is an **additive improvement specification** to be applied on top of the existing master blueprint. The core architecture (7-phase pipeline, FastAPI backend, Streamlit frontend, Render + Streamlit Cloud deployment) remains unchanged. Every instruction below is either a **new feature to add**, a **UI improvement to apply**, or a **terminology replacement to enforce globally**. Do not re-implement what already exists — only build what is specified here.

---

## 1. Terminology Replacements — Apply Globally Across All Files

The following academic jargon must be replaced everywhere it appears: in UI labels, page titles, chart titles, API response keys, variable names where readable, and all user-facing text. Backend internal variable names may keep short forms.

| Replace This | With This |
|---|---|
| Ablation Study | Comparative Analysis |
| Ablation | Model Comparison |
| C1, C2, C3, C4, C5, C6 (as labels) | Baseline ARIMA / Standard Deep Learning / NL-Enhanced Deep Learning / Standard Hybrid / NL Hybrid / Full Optimized Model |
| Directional Accuracy | Trend Prediction Accuracy |
| Theils_U | Forecast Reliability Score |
| val_loss / train_loss (UI text) | Validation Error / Training Error |
| Neutrosophication | Market Signal Encoding |
| Fluct / Fluctuation (UI text) | Daily Price Movement |
| Ticker | Stock Symbol |
| MAPE | Average Error Rate |
| BiLSTM Weight α | Deep Learning Model Weight |
| ARIMA Weight β | Statistical Model Weight |
| Pipeline | Analysis Engine |
| Test Set | Evaluation Period |
| Entropy Window | Uncertainty Lookback Window |

---

## 2. New Feature: Raw Data Visualization Dashboard

### 2.1 Location
Add as the **first tab** inside `frontend/pages/01_predict.py` after a successful forecast run. This appears before any model output. Tab label: **"Market Data"**.

### 2.2 Sub-section A — Price History Chart

Render a full interactive Plotly chart showing:
- Closing price `V_t` as a continuous line (color: `#0D9488`)
- A shaded vertical band marking the boundary between training period (left, light blue fill opacity 0.08) and the evaluation period (right, light amber fill opacity 0.08)
- Annotations on the chart: "Training Period" label at the left band center, "Evaluation Period" label at the right band center
- Chart title: `"{TICKER} — Historical Closing Price"`
- X-axis: actual calendar dates (not integer indices)
- Y-axis: price in currency
- Hover tooltip: date + price formatted to 2 decimal places

```python
def plot_price_history(prices, dates, train_end_idx, ticker):
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
```

### 2.3 Sub-section B — Daily Price Movement Chart

Below the price history chart, render a second chart showing the daily fluctuation series `U_t = V_t - V_{t-1}`:
- Bar chart where positive bars are `#0D9488` (teal/green) and negative bars are `#F43F5E` (red)
- A horizontal zero line in white at opacity 0.4
- Chart title: `"{TICKER} — Daily Price Movement (Gain / Loss)"`
- Y-axis label: "Price Change"
- Add a rolling 20-day moving average of absolute movement as a separate line (color: `#F59E0B`, label: "20-Day Avg Volatility")

```python
def plot_daily_movements(fluctuations, dates, ticker):
    colors = ['#0D9488' if v >= 0 else '#F43F5E' for v in fluctuations]
    roll_vol = pd.Series(np.abs(fluctuations)).rolling(20).mean().values

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dates[1:], y=fluctuations,
        marker_color=colors,
        name='Daily Movement',
        hovertemplate='%{x|%d %b %Y}<br>Change: %{y:.2f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=dates[1:], y=roll_vol,
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
```

---

## 3. New Feature: Neutrosophic Mapping Visualization

### 3.1 Location
Add as the **second tab** in the predict page after a successful run. Tab label: **"Market Signal Encoding"**.

### 3.2 Triple Time Series Chart

Render a stacked area chart of the three neutrosophic membership values over the full dataset timeline:

```python
def plot_neutrosophic_mapping(triples, dates_trimmed, ticker):
    T_vals = triples[:, 0]
    I_vals = triples[:, 1]
    F_vals = triples[:, 2]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates_trimmed, y=T_vals,
        name='Truth (T) — Bullish Strength',
        fill='tozeroy',
        mode='lines',
        line=dict(color='#0D9488', width=1.2),
        fillcolor='rgba(13,148,136,0.18)',
        hovertemplate='%{x|%d %b %Y}<br>T: %{y:.3f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=dates_trimmed, y=F_vals,
        name='Falsity (F) — Bearish Pressure',
        fill='tozeroy',
        mode='lines',
        line=dict(color='#F43F5E', width=1.2),
        fillcolor='rgba(244,63,94,0.18)',
        hovertemplate='%{x|%d %b %Y}<br>F: %{y:.3f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=dates_trimmed, y=I_vals,
        name='Indeterminacy (I) — Market Uncertainty',
        mode='lines',
        line=dict(color='#F59E0B', width=1.5, dash='dot'),
        hovertemplate='%{x|%d %b %Y}<br>I: %{y:.3f}<extra></extra>'
    ))
    fig.add_hline(
        y=0.333, line_dash='dot', line_color='white',
        opacity=0.35, annotation_text='Neutral (1/3)',
        annotation_font_size=9, annotation_font_color='#94A3B8'
    )
    fig.update_layout(
        title=f'{ticker} — Neutrosophic Market Signal Encoding (T, I, F)',
        xaxis_title='Date',
        yaxis_title='Membership Value [0 – 1]',
        yaxis=dict(range=[0, 1]),
        template='plotly_dark',
        height=360,
        legend=dict(orientation='h', y=1.08),
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
```

### 3.3 Interpretation Row Below Chart

Below the chart, render three `st.metric` cards in columns:

```python
col1, col2, col3 = st.columns(3)
col1.metric("Avg Bullish Signal (T)", f"{T_vals.mean():.3f}", help="Mean truth membership across all days. Above 0.5 = structurally bullish dataset.")
col2.metric("Avg Uncertainty (I)", f"{I_vals.mean():.3f}", help="Mean indeterminacy. Closer to 1.0 = highly inconsistent market history.")
col3.metric("Avg Bearish Signal (F)", f"{F_vals.mean():.3f}", help="Mean falsity membership. Above 0.5 = structurally bearish dataset.")
```

### 3.4 Explanation Expander

Add an `st.expander("How Market Signal Encoding Works")` below with clean prose (no bullet points) explaining what T, I, F represent in plain English for a non-technical investor audience.

---

## 4. New Feature: 30-Day Future Price Projection

### 4.1 Backend Changes

#### 4.1.1 Add method to `ml/pipeline.py`

Add a new method `forecast_future(n_days=30)` to `ForecastPipeline` that runs **after** `run()` completes:

```python
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
      future_dates: list of ISO date strings (business days)
      bilstm_future: list of float (Bi-LSTM n_days forecasts)
      arima_future: list of float (ARIMA n_days forecasts)
      combined_future: list of float (weighted combination)
      upper_band: list of float (combined + confidence margin)
      lower_band: list of float (combined - confidence margin)
      last_known_price: float
      trend_label: "Bullish" | "Bearish" | "Neutral"
      trend_strength: float (0–100)
      recommendation: "BUY" | "HOLD" | "SELL"
      recommendation_reason: str
    """
    import pandas as pd
    from datetime import timedelta

    # Generate future business dates
    last_date = pd.Timestamp.today()
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
    arima_model = ARIMA(full_series.astype(np.float64), order=self.arima.order).fit()
    arima_raw = arima_model.forecast(steps=n_days).values

    # Bi-LSTM rolling forecast using last window
    last_window = self._triples_full[-config.SEQUENCE_WINDOW:]
    bilstm_raw = []
    current_window = last_window.copy()
    predictor = BiLSTMPredictor(self.bilstm_trainer.model)

    for step in range(n_days):
        x = current_window.reshape(1, config.SEQUENCE_WINDOW, 3)
        pred_fluct = float(self.bilstm_trainer.model.predict(x, verbose=0)[0][0])
        bilstm_raw.append(pred_fluct)
        # Synthesize next triple from predicted fluctuation using stored normalizer
        next_T = self.normalizer._truth(pred_fluct)
        next_F = self.normalizer._falsity(pred_fluct)
        next_I = float(np.mean([current_window[-1, 1]]))  # propagate last entropy
        next_triple = np.array([[next_T, next_I, next_F]], dtype=np.float32)
        current_window = np.vstack([current_window[1:], next_triple])

    bilstm_raw = np.array(bilstm_raw, dtype=np.float32)

    # Reconstruct prices from fluctuations
    last_price = float(self._prices_full[-1])
    bilstm_prices, arima_prices, combined_prices = [], [], []
    current_price = last_price

    for i in range(n_days):
        bl = current_price + float(bilstm_raw[i])
        ar = current_price + float(arima_raw[i])
        cb = self.static_optimizer.alpha_star * bl + self.static_optimizer.beta_star * ar
        bilstm_prices.append(bl)
        arima_prices.append(ar)
        combined_prices.append(cb)
        current_price = cb

    combined_prices = np.array(combined_prices)

    # Confidence bands from test set error
    test_errors = np.abs(np.array(self._actual_test) - np.array(self._predicted_test))
    margin = float(np.std(test_errors) * 1.5)
    upper_band = (combined_prices + margin).tolist()
    lower_band = (combined_prices - margin).tolist()

    # Trend detection
    price_change_pct = (combined_prices[-1] - last_price) / last_price * 100
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

    if trend_label == "Bullish" and da > 60 and price_change_pct > 5:
        rec = "BUY"
        reason = (f"The model projects a {price_change_pct:.1f}% price increase over 30 days "
                  f"with {da:.1f}% trend prediction accuracy. Strong bullish signal with high model confidence.")
    elif trend_label == "Bearish" and da > 60 and price_change_pct < -5:
        rec = "SELL"
        reason = (f"The model projects a {abs(price_change_pct):.1f}% price decline over 30 days "
                  f"with {da:.1f}% trend prediction accuracy. Strong bearish signal — consider reducing exposure.")
    else:
        rec = "HOLD"
        reason = (f"The model projects a {price_change_pct:.1f}% price change over 30 days. "
                  f"The signal is not strong enough ({da:.1f}% accuracy) to recommend directional action. "
                  f"Monitor for clearer trend confirmation.")

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
```

#### 4.1.2 Store intermediate results in `run()`

At the end of `run()`, store these on `self` so `forecast_future()` can access them:

```python
# Store for future forecasting access
self._fluct_train = fluct_train
self._fluct_val   = fluct_val
self._fluct_test  = fluct_test
self._prices_full = prices
self._triples_full = np.vstack([triples_train, triples_val, triples_test])
self._actual_test    = actual_test.tolist()
self._predicted_test = final_test.tolist()
self._metrics = metrics
```

#### 4.1.3 Update `PredictResponse` schema in `app/models/schemas.py`

Add these new fields to `PredictResponse`:

```python
future_dates: List[str]
bilstm_future: List[float]
arima_future: List[float]
combined_future: List[float]
upper_band: List[float]
lower_band: List[float]
last_known_price: float
trend_label: str           # "Bullish" | "Bearish" | "Neutral"
trend_strength: float      # 0–100
recommendation: str        # "BUY" | "HOLD" | "SELL"
recommendation_reason: str
projected_change_pct: float
confidence_margin: float
```

#### 4.1.4 Update `predict.py` route to call `forecast_future()`

```python
results = pipeline.run()
future = pipeline.forecast_future(n_days=30)
results.update(future)
return PredictResponse(**results)
```

### 4.2 Frontend: 30-Day Forecast Tab

Add as the **third tab** in the predict page. Tab label: **"30-Day Outlook"**.

#### 4.2.1 Recommendation Banner

At the top of this tab, render a full-width color-coded banner based on `recommendation`:

```python
rec = data["recommendation"]
trend = data["trend_label"]
change = data["projected_change_pct"]
strength = data["trend_strength"]
reason = data["recommendation_reason"]

colors = {"BUY": "#0D9488", "HOLD": "#F59E0B", "SELL": "#F43F5E"}
icons  = {"BUY": "📈", "HOLD": "⏸", "SELL": "📉"}
bg     = {"BUY": "#0A2018", "HOLD": "#1C1500", "SELL": "#1C0A0A"}

st.markdown(f"""
<div style="background:{bg[rec]};border:1.5px solid {colors[rec]};
border-radius:8px;padding:18px 24px;margin-bottom:16px;">
  <div style="display:flex;align-items:center;gap:16px;">
    <span style="font-size:2.2rem">{icons[rec]}</span>
    <div>
      <div style="color:{colors[rec]};font-size:1.5rem;font-weight:700;
      letter-spacing:2px">{rec} — {trend} Signal</div>
      <div style="color:#94A3B8;font-size:0.92rem;margin-top:4px">{reason}</div>
    </div>
    <div style="margin-left:auto;text-align:right">
      <div style="color:{colors[rec]};font-size:1.8rem;font-weight:700">
        {'↑' if change > 0 else '↓'}{abs(change):.1f}%
      </div>
      <div style="color:#64748B;font-size:0.82rem">Projected 30-day change</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
```

#### 4.2.2 Trend Strength Gauge

Below the banner, show a horizontal progress bar styled as a trend meter:

```python
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown(f"**Trend Strength:** {data['trend_strength']:.0f} / 100")
    st.progress(int(data['trend_strength']))
    st.caption(f"Confidence Margin: ±{data['confidence_margin']:.2f} per forecast point | Model Trend Prediction Accuracy: {data['metrics']['Directional_Accuracy']:.1f}%")
```

#### 4.2.3 30-Day Projection Chart

```python
def plot_future_projection(data, ticker):
    last_price = data["last_known_price"]
    dates = data["future_dates"]
    combined = data["combined_future"]
    upper = data["upper_band"]
    lower = data["lower_band"]
    bilstm = data["bilstm_future"]
    arima  = data["arima_future"]
    trend  = data["trend_label"]

    trend_color = {"Bullish": "#0D9488", "Bearish": "#F43F5E", "Neutral": "#F59E0B"}[trend]

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
```

#### 4.2.4 Disclaimer

Below the chart, add:

```python
st.caption("⚠️  This projection is generated by a machine learning model trained on historical price patterns. It is not financial advice. Past performance does not guarantee future results. Always conduct independent research before making investment decisions.")
```

---

## 5. New Feature: Model Accuracy & Performance Dashboard

### 5.1 Location
Add as the **fourth tab** in the predict page. Tab label: **"Model Performance"**.

### 5.2 Forecast vs Actual Chart

Replace the current single forecast chart with this enhanced version:

```python
def plot_forecast_vs_actual(data, ticker):
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
```

### 5.3 Metrics Display

Replace the current `show_metrics()` component entirely with this version using renamed labels:

```python
def show_metrics_v2(metrics: dict) -> None:
    st.markdown("#### Performance Metrics — Evaluation Period")
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        label="Forecast Error (RMSE)",
        value=f"{metrics['RMSE']:.4f}",
        help="Root Mean Squared Error. Lower is better. Penalizes large prediction misses more heavily."
    )
    col2.metric(
        label="Average Error (MAE)",
        value=f"{metrics['MAE']:.4f}",
        help="Mean Absolute Error. Average price difference between forecast and actual in currency units."
    )
    col3.metric(
        label="Average Error Rate",
        value=f"{metrics['MAPE']:.2f}%",
        help="Mean Absolute Percentage Error. Scale-independent measure. Below 2% is considered strong."
    )
    col4.metric(
        label="Forecast Reliability",
        value=f"{metrics['Theils_U']:.4f}",
        help="Forecast Reliability Score (Theil's U). 0 = perfect. Below 0.05 = highly reliable."
    )
    col5.metric(
        label="Trend Prediction Accuracy",
        value=f"{metrics['Directional_Accuracy']:.1f}%",
        help="Percentage of days the model correctly predicted the direction (up or down). Above 55% outperforms random."
    )
```

### 5.4 Model Weight Display

Below metrics, show a two-column weight card:

```python
st.markdown("#### Ensemble Model Weights (Optimized on Validation Data)")
cA, cB = st.columns(2)
cA.metric("Deep Learning Model Weight", f"{data['alpha']:.0%}",
          help="Proportion of the final forecast contributed by the Bi-directional LSTM model.")
cB.metric("Statistical Model Weight", f"{data['beta']:.0%}",
          help="Proportion of the final forecast contributed by the ARIMA statistical model.")
```

### 5.5 Residual Error Distribution Chart

Add a histogram of per-day forecast errors in the evaluation period:

```python
def plot_error_distribution(actual, predicted, ticker):
    errors = np.array(actual) - np.array(predicted)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=errors, nbinsx=30,
        marker_color='#0D9488', opacity=0.75,
        name='Forecast Error'
    ))
    fig.add_vline(x=0, line_dash='dash', line_color='white', opacity=0.5)
    fig.add_vline(x=np.mean(errors), line_dash='dot', line_color='#F59E0B',
                  annotation_text=f'Mean: {np.mean(errors):.2f}',
                  annotation_font_color='#F59E0B')

    fig.update_layout(
        title=f'{ticker} — Forecast Error Distribution (Evaluation Period)',
        xaxis_title='Error (Actual − Forecast)',
        yaxis_title='Frequency',
        template='plotly_dark',
        height=280,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("A distribution centered near zero with low spread indicates a well-calibrated model.")
```

---

## 6. New Feature: Comparative Analysis Page (Replaces Ablation Study Page)

Rename `frontend/pages/02_ablation.py` to `frontend/pages/02_analysis.py`. Update page title and all labels:

### 6.1 Page Header

```python
st.title("📊 Comparative Model Analysis")
st.markdown("""
Compare how each model component contributes to the final forecast accuracy.
This analysis runs the same stock data through six progressively more complete
model configurations to demonstrate the value each component adds.
""")
```

### 6.2 Configuration Table — User-Facing Labels

Replace all C1-C6 labels with descriptive names:

```python
configs_display = {
    "Baseline (ARIMA Only)":                  "Linear statistical model only. No deep learning. Establishes the minimum viable baseline.",
    "Standard Deep Learning":                 "Bi-LSTM with conventional Min-Max normalization. No market signal encoding.",
    "NL-Enhanced Deep Learning":              "Bi-LSTM with Neutrosophic market signal encoding. Demonstrates encoding contribution.",
    "Standard Hybrid Model":                  "ARIMA + Bi-LSTM with conventional normalization and equal weights.",
    "NL Hybrid Model":                        "ARIMA + Bi-LSTM with Neutrosophic encoding and equal weights.",
    "Full Optimized Model (Our Approach)":    "Complete system: NL encoding + Bi-LSTM + ARIMA + optimized weights.",
}
```

### 6.3 Results Visualization

After running, show a grouped bar chart titled **"Forecast Error Across Model Configurations (Lower = Better)"** and a separate line chart titled **"Trend Prediction Accuracy Across Configurations (Higher = Better)"**. Both use the descriptive names above as X-axis labels.

---

## 7. UI Improvements — Global

### 7.1 Sidebar Enhancement

Replace the plain Streamlit sidebar with a styled version. Add to `frontend/streamlit_app.py`:

```python
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
```

### 7.2 Home Page Improvement

Replace current home page content with a three-column feature card layout:

```python
st.title("📈 Neutrosophic AI Stock Forecasting")
st.markdown("An intelligent stock analysis system combining market uncertainty modeling with deep learning and statistical forecasting.")
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style='background:#0A2018;border:1px solid #0D9488;border-radius:8px;padding:18px'>
    <div style='font-size:1.5rem'>🔬</div>
    <div style='font-weight:700;color:#0D9488;margin:8px 0 4px'>Market Signal Encoding</div>
    <div style='color:#94A3B8;font-size:0.88rem'>Transforms raw price movements into rich three-dimensional signals capturing bullish momentum, market uncertainty, and bearish pressure.</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style='background:#0A0A1F;border:1px solid #6D28D9;border-radius:8px;padding:18px'>
    <div style='font-size:1.5rem'>🧠</div>
    <div style='font-weight:700;color:#7C3AED;margin:8px 0 4px'>Deep Learning Engine</div>
    <div style='color:#94A3B8;font-size:0.88rem'>Bi-directional LSTM reads 60-day market patterns forward and backward to learn which signals reliably predict future price movement.</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style='background:#1A1200;border:1px solid #F59E0B;border-radius:8px;padding:18px'>
    <div style='font-size:1.5rem'>📊</div>
    <div style='font-weight:700;color:#F59E0B;margin:8px 0 4px'>30-Day Outlook</div>
    <div style='color:#94A3B8;font-size:0.88rem'>Projects the next 30 trading days with confidence bands, trend classification, and a clear BUY / HOLD / SELL recommendation.</div>
    </div>
    """, unsafe_allow_html=True)
```

### 7.3 Page Tab Structure for `01_predict.py`

Wrap all output content in four tabs immediately after the run button completes:

```python
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Market Data",
    "🔬  Market Signal Encoding",
    "🔮  30-Day Outlook",
    "📈  Model Performance"
])
```

### 7.4 Loading State Improvement

Replace the plain spinner with a staged progress display:

```python
progress = st.progress(0, text="Connecting to market data source...")
time.sleep(0.3); progress.progress(10, text="Downloading historical price data...")
time.sleep(0.3); progress.progress(20, text="Running stationarity checks...")
time.sleep(0.3); progress.progress(30, text="Encoding market signals (T, I, F)...")
# After API call starts:
progress.progress(50, text="Training Deep Learning model...")
# Cannot track actual training, so after response:
progress.progress(85, text="Fitting Statistical model...")
time.sleep(0.2); progress.progress(95, text="Optimizing ensemble weights...")
time.sleep(0.2); progress.progress(100, text="Analysis complete.")
progress.empty()
```

---

## 8. Implementation Checklist for the Code Generator

When implementing these changes, follow this exact order:

```
Step 1   Apply all terminology replacements globally (Section 1)
Step 2   Add plot_price_history() and plot_daily_movements() to frontend/components/charts.py
Step 3   Add plot_neutrosophic_mapping() to frontend/components/neutrosophic_viz.py
Step 4   Add internal state storage to ml/pipeline.py run() method
Step 5   Implement forecast_future() method in ml/pipeline.py
Step 6   Update PredictResponse schema with new fields
Step 7   Update /api/predict route to call forecast_future() and merge results
Step 8   Implement plot_future_projection() and recommendation banner in frontend
Step 9   Replace show_metrics() with show_metrics_v2()
Step 10  Implement plot_forecast_vs_actual() and plot_error_distribution()
Step 11  Restructure 01_predict.py into four-tab layout
Step 12  Rename and rewrite 02_ablation.py → 02_analysis.py
Step 13  Apply sidebar and home page UI improvements
Step 14  Add loading progress bar
Step 15  Run full QA: verify all old terminology is replaced, all tabs render, no import errors
```

---

## 9. Files Modified Summary

```
Modified   ml/pipeline.py                     add state storage + forecast_future()
Modified   app/models/schemas.py              add 12 new response fields
Modified   app/api/routes/predict.py          call forecast_future(), merge results
Modified   frontend/streamlit_app.py          new home page layout + sidebar
Modified   frontend/pages/01_predict.py       four-tab layout + all new charts
Renamed    frontend/pages/02_ablation.py      → 02_analysis.py (full rewrite)
Modified   frontend/components/charts.py      add price history, daily movement, forecast v2, error dist charts
Modified   frontend/components/neutrosophic_viz.py  add neutrosophic mapping chart
Modified   frontend/components/metrics_display.py   replace with metrics_v2
No change  ml/neutrosophic/*, ml/arima/*, ml/bilstm/*, ml/aggregator/*, ml/evaluation/*
No change  Dockerfile, render.yaml, .github/workflows/deploy.yml, requirements.txt
```