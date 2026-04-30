import streamlit as st

st.title("ℹ️ About This Model")

st.markdown(r"""
## Theoretical Foundation

### Neutrosophic Logic (Smarandache, 1999)
Standard normalization collapses each price movement to a scalar.
Neutrosophic normalization preserves **three independent dimensions**:

```
X_t = (T(U_t), I(U_t), F(U_t))

T(U_t) — Truth:         Degree of bullish momentum   ∈ [0, 1]
I(U_t) — Indeterminacy: Shannon entropy of recent     ∈ [0, 1]
                         market inconsistency
F(U_t) — Falsity:       Degree of bearish pressure    ∈ [0, 1]
```

### Benchmark Length
```
len = mean(|U_t|) over training period

T(U_t):
  0                        if U_t ≤ -0.5·len
  U_t/(1.5·len) + 1/3     if -0.5·len < U_t < len
  1                        if U_t ≥ len
```

### Shannon Information Entropy
The indeterminacy membership I(U_t) captures **how chaotic** the recent
market has been, computed over the last m=10 days:

```
I(U_t) = -Σ p(L_n)·log₂(p(L_n)) / log₂(5)
```

A trending market has I ≈ 0. A whipsaw market has I ≈ 1.

---

## Model Architecture

### Bi-directional LSTM
```
Input (60, 3) → BiLSTM(128) → Dropout(0.3)
             → BiLSTM(64)  → Dropout(0.2)
             → Dense(32)   → Dense(1)
Loss: Huber(δ=1.0)   Optimizer: Adam(lr=0.001)
```

### ARIMA
```
Order selected by AIC minimization: ARIMA(p*, 0, q*)
Expanding window one-step-ahead forecasting
```

### Weighted Aggregation
```
P̂_final = α·P̂_BiLSTM + (1-α)·P̂_ARIMA

α* = argmin RMSE on validation set
     via grid search α ∈ {0.00, 0.01, ..., 1.00}
```

---

## Why the Hybrid Works

The ensemble reduces error variance because:
- **Bi-LSTM errors** arise from non-linear pattern failures
- **ARIMA errors** arise from linear assumption violations

These error sources are structurally uncorrelated, so:
```
Var(ensemble) = α²·Var(BiLSTM) + β²·Var(ARIMA) + 2αβ·Cov(ε₁,ε₂)
                                                   ≈ 0
```

---

## Literature Evidence

| Comparison | Improvement | Source |
|------------|-------------|--------|
| NL front-end vs standard | +11.71% accuracy | Saravanaraj et al. (2025) |
| Bi-LSTM vs LSTM | +16 pp accuracy | Saravanaraj et al. (2025) |
| Hybrid vs ARIMA alone | 8–12% RMSE reduction | Dhanalakshmi et al. (2025) |

---

## Stack
- **Data:** yfinance (Yahoo Finance)
- **ML:** TensorFlow 2.15, statsmodels, pmdarima
- **API:** FastAPI + Uvicorn
- **UI:** Streamlit + Plotly
- **Deploy:** Render.com + Streamlit Community Cloud
""")
