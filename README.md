# Neutrosophic Bi-LSTM + ARIMA Stock Forecasting System

![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-FF4B4B?logo=streamlit)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-FF6F00?logo=tensorflow)

A complete, end-to-end stock price forecasting system that combines advanced Deep Learning (**Bi-directional LSTM**) with classical statistical time-series forecasting (**ARIMA**), enhanced by a novel data preprocessing technique utilizing **Neutrosophic Logic**.

This project provides both an interactive frontend dashboard for stock analysis and a robust backend API.

---

## Theoretical Foundation

### 1. Neutrosophic Logic Normalization
Standard Min-Max normalization collapses price fluctuations into a simple scalar, ignoring the "chaos" or market uncertainty. This system transforms raw market data into a 3-Dimensional Neutrosophic Set:
- **T (Truth):** Degree of bullish momentum.
- **I (Indeterminacy):** Shannon entropy of recent market inconsistency.
- **F (Falsity):** Degree of bearish pressure.

### 2. Bi-directional LSTM
Reads the 3D Neutrosophic sequences both forwards and backwards in time, learning non-linear patterns from semantically rich features.

### 3. ARIMA
Captures the linear autocorrelation and short-term momentum in the time series that Deep Learning models often struggle to model efficiently.

### 4. Weighted Ensemble
The predictions from both the Bi-LSTM and ARIMA models are optimally combined ($P_{final} = \alpha P_{BiLSTM} + \beta P_{ARIMA}$). Because the errors from the two models are structurally uncorrelated, the hybrid ensemble significantly reduces overall error variance.

---

## Key Features
- **Real-Time Data Fetching:** Directly downloads live OHLCV data using Yahoo Finance (`yfinance`).
- **Stationarity Enforcement:** Automatically runs Augmented Dickey-Fuller (ADF) tests and applies differencing.
- **Comparative Analysis Dashboard:** Run six model configurations, from baseline ARIMA to the Full Optimized Model, to quantify the contribution of each component.
- **Dynamic Weight Optimization:** Uses Grid Search to find the mathematically optimal blend of Bi-LSTM and ARIMA forecasts on the validation set.

---

## Project Structure

```text
neutrosophic-stock-forecast/
├── app/                  # FastAPI backend application
│   ├── api/routes/       # API Endpoints (predict, train, health)
│   ├── core/             # Configuration and Loguru logger
│   └── services/         # yfinance data collection
├── frontend/             # Streamlit Interactive UI
│   ├── components/       # Plotly charts and metrics
│   └── pages/            # Predict, Comparative Analysis, and About pages
├── ml/                   # Machine Learning analysis engine
│   ├── aggregator/       # Weight optimization logic
│   ├── arima/            # Auto-ARIMA statistical modeling
│   ├── bilstm/           # TensorFlow Keras architecture
│   ├── evaluation/       # RMSE, MAE, MAPE, Theil's U metrics
│   └── neutrosophic/     # Entropy and Neutrosophic transform
├── data/                 # Local storage for saved models
└── MASTER_BLUEPRINT.md   # Architectural reference document
```

---

## Setup and Installation

### Prerequisites
- Python 3.11+
- Git

### 1. Clone the repository
```bash
git clone https://github.com/your-username/neutrosophic-stock-forecast.git
cd neutrosophic-stock-forecast
```

### 2. Create a Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Setup Environment Variables
Copy the example environment file:
```bash
# Windows
copy .env.example .env

# Linux / macOS
cp .env.example .env
```

---

## Running the Application

This system consists of two parts that need to be run concurrently: the **FastAPI Backend** and the **Streamlit Frontend**.

### Terminal 1: Start the Backend API
The backend handles the heavy lifting, data fetching, model training, and forecasting.
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
The API will be available at: `http://localhost:8000`  
API Documentation (Swagger UI): `http://localhost:8000/docs`

### Terminal 2: Start the Frontend UI
The frontend provides the interactive user dashboard.
```bash
streamlit run frontend/streamlit_app.py
```
The application will open in your browser automatically at: `http://localhost:8501`

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Check API status |
| `POST` | `/api/predict` | Run the full 7-phase forecasting analysis engine for a given stock symbol |
| `POST` | `/api/train` | Run the analysis engine asynchronously in the background |

**Example Request:**
```bash
curl -X 'POST' \
  'http://localhost:8000/api/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "ticker": "AAPL",
  "use_adaptive": false,
  "period": "5y"
}'
```

---

