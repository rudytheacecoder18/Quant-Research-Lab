# 📊 Quant Research Lab V1

A professional equity research terminal supporting **US and Indian markets**, built in Python with Streamlit.

---

## Features

| Module | What it does |
|---|---|
| **Performance Dashboard** | CAGR, Sharpe, Sortino, Max Drawdown, Equity Curves |
| **Portfolio Simulator** | Custom weights, blended metrics, portfolio vs individual |
| **Correlation Analysis** | Heatmap + diversification insights |
| **Strategy Lab** | SMA Crossover vs Buy & Hold backtesting |

---

## Supported Tickers

- **US:** `AAPL`, `MSFT`, `NVDA`, `GOOGL`, `SPY`, etc.
- **India:** `RELIANCE.NS`, `TCS.NS`, `HDFCBANK.NS`, `NIFTYBEES.NS`, etc.
- **Mixed:** Combine both in one analysis

---

## Setup

### 1. Clone / download the project

```bash
cd quant-research-lab
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the dashboard

```bash
streamlit run src/dashboard/app.py
```

Then open your browser to `http://localhost:8501`

---

## Project Structure

```
quant-research-lab/
├── src/
│   ├── data/
│   │   └── data_loader.py       # yfinance data fetching + cleaning
│   ├── analytics/
│   │   ├── returns.py           # CAGR, total return, cumulative returns
│   │   └── risk.py              # Volatility, Sharpe, Sortino, Max Drawdown
│   ├── portfolio/
│   │   └── portfolio_engine.py  # Multi-asset portfolio + correlation matrix
│   ├── strategies/
│   │   ├── buy_hold.py          # Buy & Hold baseline
│   │   └── sma_crossover.py     # SMA Crossover backtest
│   ├── visualization/
│   │   └── charts.py            # All Plotly charts (dark theme)
│   └── dashboard/
│       └── app.py               # Main Streamlit app
└── requirements.txt
```

---

## Concepts Covered

**Finance:** Returns, CAGR, Benchmarking, Portfolio Theory  
**Statistics:** Mean, Standard Deviation, Correlation  
**Quant:** Sharpe Ratio, Sortino Ratio, Max Drawdown, Backtesting  
**Software:** APIs, Data Pipelines, Modular Design  
**Data Science:** Interactive Visualisations

---

## Version Roadmap

- **V1 (current):** Core analytics, portfolio simulator, SMA backtesting
- **V2:** VaR, Monte Carlo simulation, portfolio optimisation
- **V3:** Factor investing (Fama-French), ML alpha models
