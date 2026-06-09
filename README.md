# 📊 Quant Research Lab — V2

A professional equity research terminal for **US and Indian markets**, built in Python + Streamlit.

---

## What's in V2

| Tab | Feature | Concepts |
|---|---|---|
| 📊 Performance | CAGR, Sharpe, Sortino, Drawdown, Rolling Vol | Returns, Risk-adjusted metrics |
| 🧮 Portfolio | Custom weights, blended metrics, correlation heatmap | Portfolio theory |
| 🎲 Monte Carlo | 5,000+ random portfolios → risk/return cloud | Markowitz, diversification |
| ⚡ Optimizer | Exact Max Sharpe & Min Variance weights via SciPy | Constrained optimisation |
| 🛡️ VaR | Historical, Parametric, CVaR at 95% & 99% | Risk management, tail risk |
| 📐 CAPM | Beta, Alpha, R², Security Market Line | Factor models, regression |
| 🔬 Strategy Lab | Buy & Hold, SMA, EMA, RSI, MACD + heatmap | Backtesting, technical analysis |

---

## Setup

```bash
# 1. Enter project folder
cd quant-research-lab

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
streamlit run src/dashboard/app.py
```

Open `http://localhost:8501`

---

## Project Structure

```
quant-research-lab/
├── src/
│   ├── data/
│   │   └── data_loader.py          # yfinance, US + Indian markets
│   ├── analytics/
│   │   ├── returns.py              # CAGR, cumulative returns
│   │   ├── risk.py                 # Volatility, Sharpe, Sortino, Drawdown
│   │   ├── var.py                  # Historical VaR, Parametric VaR, CVaR
│   │   └── capm.py                 # Beta, Alpha, Expected Return, SML
│   ├── portfolio/
│   │   ├── portfolio_engine.py     # Blended portfolio metrics
│   │   ├── monte_carlo.py          # Random portfolio simulation
│   │   └── optimizer.py           # SciPy Max Sharpe / Min Variance
│   ├── strategies/
│   │   ├── buy_hold.py
│   │   ├── sma_crossover.py
│   │   ├── ema_strategy.py
│   │   ├── rsi_strategy.py
│   │   └── macd_strategy.py
│   ├── visualization/
│   │   ├── charts.py               # V1 charts (equity, drawdown, heatmap)
│   │   └── frontier_charts.py      # V2 charts (MC scatter, SML, MACD, RSI)
│   └── dashboard/
│       └── app.py                  # Main Streamlit app (7 tabs)
└── requirements.txt
```

---

## Quick Start Tickers

| Preset | Tickers |
|---|---|
| US Tech Giants | AAPL, MSFT, NVDA, GOOGL, SPY |
| Indian Blue Chips | RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS, NIFTYBEES.NS |
| Global Diversified | AAPL, MSFT, RELIANCE.NS, TCS.NS, SPY |
| Defensive Mix | JNJ, KO, PG, SPY, GLD |

Benchmark for CAPM: `SPY` (US) or `NIFTYBEES.NS` (India)

---

## Roadmap

- **V3:** Fama-French 3-factor model, rolling beta, ML alpha signals, PDF export
