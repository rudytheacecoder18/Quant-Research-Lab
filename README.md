# 📊 Quant Research Lab V1

A quantitative finance research platform built with Python and Streamlit for portfolio analysis, risk analytics, and strategy backtesting across US and Indian equity markets.

---

## Overview

Quant Research Lab is designed to help investors and aspiring quants analyze portfolios, evaluate risk-adjusted performance, study diversification effects, and compare investment strategies through an interactive dashboard.

The project combines financial data analysis, portfolio analytics, and quantitative research techniques in a modular Python codebase.

---

## Features

### Performance Analytics

* CAGR (Compound Annual Growth Rate)
* Annualized Volatility
* Sharpe Ratio
* Sortino Ratio
* Maximum Drawdown
* Equity Curve Visualization

### Portfolio Simulator

* Multi-asset portfolio construction
* Custom portfolio weights
* Portfolio vs Individual Asset comparison
* Risk-return analysis

### Correlation Analysis

* Correlation Matrix
* Diversification Insights
* Asset Relationship Visualization

### Strategy Lab

* Buy & Hold Benchmark
* SMA Crossover Strategy
* Strategy Performance Comparison
* Backtesting Framework

---

## Supported Markets

### US Equities

Examples:

* AAPL
* MSFT
* NVDA
* GOOGL
* SPY

### Indian Equities

Examples:

* RELIANCE.NS
* TCS.NS
* INFY.NS
* HDFCBANK.NS
* NIFTYBEES.NS

---

## Project Structure

```text
src/
├── analytics/
│   ├── returns.py
│   └── risk.py
├── data/
│   └── data_loader.py
├── portfolio/
│   └── portfolio_engine.py
├── strategies/
│   ├── buy_hold.py
│   └── sma_crossover.py
├── visualization/
│   └── charts.py
└── dashboard/
    └── app.py
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/rudytheacecoder18/Quant-Research-Lab.git
cd Quant-Research-Lab
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

### Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the dashboard:

```bash
streamlit run src/dashboard/app.py
```

---

## Tech Stack

* Python
* Streamlit
* Pandas
* NumPy
* Plotly
* SciPy
* yFinance

---

## Future Roadmap

### V2 – Portfolio Optimization

* Efficient Frontier
* Monte Carlo Portfolio Simulation
* Maximum Sharpe Portfolio
* Minimum Variance Portfolio

### V3 – Risk Analytics

* Value at Risk (VaR)
* Conditional VaR
* Rolling Volatility
* Risk Attribution

### V4 – Asset Pricing

* CAPM
* Alpha & Beta Analysis
* Market Exposure Analytics

### V5 – Quantitative Research

* Fama-French Factor Models
* Factor Attribution
* Multi-Factor Portfolio Research

---

## Learning Objectives

This project serves as a practical exploration of:

* Portfolio Management
* Quantitative Finance
* Risk Analytics
* Financial Data Science
* Algorithmic Strategy Evaluation

---

## Author

Rudraksh Mehta

GitHub: https://github.com/rudytheacecoder18
