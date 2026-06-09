"""
MONTE CARLO PORTFOLIO SIMULATION
==================================
One of the most iconic visuals in all of quant finance.

The idea is brilliantly simple:
  1. Generate thousands of random portfolios (random weight combinations)
  2. For each portfolio, calculate its Expected Return and Volatility
  3. Plot every portfolio as a dot on a Risk vs Return chart
  4. The resulting "cloud" of dots reveals the Efficient Frontier

What you'll see:
  - A curved boundary on the upper-left edge of the cloud
  - That curve is the Efficient Frontier — the set of portfolios
    that give the MAXIMUM return for a given level of risk
  - Anything inside the cloud is "inefficient" — you could do better

Key portfolios highlighted:
  ★ Max Sharpe  — best risk-adjusted return (the "sweet spot")
  ◆ Min Variance — lowest possible risk (for the risk-averse)

This is the foundation of Modern Portfolio Theory (Markowitz, 1952).
Harry Markowitz won the Nobel Prize for this idea.
"""

import numpy as np
import pandas as pd
from src.analytics.returns import daily_returns, cagr
from src.analytics.risk    import volatility


def run_simulation(
    prices          : pd.DataFrame,
    n_portfolios    : int   = 5000,
    risk_free_rate  : float = 0.06,
    random_seed     : int   = 42
) -> pd.DataFrame:
    """
    Simulate thousands of random portfolios and return their stats.

    Parameters
    ----------
    prices         : Price DataFrame (dates × tickers)
    n_portfolios   : How many random portfolios to generate (5000 is standard)
    risk_free_rate : Used to calculate Sharpe Ratio for each portfolio
    random_seed    : For reproducibility — same seed = same random portfolios

    Returns
    -------
    pd.DataFrame with columns:
      - Return     : Annualised expected return
      - Volatility : Annualised volatility (risk)
      - Sharpe     : Sharpe Ratio
      - w_TICKER   : Weight assigned to each asset
    """
    np.random.seed(random_seed)

    n_assets = len(prices.columns)
    tickers  = list(prices.columns)

    # Pre-compute daily returns and annualised stats (faster than recomputing each loop)
    daily_rets   = daily_returns(prices)
    mean_returns = daily_rets.mean() * 252          # Annualised expected return per asset
    cov_matrix   = daily_rets.cov() * 252           # Annualised covariance matrix

    # Storage — pre-allocate for speed
    results = np.zeros((n_portfolios, 3 + n_assets))
    # columns: [Return, Volatility, Sharpe, w1, w2, ..., wN]

    for i in range(n_portfolios):
        # Step 1: Generate random weights
        raw_weights = np.random.random(n_assets)       # Random numbers 0–1
        weights     = raw_weights / raw_weights.sum()  # Normalise so they sum to 1

        # Step 2: Portfolio Return = weighted sum of individual returns
        port_return = np.dot(weights, mean_returns)

        # Step 3: Portfolio Variance using the covariance matrix
        # Formula: w^T × Σ × w  (this is the matrix form of portfolio variance)
        # It automatically accounts for correlations between assets
        port_variance = weights @ cov_matrix.values @ weights
        port_vol      = np.sqrt(port_variance)

        # Step 4: Sharpe Ratio
        sharpe = (port_return - risk_free_rate) / port_vol

        # Store results
        results[i, 0]  = port_return
        results[i, 1]  = port_vol
        results[i, 2]  = sharpe
        results[i, 3:] = weights

    # Package into a DataFrame
    weight_cols = [f"w_{t}" for t in tickers]
    columns     = ["Return", "Volatility", "Sharpe"] + weight_cols

    df = pd.DataFrame(results, columns=columns)

    return df


def get_max_sharpe(simulation_df: pd.DataFrame, tickers: list[str]) -> dict:
    """
    Find the portfolio with the highest Sharpe Ratio from the simulation.

    This is the "optimal" portfolio — maximum return per unit of risk.
    """
    idx = simulation_df["Sharpe"].idxmax()
    row = simulation_df.loc[idx]

    weights = {t: row[f"w_{t}"] for t in tickers}

    return {
        "Return"    : row["Return"],
        "Volatility": row["Volatility"],
        "Sharpe"    : row["Sharpe"],
        "Weights"   : weights,
    }


def get_min_variance(simulation_df: pd.DataFrame, tickers: list[str]) -> dict:
    """
    Find the portfolio with the lowest volatility from the simulation.

    This is for the risk-averse investor — minimum possible risk.
    """
    idx = simulation_df["Volatility"].idxmin()
    row = simulation_df.loc[idx]

    weights = {t: row[f"w_{t}"] for t in tickers}

    return {
        "Return"    : row["Return"],
        "Volatility": row["Volatility"],
        "Sharpe"    : row["Sharpe"],
        "Weights"   : weights,
    }
