"""
PORTFOLIO ENGINE
================
Combines multiple assets into a portfolio and calculates combined metrics.

Key concept:
  A portfolio is NOT just the average of its parts.
  Because assets move differently (correlation), combining them
  can REDUCE overall risk — this is called diversification.

  Famous example:
    Asset A: 20% vol,  Asset B: 20% vol
    If perfectly correlated (move together): Portfolio vol = 20%
    If uncorrelated (move independently):   Portfolio vol ≈ 14%
    If negatively correlated (move opposite): Portfolio vol < 14%

  This is the core idea behind Modern Portfolio Theory (Markowitz, 1952).
"""

import numpy as np
import pandas as pd
from src.analytics.returns import daily_returns, cagr, cumulative_returns, growth_of_investment
from src.analytics.risk   import volatility, sharpe_ratio, sortino_ratio, max_drawdown


def validate_weights(tickers: list[str], weights: list[float]) -> np.ndarray:
    """
    Validate and normalise portfolio weights.

    Rules:
      - One weight per ticker
      - Weights must sum to 1 (or we normalise them)
      - No negative weights (no short-selling in V1)
    """
    if len(tickers) != len(weights):
        raise ValueError(f"Got {len(tickers)} tickers but {len(weights)} weights.")

    weights = np.array(weights, dtype=float)

    if np.any(weights < 0):
        raise ValueError("Negative weights not supported in V1 (no short-selling).")

    total = weights.sum()
    if not np.isclose(total, 1.0, atol=0.01):
        print(f"⚠️  Weights sum to {total:.2%}. Auto-normalising to 100%.")
        weights = weights / total

    return weights


def portfolio_returns(prices: pd.DataFrame, weights: list[float]) -> pd.Series:
    """
    Calculate daily portfolio returns given asset weights.

    How it works:
      Each day, portfolio return = sum of (weight × asset return)
      Example with 60/40 portfolio:
        AAPL returned +2%, MSFT returned -1%
        Portfolio return = 0.6 × 2% + 0.4 × (-1%) = 0.8%
    """
    w = validate_weights(list(prices.columns), weights)
    daily_rets = daily_returns(prices)

    # Matrix multiplication: (days × assets) dot (assets,) = (days,)
    return daily_rets.dot(w)


def portfolio_cumulative_returns(prices: pd.DataFrame, weights: list[float]) -> pd.Series:
    """Cumulative return of the portfolio over time — for plotting equity curves."""
    port_rets = portfolio_returns(prices, weights)
    return (1 + port_rets).cumprod()


def portfolio_growth(prices: pd.DataFrame, weights: list[float], initial: float = 100_000) -> pd.Series:
    """Value of initial investment in the portfolio over time."""
    return portfolio_cumulative_returns(prices, weights) * initial


def portfolio_metrics(
    prices       : pd.DataFrame,
    weights      : list[float],
    risk_free_rate: float = 0.06
) -> dict:
    """
    Full set of performance metrics for a portfolio.

    Returns a dictionary with all key metrics.
    """
    w          = validate_weights(list(prices.columns), weights)
    port_rets  = portfolio_returns(prices, weights)

    # We need a "prices" series for the portfolio to reuse our risk functions
    # Trick: treat cumulative portfolio value as a "price" series
    port_value = (1 + port_rets).cumprod()
    port_df    = port_value.to_frame(name="Portfolio")

    n_years    = len(prices) / 252
    total_ret  = port_value.iloc[-1] - 1
    port_cagr  = (port_value.iloc[-1]) ** (1 / n_years) - 1
    port_vol   = port_rets.std() * np.sqrt(252)
    port_sharpe = (port_cagr - risk_free_rate) / port_vol

    downside = port_rets.copy()
    downside[downside > 0] = 0
    port_sortino = (port_cagr - risk_free_rate) / (downside.std() * np.sqrt(252))

    rolling_max = port_value.cummax()
    port_mdd    = ((port_value - rolling_max) / rolling_max).min()

    return {
        "Total Return"  : round(total_ret,    4),
        "CAGR"          : round(port_cagr,    4),
        "Volatility"    : round(port_vol,     4),
        "Sharpe Ratio"  : round(port_sharpe,  4),
        "Sortino Ratio" : round(port_sortino, 4),
        "Max Drawdown"  : round(port_mdd,     4),
    }


def correlation_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Correlation matrix between all assets.

    Correlation ranges from -1 to +1:
      +1.0 : Assets move perfectly together (no diversification benefit)
       0.0 : Assets move independently (good diversification)
      -1.0 : Assets move perfectly opposite (best diversification)

    Rule of thumb:
      > 0.7 : Highly correlated — little benefit to holding both
      < 0.3 : Low correlation  — good diversifier
    """
    daily_rets = daily_returns(prices)
    return daily_rets.corr().round(3)


def covariance_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Annualised covariance matrix.
    Used internally for portfolio variance calculations.
    (Correlation = covariance normalised by individual volatilities)
    """
    daily_rets = daily_returns(prices)
    return daily_rets.cov() * 252


def portfolio_variance_analytical(prices: pd.DataFrame, weights: list[float]) -> float:
    """
    Calculate portfolio variance using the covariance matrix formula.

    Formula: w^T × Σ × w
      where w = weight vector, Σ = covariance matrix

    This is the "textbook" way — faster than computing from historical returns.
    """
    w     = validate_weights(list(prices.columns), weights)
    cov   = covariance_matrix(prices)
    var   = w @ cov.values @ w  # matrix multiplication
    return float(var)
