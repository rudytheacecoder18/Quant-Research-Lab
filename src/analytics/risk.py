"""
RISK MODULE
===========
Calculates risk metrics for each asset.

Key concepts explained:
  - Volatility:     How much does the price jump around day to day?
                    High vol = risky, Low vol = stable
                    Measured as annualised standard deviation of returns

  - Sharpe Ratio:   Return per unit of risk taken
                    Formula: (Portfolio Return - Risk-Free Rate) / Volatility
                    Higher is better. >1 is good, >2 is great.

  - Sortino Ratio:  Like Sharpe, but only penalises downside risk (drops)
                    Sharpe treats upside volatility as bad too — Sortino doesn't

  - Max Drawdown:   The worst peak-to-trough loss you would have experienced
                    e.g. -35% means at its worst, ₹1,00,000 became ₹65,000
                    This is what keeps investors up at night

  - Calmar Ratio:   CAGR / |Max Drawdown|
                    Measures return earned per unit of drawdown risk
"""

import numpy as np
import pandas as pd
from .returns import daily_returns, log_returns, cagr


TRADING_DAYS = 252  # Standard annualisation factor


def volatility(prices: pd.DataFrame, annualise: bool = True) -> pd.Series:
    """
    Annualised volatility (standard deviation of daily returns).

    Why annualise?
      Daily vol * sqrt(252) gives the yearly equivalent.
      This is the market standard and lets you compare across time frames.

    Example: 1.2% daily vol → ~19% annual vol
    """
    daily_rets = log_returns(prices)
    vol = daily_rets.std()

    if annualise:
        vol = vol * np.sqrt(TRADING_DAYS)

    return vol


def sharpe_ratio(prices: pd.DataFrame, risk_free_rate: float = 0.06) -> pd.Series:
    """
    Sharpe Ratio: Risk-adjusted return.

    Parameters
    ----------
    risk_free_rate : Annual risk-free rate (default 6% — approximate India T-bill / US ~5%)
                     This is the return you'd get by doing NOTHING (just holding cash/bonds)

    Interpretation:
      Sharpe < 0 : Worse than risk-free
      Sharpe 0–1 : Okay
      Sharpe 1–2 : Good
      Sharpe > 2 : Excellent
    """
    annual_cagr = cagr(prices)
    annual_vol  = volatility(prices)

    return (annual_cagr - risk_free_rate) / annual_vol


def sortino_ratio(prices: pd.DataFrame, risk_free_rate: float = 0.06) -> pd.Series:
    """
    Sortino Ratio: Like Sharpe, but only penalises downside moves.

    Why Sortino?
      Sharpe penalises ALL volatility including upside.
      But investors only care about downside risk (losses).
      Sortino is more realistic for skewed return distributions.
    """
    annual_cagr = cagr(prices)
    daily_rets  = daily_returns(prices)

    # Only keep negative returns (downside)
    downside = daily_rets.copy()
    downside[downside > 0] = 0

    # Downside deviation (annualised)
    downside_vol = downside.std() * np.sqrt(TRADING_DAYS)

    return (annual_cagr - risk_free_rate) / downside_vol


def max_drawdown(prices: pd.DataFrame) -> pd.Series:
    """
    Maximum Drawdown: The worst peak-to-trough decline.

    How it's calculated:
      1. Find the running maximum price (the "peak" so far)
      2. Drawdown on each day = (Today's price - Peak) / Peak
      3. Max Drawdown = the worst (most negative) value ever

    Returns a negative number, e.g. -0.35 means -35%
    """
    # Running maximum (the highest price seen up to that point)
    rolling_max = prices.cummax()

    # Drawdown on each day
    drawdowns = (prices - rolling_max) / rolling_max

    # Worst drawdown
    return drawdowns.min()


def drawdown_series(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Full drawdown time series — used for plotting drawdown charts.
    Shows how far below the peak you are on each day.
    """
    rolling_max = prices.cummax()
    return (prices - rolling_max) / rolling_max


def calmar_ratio(prices: pd.DataFrame) -> pd.Series:
    """
    Calmar Ratio = CAGR / |Max Drawdown|

    Measures how much return you get per unit of drawdown risk.
    Higher is better.
    """
    annual_cagr = cagr(prices)
    mdd         = max_drawdown(prices).abs()  # Make positive for division

    return annual_cagr / mdd


def rolling_volatility(prices: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """
    Rolling 30-day volatility — shows how risk changes over time.

    This is more insightful than a single volatility number because:
      - Markets go through calm periods and crisis periods
      - Rolling vol reveals when things got scary (e.g. COVID crash)
    """
    daily_rets = log_returns(prices)
    return daily_rets.rolling(window).std() * np.sqrt(TRADING_DAYS)


def full_risk_report(prices: pd.DataFrame, risk_free_rate: float = 0.06) -> pd.DataFrame:
    """
    Generate a complete risk metrics table for all assets.

    Returns a DataFrame with one row per ticker, columns = metrics.
    This is what gets displayed in the dashboard summary table.
    """
    from .returns import total_return

    metrics = pd.DataFrame({
        "Total Return"  : total_return(prices),
        "CAGR"          : cagr(prices),
        "Volatility"    : volatility(prices),
        "Sharpe Ratio"  : sharpe_ratio(prices, risk_free_rate),
        "Sortino Ratio" : sortino_ratio(prices, risk_free_rate),
        "Max Drawdown"  : max_drawdown(prices),
        "Calmar Ratio"  : calmar_ratio(prices),
    })

    return metrics.round(4)
