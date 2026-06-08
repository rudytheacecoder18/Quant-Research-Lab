"""
RETURNS MODULE
==============
Handles everything related to calculating investment returns.

Key concepts explained:
  - Daily Return:  How much % did the stock move each day?
  - Total Return:  If you invested ₹1,00,000, what is it worth now?
  - CAGR:          "Compound Annual Growth Rate" — annualised return
                   e.g. 15% CAGR means your money grew at 15% per year on average
  - Log Returns:   Used in statistics because they're more math-friendly
                   (we'll use these for volatility calculations)
"""

import numpy as np
import pandas as pd


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate simple daily percentage returns.

    Formula: (Price_today / Price_yesterday) - 1

    Example: Stock was ₹100 yesterday, ₹103 today → return = 3%
    """
    return prices.pct_change().dropna()


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate logarithmic daily returns.

    Formula: ln(Price_today / Price_yesterday)

    Why use log returns?
      - They're symmetric: +10% and -10% cancel out
      - They're additive over time (easier for statistics)
      - Assumed to be normally distributed (needed for Sharpe, VaR, etc.)
    """
    return np.log(prices / prices.shift(1)).dropna()


def total_return(prices: pd.DataFrame) -> pd.Series:
    """
    Total return over the full period for each asset.

    Formula: (End Price / Start Price) - 1

    Returns a Series: {ticker: total_return_as_decimal}
    Example: 0.45 means +45% total return
    """
    return (prices.iloc[-1] / prices.iloc[0]) - 1


def cagr(prices: pd.DataFrame) -> pd.Series:
    """
    Compound Annual Growth Rate — the annualised return.

    Formula: (End/Start) ^ (1/years) - 1

    Why CAGR?
      - A 50% total return over 10 years is very different from 50% over 2 years
      - CAGR normalises this so you can compare any two assets fairly
    """
    n_days  = len(prices)
    n_years = n_days / 252  # 252 = average trading days per year

    start = prices.iloc[0]
    end   = prices.iloc[-1]

    return (end / start) ** (1 / n_years) - 1


def cumulative_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Cumulative return series — used for plotting equity curves.

    This shows: "If I invested ₹1 on day 1, what is it worth on each day?"

    Example output for one ticker:
      2021-01-01: 1.00   (starting value)
      2021-06-01: 1.23   (up 23%)
      2022-01-01: 1.45   (up 45% total)
    """
    rets = daily_returns(prices)
    return (1 + rets).cumprod()


def growth_of_investment(prices: pd.DataFrame, initial: float = 100_000) -> pd.DataFrame:
    """
    Show the actual rupee/dollar value of an initial investment over time.

    Parameters
    ----------
    initial : Starting investment amount (default ₹1,00,000)

    Returns
    -------
    DataFrame showing value of investment on each day
    """
    cum = cumulative_returns(prices)
    return cum * initial
