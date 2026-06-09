"""
EMA CROSSOVER STRATEGY
=======================
Same logic as SMA Crossover, but uses Exponential Moving Averages.

SMA vs EMA:
  SMA (Simple Moving Average):
    - Equal weight to ALL past days in the window
    - A 50-day SMA on day 50 = average of days 1–50
    - Problem: a big price move from 50 days ago has the SAME weight
      as yesterday's price move

  EMA (Exponential Moving Average):
    - More weight on RECENT prices, less on older prices
    - Reacts FASTER to new price trends
    - Most traders prefer EMA because it's more responsive
    - Used in MACD (next strategy)

When to use EMA vs SMA:
  - EMA: better for fast-moving markets, catches trends earlier
  - SMA: smoother, fewer false signals, better for slower markets
"""

import numpy as np
import pandas as pd


def run(
    prices     : pd.DataFrame,
    fast_window: int   = 12,
    slow_window: int   = 26,
    initial    : float = 100_000,
) -> dict:
    """
    EMA Crossover strategy.

    Default windows (12/26) are the same used in MACD.

    Parameters
    ----------
    fast_window : Short EMA (default 12)
    slow_window : Long EMA (default 26)
    initial     : Starting capital
    """
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    price_series = prices.iloc[:, 0]

    # EMA: pandas ewm() = exponentially weighted moving average
    # span = window size (equivalent to period in SMA)
    # adjust=False: use the recursive EMA formula (standard in trading)
    fast_ema = price_series.ewm(span=fast_window, adjust=False).mean()
    slow_ema = price_series.ewm(span=slow_window, adjust=False).mean()

    # Signal: 1 when fast EMA is above slow EMA (uptrend), 0 otherwise
    signal = (fast_ema > slow_ema).astype(int).shift(1).fillna(0)

    daily_rets    = price_series.pct_change()
    strategy_rets = daily_rets * signal

    equity_curve  = (1 + strategy_rets).cumprod() * initial

    n_years    = len(prices) / 252
    total_ret  = equity_curve.iloc[-1] / initial - 1
    strat_cagr = (1 + total_ret) ** (1 / n_years) - 1
    strat_vol  = strategy_rets.std() * np.sqrt(252)
    sharpe     = (strat_cagr - 0.06) / strat_vol if strat_vol > 0 else 0

    rolling_max = equity_curve.cummax()
    mdd         = ((equity_curve - rolling_max) / rolling_max).min()

    trades = signal.diff().abs().sum()

    metrics = {
        "Strategy"    : f"EMA {fast_window}/{slow_window} Crossover",
        "Total Return": round(total_ret,  4),
        "CAGR"        : round(strat_cagr, 4),
        "Volatility"  : round(strat_vol,  4),
        "Sharpe Ratio": round(sharpe,     4),
        "Max Drawdown": round(float(mdd), 4),
        "Num Trades"  : int(trades),
    }

    signals_df = pd.DataFrame({
        "Price"   : price_series,
        "Fast EMA": fast_ema,
        "Slow EMA": slow_ema,
        "Signal"  : signal,
    })

    return {
        "equity_curve": equity_curve,
        "signals"     : signals_df,
        "metrics"     : metrics,
    }
