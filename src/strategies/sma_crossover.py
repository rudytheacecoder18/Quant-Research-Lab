"""
SMA CROSSOVER STRATEGY
=======================
One of the most classic quantitative trading strategies.

How it works:
  1. Calculate a FAST moving average (e.g. 50-day SMA)
  2. Calculate a SLOW moving average (e.g. 200-day SMA)
  3. BUY signal:  When Fast SMA crosses ABOVE Slow SMA ("Golden Cross")
  4. SELL signal: When Fast SMA crosses BELOW Slow SMA ("Death Cross")

Intuition:
  - When the short-term trend (fast) is above long-term trend (slow),
    the asset is in an uptrend → be invested
  - When short-term drops below long-term → trend is reversing → get out

This is a TREND FOLLOWING strategy — it catches big moves but has lots of
small false signals (called "whipsaws").
"""

import numpy as np
import pandas as pd
from src.analytics.returns import daily_returns, total_return, cagr
from src.analytics.risk    import volatility, sharpe_ratio, max_drawdown


def run(
    prices     : pd.DataFrame,
    fast_window: int   = 50,
    slow_window: int   = 200,
    initial    : float = 100_000
) -> dict:
    """
    Run SMA Crossover strategy on a single asset.

    Parameters
    ----------
    prices      : Price series (single-column DataFrame)
    fast_window : Short-term SMA period (default: 50 days)
    slow_window : Long-term SMA period (default: 200 days)
    initial     : Starting capital

    Returns
    -------
    dict with:
      - equity_curve : pd.Series — portfolio value over time
      - signals      : pd.DataFrame — SMA lines + buy/sell signal
      - metrics      : dict — performance summary
    """
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    price_series = prices.iloc[:, 0]

    # ── Calculate Moving Averages ──────────────────────────────────────────
    fast_sma = price_series.rolling(window=fast_window).mean()
    slow_sma = price_series.rolling(window=slow_window).mean()

    # ── Generate Signal ────────────────────────────────────────────────────
    # Signal = 1 when fast > slow (invested), 0 when fast < slow (in cash)
    signal = (fast_sma > slow_sma).astype(int)

    # Shift signal by 1 day — we can only act on yesterday's signal today
    # (avoids "look-ahead bias" — peeking at the future)
    signal = signal.shift(1).fillna(0)

    # ── Calculate Returns ──────────────────────────────────────────────────
    daily_rets    = price_series.pct_change()
    strategy_rets = daily_rets * signal  # Return = 0 when out of market

    # ── Build Equity Curve ─────────────────────────────────────────────────
    equity_curve = (1 + strategy_rets).cumprod() * initial

    # ── Metrics ────────────────────────────────────────────────────────────
    n_years    = len(prices) / 252
    total_ret  = equity_curve.iloc[-1] / initial - 1
    strat_cagr = (1 + total_ret) ** (1 / n_years) - 1
    strat_vol  = strategy_rets.std() * np.sqrt(252)
    strat_sharpe = (strat_cagr - 0.06) / strat_vol if strat_vol > 0 else 0

    rolling_max = equity_curve.cummax()
    strat_mdd   = ((equity_curve - rolling_max) / rolling_max).min()

    # Count number of trades (signal changes)
    trades = signal.diff().abs().sum()

    metrics = {
        "Strategy"    : f"SMA {fast_window}/{slow_window} Crossover",
        "Total Return": round(total_ret,    4),
        "CAGR"        : round(strat_cagr,   4),
        "Volatility"  : round(strat_vol,    4),
        "Sharpe Ratio": round(strat_sharpe, 4),
        "Max Drawdown": round(strat_mdd,    4),
        "Num Trades"  : int(trades),
    }

    signals_df = pd.DataFrame({
        "Price"   : price_series,
        "Fast SMA": fast_sma,
        "Slow SMA": slow_sma,
        "Signal"  : signal,
    })

    return {
        "equity_curve": equity_curve,
        "signals"     : signals_df,
        "metrics"     : metrics,
    }
