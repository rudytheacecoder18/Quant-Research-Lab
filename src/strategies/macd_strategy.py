"""
MACD STRATEGY — Moving Average Convergence Divergence
=======================================================
MACD is one of the most widely used technical indicators.
It combines trend-following AND momentum into a single indicator.

Components:
  MACD Line   = EMA(12) − EMA(26)
                The difference between fast and slow EMAs
                Positive = short-term trend is above long-term (bullish)
                Negative = short-term below long-term (bearish)

  Signal Line = EMA(9) of the MACD Line
                A smoothed version of the MACD line
                Used to generate buy/sell signals

  Histogram   = MACD Line − Signal Line
                Visual representation of the gap
                Bars growing → momentum increasing
                Bars shrinking → momentum fading

Trading Rules:
  BUY  when MACD Line crosses ABOVE Signal Line (bullish crossover)
  SELL when MACD Line crosses BELOW Signal Line (bearish crossover)

Why traders like MACD:
  - Captures both trend direction AND momentum
  - Works across many asset classes and time frames
  - Histogram makes it easy to see momentum shifts visually
"""

import numpy as np
import pandas as pd


def compute_macd(
    prices      : pd.Series,
    fast_period : int = 12,
    slow_period : int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    """
    Calculate MACD components for a price series.

    Returns
    -------
    DataFrame with: MACD, Signal, Histogram
    """
    ema_fast  = prices.ewm(span=fast_period,   adjust=False).mean()
    ema_slow  = prices.ewm(span=slow_period,   adjust=False).mean()

    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram   = macd_line - signal_line

    return pd.DataFrame({
        "MACD"     : macd_line,
        "Signal"   : signal_line,
        "Histogram": histogram,
    })


def run(
    prices        : pd.DataFrame,
    fast_period   : int   = 12,
    slow_period   : int   = 26,
    signal_period : int   = 9,
    initial       : float = 100_000,
) -> dict:
    """
    MACD crossover strategy.

    BUY  when MACD crosses above Signal Line.
    SELL when MACD crosses below Signal Line.
    """
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    price_series = prices.iloc[:, 0]
    macd_df      = compute_macd(price_series, fast_period, slow_period, signal_period)

    # Signal: 1 when MACD > Signal (bullish), 0 otherwise
    signal = (macd_df["MACD"] > macd_df["Signal"]).astype(int).shift(1).fillna(0)

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
        "Strategy"    : f"MACD ({fast_period},{slow_period},{signal_period})",
        "Total Return": round(total_ret,  4),
        "CAGR"        : round(strat_cagr, 4),
        "Volatility"  : round(strat_vol,  4),
        "Sharpe Ratio": round(sharpe,     4),
        "Max Drawdown": round(float(mdd), 4),
        "Num Trades"  : int(trades),
    }

    return {
        "equity_curve": equity_curve,
        "macd_df"     : macd_df,
        "signal"      : signal,
        "metrics"     : metrics,
    }
