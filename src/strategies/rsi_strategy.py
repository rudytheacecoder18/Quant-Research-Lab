"""
RSI STRATEGY — Relative Strength Index
=========================================
RSI is a "momentum oscillator" — it measures how fast prices are moving
and whether an asset is overbought or oversold.

How RSI is calculated:
  1. Take the last 14 days of price changes
  2. Separate gains (positive days) and losses (negative days)
  3. Average Gain / Average Loss = Relative Strength (RS)
  4. RSI = 100 − (100 / (1 + RS))

RSI ranges from 0 to 100:
  RSI > 70 → Overbought (price rose too fast, may reverse downward)
  RSI < 30 → Oversold  (price fell too fast, may bounce upward)
  RSI = 50 → Neutral

Trading Rules (contrarian / mean-reversion):
  BUY  when RSI < 30 (oversold — expected to bounce)
  SELL when RSI > 70 (overbought — expected to fall)

This is the OPPOSITE of trend-following (SMA/EMA).
RSI bets that extremes will revert to normal.
In sideways/choppy markets, RSI tends to work better than SMA.
In trending markets, SMA/EMA tends to work better.
"""

import numpy as np
import pandas as pd


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate RSI for a price series.

    Parameters
    ----------
    prices : Price series
    period : Lookback window (default 14 — standard)
    """
    delta  = prices.diff()

    gains  = delta.clip(lower=0)   # Keep only positive moves
    losses = (-delta).clip(lower=0)  # Keep only negative moves (make positive)

    # Exponential moving average of gains and losses
    avg_gain = gains.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = losses.ewm(com=period - 1, min_periods=period).mean()

    rs  = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def run(
    prices      : pd.DataFrame,
    rsi_period  : int   = 14,
    oversold    : float = 30,
    overbought  : float = 70,
    initial     : float = 100_000,
) -> dict:
    """
    RSI mean-reversion strategy.

    BUY when RSI drops below oversold threshold.
    SELL when RSI rises above overbought threshold.

    Parameters
    ----------
    rsi_period : RSI calculation window (default 14)
    oversold   : RSI buy threshold (default 30)
    overbought : RSI sell threshold (default 70)
    initial    : Starting capital
    """
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    price_series = prices.iloc[:, 0]
    rsi          = compute_rsi(price_series, rsi_period)

    # Generate signal:
    #   Enter (buy)  when RSI < oversold
    #   Exit  (sell) when RSI > overbought
    #   Hold position in between

    signal   = pd.Series(np.nan, index=price_series.index)
    position = 0  # 0 = out of market, 1 = invested

    for i in range(len(rsi)):
        r = rsi.iloc[i]
        if pd.isna(r):
            signal.iloc[i] = 0
            continue

        if position == 0 and r < oversold:
            position = 1   # BUY
        elif position == 1 and r > overbought:
            position = 0   # SELL

        signal.iloc[i] = position

    # Shift by 1 day to avoid look-ahead bias
    signal = signal.shift(1).fillna(0)

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
        "Strategy"    : f"RSI({rsi_period}) {oversold}/{overbought}",
        "Total Return": round(total_ret,  4),
        "CAGR"        : round(strat_cagr, 4),
        "Volatility"  : round(strat_vol,  4),
        "Sharpe Ratio": round(sharpe,     4),
        "Max Drawdown": round(float(mdd), 4),
        "Num Trades"  : int(trades),
    }

    return {
        "equity_curve": equity_curve,
        "rsi_series"  : rsi,
        "signal"      : signal,
        "metrics"     : metrics,
    }
