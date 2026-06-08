"""
BUY & HOLD STRATEGY
====================
The simplest possible strategy: buy on Day 1, hold forever.

This is the benchmark everything else is compared against.
Surprisingly, most active strategies FAIL to beat buy & hold over long periods.
That's why it's so important — it sets the bar.
"""

import pandas as pd
from src.analytics.returns import cumulative_returns, growth_of_investment, cagr, total_return
from src.analytics.risk    import volatility, sharpe_ratio, max_drawdown


def run(prices: pd.DataFrame, initial: float = 100_000) -> dict:
    """
    Run Buy & Hold strategy on a single asset's prices.

    Parameters
    ----------
    prices  : Price series (single asset as DataFrame or Series)
    initial : Starting capital

    Returns
    -------
    dict with:
      - equity_curve : pd.Series — portfolio value over time
      - metrics      : dict      — performance summary
    """
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    cum_ret     = cumulative_returns(prices).iloc[:, 0]
    equity_curve = cum_ret * initial

    asset_df = prices

    metrics = {
        "Strategy"    : "Buy & Hold",
        "Total Return": round(total_return(asset_df).iloc[0], 4),
        "CAGR"        : round(cagr(asset_df).iloc[0], 4),
        "Volatility"  : round(volatility(asset_df).iloc[0], 4),
        "Sharpe Ratio": round(sharpe_ratio(asset_df).iloc[0], 4),
        "Max Drawdown": round(max_drawdown(asset_df).iloc[0], 4),
    }

    return {
        "equity_curve": equity_curve,
        "metrics"     : metrics,
    }
