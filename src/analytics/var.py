"""
VALUE AT RISK (VaR)
====================
VaR answers one of the most important questions in risk management:

  "How much could I lose tomorrow, and with what probability?"

Example interpretation:
  95% VaR = -2.1%  →  On 95% of days, you won't lose more than 2.1%
                       But on 5% of days (worst days), you WILL lose more

  99% VaR = -3.8%  →  On 99% of days, you won't lose more than 3.8%
                       Only 1% of days are worse than this

Three methods covered:
  1. Historical VaR  — look at actual past returns and take the percentile
                       Simple, no assumptions, but depends on history
  2. Parametric VaR  — assume returns are normally distributed, use statistics
                       Fast, mathematically clean, but assumes normality
  3. CVaR (Expected Shortfall) — average loss in the WORST scenarios
                       More informative than VaR alone

VaR is used by every bank, hedge fund, and regulator in the world.
It's a core part of Basel III (global banking regulation).
"""

import numpy as np
import pandas as pd
from src.analytics.returns  import daily_returns
from src.portfolio.portfolio_engine import portfolio_returns


def historical_var(
    returns        : pd.Series,
    confidence     : float = 0.95,
    holding_period : int   = 1,
) -> float:
    """
    Historical VaR: Just take the percentile of actual past returns.

    Parameters
    ----------
    returns        : Daily return series
    confidence     : 0.95 for 95% VaR, 0.99 for 99% VaR
    holding_period : Days to hold (1 = daily, 10 = 10-day VaR)

    Returns
    -------
    VaR as a negative decimal (e.g. -0.021 means -2.1% loss)
    """
    # Scale to holding period using square-root-of-time rule
    scaled = returns * np.sqrt(holding_period)

    # The percentile at (1 - confidence) gives the loss threshold
    # e.g. at 95% confidence: np.percentile(returns, 5)
    var = np.percentile(scaled, (1 - confidence) * 100)

    return float(var)


def parametric_var(
    returns        : pd.Series,
    confidence     : float = 0.95,
    holding_period : int   = 1,
) -> float:
    """
    Parametric VaR (Variance-Covariance method).

    Assumes returns are normally distributed.
    Uses mean and standard deviation to analytically compute VaR.

    Formula: VaR = μ - z × σ
      where z is the z-score for the confidence level
        z = 1.645 for 95%, z = 2.326 for 99%

    Faster than historical, but the normal distribution assumption
    often underestimates tail risk (real markets have "fat tails").
    """
    from scipy.stats import norm

    mu    = returns.mean() * holding_period
    sigma = returns.std()  * np.sqrt(holding_period)

    # z-score: how many standard deviations from the mean
    z_score = norm.ppf(1 - confidence)   # negative value (left tail)

    var = mu + z_score * sigma

    return float(var)


def cvar(
    returns    : pd.Series,
    confidence : float = 0.95,
) -> float:
    """
    Conditional VaR (CVaR) / Expected Shortfall (ES).

    VaR tells you the threshold. CVaR tells you:
      "Given that we exceeded VaR, what's the average loss?"

    Example:
      95% VaR  = -2.1%  →  "5% of days are worse than this"
      95% CVaR = -3.4%  →  "On those worst 5% of days, average loss is 3.4%"

    CVaR is considered a better risk measure than VaR because
    it captures the severity of losses in the tail, not just the threshold.
    """
    var       = historical_var(returns, confidence)
    tail_rets = returns[returns <= var]  # Only the worst days

    if len(tail_rets) == 0:
        return var

    return float(tail_rets.mean())


def full_var_report(
    prices         : pd.DataFrame,
    weights        : list[float]  = None,
    portfolio_value: float        = 100_000,
    risk_free_rate : float        = 0.06,
) -> dict:
    """
    Complete VaR report for either individual assets or a portfolio.

    Parameters
    ----------
    prices          : Price DataFrame
    weights         : If provided, compute portfolio VaR; else per-asset
    portfolio_value : Total value in rupees/dollars (for absolute loss numbers)

    Returns
    -------
    dict with VaR metrics at 95% and 99% confidence
    """
    if weights is not None:
        # Portfolio-level VaR
        rets = portfolio_returns(prices, weights)
        label = "Portfolio"
    else:
        # Per-asset (use equal weight for simplicity)
        n = len(prices.columns)
        rets = portfolio_returns(prices, [1/n]*n)
        label = "Equal-Weight Portfolio"

    var_95  = historical_var(rets, 0.95)
    var_99  = historical_var(rets, 0.99)
    pvar_95 = parametric_var(rets, 0.95)
    pvar_99 = parametric_var(rets, 0.99)
    cvar_95 = cvar(rets, 0.95)
    cvar_99 = cvar(rets, 0.99)

    return {
        "label"           : label,
        "Historical VaR 95%"  : round(var_95,  4),
        "Historical VaR 99%"  : round(var_99,  4),
        "Parametric VaR 95%"  : round(pvar_95, 4),
        "Parametric VaR 99%"  : round(pvar_99, 4),
        "CVaR 95%"            : round(cvar_95, 4),
        "CVaR 99%"            : round(cvar_99, 4),
        # Absolute loss values
        "1-Day Loss at 95% (₹/$)" : round(abs(var_95)  * portfolio_value, 2),
        "1-Day Loss at 99% (₹/$)" : round(abs(var_99)  * portfolio_value, 2),
        "returns_series"          : rets,   # for plotting
    }


def var_series(returns: pd.Series, window: int = 252, confidence: float = 0.95) -> pd.Series:
    """
    Rolling VaR — shows how portfolio risk has changed over time.

    Uses a 1-year rolling window of historical returns.
    Useful for seeing when the portfolio became riskier.
    """
    return returns.rolling(window).apply(
        lambda x: historical_var(pd.Series(x), confidence), raw=False
    )
