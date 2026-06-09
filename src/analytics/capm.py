"""
CAPM — Capital Asset Pricing Model
=====================================
One of the most famous models in all of finance.

The Core Idea:
  Every stock has two types of risk:
    1. Market Risk (Systematic)   — risk you CAN'T diversify away
                                    (e.g. recession, rate hikes affect everyone)
    2. Idiosyncratic Risk         — risk specific to one company
                                    (e.g. CEO scandal, product failure)

  CAPM says: investors are only COMPENSATED for market risk.
  Because idiosyncratic risk can be diversified away, the market
  won't pay you extra for taking it.

Key Metrics:
  Beta (β)  — How much does the stock move relative to the market?
              β = 1.0 → moves exactly like the market
              β > 1.0 → more volatile than market (amplified)
              β < 1.0 → less volatile (defensive)
              β < 0.0 → moves opposite to market (rare — gold, some bonds)

  Alpha (α) — Return ABOVE what CAPM predicts.
              α > 0 → the stock outperformed expectations (skill / luck)
              α < 0 → underperformed (bad management / bad luck)
              This is what fund managers are paid to generate.

  Expected Return formula:
              E(R_i) = R_f + β_i × (E(R_m) - R_f)
              where:
                R_f      = Risk-free rate
                E(R_m)   = Expected market return
                β_i × (E(R_m) - R_f) = "Market Risk Premium"
"""

import numpy as np
import pandas as pd
from scipy.stats         import linregress
from src.analytics.returns import daily_returns


def beta(
    asset_prices  : pd.Series,
    market_prices : pd.Series,
) -> float:
    """
    Calculate Beta: how much the asset moves per 1% market move.

    Uses linear regression of asset returns vs market returns.
    The slope of the regression line IS the beta.

    Parameters
    ----------
    asset_prices  : Price series of the stock
    market_prices : Price series of the market benchmark (S&P 500 / Nifty)

    Returns
    -------
    Beta as a float
    """
    # Align the two series (same dates only)
    combined = pd.DataFrame({"asset": asset_prices, "market": market_prices}).dropna()

    asset_rets  = combined["asset"].pct_change().dropna()
    market_rets = combined["market"].pct_change().dropna()

    # Linear regression: asset_return = alpha + beta × market_return
    slope, intercept, r_value, p_value, std_err = linregress(market_rets, asset_rets)

    return float(slope)


def alpha_beta(
    asset_prices  : pd.Series,
    market_prices : pd.Series,
    risk_free_rate: float = 0.06,
) -> dict:
    """
    Full CAPM regression: compute Alpha, Beta, R², and p-value.

    This tells you:
      - Beta: market sensitivity
      - Alpha: excess return (annualised)
      - R²: how much of the stock's movement is explained by the market
            R² = 0.8 means 80% of price moves are driven by market moves
      - p-value: statistical significance (< 0.05 = statistically significant)

    Parameters
    ----------
    risk_free_rate : Annual. Converted to daily internally.
    """
    combined = pd.DataFrame({"asset": asset_prices, "market": market_prices}).dropna()

    asset_rets  = combined["asset"].pct_change().dropna()
    market_rets = combined["market"].pct_change().dropna()

    # Convert annual risk-free rate to daily
    rf_daily = risk_free_rate / 252

    # Excess returns (above risk-free rate)
    excess_asset  = asset_rets  - rf_daily
    excess_market = market_rets - rf_daily

    # OLS Regression: excess_asset = alpha + beta × excess_market
    slope, intercept, r_value, p_value, std_err = linregress(excess_market, excess_asset)

    # Annualise alpha (daily alpha × 252 trading days)
    annual_alpha = intercept * 252

    return {
        "Beta"     : round(float(slope),        4),
        "Alpha"    : round(float(annual_alpha),  4),   # Annualised
        "R²"       : round(float(r_value**2),    4),
        "p-value"  : round(float(p_value),       4),
        "Std Error": round(float(std_err),       4),
    }


def expected_return_capm(
    beta_value     : float,
    risk_free_rate : float = 0.06,
    market_return  : float = 0.12,   # Historical average ~12% for broad market
) -> float:
    """
    CAPM formula: E(R_i) = R_f + β_i × (E(R_m) - R_f)

    Parameters
    ----------
    beta_value     : The asset's beta
    risk_free_rate : Annual risk-free rate (e.g. 0.06 = 6%)
    market_return  : Expected market return (e.g. 0.12 = 12%)

    Returns
    -------
    Expected annual return as a decimal (e.g. 0.14 = 14%)
    """
    market_risk_premium = market_return - risk_free_rate
    return risk_free_rate + beta_value * market_risk_premium


def full_capm_report(
    prices         : pd.DataFrame,
    market_prices  : pd.Series,
    risk_free_rate : float = 0.06,
    market_return  : float = 0.12,
) -> pd.DataFrame:
    """
    Run CAPM analysis for all assets in the DataFrame.

    Parameters
    ----------
    prices        : Multi-asset price DataFrame
    market_prices : Benchmark (SPY for US, NIFTYBEES.NS or ^NSEI for India)

    Returns
    -------
    DataFrame: one row per asset, columns = [Beta, Alpha, Expected Return, R², p-value]
    """
    rows = []

    for ticker in prices.columns:
        ab = alpha_beta(prices[ticker], market_prices, risk_free_rate)
        exp_ret = expected_return_capm(ab["Beta"], risk_free_rate, market_return)

        rows.append({
            "Ticker"         : ticker,
            "Beta"           : ab["Beta"],
            "Alpha (Ann.)"   : ab["Alpha"],
            "Expected Return": round(exp_ret, 4),
            "R²"             : ab["R²"],
            "p-value"        : ab["p-value"],
        })

    return pd.DataFrame(rows).set_index("Ticker")


def beta_interpretation(b: float) -> str:
    """Human-readable interpretation of a beta value."""
    if b < -0.5:  return "🔵 Strongly counter-cyclical (moves opposite to market)"
    if b < 0:     return "🔵 Slightly counter-cyclical"
    if b < 0.5:   return "🟢 Very defensive (low market sensitivity)"
    if b < 0.8:   return "🟢 Defensive (less volatile than market)"
    if b < 1.2:   return "🟡 Market-like (moves roughly with market)"
    if b < 1.5:   return "🟠 Slightly aggressive (more volatile than market)"
    return            "🔴 Highly aggressive (amplifies market moves)"
