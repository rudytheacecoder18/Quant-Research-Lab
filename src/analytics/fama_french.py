"""
FAMA-FRENCH 3-FACTOR MODEL
============================
The most important factor model in academic finance.
Published by Eugene Fama & Kenneth French in 1992.
Fama won the Nobel Prize in 2013 partly for this work.

The Problem with CAPM:
  CAPM explains stock returns using just ONE factor: the market.
  But in practice, stocks with certain characteristics systematically
  outperform — CAPM can't explain this.

The Three Factors:
  1. Market (Mkt-RF) — excess market return over risk-free rate
                        Same as CAPM's market risk premium
                        Captures broad market exposure

  2. SMB — "Small Minus Big"
            Small-cap stocks historically outperform large-cap stocks.
            SMB = return of small stocks − return of large stocks
            A positive SMB loading means the portfolio tilts toward
            small companies (higher expected return, higher risk)

  3. HML — "High Minus Low"
            Value stocks (high book-to-market) outperform growth stocks
            (low book-to-market) over time.
            HML = return of value stocks − return of growth stocks
            A positive HML loading = value tilt

The Model:
  R_i - R_f = α + β₁(Mkt-RF) + β₂(SMB) + β₃(HML) + ε

  Where:
    α        = Alpha (excess return not explained by the 3 factors)
    β₁       = Market beta (same concept as CAPM)
    β₂       = Size beta (positive = small-cap tilt)
    β₃       = Value beta (positive = value tilt)

Why it matters for interviews:
  - FF3 is the standard benchmark for evaluating fund performance
  - Every CFA, quant researcher, and portfolio manager knows it
  - Alpha against FF3 is far more meaningful than raw alpha
    because it controls for well-known risk premia

Data Source:
  Kenneth French's Data Library (public):
  https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
"""

import io
import zipfile
import urllib.request
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import linregress


# ── Factor data fetching ─────────────────────────────────────────────────────

FF_DAILY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)


def fetch_ff_factors(start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Download daily Fama-French 3 factors from Kenneth French's website.

    Returns a DataFrame with columns:
      Mkt-RF : Daily excess market return (market return minus risk-free)
      SMB    : Small Minus Big factor
      HML    : High Minus Low factor
      RF     : Daily risk-free rate (1-month T-bill)

    All values are in DECIMAL form (e.g. 0.01 = 1% daily return).

    Falls back to synthetic proxies if the server is unreachable.
    """
    try:
        with urllib.request.urlopen(FF_DAILY_URL, timeout=10) as response:
            zip_bytes = response.read()

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            csv_name = [n for n in zf.namelist() if n.endswith(".CSV") or n.endswith(".csv")][0]
            with zf.open(csv_name) as f:
                raw = f.read().decode("latin-1")

        # French's CSV has a header block of text before the data — skip it
        lines = raw.splitlines()
        data_start = next(
            i for i, line in enumerate(lines)
            if line.strip() and line.strip()[0].isdigit()
        )
        csv_clean = "\n".join(lines[data_start:])

        # Find where the annual data section starts (if present) and stop there
        end_idx = None
        for i, line in enumerate(csv_clean.splitlines()):
            parts = line.strip().split(",")
            # Annual section rows have 4-digit year with no month/day
            if len(parts) >= 2 and len(parts[0].strip()) == 4:
                end_idx = i
                break

        if end_idx:
            csv_clean = "\n".join(csv_clean.splitlines()[:end_idx])

        df = pd.read_csv(
            io.StringIO(csv_clean),
            header=None,
            names=["Date", "Mkt-RF", "SMB", "HML", "RF"],
        )
        df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m%d", errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date")

        # French reports in percent — convert to decimal
        for col in ["Mkt-RF", "SMB", "HML", "RF"]:
            df[col] = pd.to_numeric(df[col], errors="coerce") / 100.0

        df = df.dropna()

    except Exception as e:
        print(f"⚠️  Could not fetch FF factors from French's website: {e}")
        print("   Using synthetic factor proxies instead.")
        df = _synthetic_ff_factors(start_date, end_date)

    # Filter to requested date range
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]

    return df.sort_index()


def _synthetic_ff_factors(start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Generate synthetic Fama-French factors as a fallback.

    Uses historically realistic parameters from Fama-French (1993):
      Mkt-RF : ~8% annual premium, ~18% annual vol
      SMB    : ~3% annual premium, ~11% vol
      HML    : ~4% annual premium, ~12% vol
      RF     : ~5% annual (approximating T-bill rate)

    NOTE: These are synthetic — they capture realistic statistical
    properties but are NOT real historical factor returns.
    A warning is shown in the dashboard when this path is taken.
    """
    np.random.seed(99)

    end   = pd.Timestamp(end_date)   if end_date   else pd.Timestamp.today()
    start = pd.Timestamp(start_date) if start_date else end - pd.Timedelta(days=5*365)

    dates = pd.bdate_range(start, end)   # Business days only
    n     = len(dates)

    # Annualised parameters → daily
    params = {
        "Mkt-RF": (0.08 / 252, 0.18 / np.sqrt(252)),
        "SMB"   : (0.03 / 252, 0.11 / np.sqrt(252)),
        "HML"   : (0.04 / 252, 0.12 / np.sqrt(252)),
        "RF"    : (0.05 / 252, 0.001),
    }

    data = {col: np.random.normal(mu, sigma, n) for col, (mu, sigma) in params.items()}
    data["RF"] = np.abs(data["RF"])   # risk-free rate is never negative

    return pd.DataFrame(data, index=dates)


# ── Regression ────────────────────────────────────────────────────────────────

def run_ff3_regression(
    asset_returns : pd.Series,
    ff_factors    : pd.DataFrame,
) -> dict:
    """
    Run the Fama-French 3-Factor regression for a single asset.

    Regression equation:
      R_i - RF = α + β₁(Mkt-RF) + β₂(SMB) + β₃(HML) + ε

    Parameters
    ----------
    asset_returns : Daily return series for one asset
    ff_factors    : DataFrame with columns [Mkt-RF, SMB, HML, RF]

    Returns
    -------
    dict with alpha, betas, R², t-stats, p-values
    """
    from sklearn.linear_model import LinearRegression

    # Align dates — inner join so both series cover the same period
    combined = pd.DataFrame({
        "asset": asset_returns,
        "Mkt-RF": ff_factors["Mkt-RF"],
        "SMB"   : ff_factors["SMB"],
        "HML"   : ff_factors["HML"],
        "RF"    : ff_factors["RF"],
    }).dropna()

    if len(combined) < 60:
        return {"error": "Insufficient overlapping data (need 60+ days)"}

    # Excess return of the asset over the risk-free rate
    y = (combined["asset"] - combined["RF"]).values
    X = combined[["Mkt-RF", "SMB", "HML"]].values

    # OLS: y = X @ betas + alpha
    X_with_const = np.column_stack([np.ones(len(X)), X])
    coeffs, residuals, _, _ = np.linalg.lstsq(X_with_const, y, rcond=None)

    alpha_daily = coeffs[0]
    beta_mkt    = coeffs[1]
    beta_smb    = coeffs[2]
    beta_hml    = coeffs[3]

    # Annualise alpha
    alpha_annual = alpha_daily * 252

    # R-squared
    y_pred  = X_with_const @ coeffs
    ss_res  = np.sum((y - y_pred) ** 2)
    ss_tot  = np.sum((y - y.mean()) ** 2)
    r2      = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Standard errors and t-statistics
    n, k    = X_with_const.shape
    sigma2  = ss_res / (n - k)
    cov_mat = sigma2 * np.linalg.inv(X_with_const.T @ X_with_const)
    se      = np.sqrt(np.diag(cov_mat))

    from scipy.stats import t as t_dist
    t_stats = coeffs / se
    p_vals  = 2 * t_dist.sf(np.abs(t_stats), df=n - k)

    return {
        "Alpha (Ann.)"   : round(float(alpha_annual), 4),
        "Beta Market"    : round(float(beta_mkt),     4),
        "Beta SMB"       : round(float(beta_smb),     4),
        "Beta HML"       : round(float(beta_hml),     4),
        "R²"             : round(float(r2),           4),
        "t-stat Alpha"   : round(float(t_stats[0]),   4),
        "t-stat Mkt"     : round(float(t_stats[1]),   4),
        "t-stat SMB"     : round(float(t_stats[2]),   4),
        "t-stat HML"     : round(float(t_stats[3]),   4),
        "p-val Alpha"    : round(float(p_vals[0]),    4),
        "n_obs"          : int(n),
    }


def full_ff3_report(
    prices     : pd.DataFrame,
    ff_factors : pd.DataFrame,
) -> pd.DataFrame:
    """
    Run FF3 regression for every asset in the prices DataFrame.

    Returns a summary DataFrame: one row per asset.
    """
    from src.analytics.returns import daily_returns

    rets = daily_returns(prices)
    rows = []

    for ticker in rets.columns:
        result = run_ff3_regression(rets[ticker], ff_factors)
        if "error" not in result:
            result["Ticker"] = ticker
            rows.append(result)
        else:
            rows.append({"Ticker": ticker, "error": result["error"]})

    df = pd.DataFrame(rows).set_index("Ticker")
    return df


def interpret_ff3(row: pd.Series) -> str:
    """
    Plain-English interpretation of FF3 factor loadings for one asset.
    """
    lines = []

    # Alpha
    a = row.get("Alpha (Ann.)", 0)
    sig = "✅ statistically significant (p < 0.05)" if row.get("p-val Alpha", 1) < 0.05 else "⚠️ not statistically significant"
    if a > 0:
        lines.append(f"**Alpha: +{a*100:.2f}%/yr** — outperforms the 3-factor model ({sig})")
    else:
        lines.append(f"**Alpha: {a*100:.2f}%/yr** — underperforms the 3-factor model ({sig})")

    # Market beta
    b_mkt = row.get("Beta Market", 1)
    if b_mkt > 1.2:
        lines.append(f"**Market β: {b_mkt:.2f}** — aggressive, amplifies market moves")
    elif b_mkt < 0.8:
        lines.append(f"**Market β: {b_mkt:.2f}** — defensive, dampens market swings")
    else:
        lines.append(f"**Market β: {b_mkt:.2f}** — market-like exposure")

    # SMB
    b_smb = row.get("Beta SMB", 0)
    if b_smb > 0.2:
        lines.append(f"**SMB β: +{b_smb:.2f}** — small-cap tilt (higher risk, historically higher return)")
    elif b_smb < -0.2:
        lines.append(f"**SMB β: {b_smb:.2f}** — large-cap tilt (more stable)")
    else:
        lines.append(f"**SMB β: {b_smb:.2f}** — neutral size exposure")

    # HML
    b_hml = row.get("Beta HML", 0)
    if b_hml > 0.2:
        lines.append(f"**HML β: +{b_hml:.2f}** — value tilt (high book-to-market)")
    elif b_hml < -0.2:
        lines.append(f"**HML β: {b_hml:.2f}** — growth tilt (low book-to-market, e.g. tech stocks)")
    else:
        lines.append(f"**HML β: {b_hml:.2f}** — neutral style exposure")

    # R²
    r2 = row.get("R²", 0)
    lines.append(f"**R²: {r2:.3f}** — {r2*100:.0f}% of this asset's returns are explained by the 3 factors")

    return "\n\n".join(lines)
