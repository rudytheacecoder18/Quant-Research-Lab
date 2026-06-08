"""
DATA LOADER
===========
This is the foundation of our entire project.
Think of it as the "pipeline" that fetches stock prices
from Yahoo Finance and cleans them up for analysis.

What it does:
  1. Takes a list of ticker symbols
  2. Downloads historical price data from the internet (via yfinance)
  3. Cleans the data (removes bad values, fills gaps)
  4. Returns it in a consistent format

Supports:
  - US stocks:     AAPL, MSFT, NVDA, SPY
  - Indian stocks: RELIANCE.NS, TCS.NS, NIFTYBEES.NS
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def load_price_data(
    tickers: list[str],
    start_date: str = None,
    end_date: str = None,
    period: str = "3y"
) -> pd.DataFrame:
    """
    Download and clean historical closing prices for a list of tickers.

    Parameters
    ----------
    tickers    : list of ticker symbols e.g. ["AAPL", "RELIANCE.NS"]
    start_date : "YYYY-MM-DD" — if provided, overrides `period`
    end_date   : "YYYY-MM-DD" — defaults to today
    period     : shorthand like "1y", "3y", "5y" (used if no start_date given)

    Returns
    -------
    pd.DataFrame with dates as index, tickers as columns, adjusted close prices as values
    """

    # Validate input
    if not tickers:
        raise ValueError("Please provide at least one ticker symbol.")

    # Clean tickers — strip whitespace, uppercase US tickers
    tickers = [t.strip() for t in tickers]

    print(f"📥 Fetching data for: {tickers}")

    # Determine date range
    if start_date:
        kwargs = {"start": start_date, "end": end_date or datetime.today().strftime("%Y-%m-%d")}
    else:
        kwargs = {"period": period}

    # Download from Yahoo Finance
    # auto_adjust=True means prices are already adjusted for splits & dividends
    raw = yf.download(
        tickers,
        progress=False,
        auto_adjust=True,
        **kwargs
    )

    # yfinance returns multi-level columns when multiple tickers are given
    # We only want the "Close" prices (daily closing price)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        # Single ticker — yfinance returns flat columns
        prices = raw[["Close"]]
        prices.columns = tickers

    # Drop rows where ALL tickers are missing (weekend/holiday gaps are fine to keep)
    prices = prices.dropna(how="all")

    # Forward-fill gaps (e.g. Indian market holiday while US market is open)
    prices = prices.ffill()

    # Drop any remaining NaN columns (bad tickers)
    prices = prices.dropna(axis=1, how="all")

    # Warn about any tickers that failed
    fetched = list(prices.columns)
    failed  = [t for t in tickers if t not in fetched]
    if failed:
        print(f"⚠️  Could not fetch data for: {failed}. Check ticker symbols.")

    print(f"✅ Successfully loaded {len(fetched)} assets | {len(prices)} trading days")
    print(f"   Date range: {prices.index[0].date()} → {prices.index[-1].date()}")

    return prices


def get_asset_info(ticker: str) -> dict:
    """
    Fetch basic information about a stock (name, sector, currency, etc.)

    Parameters
    ----------
    ticker : e.g. "AAPL" or "RELIANCE.NS"

    Returns
    -------
    dict with keys: name, sector, currency, exchange, market_cap
    """
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info

        return {
            "ticker"    : ticker,
            "name"      : info.get("longName", ticker),
            "sector"    : info.get("sector", "N/A"),
            "currency"  : info.get("currency", "N/A"),
            "exchange"  : info.get("exchange", "N/A"),
            "market_cap": info.get("marketCap", None),
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


def detect_market(ticker: str) -> str:
    """
    Detect whether a ticker is Indian (.NS / .BO) or US.

    Returns "IN" for Indian, "US" for US markets.
    """
    if ticker.upper().endswith(".NS") or ticker.upper().endswith(".BO"):
        return "IN"
    return "US"


def format_market_cap(market_cap: float, currency: str) -> str:
    """Convert raw market cap number to a readable string."""
    if not market_cap:
        return "N/A"

    if currency == "INR":
        cr = market_cap / 1e7  # Convert to Crores
        if cr >= 1_00_000:
            return f"₹{cr/1_00_000:.2f}L Cr"
        return f"₹{cr:,.0f} Cr"
    else:
        if market_cap >= 1e12:
            return f"${market_cap/1e12:.2f}T"
        elif market_cap >= 1e9:
            return f"${market_cap/1e9:.2f}B"
        return f"${market_cap/1e6:.2f}M"


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test with a mix of US and Indian tickers
    test_tickers = ["AAPL", "MSFT", "RELIANCE.NS", "TCS.NS"]

    prices = load_price_data(test_tickers, period="1y")
    print("\nFirst 3 rows of price data:")
    print(prices.head(3))
    print("\nShape:", prices.shape)

    print("\n── Asset Info ──")
    for t in test_tickers[:2]:  # Just test 2 to keep it fast
        info = get_asset_info(t)
        mc   = format_market_cap(info.get("market_cap"), info.get("currency", "USD"))
        print(f"  {t}: {info['name']} | {info['sector']} | Market Cap: {mc}")
