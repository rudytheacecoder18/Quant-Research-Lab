"""
QUANT RESEARCH LAB — MAIN DASHBOARD
=====================================
Built with Streamlit. Run with:   streamlit run src/dashboard/app.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
import pandas as pd
import numpy as np

from src.data.data_loader            import load_price_data, get_asset_info, detect_market, format_market_cap
from src.analytics.returns           import growth_of_investment, cumulative_returns
from src.analytics.risk              import full_risk_report, drawdown_series, rolling_volatility
from src.portfolio.portfolio_engine  import (
    portfolio_metrics, portfolio_growth, portfolio_cumulative_returns,
    correlation_matrix
)
from src.strategies                  import buy_hold, sma_crossover
from src.visualization.charts        import (
    equity_curve, drawdown_chart, rolling_volatility_chart, correlation_heatmap,
    strategy_comparison, sma_signals_chart, metrics_bar_chart
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quant Research Lab",
    page_icon ="📈",
    layout    ="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
    background: #1a1d27;
    border: 1px solid #2d3142;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
  }
  .metric-label { color: #9e9e9e; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }
  .metric-value { color: #e0e0e0; font-size: 22px; font-weight: 700; margin-top: 4px; }
  .metric-pos   { color: #4CAF50 !important; }
  .metric-neg   { color: #F44336 !important; }
  div[data-testid="stSidebar"] { background: #0e1117; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Quant Research Lab")
    st.markdown("---")

    # Preset portfolios
    preset = st.selectbox("Quick Presets", [
        "Custom",
        "US Tech Giants",
        "Indian Blue Chips",
        "Global Diversified",
    ])

    presets = {
        "US Tech Giants"      : ["AAPL", "MSFT", "NVDA", "GOOGL", "SPY"],
        "Indian Blue Chips"   : ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "NIFTYBEES.NS"],
        "Global Diversified"  : ["AAPL", "MSFT", "RELIANCE.NS", "TCS.NS", "SPY"],
    }

    if preset != "Custom":
        default_tickers = ", ".join(presets[preset])
    else:
        default_tickers = "AAPL, MSFT, NVDA, SPY"

    ticker_input = st.text_area(
        "Tickers (comma-separated)",
        value=default_tickers,
        help="US: AAPL, MSFT, SPY  |  India: RELIANCE.NS, TCS.NS, NIFTYBEES.NS",
    )

    period_map = {"1 Year": "1y", "2 Years": "2y", "3 Years": "3y", "5 Years": "5y"}
    period_label = st.select_slider("Time Period", options=list(period_map.keys()), value="3 Years")
    period = period_map[period_label]

    risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 6.0, 0.5) / 100

    initial_investment = st.number_input("Initial Investment", value=100_000, step=10_000)

    st.markdown("---")
    run_button = st.button("🚀 Run Analysis", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("""
    **Quant Research Lab V1**
    
    Built with Python, yfinance, and Streamlit.
    
    Metrics covered:
    - CAGR, Volatility
    - Sharpe & Sortino Ratio  
    - Max Drawdown
    - Correlation Analysis
    - Strategy Backtesting
    """)


# ── Main Content ──────────────────────────────────────────────────────────────
st.title("📊 Quant Research Lab")
st.caption("Professional Equity Research Terminal · Supports US & Indian Markets")

if not run_button:
    # Landing state
    st.markdown("""
    ### Welcome 👋

    This tool lets you:
    1. **Analyse** any US or Indian stocks with professional metrics
    2. **Build** a custom portfolio and see combined risk/return
    3. **Backtest** strategies like SMA Crossover vs Buy & Hold
    4. **Explore** correlations to find diversification opportunities

    **Get started:** Enter tickers in the sidebar and click *Run Analysis*.

    > 💡 Try the **Quick Presets** for instant examples!
    """)
    st.stop()


# ── Load Data ─────────────────────────────────────────────────────────────────
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

with st.spinner("📥 Fetching market data…"):
    try:
        prices = load_price_data(tickers, period=period)
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.stop()

if prices.empty or len(prices.columns) == 0:
    st.error("No valid tickers found. Please check your symbols.")
    st.stop()

fetched_tickers = list(prices.columns)
currency_symbol = "₹" if all(detect_market(t) == "IN" for t in fetched_tickers) else "$"

# Fix currency for mixed portfolios
if any(detect_market(t) == "IN" for t in fetched_tickers) and any(detect_market(t) == "US" for t in fetched_tickers):
    currency_symbol = ""  # Mixed — don't show a single currency

st.success(f"✅ Loaded **{len(fetched_tickers)} assets** | {len(prices)} trading days | {prices.index[0].date()} → {prices.index[-1].date()}")


# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Performance",
    "🧮 Portfolio",
    "🔗 Correlation",
    "⚡ Strategy Lab",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1: PERFORMANCE DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Individual Asset Performance")

    # ── Metrics Table ──────────────────────────────────────────────────────
    metrics_df = full_risk_report(prices, risk_free_rate)

    # Format for display
    display_df = metrics_df.copy()
    pct_cols   = ["Total Return", "CAGR", "Volatility", "Max Drawdown"]
    ratio_cols = ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio"]

    for col in pct_cols:
        display_df[col] = display_df[col].apply(lambda x: f"{x*100:.2f}%")
    for col in ratio_cols:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}")

    st.dataframe(display_df, use_container_width=True)

    # ── Equity Curves ──────────────────────────────────────────────────────
    st.markdown("#### Equity Curves")
    st.caption(f"Growth of {currency_symbol}{initial_investment:,.0f} invested in each asset")
    growth_df = growth_of_investment(prices, initial=initial_investment)
    st.plotly_chart(equity_curve(growth_df, currency=currency_symbol), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Drawdown")
        st.caption("How far below the peak each asset has fallen")
        dd_df = drawdown_series(prices)
        st.plotly_chart(drawdown_chart(dd_df), use_container_width=True)

    with col2:
        st.markdown("#### Rolling 30-Day Volatility")
        st.caption("How risk has changed over time — spikes = turbulent periods")
        rv_df = rolling_volatility(prices)
        st.plotly_chart(rolling_volatility_chart(rv_df), use_container_width=True)

    # ── Metric Comparisons ─────────────────────────────────────────────────
    st.markdown("#### Metric Comparisons")
    metric_choice = st.selectbox(
        "Select metric to compare",
        ["Sharpe Ratio", "Sortino Ratio", "CAGR", "Volatility", "Max Drawdown"]
    )
    st.plotly_chart(metrics_bar_chart(metrics_df, metric_choice), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2: PORTFOLIO SIMULATOR
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Portfolio Simulator")
    st.caption("Combine your assets with custom weights and see the blended result")

    n = len(fetched_tickers)
    equal_weight = round(1.0 / n, 4)

    st.markdown("**Set Portfolio Weights**")
    st.caption("Weights will be auto-normalised if they don't sum to 100%")

    weight_cols = st.columns(min(n, 4))
    weights     = []

    for i, ticker in enumerate(fetched_tickers):
        col = weight_cols[i % len(weight_cols)]
        w   = col.number_input(ticker, min_value=0.0, max_value=1.0, value=equal_weight, step=0.05)
        weights.append(w)

    total_w = sum(weights)
    if not np.isclose(total_w, 1.0, atol=0.01):
        st.warning(f"Weights sum to {total_w*100:.1f}%. They will be auto-normalised.")

    if total_w > 0:
        # Normalise
        norm_weights = [w / total_w for w in weights]

        # Display normalised allocation
        alloc_df = pd.DataFrame({
            "Asset" : fetched_tickers,
            "Weight": [f"{w*100:.1f}%" for w in norm_weights],
        })
        st.dataframe(alloc_df, use_container_width=False, hide_index=True)

        # Calculate portfolio metrics
        port_m = portfolio_metrics(prices, norm_weights, risk_free_rate)

        # Display metric cards
        st.markdown("#### Portfolio Performance")
        m_cols = st.columns(6)
        cards  = [
            ("Total Return",  f"{port_m['Total Return']*100:.2f}%",  port_m['Total Return'] > 0),
            ("CAGR",          f"{port_m['CAGR']*100:.2f}%",          port_m['CAGR'] > 0),
            ("Volatility",    f"{port_m['Volatility']*100:.2f}%",    None),
            ("Sharpe Ratio",  f"{port_m['Sharpe Ratio']:.3f}",       port_m['Sharpe Ratio'] > 1),
            ("Sortino Ratio", f"{port_m['Sortino Ratio']:.3f}",      port_m['Sortino Ratio'] > 1),
            ("Max Drawdown",  f"{port_m['Max Drawdown']*100:.2f}%",  False),
        ]

        for col, (label, value, is_positive) in zip(m_cols, cards):
            color_class = ""
            if is_positive is True:  color_class = "metric-pos"
            if is_positive is False and label in ["Total Return", "CAGR", "Sharpe Ratio", "Sortino Ratio"]:
                color_class = "metric-neg"
            if label == "Max Drawdown": color_class = "metric-neg"

            col.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value {color_class}">{value}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Portfolio equity curve vs individual assets
        st.markdown("#### Portfolio vs Individual Assets")
        port_growth  = portfolio_growth(prices, norm_weights, initial=initial_investment)
        indiv_growth = growth_of_investment(prices, initial=initial_investment)

        combined = indiv_growth.copy()
        combined["Portfolio (Blended)"] = port_growth.values

        st.plotly_chart(equity_curve(combined, currency=currency_symbol, title="Portfolio vs Individual Assets"), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3: CORRELATION ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Correlation Analysis")
    st.caption("Understand how your assets move relative to each other")

    corr = correlation_matrix(prices)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.plotly_chart(correlation_heatmap(corr), use_container_width=True)

    with col2:
        st.markdown("#### How to Read This")
        st.markdown("""
        **Values range from -1 to +1:**

        🔴 **Near +1.0**  
        Assets move together.  
        Little diversification benefit.

        ⬜ **Near 0.0**  
        Assets move independently.  
        Good diversifiers!

        🔵 **Near -1.0**  
        Assets move opposite.  
        Best possible diversifier.

        ---
        **Rule of thumb:**  
        Correlation > 0.7 = watch out.  
        Look for assets < 0.3 to reduce portfolio risk.
        """)

    # Correlation table
    st.markdown("#### Correlation Matrix")
    st.dataframe(corr.style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1).format("{:.3f}"), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4: STRATEGY LAB
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Strategy Lab")
    st.caption("Backtest strategies and compare against Buy & Hold")

    strategy_ticker = st.selectbox("Select Asset to Backtest", fetched_tickers)
    asset_prices    = prices[[strategy_ticker]]

    col1, col2 = st.columns(2)
    with col1:
        fast_window = st.slider("Fast SMA (days)", 10, 100, 50, 5)
    with col2:
        slow_window = st.slider("Slow SMA (days)", 50, 300, 200, 10)

    if fast_window >= slow_window:
        st.warning("⚠️ Fast window must be smaller than Slow window.")
    else:
        # Run strategies
        bh_result  = buy_hold.run(asset_prices, initial=initial_investment)
        sma_result = sma_crossover.run(asset_prices, fast_window, slow_window, initial=initial_investment)

        # ── Equity Curve Comparison ────────────────────────────────────────
        st.markdown("#### Strategy Performance Comparison")
        curves = {
            "Buy & Hold"                             : bh_result["equity_curve"],
            f"SMA {fast_window}/{slow_window} Cross" : sma_result["equity_curve"],
        }
        st.plotly_chart(strategy_comparison(curves, currency=currency_symbol), use_container_width=True)

        # ── Metrics Comparison ─────────────────────────────────────────────
        st.markdown("#### Metrics Comparison")
        metrics_data = {
            k: v for k, v in bh_result["metrics"].items() if k != "Strategy"
        }
        sma_data = {
            k: v for k, v in sma_result["metrics"].items()
            if k not in ("Strategy", "Num Trades")
        }

        compare_df = pd.DataFrame({
            "Buy & Hold"                             : metrics_data,
            f"SMA {fast_window}/{slow_window} Cross" : sma_data,
        }).T

        # Format
        disp = compare_df.copy()
        for col in ["Total Return", "CAGR", "Volatility", "Max Drawdown"]:
            if col in disp.columns:
                disp[col] = disp[col].apply(lambda x: f"{x*100:.2f}%")
        for col in ["Sharpe Ratio"]:
            if col in disp.columns:
                disp[col] = disp[col].apply(lambda x: f"{x:.3f}")

        st.dataframe(disp, use_container_width=True)

        # ── SMA Signal Chart ───────────────────────────────────────────────
        st.markdown("#### Price Chart with SMA Signals")
        st.caption("▲ Green = Buy signal (Golden Cross)   ▼ Red = Sell signal (Death Cross)")
        st.plotly_chart(
            sma_signals_chart(sma_result["signals"], ticker=strategy_ticker),
            use_container_width=True
        )

        # ── Key Insight ────────────────────────────────────────────────────
        bh_sharpe  = bh_result["metrics"]["Sharpe Ratio"]
        sma_sharpe = sma_result["metrics"]["Sharpe Ratio"]

        if sma_sharpe > bh_sharpe:
            st.success(f"✅ SMA strategy outperformed Buy & Hold on a risk-adjusted basis (Sharpe: {sma_sharpe:.3f} vs {bh_sharpe:.3f})")
        else:
            st.info(f"ℹ️ Buy & Hold outperformed SMA on a risk-adjusted basis (Sharpe: {bh_sharpe:.3f} vs {sma_sharpe:.3f}) — showing that active strategies often struggle to beat passive investing!")

        st.caption(f"Number of trades generated by SMA strategy: **{sma_result['metrics']['Num Trades']}**")
