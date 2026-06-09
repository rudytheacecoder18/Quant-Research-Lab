"""
QUANT RESEARCH LAB — V2 DASHBOARD
====================================
Run with:   streamlit run src/dashboard/app.py

New in V2:
  • Monte Carlo Simulation + Efficient Frontier
  • Portfolio Optimizer (Max Sharpe / Min Variance)
  • Value at Risk (Historical, Parametric, CVaR)
  • CAPM / Beta / Alpha analysis
  • EMA, RSI, MACD strategies + full comparison table
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
import streamlit as st

# ── Data ──────────────────────────────────────────────────────────────────────
from src.data.data_loader import (
    load_price_data, detect_market, format_market_cap,
)

# ── Analytics ────────────────────────────────────────────────────────────────
from src.analytics.returns import growth_of_investment, cumulative_returns
from src.analytics.risk    import full_risk_report, drawdown_series, rolling_volatility
from src.analytics.var     import full_var_report, historical_var, cvar as compute_cvar
from src.analytics.capm    import full_capm_report, beta_interpretation

# ── Portfolio ────────────────────────────────────────────────────────────────
from src.portfolio.portfolio_engine import (
    portfolio_metrics, portfolio_growth,
    portfolio_returns, correlation_matrix,
)
from src.portfolio.monte_carlo import (
    run_simulation, get_max_sharpe, get_min_variance,
)
from src.portfolio.optimizer import optimize, efficient_frontier_curve

# ── Strategies ────────────────────────────────────────────────────────────────
from src.strategies import buy_hold, sma_crossover, ema_strategy, rsi_strategy, macd_strategy

# ── Charts ────────────────────────────────────────────────────────────────────
from src.visualization.charts import (
    equity_curve, drawdown_chart, rolling_volatility_chart,
    correlation_heatmap, metrics_bar_chart, strategy_comparison,
    sma_signals_chart,
)
from src.visualization.frontier_charts import (
    monte_carlo_scatter, weights_bar,
    var_distribution,
    security_market_line, beta_bar,
    all_strategies_equity, macd_chart, rsi_chart, strategy_heatmap,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quant Research Lab",
    page_icon ="📈",
    layout    ="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .metric-card {
    background:#1a1d27; border:1px solid #2d3142;
    border-radius:8px; padding:16px; text-align:center;
  }
  .metric-label { color:#9e9e9e; font-size:11px; text-transform:uppercase; letter-spacing:1px; }
  .metric-value { color:#e0e0e0; font-size:20px; font-weight:700; margin-top:4px; }
  .pos { color:#4CAF50 !important; }
  .neg { color:#F44336 !important; }
  .neu { color:#FFC107 !important; }
  div[data-testid="stSidebar"] { background:#0e1117; }
  .stTabs [data-baseweb="tab"] { font-size:14px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def fmt_pct(v):  return f"{v*100:.2f}%"
def fmt_rat(v):  return f"{v:.3f}"
def fmt_abs(v, sym): return f"{sym}{abs(v):,.0f}"

def metric_card(col, label, value, positive=None):
    cls = ""
    if positive is True:  cls = "pos"
    if positive is False: cls = "neg"
    col.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value {cls}">{value}</div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Quant Research Lab")
    st.markdown("### V2 — Professional Edition")
    st.markdown("---")

    preset = st.selectbox("Quick Presets", [
        "Custom",
        "US Tech Giants",
        "Indian Blue Chips",
        "Global Diversified",
        "Defensive Mix",
    ])
    presets = {
        "US Tech Giants"   : "AAPL, MSFT, NVDA, GOOGL, SPY",
        "Indian Blue Chips": "RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS, NIFTYBEES.NS",
        "Global Diversified": "AAPL, MSFT, RELIANCE.NS, TCS.NS, SPY",
        "Defensive Mix"    : "JNJ, KO, PG, SPY, GLD",
    }
    default_tickers = presets.get(preset, "AAPL, MSFT, NVDA, SPY")

    ticker_input = st.text_area(
        "Tickers (comma-separated)",
        value=default_tickers,
        help="US: AAPL, MSFT, SPY  |  India: RELIANCE.NS, TCS.NS",
    )

    years = st.slider(
        "Time Period (Years)",
        min_value=1.0,
        max_value=5.0,
        value=3.0,
        step=0.25,
    )
    st.caption(f"≈ {int(years * 365.25):,} calendar days  ·  ~{int(years * 252):,} trading days")

    rf_rate   = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 6.0, 0.5) / 100
    mkt_ret   = st.slider("Expected Market Return (%)", 5.0, 20.0, 12.0, 0.5) / 100
    init_inv  = st.number_input("Initial Investment", value=100_000, step=10_000)

    st.markdown("---")

    # Benchmark selector for CAPM
    benchmark_input = st.text_input(
        "Benchmark Ticker (for CAPM/Beta)",
        value="SPY",
        help="SPY = S&P500  |  ^NSEI = Nifty50  |  NIFTYBEES.NS = Nifty ETF",
    )

    # Monte Carlo settings
    with st.expander("⚙️ Monte Carlo Settings"):
        n_portfolios = st.slider("Number of Portfolios", 1000, 10000, 5000, 500)
        mc_seed      = st.number_input("Random Seed", value=42, step=1)

    st.markdown("---")
    run_btn = st.button("🚀 Run Analysis", use_container_width=True, type="primary")

    st.markdown("""
    **V2 Features:**
    - ✅ Performance Dashboard
    - ✅ Portfolio Simulator
    - ✅ Correlation Analysis
    - ✅ Monte Carlo Simulation
    - ✅ Efficient Frontier
    - ✅ Portfolio Optimizer
    - ✅ Value at Risk (VaR/CVaR)
    - ✅ CAPM / Beta / Alpha
    - ✅ EMA, RSI, MACD Strategies
    """)


# ─────────────────────────────────────────────────────────────────────────────
# LANDING
# ─────────────────────────────────────────────────────────────────────────────
st.title("📊 Quant Research Lab  ·  V2")
st.caption("Professional Equity Research Terminal  ·  US & Indian Markets")

if not run_btn:
    c1, c2, c3 = st.columns(3)
    c1.info("**📊 V1 Features**\n\nPerformance metrics · Portfolio simulator · Correlation analysis · SMA backtesting")
    c2.success("**🆕 V2 New**\n\nMonte Carlo · Efficient Frontier · Optimizer · VaR/CVaR · CAPM/Beta · EMA · RSI · MACD")
    c3.warning("**🗺️ Roadmap**\n\nFama-French factors · ML alpha models · Live market data · PDF reports")
    st.markdown("---")
    st.markdown("**Get started:** Enter tickers in the sidebar and click *Run Analysis*.")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

with st.spinner("📥 Fetching market data…"):
    try:
        prices = load_price_data(tickers, years=years)
    except Exception as e:
        st.error(f"❌ Data error: {e}")
        st.stop()

    # Benchmark for CAPM
    bench_ticker = benchmark_input.strip().upper()
    try:
        bench_prices = load_price_data([bench_ticker], years=years)
        bench_series = bench_prices.iloc[:, 0]
        # Align to same dates as main prices
        bench_series = bench_series.reindex(prices.index).ffill()
    except Exception:
        bench_series = None

if prices.empty:
    st.error("No valid tickers. Please check your symbols.")
    st.stop()

tickers = list(prices.columns)
n       = len(tickers)
mixed   = any(detect_market(t) == "IN" for t in tickers) and any(detect_market(t) == "US" for t in tickers)
cur     = "" if mixed else ("₹" if detect_market(tickers[0]) == "IN" else "$")

st.success(
    f"✅ **{n} assets** · {len(prices)} trading days · "
    f"{prices.index[0].date()} → {prices.index[-1].date()}"
    + (f"  |  Benchmark: **{bench_ticker}**" if bench_series is not None else "")
)

equal_w = [1.0 / n] * n


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_perf, tab_port, tab_mc, tab_opt, tab_var, tab_capm, tab_strat = st.tabs([
    "📊 Performance",
    "🧮 Portfolio",
    "🎲 Monte Carlo",
    "⚡ Optimizer",
    "🛡️ Risk (VaR)",
    "📐 CAPM / Beta",
    "🔬 Strategy Lab",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab_perf:
    st.subheader("Individual Asset Performance")

    metrics_df = full_risk_report(prices, rf_rate)

    disp = metrics_df.copy()
    for col in ["Total Return","CAGR","Volatility","Max Drawdown"]:
        disp[col] = disp[col].apply(fmt_pct)
    for col in ["Sharpe Ratio","Sortino Ratio","Calmar Ratio"]:
        disp[col] = disp[col].apply(fmt_rat)
    st.dataframe(disp, use_container_width=True)

    st.markdown("#### Equity Curves")
    gdf = growth_of_investment(prices, initial=init_inv)
    st.plotly_chart(equity_curve(gdf, currency=cur), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Drawdown")
        st.plotly_chart(drawdown_chart(drawdown_series(prices)), use_container_width=True)
    with c2:
        st.markdown("#### Rolling 30-Day Volatility")
        st.plotly_chart(rolling_volatility_chart(rolling_volatility(prices)), use_container_width=True)

    st.markdown("#### Metric Comparison")
    m_choice = st.selectbox("Metric", ["Sharpe Ratio","Sortino Ratio","CAGR","Volatility","Max Drawdown"])
    st.plotly_chart(metrics_bar_chart(metrics_df, m_choice), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PORTFOLIO SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
with tab_port:
    st.subheader("Portfolio Simulator")

    w_cols   = st.columns(min(n, 4))
    user_w   = []
    for i, t in enumerate(tickers):
        wv = w_cols[i % len(w_cols)].number_input(t, 0.0, 1.0, round(1/n, 3), 0.05, key=f"pw_{t}")
        user_w.append(wv)

    total_w = sum(user_w)
    if total_w == 0:
        st.warning("All weights are zero.")
        st.stop()

    norm_w = [w / total_w for w in user_w]
    if not np.isclose(total_w, 1.0, atol=0.01):
        st.caption(f"Weights normalised from {total_w*100:.1f}% → 100%")

    pm = portfolio_metrics(prices, norm_w, rf_rate)

    st.markdown("#### Portfolio Metrics")
    mc2 = st.columns(6)
    cards = [
        ("Total Return",  fmt_pct(pm["Total Return"]),  pm["Total Return"]  > 0),
        ("CAGR",          fmt_pct(pm["CAGR"]),          pm["CAGR"]          > 0),
        ("Volatility",    fmt_pct(pm["Volatility"]),    None),
        ("Sharpe",        fmt_rat(pm["Sharpe Ratio"]),  pm["Sharpe Ratio"]  > 1),
        ("Sortino",       fmt_rat(pm["Sortino Ratio"]), pm["Sortino Ratio"] > 1),
        ("Max Drawdown",  fmt_pct(pm["Max Drawdown"]),  False),
    ]
    for col, (lbl, val, pos) in zip(mc2, cards):
        metric_card(col, lbl, val, pos)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Portfolio vs Individual Assets")
    combined = growth_of_investment(prices, initial=init_inv).copy()
    combined["Portfolio (Blended)"] = portfolio_growth(prices, norm_w, init_inv).values
    st.plotly_chart(equity_curve(combined, currency=cur, title="Portfolio vs Individuals"), use_container_width=True)

    st.markdown("#### Correlation")
    st.plotly_chart(correlation_heatmap(correlation_matrix(prices)), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MONTE CARLO
# ══════════════════════════════════════════════════════════════════════════════
with tab_mc:
    st.subheader("Monte Carlo Portfolio Simulation")
    st.caption("Generates thousands of random weight combinations to map the risk/return universe.")

    with st.spinner(f"Running {n_portfolios:,} simulations…"):
        sim_df  = run_simulation(prices, n_portfolios=n_portfolios,
                                 risk_free_rate=rf_rate, random_seed=int(mc_seed))
        ms      = get_max_sharpe(sim_df, tickers)
        mv      = get_min_variance(sim_df, tickers)

        with st.spinner("Computing Efficient Frontier curve…"):
            try:
                frontier_df = efficient_frontier_curve(prices, n_points=60, risk_free_rate=rf_rate)
            except Exception:
                frontier_df = None

    st.plotly_chart(
        monte_carlo_scatter(sim_df, ms, mv, frontier_df),
        use_container_width=True,
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### ⭐ Max Sharpe Portfolio")
        st.caption("Best risk-adjusted return")
        ms_rows = st.columns(3)
        metric_card(ms_rows[0], "Return",     fmt_pct(ms["Return"]),     True)
        metric_card(ms_rows[1], "Volatility", fmt_pct(ms["Volatility"]), None)
        metric_card(ms_rows[2], "Sharpe",     fmt_rat(ms["Sharpe"]),     ms["Sharpe"] > 1)
        st.plotly_chart(weights_bar(ms["Weights"], "Max Sharpe Weights"), use_container_width=True)

    with c2:
        st.markdown("#### 💎 Min Variance Portfolio")
        st.caption("Lowest possible risk")
        mv_rows = st.columns(3)
        metric_card(mv_rows[0], "Return",     fmt_pct(mv["Return"]),     True)
        metric_card(mv_rows[1], "Volatility", fmt_pct(mv["Volatility"]), None)
        metric_card(mv_rows[2], "Sharpe",     fmt_rat(mv["Sharpe"]),     mv["Sharpe"] > 1)
        st.plotly_chart(weights_bar(mv["Weights"], "Min Variance Weights"), use_container_width=True)

    with st.expander("📋 Full Simulation Statistics"):
        st.markdown(f"""
        | Stat | Value |
        |---|---|
        | Portfolios simulated | {len(sim_df):,} |
        | Return range | {fmt_pct(sim_df['Return'].min())} → {fmt_pct(sim_df['Return'].max())} |
        | Volatility range | {fmt_pct(sim_df['Volatility'].min())} → {fmt_pct(sim_df['Volatility'].max())} |
        | Sharpe range | {fmt_rat(sim_df['Sharpe'].min())} → {fmt_rat(sim_df['Sharpe'].max())} |
        """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════
with tab_opt:
    st.subheader("Portfolio Optimizer")
    st.caption("Uses SciPy constrained optimisation to find the mathematically exact optimal weights.")

    if n < 2:
        st.warning("Need at least 2 assets for optimisation.")
    else:
        with st.spinner("Optimising portfolios…"):
            try:
                opt_ms = optimize(prices, "max_sharpe",   rf_rate)
                opt_mv = optimize(prices, "min_variance", rf_rate)
            except Exception as e:
                st.error(f"Optimisation failed: {e}")
                st.stop()

        c1, c2 = st.columns(2)

        for col, result, label, icon in [
            (c1, opt_ms, "Max Sharpe",    "⭐"),
            (c2, opt_mv, "Min Variance",  "💎"),
        ]:
            with col:
                status = "✅ Converged" if result["success"] else "⚠️ Did not converge"
                st.markdown(f"#### {icon} {label}  —  {status}")
                rc = st.columns(3)
                metric_card(rc[0], "Return",     fmt_pct(result["Return"]),     result["Return"] > 0)
                metric_card(rc[1], "Volatility", fmt_pct(result["Volatility"]), None)
                metric_card(rc[2], "Sharpe",     fmt_rat(result["Sharpe"]),     result["Sharpe"] > 1)
                st.markdown("<br>", unsafe_allow_html=True)
                st.plotly_chart(
                    weights_bar(result["weights"], f"{label} — Optimal Weights"),
                    use_container_width=True,
                )

                # Weight table
                wdf = pd.DataFrame({
                    "Asset" : list(result["weights"].keys()),
                    "Weight": [fmt_pct(v) for v in result["weights"].values()],
                })
                st.dataframe(wdf, hide_index=True, use_container_width=True)

        # Side-by-side comparison
        st.markdown("#### Optimizer vs Equal-Weight Comparison")
        eq_pm = portfolio_metrics(prices, equal_w, rf_rate)
        compare = pd.DataFrame({
            "Equal Weight" : {
                "Return": fmt_pct(eq_pm["Total Return"]), "CAGR": fmt_pct(eq_pm["CAGR"]),
                "Vol": fmt_pct(eq_pm["Volatility"]), "Sharpe": fmt_rat(eq_pm["Sharpe Ratio"]),
                "Max DD": fmt_pct(eq_pm["Max Drawdown"]),
            },
            "Max Sharpe" : {
                "Return": fmt_pct(opt_ms["Return"]*3), "CAGR": fmt_pct(opt_ms["Return"]),
                "Vol": fmt_pct(opt_ms["Volatility"]), "Sharpe": fmt_rat(opt_ms["Sharpe"]),
                "Max DD": "—",
            },
            "Min Variance" : {
                "Return": "—", "CAGR": fmt_pct(opt_mv["Return"]),
                "Vol": fmt_pct(opt_mv["Volatility"]), "Sharpe": fmt_rat(opt_mv["Sharpe"]),
                "Max DD": "—",
            },
        }).T
        st.dataframe(compare, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — VALUE AT RISK
# ══════════════════════════════════════════════════════════════════════════════
with tab_var:
    st.subheader("Value at Risk (VaR) & CVaR")
    st.caption("How much could this portfolio lose, and with what probability?")

    var_report = full_var_report(
        prices,
        weights         = equal_w,
        portfolio_value = init_inv,
        risk_free_rate  = rf_rate,
    )

    rets_series = var_report.pop("returns_series")

    # ── Top metric cards ──────────────────────────────────────────────────
    st.markdown("#### Equal-Weight Portfolio — 1-Day VaR")
    vc = st.columns(6)
    var_cards = [
        ("Hist VaR 95%",  fmt_pct(var_report["Historical VaR 95%"]),  False),
        ("Hist VaR 99%",  fmt_pct(var_report["Historical VaR 99%"]),  False),
        ("Param VaR 95%", fmt_pct(var_report["Parametric VaR 95%"]), False),
        ("Param VaR 99%", fmt_pct(var_report["Parametric VaR 99%"]), False),
        ("CVaR 95%",      fmt_pct(var_report["CVaR 95%"]),           False),
        ("CVaR 99%",      fmt_pct(var_report["CVaR 99%"]),           False),
    ]
    for col, (lbl, val, pos) in zip(vc, var_cards):
        metric_card(col, lbl, val, pos)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Absolute loss cards ───────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        loss_95 = var_report["1-Day Loss at 95% (₹/$)"]
        st.metric(
            label=f"Max 1-Day Loss at 95% Confidence (out of {cur}{init_inv:,.0f})",
            value=f"{cur}{loss_95:,.0f}",
            delta=f"-{loss_95/init_inv*100:.2f}% of portfolio",
            delta_color="inverse",
        )
    with c2:
        loss_99 = var_report["1-Day Loss at 99% (₹/$)"]
        st.metric(
            label=f"Max 1-Day Loss at 99% Confidence (out of {cur}{init_inv:,.0f})",
            value=f"{cur}{loss_99:,.0f}",
            delta=f"-{loss_99/init_inv*100:.2f}% of portfolio",
            delta_color="inverse",
        )

    # ── Distribution chart ────────────────────────────────────────────────
    st.markdown("#### Return Distribution")
    st.plotly_chart(
        var_distribution(
            rets_series,
            var_95  = var_report["Historical VaR 95%"],
            var_99  = var_report["Historical VaR 99%"],
            cvar_95 = var_report["CVaR 95%"],
        ),
        use_container_width=True,
    )

    # ── Per-asset VaR table ───────────────────────────────────────────────
    st.markdown("#### Per-Asset VaR")
    from src.analytics.returns import daily_returns as _dr
    asset_rets = _dr(prices)
    var_rows = []
    for t in tickers:
        r = asset_rets[t].dropna()
        var_rows.append({
            "Ticker"         : t,
            "Hist VaR 95%"   : fmt_pct(historical_var(r, 0.95)),
            "Hist VaR 99%"   : fmt_pct(historical_var(r, 0.99)),
            "CVaR 95%"       : fmt_pct(compute_cvar(r, 0.95)),
            f"1-Day Loss 95% ({cur})" : fmt_abs(historical_var(r, 0.95) * init_inv, cur),
        })
    st.dataframe(pd.DataFrame(var_rows).set_index("Ticker"), use_container_width=True)

    with st.expander("📖 How to interpret VaR"):
        st.markdown("""
        **Historical VaR 95%** = On 95% of days, you won't lose more than this amount.
        The worst 5% of days exceed this threshold.

        **VaR 99%** = Even stricter — only 1% of days are worse.

        **CVaR (Expected Shortfall)** = *Given* that we're in the worst 5% of days,
        what is the *average* loss? Always worse than VaR — this captures tail severity.

        **Parametric VaR** assumes a normal distribution. It's faster but underestimates
        losses during crises (fat tails). Historical VaR makes no distribution assumption.
        """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — CAPM / BETA / ALPHA
# ══════════════════════════════════════════════════════════════════════════════
with tab_capm:
    st.subheader("CAPM — Capital Asset Pricing Model")
    st.caption(f"Benchmark: **{bench_ticker}**  |  Risk-free rate: **{rf_rate*100:.1f}%**  |  Expected market return: **{mkt_ret*100:.1f}%**")

    if bench_series is None:
        st.warning(f"Could not load benchmark '{bench_ticker}'. Check the ticker in the sidebar.")
    elif n < 1:
        st.warning("Need at least 1 asset.")
    else:
        capm_df = full_capm_report(prices, bench_series, rf_rate, mkt_ret)

        # Add actual realised return for the SML chart
        from src.analytics.returns import cagr as _cagr
        capm_df["Actual Return"] = _cagr(prices)

        # ── CAPM table ────────────────────────────────────────────────────
        st.markdown("#### CAPM Metrics")
        disp_capm = capm_df.copy()
        for col in ["Beta", "R²", "p-value"]:
            disp_capm[col] = disp_capm[col].apply(fmt_rat)
        for col in ["Alpha (Ann.)", "Expected Return", "Actual Return"]:
            if col in disp_capm.columns:
                disp_capm[col] = disp_capm[col].apply(fmt_pct)
        st.dataframe(disp_capm, use_container_width=True)

        # ── Beta cards ────────────────────────────────────────────────────
        st.markdown("#### Beta Breakdown")
        beta_cols = st.columns(min(n, 4))
        for i, (ticker, row) in enumerate(capm_df.iterrows()):
            b = row["Beta"]
            interp = beta_interpretation(b)
            with beta_cols[i % len(beta_cols)]:
                st.markdown(f"**{ticker}**")
                st.markdown(f"β = `{b:.3f}`")
                st.caption(interp)

        # ── Charts ────────────────────────────────────────────────────────
        c1, c2 = st.columns([3, 2])
        with c1:
            st.plotly_chart(security_market_line(capm_df, rf_rate, mkt_ret), use_container_width=True)
        with c2:
            st.plotly_chart(beta_bar(capm_df), use_container_width=True)

        with st.expander("📖 CAPM Concepts"):
            st.markdown("""
            **Beta (β):**
            - β = 1.0 → moves exactly with the market
            - β > 1.0 → amplifies market moves (higher risk & reward)
            - β < 1.0 → defensive, dampens market swings
            - β < 0.0 → moves opposite (rare — gold, some bonds)

            **Alpha (α):**
            Return *above* what CAPM predicts. A positive alpha means the asset
            outperformed its risk-adjusted expectation. This is what fund managers chase.

            **R²:**
            Percentage of the asset's returns explained by market movements.
            R² = 0.85 means 85% of price moves are driven by the market.

            **Security Market Line (SML):**
            The yellow dashed line is what CAPM *predicts*. Assets above the line
            delivered positive alpha. Assets below underperformed their risk-adjusted expectation.
            """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — STRATEGY LAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_strat:
    st.subheader("Strategy Lab")
    st.caption("Compare Buy & Hold against SMA, EMA, RSI, and MACD strategies")

    strat_ticker = st.selectbox("Select Asset", tickers)
    asset_px     = prices[[strat_ticker]]

    # ── Strategy parameters ───────────────────────────────────────────────
    with st.expander("⚙️ Strategy Parameters", expanded=True):
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            st.markdown("**SMA Crossover**")
            sma_fast = st.slider("SMA Fast", 10, 100, 50, 5, key="sma_f")
            sma_slow = st.slider("SMA Slow", 50, 300, 200, 10, key="sma_s")
        with pc2:
            st.markdown("**EMA Crossover**")
            ema_fast = st.slider("EMA Fast", 5, 50, 12, 1, key="ema_f")
            ema_slow = st.slider("EMA Slow", 10, 100, 26, 1, key="ema_s")
        with pc3:
            st.markdown("**RSI**")
            rsi_period   = st.slider("RSI Period", 5, 30, 14, 1, key="rsi_p")
            rsi_oversold = st.slider("Oversold",  10, 40, 30, 5, key="rsi_os")
            rsi_overbought = st.slider("Overbought", 60, 90, 70, 5, key="rsi_ob")

    if sma_fast >= sma_slow:
        st.warning("SMA Fast must be < SMA Slow.")
        st.stop()

    # ── Run all strategies ────────────────────────────────────────────────
    with st.spinner("Running backtests…"):
        res_bh   = buy_hold.run(asset_px,     initial=init_inv)
        res_sma  = sma_crossover.run(asset_px, sma_fast, sma_slow,  initial=init_inv)
        res_ema  = ema_strategy.run(asset_px,  ema_fast, ema_slow,  initial=init_inv)
        res_rsi  = rsi_strategy.run(asset_px,  rsi_period, rsi_oversold, rsi_overbought, initial=init_inv)
        res_macd = macd_strategy.run(asset_px, initial=init_inv)

    all_results = {
        "Buy & Hold"                              : res_bh,
        f"SMA {sma_fast}/{sma_slow}"              : res_sma,
        f"EMA {ema_fast}/{ema_slow}"              : res_ema,
        f"RSI({rsi_period}) {rsi_oversold}/{rsi_overbought}": res_rsi,
        "MACD (12,26,9)"                          : res_macd,
    }

    # ── Equity curve comparison ───────────────────────────────────────────
    curves = {name: r["equity_curve"] for name, r in all_results.items()}
    st.plotly_chart(all_strategies_equity(curves, currency=cur), use_container_width=True)

    # ── Metrics heatmap ───────────────────────────────────────────────────
    st.markdown("#### Strategy Comparison Matrix")
    metric_keys = ["Total Return", "CAGR", "Volatility", "Sharpe Ratio", "Max Drawdown", "Num Trades"]
    table_rows  = {}
    for name, r in all_results.items():
        m = r["metrics"]
        table_rows[name] = {k: m.get(k, 0) for k in metric_keys}
    metrics_tbl = pd.DataFrame(table_rows).T
    st.plotly_chart(strategy_heatmap(metrics_tbl), use_container_width=True)

    # ── Formatted table ───────────────────────────────────────────────────
    disp_tbl = metrics_tbl.copy()
    for col in ["Total Return", "CAGR", "Volatility", "Max Drawdown"]:
        disp_tbl[col] = disp_tbl[col].apply(lambda x: fmt_pct(float(x)))
    disp_tbl["Sharpe Ratio"] = disp_tbl["Sharpe Ratio"].apply(lambda x: fmt_rat(float(x)))
    disp_tbl["Num Trades"]   = disp_tbl["Num Trades"].apply(lambda x: str(int(float(x))))
    st.dataframe(disp_tbl, use_container_width=True)

    # ── Individual indicator charts ───────────────────────────────────────
    st.markdown("---")
    ic1, ic2 = st.columns(2)

    with ic1:
        st.markdown("#### MACD Chart")
        st.plotly_chart(
            macd_chart(asset_px.iloc[:, 0], res_macd["macd_df"], strat_ticker),
            use_container_width=True,
        )

    with ic2:
        st.markdown("#### RSI Chart")
        st.plotly_chart(
            rsi_chart(asset_px.iloc[:, 0], res_rsi["rsi_series"],
                      rsi_oversold, rsi_overbought, strat_ticker),
            use_container_width=True,
        )

    st.markdown("#### SMA Signal Chart")
    st.plotly_chart(sma_signals_chart(res_sma["signals"], strat_ticker), use_container_width=True)

    # ── Best strategy callout ──────────────────────────────────────────────
    best_name = max(all_results, key=lambda k: all_results[k]["metrics"]["Sharpe Ratio"])
    best_sh   = all_results[best_name]["metrics"]["Sharpe Ratio"]
    bh_sh     = res_bh["metrics"]["Sharpe Ratio"]

    if best_name == "Buy & Hold":
        st.info(f"🏆 **Buy & Hold** is the best strategy for **{strat_ticker}** on a risk-adjusted basis (Sharpe: {bh_sh:.3f}). Active strategies often struggle to beat passive investing — this is a key insight in quant finance.")
    else:
        st.success(f"🏆 **{best_name}** outperforms Buy & Hold for **{strat_ticker}** (Sharpe: {best_sh:.3f} vs {bh_sh:.3f})")
