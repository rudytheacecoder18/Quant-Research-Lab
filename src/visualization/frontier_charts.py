"""
FRONTIER CHARTS
===============
Advanced visualisations for V2 features:
  - Monte Carlo scatter (the iconic "bullet" cloud)
  - Efficient Frontier curve with highlighted portfolios
  - VaR return distribution with tail shading
  - CAPM Security Market Line
  - Strategy comparison table heat-map
  - MACD / RSI indicator sub-plots
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Shared palette / theme ────────────────────────────────────────────────────
COLORS = [
    "#2196F3", "#FF5722", "#4CAF50", "#9C27B0",
    "#FFC107", "#00BCD4", "#F44336", "#607D8B",
]

THEME = {
    "bg"    : "#0e1117",
    "paper" : "#1a1d27",
    "grid"  : "#2d3142",
    "text"  : "#e0e0e0",
    "sub"   : "#9e9e9e",
}


def _base(title: str = "", height: int = 500) -> dict:
    return dict(
        title        = dict(text=title, font=dict(size=16, color=THEME["text"])),
        height       = height,
        plot_bgcolor = THEME["paper"],
        paper_bgcolor= THEME["bg"],
        font         = dict(color=THEME["text"], family="Inter, sans-serif"),
        xaxis        = dict(gridcolor=THEME["grid"], showgrid=True, zeroline=False),
        yaxis        = dict(gridcolor=THEME["grid"], showgrid=True, zeroline=False),
        legend       = dict(bgcolor=THEME["paper"], bordercolor=THEME["grid"]),
        margin       = dict(l=50, r=20, t=55, b=50),
        hovermode    = "closest",
    )


# ─────────────────────────────────────────────────────────────────────────────
# MONTE CARLO / EFFICIENT FRONTIER
# ─────────────────────────────────────────────────────────────────────────────

def monte_carlo_scatter(
    sim_df        : pd.DataFrame,
    max_sharpe    : dict,
    min_variance  : dict,
    frontier_df   : pd.DataFrame = None,
) -> go.Figure:
    """
    The iconic Markowitz bullet — thousands of random portfolios as a scatter.

    Colour-coded by Sharpe Ratio so the efficient zone stands out.
    Highlights Max Sharpe (star) and Min Variance (diamond).
    Optional: overlay the exact Efficient Frontier curve.
    """
    fig = go.Figure()

    # ── Cloud of random portfolios (colour = Sharpe) ──────────────────────
    fig.add_trace(go.Scatter(
        x    = sim_df["Volatility"] * 100,
        y    = sim_df["Return"]     * 100,
        mode = "markers",
        name = "Random Portfolios",
        marker = dict(
            color     = sim_df["Sharpe"],
            colorscale= "Viridis",
            size      = 3,
            opacity   = 0.6,
            colorbar  = dict(
                title      = "Sharpe",
                tickformat = ".2f",
                len        = 0.7,
                x          = 1.02,
            ),
        ),
        hovertemplate=(
            "Volatility: %{x:.2f}%<br>"
            "Return: %{y:.2f}%<br>"
            "Sharpe: %{marker.color:.3f}<extra></extra>"
        ),
    ))

    # ── Efficient Frontier curve (if provided) ────────────────────────────
    if frontier_df is not None and len(frontier_df) > 0:
        fig.add_trace(go.Scatter(
            x    = frontier_df["Volatility"] * 100,
            y    = frontier_df["Return"]     * 100,
            mode = "lines",
            name = "Efficient Frontier",
            line = dict(color="#FFD700", width=2.5),
            hovertemplate="Vol: %{x:.2f}%  Ret: %{y:.2f}%<extra>Frontier</extra>",
        ))

    # ── Max Sharpe ────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x    = [max_sharpe["Volatility"] * 100],
        y    = [max_sharpe["Return"]     * 100],
        mode = "markers",
        name = f"Max Sharpe  ({max_sharpe['Sharpe']:.3f})",
        marker = dict(symbol="star", size=18, color="#FFD700",
                      line=dict(color="white", width=1)),
        hovertemplate=(
            "<b>Max Sharpe Portfolio</b><br>"
            "Volatility: %{x:.2f}%<br>"
            "Return: %{y:.2f}%<br>"
            f"Sharpe: {max_sharpe['Sharpe']:.3f}<extra></extra>"
        ),
    ))

    # ── Min Variance ──────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x    = [min_variance["Volatility"] * 100],
        y    = [min_variance["Return"]     * 100],
        mode = "markers",
        name = f"Min Variance  ({min_variance['Volatility']*100:.1f}% vol)",
        marker = dict(symbol="diamond", size=14, color="#00E5FF",
                      line=dict(color="white", width=1)),
        hovertemplate=(
            "<b>Min Variance Portfolio</b><br>"
            "Volatility: %{x:.2f}%<br>"
            "Return: %{y:.2f}%<extra></extra>"
        ),
    ))

    layout = _base("Monte Carlo Simulation — Efficient Frontier", height=520)
    layout["xaxis"]["title"] = "Annualised Volatility (%)"
    layout["yaxis"]["title"] = "Annualised Return (%)"
    fig.update_layout(**layout)

    return fig


def weights_bar(weights: dict, title: str = "Portfolio Weights") -> go.Figure:
    """Horizontal bar chart of optimal portfolio weights."""
    tickers = list(weights.keys())
    vals    = [w * 100 for w in weights.values()]
    colors  = [COLORS[i % len(COLORS)] for i in range(len(tickers))]

    fig = go.Figure(go.Bar(
        x             = vals,
        y             = tickers,
        orientation   = "h",
        marker_color  = colors,
        text          = [f"{v:.1f}%" for v in vals],
        textposition  = "outside",
        hovertemplate = "<b>%{y}</b>: %{x:.2f}%<extra></extra>",
    ))

    layout = _base(title, height=max(280, 50 * len(tickers) + 80))
    layout["xaxis"]["title"]   = "Weight (%)"
    layout["xaxis"]["range"]   = [0, max(vals) * 1.25]
    layout["yaxis"]["autorange"] = "reversed"
    fig.update_layout(**layout)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# VALUE AT RISK
# ─────────────────────────────────────────────────────────────────────────────

def var_distribution(
    returns : pd.Series,
    var_95  : float,
    var_99  : float,
    cvar_95 : float,
) -> go.Figure:
    """
    Return distribution histogram with VaR / CVaR thresholds shaded.

    Shows:
      - Full distribution of daily returns
      - Red zone (beyond 99% VaR) — the extreme tail
      - Orange zone (between 95% and 99%) — the severe tail
      - Vertical lines for each threshold
    """
    rets_pct = returns * 100

    fig = go.Figure()

    # ── Histogram ─────────────────────────────────────────────────────────
    fig.add_trace(go.Histogram(
        x         = rets_pct,
        nbinsx    = 80,
        name      = "Daily Returns",
        marker    = dict(color="#2196F3", opacity=0.7,
                         line=dict(color=THEME["paper"], width=0.3)),
        hovertemplate = "Return: %{x:.2f}%<br>Count: %{y}<extra></extra>",
    ))

    # ── VaR / CVaR vertical lines ─────────────────────────────────────────
    for val, label, color in [
        (var_95  * 100, "95% VaR",  "#FFA726"),
        (var_99  * 100, "99% VaR",  "#EF5350"),
        (cvar_95 * 100, "95% CVaR", "#AB47BC"),
    ]:
        fig.add_vline(
            x           = val,
            line_dash   = "dash",
            line_color  = color,
            line_width  = 2,
            annotation  = dict(
                text      = f"<b>{label}<br>{val:.2f}%</b>",
                font      = dict(color=color, size=11),
                bgcolor   = THEME["paper"],
                borderpad = 4,
            ),
            annotation_position = "top",
        )

    layout = _base("Daily Return Distribution & VaR Thresholds", height=420)
    layout["xaxis"]["title"] = "Daily Return (%)"
    layout["yaxis"]["title"] = "Frequency"
    layout["bargap"]         = 0.05
    fig.update_layout(**layout)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CAPM
# ─────────────────────────────────────────────────────────────────────────────

def security_market_line(
    capm_df       : pd.DataFrame,
    risk_free_rate: float = 0.06,
    market_return : float = 0.12,
) -> go.Figure:
    """
    Security Market Line (SML) — the graphical representation of CAPM.

    The SML shows what return CAPM predicts for each beta.
    Assets ABOVE the line have positive alpha (outperformed).
    Assets BELOW the line have negative alpha (underperformed).
    """
    fig = go.Figure()

    # ── SML line ──────────────────────────────────────────────────────────
    beta_range  = np.linspace(-0.2, capm_df["Beta"].max() * 1.3, 100)
    sml_returns = risk_free_rate + beta_range * (market_return - risk_free_rate)

    fig.add_trace(go.Scatter(
        x    = beta_range,
        y    = sml_returns * 100,
        mode = "lines",
        name = "Security Market Line (CAPM)",
        line = dict(color="#FFD700", width=2, dash="dash"),
        hovertemplate = "Beta: %{x:.2f}<br>Expected: %{y:.2f}%<extra>SML</extra>",
    ))

    # ── Asset scatter ─────────────────────────────────────────────────────
    for i, (ticker, row) in enumerate(capm_df.iterrows()):
        actual_ret = row.get("Actual Return", row.get("Expected Return", 0)) * 100
        alpha_val  = row["Alpha (Ann.)"]
        color      = "#4CAF50" if alpha_val >= 0 else "#F44336"
        symbol     = "circle" if alpha_val >= 0 else "circle-open"

        fig.add_trace(go.Scatter(
            x    = [row["Beta"]],
            y    = [actual_ret],
            mode = "markers+text",
            name = ticker,
            text = [ticker],
            textposition = "top center",
            marker = dict(size=12, color=color, symbol=symbol,
                          line=dict(color=color, width=2)),
            hovertemplate=(
                f"<b>{ticker}</b><br>"
                f"Beta: {row['Beta']:.3f}<br>"
                f"Alpha: {alpha_val*100:.2f}%<br>"
                f"R²: {row['R²']:.3f}<extra></extra>"
            ),
        ))

    # ── Market point ──────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[1.0], y=[market_return * 100],
        mode="markers+text", name="Market",
        text=["Market"], textposition="top right",
        marker=dict(size=14, color="white", symbol="diamond"),
        hovertemplate="<b>Market</b><br>Beta: 1.0<br>Return: %{y:.2f}%<extra></extra>",
    ))

    layout = _base("CAPM — Security Market Line", height=480)
    layout["xaxis"]["title"]    = "Beta (β)"
    layout["yaxis"]["title"]    = "Annualised Return (%)"
    layout["yaxis"]["ticksuffix"] = "%"
    fig.update_layout(**layout)

    return fig


def beta_bar(capm_df: pd.DataFrame) -> go.Figure:
    """Bar chart of beta values with a reference line at 1.0."""
    tickers = capm_df.index.tolist()
    betas   = capm_df["Beta"].tolist()
    colors  = ["#4CAF50" if b < 1 else "#FF5722" for b in betas]

    fig = go.Figure(go.Bar(
        x             = tickers,
        y             = betas,
        marker_color  = colors,
        hovertemplate = "<b>%{x}</b>  β = %{y:.3f}<extra></extra>",
    ))

    # Reference line at beta = 1 (= market)
    fig.add_hline(y=1.0, line_dash="dash", line_color="#FFD700", line_width=1.5,
                  annotation_text="Market β = 1", annotation_position="top right",
                  annotation_font_color="#FFD700")

    layout = _base("Beta — Market Sensitivity", height=350)
    layout["yaxis"]["title"] = "Beta (β)"
    fig.update_layout(**layout)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY LAB
# ─────────────────────────────────────────────────────────────────────────────

def all_strategies_equity(curves: dict, currency: str = "₹") -> go.Figure:
    """Overlay equity curves for ALL strategies on one chart."""
    fig = go.Figure()

    for i, (name, curve) in enumerate(curves.items()):
        width = 2.5 if name == "Buy & Hold" else 1.8
        dash  = "dot" if name == "Buy & Hold" else "solid"
        fig.add_trace(go.Scatter(
            x    = curve.index,
            y    = curve.values,
            name = name,
            line = dict(color=COLORS[i % len(COLORS)], width=width, dash=dash),
            hovertemplate = f"<b>{name}</b>: {currency}%{{y:,.0f}}<extra></extra>",
        ))

    layout = _base("All Strategies — Equity Curve Comparison", height=460)
    layout["yaxis"]["tickprefix"] = currency
    layout["yaxis"]["tickformat"] = ",.0f"
    layout["hovermode"]           = "x unified"
    fig.update_layout(**layout)

    return fig


def macd_chart(price_series: pd.Series, macd_df: pd.DataFrame, ticker: str = "") -> go.Figure:
    """
    Classic two-panel MACD chart:
      Top:    Price
      Bottom: MACD line, Signal line, Histogram (green/red bars)
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.04,
    )

    # ── Price ─────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=price_series.index, y=price_series,
        name="Price", line=dict(color="#90CAF9", width=1.5),
    ), row=1, col=1)

    # ── MACD line ─────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=macd_df.index, y=macd_df["MACD"],
        name="MACD", line=dict(color="#2196F3", width=1.5),
    ), row=2, col=1)

    # ── Signal line ───────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=macd_df.index, y=macd_df["Signal"],
        name="Signal", line=dict(color="#FF5722", width=1.5),
    ), row=2, col=1)

    # ── Histogram (green / red) ───────────────────────────────────────────
    hist_colors = ["#4CAF50" if v >= 0 else "#F44336" for v in macd_df["Histogram"]]
    fig.add_trace(go.Bar(
        x=macd_df.index, y=macd_df["Histogram"],
        name="Histogram", marker_color=hist_colors, opacity=0.7,
    ), row=2, col=1)

    # ── Layout ────────────────────────────────────────────────────────────
    fig.update_layout(
        title        = dict(text=f"{ticker} — MACD", font=dict(size=16, color=THEME["text"])),
        height       = 480,
        plot_bgcolor = THEME["paper"],
        paper_bgcolor= THEME["bg"],
        font         = dict(color=THEME["text"]),
        legend       = dict(bgcolor=THEME["paper"]),
        margin       = dict(l=50, r=20, t=55, b=40),
        hovermode    = "x unified",
        bargap       = 0,
    )
    for row in [1, 2]:
        fig.update_xaxes(gridcolor=THEME["grid"], row=row, col=1)
        fig.update_yaxes(gridcolor=THEME["grid"], row=row, col=1)

    return fig


def rsi_chart(price_series: pd.Series, rsi_series: pd.Series,
              oversold: float = 30, overbought: float = 70,
              ticker: str = "") -> go.Figure:
    """
    Two-panel RSI chart:
      Top:    Price
      Bottom: RSI oscillator with overbought/oversold bands
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.04,
    )

    # ── Price ─────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=price_series.index, y=price_series,
        name="Price", line=dict(color="#90CAF9", width=1.5),
    ), row=1, col=1)

    # ── RSI ───────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=rsi_series.index, y=rsi_series,
        name="RSI", line=dict(color="#FF5722", width=1.5),
        fill="tozeroy", fillcolor="rgba(255,87,34,0.07)",
    ), row=2, col=1)

    # ── Overbought / Oversold bands ───────────────────────────────────────
    for level, color, label in [
        (overbought, "#F44336", f"Overbought ({overbought})"),
        (oversold,   "#4CAF50", f"Oversold ({oversold})"),
        (50,         "#9e9e9e", "Neutral (50)"),
    ]:
        fig.add_hline(
            y=level, line_dash="dash", line_color=color, line_width=1.2,
            annotation_text=label, annotation_position="right",
            annotation_font_color=color, row=2, col=1,
        )

    fig.update_layout(
        title         = dict(text=f"{ticker} — RSI ({14})", font=dict(size=16, color=THEME["text"])),
        height        = 480,
        plot_bgcolor  = THEME["paper"],
        paper_bgcolor = THEME["bg"],
        font          = dict(color=THEME["text"]),
        legend        = dict(bgcolor=THEME["paper"]),
        margin        = dict(l=50, r=20, t=55, b=40),
        hovermode     = "x unified",
    )
    for row in [1, 2]:
        fig.update_xaxes(gridcolor=THEME["grid"], row=row, col=1)
        fig.update_yaxes(gridcolor=THEME["grid"], row=row, col=1)

    return fig


def strategy_heatmap(metrics_table: pd.DataFrame) -> go.Figure:
    """
    Colour-coded metrics table across all strategies.
    Green = better, Red = worse (column-wise normalised).
    """
    # Columns where LOWER is better
    lower_is_better = {"Volatility", "Max Drawdown", "Num Trades"}

    z_matrix = []
    for col in metrics_table.columns:
        col_vals = pd.to_numeric(metrics_table[col], errors="coerce").fillna(0)
        mn, mx   = col_vals.min(), col_vals.max()
        if mx == mn:
            normed = [0.5] * len(col_vals)
        else:
            normed = ((col_vals - mn) / (mx - mn)).tolist()
        if col in lower_is_better:
            normed = [1 - v for v in normed]
        z_matrix.append(normed)

    z_array = np.array(z_matrix).T   # shape: (strategies, metrics)

    text_array = []
    for _, row in metrics_table.iterrows():
        row_text = []
        for col in metrics_table.columns:
            val = row[col]
            if col in {"Total Return", "CAGR", "Volatility", "Max Drawdown"}:
                row_text.append(f"{float(val)*100:.2f}%")
            elif col == "Num Trades":
                row_text.append(str(int(val)))
            else:
                row_text.append(f"{float(val):.3f}")
        text_array.append(row_text)

    fig = go.Figure(go.Heatmap(
        z            = z_array,
        x            = list(metrics_table.columns),
        y            = list(metrics_table.index),
        text         = text_array,
        texttemplate = "%{text}",
        textfont     = dict(size=11, color="white"),
        colorscale   = "RdYlGn",
        showscale    = False,
        hovertemplate= "<b>%{y}</b> — %{x}: %{text}<extra></extra>",
    ))

    fig.update_layout(
        title         = dict(text="Strategy Comparison Matrix", font=dict(size=16, color=THEME["text"])),
        height        = max(300, 60 * len(metrics_table) + 120),
        plot_bgcolor  = THEME["paper"],
        paper_bgcolor = THEME["bg"],
        font          = dict(color=THEME["text"]),
        margin        = dict(l=10, r=10, t=55, b=40),
        xaxis         = dict(side="top"),
    )

    return fig
