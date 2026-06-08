"""
CHARTS MODULE
=============
All visualizations for the dashboard using Plotly.

Plotly creates interactive charts — you can hover, zoom, and pan.
This is far more professional than matplotlib for a web app.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express       as px
from plotly.subplots        import make_subplots


# ── Colour palette ────────────────────────────────────────────────────────────
COLORS = [
    "#2196F3",  # Blue
    "#FF5722",  # Orange
    "#4CAF50",  # Green
    "#9C27B0",  # Purple
    "#FFC107",  # Amber
    "#00BCD4",  # Cyan
    "#F44336",  # Red
    "#607D8B",  # Blue Grey
]

THEME = {
    "bg"       : "#0e1117",
    "paper"    : "#1a1d27",
    "grid"     : "#2d3142",
    "text"     : "#e0e0e0",
    "subtext"  : "#9e9e9e",
}


def _base_layout(title: str = "", height: int = 450) -> dict:
    """Standard dark-theme layout applied to all charts."""
    return dict(
        title       = dict(text=title, font=dict(size=16, color=THEME["text"])),
        height      = height,
        plot_bgcolor= THEME["paper"],
        paper_bgcolor= THEME["bg"],
        font        = dict(color=THEME["text"], family="Inter, sans-serif"),
        xaxis       = dict(gridcolor=THEME["grid"], showgrid=True, zeroline=False),
        yaxis       = dict(gridcolor=THEME["grid"], showgrid=True, zeroline=False),
        legend      = dict(bgcolor=THEME["paper"], bordercolor=THEME["grid"]),
        margin      = dict(l=40, r=20, t=50, b=40),
        hovermode   = "x unified",
    )


def equity_curve(growth_df: pd.DataFrame, currency: str = "₹", title: str = "") -> go.Figure:
    """
    Plot the growth of an initial investment over time.

    Shows: If you invested ₹1,00,000 on day 1, what's it worth each day?
    """
    fig = go.Figure()

    for i, col in enumerate(growth_df.columns):
        fig.add_trace(go.Scatter(
            x        = growth_df.index,
            y        = growth_df[col],
            name     = col,
            line     = dict(color=COLORS[i % len(COLORS)], width=2),
            hovertemplate=f"<b>{col}</b>: {currency}%{{y:,.0f}}<extra></extra>",
        ))

    fig.update_layout(**_base_layout(title or f"Growth of {currency}1,00,000", height=420))
    fig.update_yaxes(tickprefix=currency, tickformat=",.0f")

    return fig


def drawdown_chart(drawdown_df: pd.DataFrame, title: str = "Drawdown") -> go.Figure:
    """
    Visualise losses from peak — shows when and how bad drawdowns were.
    All values are negative (or zero at the peak).
    """
    fig = go.Figure()

    for i, col in enumerate(drawdown_df.columns):
        fig.add_trace(go.Scatter(
            x        = drawdown_df.index,
            y        = drawdown_df[col] * 100,  # Convert to %
            name     = col,
            line     = dict(color=COLORS[i % len(COLORS)], width=1.5),
            fill     = "tozeroy",
            fillcolor= COLORS[i % len(COLORS)].replace(")", ", 0.15)").replace("rgb", "rgba"),
            hovertemplate="<b>" + col + "</b>: %{y:.1f}%<extra></extra>",
        ))

    fig.update_layout(**_base_layout(title, height=350))
    fig.update_yaxes(ticksuffix="%")

    return fig


def rolling_volatility_chart(rolling_vol: pd.DataFrame, title: str = "Rolling 30-Day Volatility") -> go.Figure:
    """
    Shows how risk has changed over time.
    Spikes = periods of high uncertainty (COVID, rate hikes, earnings, etc.)
    """
    fig = go.Figure()

    for i, col in enumerate(rolling_vol.columns):
        fig.add_trace(go.Scatter(
            x        = rolling_vol.index,
            y        = rolling_vol[col] * 100,
            name     = col,
            line     = dict(color=COLORS[i % len(COLORS)], width=1.5),
            hovertemplate="<b>" + col + "</b>: %{y:.1f}%<extra></extra>",
        ))

    fig.update_layout(**_base_layout(title, height=350))
    fig.update_yaxes(ticksuffix="%")

    return fig


def correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """
    Heatmap of asset correlations.

    Colour scale:
      Red   = High positive correlation (move together)
      White = Uncorrelated
      Blue  = Negative correlation (move opposite)

    Look for blue/white cells — those are your diversifiers!
    """
    fig = go.Figure(data=go.Heatmap(
        z           = corr_matrix.values,
        x           = corr_matrix.columns.tolist(),
        y           = corr_matrix.index.tolist(),
        colorscale  = "RdBu_r",
        zmid        = 0,
        zmin        = -1,
        zmax        = 1,
        text        = corr_matrix.round(2).values,
        texttemplate="%{text}",
        textfont    = dict(size=12, color="white"),
        hovertemplate="<b>%{x} vs %{y}</b>: %{z:.3f}<extra></extra>",
        colorbar    = dict(
            title     = "Correlation",
            tickformat= ".1f",
            len       = 0.8,
        ),
    ))

    fig.update_layout(**_base_layout("Correlation Heatmap", height=420))

    return fig


def strategy_comparison(results: dict, currency: str = "₹", title: str = "Strategy Comparison") -> go.Figure:
    """
    Compare equity curves of multiple strategies on the same chart.

    results: dict of {strategy_name: equity_curve_series}
    """
    fig = go.Figure()

    for i, (name, curve) in enumerate(results.items()):
        fig.add_trace(go.Scatter(
            x        = curve.index,
            y        = curve.values,
            name     = name,
            line     = dict(color=COLORS[i % len(COLORS)], width=2),
            hovertemplate=f"<b>{name}</b>: {currency}%{{y:,.0f}}<extra></extra>",
        ))

    fig.update_layout(**_base_layout(title, height=420))
    fig.update_yaxes(tickprefix=currency, tickformat=",.0f")

    return fig


def sma_signals_chart(signals_df: pd.DataFrame, ticker: str = "") -> go.Figure:
    """
    Plot price with SMA lines and buy/sell markers.
    Helps visualise when the strategy entered and exited.
    """
    fig = go.Figure()

    # Price
    fig.add_trace(go.Scatter(
        x=signals_df.index, y=signals_df["Price"],
        name="Price", line=dict(color="#78909C", width=1), opacity=0.8,
    ))

    # Fast SMA
    fig.add_trace(go.Scatter(
        x=signals_df.index, y=signals_df["Fast SMA"],
        name="Fast SMA", line=dict(color="#2196F3", width=1.5, dash="dot"),
    ))

    # Slow SMA
    fig.add_trace(go.Scatter(
        x=signals_df.index, y=signals_df["Slow SMA"],
        name="Slow SMA", line=dict(color="#FF5722", width=1.5, dash="dot"),
    ))

    # Buy/Sell signals
    buy_dates  = signals_df[signals_df["Signal"].diff() ==  1].index
    sell_dates = signals_df[signals_df["Signal"].diff() == -1].index

    fig.add_trace(go.Scatter(
        x=buy_dates, y=signals_df.loc[buy_dates, "Price"],
        mode="markers", name="Buy",
        marker=dict(symbol="triangle-up", size=10, color="#4CAF50"),
    ))
    fig.add_trace(go.Scatter(
        x=sell_dates, y=signals_df.loc[sell_dates, "Price"],
        mode="markers", name="Sell",
        marker=dict(symbol="triangle-down", size=10, color="#F44336"),
    ))

    fig.update_layout(**_base_layout(f"{ticker} SMA Crossover Signals", height=420))

    return fig


def metrics_bar_chart(metrics_df: pd.DataFrame, metric: str = "Sharpe Ratio") -> go.Figure:
    """Simple bar chart comparing a single metric across all assets."""
    values = metrics_df[metric].sort_values(ascending=False)
    colors = ["#4CAF50" if v > 0 else "#F44336" for v in values]

    fig = go.Figure(go.Bar(
        x             = values.index,
        y             = values.values,
        marker_color  = colors,
        hovertemplate = "<b>%{x}</b>: %{y:.3f}<extra></extra>",
    ))

    fig.update_layout(**_base_layout(f"{metric} Comparison", height=350))

    return fig
