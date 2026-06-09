"""
FAMA-FRENCH CHARTS
==================
Visualisations for the FF3 factor model tab.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS = ["#2196F3","#FF5722","#4CAF50","#9C27B0","#FFC107","#00BCD4","#F44336","#607D8B"]
THEME  = {"bg":"#0e1117","paper":"#1a1d27","grid":"#2d3142","text":"#e0e0e0","sub":"#9e9e9e"}


def _base(title="", height=460):
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
        hovermode    = "x unified",
    )


def factor_returns_chart(ff_factors: pd.DataFrame) -> go.Figure:
    """
    Cumulative returns of each FF factor over time.
    Shows whether each risk premium has been positive in this period.
    """
    fig = go.Figure()

    factor_colors = {"Mkt-RF": "#2196F3", "SMB": "#4CAF50", "HML": "#FF5722"}

    for factor, color in factor_colors.items():
        if factor not in ff_factors.columns:
            continue
        cum = (1 + ff_factors[factor]).cumprod()
        fig.add_trace(go.Scatter(
            x    = cum.index,
            y    = (cum - 1) * 100,
            name = factor,
            line = dict(color=color, width=2),
            hovertemplate=f"<b>{factor}</b>: %{{y:.2f}}%<extra></extra>",
        ))

    layout = _base("Fama-French Factor Cumulative Returns", height=380)
    layout["yaxis"]["ticksuffix"] = "%"
    layout["yaxis"]["title"]      = "Cumulative Return (%)"
    fig.update_layout(**layout)
    return fig


def ff3_alpha_chart(ff3_df: pd.DataFrame) -> go.Figure:
    """
    Bar chart of FF3 alpha for each asset.
    Green = positive alpha (outperformed), Red = negative.
    """
    valid = ff3_df.dropna(subset=["Alpha (Ann.)"])
    alphas = valid["Alpha (Ann.)"].sort_values(ascending=False)

    colors = ["#4CAF50" if a >= 0 else "#F44336" for a in alphas]
    sig    = valid.loc[alphas.index, "p-val Alpha"] < 0.05

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x            = alphas.index.tolist(),
        y            = (alphas * 100).tolist(),
        marker_color = colors,
        marker_line  = dict(
            color = ["white" if s else "rgba(0,0,0,0)" for s in sig],
            width = 2,
        ),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "FF3 Alpha: %{y:.2f}%/yr<br>"
            "<extra></extra>"
        ),
    ))

    fig.add_hline(y=0, line_color="#9e9e9e", line_width=1)

    layout = _base("FF3 Alpha — Excess Return vs 3-Factor Model", height=360)
    layout["yaxis"]["title"]    = "Annualised Alpha (%)"
    layout["yaxis"]["ticksuffix"] = "%"
    fig.update_layout(**layout)

    fig.add_annotation(
        text="White border = statistically significant (p < 0.05)",
        xref="paper", yref="paper", x=0.01, y=1.06,
        showarrow=False, font=dict(size=11, color=THEME["sub"]),
    )

    return fig


def factor_betas_grouped(ff3_df: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart comparing Market, SMB, and HML betas across all assets.
    Instantly shows which assets are aggressive/defensive/value/growth.
    """
    valid   = ff3_df.dropna(subset=["Beta Market","Beta SMB","Beta HML"])
    tickers = valid.index.tolist()

    fig = go.Figure()

    for factor, color, col in [
        ("Market β", "#2196F3", "Beta Market"),
        ("SMB β",    "#4CAF50", "Beta SMB"),
        ("HML β",    "#FF5722", "Beta HML"),
    ]:
        fig.add_trace(go.Bar(
            name=factor,
            x=tickers,
            y=valid[col].tolist(),
            marker_color=color,
            opacity=0.85,
            hovertemplate=f"<b>%{{x}}</b> {factor}: %{{y:.3f}}<extra></extra>",
        ))

    fig.add_hline(y=0, line_color="#9e9e9e", line_width=1)
    fig.add_hline(y=1, line_dash="dot", line_color="#2196F3", line_width=1,
                  annotation_text="Market β = 1", annotation_position="right",
                  annotation_font_color="#2196F3")

    layout = _base("Factor Loadings — Market, SMB, HML", height=400)
    layout["barmode"] = "group"
    layout["yaxis"]["title"] = "Factor Beta"
    fig.update_layout(**layout)
    return fig


def r_squared_chart(ff3_df: pd.DataFrame) -> go.Figure:
    """
    How much of each asset's return variance is explained by the 3 factors?
    High R² = mostly a beta play. Low R² = more idiosyncratic.
    """
    valid = ff3_df.dropna(subset=["R²"]).sort_values("R²", ascending=True)

    fig = go.Figure(go.Bar(
        x            = (valid["R²"] * 100).tolist(),
        y            = valid.index.tolist(),
        orientation  = "h",
        marker_color = [COLORS[i % len(COLORS)] for i in range(len(valid))],
        text         = [f"{v:.1f}%" for v in valid["R²"] * 100],
        textposition = "outside",
        hovertemplate="<b>%{y}</b>: R² = %{x:.1f}%<extra></extra>",
    ))

    layout = _base("R² — How Much is Explained by the 3 Factors?", height=max(280, 55*len(valid)+100))
    layout["xaxis"]["title"] = "R² (%)"
    layout["xaxis"]["range"] = [0, 110]
    layout["yaxis"]["autorange"] = "reversed"
    fig.update_layout(**layout)
    return fig
