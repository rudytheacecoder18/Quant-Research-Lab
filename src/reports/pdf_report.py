"""
PDF REPORT GENERATOR
=====================
Generates a professional multi-page research report using ReportLab.

What's in the report:
  Page 1  — Cover page (title, tickers, date range, generated timestamp)
  Page 2  — Executive Summary (top metrics for each asset as a table)
  Page 3  — Portfolio Analysis (weights, blended metrics)
  Page 4  — Risk Analysis (VaR, CVaR table)
  Page 5  — Strategy Comparison table
  Page 6+ — Per-asset detail pages (CAGR, Sharpe, Drawdown, CAPM)
  Final   — FF3 factor loadings table (if available)

ReportLab concepts used:
  - Canvas:     low-level drawing (lines, rectangles, text at exact coordinates)
  - SimpleDocTemplate: higher-level document with automatic page flow
  - Paragraph:  styled text that wraps automatically
  - Table:      data grid with styling
  - Spacer:     vertical whitespace between elements
  - PageBreak:  forces a new page
"""

import io
from datetime import datetime

from reportlab.lib               import colors
from reportlab.lib.pagesizes     import A4
from reportlab.lib.styles        import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units         import cm
from reportlab.lib.enums         import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus          import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether,
)
from reportlab.platypus.flowables import Flowable

import pandas as pd
import numpy as np


# ── Colour palette (matching the dark dashboard's accent colours) ─────────────
C_DARK      = colors.HexColor("#0e1117")
C_PANEL     = colors.HexColor("#1a1d27")
C_BORDER    = colors.HexColor("#2d3142")
C_BLUE      = colors.HexColor("#2196F3")
C_GREEN     = colors.HexColor("#4CAF50")
C_RED       = colors.HexColor("#F44336")
C_AMBER     = colors.HexColor("#FFC107")
C_TEXT      = colors.HexColor("#212121")
C_SUBTEXT   = colors.HexColor("#616161")
C_HEADER_BG = colors.HexColor("#1565C0")
C_ALT_ROW   = colors.HexColor("#F5F7FF")
WHITE       = colors.white
BLACK       = colors.black


# ── Styles ────────────────────────────────────────────────────────────────────

def _styles():
    base = getSampleStyleSheet()

    custom = {
        "cover_title": ParagraphStyle(
            "cover_title", fontSize=28, fontName="Helvetica-Bold",
            textColor=WHITE, alignment=TA_CENTER, spaceAfter=10,
        ),
        "cover_sub": ParagraphStyle(
            "cover_sub", fontSize=13, fontName="Helvetica",
            textColor=colors.HexColor("#90CAF9"), alignment=TA_CENTER, spaceAfter=6,
        ),
        "cover_meta": ParagraphStyle(
            "cover_meta", fontSize=10, fontName="Helvetica",
            textColor=colors.HexColor("#B0BEC5"), alignment=TA_CENTER, spaceAfter=4,
        ),
        "section_header": ParagraphStyle(
            "section_header", fontSize=14, fontName="Helvetica-Bold",
            textColor=C_BLUE, spaceBefore=14, spaceAfter=6,
            borderPad=4,
        ),
        "body": ParagraphStyle(
            "body", fontSize=9, fontName="Helvetica",
            textColor=C_TEXT, spaceAfter=4, leading=14,
        ),
        "caption": ParagraphStyle(
            "caption", fontSize=8, fontName="Helvetica-Oblique",
            textColor=C_SUBTEXT, spaceAfter=6, alignment=TA_LEFT,
        ),
        "disclaimer": ParagraphStyle(
            "disclaimer", fontSize=7, fontName="Helvetica-Oblique",
            textColor=C_SUBTEXT, spaceAfter=4, alignment=TA_CENTER,
        ),
        "toc_item": ParagraphStyle(
            "toc_item", fontSize=10, fontName="Helvetica",
            textColor=C_TEXT, spaceAfter=3, leftIndent=20,
        ),
    }
    return custom


# ── Helper: standard table style ──────────────────────────────────────────────

def _table_style(has_header=True):
    cmds = [
        ("FONTNAME",    (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1 if has_header else 0), (-1, -1), [WHITE, C_ALT_ROW]),
        ("GRID",        (0, 0), (-1, -1), 0.4, C_BORDER),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0,0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING",(0, 0), (-1, -1), 7),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
    ]
    if has_header:
        cmds += [
            ("BACKGROUND",  (0, 0), (-1, 0), C_HEADER_BG),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, 0), 9),
            ("TEXTCOLOR",   (0, 0), (-1, 0), WHITE),
            ("ALIGN",       (0, 0), (-1, 0), "CENTER"),
        ]
    return TableStyle(cmds)


def _colour_cell(val_str: str, positive_is_good: bool = True):
    """Return green/red/black depending on sign of a percentage string."""
    try:
        num = float(val_str.replace("%","").replace("−","-").strip())
        if num > 0:
            return colors.HexColor("#1B5E20") if positive_is_good else colors.HexColor("#B71C1C")
        if num < 0:
            return colors.HexColor("#B71C1C") if positive_is_good else colors.HexColor("#1B5E20")
    except Exception:
        pass
    return C_TEXT


# ── Cover page ────────────────────────────────────────────────────────────────

class _CoverBackground(Flowable):
    """Dark gradient background for the cover page."""
    def draw(self):
        w, h = A4
        self.canv.setFillColor(C_DARK)
        self.canv.rect(0, 0, w, h, fill=1, stroke=0)
        self.canv.setFillColor(C_HEADER_BG)
        self.canv.rect(0, h - 5*cm, w, 5*cm, fill=1, stroke=0)
        self.canv.setFillColor(C_PANEL)
        self.canv.rect(1.5*cm, h - 13*cm, w - 3*cm, 7.5*cm, fill=1, stroke=0)


# ── Footer callback ────────────────────────────────────────────────────────────

def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(C_SUBTEXT)
    w, h = A4
    canvas.drawString(1.5*cm, 1.2*cm,
        "Quant Research Lab — Automated Report  |  For research purposes only. Not investment advice.")
    canvas.drawRightString(w - 1.5*cm, 1.2*cm, f"Page {doc.page}")
    canvas.restoreState()


# ── Formatting helpers ────────────────────────────────────────────────────────

def _pct(v, decimals=2):
    try:
        return f"{float(v)*100:.{decimals}f}%"
    except Exception:
        return str(v)

def _rat(v, decimals=3):
    try:
        return f"{float(v):.{decimals}f}"
    except Exception:
        return str(v)


# ── Main report builder ───────────────────────────────────────────────────────

def generate_report(
    tickers        : list,
    prices         : pd.DataFrame,
    metrics_df     : pd.DataFrame,
    port_weights   : list,
    port_metrics   : dict,
    var_report     : dict,
    capm_df        : pd.DataFrame,
    strategy_table : pd.DataFrame,
    ff3_df         : pd.DataFrame   = None,
    years          : float          = 3.0,
    risk_free_rate : float          = 0.06,
    benchmark      : str            = "SPY",
    using_synthetic_ff: bool        = False,
) -> bytes:
    """
    Build a complete PDF research report and return it as bytes.

    Parameters
    ----------
    tickers        : List of ticker symbols analysed
    prices         : Price DataFrame
    metrics_df     : Output of full_risk_report()
    port_weights   : List of floats (equal or user-defined)
    port_metrics   : Output of portfolio_metrics()
    var_report     : Output of full_var_report() (returns_series key removed)
    capm_df        : Output of full_capm_report()
    strategy_table : DataFrame with strategy comparison metrics
    ff3_df         : Output of full_ff3_report() — optional
    years          : Time period analysed
    risk_free_rate : Used in analysis
    benchmark      : Benchmark ticker label
    using_synthetic_ff : Whether FF data is synthetic (shows a warning)

    Returns
    -------
    PDF file as bytes — ready for st.download_button()
    """

    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=1.5*cm, rightMargin=1.5*cm,
        topMargin=1.5*cm,  bottomMargin=2.2*cm,
    )
    S      = _styles()
    story  = []

    generated_at = datetime.now().strftime("%d %B %Y, %H:%M")
    date_from    = prices.index[0].strftime("%d %b %Y")
    date_to      = prices.index[-1].strftime("%d %b %Y")

    # ─────────────────────────────────────────────────────────────────────
    # COVER PAGE
    # ─────────────────────────────────────────────────────────────────────
    story.append(_CoverBackground())

    story.append(Spacer(1, 3.5*cm))
    story.append(Paragraph("QUANT RESEARCH LAB", S["cover_title"]))
    story.append(Paragraph("Equity Research Report", S["cover_sub"]))
    story.append(Spacer(1, 0.5*cm))

    ticker_str = "  ·  ".join(tickers)
    story.append(Paragraph(ticker_str, S["cover_sub"]))
    story.append(Spacer(1, 0.8*cm))
    story.append(Paragraph(f"Analysis Period: {date_from} — {date_to}  ({years:.2f} years)", S["cover_meta"]))
    story.append(Paragraph(f"Benchmark: {benchmark}  ·  Risk-Free Rate: {risk_free_rate*100:.1f}%", S["cover_meta"]))
    story.append(Paragraph(f"Generated: {generated_at}", S["cover_meta"]))
    story.append(Spacer(1, 1.5*cm))

    # Table of contents
    toc_data = [
        ["Section", "Contents"],
        ["1", "Executive Summary — Key metrics for all assets"],
        ["2", "Portfolio Analysis — Blended metrics & allocation"],
        ["3", "Risk Analysis — VaR, CVaR at 95% and 99%"],
        ["4", "CAPM Analysis — Beta, Alpha, R²"],
        ["5", "Strategy Comparison — Buy & Hold vs active strategies"],
    ]
    if ff3_df is not None:
        toc_data.append(["6", "Fama-French 3-Factor — Alpha, factor loadings"])

    toc = Table(toc_data, colWidths=[1.5*cm, 13*cm])
    toc.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,0), C_HEADER_BG),
        ("TEXTCOLOR",    (0,0), (-1,0), WHITE),
        ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 9),
        ("FONTNAME",     (0,1), (-1,-1), "Helvetica"),
        ("TEXTCOLOR",    (0,1), (-1,-1), colors.HexColor("#E0E0E0")),
        ("BACKGROUND",   (0,1), (-1,-1), C_PANEL),
        ("GRID",         (0,0), (-1,-1), 0.4, C_BORDER),
        ("TOPPADDING",   (0,0), (-1,-1), 6),
        ("BOTTOMPADDING",(0,0), (-1,-1), 6),
        ("LEFTPADDING",  (0,0), (-1,-1), 8),
        ("ALIGN",        (0,0), (0,-1), "CENTER"),
    ]))
    story.append(toc)
    story.append(PageBreak())


    # ─────────────────────────────────────────────────────────────────────
    # SECTION 1 — EXECUTIVE SUMMARY
    # ─────────────────────────────────────────────────────────────────────
    story.append(Paragraph("1. Executive Summary", S["section_header"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_BLUE, spaceAfter=8))
    story.append(Paragraph(
        f"The following table summarises risk-adjusted performance for {len(tickers)} "
        f"assets over {years:.2f} years ({date_from} to {date_to}). "
        f"All metrics use an annualised risk-free rate of {risk_free_rate*100:.1f}%.",
        S["body"],
    ))
    story.append(Spacer(1, 0.3*cm))

    # Build summary table
    cols = ["Total Return","CAGR","Volatility","Sharpe Ratio","Sortino Ratio","Max Drawdown","Calmar Ratio"]
    header = ["Ticker"] + cols
    rows   = [header]

    for ticker in metrics_df.index:
        row_vals = [ticker]
        for col in cols:
            raw = metrics_df.loc[ticker, col]
            if col in ["Total Return","CAGR","Volatility","Max Drawdown"]:
                row_vals.append(_pct(raw))
            else:
                row_vals.append(_rat(raw))
        rows.append(row_vals)

    col_widths = [2.5*cm] + [2.3*cm]*len(cols)
    tbl = Table(rows, colWidths=col_widths, repeatRows=1)
    style_cmds = _table_style().getCommands()

    # Colour CAGR and Sharpe cells
    for r_idx in range(1, len(rows)):
        cagr_val  = metrics_df.iloc[r_idx-1]["CAGR"]
        sharpe_val= metrics_df.iloc[r_idx-1]["Sharpe Ratio"]
        mdd_val   = metrics_df.iloc[r_idx-1]["Max Drawdown"]
        c_idx_cagr   = cols.index("CAGR") + 1
        c_idx_sharpe = cols.index("Sharpe Ratio") + 1
        c_idx_mdd    = cols.index("Max Drawdown") + 1

        cagr_color   = C_GREEN if cagr_val  > 0    else C_RED
        sharpe_color = C_GREEN if sharpe_val > 1    else (C_AMBER if sharpe_val > 0 else C_RED)
        mdd_color    = C_RED

        style_cmds += [
            ("TEXTCOLOR", (c_idx_cagr,   r_idx), (c_idx_cagr,   r_idx), cagr_color),
            ("TEXTCOLOR", (c_idx_sharpe, r_idx), (c_idx_sharpe, r_idx), sharpe_color),
            ("TEXTCOLOR", (c_idx_mdd,    r_idx), (c_idx_mdd,    r_idx), mdd_color),
            ("FONTNAME",  (c_idx_cagr,   r_idx), (c_idx_cagr,   r_idx), "Helvetica-Bold"),
            ("FONTNAME",  (c_idx_sharpe, r_idx), (c_idx_sharpe, r_idx), "Helvetica-Bold"),
        ]

    tbl.setStyle(TableStyle(style_cmds))
    story.append(tbl)
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "Sharpe > 1.0 = good · > 2.0 = excellent · Max Drawdown = worst peak-to-trough loss",
        S["caption"],
    ))
    story.append(PageBreak())


    # ─────────────────────────────────────────────────────────────────────
    # SECTION 2 — PORTFOLIO ANALYSIS
    # ─────────────────────────────────────────────────────────────────────
    story.append(Paragraph("2. Portfolio Analysis", S["section_header"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_BLUE, spaceAfter=8))

    # Allocation table
    story.append(Paragraph("2.1  Allocation", S["body"]))
    alloc_rows = [["Asset", "Weight (%)"]]
    for t, w in zip(tickers, port_weights):
        alloc_rows.append([t, f"{w*100:.1f}%"])
    alloc_tbl = Table(alloc_rows, colWidths=[5*cm, 4*cm])
    alloc_tbl.setStyle(_table_style())
    story.append(alloc_tbl)
    story.append(Spacer(1, 0.4*cm))

    # Blended metrics
    story.append(Paragraph("2.2  Blended Portfolio Metrics", S["body"]))
    pm_rows = [["Metric", "Value"]]
    pm_fmt  = {
        "Total Return" : _pct, "CAGR": _pct, "Volatility": _pct,
        "Sharpe Ratio" : _rat, "Sortino Ratio": _rat,
        "Max Drawdown" : _pct,
    }
    for metric, fmt_fn in pm_fmt.items():
        if metric in port_metrics:
            pm_rows.append([metric, fmt_fn(port_metrics[metric])])
    pm_tbl = Table(pm_rows, colWidths=[5*cm, 4*cm])
    pm_tbl.setStyle(_table_style())
    story.append(pm_tbl)
    story.append(PageBreak())


    # ─────────────────────────────────────────────────────────────────────
    # SECTION 3 — RISK ANALYSIS (VaR)
    # ─────────────────────────────────────────────────────────────────────
    story.append(Paragraph("3. Risk Analysis — Value at Risk", S["section_header"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_BLUE, spaceAfter=8))
    story.append(Paragraph(
        "VaR (Value at Risk) measures the maximum expected loss over a 1-day horizon at a given "
        "confidence level. CVaR (Conditional VaR / Expected Shortfall) is the average loss on "
        "the worst days beyond the VaR threshold — a more complete picture of tail risk.",
        S["body"],
    ))
    story.append(Spacer(1, 0.3*cm))

    var_display_keys = [
        "Historical VaR 95%", "Historical VaR 99%",
        "Parametric VaR 95%", "Parametric VaR 99%",
        "CVaR 95%",           "CVaR 99%",
        "1-Day Loss at 95% (₹/$)", "1-Day Loss at 99% (₹/$)",
    ]
    var_rows = [["Metric", "Value"]]
    for key in var_display_keys:
        if key in var_report:
            val = var_report[key]
            if "Loss" in key:
                var_rows.append([key, f"{abs(float(val)):,.0f}"])
            else:
                var_rows.append([key, _pct(val)])

    var_tbl = Table(var_rows, colWidths=[8*cm, 4*cm])
    var_tbl.setStyle(_table_style())
    story.append(var_tbl)
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "Historical VaR makes no distributional assumption. Parametric VaR assumes normality — "
        "it may underestimate tail risk during market stress events.",
        S["caption"],
    ))
    story.append(PageBreak())


    # ─────────────────────────────────────────────────────────────────────
    # SECTION 4 — CAPM
    # ─────────────────────────────────────────────────────────────────────
    story.append(Paragraph("4. CAPM Analysis", S["section_header"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_BLUE, spaceAfter=8))
    story.append(Paragraph(
        f"Beta measures sensitivity to market ({benchmark}) moves. "
        "Alpha is the excess return above CAPM's prediction. "
        "R² indicates how much of return variance is explained by market movements.",
        S["body"],
    ))
    story.append(Spacer(1, 0.3*cm))

    capm_cols = ["Beta", "Alpha (Ann.)", "Expected Return", "R²", "p-value"]
    capm_header = ["Ticker"] + capm_cols
    capm_rows = [capm_header]

    for ticker in capm_df.index:
        row = [ticker]
        for col in capm_cols:
            val = capm_df.loc[ticker, col] if col in capm_df.columns else "N/A"
            if col in ["Alpha (Ann.)", "Expected Return"]:
                row.append(_pct(val))
            else:
                row.append(_rat(val, 3))
        capm_rows.append(row)

    capm_tbl = Table(capm_rows, colWidths=[2.5*cm]+[3*cm]*len(capm_cols), repeatRows=1)
    capm_tbl.setStyle(_table_style())
    story.append(capm_tbl)
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "β < 1.0 = defensive  ·  β > 1.0 = aggressive  ·  p-value < 0.05 = statistically significant alpha",
        S["caption"],
    ))
    story.append(PageBreak())


    # ─────────────────────────────────────────────────────────────────────
    # SECTION 5 — STRATEGY COMPARISON
    # ─────────────────────────────────────────────────────────────────────
    story.append(Paragraph("5. Strategy Comparison", S["section_header"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_BLUE, spaceAfter=8))
    story.append(Paragraph(
        "Backtested strategies on the selected asset vs Buy & Hold baseline. "
        "Sharpe Ratio is the primary ranking metric — it measures return per unit of risk taken.",
        S["body"],
    ))
    story.append(Spacer(1, 0.3*cm))

    strat_header = ["Strategy"] + list(strategy_table.columns)
    strat_rows   = [strat_header]
    for idx, row in strategy_table.iterrows():
        formatted = [str(idx)]
        for col, val in zip(strategy_table.columns, row):
            try:
                fval = float(val)
                if col in ["Total Return","CAGR","Volatility","Max Drawdown"]:
                    formatted.append(_pct(fval))
                elif col == "Num Trades":
                    formatted.append(str(int(fval)))
                else:
                    formatted.append(_rat(fval))
            except Exception:
                formatted.append(str(val))
        strat_rows.append(formatted)

    strat_col_w = [4.5*cm] + [2.5*cm]*(len(strat_header)-1)
    strat_tbl   = Table(strat_rows, colWidths=strat_col_w, repeatRows=1)
    strat_cmds  = _table_style().getCommands()

    # Highlight best Sharpe row
    if "Sharpe Ratio" in strategy_table.columns:
        sharpe_col_idx = list(strategy_table.columns).index("Sharpe Ratio") + 1
        try:
            best_row = strategy_table["Sharpe Ratio"].astype(float).idxmax()
            best_row_idx = list(strategy_table.index).index(best_row) + 1
            strat_cmds.append(("BACKGROUND", (0, best_row_idx), (-1, best_row_idx), colors.HexColor("#E8F5E9")))
            strat_cmds.append(("FONTNAME", (0, best_row_idx), (-1, best_row_idx), "Helvetica-Bold"))
        except Exception:
            pass

    strat_tbl.setStyle(TableStyle(strat_cmds))
    story.append(strat_tbl)
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph("Highlighted row = best Sharpe Ratio.", S["caption"]))


    # ─────────────────────────────────────────────────────────────────────
    # SECTION 6 — FAMA-FRENCH (optional)
    # ─────────────────────────────────────────────────────────────────────
    if ff3_df is not None:
        story.append(PageBreak())
        story.append(Paragraph("6. Fama-French 3-Factor Analysis", S["section_header"]))
        story.append(HRFlowable(width="100%", thickness=1, color=C_BLUE, spaceAfter=8))

        if using_synthetic_ff:
            story.append(Paragraph(
                "⚠️  Note: Fama-French factor data could not be fetched from Kenneth French's "
                "website. The regression below uses synthetic factor proxies with historically "
                "realistic parameters. Results are illustrative only.",
                ParagraphStyle("warn", parent=S["body"], textColor=colors.HexColor("#E65100"),
                               backColor=colors.HexColor("#FFF3E0"), borderPad=6),
            ))
            story.append(Spacer(1, 0.3*cm))

        story.append(Paragraph(
            "The Fama-French 3-Factor model extends CAPM with two additional risk factors: "
            "SMB (Small Minus Big — size premium) and HML (High Minus Low — value premium). "
            "Alpha here is the return unexplained by all three factors — a stricter test than CAPM alpha.",
            S["body"],
        ))
        story.append(Spacer(1, 0.3*cm))

        ff_cols    = ["Alpha (Ann.)", "Beta Market", "Beta SMB", "Beta HML", "R²", "p-val Alpha"]
        ff_header  = ["Ticker"] + ff_cols
        ff_rows    = [ff_header]
        valid_ff   = ff3_df.dropna(subset=["Alpha (Ann.)"])

        for ticker in valid_ff.index:
            row = [ticker]
            for col in ff_cols:
                val = valid_ff.loc[ticker, col] if col in valid_ff.columns else "N/A"
                if col in ["Alpha (Ann.)"]:
                    row.append(_pct(val))
                else:
                    row.append(_rat(float(val), 3) if val != "N/A" else "N/A")
            ff_rows.append(row)

        ff_col_w = [2.5*cm] + [2.6*cm]*len(ff_cols)
        ff_tbl   = Table(ff_rows, colWidths=ff_col_w, repeatRows=1)
        ff_cmds  = _table_style().getCommands()

        # Colour alpha cells
        for r_idx in range(1, len(ff_rows)):
            try:
                alpha_val = float(valid_ff.iloc[r_idx-1]["Alpha (Ann.)"])
                alpha_col = 1
                colour    = C_GREEN if alpha_val > 0 else C_RED
                ff_cmds  += [
                    ("TEXTCOLOR", (alpha_col, r_idx), (alpha_col, r_idx), colour),
                    ("FONTNAME",  (alpha_col, r_idx), (alpha_col, r_idx), "Helvetica-Bold"),
                ]
            except Exception:
                pass

        ff_tbl.setStyle(TableStyle(ff_cmds))
        story.append(ff_tbl)
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph(
            "SMB β > 0 = small-cap tilt  ·  HML β > 0 = value tilt  ·  HML β < 0 = growth tilt  ·  "
            "p-val < 0.05 = statistically significant alpha",
            S["caption"],
        ))


    # ─────────────────────────────────────────────────────────────────────
    # DISCLAIMER
    # ─────────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 1.5*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_SUBTEXT))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "DISCLAIMER: This report is generated by Quant Research Lab for educational and research "
        "purposes only. It does not constitute investment advice. Past performance is not indicative "
        "of future results. All data sourced from Yahoo Finance via yfinance.",
        S["disclaimer"],
    ))

    # ─────────────────────────────────────────────────────────────────────
    # BUILD
    # ─────────────────────────────────────────────────────────────────────
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    buf.seek(0)
    return buf.read()
