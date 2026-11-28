"""
app/pdf_generator.py - Branded one-page advisory report.  (MandiMind v7.0, NEW FILE)

WHAT'S IN v7.0
--------------
* `build_advisory_pdf()` renders a one-page A4 report with ReportLab:
    - green MandiMind header bar with logo text,
    - crop, position and hold window,
    - a colour-coded recommendation box (HOLD / SELL NOW / SELL IN TRANCHES),
    - key figures: current price, forecast, gross/net change, uncertainty,
      confidence and winning model,
    - an embedded price-trend PNG (history + forecast + 80% interval),
    - footer: "Decision support, not financial advice. Generated {date}."
* `render_trend_png()` draws the chart with Pillow (always installed with
  Streamlit), so no matplotlib / kaleido / extra system packages are needed.
* The report is English-only on purpose: ReportLab's built-in fonts have no Urdu
  glyphs. All text is restricted to characters the built-in fonts can draw.
* Defensive by design: missing history/forecast simply omits the chart, and bad
  numbers are shown as "n/a" instead of raising.
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

BRAND_GREEN = colors.HexColor("#10B981")
BRAND_DARK = colors.HexColor("#0F172A")
BRAND_SLATE = colors.HexColor("#64748B")
BRAND_LIGHT = colors.HexColor("#F1F5F9")

REC_COLORS: dict[str, tuple[str, str]] = {
    "HOLD": ("#047857", "Waiting is expected to pay"),
    "SELL NOW": ("#B91C1C", "Waiting is expected to lose money"),
    "SELL IN TRANCHES": ("#B45309", "Signal is not decisive"),
}

FOOTER_TEXT = "Decision support, not financial advice. Generated {date}."


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def _num(value: Any, digits: int = 0, signed: bool = False, suffix: str = "") -> str:
    """
    Format a number safely for the report.

    Parameters
    ----------
    value : Any
        Number-like value.
    digits : int
        Decimal places.
    signed : bool
        Always show a +/- sign.
    suffix : str
        Text appended after the number (e.g. ``"%"``).

    Returns
    -------
    str
        Formatted number or ``"n/a"`` when the value is missing/not finite.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(v):
        return "n/a"
    fmt = f"{{:{'+' if signed else ''},.{digits}f}}"
    return fmt.format(v) + suffix


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """
    Convert ``#RRGGBB`` to an RGB tuple.

    Parameters
    ----------
    hex_color : str
        Colour such as ``"#10B981"``.

    Returns
    -------
    tuple of int
        ``(r, g, b)``.
    """
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _load_font(size: int) -> Any:
    """
    Load a TrueType font for Pillow, falling back to the built-in font.

    Parameters
    ----------
    size : int
        Font size in pixels.

    Returns
    -------
    PIL.ImageFont.ImageFont
        A usable font object.
    """
    from PIL import ImageFont

    for name in ("DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except (OSError, ValueError):
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # very old Pillow
        return ImageFont.load_default()


def _prepare_series(
    history: pd.Series | None, forecast: pd.DataFrame | None, max_history: int
) -> tuple[list[pd.Timestamp], list[float], list[pd.Timestamp], list[float], list[float], list[float]]:
    """
    Clean the history and forecast inputs for plotting.

    Parameters
    ----------
    history : pd.Series or None
        Price history indexed by date.
    forecast : pd.DataFrame or None
        Columns ``date``, ``forecast`` and optionally ``lo`` and ``hi``.
    max_history : int
        Maximum number of historical points to draw.

    Returns
    -------
    tuple
        ``(hist_dates, hist_vals, fc_dates, fc_vals, lo_vals, hi_vals)``.
    """
    h_dates: list[pd.Timestamp] = []
    h_vals: list[float] = []
    if history is not None and len(history) > 0:
        h = pd.Series(history).dropna().astype(float).tail(max_history)
        h_dates = [pd.Timestamp(d) for d in h.index]
        h_vals = [float(v) for v in h.values]

    f_dates: list[pd.Timestamp] = []
    f_vals: list[float] = []
    lo: list[float] = []
    hi: list[float] = []
    if forecast is not None and len(forecast) > 0 and {"date", "forecast"} <= set(forecast.columns):
        f = forecast.dropna(subset=["forecast"]).copy()
        f_dates = [pd.Timestamp(d) for d in f["date"]]
        f_vals = [float(v) for v in f["forecast"]]
        lo = [float(v) for v in f["lo"]] if "lo" in f.columns else list(f_vals)
        hi = [float(v) for v in f["hi"]] if "hi" in f.columns else list(f_vals)
    return h_dates, h_vals, f_dates, f_vals, lo, hi


# ----------------------------------------------------------------------------
# Chart (Pillow)
# ----------------------------------------------------------------------------
def render_trend_png(
    history: pd.Series | None,
    forecast: pd.DataFrame | None = None,
    width: int = 1500,
    height: int = 560,
    max_history: int = 36,
) -> bytes | None:
    """
    Draw a small price-trend chart as PNG bytes.

    The chart shows recent history (solid dark line), the forecast (dashed green
    line) and the 80% interval (soft green band). It is drawn at 2x resolution and
    down-sampled for smooth anti-aliased edges.

    Parameters
    ----------
    history : pd.Series or None
        Price history indexed by date.
    forecast : pd.DataFrame or None
        Columns ``date``, ``forecast``, ``lo``, ``hi``.
    width, height : int
        Final image size in pixels.
    max_history : int
        Maximum number of historical points drawn.

    Returns
    -------
    bytes or None
        PNG bytes, or None when there is nothing to draw.
    """
    from PIL import Image, ImageDraw

    h_dates, h_vals, f_dates, f_vals, lo, hi = _prepare_series(history, forecast, max_history)
    if len(h_vals) < 2:
        return None

    s = 2  # supersampling factor
    W, H = width * s, height * s
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    font = _load_font(23 * s)
    font_b = _load_font(25 * s)

    left, right, top, bottom = 130 * s, 40 * s, 60 * s, 66 * s
    plot_w, plot_h = W - left - right, H - top - bottom

    all_dates = h_dates + f_dates
    n_total = len(all_dates)
    all_vals = h_vals + f_vals + lo + hi
    y_min, y_max = min(all_vals), max(all_vals)
    pad = (y_max - y_min) * 0.12 or max(abs(y_max) * 0.05, 1.0)
    y_min, y_max = y_min - pad, y_max + pad

    def px(i: int) -> float:
        return left + (plot_w * i / max(n_total - 1, 1))

    def py(v: float) -> float:
        return top + plot_h * (1 - (v - y_min) / (y_max - y_min))

    # horizontal guides + y labels
    for k in range(5):
        v = y_min + (y_max - y_min) * k / 4
        y = py(v)
        d.line([(left, y), (W - right, y)], fill=(226, 232, 240), width=s)
        label = f"{v:,.0f}"
        tw = d.textlength(label, font=font)
        d.text((left - 12 * s - tw, y - 12 * s), label, fill=(100, 116, 139), font=font)
    d.line([(left, top + plot_h), (W - right, top + plot_h)], fill=(148, 163, 184), width=2 * s)

    n_hist = len(h_vals)
    # interval band (anchored at the last actual point)
    if f_vals:
        top_pts = [(px(n_hist - 1), py(h_vals[-1]))] + [(px(n_hist + i), py(hi[i])) for i in range(len(f_vals))]
        bot_pts = [(px(n_hist - 1), py(h_vals[-1]))] + [(px(n_hist + i), py(lo[i])) for i in range(len(f_vals))]
        d.polygon(top_pts + bot_pts[::-1], fill=(209, 250, 229))

    # history line
    hist_pts = [(px(i), py(v)) for i, v in enumerate(h_vals)]
    d.line(hist_pts, fill=(15, 23, 42), width=4 * s, joint="curve")

    # forecast dashed line
    if f_vals:
        fc_pts = [(px(n_hist - 1), py(h_vals[-1]))] + [(px(n_hist + i), py(v)) for i, v in enumerate(f_vals)]
        for (x0, y0), (x1, y1) in zip(fc_pts[:-1], fc_pts[1:]):
            length = float(np.hypot(x1 - x0, y1 - y0)) or 1.0
            dash, gap = 14 * s, 9 * s
            pos = 0.0
            while pos < length:
                end = min(pos + dash, length)
                a, b = pos / length, end / length
                d.line(
                    [(x0 + (x1 - x0) * a, y0 + (y1 - y0) * a), (x0 + (x1 - x0) * b, y0 + (y1 - y0) * b)],
                    fill=(16, 185, 129), width=4 * s,
                )
                pos += dash + gap
        for x, y in fc_pts[1:]:
            r = 7 * s
            d.ellipse([x - r, y - r, x + r, y + r], fill="white", outline=(16, 185, 129), width=3 * s)

    # marker on the last actual point
    lx, ly = hist_pts[-1]
    r = 8 * s
    d.ellipse([lx - r, ly - r, lx + r, ly + r], fill=(15, 23, 42), outline="white", width=2 * s)

    # x labels: first, middle, last
    idxs = sorted({0, n_total // 2, n_total - 1})
    for i in idxs:
        text = all_dates[i].strftime("%b %Y")
        tw = d.textlength(text, font=font)
        x = min(max(px(i) - tw / 2, left - 20 * s), W - right - tw + 20 * s)
        d.text((x, top + plot_h + 12 * s), text, fill=(100, 116, 139), font=font)

    # legend
    lx0 = left
    for label, color, dashed in (("Actual", (15, 23, 42), False), ("Forecast", (16, 185, 129), True)):
        if label == "Forecast" and not f_vals:
            continue
        d.line([(lx0, 22 * s), (lx0 + 34 * s, 22 * s)], fill=color, width=4 * s)
        d.text((lx0 + 42 * s, 8 * s), label, fill=(51, 65, 85), font=font_b)
        lx0 += 42 * s + int(d.textlength(label, font=font_b)) + 34 * s
    if f_vals:
        d.rectangle([lx0, 14 * s, lx0 + 34 * s, 30 * s], fill=(209, 250, 229))
        d.text((lx0 + 42 * s, 8 * s), "80% interval", fill=(51, 65, 85), font=font_b)

    img = img.resize((width, height), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# ----------------------------------------------------------------------------
# PDF page furniture
# ----------------------------------------------------------------------------
def _draw_page(canv: rl_canvas.Canvas, doc: SimpleDocTemplate, generated: datetime) -> None:
    """
    Draw the green header bar (with logo text) and the footer on a page.

    Parameters
    ----------
    canv : reportlab.pdfgen.canvas.Canvas
        Canvas for the current page.
    doc : SimpleDocTemplate
        Document being built (for page size and margins).
    generated : datetime
        Generation timestamp shown in the footer.

    Returns
    -------
    None
    """
    page_w, page_h = A4
    bar_h = 22 * mm
    canv.saveState()

    # header bar
    canv.setFillColor(BRAND_GREEN)
    canv.rect(0, page_h - bar_h, page_w, bar_h, stroke=0, fill=1)
    canv.setFillColor(colors.HexColor("#047857"))
    canv.rect(0, page_h - bar_h, page_w, 1.6 * mm, stroke=0, fill=1)

    # logo mark + text
    mark = 11 * mm
    mx, my = doc.leftMargin, page_h - bar_h + (bar_h - mark) / 2 + 0.8 * mm
    canv.setFillColor(colors.white)
    canv.roundRect(mx, my, mark, mark, 2.2 * mm, stroke=0, fill=1)
    canv.setFillColor(BRAND_GREEN)
    canv.setFont("Helvetica-Bold", 15)
    canv.drawCentredString(mx + mark / 2, my + 3.2 * mm, "M")
    canv.setFillColor(colors.white)
    canv.setFont("Helvetica-Bold", 20)
    canv.drawString(mx + mark + 4 * mm, my + 3.4 * mm, "MandiMind")
    canv.setFont("Helvetica", 8.5)
    canv.drawRightString(page_w - doc.rightMargin, my + 5.6 * mm, "Pakistan Crop Price Intelligence")
    canv.drawRightString(page_w - doc.rightMargin, my + 1.8 * mm, "Advisory Report")

    # footer
    canv.setStrokeColor(colors.HexColor("#E2E8F0"))
    canv.setLineWidth(0.6)
    canv.line(doc.leftMargin, 16 * mm, page_w - doc.rightMargin, 16 * mm)
    canv.setFillColor(BRAND_SLATE)
    canv.setFont("Helvetica", 8)
    canv.drawString(doc.leftMargin, 11 * mm, FOOTER_TEXT.format(date=generated.strftime("%d %b %Y, %H:%M")))
    canv.drawRightString(page_w - doc.rightMargin, 11 * mm, "MandiMind v7.0")
    canv.restoreState()


def build_advisory_pdf(
    crop: str,
    qty: int,
    horizon: int,
    now_price: float,
    forecast_price: float,
    recommendation: str,
    confidence: float,
    rationale: str,
    history: pd.Series | None = None,
    forecast: pd.DataFrame | None = None,
    gross_pct: float | None = None,
    net_pct: float | None = None,
    uncertainty_pct: float | None = None,
    model_name: str | None = None,
    generated_on: datetime | None = None,
) -> bytes:
    """
    Build the branded one-page advisory report.

    Parameters
    ----------
    crop : str
        Commodity name.
    qty : int
        Number of 40 kg bags held.
    horizon : int
        Hold window in forecast periods.
    now_price, forecast_price : float
        Latest price and forecast price at the end of the hold window (PKR / 40 kg).
    recommendation : str
        ``"HOLD"``, ``"SELL NOW"`` or ``"SELL IN TRANCHES"``.
    confidence : float
        Model confidence, 0-100.
    rationale : str
        One-paragraph plain-English reason for the recommendation.
    history : pd.Series, optional
        Price history indexed by date (for the trend chart).
    forecast : pd.DataFrame, optional
        Columns ``date``, ``forecast``, ``lo``, ``hi`` (for the trend chart).
    gross_pct, net_pct, uncertainty_pct : float, optional
        Expected gross change, change after storage costs and model error band (%).
    model_name : str, optional
        Name of the winning backtest model.
    generated_on : datetime, optional
        Timestamp for the footer (defaults to now).

    Returns
    -------
    bytes
        The PDF file content.
    """
    generated = generated_on or datetime.now()
    rec = recommendation if recommendation in REC_COLORS else "SELL IN TRANCHES"
    rec_hex, rec_tag = REC_COLORS[rec]

    base = getSampleStyleSheet()
    s_title = ParagraphStyle("t", parent=base["Title"], fontName="Helvetica-Bold", fontSize=20,
                             leading=24, alignment=TA_LEFT, textColor=BRAND_DARK, spaceAfter=2)
    s_sub = ParagraphStyle("s", parent=base["Normal"], fontName="Helvetica", fontSize=10,
                           leading=14, textColor=BRAND_SLATE)
    s_rec_lab = ParagraphStyle("rl", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=8.5,
                               leading=11, textColor=colors.HexColor("#D1FAE5"))
    s_rec = ParagraphStyle("r", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=34,
                           leading=38, textColor=colors.white)
    s_why = ParagraphStyle("w", parent=base["Normal"], fontName="Helvetica", fontSize=10.5,
                           leading=15, textColor=colors.white)
    s_cell_l = ParagraphStyle("cl", parent=base["Normal"], fontName="Helvetica", fontSize=8,
                              leading=10, textColor=BRAND_SLATE)
    s_cell_v = ParagraphStyle("cv", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=14,
                              leading=17, textColor=BRAND_DARK)
    s_h2 = ParagraphStyle("h2", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=11,
                          leading=14, textColor=BRAND_DARK, spaceBefore=4, spaceAfter=4)
    s_small = ParagraphStyle("sm", parent=base["Normal"], fontName="Helvetica", fontSize=8,
                             leading=11, textColor=BRAND_SLATE)

    safe_crop = escape(str(crop))
    story: list[Any] = [
        Spacer(1, 2 * mm),
        Paragraph(f"Advisory Report: {safe_crop}", s_title),
        Paragraph(
            f"Position: <b>{int(qty):,}</b> bags (40 kg) &nbsp;|&nbsp; Hold window: "
            f"<b>{int(horizon)}</b> period(s) &nbsp;|&nbsp; Prices in PKR per 40 kg",
            s_sub,
        ),
        Spacer(1, 5 * mm),
    ]

    # recommendation box
    rec_tbl = Table(
        [[Paragraph(f"RECOMMENDATION &nbsp;-&nbsp; {escape(rec_tag).upper()}", s_rec_lab)],
         [Paragraph(escape(rec), s_rec)],
         [Paragraph(escape(rationale or ""), s_why)]],
        colWidths=[A4[0] - 40 * mm],
    )
    rec_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(rec_hex)),
        ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (0, 0), 10), ("BOTTOMPADDING", (0, -1), (-1, -1), 12),
        ("TOPPADDING", (0, 1), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -2), 2),
        ("ROUNDEDCORNERS", [6, 6, 6, 6]),
    ]))
    story += [rec_tbl, Spacer(1, 5 * mm)]

    # key figures
    def cell(label: str, value: str) -> list[Paragraph]:
        return [Paragraph(escape(label).upper(), s_cell_l), Paragraph(escape(value), s_cell_v)]

    fig_rows = [
        [cell("Current price", _num(now_price)),
         cell(f"Forecast (+{int(horizon)})", _num(forecast_price)),
         cell("Gross change", _num(gross_pct, 1, True, "%"))],
        [cell("Net after storage", _num(net_pct, 1, True, "%")),
         cell("Error band (+/-)", _num(uncertainty_pct, 1, False, "%")),
         cell("Confidence", _num(confidence, 0, False, "/100"))],
    ]
    fig_tbl = Table(fig_rows, colWidths=[(A4[0] - 40 * mm) / 3] * 3)
    fig_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BRAND_LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#E2E8F0")),
        ("INNERGRID", (0, 0), (-1, -1), 0.6, colors.white),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    value_total = _num(float(now_price) * int(qty)) if np.isfinite(now_price) else "n/a"
    story += [fig_tbl, Spacer(1, 3 * mm),
              Paragraph(f"Position value at current price: <b>PKR {value_total}</b>"
                        + (f" &nbsp;|&nbsp; Winning backtest model: <b>{escape(str(model_name))}</b>" if model_name else ""),
                        s_small),
              Spacer(1, 4 * mm)]

    # chart
    story.append(Paragraph("Price trend and forecast", s_h2))
    try:
        png = render_trend_png(history, forecast)
    except Exception:  # never let a chart problem block the report
        png = None
    if png:
        img_w = A4[0] - 40 * mm
        story.append(Image(io.BytesIO(png), width=img_w, height=img_w * 560 / 1500))
    else:
        story.append(Paragraph("Not enough history to draw a trend chart.", s_small))

    story += [Spacer(1, 4 * mm),
              Paragraph("How to read this: the forecast comes from the model that scored best in a "
                        "walk-forward backtest on this crop. The green band is the 80% interval. "
                        "The recommendation compares expected net gain (after storage and financing "
                        "costs) with the model's own error band.", s_small)]

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=22 * mm + 6 * mm, bottomMargin=22 * mm,
        title=f"MandiMind Advisory - {crop}", author="MandiMind",
    )

    def on_page(canv: rl_canvas.Canvas, d: SimpleDocTemplate) -> None:
        _draw_page(canv, d, generated)

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return buf.getvalue()