"""
app/main.py - MandiMind v9.0 | Pakistan Crop Price & Supply-Shock Intelligence

Design system : dark slate sidebar (#0F172A), airy light canvas (#F8FAFC), glassmorphism KPI
                cards, Inter typeface, zero gridlines, one stable colour per commodity.
Stack         : Streamlit + Plotly graph objects only (Scattergeo map, no mapping libraries).
Resilience    : empty/missing anomalies.parquet, metrics.json, district_profiles.parquet and
                models never break a page. Every export and chart is guarded.

WHAT CHANGED IN v9.0
--------------------
1. STATISTICS          "Why This Matters" shows fixed figures with a short CSS fade-in. The CSS ``@property``
                       counter is gone.
2. MAP                 Crowded districts are nudged apart, bubbles are smaller, and bold white-halo labels
                       sit outside their own bubble.
3. TICKER              One 120s loop over two identical, screen-wide halves, so there is no jump.
4. CHAT                "Ask MandiMind" is a sidebar expander with a form and ``chat_history`` in session state.
5. DESIGN              Glow on HOLD / SELL NOW cards, softer confidence band, restyled Pro password box,
                       sidebar profile chip with a gold PRO badge, sparkle on the chat label.

WHAT CHANGED IN v8.0
--------------------
1. LIVE MARKET TICKER  Bloomberg-style marquee above every page: latest % change per crop, green up / red down.
2. PRO MODE            Sidebar toggle + password. Unlocks "Client Portfolio Simulator" (budget -> allocation
                       ranked by risk-adjusted expected return) and "Export Center" (every table, chart and
                       metric as CSV / Excel / PDF / ZIP / HTML).
3. WHY THIS MATTERS    Home page storytelling banner with three pure-CSS count-up statistics.
4. ASK MANDIMIND       Floating chat widget on every page. Keyword routing (no LLM): crop -> price + forecast,
                       sell/hold/store -> advisor logic, risk/district -> ranking, otherwise a friendly fallback.
5. MARKET NEWS         Market Overview expander with 3-5 mock headlines generated from the loaded data.
6. POLISH              Loading skeleton, fade-in cards, back-to-top button, footer, version badge + changelog,
                       PNG download button on every Plotly chart.

WHAT CHANGED IN v7.0
--------------------
1. URDU LOCALIZATION   Sidebar toggle "English / اردو". A `LANG` dictionary ("en"/"ur")
                       translates navigation, KPIs, captions, recommendations and buttons.
                       Urdu mode injects `direction: rtl; text-align: right;` CSS. Plotly
                       legends, ticks and hover text stay in English by design (charts are
                       forced back to LTR), only Streamlit markdown and KPI cards flip.
2. ROOT CAUSE AI       Forecast Lab expander "Why did this happen?". Simulated causal
                       attribution with numpy/pandas only: the z-score of each candidate
                       driver (rainfall, monsoon, Ramadan, previous momentum, diesel, FX,
                       heat) at the anomaly date vs its history becomes a SHAP-style weight.
3. ENTERPRISE EXPORTS  Decision Advisor: branded PDF (app/pdf_generator.py), full forecast
                       CSV and a 4-sheet Excel workbook (Prices, Anomalies, Forecasts,
                       Metrics) built with openpyxl.
4. LIVE ALERTS         Sidebar "Live Alerts" feed (5 latest anomalies, red = spike, blue =
                       drop, with a plain-text reason) + Refresh button that re-reads
                       anomalies.parquet + a banner on Market Overview for the last 30 days
                       of data.
5. WHAT-IF SIMULATOR   New page: rainfall / diesel / FX sliders + Ramadan toggle applied to
                       the forecast with a regression-style proxy; before-vs-after chart and
                       a plain-language summary.
6. PERFORMANCE         `@st.cache_data` for every loader and for the whole forecast engine
                       (switching tabs never re-runs the backtest), `@st.cache_resource` for
                       model loading, uniform Plotly config, type hints and numpydoc
                       docstrings everywhere.
7. LANDING PAGE        New default page "Home": hero, Get Started button, three glass
                       feature cards and "Today's Top Movers".

Data note: the app now keeps each series' latest period (build_features(for_inference=True)),
so "Price now" is the true latest price instead of the second-to-last period used in v6.
"""

from __future__ import annotations

import html as _html
import inspect
import io
import json
import random
import re
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent
for _p in (str(BASE_DIR), str(APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline.features import (  # noqa: E402
    MODEL_FEATURE_COLS,
    build_features,
    model_filename,
    select_latest_rows,
)

try:  # PDF export is optional: the rest of the app must work without reportlab
    from pdf_generator import build_advisory_pdf  # noqa: E402

    PDF_OK = True
except Exception:  # pragma: no cover
    build_advisory_pdf = None  # type: ignore[assignment]
    PDF_OK = False

# ============================================================================
# CONFIG
# ============================================================================
PRICE = "price_pkr_per_40kg"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MIN_HISTORY = 10
PAGES = ["home", "overview", "geo", "forecast", "advisor", "whatif"]
PRO_PAGES = ["portfolio", "export"]  # unlocked by Pro mode
PRO_PASSWORD = "mandimind2025"  # demo gate for the tiered-SaaS experience, NOT real security
APP_VERSION = "9.0"
DEFAULT_USER = "Guest Analyst"  # shown in the sidebar profile chip until the viewer types a name

INK, MUTED, GRID = "#0F172A", "#64748B", "#E2E8F0"
GREEN, RED, AMBER, BLUE, PURPLE, GREY = "#10B981", "#EF4444", "#F59E0B", "#3B82F6", "#8B5CF6", "#94A3B8"
PALETTE = ["#0EA5E9", "#10B981", "#F59E0B", "#8B5CF6", "#EC4899", "#14B8A6",
           "#F97316", "#6366F1", "#84CC16", "#06B6D4", "#64748B", "#A855F7"]
CLUSTER_COLORS = {"Volatile / High Risk": RED, "Stable / Normal": GREEN}
CHART_CONFIG: dict[str, Any] = {
    "displaylogo": False, "displayModeBar": True, "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    # the built-in camera button saves the chart as a PNG
    "toImageButtonOptions": {"format": "png", "filename": "mandimind_chart", "scale": 2},
}

# district: (lat, lon, hand-tuned label position)
COORDS = {
    "Karachi": (24.8607, 67.0011, "bottom center"), "Hyderabad": (25.3960, 68.3578, "middle right"),
    "Sukkur": (27.7052, 68.8574, "middle left"), "Larkana": (27.5600, 68.2264, "bottom left"),
    "Quetta": (30.1798, 66.9750, "middle left"), "Multan": (30.1575, 71.4700, "bottom left"),
    "Bahawalpur": (29.3956, 71.6836, "bottom center"), "Faisalabad": (31.4504, 73.1350, "middle left"),
    "Lahore": (31.5204, 74.3587, "bottom center"), "Gujranwala": (32.1877, 74.1945, "middle right"),
    "Sialkot": (32.4945, 74.5229, "top right"), "Sargodha": (32.0836, 72.6711, "top left"),
    "Islamabad": (33.6844, 73.0479, "top right"), "Rawalpindi": (33.5651, 73.0169, "bottom right"),
    "Peshawar": (34.0151, 71.5249, "top left"), "Mardan": (34.1986, 72.0404, "top center"),
}
# Display-only nudges (d_lat, d_lon, label position) for districts whose bubbles would collide.
# COORDS above stays the true location. A nudge is applied only when a listed neighbour is also plotted.
# The offsets must be a few tenths of a degree: at this zoom 0.05 degrees is only about 2 pixels.
MAP_NUDGES: dict[str, tuple[float, float, str, tuple[str, ...]]] = {
    "Islamabad":  (0.38, 0.22, "middle right", ("Rawalpindi",)),
    "Rawalpindi": (-0.22, 0.05, "middle right", ("Islamabad",)),
    "Peshawar":   (-0.12, -0.30, "middle left", ("Mardan",)),
    "Mardan":     (0.35, -0.05, "top center", ("Peshawar",)),
    "Sialkot":    (0.30, 0.38, "middle right", ("Gujranwala",)),
    "Gujranwala": (0.02, -0.05, "middle right", ("Sialkot", "Lahore")),
    "Lahore":     (-0.32, 0.22, "middle right", ("Gujranwala",)),
    "Sargodha":   (0.08, -0.15, "middle left", ("Faisalabad",)),
    "Faisalabad": (-0.12, -0.15, "middle left", ("Sargodha",)),
    "Sukkur":     (0.12, 0.28, "middle right", ("Larkana",)),
    "Larkana":    (-0.12, -0.28, "middle left", ("Sukkur",)),
}
# label position for districts that never need a nudge (overrides the hint stored in COORDS)
MAP_LABEL_POS = {"Karachi": "middle left", "Hyderabad": "middle right", "Quetta": "middle left",
                 "Multan": "middle left", "Bahawalpur": "bottom center"}
LABEL_HALO = "0 0 2px #fff, 0 0 3px #fff, 0 0 5px #fff, 0 0 7px #fff"

COUNTRY_LABELS = [("AFGHANISTAN", 33.3, 65.6), ("INDIA", 28.2, 76.0), ("IRAN", 30.0, 61.7),
                  ("CHINA", 36.2, 76.0), ("Arabian Sea", 23.6, 65.0)]

# (key, feature column) pairs used by the Root Cause AI. A driver is only used when its
# column exists and has variation, so missing diesel / FX data is handled gracefully.
DRIVER_SPECS: tuple[tuple[str, str], ...] = (
    ("rain", "precipitation_sum"),
    ("monsoon", "is_monsoon_season"),
    ("ramadan", "is_ramadan_window"),
    ("momentum", "price_pct_change_1"),
    ("diesel", "diesel_pct_change_1"),
    ("fx", "fx_pct_change_1"),
    ("heat", "temperature_2m_max"),
)
RESIDUAL_WEIGHT = 0.5  # z-units reserved for "other local factors" in the attribution pool

# Regression-style What-If proxy: price_impact = 0.15*rain + 0.05*diesel + 0.10*fx
WHATIF_COEF = {"rain": 0.15, "diesel": 0.05, "fx": 0.10}

REC_EN = {"HOLD": "HOLD", "SELL_NOW": "SELL NOW", "TRANCHES": "SELL IN TRANCHES"}

st.set_page_config(page_title="MandiMind | Crop Price Intelligence", page_icon="🌾",
                   layout="wide", initial_sidebar_state="expanded")

# ============================================================================
# LOCALIZATION (LANG dictionary: "en" / "ur")
# ============================================================================
LANG: dict[str, dict[str, str]] = {
    "en": {
        "tagline": "Pakistan crop price & supply-shock intelligence",
        "lang_label": "Language / زبان",
        "nav_home": "🏠 Home",
        "nav_overview": "📊 Market Overview",
        "nav_geo": "🗺️ Geographic Risk",
        "nav_forecast": "🔮 Forecast Lab",
        "nav_advisor": "🤖 Decision Advisor",
        "nav_whatif": "🧪 What-If Simulator",
        "filters_title": "GLOBAL FILTERS",
        "date_range": "Date range",
        "districts": "Districts",
        "sidebar_footer": "v9.0 · Data Mining Project · Open data, free stack",
        "hero_sub": "{d0} → {d1} · {nc} commodities · {nd} districts · {nobs} observations",
        "na": "n/a",
        "unit": "PKR / 40kg",
        "pkr": "PKR",
        "err_load": "Could not load data: {err}",
        "err_run_pipeline": "Run the pipeline first so `data/processed/analysis_table.parquet` exists.",
        "no_data_filters": "No data for the selected filters. Widen the date range or districts.",
        "disclaimer": "Decision support, not financial advice.",
        "disclaimer_long": "Decision support, not financial advice. Based on historical data only.",
        # alerts
        "alerts_title": "🔔 Live Alerts",
        "alerts_refresh": "Refresh",
        "alerts_empty": "No anomalies detected yet.",
        "alerts_updated": "Last refreshed {t}",
        "alert_spike": "Spike",
        "alert_drop": "Drop",
        "banner_shocks": "⚠️ {n} new supply shocks detected this month.",
        # overview
        "kpi_basket": "Basket avg price",
        "kpi_vs_prev": "{x} vs previous period",
        "kpi_anoms": "Supply-shock anomalies",
        "kpi_anoms_sub": "flagged in selected window",
        "kpi_markets": "Markets monitored",
        "kpi_markets_sub": "districts in view",
        "kpi_best_ml": "Best ML crop",
        "kpi_best_ml_sub": "{x} vs naive baseline",
        "kpi_no_metrics": "no model metrics yet",
        "price_trends": "Price trends",
        "commodities": "Commodities",
        "view": "View",
        "view_index": "Index (base = 100)",
        "view_dual": "Dual axis",
        "anomalies_toggle": "Anomalies",
        "pick_one": "Pick at least one commodity.",
        "note_index": "Index view puts cheap and expensive crops on one comparable scale. Red × marks statistically flagged supply shocks.",
        "volatility": "Relative volatility (CV %)",
        "note_cv": "Std-dev ÷ mean, so cheap and expensive crops compare fairly.",
        "cv_none": "Not enough data to compute volatility.",
        # geo
        "geo_unavailable": "District profiles are not available yet. Run the clustering step of the pipeline.",
        "geo_map": "District volatility map",
        "geo_rank": "Risk ranking",
        "geo_no_coords": "None of the selected districts have coordinates configured (COORDS in main.py).",
        "geo_note": "Bubble size = price volatility. Red = volatile / high risk, green = stable. Bubbles of neighbouring districts are nudged apart, so positions are approximate.",
        "geo_missing": "Not plotted (no coordinates configured): {names}",
        "geo_no_districts": "No districts selected.",
        # forecast lab
        "fc_commodity": "Commodity",
        "fc_horizon": "Forecast horizon (periods)",
        "fc_need_history": "Need at least 10 periods of history for a trustworthy backtest. Widen the date range.",
        "kpi_winner": "Winning model",
        "kpi_winner_sub": "lowest walk-forward error",
        "kpi_mape": "Backtest MAPE",
        "kpi_mape_sub": "mean absolute % error",
        "kpi_xgb_mae": "XGBoost MAE",
        "kpi_pipeline_model": "pipeline model",
        "kpi_no_crop_metrics": "no metrics for this crop",
        "kpi_naive_mae": "Naive MAE",
        "kpi_xgb_wins": "XGBoost wins",
        "kpi_baseline_holds": "baseline holds",
        "fc_chart_title": "{crop}: history, forecast and 80% confidence interval",
        "fc_ml_estimate": "XGBoost next-period estimate for the latest data: {px} PKR / 40kg ({chg}). Per-commodity pipeline model.",
        "fc_leaderboard": "Model leaderboard (MAE, lower is better)",
        "fc_leader_note": "Green = winner, auto-selected for the forecast above.",
        "fc_backtest": "Backtest: actual vs predicted (proof the model works)",
        "fc_backtest_note": "Every prediction uses only data available before that date. No leakage.",
        "dl_forecast_csv": "⬇ Download forecast (CSV)",
        # root cause
        "rc_title": "🔍 Why did this happen? (Root Cause AI)",
        "rc_pick": "Select an anomaly to investigate",
        "rc_none": "No detected anomalies for this commodity.",
        "rc_insufficient": "Not enough data around this date to attribute causes.",
        "rc_head": "On {date}, {crop} prices changed by {pct}. The most likely drivers were:",
        "rc_line": "{i}. {name} (impact: {w}%)",
        "rc_conf": "Confidence: {c}%",
        "rc_caption": "Simulated attribution: each driver's z-score at the anomaly date versus its own history becomes its weight. It shows association, not proven causation.",
        "drv_rain_up": "Heavy rainfall disrupting supply",
        "drv_rain_down": "Dry, favourable weather easing supply",
        "drv_monsoon_up": "Monsoon / flood season supply risk",
        "drv_monsoon_down": "Outside monsoon season (normal supply)",
        "drv_ramadan_up": "Ramadan demand surge",
        "drv_ramadan_down": "Outside Ramadan (demand easing)",
        "drv_momentum_up": "Upward momentum from the previous period",
        "drv_momentum_down": "Downward momentum from the previous period",
        "drv_diesel_up": "Rising diesel prices (transport cost)",
        "drv_diesel_down": "Falling diesel prices (cheaper transport)",
        "drv_fx_up": "Weaker rupee raising import and input costs",
        "drv_fx_down": "Stronger rupee easing import and input costs",
        "drv_heat_up": "Extreme heat stressing crops",
        "drv_heat_down": "Mild temperatures (good crop conditions)",
        "drv_other": "Other local market factors",
        # advisor
        "adv_position": "Your position",
        "adv_crop": "Crop you hold",
        "adv_qty": "Quantity (40kg bags)",
        "adv_hold": "Hold window (periods)",
        "adv_storage": "Storage + financing cost (% per period)",
        "adv_no_history": "Not enough history for this crop in the selected window.",
        "adv_bad_price": "The latest price for this crop is not valid, so no advice can be given.",
        "rec_label": "RECOMMENDATION · {crop} · {qty} BAGS",
        "rec_hold": "HOLD",
        "rec_sell_now": "SELL NOW",
        "rec_tranches": "SELL IN TRANCHES",
        "why_hold": "Expected net gain of {net} after storage costs clears the model's error band (±{err}). Waiting is worth it.",
        "why_sell_now": "Expected net change of {net} after storage costs. Waiting loses money on average.",
        "why_tranches": "Net {net} sits inside the ±{err} error band, so the signal is not decisive. Sell about half now and half later to reduce regret.",
        "kpi_price_now": "Price now",
        "kpi_price_now_sub": "latest period average",
        "kpi_forecast_n": "Forecast +{h}",
        "kpi_gross": "{x} gross",
        "kpi_position": "Position value now",
        "kpi_bags": "{n} bags",
        "adv_confidence": "Model confidence",
        "adv_conf_note": "Backtest MAPE {m}% · winning model: {b}",
        "adv_compare": "Sell now vs hold (PKR, after storage cost)",
        "adv_compare_note": "Storage cost over {h} period(s): PKR {c}. Bars show total proceeds.",
        "adv_why": "Why this recommendation? (transparent scoring)",
        "adv_why_price": "Price vs history: <b>{z} σ</b> ({tag})",
        "tag_expensive": "expensive",
        "tag_cheap": "cheap",
        "tag_normal": "normal",
        "adv_why_mom": "3-period momentum: <b>{m}</b>",
        "adv_why_net": "Forecast gross <b>{g}</b> − storage <b>{s}</b> = net <b>{n}</b>",
        "adv_why_unc": "Model uncertainty over {h} period(s): <b>±{e}</b>",
        "adv_why_rule": "Rule: HOLD if net > uncertainty · SELL NOW if net < −½ uncertainty · otherwise sell in tranches.",
        "dl_pdf": "⬇ Download Advisory Report (PDF)",
        "dl_csv": "⬇ Download Forecast Data (CSV)",
        "dl_xlsx": "⬇ Download Full Analysis (Excel)",
        "pdf_unavailable": "PDF export is unavailable (reportlab is not installed).",
        "pdf_note": "The PDF report is generated in English.",
        "export_failed": "An export could not be prepared: {err}",
        # what-if
        "wi_intro": "Stress-test the forecast against rainfall, fuel and currency shocks.",
        "wi_crop": "Commodity",
        "wi_horizon": "Forecast horizon (periods)",
        "wi_rain": "Rainfall increase (%)",
        "wi_diesel": "Diesel price increase (%)",
        "wi_fx": "PKR/USD exchange rate change (%)",
        "wi_ramadan": "Ramadan effect",
        "wi_need_history": "Not enough history for this crop to simulate a scenario. Widen the date range.",
        "wi_kpi_base": "Baseline forecast",
        "wi_kpi_base_sub": "{x} vs price now",
        "wi_kpi_scen": "Scenario forecast",
        "wi_kpi_scen_sub": "{x} vs price now",
        "wi_kpi_impact": "Scenario impact",
        "wi_kpi_impact_sub": "vs baseline forecast",
        "wi_chart_title": "{crop}: baseline vs scenario forecast",
        "wi_summary": "Under this scenario, {crop} is expected to change by {pct} over {h} periods. The biggest driver is {driver}.",
        "wi_summary_none": "Under this scenario, {crop} is expected to change by {pct} over {h} periods. All sliders are at baseline, so no scenario driver is active.",
        "wi_drv_rain": "rainfall",
        "wi_drv_diesel": "diesel prices",
        "wi_drv_fx": "the exchange rate",
        "wi_drv_ramadan": "the Ramadan demand effect",
        "wi_caption": "Regression-style proxy: impact = 0.15 × rainfall + 0.05 × diesel + 0.10 × FX, plus a Ramadan lift estimated from this crop's own history ({lift}). The effect builds up evenly across the horizon. Illustrative, not calibrated.",
        # home
        "home_hero_title": "Pakistan's Crop Price Intelligence",
        "home_hero_sub": "Forecast prices, uncover why markets move, and know when to sell. Built on open data for farmers, traders and policy makers.",
        "home_cta": "Get Started",
        "feat1_t": "Predictive AI",
        "feat1_d": "Four auditable models are backtested walk-forward on every crop. The winner drives an 80% confidence forecast.",
        "feat2_t": "Root Cause Analysis",
        "feat2_d": "Pick any supply shock and see what drove it: rainfall, season, momentum, fuel or currency.",
        "feat3_t": "Farmer Advisor",
        "feat3_d": "Get a clear HOLD, SELL NOW or SELL IN TRANCHES call with confidence, plus a printable PDF report.",
        "home_movers": "Today's Top Movers",
        "movers_note": "Biggest price changes in the latest period ({date}).",
        "movers_none": "Not enough data to rank movers yet.",
        "movers_sub": "{x} vs previous period",
        "home_footer": "Decision support, not financial advice.",
    },
    "ur": {
        "tagline": "پاکستان میں فصلوں کی قیمتوں اور سپلائی شاک کی ذہانت",
        "lang_label": "Language / زبان",
        "nav_home": "🏠 ہوم",
        "nav_overview": "📊 مارکیٹ کا جائزہ",
        "nav_geo": "🗺️ جغرافیائی خطرہ",
        "nav_forecast": "🔮 پیشگوئی لیب",
        "nav_advisor": "🤖 فیصلہ مشیر",
        "nav_whatif": "🧪 «اگر ایسا ہو» سمیولیٹر",
        "filters_title": "عمومی فلٹرز",
        "date_range": "تاریخ کی حد",
        "districts": "اضلاع",
        "sidebar_footer": "ورژن 9.0 · ڈیٹا مائننگ پروجیکٹ · کھلا ڈیٹا، مفت ٹیکنالوجی",
        "hero_sub": "{d0} ← {d1} · {nc} اجناس · {nd} اضلاع · {nobs} مشاہدات",
        "na": "دستیاب نہیں",
        "unit": "روپے / 40 کلو",
        "pkr": "روپے",
        "err_load": "ڈیٹا لوڈ نہیں ہو سکا: {err}",
        "err_run_pipeline": "پہلے پائپ لائن چلائیں تاکہ `data/processed/analysis_table.parquet` فائل بن جائے۔",
        "no_data_filters": "منتخب فلٹرز کے لیے کوئی ڈیٹا موجود نہیں۔ تاریخ کی حد یا اضلاع بڑھائیں۔",
        "disclaimer": "فیصلہ سازی میں معاون، مالی مشورہ نہیں۔",
        "disclaimer_long": "فیصلہ سازی میں معاون، مالی مشورہ نہیں۔ صرف ماضی کے ڈیٹا پر مبنی۔",
        # alerts
        "alerts_title": "🔔 براہِ راست الرٹس",
        "alerts_refresh": "تازہ کریں",
        "alerts_empty": "ابھی تک کوئی گڑبڑ سامنے نہیں آئی۔",
        "alerts_updated": "آخری تازہ کاری: {t}",
        "alert_spike": "اضافہ",
        "alert_drop": "کمی",
        "banner_shocks": "⚠️ اس ماہ {n} نئے سپلائی شاک سامنے آئے ہیں۔",
        # overview
        "kpi_basket": "اجناس کی اوسط قیمت",
        "kpi_vs_prev": "پچھلی مدت کے مقابلے میں {x}",
        "kpi_anoms": "سپلائی شاک کی گڑبڑ",
        "kpi_anoms_sub": "منتخب مدت میں نشاندہی شدہ",
        "kpi_markets": "زیرِ نگرانی منڈیاں",
        "kpi_markets_sub": "زیرِ مشاہدہ اضلاع",
        "kpi_best_ml": "مشینی ماڈل کی بہترین فصل",
        "kpi_best_ml_sub": "سادہ ماڈل کے مقابلے میں {x}",
        "kpi_no_metrics": "ماڈل کے اعداد و شمار ابھی دستیاب نہیں",
        "price_trends": "قیمتوں کا رجحان",
        "commodities": "اجناس",
        "view": "منظر",
        "view_index": "انڈیکس (بنیاد = 100)",
        "view_dual": "دوہرا محور",
        "anomalies_toggle": "گڑبڑ کے نشان",
        "pick_one": "کم از کم ایک جنس منتخب کریں۔",
        "note_index": "انڈیکس منظر سستی اور مہنگی فصلوں کو ایک قابلِ موازنہ پیمانے پر لاتا ہے۔ سرخ × نشان شماریاتی طور پر نشان زد سپلائی شاک کو ظاہر کرتے ہیں۔",
        "volatility": "نسبتی اتار چڑھاؤ (CV %)",
        "note_cv": "معیاری انحراف ÷ اوسط، تاکہ سستی اور مہنگی فصلوں کا منصفانہ موازنہ ہو سکے۔",
        "cv_none": "اتار چڑھاؤ نکالنے کے لیے ڈیٹا ناکافی ہے۔",
        # geo
        "geo_unavailable": "ضلعی پروفائل ابھی دستیاب نہیں۔ پائپ لائن کا کلسٹرنگ مرحلہ چلائیں۔",
        "geo_map": "ضلعی اتار چڑھاؤ کا نقشہ",
        "geo_rank": "خطرے کی درجہ بندی",
        "geo_no_coords": "منتخب اضلاع میں سے کسی کے نقاط (main.py میں COORDS) موجود نہیں۔",
        "geo_note": "دائرے کا سائز = قیمت کا اتار چڑھاؤ۔ سرخ = غیر مستحکم / زیادہ خطرہ، سبز = مستحکم۔ قریبی اضلاع کے دائرے الگ دکھانے کے لیے تھوڑا ہٹائے گئے ہیں، اس لیے مقام تقریباً ہے۔",
        "geo_missing": "نقشے پر نہیں دکھائے گئے (نقاط موجود نہیں): {names}",
        "geo_no_districts": "کوئی ضلع منتخب نہیں۔",
        # forecast lab
        "fc_commodity": "جنس",
        "fc_horizon": "پیشگوئی کی مدت (دورانیے)",
        "fc_need_history": "قابلِ اعتماد بیک ٹیسٹ کے لیے کم از کم 10 دورانیوں کی تاریخ درکار ہے۔ تاریخ کی حد بڑھائیں۔",
        "kpi_winner": "بہترین ماڈل",
        "kpi_winner_sub": "سب سے کم بیک ٹیسٹ غلطی",
        "kpi_mape": "بیک ٹیسٹ MAPE",
        "kpi_mape_sub": "اوسط مطلق فیصد غلطی",
        "kpi_xgb_mae": "ایکس جی بوسٹ MAE",
        "kpi_pipeline_model": "پائپ لائن ماڈل",
        "kpi_no_crop_metrics": "اس فصل کے اعداد و شمار دستیاب نہیں",
        "kpi_naive_mae": "سادہ ماڈل MAE",
        "kpi_xgb_wins": "ایکس جی بوسٹ بہتر ہے",
        "kpi_baseline_holds": "سادہ ماڈل برتر ہے",
        "fc_chart_title": "{crop}: ماضی، پیشگوئی اور 80 فیصد اعتماد کا دائرہ",
        "fc_ml_estimate": "تازہ ترین ڈیٹا پر ایکس جی بوسٹ کا اگلی مدت کا اندازہ: {px} روپے / 40 کلو ({chg})۔ ہر جنس کے لیے الگ پائپ لائن ماڈل۔",
        "fc_leaderboard": "ماڈلز کی درجہ بندی (MAE، جتنا کم اتنا بہتر)",
        "fc_leader_note": "سبز = فاتح، جو اوپر کی پیشگوئی کے لیے خود بخود منتخب ہوا۔",
        "fc_backtest": "بیک ٹیسٹ: اصل بمقابلہ پیشگوئی (ماڈل کے کام کرنے کا ثبوت)",
        "fc_backtest_note": "ہر پیشگوئی صرف اس تاریخ سے پہلے کے ڈیٹا پر مبنی ہے۔ کوئی ڈیٹا لیک نہیں۔",
        "dl_forecast_csv": "⬇ پیشگوئی ڈاؤن لوڈ کریں (CSV)",
        # root cause
        "rc_title": "🔍 ایسا کیوں ہوا؟ (بنیادی وجہ کا تجزیہ)",
        "rc_pick": "تحقیق کے لیے کوئی گڑبڑ منتخب کریں",
        "rc_none": "اس جنس کے لیے کوئی گڑبڑ نہیں ملی۔",
        "rc_insufficient": "اس تاریخ کے آس پاس وجوہات کے تعین کے لیے ڈیٹا ناکافی ہے۔",
        "rc_head": "{date} کو {crop} کی قیمتوں میں {pct} کی تبدیلی آئی۔ سب سے ممکنہ وجوہات یہ تھیں:",
        "rc_line": "{i}۔ {name} (اثر: {w}%)",
        "rc_conf": "اعتماد: {c}%",
        "rc_caption": "یہ نقلی (سمیولیٹڈ) تجزیہ ہے: ہر عنصر کا گڑبڑ کی تاریخ پر اپنی تاریخی اوسط سے انحراف (زی اسکور) ہی اس کا وزن بنتا ہے۔ یہ تعلق ظاہر کرتا ہے، ثابت شدہ سبب نہیں۔",
        "drv_rain_up": "شدید بارش کے باعث رسد میں رکاوٹ",
        "drv_rain_down": "خشک اور سازگار موسم سے رسد میں آسانی",
        "drv_monsoon_up": "مون سون / سیلاب کے موسم میں رسد کا خطرہ",
        "drv_monsoon_down": "مون سون کا موسم نہیں (معمول کی رسد)",
        "drv_ramadan_up": "رمضان میں طلب کا بڑھ جانا",
        "drv_ramadan_down": "رمضان کا وقت نہیں (طلب میں کمی)",
        "drv_momentum_up": "پچھلی مدت سے قیمتوں میں اضافے کا رجحان",
        "drv_momentum_down": "پچھلی مدت سے قیمتوں میں کمی کا رجحان",
        "drv_diesel_up": "ڈیزل کی قیمت میں اضافہ (ترسیل کا خرچ)",
        "drv_diesel_down": "ڈیزل کی قیمت میں کمی (سستی ترسیل)",
        "drv_fx_up": "روپے کی کمزوری سے درآمدی اور پیداواری لاگت میں اضافہ",
        "drv_fx_down": "روپے کی مضبوطی سے درآمدی اور پیداواری لاگت میں کمی",
        "drv_heat_up": "شدید گرمی سے فصلوں پر دباؤ",
        "drv_heat_down": "معتدل درجہ حرارت (فصل کے لیے سازگار حالات)",
        "drv_other": "مقامی منڈی کے دیگر عوامل",
        # advisor
        "adv_position": "آپ کی پوزیشن",
        "adv_crop": "آپ کے پاس موجود فصل",
        "adv_qty": "مقدار (40 کلو کی بوریاں)",
        "adv_hold": "روکنے کی مدت (دورانیے)",
        "adv_storage": "ذخیرہ + مالی اخراجات (فیصد فی دورانیہ)",
        "adv_no_history": "منتخب مدت میں اس فصل کی کافی تاریخ موجود نہیں۔",
        "adv_bad_price": "اس فصل کی تازہ ترین قیمت درست نہیں، اس لیے مشورہ نہیں دیا جا سکتا۔",
        "rec_label": "سفارش · {crop} · {qty} بوریاں",
        "rec_hold": "روکیں",
        "rec_sell_now": "ابھی بیچیں",
        "rec_tranches": "قسطوں میں بیچیں",
        "why_hold": "ذخیرہ اخراجات کے بعد متوقع خالص منافع {net} ماڈل کی غلطی کی حد (±{err}) سے زیادہ ہے۔ انتظار فائدہ مند ہے۔",
        "why_sell_now": "ذخیرہ اخراجات کے بعد متوقع خالص تبدیلی {net} ہے۔ انتظار کرنے سے اوسطاً نقصان ہوگا۔",
        "why_tranches": "خالص تبدیلی {net} ماڈل کی غلطی کی حد (±{err}) کے اندر ہے، اس لیے اشارہ فیصلہ کن نہیں۔ تقریباً آدھا مال ابھی اور آدھا بعد میں بیچیں تاکہ پچھتاوے کا امکان کم ہو۔",
        "kpi_price_now": "موجودہ قیمت",
        "kpi_price_now_sub": "تازہ ترین مدت کی اوسط",
        "kpi_forecast_n": "پیشگوئی +{h}",
        "kpi_gross": "مجموعی {x}",
        "kpi_position": "موجودہ مالیت",
        "kpi_bags": "{n} بوریاں",
        "adv_confidence": "ماڈل پر اعتماد",
        "adv_conf_note": "بیک ٹیسٹ MAPE {m}% · بہترین ماڈل: {b}",
        "adv_compare": "ابھی بیچنا بمقابلہ روکنا (روپے، ذخیرہ خرچ کے بعد)",
        "adv_compare_note": "{h} دورانیے (دورانیوں) کا ذخیرہ خرچ: {c} روپے۔ ستون کل آمدنی دکھاتے ہیں۔",
        "adv_why": "یہ سفارش کیوں؟ (شفاف اسکورنگ)",
        "adv_why_price": "تاریخی اوسط کے مقابلے میں قیمت: <b>{z} σ</b> ({tag})",
        "tag_expensive": "مہنگی",
        "tag_cheap": "سستی",
        "tag_normal": "معمول کے مطابق",
        "adv_why_mom": "تین دورانیوں کا رجحان: <b>{m}</b>",
        "adv_why_net": "پیشگوئی (مجموعی) <b>{g}</b> − ذخیرہ <b>{s}</b> = خالص <b>{n}</b>",
        "adv_why_unc": "{h} دورانیے (دورانیوں) میں ماڈل کی غیر یقینی صورتحال: <b>±{e}</b>",
        "adv_why_rule": "اصول: خالص > غیر یقینی ہو تو روکیں · خالص < −½ غیر یقینی ہو تو ابھی بیچیں · ورنہ قسطوں میں بیچیں۔",
        "dl_pdf": "⬇ مشاورتی رپورٹ ڈاؤن لوڈ کریں (PDF)",
        "dl_csv": "⬇ پیشگوئی کا ڈیٹا ڈاؤن لوڈ کریں (CSV)",
        "dl_xlsx": "⬇ مکمل تجزیہ ڈاؤن لوڈ کریں (Excel)",
        "pdf_unavailable": "پی ڈی ایف برآمد دستیاب نہیں (reportlab نصب نہیں)۔",
        "pdf_note": "پی ڈی ایف رپورٹ انگریزی میں تیار ہوتی ہے۔",
        "export_failed": "ایک برآمدی فائل تیار نہیں ہو سکی: {err}",
        # what-if
        "wi_intro": "بارش، ایندھن اور کرنسی کے جھٹکوں کے مقابل پیشگوئی کو آزمائیں۔",
        "wi_crop": "جنس",
        "wi_horizon": "پیشگوئی کی مدت (دورانیے)",
        "wi_rain": "بارش میں اضافہ (%)",
        "wi_diesel": "ڈیزل کی قیمت میں اضافہ (%)",
        "wi_fx": "روپے / ڈالر کی شرحِ تبادلہ میں تبدیلی (%)",
        "wi_ramadan": "رمضان کا اثر",
        "wi_need_history": "منظرنامہ چلانے کے لیے اس فصل کی کافی تاریخ موجود نہیں۔ تاریخ کی حد بڑھائیں۔",
        "wi_kpi_base": "بنیادی پیشگوئی",
        "wi_kpi_base_sub": "موجودہ قیمت کے مقابلے میں {x}",
        "wi_kpi_scen": "منظرنامے کی پیشگوئی",
        "wi_kpi_scen_sub": "موجودہ قیمت کے مقابلے میں {x}",
        "wi_kpi_impact": "منظرنامے کا اثر",
        "wi_kpi_impact_sub": "بنیادی پیشگوئی کے مقابلے میں",
        "wi_chart_title": "{crop}: بنیادی بمقابلہ منظرنامے کی پیشگوئی",
        "wi_summary": "اس منظرنامے کے تحت {crop} کی قیمت {h} دورانیوں میں {pct} تبدیل ہونے کی توقع ہے۔ سب سے بڑا محرک {driver} ہے۔",
        "wi_summary_none": "اس منظرنامے کے تحت {crop} کی قیمت {h} دورانیوں میں {pct} تبدیل ہونے کی توقع ہے۔ تمام سلائیڈر بنیادی سطح پر ہیں، اس لیے کوئی محرک فعال نہیں۔",
        "wi_drv_rain": "بارش",
        "wi_drv_diesel": "ڈیزل کی قیمتیں",
        "wi_drv_fx": "شرحِ تبادلہ",
        "wi_drv_ramadan": "رمضان میں طلب کا اثر",
        "wi_caption": "ریگریشن طرز کا اندازہ: اثر = 0.15 × بارش + 0.05 × ڈیزل + 0.10 × شرحِ تبادلہ، اور اس فصل کی اپنی تاریخ سے نکالا گیا رمضان کا اضافہ ({lift})۔ اثر پوری مدت میں یکساں طور پر بڑھتا ہے۔ یہ صرف مثالی اندازہ ہے، باقاعدہ کیلیبریٹ شدہ نہیں۔",
        # home
        "home_hero_title": "پاکستان کی فصلوں کی قیمتوں کی ذہانت",
        "home_hero_sub": "قیمتوں کی پیشگوئی کریں، منڈیوں میں اتار چڑھاؤ کی وجہ جانیں، اور فیصلہ کریں کہ کب بیچنا ہے۔ کسانوں، تاجروں اور پالیسی سازوں کے لیے کھلے ڈیٹا پر مبنی۔",
        "home_cta": "شروع کریں",
        "feat1_t": "پیشگوئی کرنے والی مصنوعی ذہانت",
        "feat1_d": "ہر فصل پر چار قابلِ جانچ ماڈلز ماضی کے ڈیٹا پر آزمائے جاتے ہیں۔ بہترین ماڈل 80 فیصد اعتماد کی پیشگوئی دیتا ہے۔",
        "feat2_t": "بنیادی وجوہات کا تجزیہ",
        "feat2_d": "کوئی بھی سپلائی شاک منتخب کریں اور دیکھیں اس کے پیچھے کیا تھا: بارش، موسم، رجحان، ایندھن یا کرنسی۔",
        "feat3_t": "کسان مشیر",
        "feat3_d": "اعتماد کے ساتھ واضح مشورہ پائیں: روکیں، ابھی بیچیں یا قسطوں میں بیچیں۔ ساتھ ہی پرنٹ کے قابل پی ڈی ایف رپورٹ بھی۔",
        "home_movers": "آج کے سب سے بڑے اتار چڑھاؤ",
        "movers_note": "تازہ ترین مدت ({date}) میں قیمتوں کی سب سے بڑی تبدیلیاں۔",
        "movers_none": "اتار چڑھاؤ کی درجہ بندی کے لیے ابھی ڈیٹا ناکافی ہے۔",
        "movers_sub": "پچھلی مدت کے مقابلے میں {x}",
        "home_footer": "فیصلہ سازی میں معاون، مالی مشورہ نہیں۔",
    },
}


# ----------------------------------------------------------------------------
# v8.0 / v9.0 additions to the LANG dictionary. Any key missing in Urdu falls back to English.
# ----------------------------------------------------------------------------
LANG_V8: dict[str, dict[str, str]] = {
    "en": {
        "nav_portfolio": "💼 Client Portfolio Simulator",
        "nav_export": "📊 Export Center",
        "pro_toggle": "💎 Pro mode",
        "pro_pwd": "Pro password",
        "pro_wrong": "Wrong password. Try again.",
        "pro_active": "PRO UNLOCKED · 2 extra pages",
        "pro_hint": "Unlocks the Portfolio Simulator and the Export Center.",
        "pro_locked": "This page is part of MandiMind Pro. Switch on Pro mode in the sidebar and enter the password.",
        "why_title": "Why This Matters",
        "why_s1": "of Pakistan's population depends on agriculture",
        "why_s2": "lost to supply shocks in 2023",
        "why_s3": "of farmers have access to price forecasts",
        "why_pre_only": "Only",
        "why_pkr_billion": "PKR billion",
        "why_close": "MandiMind exists to close this gap.",
        "news_title": "📰 Market News",
        "news_caption": "Simulated headlines generated from the loaded data. This is not a live news feed.",
        "chat_launcher": "💬 Ask MandiMind ✨",
        "chat_title": "Ask MandiMind",
        "chat_welcome": "Hi! I can answer questions about crop prices, forecasts, sell/hold decisions and district risk. Try one of the suggestions below.",
        "chat_placeholder": "e.g. What is the price of wheat?",
        "chat_send": "Send",
        "chat_clear": "Clear chat",
        "chat_fallback": "I can help with crop prices, forecasts, and risk analysis. Try asking about Wheat, Sugar, or your district.",
        "footer_made": "Made with ❤️ in Pakistan 🇵🇰",
        "changelog_title": "What's new in v9.0",
        "user_name": "Your display name",
        "user_free": "Free plan",
        "user_pro": "Pro plan",
        "back_to_top": "Back to top",
        "pf_intro": "Enter a budget and MandiMind ranks every crop by risk-adjusted expected return over the chosen horizon, then proposes an allocation. Forecasts are statistical baselines, not guarantees.",
        "pf_inputs": "Your mandate",
        "pf_budget": "Budget (PKR)",
        "pf_horizon": "Holding period (periods)",
        "pf_max_pos": "Maximum number of crops",
        "pf_none": "No crop shows a positive expected return after storage costs for this horizon. Holding cash (or selling existing stock) looks safer.",
        "pf_kpi_invested": "Invested",
        "pf_kpi_cash": "{x} stays in cash",
        "pf_kpi_value": "Expected value",
        "pf_kpi_profit": "Expected profit",
        "pf_kpi_down": "Pessimistic case",
        "pf_kpi_down_sub": "Lower edge of the 80% band",
        "pf_alloc": "Proposed allocation",
        "pf_range": "Net return by crop (with 80% band)",
        "pf_rank": "Opportunity ranking (all crops)",
        "pf_plan": "Purchase plan",
        "pf_dl": "⬇ Download plan (CSV)",
        "pf_note": "Score = net expected return ÷ forecast uncertainty. Weights are proportional to the score. Bags are 40 kg each.",
        "ex_intro": "Everything in MandiMind, ready to take away. All exports respect the date and district filters in the sidebar.",
        "ex_tables": "Tables (CSV)",
        "ex_reports": "Reports (Excel / PDF / bundle)",
        "ex_charts": "Charts (interactive HTML)",
        "ex_metrics": "Model metrics",
        "ex_horizon": "Forecast horizon (periods)",
        "ex_crop": "Commodity for single-crop reports",
        "ex_png_hint": "Every chart in the app also has a camera icon in its toolbar that saves a PNG.",
    },
    "ur": {
        "nav_portfolio": "💼 کلائنٹ پورٹ فولیو سمیولیٹر",
        "nav_export": "📊 ایکسپورٹ سینٹر",
        "pro_toggle": "💎 پرو موڈ",
        "pro_pwd": "پرو پاس ورڈ",
        "pro_wrong": "غلط پاس ورڈ۔ دوبارہ کوشش کریں۔",
        "pro_active": "پرو فعال · 2 اضافی صفحات",
        "pro_locked": "یہ صفحہ مینڈی مائنڈ پرو کا حصہ ہے۔ سائیڈ بار میں پرو موڈ آن کریں اور پاس ورڈ درج کریں۔",
        "why_title": "یہ کیوں اہم ہے",
        "why_s1": "پاکستان کی آبادی زراعت پر انحصار کرتی ہے",
        "why_s2": "2023 میں سپلائی جھٹکوں سے نقصان",
        "why_s3": "کسانوں کو قیمتوں کی پیشگوئی تک رسائی حاصل ہے",
        "why_pre_only": "صرف",
        "why_pkr_billion": "ارب روپے",
        "why_close": "مینڈی مائنڈ اس خلا کو پُر کرنے کے لیے بنا ہے۔",
        "news_title": "📰 مارکیٹ کی خبریں",
        "news_caption": "یہ سرخیاں لوڈ شدہ ڈیٹا سے خودکار طور پر بنائی گئی ہیں، یہ براہ راست خبروں کا سلسلہ نہیں۔",
        "chat_launcher": "💬 مینڈی مائنڈ سے پوچھیں ✨",
        "chat_title": "مینڈی مائنڈ سے پوچھیں",
        "chat_fallback": "میں فصلوں کی قیمتوں، پیشگوئیوں اور رسک تجزیے میں مدد کر سکتا ہوں۔ گندم، چینی یا اپنے ضلع کے بارے میں پوچھیں۔",
        "chat_send": "بھیجیں",
        "chat_clear": "چیٹ صاف کریں",
        "footer_made": "پاکستان میں ❤️ کے ساتھ بنایا گیا 🇵🇰",
        "changelog_title": "ورژن 9.0 میں نیا کیا ہے",
        "user_name": "آپ کا نام",
        "user_free": "مفت پلان",
        "user_pro": "پرو پلان",
        "back_to_top": "اوپر جائیں",
    },
}
LANG["en"].update(LANG_V8["en"])
LANG["ur"].update(LANG_V8["ur"])

def get_lang() -> str:
    """
    Return the active interface language code.

    Returns
    -------
    str
        ``"en"`` or ``"ur"`` (falls back to ``"en"`` for anything unknown).
    """
    code = st.session_state.get("lang", "en")
    return code if code in LANG else "en"


def T(key: str, lang: str | None = None, **kw: Any) -> str:
    """
    Translate a UI string and fill its placeholders.

    Parameters
    ----------
    key : str
        Key in the ``LANG`` dictionary.
    lang : str, optional
        Force a language (e.g. ``"en"`` for Plotly labels); defaults to the active one.
    **kw : Any
        Values for ``{placeholders}`` in the string.

    Returns
    -------
    str
        The translated text; the English text or the key itself if missing.
    """
    code = lang or get_lang()
    text = LANG.get(code, {}).get(key) or LANG["en"].get(key) or key
    if not kw:
        return text
    try:
        return text.format(**kw)
    except (KeyError, IndexError, ValueError):
        return text


def nm(text: Any) -> str:
    """
    Wrap a number-like string so it keeps left-to-right order inside RTL text.

    Parameters
    ----------
    text : Any
        Value to isolate (e.g. ``"+3.2%"``).

    Returns
    -------
    str
        HTML ``<bdi dir="ltr">`` fragment.
    """
    return f'<bdi dir="ltr">{text}</bdi>'


# ============================================================================
# DESIGN SYSTEM (CSS)
# ============================================================================
BASE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Noto+Naskh+Arabic:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Inter', 'Noto Naskh Arabic', sans-serif; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

/* airy canvas with soft colour blooms so the glass cards have something to refract */
.stApp {
  background:
    radial-gradient(900px 420px at 8% -8%,  rgba(16,185,129,.13), transparent 62%),
    radial-gradient(800px 420px at 100% 0%, rgba(59,130,246,.12), transparent 60%),
    #F8FAFC;
  color: #0F172A;
}
.block-container { padding-top: 1.4rem; max-width: 1480px; }
[data-testid="stMain"] p, [data-testid="stMain"] label, [data-testid="stMain"] li { color: #1E293B; }

/* sidebar */
section[data-testid="stSidebar"] { background: #0F172A; }
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] [role="radiogroup"] p { color: #E2E8F0 !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(148,163,184,.25); }

/* hero */
.hero { background: linear-gradient(120deg,#0F172A 0%,#134E4A 62%,#10B981 140%);
        border-radius: 22px; padding: 26px 32px; margin-bottom: 22px;
        box-shadow: 0 18px 40px -18px rgba(15,23,42,.5); }
.hero h1 { margin:0 !important; font-size:28px; font-weight:800; letter-spacing:-.02em; color:#fff !important; }
.hero p  { margin:6px 0 0 0 !important; font-size:13.5px; color:#CBD5E1 !important; }

/* landing hero + feature cards */
.hero-xl { background: linear-gradient(125deg,#0F172A 0%,#134E4A 58%,#10B981 150%);
           border-radius: 28px; padding: 54px 48px 50px 48px; margin-bottom: 18px; position:relative; overflow:hidden;
           box-shadow: 0 30px 60px -26px rgba(15,23,42,.55); }
.hero-xl::after { content:""; position:absolute; right:-80px; top:-80px; width:340px; height:340px; border-radius:50%;
                  background:rgba(255,255,255,.08); }
.hero-xl .badge { display:inline-block; font-size:12px; font-weight:700; letter-spacing:.12em; color:#A7F3D0 !important;
                  background:rgba(16,185,129,.18); border:1px solid rgba(167,243,208,.35); padding:5px 12px; border-radius:999px; }
.hero-xl h1 { margin:16px 0 10px 0 !important; font-size:46px; line-height:1.1; font-weight:800; letter-spacing:-.025em; color:#fff !important; }
.hero-xl p  { margin:0 !important; font-size:17px; max-width:720px; color:#CBD5E1 !important; line-height:1.6; }
.feat { border-radius:20px; padding:24px 24px 20px 24px; min-height:190px;
        background: rgba(255,255,255,.62);
        -webkit-backdrop-filter: blur(16px) saturate(160%); backdrop-filter: blur(16px) saturate(160%);
        border: 1px solid rgba(255,255,255,.85);
        box-shadow: 0 1px 0 rgba(255,255,255,.9) inset, 0 14px 34px -16px rgba(15,23,42,.25); }
.feat .ico { font-size:30px; margin-bottom:8px; }
.feat h3 { margin:0 0 6px 0 !important; font-size:18px; font-weight:700; color:#0F172A !important; }
.feat p  { margin:0 !important; font-size:14px; line-height:1.6; color:#475569 !important; }
.foot { text-align:center; color:#64748B !important; font-size:13px; margin:34px 0 8px 0; }

/* glassmorphism KPI */
.kpi { position:relative; overflow:hidden; border-radius:18px; padding:18px 22px; height:126px;
       background: rgba(255,255,255,.62);
       -webkit-backdrop-filter: blur(16px) saturate(160%); backdrop-filter: blur(16px) saturate(160%);
       border: 1px solid rgba(255,255,255,.85);
       box-shadow: 0 1px 0 rgba(255,255,255,.9) inset, 0 12px 32px -14px rgba(15,23,42,.22); }
.kpi::before { content:""; position:absolute; left:0; top:0; right:0; height:4px; background:var(--c,#10B981); }
.kpi::after  { content:""; position:absolute; right:-30px; top:-30px; width:110px; height:110px; border-radius:50%;
               background: var(--c,#10B981); opacity:.12; filter: blur(6px); }
.kpi .t { font-size:11px; font-weight:700; color:#64748B !important; text-transform:uppercase; letter-spacing:.08em; }
.kpi .v { font-size:28px; font-weight:800; color:#0F172A !important; line-height:1.15; margin-top:6px; }
.kpi .v small { font-size:13px; font-weight:600; color:#64748B; margin-left:4px; }
.kpi .d { font-size:12.5px; font-weight:600; margin-top:4px; color:var(--c,#10B981) !important; }

/* panels + recommendation */
.sec { font-size:16px; font-weight:700; color:#0F172A; margin:2px 0 6px 0; }
.note { font-size:12.5px; color:#64748B !important; margin:-4px 0 14px 0; }
.rec { border-radius:24px; padding:34px 38px; margin-bottom:18px; position:relative; overflow:hidden;
       box-shadow: 0 24px 50px -22px var(--shadow); }
.rec::after { content:""; position:absolute; right:-60px; top:-60px; width:260px; height:260px; border-radius:50%;
              background:rgba(255,255,255,.14); }
.rec .lab { font-size:12px; font-weight:700; letter-spacing:.14em; color:rgba(255,255,255,.8) !important; }
.rec h2 { margin:4px 0 8px 0 !important; font-size:54px; line-height:1.15; font-weight:800; color:#fff !important; letter-spacing:-.02em; }
.rec p  { margin:0 !important; font-size:15.5px; color:rgba(255,255,255,.94) !important; max-width:760px; }
.rec.rec-HOLD { --glow:16,185,129; }
.rec.rec-SELL_NOW { --glow:239,68,68; }
.rec.rec-HOLD, .rec.rec-SELL_NOW {
  box-shadow: 0 0 0 1px rgba(var(--glow),.45), 0 0 26px rgba(var(--glow),.45), 0 24px 50px -22px var(--shadow);
  animation: fadeUp .55s ease both, glow 3.6s ease-in-out .6s infinite; }
@keyframes glow {
  0%, 100% { box-shadow: 0 0 0 1px rgba(var(--glow),.45), 0 0 22px rgba(var(--glow),.38), 0 24px 50px -22px var(--shadow); }
  50%      { box-shadow: 0 0 0 1px rgba(var(--glow),.60), 0 0 40px rgba(var(--glow),.60), 0 24px 50px -22px var(--shadow); }
}

/* sidebar widgets on the dark slate background */
section[data-testid="stSidebar"] [data-testid="stExpander"] details {
  background: rgba(255,255,255,.04); border: 1px solid rgba(148,163,184,.30); border-radius: 12px; }
section[data-testid="stSidebar"] [data-testid="stExpander"] details summary,
section[data-testid="stSidebar"] [data-testid="stExpander"] details summary:hover { background: transparent !important; }
section[data-testid="stSidebar"] [data-testid="stExpander"] summary p,
section[data-testid="stSidebar"] [data-testid="stExpander"] summary svg { color:#E2E8F0 !important; fill:#E2E8F0 !important; }
section[data-testid="stSidebar"] .stButton > button {
  background: rgba(255,255,255,.08); border: 1px solid rgba(148,163,184,.45); border-radius: 10px; }
section[data-testid="stSidebar"] .stButton > button:hover { background: rgba(16,185,129,.22); border-color:#10B981; }
section[data-testid="stSidebar"] .stButton > button p { color:#E2E8F0 !important; font-weight:600; }

/* live alerts (sidebar) */
.alert-feed { max-height:300px; overflow-y:auto; padding-right:4px; }
.alert-item { display:flex; gap:10px; padding:9px 0; border-bottom:1px solid rgba(148,163,184,.22); }
.alert-item .dot { width:10px; height:10px; border-radius:50%; margin-top:5px; flex:0 0 10px; animation:pulse 2.2s infinite; }
.alert-item .ttl { font-size:13px; font-weight:700; color:#F1F5F9 !important; }
.alert-item .meta { font-size:12px; color:#94A3B8 !important; }
.alert-item .why { font-size:12px; color:#CBD5E1 !important; margin-top:2px; }
@keyframes pulse { 0% { box-shadow:0 0 0 0 rgba(148,163,184,.55); } 70% { box-shadow:0 0 0 7px rgba(148,163,184,0); } 100% { box-shadow:0 0 0 0 rgba(148,163,184,0); } }
.banner { background:linear-gradient(90deg,#FEF3C7,#FDE68A); border:1px solid #F59E0B; border-radius:14px;
          padding:12px 18px; margin-bottom:16px; font-weight:600; color:#92400E !important; }

/* root cause */
.rc-box { border-radius:18px; padding:20px 24px; margin:6px 0 10px 0; line-height:1.75; font-size:15px;
          background: rgba(255,255,255,.7); border:1px solid rgba(255,255,255,.9);
          box-shadow: 0 12px 30px -16px rgba(15,23,42,.25); color:#1E293B; }
.rc-box .rc-line { padding-left:6px; }
.rc-box .rc-conf { margin-top:8px; font-weight:700; color:#047857; }

/* primary button in brand green */
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"] {
  background:#10B981; border-color:#10B981; color:#fff; font-weight:700; border-radius:12px; padding:.55rem 1.6rem; }
.stButton > button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover { background:#059669; border-color:#059669; }
.stButton > button[kind="primary"] p, [data-testid="stBaseButton-primary"] p,
[data-testid="stMain"] .stButton > button[kind="primary"] p { color:#fff !important; font-weight:700; }
</style>
"""

RTL_CSS = """
<style>
[data-testid="stMain"], section[data-testid="stSidebar"] { direction: rtl; text-align: right; }
[data-testid="stMain"] .stMarkdown, [data-testid="stMain"] label, [data-testid="stMain"] p,
.hero, .hero-xl, .feat, .kpi, .rec, .rc-box, .sec, .note, .banner, .foot, .alert-item,
[data-testid="stExpander"] summary, [data-testid="stCaptionContainer"] { direction: rtl; text-align: right; }
.kpi .v small { margin-left: 0; margin-right: 4px; }
.kpi .v, .kpi .d { unicode-bidi: plaintext; }
.alert-item { flex-direction: row; }
/* Plotly charts stay left-to-right in English so legends, ticks and hover remain readable */
[data-testid="stPlotlyChart"], [data-testid="stPlotlyChart"] *, .js-plotly-plot, .js-plotly-plot * {
  direction: ltr !important; text-align: left; }
/* sliders keep their LTR geometry; only their labels flip */
[data-testid="stSlider"] { direction: ltr; }
[data-testid="stSlider"] label, [data-testid="stSlider"] label p { direction: rtl; text-align: right; }
[data-testid="stDateInput"], [data-testid="stDateInput"] input, [data-testid="stNumberInput"] input { direction: ltr; text-align: left; }
[data-testid="stDateInput"] label, [data-testid="stDateInput"] label p { direction: rtl; text-align: right; }
html, body, [class*="css"], .stApp { font-family: 'Inter', 'Noto Naskh Arabic', sans-serif; line-height: 1.7; }
</style>
"""


V8_CSS = """
<style>
/* ---- Feature 1: live market ticker ------------------------------------- */
.ticker { overflow:hidden; white-space:nowrap; direction:ltr; text-align:left; border-radius:14px; margin-bottom:14px;
          background:linear-gradient(90deg,#0B1220,#0F172A 50%,#0B1220); border:1px solid #1E293B;
          box-shadow:0 12px 28px -16px rgba(15,23,42,.6); position:relative; }
.ticker::before, .ticker::after { content:""; position:absolute; top:0; bottom:0; width:48px; z-index:2; pointer-events:none; }
.ticker::before { left:0;  background:linear-gradient(90deg,#0B1220,rgba(11,18,32,0)); }
.ticker::after  { right:0; background:linear-gradient(270deg,#0B1220,rgba(11,18,32,0)); }
.ticker-track { display:flex; width:max-content; animation:marquee 120s linear infinite; will-change:transform;
                backface-visibility:hidden; }
.ticker:hover .ticker-track { animation-play-state:paused; }
/* two identical halves, each at least one screen wide, so moving by exactly -50% is seamless */
.ticker-group { display:flex; flex:0 0 auto; align-items:center; justify-content:space-around; min-width:100vw; padding:10px 0; }
.tk { display:inline-block; flex:0 0 auto; margin:0 6px; font-size:13px; font-weight:600; color:#E2E8F0; font-variant-numeric:tabular-nums; }
.tk b { color:#F8FAFC; letter-spacing:.02em; }
.tk .px { color:#94A3B8; font-weight:500; margin:0 4px; }
.tk .up { color:#34D399; font-weight:700; }
.tk .dn { color:#F87171; font-weight:700; }
.tk .fl { color:#94A3B8; font-weight:700; }
.tk-sep { color:#334155; margin:0 14px; }
@keyframes marquee { from { transform:translate3d(0,0,0); } to { transform:translate3d(-50%,0,0); } }

/* ---- Feature 3: why this matters --------------------------------------- */
.why-wrap { container-type:inline-size; border-radius:26px; padding:34px 38px 30px 38px; margin:6px 0 22px 0;
            background:linear-gradient(135deg,rgba(255,255,255,.72),rgba(236,253,245,.7));
            -webkit-backdrop-filter:blur(16px) saturate(160%); backdrop-filter:blur(16px) saturate(160%);
            border:1px solid rgba(255,255,255,.9); box-shadow:0 1px 0 rgba(255,255,255,.9) inset, 0 22px 46px -24px rgba(15,23,42,.35); }
.why-title { font-size:13px; font-weight:800; letter-spacing:.16em; text-transform:uppercase; color:#047857; margin-bottom:16px; }
.why-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:20px; }
.why-card { position:relative; overflow:hidden; border-radius:22px; padding:26px 18px 22px 18px; text-align:center;
            background:linear-gradient(160deg,rgba(255,255,255,.92),rgba(255,255,255,.55));
            -webkit-backdrop-filter:blur(18px) saturate(170%); backdrop-filter:blur(18px) saturate(170%);
            border:1px solid rgba(255,255,255,.95);
            box-shadow:0 1px 0 #fff inset, 0 0 0 1px rgba(148,163,184,.18), 0 20px 38px -22px rgba(var(--rgb),.75); }
.why-card::before { content:""; position:absolute; left:0; right:0; top:0; height:5px;
                    background:linear-gradient(90deg,var(--a),var(--b)); }
.why-card::after { content:""; position:absolute; right:-45px; top:-45px; width:150px; height:150px; border-radius:50%;
                   background:rgba(var(--rgb),.13); filter:blur(8px); }
.why-card { animation:fadeUp .7s ease both; }
.why-card:nth-child(2) { animation-delay:.15s; }
.why-card:nth-child(3) { animation-delay:.3s; }
.why-num { display:flex; align-items:baseline; justify-content:center; gap:6px; direction:ltr; min-height:92px; line-height:1.2;
           font-size:clamp(40px,6.6cqw,76px); font-weight:800; letter-spacing:-.04em; color:var(--a); font-variant-numeric:tabular-nums; }
.why-num small { font-size:clamp(14px,2.1cqw,25px); font-weight:800; letter-spacing:0; color:#64748B; white-space:nowrap; }
.why-txt { margin-top:6px; font-size:15px; font-weight:500; line-height:1.55; color:#475569; }
.why-close { margin:22px 0 0 0 !important; text-align:center; font-size:19px; font-weight:700; color:#0F172A !important; }
/* stack the cards when the banner itself is narrow (this accounts for the open sidebar) */
@container (max-width:680px) { .why-grid { grid-template-columns:1fr; } .why-num { font-size:64px; } .why-num small { font-size:22px; } }

/* ---- Feature 5: news cards --------------------------------------------- */
.news-card { border-radius:16px; padding:14px 18px; margin:0 0 10px 0; background:rgba(255,255,255,.78);
             border:1px solid #E2E8F0; border-left:5px solid var(--c,#10B981);
             box-shadow:0 10px 26px -18px rgba(15,23,42,.35); animation:fadeUp .5s ease both; }
.news-card .tag { display:inline-block; font-size:10.5px; font-weight:800; letter-spacing:.1em; padding:2px 9px; border-radius:999px;
                  color:var(--c,#10B981) !important; background:rgba(148,163,184,.16); }
.news-card .hl { font-size:15.5px; font-weight:700; color:#0F172A !important; margin:6px 0 4px 0; line-height:1.4; }
.news-card .meta { font-size:12px; color:#64748B !important; }

/* ---- Feature 4: sidebar chat -------------------------------------------- */
section[data-testid="stSidebar"] [data-testid="stChatMessage"] { background:rgba(255,255,255,.07); border-radius:12px;
  padding:10px; gap:8px; }
section[data-testid="stSidebar"] [data-testid="stChatMessage"] :is(p, li, strong, em) {
  color:#E2E8F0 !important; font-size:13px; line-height:1.55; unicode-bidi:plaintext; }
section[data-testid="stSidebar"] [data-testid="stForm"] { border-color:rgba(148,163,184,.3); border-radius:12px; padding:10px; }
section[data-testid="stSidebar"] .st-key-chat_q [data-testid="stTextInputRootElement"] {
  background:rgba(2,6,23,.55) !important; border:1px solid rgba(16,185,129,.45) !important; border-radius:10px !important; }
section[data-testid="stSidebar"] .st-key-chat_q [data-testid="stTextInputRootElement"] * { background-color:transparent !important; }
section[data-testid="stSidebar"] .st-key-chat_q input { color:#F8FAFC !important; -webkit-text-fill-color:#F8FAFC; caret-color:#10B981; }
section[data-testid="stSidebar"] .st-key-chat_q input::placeholder { color:#94A3B8; -webkit-text-fill-color:#94A3B8; }
section[data-testid="stSidebar"] .st-key-chat_q [data-testid="InputInstructions"] { display:none; }  /* overlaps in a narrow box */
section[data-testid="stSidebar"] [data-testid="stExpander"] :is(.stButton p, input) { unicode-bidi:plaintext; }
section[data-testid="stSidebar"] [data-testid="stFormSubmitButton"] button {
  background:#10B981; border-color:#10B981; border-radius:10px; }
section[data-testid="stSidebar"] [data-testid="stFormSubmitButton"] button p { color:#fff !important; font-weight:700; }

/* ---- Feature 6: polish -------------------------------------------------- */
@keyframes fadeUp { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:none; } }
.hero, .hero-xl, .kpi, .feat, .rc-box, .banner { animation:fadeUp .55s ease both; }
.rec:not(.rec-HOLD):not(.rec-SELL_NOW) { animation:fadeUp .55s ease both; }
.sk { border-radius:16px; background:linear-gradient(90deg,#E2E8F0 25%,#F1F5F9 37%,#E2E8F0 63%); background-size:400% 100%;
      animation:shimmer 1.4s ease infinite; }
@keyframes shimmer { 0% { background-position:100% 50%; } 100% { background-position:0 50%; } }
.sk-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:16px; margin-top:16px; }
.sk-hero { height:96px; } .sk-kpi { height:126px; } .sk-wide { grid-column:span 3; height:330px; } .sk-side { height:330px; }
.mm-footer { text-align:center; color:#64748B !important; font-size:13.5px; font-weight:600; margin:40px 0 24px 0; padding-top:18px;
             border-top:1px solid #E2E8F0; }
.ver-badge { display:inline-block; font-size:11px; font-weight:800; letter-spacing:.12em; color:#A7F3D0; background:rgba(16,185,129,.18);
             border:1px solid rgba(167,243,208,.4); padding:3px 11px; border-radius:999px; }
.pro-err { font-size:12.5px; font-weight:600; color:#FCA5A5 !important; margin:2px 0 6px 0; }
/* Pro password + display-name inputs on the dark sidebar */
section[data-testid="stSidebar"] :is(.st-key-pro_pwd, .st-key-user_name) [data-testid="stTextInputRootElement"] {
  background:rgba(2,6,23,.55) !important; border:1px solid rgba(251,191,36,.38) !important; border-radius:12px !important;
  box-shadow:0 1px 0 rgba(255,255,255,.05) inset; transition:border-color .2s, box-shadow .2s; overflow:hidden; }
section[data-testid="stSidebar"] :is(.st-key-pro_pwd, .st-key-user_name) [data-testid="stTextInputRootElement"]:focus-within {
  border-color:#FBBF24 !important; box-shadow:0 0 0 3px rgba(251,191,36,.18), 0 0 18px rgba(251,191,36,.22); }
section[data-testid="stSidebar"] :is(.st-key-pro_pwd, .st-key-user_name) [data-testid="stTextInputRootElement"] * { background-color:transparent !important; }
section[data-testid="stSidebar"] :is(.st-key-pro_pwd, .st-key-user_name) input { color:#F8FAFC !important; -webkit-text-fill-color:#F8FAFC; caret-color:#FBBF24; letter-spacing:.04em; }
section[data-testid="stSidebar"] :is(.st-key-pro_pwd, .st-key-user_name) button svg { color:#FCD34D !important; fill:#FCD34D; }
section[data-testid="stSidebar"] [data-testid="InputInstructions"] { color:#94A3B8 !important; }
.user-chip { display:flex; align-items:center; gap:11px; padding:10px 12px; margin:0 0 4px 0; border-radius:14px;
             background:rgba(255,255,255,.05); border:1px solid rgba(148,163,184,.28); }
.user-chip.is-pro { border-color:rgba(251,191,36,.55); box-shadow:0 0 20px -6px rgba(251,191,36,.5); }
.user-chip .av { flex:0 0 36px; width:36px; height:36px; border-radius:50%; display:flex; align-items:center; justify-content:center;
                 font-weight:800; color:#fff; background:linear-gradient(135deg,#047857,#10B981); }
.user-chip.is-pro .av { color:#422006; background:linear-gradient(135deg,#FDE68A,#F59E0B); }
.user-chip .who { min-width:0; flex:1; }
.user-chip .row { display:flex; align-items:center; gap:8px; }
.user-chip .name { font-size:14px; font-weight:700; color:#F8FAFC; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.user-chip .plan { font-size:11.5px; color:#94A3B8; }
.gold-pro { flex:0 0 auto; font-size:10.5px; font-weight:900; letter-spacing:.12em; color:#422006; padding:2px 8px; border-radius:6px;
            background:linear-gradient(110deg,#B45309,#FBBF24 28%,#FEF3C7 48%,#FBBF24 68%,#D97706);
            box-shadow:0 0 12px rgba(251,191,36,.65); }
.chlog { margin:0; padding-left:18px; font-size:12.5px; line-height:1.6; color:#CBD5E1; }
.to-top-fallback { position:fixed; left:22px; bottom:22px; z-index:9999; width:44px; height:44px; border-radius:50%; background:#10B981;
                   color:#fff !important; font-size:20px; font-weight:700; text-align:center; line-height:44px; text-decoration:none; }
@media (prefers-reduced-motion: reduce) {
  .ticker-track, .hero, .hero-xl, .kpi, .feat, .rc-box, .rec, .banner, .why-card, .news-card, .sk { animation:none !important; }
}
</style>
"""

V8_RTL_CSS = """
<style>
.ticker, .ticker * { direction:ltr !important; text-align:left; }
.why-wrap, .why-card, .news-card, .mm-footer { direction:rtl; text-align:right; }
.news-card { border-left:1px solid #E2E8F0; border-right:5px solid var(--c,#10B981); }
.why-num { direction:rtl; }
/* toggle switches keep left-to-right geometry, otherwise the knob slides out of its track */
[data-testid="stCheckbox"] label > div:not([data-testid="stWidgetLabel"]) { direction:ltr; }
.why-card, .why-close { text-align:center; }
.user-chip { direction:rtl; text-align:right; }
section[data-testid="stSidebar"] .st-key-chat_q input { direction:ltr; text-align:left; }  /* questions are English */
</style>
"""


def inject_css() -> None:
    """
    Inject the design-system CSS, plus the RTL layer when Urdu is selected.

    Returns
    -------
    None
    """
    st.markdown(BASE_CSS, unsafe_allow_html=True)
    st.markdown(V8_CSS, unsafe_allow_html=True)
    if get_lang() == "ur":
        st.markdown(RTL_CSS, unsafe_allow_html=True)
        st.markdown(V8_RTL_CSS, unsafe_allow_html=True)


# ============================================================================
# UI HELPERS
# ============================================================================
def show(fig: go.Figure, key: str | None = None, config: dict[str, Any] | None = None) -> None:
    """
    Render a Plotly figure with the uniform, clean toolbar on old and new Streamlit.

    Parameters
    ----------
    fig : go.Figure
        Figure to draw.
    key : str, optional
        Unique element key (needed when two identical charts are on one page).
    config : dict, optional
        Plotly config; defaults to ``CHART_CONFIG``.

    Returns
    -------
    None
    """
    cfg = config or CHART_CONFIG
    if "width" in inspect.signature(st.plotly_chart).parameters:
        st.plotly_chart(fig, width="stretch", config=cfg, key=key)
    else:  # older Streamlit
        st.plotly_chart(fig, use_container_width=True, config=cfg, key=key)


def style_plotly(fig: go.Figure, height: int = 400, legend: bool = True, hover: str = "x unified") -> go.Figure:
    """
    Apply one consistent, gridline-free, Inter-typeset look to every chart.

    Parameters
    ----------
    fig : go.Figure
        Figure to style (modified in place).
    height : int
        Chart height in pixels.
    legend : bool
        Whether to show the legend.
    hover : str
        Plotly hover mode.

    Returns
    -------
    go.Figure
        The same figure, for chaining.
    """
    fig.update_layout(
        height=height, hovermode=hover, showlegend=legend,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12, color="#334155"),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
                    title_text="", bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=INK, font=dict(family="Inter, sans-serif", color="#fff"), bordercolor=INK),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showline=True, linecolor=GRID, ticks="outside",
                     tickcolor=GRID)
    fig.update_yaxes(showgrid=False, zeroline=False, showline=False)
    return fig


def kpi(col: Any, title: str, value: str, delta: str, color: str = GREEN) -> None:
    """
    Draw one glassmorphism KPI card in a Streamlit column.

    Parameters
    ----------
    col : Any
        Streamlit column (or any object with ``markdown``).
    title, value, delta : str
        Card title, headline value (HTML allowed) and sub-line.
    color : str
        Accent colour.

    Returns
    -------
    None
    """
    col.markdown(f'<div class="kpi" style="--c:{color}"><div class="t">{title}</div>'
                 f'<div class="v">{value}</div><div class="d">{delta}</div></div>', unsafe_allow_html=True)


def sec(text: str) -> None:
    """
    Render a section heading.

    Parameters
    ----------
    text : str
        Heading text.

    Returns
    -------
    None
    """
    st.markdown(f'<div class="sec">{text}</div>', unsafe_allow_html=True)


def note(text: str) -> None:
    """
    Render a small grey caption under a chart.

    Parameters
    ----------
    text : str
        Caption text.

    Returns
    -------
    None
    """
    st.markdown(f'<p class="note">{text}</p>', unsafe_allow_html=True)


def fmt(x: Any) -> str:
    """
    Format a price with thousands separators.

    Parameters
    ----------
    x : Any
        Number or None.

    Returns
    -------
    str
        ``"1,234"`` or the localized "n/a".
    """
    try:
        v = float(x)
    except (TypeError, ValueError):
        return T("na")
    return T("na") if not np.isfinite(v) else f"{v:,.0f}"


def pct_str(x: Any, digits: int = 1) -> str:
    """
    Format a percentage change with a sign.

    Parameters
    ----------
    x : Any
        Percentage value (e.g. ``3.2`` for 3.2%).
    digits : int
        Decimal places.

    Returns
    -------
    str
        ``"+3.2%"`` or the localized "n/a".
    """
    try:
        v = float(x)
    except (TypeError, ValueError):
        return T("na")
    return T("na") if not np.isfinite(v) else f"{v:+.{digits}f}%"


def snap(series: pd.Series, dates: Any, tol_days: int = 20) -> pd.Series:
    """
    Map anomaly dates onto the nearest date that exists in a plotted series.

    Parameters
    ----------
    series : pd.Series
        Plotted series indexed by date.
    dates : array-like
        Anomaly dates.
    tol_days : int
        Maximum distance (days) for a match.

    Returns
    -------
    pd.Series
        The subset of ``series`` closest to the anomaly dates.
    """
    if series.empty or len(dates) == 0:
        return series.iloc[0:0]
    pos = series.index.get_indexer(pd.DatetimeIndex(dates), method="nearest",
                                   tolerance=pd.Timedelta(days=tol_days))
    pos = np.unique(pos[pos >= 0])
    return series.iloc[pos]


def sanitize_choice(key: str, options: list[Any], multi: bool = False) -> None:
    """
    Drop a stale widget value from session state if it is no longer a valid option.

    Parameters
    ----------
    key : str
        Widget key.
    options : list
        Currently valid options.
    multi : bool
        True for multiselect (value is a list).

    Returns
    -------
    None
    """
    if key not in st.session_state:
        return
    val = st.session_state[key]
    if multi:
        st.session_state[key] = [v for v in (val or []) if v in options]
    elif val not in options:
        del st.session_state[key]


def goto_page(page_key: str) -> None:
    """
    Callback that switches the navigation radio to another page.

    Parameters
    ----------
    page_key : str
        One of ``PAGES``.

    Returns
    -------
    None
    """
    st.session_state["page"] = page_key


def bump_alerts() -> None:
    """
    Callback for the Live Alerts Refresh button: invalidate the anomalies cache.

    Returns
    -------
    None
    """
    st.session_state["alerts_n"] = int(st.session_state.get("alerts_n", 0)) + 1
    st.session_state["alerts_ts"] = datetime.now().strftime("%H:%M:%S")


# ============================================================================
# DATA LAYER (all cached)
# ============================================================================
def _read_optional_parquet(name: str) -> pd.DataFrame:
    """
    Read an optional parquet file, returning an empty frame on any failure.

    Parameters
    ----------
    name : str
        File name inside ``data/processed``.

    Returns
    -------
    pd.DataFrame
        The table, or an empty DataFrame.
    """
    try:
        return pd.read_parquet(PROCESSED_DIR / name)
    except Exception:
        return pd.DataFrame()


def _read_metrics() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """
    Read models/metrics.json defensively.

    Returns
    -------
    tuple
        ``(per_commodity_metrics, meta)``; both empty if the file is missing or invalid.
    """
    try:
        raw = json.loads((MODELS_DIR / "metrics.json").read_text(encoding="utf-8"))
    except Exception:
        return {}, {}
    if not isinstance(raw, dict):
        return {}, {}
    meta = raw.get("__meta__", {})
    metrics = {k: v for k, v in raw.items() if isinstance(v, dict) and not str(k).startswith("__")}
    return metrics, meta if isinstance(meta, dict) else {}


@st.cache_data(show_spinner=False)
def load_core() -> dict[str, Any]:
    """
    Load and feature-engineer the price data, district profiles and model metrics.

    Returns
    -------
    dict
        Keys: ``df`` (features incl. each series' latest period), ``latest`` (rows to
        score for the next period), ``profiles``, ``metrics`` and ``meta``.

    Raises
    ------
    FileNotFoundError
        If ``analysis_table.parquet`` does not exist.
    ValueError
        If required columns are missing.
    """
    path = PROCESSED_DIR / "analysis_table.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    raw = pd.read_parquet(path)
    missing = {"date", "commodity_name", PRICE} - set(raw.columns)
    if missing:
        raise ValueError(f"analysis_table.parquet is missing columns: {sorted(missing)}")
    raw = raw.dropna(subset=["date", "commodity_name"]).copy()
    raw["date"] = pd.to_datetime(raw["date"]).astype("datetime64[ns]")
    raw[PRICE] = pd.to_numeric(raw[PRICE], errors="coerce")
    raw = raw.dropna(subset=[PRICE])

    df = build_features(raw, for_inference=True)
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
    metrics, meta = _read_metrics()
    return {
        "df": df,
        "latest": select_latest_rows(df),
        "profiles": _read_optional_parquet("district_profiles.parquet"),
        "metrics": metrics,
        "meta": meta,
    }


def _empty_anomalies() -> pd.DataFrame:
    """
    Build an empty anomalies table with the standard columns.

    Returns
    -------
    pd.DataFrame
        Empty frame with ``date``, ``commodity_name``, ``pct_change``, ``direction``
        and ``obs_date`` columns.
    """
    return pd.DataFrame({
        "date": pd.Series(dtype="datetime64[ns]"),
        "commodity_name": pd.Series(dtype="object"),
        "pct_change": pd.Series(dtype="float64"),
        "direction": pd.Series(dtype="object"),
        "obs_date": pd.Series(dtype="datetime64[ns]"),
    })


def alert_token() -> tuple[float, int]:
    """
    Cache key for the anomalies loader: file modification time + refresh counter.

    Returns
    -------
    tuple
        ``(mtime, refresh_count)``. A new value forces ``load_anomalies`` to re-read.
    """
    try:
        mtime = (PROCESSED_DIR / "anomalies.parquet").stat().st_mtime
    except OSError:
        mtime = 0.0
    return (float(mtime), int(st.session_state.get("alerts_n", 0)))


@st.cache_data(show_spinner=False)
def load_anomalies(token: tuple[float, int]) -> pd.DataFrame:
    """
    Read anomalies.parquet and attach % change, direction and the matching price date.

    Parameters
    ----------
    token : tuple
        Cache key from ``alert_token()``; changing it re-reads the file.

    Returns
    -------
    pd.DataFrame
        Anomalies with ``pct_change`` (period % change of the district's price, or the crop's average
        when the table has no districts),
        ``direction`` (``"spike"``/``"drop"``) and ``obs_date`` (nearest price date).
        Empty (never raises) if the file is missing or malformed.
    """
    try:
        raw = pd.read_parquet(PROCESSED_DIR / "anomalies.parquet")
    except Exception:
        return _empty_anomalies()
    if raw.empty:
        return _empty_anomalies()

    raw = raw.rename(columns={"commodity": "commodity_name", "crop": "commodity_name",
                              "ds": "date", "timestamp": "date"})
    raw = raw.loc[:, ~raw.columns.duplicated()]
    if not {"date", "commodity_name"} <= set(raw.columns):
        return _empty_anomalies()
    for flag in ("is_anomaly", "anomaly", "is_supply_shock"):
        if flag in raw.columns:
            try:
                raw = raw[raw[flag].fillna(0).astype(bool)]
            except Exception:
                pass
            break
    raw = raw.dropna(subset=["date", "commodity_name"]).copy()
    if raw.empty:
        return _empty_anomalies()
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").astype("datetime64[ns]")
    raw = raw.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    try:
        df = load_core()["df"]
        # district-level anomalies are measured against their own district's series
        keys = ["commodity_name"]
        if "admin2_name" in raw.columns and "admin2_name" in df.columns:
            keys.append("admin2_name")
        by_series = df.groupby(keys + ["date"])[PRICE].mean().reset_index().sort_values("date")
        by_series["pct_change"] = (by_series.groupby(keys)[PRICE].pct_change(fill_method=None)
                                   .replace([np.inf, -np.inf], np.nan) * 100)
        by_series["obs_date"] = by_series["date"]
        lookup = by_series[keys + ["date", "pct_change", "obs_date"]].rename(columns={"date": "_key"})
        raw["_key"] = raw["date"]
        raw = raw.drop(columns=[c for c in ("pct_change", "obs_date") if c in raw.columns])
        raw = pd.merge_asof(raw.sort_values("_key"), lookup.sort_values("_key"), on="_key",
                            by=keys, direction="nearest", tolerance=pd.Timedelta(days=20))
        raw = raw.drop(columns="_key")
    except Exception:
        raw["pct_change"] = np.nan
        raw["obs_date"] = raw["date"]

    if "price_pct_change_1" in raw.columns:  # fall back to the table's own figure
        own = pd.to_numeric(raw["price_pct_change_1"], errors="coerce") * 100
        raw["pct_change"] = raw["pct_change"].fillna(own)
    raw["obs_date"] = pd.to_datetime(raw["obs_date"]).fillna(raw["date"]).astype("datetime64[ns]")
    raw["direction"] = np.where(raw["pct_change"] < 0, "drop", "spike")
    return raw.sort_values("date").reset_index(drop=True)


def district_of(row: Any) -> str:
    """
    Return an anomaly row's district name, or an empty string when unknown.

    Parameters
    ----------
    row : Any
        A row (Series or namedtuple-like) that may have ``admin2_name``.

    Returns
    -------
    str
        District name or ``""``.
    """
    val = row.get("admin2_name") if hasattr(row, "get") else None
    return "" if val is None or (isinstance(val, float) and np.isnan(val)) or pd.isna(val) else str(val)


def filter_prices(df: pd.DataFrame, d0: pd.Timestamp, d1: pd.Timestamp, districts: list[str]) -> pd.DataFrame:
    """
    Apply the global date and district filters to the price table.

    Parameters
    ----------
    df : pd.DataFrame
        Feature table.
    d0, d1 : pd.Timestamp
        Inclusive date bounds.
    districts : list of str
        Selected districts (empty list = no district filter).

    Returns
    -------
    pd.DataFrame
        Filtered copy-safe view.
    """
    out = df[(df["date"] >= d0) & (df["date"] <= d1)]
    if districts and "admin2_name" in out.columns:
        out = out[out["admin2_name"].isin(districts)]
    return out


def filter_anomalies(an: pd.DataFrame, d0: pd.Timestamp, d1: pd.Timestamp, districts: list[str]) -> pd.DataFrame:
    """
    Apply the global date and district filters to the anomalies table.

    Parameters
    ----------
    an : pd.DataFrame
        Anomalies table.
    d0, d1 : pd.Timestamp
        Inclusive date bounds.
    districts : list of str
        Selected districts.

    Returns
    -------
    pd.DataFrame
        Filtered anomalies (empty-safe).
    """
    if an.empty or "date" not in an.columns:
        return an.copy()
    out = an[(an["date"] >= d0) & (an["date"] <= d1)]
    if districts and "admin2_name" in out.columns:
        out = out[out["admin2_name"].isin(districts)]
    return out


def crop_series(df: pd.DataFrame, crop: str) -> pd.Series:
    """
    Average price per period for one commodity (positive, non-null values only).

    Parameters
    ----------
    df : pd.DataFrame
        Feature table (already filtered).
    crop : str
        Commodity name.

    Returns
    -------
    pd.Series
        Price series indexed by date.
    """
    s = df.loc[df["commodity_name"] == crop].groupby("date")[PRICE].mean().sort_index().dropna()
    return s[s > 0]


def data_token(df_all: pd.DataFrame) -> str:
    """
    Fingerprint of the loaded data, used as a cache key for derived results.

    Parameters
    ----------
    df_all : pd.DataFrame
        Full feature table.

    Returns
    -------
    str
        Row count and last date.
    """
    return f"{len(df_all)}|{df_all['date'].max():%Y%m%d}"


# ============================================================================
# FORECAST ENGINE (4 auditable models, walk-forward backtest, fully cached)
# ============================================================================
def _naive(h: list[float]) -> float:
    """Last observed value (numpydoc: h = history list; returns float)."""
    return h[-1]


def _ma3(h: list[float]) -> float:
    """Mean of the last three values."""
    return float(np.mean(h[-3:]))


def _drift(h: list[float]) -> float:
    """Last value plus half of the recent 3-period average drift."""
    k = min(3, len(h) - 1)
    return h[-1] + (h[-1] - h[-1 - k]) / k * 0.5


def _seasonal(h: list[float], p: int = 12) -> float:
    """Value one season (``p`` periods) ago, or the last value if history is short."""
    return h[-p] if len(h) >= p else h[-1]


MODELS: dict[str, Callable[[list[float]], float]] = {
    "Naive": _naive, "MA3": _ma3, "Drift": _drift, "Seasonal": _seasonal,
}


def backtest(y: pd.Series) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray], pd.Index, np.ndarray]:
    """
    Walk-forward backtest of all baseline models on the last 25% of the series.

    Parameters
    ----------
    y : pd.Series
        Price series indexed by date (at least ``MIN_HISTORY`` points).

    Returns
    -------
    tuple
        ``(leaderboard, errors_by_model, predictions_by_model, test_index, test_actuals)``.
        The leaderboard has columns Model, MAE, MAPE and is sorted by MAE.
    """
    vals = y.values.astype(float)
    n_test = max(6, int(len(vals) * 0.25))
    start = len(vals) - n_test
    rows: list[dict[str, Any]] = []
    errs: dict[str, np.ndarray] = {}
    preds: dict[str, np.ndarray] = {}
    for name, fn in MODELS.items():
        p = np.array([fn(list(vals[:i])) for i in range(start, len(vals))])
        e = vals[start:] - p
        errs[name], preds[name] = e, p
        rows.append({"Model": name, "MAE": float(np.mean(np.abs(e))),
                     "MAPE": float(np.mean(np.abs(e) / np.maximum(vals[start:], 1e-9)) * 100)})
    board = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    return board, errs, preds, y.index[start:], vals[start:]


def make_forecast(y: pd.Series, h: int) -> tuple[pd.DataFrame, pd.DataFrame, str, float, pd.DataFrame]:
    """
    Pick the best backtested model and roll it forward ``h`` periods with an 80% band.

    Parameters
    ----------
    y : pd.Series
        Price series indexed by date.
    h : int
        Forecast horizon in periods.

    Returns
    -------
    tuple
        ``(forecast_df, leaderboard, best_model_name, sigma, backtest_df)``. ``forecast_df``
        has columns date, forecast, lo, hi.
    """
    board, errs, preds, t_idx, t_act = backtest(y)
    best = str(board.iloc[0]["Model"])
    fn = MODELS[best]
    sigma = float(np.std(errs[best])) if len(errs[best]) > 1 else float(y.std())
    if not np.isfinite(sigma):
        sigma = 0.0
    hist, out = list(y.values.astype(float)), []
    for _ in range(h):
        out.append(fn(hist))
        hist.append(out[-1])
    steps = np.arange(1, h + 1)
    band = 1.2816 * sigma * np.sqrt(steps)  # 80% interval
    sd = y.index.to_series().diff().dt.days.median()
    if pd.notna(sd) and 25 <= sd <= 35:
        dates = [y.index[-1] + pd.DateOffset(months=int(i)) for i in steps]
    else:
        dates = [y.index[-1] + pd.Timedelta(days=float(sd if pd.notna(sd) and sd else 30) * int(i)) for i in steps]
    fc = pd.DataFrame({"date": dates, "forecast": out, "lo": np.array(out) - band, "hi": np.array(out) + band})
    bt = pd.DataFrame({"date": t_idx, "actual": t_act, "pred": preds[best]})
    return fc, board, best, sigma, bt


@st.cache_data(show_spinner="Running walk-forward backtest...")
def forecast_bundle(crop: str, horizon: int, d0_iso: str, d1_iso: str,
                    districts: tuple[str, ...], token: str) -> dict[str, Any] | None:
    """
    Cached forecast engine: filter, backtest, forecast. Reused by every page.

    Because the arguments are plain values, switching between tabs (or changing an
    unrelated widget) hits the cache and never re-runs the backtest.

    Parameters
    ----------
    crop : str
        Commodity name.
    horizon : int
        Forecast horizon in periods.
    d0_iso, d1_iso : str
        ISO dates of the global filter window.
    districts : tuple of str
        Selected districts.
    token : str
        Data fingerprint (``data_token``) so the cache refreshes when data changes.

    Returns
    -------
    dict or None
        Keys ``y``, ``fc``, ``board``, ``best``, ``sigma``, ``bt``; None if the series is
        shorter than ``MIN_HISTORY``.
    """
    df = filter_prices(load_core()["df"], pd.Timestamp(d0_iso), pd.Timestamp(d1_iso), list(districts))
    y = crop_series(df, crop)
    if len(y) < MIN_HISTORY:
        return None
    fc, board, best, sigma, bt = make_forecast(y, horizon)
    return {"y": y, "fc": fc, "board": board, "best": best, "sigma": sigma, "bt": bt}


@st.cache_resource(show_spinner=False)
def load_model(crop: str) -> Any | None:
    """
    Load a commodity's trained XGBoost model once per server process.

    Parameters
    ----------
    crop : str
        Commodity name.

    Returns
    -------
    Any or None
        The unpickled model, or None if the file or xgboost is unavailable.
    """
    path = MODELS_DIR / "per_commodity" / f"{model_filename(crop)}.pkl"
    if not path.exists():
        return None
    try:
        import joblib

        return joblib.load(path)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def ml_next_estimate(crop: str, token: str) -> dict[str, float] | None:
    """
    Score the latest data with the commodity's XGBoost model (next-period price).

    Parameters
    ----------
    crop : str
        Commodity name.
    token : str
        Data fingerprint for cache invalidation.

    Returns
    -------
    dict or None
        ``{"price", "now", "chg_pct"}`` (averaged over districts), or None when the
        model/features are unavailable.
    """
    model = load_model(crop)
    latest = load_core()["latest"]
    if model is None or latest.empty:
        return None
    rows = latest[latest["commodity_name"] == crop]
    if rows.empty:
        return None
    try:
        X = rows.reindex(columns=MODEL_FEATURE_COLS)
        pred_ret = np.asarray(model.predict(X), dtype=float)
        now = rows[PRICE].to_numpy(dtype=float)
        est = float(np.nanmean(now * np.exp(pred_ret)))
        cur = float(np.nanmean(now))
    except Exception:
        return None
    if not (np.isfinite(est) and np.isfinite(cur) and cur > 0):
        return None
    return {"price": est, "now": cur, "chg_pct": (est / cur - 1) * 100}


# ============================================================================
# ROOT CAUSE AI (simulated causal attribution, numpy/pandas only)
# ============================================================================
@st.cache_data(show_spinner=False)
def root_cause_analysis(crop: str, date_iso: str, district: str = "") -> dict[str, Any] | None:
    """
    SHAP-style attribution of one price move to candidate drivers.

    For each driver the z-score of its value at the anomaly date versus its own
    history is computed. A driver "aligns" with the move when it pushes in the
    observed direction (e.g. high rainfall and a price spike). Aligned strengths
    are normalised, together with a small residual ("other local factors"), into
    impact weights that sum to 100%. Momentum uses the PREVIOUS period's change so
    the move being explained is never used to explain itself.

    Parameters
    ----------
    crop : str
        Commodity name.
    date_iso : str
        ISO date of the anomaly (snapped to the nearest price date).
    district : str, default ""
        District to analyse; empty means the crop's average over all districts.

    Returns
    -------
    dict or None
        Keys ``date``, ``crop``, ``pct``, ``drivers`` (top 3 with ``label``, ``weight``,
        ``z``), ``confidence`` and ``n_available``; None if there is not enough data.
    """
    df = load_core()["df"]
    d = df[df["commodity_name"] == crop]
    if district and "admin2_name" in d.columns:
        dd = d[d["admin2_name"] == district]
        if not dd.empty:
            d = dd
    if d.empty:
        return None
    cols = [c for _, c in DRIVER_SPECS if c in d.columns]
    agg = d.groupby("date")[cols + [PRICE]].mean().sort_index()
    if len(agg) < 6:
        return None
    agg["_pct"] = agg[PRICE].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan) * 100
    pos = int(agg.index.get_indexer([pd.Timestamp(date_iso)], method="nearest")[0])
    if pos < 1:
        return None
    pct = float(agg["_pct"].iloc[pos])
    if not np.isfinite(pct):
        return None
    direction = 1 if pct >= 0 else -1

    pool: list[tuple[str, float, int, float]] = []  # (key, aligned strength, sign of z, z)
    n_avail = 0
    for key, col in DRIVER_SPECS:
        if col not in agg.columns:
            continue
        s = agg[col].astype(float)
        if key == "momentum":
            s = s.shift(1)  # momentum from the previous period
        val, mu, sd = s.iloc[pos], s.mean(), s.std(ddof=0)
        if not (np.isfinite(val) and np.isfinite(mu) and np.isfinite(sd)) or sd < 1e-12:
            continue
        n_avail += 1
        z = float(np.clip((val - mu) / sd, -4, 4))
        aligned = max(0.0, z * direction)
        if aligned > 0.05:
            pool.append((key, aligned, 1 if z >= 0 else -1, z))
    pool.append(("other", RESIDUAL_WEIGHT, direction, 0.0))
    total = sum(p[1] for p in pool)
    ranked = sorted(pool, key=lambda p: -p[1])
    drivers = [{
        "key": k,
        "label": "drv_other" if k == "other" else f"drv_{k}_{'up' if sign > 0 else 'down'}",
        "weight": w / total * 100,
        "z": z,
    } for k, w, sign, z in ranked[:3]]
    other_share = next(w for k, w, _, _ in pool if k == "other") / total
    conf = 100 * float(np.clip(0.15 + 0.65 * (1 - other_share) + 0.15 * (n_avail / len(DRIVER_SPECS)), 0.20, 0.95))
    return {"date": agg.index[pos].isoformat(), "crop": crop, "pct": pct, "drivers": drivers,
            "confidence": int(round(conf)), "n_available": n_avail}


def explanation_html(res: dict[str, Any], lang: str | None = None) -> str:
    """
    Render a root-cause result as the natural-language explanation.

    Parameters
    ----------
    res : dict
        Output of ``root_cause_analysis``.
    lang : str, optional
        Language code; defaults to the active one.

    Returns
    -------
    str
        HTML block: "On {date}, {crop} prices changed by {pct}%. The most likely drivers
        were: 1. ... Confidence: {c}%".
    """
    when = pd.Timestamp(res["date"]).strftime("%d %b %Y")
    lines = [f'<div>{T("rc_head", lang, date=nm(when), crop=res["crop"], pct=nm(pct_str(res["pct"])))}</div>']
    for i, drv in enumerate(res["drivers"], 1):
        line = T("rc_line", lang, i=i, name=T(drv["label"], lang), w=nm(f'{drv["weight"]:.0f}'))
        lines.append(f'<div class="rc-line">{line}</div>')
    lines.append(f'<div class="rc-conf">{T("rc_conf", lang, c=nm(res["confidence"]))}</div>')
    return '<div class="rc-box">' + "".join(lines) + "</div>"


# ============================================================================
# ADVISOR LOGIC
# ============================================================================
def compute_advice(y: pd.Series, fc: pd.DataFrame, board: pd.DataFrame, sigma: float,
                   horizon: int, storage: float) -> dict[str, Any]:
    """
    Turn a forecast into a HOLD / SELL NOW / SELL IN TRANCHES recommendation.

    Parameters
    ----------
    y : pd.Series
        Price history.
    fc : pd.DataFrame
        Forecast frame from ``make_forecast``.
    board : pd.DataFrame
        Backtest leaderboard (winner first).
    sigma : float
        Winning model's backtest error standard deviation.
    horizon : int
        Hold window in periods.
    storage : float
        Storage + financing cost, % per period.

    Returns
    -------
    dict
        Prices, gross/net change, error band, MAPE, z-score, momentum, confidence and
        ``rec`` in {"HOLD", "SELL_NOW", "TRANCHES"}.
    """
    now_px, exp_px = float(y.iloc[-1]), float(fc["forecast"].iloc[-1])
    gross = (exp_px / now_px - 1) * 100 if now_px > 0 else float("nan")
    net = gross - storage * horizon
    err_pct = sigma / now_px * 100 * np.sqrt(horizon) if now_px > 0 else float("nan")
    mape = float(board.iloc[0]["MAPE"])
    std = float(y.std())
    z = (now_px - float(y.mean())) / std if std and np.isfinite(std) else 0.0
    mom = (now_px / float(y.iloc[-4]) - 1) * 100 if len(y) > 4 and float(y.iloc[-4]) > 0 else 0.0
    conf = int(np.clip(100 * (1 - mape / 20), 10, 95)) if np.isfinite(mape) else 10
    if net > err_pct:
        rec = "HOLD"
    elif net < -0.5 * err_pct:
        rec = "SELL_NOW"
    else:
        rec = "TRANCHES"
    return {"now_px": now_px, "exp_px": exp_px, "gross": gross, "net": net, "err_pct": err_pct,
            "mape": mape, "z": z, "mom": mom, "conf": conf, "rec": rec}


def advice_reason(adv: dict[str, Any], lang: str | None = None, wrap: Callable[[str], str] = lambda s: s) -> str:
    """
    Plain-language reason for a recommendation.

    Parameters
    ----------
    adv : dict
        Output of ``compute_advice``.
    lang : str, optional
        Language code; defaults to the active one.
    wrap : callable
        Applied to numbers (use ``nm`` for HTML so RTL text keeps signs in order).

    Returns
    -------
    str
        One-paragraph reason.
    """
    key = {"HOLD": "why_hold", "SELL_NOW": "why_sell_now", "TRANCHES": "why_tranches"}[adv["rec"]]
    return T(key, lang, net=wrap(pct_str(adv["net"])), err=wrap(f'{adv["err_pct"]:.1f}%'))


def rec_style(rec: str) -> tuple[str, str, str]:
    """
    Visual style of a recommendation.

    Parameters
    ----------
    rec : str
        ``"HOLD"``, ``"SELL_NOW"`` or ``"TRANCHES"``.

    Returns
    -------
    tuple
        ``(translation_key, css_gradient, shadow_rgba)``.
    """
    return {
        "HOLD": ("rec_hold", "linear-gradient(135deg,#047857,#10B981)", "rgba(16,185,129,.6)"),
        "SELL_NOW": ("rec_sell_now", "linear-gradient(135deg,#B91C1C,#EF4444)", "rgba(239,68,68,.6)"),
        "TRANCHES": ("rec_tranches", "linear-gradient(135deg,#B45309,#F59E0B)", "rgba(245,158,11,.6)"),
    }[rec]


# ============================================================================
# WHAT-IF SIMULATOR
# ============================================================================
@st.cache_data(show_spinner=False)
def ramadan_lift(crop: str, token: str) -> float:
    """
    Estimate a crop's typical Ramadan price lift from its own history.

    Compares the average period-on-period price change inside Ramadan windows with
    the change outside them, clipped to a sensible 2%-15% range. Falls back to 5%.

    Parameters
    ----------
    crop : str
        Commodity name.
    token : str
        Data fingerprint for cache invalidation.

    Returns
    -------
    float
        Lift as a fraction (0.05 = +5%).
    """
    df = load_core()["df"]
    d = df[df["commodity_name"] == crop]
    if d.empty or "is_ramadan_window" not in d.columns:
        return 0.05
    agg = d.groupby("date").agg(price=(PRICE, "mean"), ram=("is_ramadan_window", "mean")).sort_index()
    pct = agg["price"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    inside, outside = pct[agg["ram"] > 0.5].dropna(), pct[agg["ram"] <= 0.5].dropna()
    if len(inside) < 2 or len(outside) < 2:
        return 0.05
    lift = float(inside.mean() - outside.mean())
    return float(np.clip(lift, 0.02, 0.15)) if np.isfinite(lift) else 0.05


def simulate_scenario(fc: pd.DataFrame, rain: float, diesel: float, fx: float,
                      ramadan_on: bool, ramadan_effect: float) -> dict[str, Any]:
    """
    Apply scenario multipliers to a baseline forecast.

    ``price_impact = 0.15*rain + 0.05*diesel + 0.10*fx (+ Ramadan lift)`` with all
    inputs as fractions. The impact builds up linearly across the horizon.

    Parameters
    ----------
    fc : pd.DataFrame
        Baseline forecast (date, forecast, lo, hi).
    rain, diesel, fx : float
        Scenario changes in percent (e.g. ``25`` = +25%).
    ramadan_on : bool
        Whether the Ramadan demand effect is switched on.
    ramadan_effect : float
        Ramadan lift as a fraction.

    Returns
    -------
    dict
        ``sim`` (forecast frame with ``baseline`` and ``simulated``), ``parts`` (impact
        per driver, as fractions) and ``total`` (sum, as a fraction).
    """
    parts = {
        "rain": WHATIF_COEF["rain"] * rain / 100.0,
        "diesel": WHATIF_COEF["diesel"] * diesel / 100.0,
        "fx": WHATIF_COEF["fx"] * fx / 100.0,
        "ramadan": float(ramadan_effect) if ramadan_on else 0.0,
    }
    total = float(sum(parts.values()))
    h = len(fc)
    ramp = np.arange(1, h + 1) / max(h, 1)
    sim = pd.DataFrame({"date": fc["date"].values, "baseline": fc["forecast"].values})
    sim["simulated"] = sim["baseline"] * (1 + total * ramp)
    return {"sim": sim, "parts": parts, "total": total}


# ============================================================================
# EXPORTS (cached): PDF + Excel
# ============================================================================
def _style_workbook(wb: Any) -> None:
    """
    Apply a clean header style, widths, freeze panes and number formats to a workbook.

    Parameters
    ----------
    wb : openpyxl.Workbook
        Workbook modified in place.

    Returns
    -------
    None
    """
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    fill = PatternFill("solid", fgColor="10B981")
    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        for cell in ws[1]:
            if cell.value is not None:
                cell.font = Font(name="Arial", bold=True, color="FFFFFF")
                cell.fill = fill
                cell.alignment = Alignment(horizontal="center", vertical="center")
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                if isinstance(cell.value, float):
                    cell.number_format = "#,##0.00"
                elif isinstance(cell.value, datetime):
                    cell.number_format = "yyyy-mm-dd"
        for idx, col in enumerate(ws.columns, 1):
            longest = max((len(str(c.value)) for c in list(col)[:300] if c.value is not None), default=8)
            ws.column_dimensions[get_column_letter(idx)].width = min(max(longest + 2, 10), 42)


def _or_placeholder(frame: pd.DataFrame, text: str = "No data available") -> pd.DataFrame:
    """
    Replace an empty table by a one-cell placeholder so the Excel sheet is never blank.

    Parameters
    ----------
    frame : pd.DataFrame
        Table to write.
    text : str
        Placeholder message.

    Returns
    -------
    pd.DataFrame
        The frame itself, or a one-row note when it is empty.
    """
    return frame if not frame.empty else pd.DataFrame({"info": [text]})


@st.cache_data(show_spinner=False)
def build_excel_report(crop: str, horizon: int, d0_iso: str, d1_iso: str,
                       districts: tuple[str, ...], token: str, a_token: tuple[float, int]) -> bytes:
    """
    Build the multi-sheet Excel workbook: Prices, Anomalies, Forecasts, Metrics.

    Parameters
    ----------
    crop : str
        Commodity name.
    horizon : int
        Forecast horizon in periods.
    d0_iso, d1_iso : str
        Global date filter.
    districts : tuple of str
        Selected districts.
    token : str
        Data fingerprint.
    a_token : tuple
        Anomalies cache key.

    Returns
    -------
    bytes
        The .xlsx file content (every sheet exists even when data is missing).
    """
    core = load_core()
    d0, d1, dl = pd.Timestamp(d0_iso), pd.Timestamp(d1_iso), list(districts)
    prices = filter_prices(core["df"], d0, d1, dl)
    prices = prices[prices["commodity_name"] == crop]
    keep = [c for c in ("date", "commodity_name", "admin2_name", PRICE, "temperature_2m_max", "precipitation_sum")
            if c in prices.columns]
    prices = prices[keep].sort_values([c for c in ("admin2_name", "date") if c in keep]).reset_index(drop=True)

    an = filter_anomalies(load_anomalies(a_token), d0, d1, dl)
    an = an[an["commodity_name"] == crop].drop(columns=["obs_date"], errors="ignore") if not an.empty else an

    bundle = forecast_bundle(crop, horizon, d0_iso, d1_iso, districts, token)
    fc = bundle["fc"] if bundle else pd.DataFrame()
    bt = bundle["bt"] if bundle else pd.DataFrame()
    board = bundle["board"] if bundle else pd.DataFrame()

    xg = core["metrics"].get(crop, {})
    xrows = [{"Metric": k, "Value": (", ".join(f'{f["feature"]} ({f["importance"]:.2f})' for f in v)
                                    if k == "top_features" and isinstance(v, list) else v)}
             for k, v in xg.items()]
    info = pd.DataFrame({"Field": ["Commodity", "Window", "Districts", "Horizon (periods)", "Winning backtest model",
                                   "Generated"],
                         "Value": [crop, f"{d0:%Y-%m-%d} to {d1:%Y-%m-%d}", len(dl) or "all", horizon,
                                   bundle["best"] if bundle else "n/a", datetime.now().strftime("%Y-%m-%d %H:%M")]})

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        _or_placeholder(prices).to_excel(writer, sheet_name="Prices", index=False)
        _or_placeholder(an).to_excel(writer, sheet_name="Anomalies", index=False)
        _or_placeholder(fc).to_excel(writer, sheet_name="Forecasts", index=False, startcol=0)
        if not bt.empty:
            bt.rename(columns={"actual": "backtest_actual", "pred": "backtest_predicted"}).to_excel(
                writer, sheet_name="Forecasts", index=False, startcol=6)
        _or_placeholder(board).to_excel(writer, sheet_name="Metrics", index=False, startcol=0)
        _or_placeholder(pd.DataFrame(xrows), "No XGBoost metrics for this crop").to_excel(
            writer, sheet_name="Metrics", index=False, startcol=5)
        info.to_excel(writer, sheet_name="Metrics", index=False, startcol=9)
        _style_workbook(writer.book)
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def make_advisory_pdf(crop: str, qty: int, horizon: int, storage: float, d0_iso: str, d1_iso: str,
                      districts: tuple[str, ...], token: str) -> bytes | None:
    """
    Build the branded one-page PDF advisory report (English).

    Parameters
    ----------
    crop : str
        Commodity name.
    qty : int
        Bags held (40 kg each).
    horizon : int
        Hold window in periods.
    storage : float
        Storage + financing cost, % per period.
    d0_iso, d1_iso : str
        Global date filter.
    districts : tuple of str
        Selected districts.
    token : str
        Data fingerprint.

    Returns
    -------
    bytes or None
        PDF bytes, or None if reportlab is missing or there is too little history.
    """
    if not PDF_OK or build_advisory_pdf is None:
        return None
    b = forecast_bundle(crop, horizon, d0_iso, d1_iso, districts, token)
    if b is None:
        return None
    adv = compute_advice(b["y"], b["fc"], b["board"], b["sigma"], horizon, storage)
    return build_advisory_pdf(
        crop=crop, qty=qty, horizon=horizon, now_price=adv["now_px"], forecast_price=adv["exp_px"],
        recommendation=REC_EN[adv["rec"]], confidence=adv["conf"],
        rationale=advice_reason(adv, "en"), history=b["y"], forecast=b["fc"],
        gross_pct=adv["gross"], net_pct=adv["net"], uncertainty_pct=adv["err_pct"], model_name=b["best"],
    )


def safe_filename(text: str) -> str:
    """
    Make a string safe for a download file name.

    Parameters
    ----------
    text : str
        Raw text (e.g. a commodity name).

    Returns
    -------
    str
        Letters, digits, dashes and underscores only.
    """
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text).strip("_") or "export"


# ============================================================================
# LIVE ALERTS
# ============================================================================
@st.cache_data(show_spinner=False)
def alert_feed(a_token: tuple[float, int], lang: str, n: int = 5) -> list[dict[str, str]]:
    """
    Build the sidebar feed: the ``n`` most recent anomalies with a plain-text reason.

    Parameters
    ----------
    a_token : tuple
        Anomalies cache key.
    lang : str
        Language code for the reason text.
    n : int
        Number of items.

    Returns
    -------
    list of dict
        Each item has ``crop``, ``date``, ``pct``, ``direction`` and ``reason``.
    """
    an = load_anomalies(a_token)
    if an.empty:
        return []
    an = an.copy()
    an["_abs"] = an["pct_change"].abs().fillna(0)
    an = (an.sort_values(["date", "_abs"], ascending=[False, False])
            .drop_duplicates(subset=[c for c in ("commodity_name", "obs_date", "admin2_name") if c in an.columns])
            .head(n))
    feed: list[dict[str, str]] = []
    for _, row in an.iterrows():
        dist = district_of(row)
        res = root_cause_analysis(str(row["commodity_name"]), pd.Timestamp(row["obs_date"]).isoformat(), dist)
        reason = T(res["drivers"][0]["label"], lang) if res else T("drv_other", lang)
        feed.append({"crop": str(row["commodity_name"]) + (f" · {dist}" if dist else ""), "date": f'{row["date"]:%d %b %Y}',
                     "pct": pct_str(row["pct_change"]), "direction": str(row["direction"]), "reason": reason})
    return feed


def render_alerts_sidebar(a_token: tuple[float, int]) -> None:
    """
    Draw the "Live Alerts" expander in the sidebar.

    Parameters
    ----------
    a_token : tuple
        Anomalies cache key.

    Returns
    -------
    None
    """
    with st.expander(T("alerts_title"), expanded=False):
        st.button(T("alerts_refresh"), key="alerts_refresh_btn", on_click=bump_alerts)
        feed = alert_feed(a_token, get_lang())
        if not feed:
            st.caption(T("alerts_empty"))
        else:
            items = []
            for it in feed:
                color = RED if it["direction"] == "spike" else BLUE
                label = T("alert_spike") if it["direction"] == "spike" else T("alert_drop")
                items.append(
                    f'<div class="alert-item"><span class="dot" style="background:{color}"></span><div>'
                    f'<div class="ttl">{it["crop"]} · {nm(it["pct"])}</div>'
                    f'<div class="meta">{label} · {nm(it["date"])}</div>'
                    f'<div class="why">{it["reason"]}</div></div></div>')
            st.markdown('<div class="alert-feed">' + "".join(items) + "</div>", unsafe_allow_html=True)
        st.caption(T("alerts_updated", t=st.session_state.get("alerts_ts", "")))


# ============================================================================
# PAGE VIEW MODEL
# ============================================================================
@dataclass(frozen=True)
class View:
    """Everything a page needs: filtered data plus the active filters."""

    df: pd.DataFrame
    an: pd.DataFrame
    an_all: pd.DataFrame
    commodities: list[str]
    n_dist: int
    d0: pd.Timestamp
    d1: pd.Timestamp
    districts: tuple[str, ...]
    token: str
    a_token: tuple[float, int]
    colors: dict[str, str]
    profiles: pd.DataFrame
    metrics: dict[str, dict[str, Any]]
    df_all: pd.DataFrame

    def bundle(self, crop: str, horizon: int) -> dict[str, Any] | None:
        """
        Cached forecast bundle for a crop under the current filters.

        Parameters
        ----------
        crop : str
            Commodity name.
        horizon : int
            Forecast horizon in periods.

        Returns
        -------
        dict or None
            See ``forecast_bundle``.
        """
        return forecast_bundle(crop, horizon, self.d0.isoformat(), self.d1.isoformat(),
                               self.districts, self.token)


# ============================================================================
# PAGE 0 - HOME (landing)
# ============================================================================
@st.cache_data(show_spinner=False)
def top_movers(token: str, n: int = 3) -> tuple[pd.DataFrame, str]:
    """
    The ``n`` commodities with the biggest price change in the latest period.

    Parameters
    ----------
    token : str
        Data fingerprint for cache invalidation.
    n : int
        How many movers to return.

    Returns
    -------
    tuple
        ``(movers_df, latest_date_text)``. ``movers_df`` has columns commodity, price, pct.
    """
    df = load_core()["df"]
    basket = df.groupby(["commodity_name", "date"])[PRICE].mean().reset_index()
    rows = []
    for crop, g in basket.groupby("commodity_name"):
        g = g.sort_values("date")
        if len(g) < 2 or g[PRICE].iloc[-2] <= 0:
            continue
        rows.append({"commodity": crop, "price": float(g[PRICE].iloc[-1]),
                     "pct": float(g[PRICE].iloc[-1] / g[PRICE].iloc[-2] - 1) * 100,
                     "date": g["date"].iloc[-1]})
    if not rows:
        return pd.DataFrame(columns=["commodity", "price", "pct"]), ""
    out = pd.DataFrame(rows)
    out["_abs"] = out["pct"].abs()
    latest = out["date"].max()
    return out.sort_values("_abs", ascending=False).head(n).drop(columns=["_abs", "date"]), f"{latest:%d %b %Y}"


def page_home(v: View) -> None:
    """
    Render the marketing-grade landing page.

    Parameters
    ----------
    v : View
        Page view model (only the unfiltered data is used here).

    Returns
    -------
    None
    """
    st.markdown(f'<div class="hero-xl"><span class="badge">🌾 MANDIMIND · v{APP_VERSION}</span>'
                f'<h1>{T("home_hero_title")}</h1><p>{T("home_hero_sub")}</p></div>', unsafe_allow_html=True)
    st.button(T("home_cta"), type="primary", key="cta_start", on_click=goto_page, args=("overview",))
    st.write("")
    render_why_matters()

    c1, c2, c3 = st.columns(3)
    for col, icon, t_key, d_key in ((c1, "📈", "feat1_t", "feat1_d"), (c2, "🔍", "feat2_t", "feat2_d"),
                                    (c3, "🌾", "feat3_t", "feat3_d")):
        col.markdown(f'<div class="feat"><div class="ico">{icon}</div><h3>{T(t_key)}</h3>'
                     f'<p>{T(d_key)}</p></div>', unsafe_allow_html=True)

    st.write("")
    sec(T("home_movers"))
    movers, latest = top_movers(v.token, 3)
    if movers.empty:
        st.info(T("movers_none"))
    else:
        note(T("movers_note", date=nm(latest)))
        cols = st.columns(3)
        for col, r in zip(cols, movers.itertuples()):
            up = r.pct >= 0
            kpi(col, r.commodity, f'{fmt(r.price)}<small>{T("unit")}</small>',
                f'{"▲" if up else "▼"} {T("movers_sub", x=nm(pct_str(r.pct)))}',
                RED if up else BLUE)
    st.markdown(f'<div class="foot">{T("home_footer")}</div>', unsafe_allow_html=True)


# ============================================================================
# PAGE 1 - MARKET OVERVIEW
# ============================================================================
def page_overview(v: View) -> None:
    """
    Render Market Overview: shock banner, KPI row, price trends and volatility ranking.

    Parameters
    ----------
    v : View
        Page view model.

    Returns
    -------
    None
    """
    df, an, commodities = v.df, v.an, v.commodities

    # banner: anomalies in the 30 days up to the latest data point
    if not v.an_all.empty:
        ref = v.df_all["date"].max()
        n_new = int((v.an_all["date"] > ref - pd.Timedelta(days=30)).sum())
        if n_new > 0:
            st.markdown(f'<div class="banner">{T("banner_shocks", n=nm(n_new))}</div>', unsafe_allow_html=True)

    render_news(v)
    basket = df.groupby("date")[PRICE].mean().sort_index().dropna()
    last = basket.iloc[-1] if len(basket) else np.nan
    prev = basket.iloc[-2] if len(basket) > 1 else np.nan
    chg = (last / prev - 1) * 100 if pd.notna(prev) and prev else np.nan
    top_name, top_sub = T("na"), T("kpi_no_metrics")
    scored = {k: m for k, m in v.metrics.items() if isinstance(m.get("improvement_pct"), (int, float))
              and np.isfinite(m["improvement_pct"])}
    if scored:
        k, m = max(scored.items(), key=lambda kv: kv[1]["improvement_pct"])
        top_name, top_sub = k, T("kpi_best_ml_sub", x=nm(f"{m['improvement_pct']:+.1f}%"))

    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, T("kpi_basket"), f'{fmt(last)}<small>{T("unit")}</small>',
        T("na") if pd.isna(chg) else T("kpi_vs_prev", x=nm(f"{chg:+.1f}%")),
        RED if (pd.notna(chg) and chg > 0) else GREEN)
    kpi(k2, T("kpi_anoms"), f"{len(an):,}", T("kpi_anoms_sub"), RED)
    kpi(k3, T("kpi_markets"), f"{v.n_dist}", T("kpi_markets_sub"), BLUE)
    kpi(k4, T("kpi_best_ml"), top_name, top_sub, PURPLE)
    st.write("")

    left, right = st.columns([2.2, 1])
    with left:
        sec(T("price_trends"))
        c1, c2, c3 = st.columns([2.6, 1.7, 1.2])
        default = [c for c in ["Wheat", "Rice (coarse)", "Sugar"] if c in commodities] or commodities[:3]
        sanitize_choice("ov_crops", commodities, multi=True)
        if "ov_crops" not in st.session_state:
            st.session_state["ov_crops"] = default
        crops = c1.multiselect(T("commodities"), commodities, key="ov_crops")
        mode = c2.radio(T("view"), ["index", "dual"], format_func=lambda k: T(f"view_{k}"),
                        horizontal=True, key="ov_mode")
        show_an = c3.toggle(T("anomalies_toggle"), value=True, key="ov_anoms")

        if mode == "dual":
            crops = crops[:2]
        series = {c: df[df["commodity_name"] == c].groupby("date")[PRICE].mean().sort_index() for c in crops}
        series = {c: s.dropna() for c, s in series.items() if not s.dropna().empty and s.dropna().iloc[0] != 0}

        if not series:
            st.info(T("pick_one"))
        else:
            dual = mode == "dual" and len(series) == 2
            fig = go.Figure()
            names = list(series)
            plotted: dict[str, pd.Series] = {}
            for i, c in enumerate(names):
                s = series[c]
                y = s if dual else s / s.iloc[0] * 100
                plotted[c] = y
                fig.add_trace(go.Scatter(
                    x=y.index, y=y.values, name=c, mode="lines", yaxis="y2" if (dual and i == 1) else "y",
                    line=dict(color=v.colors.get(c, BLUE), width=2.8, shape="spline", smoothing=0.5),
                    hovertemplate=f"{c}: " + "%{y:,.1f}<extra></extra>"))
            if show_an and {"date", "commodity_name"} <= set(an.columns) and not an.empty:
                first_marker = True
                for i, c in enumerate(names):
                    dts = an.loc[an["commodity_name"] == c, "date"].dropna().unique()
                    pts = snap(plotted[c], dts)
                    if pts.empty:
                        continue
                    fig.add_trace(go.Scatter(
                        x=pts.index, y=pts.values, mode="markers", name="Anomaly", showlegend=first_marker,
                        legendgroup="anomaly", yaxis="y2" if (dual and i == 1) else "y",
                        marker=dict(symbol="x", size=11, color=RED, line=dict(width=2.5, color=RED)),
                        hovertemplate=f"{c} anomaly: " + "%{y:,.1f}<extra></extra>"))
                    first_marker = False
            style_plotly(fig, 430)
            if dual:
                fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False, zeroline=False,
                                              title=dict(text=f"{names[1]} (PKR)", font=dict(color=v.colors.get(names[1], BLUE)))),
                                  yaxis=dict(title=dict(text=f"{names[0]} (PKR)", font=dict(color=v.colors.get(names[0], BLUE)))))
            else:
                fig.update_layout(yaxis=dict(title=dict(text="Index (start = 100)")))
                fig.add_hline(y=100, line_dash="dot", line_color=GREY, line_width=1)
            show(fig, key="ov_trend")
            note(T("note_index"))

    with right:
        sec(T("volatility"))
        g = df.groupby("commodity_name")[PRICE]
        cv = ((g.std() / g.mean()) * 100).replace([np.inf, -np.inf], np.nan).dropna().sort_values().tail(10)
        if cv.empty:
            st.info(T("cv_none"))
        else:
            fig = go.Figure(go.Bar(
                x=cv.values, y=cv.index, orientation="h",
                marker=dict(color=cv.values, colorscale=[[0, "#FDE68A"], [0.5, AMBER], [1, RED]], line=dict(width=0)),
                text=[f"{x:.1f}%" for x in cv.values], textposition="outside", cliponaxis=False,
                textfont=dict(size=12, color="#334155"), hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
            style_plotly(fig, 430, legend=False, hover="closest")
            fig.update_xaxes(visible=False, range=[0, float(cv.max()) * 1.35])
            fig.update_layout(margin=dict(l=10, r=40, t=30, b=10))
            show(fig, key="ov_cv")
            note(T("note_cv"))


# ============================================================================
# PAGE 2 - GEOGRAPHIC RISK
# ============================================================================
def map_positions(profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Attach display coordinates and label positions, nudging apart districts that would overlap.

    Parameters
    ----------
    profiles : pd.DataFrame
        District profiles with an ``admin2_name`` column.

    Returns
    -------
    pd.DataFrame
        Rows with known coordinates, plus ``lat``, ``lon`` and ``pos``.
    """
    m = profiles[profiles["admin2_name"].isin(list(COORDS))].copy()
    present = set(m["admin2_name"])
    lats, lons, labels = [], [], []
    for name in m["admin2_name"]:
        lat, lon, pos = COORDS[name]
        pos = MAP_LABEL_POS.get(name, pos)
        nudge = MAP_NUDGES.get(name)
        if nudge and present & set(nudge[3]):
            lat, lon, pos = lat + nudge[0], lon + nudge[1], nudge[2]
        lats.append(lat)
        lons.append(lon)
        labels.append(pos)
    return m.assign(lat=lats, lon=lons, pos=labels)


def label_font() -> dict[str, Any]:
    """
    Bold map-label font with a white halo where the installed Plotly supports text shadows.

    Returns
    -------
    dict
        ``textfont`` settings (Scattergeo has no text background option).
    """
    # Do not add ``bgcolor`` / ``borderpad`` here: Plotly rejects both on map text and the page would crash.
    font: dict[str, Any] = dict(size=11.5, color=INK, family="Inter, sans-serif")
    try:
        go.scattergeo.Textfont(shadow=LABEL_HALO)
    except ValueError:  # older Plotly: the <b> tags still keep labels bold
        return font
    return {**font, "shadow": LABEL_HALO}


def page_geo(v: View) -> None:
    """
    Render the district volatility map and risk ranking.

    Parameters
    ----------
    v : View
        Page view model.

    Returns
    -------
    None
    """
    p = v.profiles.copy()
    needed = {"admin2_name", "cluster_name", "price_volatility"}
    if p.empty or not needed <= set(p.columns):
        st.info(T("geo_unavailable"))
        return
    if v.districts:
        p = p[p["admin2_name"].isin(v.districts)]
    p = p.dropna(subset=["admin2_name", "price_volatility"])
    p["cluster_name"] = p["cluster_name"].fillna("n/a")
    m = map_positions(p)

    left, right = st.columns([1.8, 1])
    with left:
        sec(T("geo_map"))
        if m.empty:
            st.info(T("geo_no_coords"))
        else:
            vmin, vmax = float(m["price_volatility"].min()), float(m["price_volatility"].max())
            m["size"] = 13 + 15 * (m["price_volatility"] - vmin) / ((vmax - vmin) or 1.0)
            spare = iter([BLUE, PURPLE, AMBER, "#EC4899"])
            fig = go.Figure()
            for cl in m["cluster_name"].unique():
                s = m[m["cluster_name"] == cl]
                colr = CLUSTER_COLORS.get(cl) or next(spare, GREY)
                hover = [f"<b>{r.admin2_name}</b><br>{cl}<br>Volatility: {r.price_volatility:,.0f}"
                         + (f"<br>Avg price: {r.avg_price:,.0f}" if "avg_price" in s.columns and pd.notna(r.avg_price) else "")
                         for r in s.itertuples()]
                fig.add_trace(go.Scattergeo(
                    lon=s["lon"], lat=s["lat"], mode="markers", name=cl, hovertext=hover, hoverinfo="text",
                    marker=dict(size=s["size"], sizemode="diameter", color=colr, opacity=0.85,
                                line=dict(color="#fff", width=2))))
            # Labels are drawn after every bubble. Each label point carries an invisible marker of the same
            # size, so Plotly offsets the text clear of its own bubble.
            for pos in m["pos"].unique():
                s = m[m["pos"] == pos]
                fig.add_trace(go.Scattergeo(
                    lon=s["lon"], lat=s["lat"], mode="markers+text", text=[f"<b>{d}</b>" for d in s["admin2_name"]],
                    textposition=pos, showlegend=False, hoverinfo="skip",
                    marker=dict(size=s["size"] + 2, sizemode="diameter", color="rgba(0,0,0,0)", line=dict(width=0)),
                    textfont=label_font()))
            for nm_, la, lo in COUNTRY_LABELS:
                fig.add_trace(go.Scattergeo(lon=[lo], lat=[la], mode="text", text=[nm_], showlegend=False,
                                            hoverinfo="skip",
                                            textfont=dict(size=11 if nm_.isupper() else 10, color="#94A3B8",
                                                          family="Inter, sans-serif")))
            fig.update_geos(
                scope="asia", projection_type="mercator", resolution=50,
                lataxis_range=[23, 37], lonaxis_range=[60, 78], center=dict(lat=30, lon=69),
                showcountries=True, countrycolor="#94A3B8", countrywidth=1,
                showland=True, landcolor="#FFFFFF", showocean=True, oceancolor="#DBEAFE",
                showcoastlines=True, coastlinecolor="#94A3B8", showframe=False, bgcolor="rgba(0,0,0,0)")
            style_plotly(fig, 640, hover="closest")
            fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="left", x=0.01,
                                          bgcolor="rgba(255,255,255,.88)", bordercolor=GRID, borderwidth=1))
            show(fig, key="geo_map", config={**CHART_CONFIG, "scrollZoom": False})  # page scroll must not zoom
            note(T("geo_note"))
        missing = sorted(set(p["admin2_name"]) - set(COORDS))
        if missing:
            st.caption(T("geo_missing", names=", ".join(missing)))

    with right:
        sec(T("geo_rank"))
        rk = p.sort_values("price_volatility")
        if rk.empty:
            st.info(T("geo_no_districts"))
        else:
            fig = go.Figure(go.Bar(
                x=rk["price_volatility"], y=rk["admin2_name"], orientation="h",
                marker_color=[CLUSTER_COLORS.get(c, BLUE) for c in rk["cluster_name"]],
                text=[f"{x:,.0f}" for x in rk["price_volatility"]], textposition="outside", cliponaxis=False,
                hovertemplate="%{y}: %{x:,.0f}<extra></extra>"))
            style_plotly(fig, max(320, 34 * len(rk) + 60), legend=False, hover="closest")
            fig.update_xaxes(visible=False, range=[0, float(rk["price_volatility"].max()) * 1.3])
            fig.update_layout(margin=dict(l=10, r=40, t=30, b=10))
            show(fig, key="geo_rank")


# ============================================================================
# PAGE 3 - FORECAST LAB (+ Root Cause AI)
# ============================================================================
def render_root_cause(v: View, crop: str) -> None:
    """
    Draw the "Why did this happen? (Root Cause AI)" expander.

    Parameters
    ----------
    v : View
        Page view model.
    crop : str
        Commodity currently selected in the Forecast Lab.

    Returns
    -------
    None
    """
    with st.expander(T("rc_title"), expanded=False):
        cand = v.an_all[v.an_all["commodity_name"] == crop] if not v.an_all.empty else v.an_all
        if cand.empty:
            st.info(T("rc_none"))
            return
        cand = cand.sort_values("date", ascending=False).head(200).reset_index(drop=True)
        labels: list[str] = []
        for _, r in cand.iterrows():
            dist = district_of(r)
            labels.append(f'{r["date"]:%Y-%m-%d} · {crop}' + (f" · {dist}" if dist else "")
                          + f' · {pct_str(r["pct_change"])}')
        idx = st.selectbox(T("rc_pick"), list(range(len(cand))), format_func=lambda i: labels[i],
                           key=f"rc_pick_{crop}")
        row = cand.iloc[int(idx)]
        res = root_cause_analysis(crop, pd.Timestamp(row["obs_date"]).isoformat(), district_of(row))
        if res is None:
            st.info(T("rc_insufficient"))
            return
        st.markdown(explanation_html(res), unsafe_allow_html=True)
        fig = go.Figure(go.Bar(
            x=[d["weight"] for d in res["drivers"]][::-1],
            y=[T(d["label"], "en") for d in res["drivers"]][::-1], orientation="h",
            marker_color=[AMBER if d["key"] == "other" else PURPLE for d in res["drivers"]][::-1],
            text=[f'{d["weight"]:.0f}%' for d in res["drivers"]][::-1], textposition="outside", cliponaxis=False,
            hovertemplate="%{y}: %{x:.0f}%<extra></extra>"))
        style_plotly(fig, 210, legend=False, hover="closest")
        fig.update_xaxes(visible=False, range=[0, max(d["weight"] for d in res["drivers"]) * 1.3])
        fig.update_layout(margin=dict(l=10, r=50, t=10, b=10))
        show(fig, key=f"rc_chart_{crop}")
        note(T("rc_caption"))


def page_forecast(v: View) -> None:
    """
    Render the Forecast Lab: KPIs, forecast chart, leaderboard, backtest, Root Cause AI.

    Parameters
    ----------
    v : View
        Page view model.

    Returns
    -------
    None
    """
    sanitize_choice("fc_crop", v.commodities)
    c1, c2, _ = st.columns([1.2, 1.2, 2.6])
    crop = c1.selectbox(T("fc_commodity"), v.commodities, key="fc_crop")
    horizon = c2.slider(T("fc_horizon"), 1, 6, 3, key="fc_horizon")
    b = v.bundle(crop, horizon)
    if b is None:
        st.warning(T("fc_need_history"))
        return
    y, fc, board, best, bt = b["y"], b["fc"], b["board"], b["best"], b["bt"]
    xg = v.metrics.get(crop, {})
    mae_x, mae_n = xg.get("mae_xgb"), xg.get("mae_naive")

    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, T("kpi_winner"), best, T("kpi_winner_sub"), GREEN)
    kpi(k2, T("kpi_mape"), f"{board.iloc[0]['MAPE']:.1f}%", T("kpi_mape_sub"), BLUE)
    kpi(k3, T("kpi_xgb_mae"), f"{mae_x:.1f}" if isinstance(mae_x, (int, float)) else T("na"),
        T("kpi_pipeline_model") if xg else T("kpi_no_crop_metrics"), PURPLE)
    kpi(k4, T("kpi_naive_mae"), f"{mae_n:.1f}" if isinstance(mae_n, (int, float)) else T("na"),
        (T("kpi_xgb_wins") if xg.get("use_ml") else T("kpi_baseline_holds")) if xg else T("kpi_no_crop_metrics"), AMBER)
    st.write("")

    sec(T("fc_chart_title", crop=crop))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y.index, y=y.values, name="Actual", mode="lines",
                             line=dict(color=INK, width=2.8), hovertemplate="Actual: %{y:,.0f}<extra></extra>"))
    # bridge the interval from the last actual point so the area connects to history
    bx = [y.index[-1]] + list(fc["date"])
    fig.add_trace(go.Scatter(x=bx, y=[y.iloc[-1]] + list(fc["hi"]), mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=bx, y=[y.iloc[-1]] + list(fc["lo"]), mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(16, 185, 129, 0.1)", name="80% interval",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=bx, y=[y.iloc[-1]] + list(fc["forecast"]), name=f"Forecast ({best})",
                             mode="lines+markers", line=dict(color=GREEN, width=3, dash="dash"),
                             marker=dict(size=8, color="#fff", line=dict(color=GREEN, width=2.5)),
                             hovertemplate="Forecast: %{y:,.0f}<extra></extra>"))
    fig.add_vline(x=y.index[-1], line_dash="dot", line_color=GREY, line_width=1)
    style_plotly(fig, 420)
    fig.update_layout(yaxis=dict(title=dict(text="PKR / 40kg")))
    show(fig, key="fc_main")

    est = ml_next_estimate(crop, v.token)
    if est:
        note(T("fc_ml_estimate", px=nm(f'{est["price"]:,.0f}'), chg=nm(pct_str(est["chg_pct"]))))

    a, bcol = st.columns(2)
    with a:
        sec(T("fc_leaderboard"))
        s = board.sort_values("MAE", ascending=False)
        fig = go.Figure(go.Bar(
            x=s["MAE"], y=s["Model"], orientation="h",
            marker_color=[GREEN if mname == best else "#CBD5E1" for mname in s["Model"]],
            text=[f"{x:,.1f}" for x in s["MAE"]], textposition="outside", cliponaxis=False,
            hovertemplate="%{y}: %{x:,.1f} PKR<extra></extra>"))
        style_plotly(fig, 330, legend=False, hover="closest")
        fig.update_xaxes(visible=False, range=[0, float(s["MAE"].max()) * 1.3 or 1.0])
        fig.update_layout(margin=dict(l=10, r=40, t=30, b=10))
        show(fig, key="fc_board")
        note(T("fc_leader_note"))
    with bcol:
        sec(T("fc_backtest"))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=bt["date"], y=bt["actual"], name="Actual", mode="lines",
                                 line=dict(color=INK, width=2.6), hovertemplate="Actual: %{y:,.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=bt["date"], y=bt["pred"], name="Predicted", mode="lines+markers",
                                 line=dict(color=AMBER, width=2.4, dash="dot"), marker=dict(size=6),
                                 hovertemplate="Predicted: %{y:,.0f}<extra></extra>"))
        style_plotly(fig, 330)
        show(fig, key="fc_bt")
        note(T("fc_backtest_note"))

    render_root_cause(v, crop)
    st.download_button(T("dl_forecast_csv"), fc.to_csv(index=False).encode(),
                       file_name=f"mandimind_forecast_{safe_filename(crop)}.csv", mime="text/csv",
                       key="dl_fc_csv")


# ============================================================================
# PAGE 4 - DECISION ADVISOR
# ============================================================================
def page_advisor(v: View) -> None:
    """
    Render the Decision Advisor with PDF, CSV and Excel exports.

    Parameters
    ----------
    v : View
        Page view model.

    Returns
    -------
    None
    """
    sanitize_choice("adv_crop", v.commodities)
    left, right = st.columns([1, 2.6])
    with left:
        sec(T("adv_position"))
        crop = st.selectbox(T("adv_crop"), v.commodities, key="adv_crop")
        qty = int(st.number_input(T("adv_qty"), min_value=1, value=100, step=10, key="adv_qty"))
        horizon = st.slider(T("adv_hold"), 1, 6, 2, key="adv_horizon")
        storage = float(st.number_input(T("adv_storage"), 0.0, 10.0, 0.8, 0.1, key="adv_storage"))

    b = v.bundle(crop, horizon)
    if b is None:
        with right:
            st.warning(T("adv_no_history"))
        return
    y, fc, board, best = b["y"], b["fc"], b["board"], b["best"]
    if float(y.iloc[-1]) <= 0:
        with right:
            st.warning(T("adv_bad_price"))
        return

    adv = compute_advice(y, fc, board, b["sigma"], horizon, storage)
    now_px, exp_px, gross = adv["now_px"], adv["exp_px"], adv["gross"]
    label_key, color, shadow = rec_style(adv["rec"])

    with right:
        st.markdown(f'<div class="rec rec-{adv["rec"]}" style="background:{color};--shadow:{shadow}">'
                    f'<div class="lab">{T("rec_label", crop=crop.upper(), qty=nm(f"{qty:,}"))}</div>'
                    f'<h2>{T(label_key)}</h2><p>{advice_reason(adv, wrap=nm)}</p></div>', unsafe_allow_html=True)

        k1, k2, k3 = st.columns(3)
        kpi(k1, T("kpi_price_now"), f'{fmt(now_px)}<small>{T("unit")}</small>', T("kpi_price_now_sub"), BLUE)
        kpi(k2, T("kpi_forecast_n", h=horizon), f'{fmt(exp_px)}<small>{T("unit")}</small>',
            T("kpi_gross", x=nm(pct_str(gross))), GREEN if gross >= 0 else RED)
        kpi(k3, T("kpi_position"), f'{now_px * qty:,.0f}<small>{T("pkr")}</small>',
            T("kpi_bags", n=nm(f"{qty:,}")), PURPLE)
        st.write("")

        g1, g2 = st.columns([1, 1.7])
        with g1:
            sec(T("adv_confidence"))
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=adv["conf"],
                number=dict(suffix="<span style='font-size:16px'>/100</span>", font=dict(size=40, color=INK)),
                gauge=dict(axis=dict(range=[0, 100], tickwidth=0, tickcolor="rgba(0,0,0,0)"),
                           bar=dict(color=INK, thickness=0.22), bgcolor="rgba(0,0,0,0)", borderwidth=0,
                           steps=[dict(range=[0, 40], color="#FECACA"), dict(range=[40, 70], color="#FDE68A"),
                                  dict(range=[70, 100], color="#A7F3D0")])))
            style_plotly(gauge, 270, legend=False, hover="closest")
            show(gauge, key="adv_gauge")
            note(T("adv_conf_note", m=nm(f'{adv["mape"]:.1f}'), b=best))
        with g2:
            sec(T("adv_compare"))
            cost = now_px * qty * storage / 100 * horizon
            labels = ["Sell now", "Hold · pessimistic", "Hold · expected", "Hold · optimistic"]
            vals = [now_px * qty, fc["lo"].iloc[-1] * qty - cost, exp_px * qty - cost, fc["hi"].iloc[-1] * qty - cost]
            fig = go.Figure(go.Bar(
                x=labels, y=vals, marker_color=["#64748B", RED, AMBER, GREEN],
                text=[f"{x:,.0f}" for x in vals], textposition="outside", cliponaxis=False,
                hovertemplate="%{x}: %{y:,.0f} PKR<extra></extra>"))
            fig.add_hline(y=vals[0], line_dash="dot", line_color=GREY, line_width=1)
            style_plotly(fig, 270, legend=False, hover="closest")
            lo_v, hi_v = min(vals), max(vals)
            pad = (hi_v - lo_v) or abs(hi_v) * 0.1 or 1.0
            fig.update_yaxes(visible=False, range=[max(0, lo_v - pad * 0.6), hi_v + pad * 0.35])
            show(fig, key="adv_compare")
            note(T("adv_compare_note", h=nm(horizon), c=nm(f"{cost:,.0f}")))

        with st.expander(T("adv_why")):
            tag = T("tag_expensive") if adv["z"] > .5 else T("tag_cheap") if adv["z"] < -.5 else T("tag_normal")
            items = [
                T("adv_why_price", z=nm(f'{adv["z"]:+.2f}'), tag=tag),
                T("adv_why_mom", m=nm(pct_str(adv["mom"]))),
                T("adv_why_net", g=nm(pct_str(gross)), s=nm(f"{storage * horizon:.1f}%"), n=nm(pct_str(adv["net"]))),
                T("adv_why_unc", h=nm(horizon), e=nm(f'{adv["err_pct"]:.1f}%')),
                T("adv_why_rule"),
            ]
            st.markdown("<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>", unsafe_allow_html=True)

        # ---- exports: PDF, CSV, Excel -------------------------------------
        d0i, d1i = v.d0.isoformat(), v.d1.isoformat()
        e1, e2, e3 = st.columns(3)
        try:
            pdf_bytes = make_advisory_pdf(crop, qty, horizon, storage, d0i, d1i, v.districts, v.token)
        except Exception as exc:  # never crash the page because of an export
            pdf_bytes = None
            st.warning(T("export_failed", err=str(exc)))
        if pdf_bytes:
            e1.download_button(T("dl_pdf"), pdf_bytes, file_name=f"mandimind_advisory_{safe_filename(crop)}.pdf",
                               mime="application/pdf", key="dl_pdf")
            e1.caption(T("pdf_note"))
        elif not PDF_OK:
            e1.warning(T("pdf_unavailable"))

        fc_out = fc.copy()
        fc_out.insert(0, "commodity", crop)
        e2.download_button(T("dl_csv"), fc_out.to_csv(index=False).encode(),
                           file_name=f"mandimind_forecast_{safe_filename(crop)}.csv", mime="text/csv", key="dl_csv")
        try:
            xlsx = build_excel_report(crop, horizon, d0i, d1i, v.districts, v.token, v.a_token)
            e3.download_button(T("dl_xlsx"), xlsx, file_name=f"mandimind_analysis_{safe_filename(crop)}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               key="dl_xlsx")
        except Exception as exc:
            st.warning(T("export_failed", err=str(exc)))
        st.caption(T("disclaimer_long"))


# ============================================================================
# PAGE 5 - WHAT-IF SIMULATOR
# ============================================================================
def page_whatif(v: View) -> None:
    """
    Render the What-If Simulator (scenario stress testing).

    Parameters
    ----------
    v : View
        Page view model.

    Returns
    -------
    None
    """
    note(T("wi_intro"))
    sanitize_choice("wi_crop", v.commodities)
    left, right = st.columns([1, 2.4])
    with left:
        crop = st.selectbox(T("wi_crop"), v.commodities, key="wi_crop")
        horizon = st.slider(T("wi_horizon"), 1, 6, 3, key="wi_horizon")
        rain = st.slider(T("wi_rain"), -50, 200, 0, 5, key="wi_rain")
        diesel = st.slider(T("wi_diesel"), 0, 100, 0, 5, key="wi_diesel")
        fx = st.slider(T("wi_fx"), -30, 50, 0, 5, key="wi_fx")
        ram_on = st.toggle(T("wi_ramadan"), value=False, key="wi_ramadan")

    b = v.bundle(crop, horizon)
    if b is None:
        with right:
            st.warning(T("wi_need_history"))
        return
    y, fc = b["y"], b["fc"]
    now_px = float(y.iloc[-1])
    lift = ramadan_lift(crop, v.token)
    res = simulate_scenario(fc, float(rain), float(diesel), float(fx), bool(ram_on), lift)
    sim = res["sim"]
    base_chg = (float(sim["baseline"].iloc[-1]) / now_px - 1) * 100
    scen_chg = (float(sim["simulated"].iloc[-1]) / now_px - 1) * 100
    impact = res["total"] * 100

    with right:
        k1, k2, k3 = st.columns(3)
        kpi(k1, T("wi_kpi_base"), f'{fmt(sim["baseline"].iloc[-1])}<small>{T("unit")}</small>',
            T("wi_kpi_base_sub", x=nm(pct_str(base_chg))), GREY)
        kpi(k2, T("wi_kpi_scen"), f'{fmt(sim["simulated"].iloc[-1])}<small>{T("unit")}</small>',
            T("wi_kpi_scen_sub", x=nm(pct_str(scen_chg))), GREEN if scen_chg <= base_chg else RED)
        kpi(k3, T("wi_kpi_impact"), nm(pct_str(impact)), T("wi_kpi_impact_sub"), AMBER)
        st.write("")

        sec(T("wi_chart_title", crop=crop))
        hist = y.tail(24)
        bx = [hist.index[-1]] + list(sim["date"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Actual", mode="lines",
                                 line=dict(color=INK, width=2.6), hovertemplate="Actual: %{y:,.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=bx, y=[hist.iloc[-1]] + list(sim["baseline"]), name="Original forecast",
                                 mode="lines", line=dict(color=GREY, width=2.6, dash="dot"),
                                 hovertemplate="Original: %{y:,.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=bx, y=[hist.iloc[-1]] + list(sim["simulated"]), name="Simulated scenario",
                                 mode="lines+markers", line=dict(color=GREEN, width=3.2),
                                 marker=dict(size=7, color="#fff", line=dict(color=GREEN, width=2.4)),
                                 hovertemplate="Scenario: %{y:,.0f}<extra></extra>"))
        fig.add_vline(x=hist.index[-1], line_dash="dot", line_color=GREY, line_width=1)
        style_plotly(fig, 400)
        fig.update_layout(yaxis=dict(title=dict(text="PKR / 40kg")))
        show(fig, key="wi_chart")

        parts = {k: abs(x) for k, x in res["parts"].items()}
        if max(parts.values()) < 1e-9:
            text = T("wi_summary_none", crop=crop, pct=nm(pct_str(scen_chg)), h=nm(horizon))
        else:
            text = T("wi_summary", crop=crop, pct=nm(pct_str(scen_chg)), h=nm(horizon),
                     driver=T(f"wi_drv_{max(parts, key=parts.get)}"))
        st.markdown(f'<div class="rc-box">{text}</div>', unsafe_allow_html=True)
        note(T("wi_caption", lift=nm(f"+{lift * 100:.1f}%")))


# ============================================================================
# v8.0 - SHARED HELPERS
# ============================================================================
PROVINCE: dict[str, str] = {
    "Karachi": "Sindh", "Hyderabad": "Sindh", "Sukkur": "Sindh", "Larkana": "Sindh",
    "Quetta": "Balochistan",
    "Multan": "Punjab", "Bahawalpur": "Punjab", "Faisalabad": "Punjab", "Lahore": "Punjab",
    "Gujranwala": "Punjab", "Sialkot": "Punjab", "Sargodha": "Punjab", "Rawalpindi": "Punjab",
    "Islamabad": "Islamabad", "Peshawar": "Khyber Pakhtunkhwa", "Mardan": "Khyber Pakhtunkhwa",
}


def show_df(frame: pd.DataFrame) -> None:
    """
    Render a table full-width on old and new Streamlit versions.

    Parameters
    ----------
    frame : pd.DataFrame
        Table to display (index hidden).

    Returns
    -------
    None
    """
    try:
        st.dataframe(frame, width="stretch", hide_index=True)
    except Exception:
        st.dataframe(frame, use_container_width=True, hide_index=True)


@st.cache_data(show_spinner=False)
def market_changes(token: str) -> pd.DataFrame:
    """
    Latest price and period-on-period % change of every commodity (district average).

    Parameters
    ----------
    token : str
        Data fingerprint for cache invalidation.

    Returns
    -------
    pd.DataFrame
        Columns ``commodity``, ``price``, ``pct``; empty if there is not enough history.
    """
    df = load_core()["df"]
    basket = df.groupby(["commodity_name", "date"])[PRICE].mean().reset_index()
    rows: list[dict[str, Any]] = []
    for crop, g in basket.groupby("commodity_name"):
        g = g.sort_values("date")
        if len(g) < 2 or not g[PRICE].iloc[-2] > 0:
            continue
        rows.append({"commodity": str(crop), "price": float(g[PRICE].iloc[-1]),
                     "pct": float(g[PRICE].iloc[-1] / g[PRICE].iloc[-2] - 1) * 100})
    return pd.DataFrame(rows, columns=["commodity", "price", "pct"])


# ============================================================================
# v8.0 - KILLER FEATURE 1: LIVE MARKET TICKER
# ============================================================================
def render_ticker(token: str) -> None:
    """
    Draw the Bloomberg-style scrolling ticker tape above the page content.

    Parameters
    ----------
    token : str
        Data fingerprint (selects the cached price changes).

    Returns
    -------
    None
    """
    ch = market_changes(token)
    if ch.empty:
        return
    items: list[str] = []
    for r in ch.sort_values("commodity").itertuples():
        if abs(r.pct) < 0.05:
            cls, arrow = "fl", "▬"
        elif r.pct > 0:
            cls, arrow = "up", "▲"
        else:
            cls, arrow = "dn", "▼"
        items.append(f'<span class="tk"><b>{_html.escape(r.commodity)}</b><span class="px">{r.price:,.0f}</span>'
                     f'<span class="{cls}">{r.pct:+.1f}% {arrow}</span></span><span class="tk-sep">|</span>')
    track = "".join(items)
    # With few crops, repeat the set inside each half so it still reads as a continuous tape.
    group = f'<div class="ticker-group">{track * max(1, -(-10 // len(items)))}</div>'
    # exactly two identical halves; the 120s duration lives in the CSS only
    st.markdown(f'<div class="ticker"><div class="ticker-track">{group}{group}</div></div>', unsafe_allow_html=True)


# ============================================================================
# "WHY THIS MATTERS" BANNER
# ============================================================================
def render_why_matters() -> None:
    """
    Draw the full-width storytelling banner with three headline statistics.

    The figures are plain HTML (no scripted or CSS counter), so they are always visible;
    the cards only fade in.

    Returns
    -------
    None
    """
    percent = "<small>%</small>"
    first, third = f"60{percent}", f"12{percent}"
    if get_lang() == "ur":  # keep "60%" and "12%" left-to-right inside the right-to-left row
        first, third = nm(first), nm(third)
    only = f'<small>{T("why_pre_only")}</small>'
    stats = (
        (first, "why_s1", "#047857", "16,185,129"),
        (f'200<small>{T("why_pkr_billion")}</small>', "why_s2", "#B45309", "245,158,11"),
        (f"{only}{third}", "why_s3", "#1D4ED8", "59,130,246"),
    )
    cards = "".join(
        f'<div class="why-card" style="--a:{colour};--b:{colour};--rgb:{rgb}">'
        f'<div class="why-num">{figure}</div><div class="why-txt">{T(caption)}</div></div>'
        for figure, caption, colour, rgb in stats)
    st.markdown(f'<div class="why-wrap"><div class="why-title">{T("why_title")}</div>'
                f'<div class="why-grid">{cards}</div><p class="why-close">{T("why_close")}</p></div>',
                unsafe_allow_html=True)


# ============================================================================
# v8.0 - KILLER FEATURE 5: NEWS INTELLIGENCE (mock feed driven by the data)
# ============================================================================
def _rain_event(df_all: pd.DataFrame) -> str | None:
    """
    Detect an unusual rainfall spike in the latest period.

    Parameters
    ----------
    df_all : pd.DataFrame
        Full feature table.

    Returns
    -------
    str or None
        The affected province (or district), or None when rainfall is normal or missing.
    """
    if "precipitation_sum" not in df_all.columns:
        return None
    per_date = df_all.groupby("date")["precipitation_sum"].mean().dropna()
    if len(per_date) < 12 or not per_date.std() > 0:
        return None
    z = (per_date.iloc[-1] - per_date.mean()) / per_date.std()
    if not (z > 1.5 and per_date.iloc[-1] > 0):
        return None
    last = df_all[df_all["date"] == df_all["date"].max()]
    if "admin2_name" in last.columns and last["precipitation_sum"].notna().any():
        wettest = str(last.loc[last["precipitation_sum"].idxmax(), "admin2_name"])
        return PROVINCE.get(wettest, wettest)
    return "key growing regions"


def build_news(v: View) -> list[dict[str, str]]:
    """
    Generate 3-5 news items from the actual data (price moves, rain, anomalies, Ramadan).

    Parameters
    ----------
    v : View
        Page view model.

    Returns
    -------
    list of dict
        Items with ``tag``, ``color``, ``headline`` (3 to 5 of them, most relevant first).
    """
    ch = market_changes(v.token)
    news: list[dict[str, str]] = []

    def add(tag: str, color: str, headline: str) -> None:
        news.append({"tag": tag, "color": color, "headline": headline})

    if not ch.empty:
        for r in ch.sort_values("pct").itertuples():
            if r.pct <= -5 and len(news) < 2:
                region = "Punjab" if r.commodity.lower().startswith("wheat") else "major producing districts"
                add("SUPPLY", BLUE, f"{r.commodity} prices fall sharply ({r.pct:.1f}%) as {region} harvest "
                                    f"exceeds expectations — sources.")
        for r in ch.sort_values("pct", ascending=False).itertuples():
            if r.pct >= 5 and len(news) < 3:
                add("PRICES", RED, f"{r.commodity} jumps {r.pct:.1f}% as buyers scramble for tighter supply — traders.")
    rain = _rain_event(v.df_all)
    if rain:
        add("WEATHER", AMBER, f"Heavy rainfall disrupts supply chains in {rain}.")
    if not v.an_all.empty:
        recent = v.an_all[v.an_all["date"] > v.df_all["date"].max() - pd.Timedelta(days=30)]
        if not recent.empty:
            row = recent.sort_values("date").iloc[-1]
            where = district_of(row)
            move = row.get("pct_change")
            move_txt = f" ({pct_str(move)})" if pd.notna(move) else ""
            add("ALERT", RED, f"Unusual price move{move_txt} flagged in {row['commodity_name']}"
                              f"{' around ' + where if where else ''} — MandiMind anomaly desk.")
    try:
        latest = v.df_all[v.df_all["date"] == v.df_all["date"].max()]
        if "is_ramadan_window" in latest.columns and int(latest["is_ramadan_window"].max() or 0) == 1:
            add("DEMAND", GREEN, "Ramadan buying window lifts demand for staples across city markets.")
    except Exception:
        pass
    if ch.empty or not (ch["pct"].abs() >= 5).any():
        add("MARKETS", GREEN, "Markets stable across Pakistan.")
    if not ch.empty:
        basket = float(ch["pct"].mean())
        add("MARKETS", GREEN if basket >= 0 else BLUE,
            f"Average commodity price moved {basket:+.1f}% in the latest period across {len(ch)} tracked crops.")
        top = ch.sort_values("pct", ascending=False).iloc[0]
        add("MOVERS", GREEN, f"{top['commodity']} leads the board at {top['pct']:+.1f}% to PKR {top['price']:,.0f} per 40 kg.")
        vol = v.df_all.groupby("commodity_name")[PRICE].agg(["std", "mean"])
        cv = (vol["std"] / vol["mean"]).replace([np.inf, -np.inf], np.nan).dropna()
        if not cv.empty:
            add("RISK", AMBER, f"{cv.idxmax()} remains the most volatile commodity in the dataset "
                               f"(variation {cv.max() * 100:.0f}% of its average price).")
    if not news:
        add("MARKETS", GREEN, "Markets stable across Pakistan.")
    return news[:5] if len(news) >= 3 else news


def render_news(v: View) -> None:
    """
    Draw the "Market News" expander with timestamped news cards.

    Timestamps are random but seeded by the data fingerprint and the hour, so they stay
    stable while the user clicks around and refresh every hour.

    Parameters
    ----------
    v : View
        Page view model.

    Returns
    -------
    None
    """
    with st.expander(T("news_title"), expanded=False):
        try:
            items = build_news(v)
        except Exception:
            items = [{"tag": "MARKETS", "color": GREEN, "headline": "Markets stable across Pakistan."}]
        rng = random.Random(f"{v.token}|{datetime.now():%Y%m%d%H}")
        minutes = sorted(rng.sample(range(7, 600), k=len(items)))
        for it, mins in zip(items, minutes):
            when = f"{mins} minutes ago" if mins < 60 else f"{mins // 60} hour{'s' if mins // 60 > 1 else ''} ago"
            st.markdown(f'<div class="news-card" style="--c:{it["color"]}"><span class="tag">{it["tag"]}</span>'
                        f'<div class="hl">{_html.escape(it["headline"])}</div>'
                        f'<div class="meta">MandiMind Wire · {when}</div></div>', unsafe_allow_html=True)
        st.caption(T("news_caption"))


# ============================================================================
# v8.0 - KILLER FEATURE 4: "ASK MANDIMIND" CHATBOT (keyword routing, no LLM)
# ============================================================================
CROP_ALIASES: dict[str, str] = {
    "gandum": "wheat", "atta": "wheat", "chawal": "rice", "cheeni": "sugar", "chini": "sugar",
    "tamatar": "tomato", "pyaz": "onion", "aloo": "potato", "makai": "maize",
}
NAME_NOISE = {"coarse", "fine", "local", "imported", "fresh", "dry", "grade", "price", "prices"}
SELL_RE = re.compile(r"\b(sell|selling|hold|holding|store|storing|storage|keep)\b")
CHAT_SUGGESTIONS = ("What is the price of wheat?", "Should I sell my sugar?", "Show me the riskiest district")


def _stem(word: str) -> str:
    """
    Crude singular form so "tomatoes" matches "Tomato".

    Parameters
    ----------
    word : str
        Lower-case word.

    Returns
    -------
    str
        The word without a plural ending.
    """
    for suffix in ("ies", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def find_crop(question: str, crops: list[str]) -> str | None:
    """
    Find the commodity mentioned in a question.

    Parameters
    ----------
    question : str
        User text.
    crops : list of str
        Available commodity names.

    Returns
    -------
    str or None
        The best-matching commodity, or None.
    """
    tokens = [CROP_ALIASES.get(t, t) for t in re.findall(r"[a-z]+", question.lower())]
    stems = {_stem(t) for t in tokens}
    best, best_len = None, 0
    for crop in crops:
        for w in re.findall(r"[a-z]+", crop.lower()):
            if len(w) >= 3 and w not in NAME_NOISE and _stem(w) in stems and len(w) > best_len:
                best, best_len = crop, len(w)
    return best


def district_risk(v: View) -> tuple[pd.DataFrame, str]:
    """
    Rank districts by price volatility (profiles table, or computed from prices as a fallback).

    Parameters
    ----------
    v : View
        Page view model.

    Returns
    -------
    tuple
        ``(ranking, kind)``; ranking has ``admin2_name``, ``score`` and optional ``cluster_name``.
        ``kind`` is ``"profile"`` or ``"computed"``.
    """
    p = v.profiles
    if not p.empty and {"admin2_name", "price_volatility"} <= set(p.columns):
        out = p.dropna(subset=["admin2_name", "price_volatility"])
        if v.districts:
            out = out[out["admin2_name"].isin(v.districts)]
        if not out.empty:
            cols = [c for c in ("admin2_name", "price_volatility", "cluster_name") if c in out.columns]
            out = out[cols].rename(columns={"price_volatility": "score"}).sort_values("score", ascending=False)
            return out.reset_index(drop=True), "profile"
    if "admin2_name" in v.df.columns and not v.df.empty:
        g = v.df.groupby(["admin2_name", "commodity_name"])[PRICE]
        cv = (g.std() / g.mean() * 100).replace([np.inf, -np.inf], np.nan).groupby(level=0).mean().dropna()
        out = cv.sort_values(ascending=False).rename("score").reset_index()
        return out, "computed"
    return pd.DataFrame(columns=["admin2_name", "score"]), "computed"


def _answer_price(crop: str, v: View) -> str:
    """
    Chat answer: current price and forecast of one crop.

    Parameters
    ----------
    crop : str
        Commodity name.
    v : View
        Page view model.

    Returns
    -------
    str
        Markdown reply.
    """
    y = crop_series(v.df, crop)
    if y.empty:
        return f"I have no price data for **{crop}** in the current filters. Widen the date range or districts."
    now = float(y.iloc[-1])
    change = f", {(now / float(y.iloc[-2]) - 1) * 100:+.1f}% vs the previous period" if len(y) > 1 and y.iloc[-2] > 0 else ""
    lines = [f"**{crop}** is at **PKR {now:,.0f}** per 40 kg ({y.index[-1]:%d %b %Y}{change})."]
    b = v.bundle(crop, 3)
    if b is not None:
        fc = b["fc"]
        path = " → ".join(f"{x:,.0f}" for x in fc["forecast"])
        lines.append(f"**Forecast** ({b['best']} model, next {len(fc)} periods): {path}. "
                     f"80% range at the end: {fc['lo'].iloc[-1]:,.0f} to {fc['hi'].iloc[-1]:,.0f}.")
    else:
        lines.append("There is too little history in the current filters to forecast this crop.")
    est = ml_next_estimate(crop, v.token)
    if est:
        lines.append(f"The XGBoost model estimates **{est['price']:,.0f}** next period ({est['chg_pct']:+.1f}%).")
    lines.append(f"Ask “should I sell {crop.lower()}?” for a sell/hold recommendation.")
    return "\n\n".join(lines)


def _answer_advice(crop: str, v: View) -> str:
    """
    Chat answer: run the Decision Advisor logic with default assumptions.

    Parameters
    ----------
    crop : str
        Commodity name.
    v : View
        Page view model.

    Returns
    -------
    str
        Markdown reply.
    """
    horizon, storage = 2, 0.8
    b = v.bundle(crop, horizon)
    if b is None or float(b["y"].iloc[-1]) <= 0:
        return f"There is not enough usable history for **{crop}** to give advice in the current filters."
    adv = compute_advice(b["y"], b["fc"], b["board"], b["sigma"], horizon, storage)
    return (f"**{crop}: {REC_EN[adv['rec']]}** (confidence {adv['conf']}/100)\n\n"
            f"{advice_reason(adv, 'en')}\n\n"
            f"Assumptions: {horizon}-period hold, {storage}% storage cost per period. "
            f"Open the 🤖 Decision Advisor page to use your own quantity and costs. "
            f"_Decision support, not financial advice._")


def _answer_risk(v: View) -> str:
    """
    Chat answer: district risk ranking.

    Parameters
    ----------
    v : View
        Page view model.

    Returns
    -------
    str
        Markdown reply.
    """
    rk, kind = district_risk(v)
    if rk.empty:
        return "District risk data is not available yet. Run the clustering step of the pipeline."
    unit = "volatility" if kind == "profile" else "average price variability"
    suffix = "" if kind == "profile" else "%"
    lines = [f"**Riskiest district: {rk.iloc[0]['admin2_name']}.** Top districts by {unit}:"]
    for i, r in enumerate(rk.head(5).itertuples(), 1):
        cluster = f" · {r.cluster_name}" if hasattr(r, "cluster_name") and pd.notna(r.cluster_name) else ""
        lines.append(f"{i}. **{r.admin2_name}** — {r.score:,.0f}{suffix}{cluster}")
    lines.append("\nSee the 🗺️ Geographic Risk page for the full map.")
    return "\n".join(lines)


def generate_chat_response(query: str, v: View) -> str:
    """
    Route a question to the right answer with keyword matching.

    Routing order: sell/hold/store, risk/district, crop name, friendly fallback.

    Parameters
    ----------
    query : str
        User text.
    v : View
        Page view model.

    Returns
    -------
    str
        Markdown reply.
    """
    ql = query.lower()
    crop = find_crop(query, v.commodities)
    try:
        if SELL_RE.search(ql):
            if crop:
                return _answer_advice(crop, v)
            return "Which crop are you thinking about? Available: " + ", ".join(v.commodities[:12]) + "."
        if "risk" in ql or "volatil" in ql or ("district" in ql and not crop):
            return _answer_risk(v)
        if crop:
            return _answer_price(crop, v)
    except Exception as exc:  # the chat must never take the page down
        return f"Sorry, I could not compute that ({type(exc).__name__}). Try another question."
    return T("chat_fallback")


def chat_submit(question: str, v: View) -> None:
    """
    Append a question and its answer to ``st.session_state["chat_history"]``.

    Parameters
    ----------
    question : str
        User text (ignored when empty).
    v : View
        Page view model.

    Returns
    -------
    None
    """
    question = (question or "").strip()
    if not question:
        return
    hist = st.session_state.setdefault("chat_history", [])
    hist.append({"role": "user", "content": question})
    hist.append({"role": "assistant", "content": generate_chat_response(question, v)})
    st.session_state["chat_history"] = hist[-30:]


def chat_clear() -> None:
    """
    Reset the conversation to the welcome message.

    Returns
    -------
    None
    """
    st.session_state["chat_history"] = [{"role": "assistant", "content": T("chat_welcome")}]


def render_chat(v: View) -> None:
    """
    Draw the "Ask MandiMind" expander in the sidebar (on every page).

    Call this after the page has been drawn: each message ends with ``st.rerun()``, and every
    page widget must already exist on that run so it keeps its value.

    Parameters
    ----------
    v : View
        Page view model.

    Returns
    -------
    None
    """
    if "chat_history" not in st.session_state:
        chat_clear()
    with st.expander(T("chat_launcher"), expanded=False):
        try:
            box = st.container(height=320)
        except TypeError:  # older Streamlit has no fixed-height containers
            box = st.container()
        with box:
            for message in st.session_state["chat_history"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        with st.form("chat_form", clear_on_submit=True):
            query = st.text_input(T("chat_placeholder"), key="chat_q", label_visibility="collapsed",
                                  placeholder=T("chat_placeholder"))
            sent = st.form_submit_button(T("chat_send"))
        if sent and query.strip():
            chat_submit(query, v)
            st.rerun()
        for i, suggestion in enumerate(CHAT_SUGGESTIONS):
            if st.button(suggestion, key=f"chat_sug_{i}"):
                chat_submit(suggestion, v)
                st.rerun()
        if st.button(T("chat_clear"), key="chat_clear_btn"):
            chat_clear()
            st.rerun()


# ============================================================================
# v8.0 - KILLER FEATURE 2a: CLIENT PORTFOLIO SIMULATOR (Pro)
# ============================================================================
def page_portfolio(v: View) -> None:
    """
    Render the Client Portfolio Simulator: turn a budget into a crop allocation.

    Every crop is scored by its storage-adjusted expected return divided by its forecast
    uncertainty; the budget is split over the best positive-score crops in proportion to the score.

    Parameters
    ----------
    v : View
        Page view model.

    Returns
    -------
    None
    """
    note(T("pf_intro"))
    left, right = st.columns([1, 2.6])
    with left:
        sec(T("pf_inputs"))
        budget = int(st.number_input(T("pf_budget"), min_value=10_000, value=1_000_000, step=50_000, key="pf_budget"))
        horizon = st.slider(T("pf_horizon"), 1, 6, 3, key="pf_horizon")
        storage = float(st.number_input(T("adv_storage"), 0.0, 10.0, 0.8, 0.1, key="pf_storage"))
        max_pos = st.slider(T("pf_max_pos"), 1, 6, 4, key="pf_max_pos")
    rows: list[dict[str, Any]] = []
    for crop in v.commodities:
        b = v.bundle(crop, horizon)
        if b is None or float(b["y"].iloc[-1]) <= 0:
            continue
        adv = compute_advice(b["y"], b["fc"], b["board"], b["sigma"], horizon, storage)
        now_px = adv["now_px"]
        cost = storage * horizon
        pess = (float(b["fc"]["lo"].iloc[-1]) / now_px - 1) * 100 - cost
        opt = (float(b["fc"]["hi"].iloc[-1]) / now_px - 1) * 100 - cost
        unc = max(float(adv["err_pct"]), 0.5) if np.isfinite(adv["err_pct"]) else 0.5
        rows.append({"crop": crop, "now": now_px, "expected": adv["exp_px"], "net": float(adv["net"]),
                     "pess": pess, "opt": opt, "unc": unc, "score": float(adv["net"]) / unc,
                     "conf": adv["conf"], "model": b["best"]})
    if not rows:
        with right:
            st.warning(T("fc_need_history"))
        return
    tbl = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    pick = tbl[(tbl["net"] > 0) & (tbl["score"] > 0)].head(max_pos).copy()
    rank_view = pd.DataFrame({
        "Crop": tbl["crop"], f"Price now ({T('unit')})": tbl["now"].round(0), "Expected price": tbl["expected"].round(0),
        "Net return %": tbl["net"].round(2), "Uncertainty ±%": tbl["unc"].round(2), "Score": tbl["score"].round(2),
        "Confidence": tbl["conf"], "Model": tbl["model"]})
    with right:
        if pick.empty:
            st.warning(T("pf_none"))
            sec(T("pf_rank"))
            show_df(rank_view)
            return
        pick["alloc"] = budget * pick["score"] / pick["score"].sum()
        pick["bags"] = np.floor(pick["alloc"] / pick["now"]).astype(int)
        pick["invested"] = pick["bags"] * pick["now"]
        pick["exp_val"] = pick["invested"] * (1 + pick["net"] / 100)
        pick["pess_val"] = pick["invested"] * (1 + pick["pess"] / 100)
        pick["opt_val"] = pick["invested"] * (1 + pick["opt"] / 100)
        invested = float(pick["invested"].sum())
        cash = budget - invested
        exp_total, pess_total = float(pick["exp_val"].sum()) + cash, float(pick["pess_val"].sum()) + cash
        profit = exp_total - budget
        k1, k2, k3, k4 = st.columns(4)
        kpi(k1, T("pf_kpi_invested"), f'{invested:,.0f}<small>{T("pkr")}</small>',
            T("pf_kpi_cash", x=nm(f"{cash:,.0f}")), BLUE)
        kpi(k2, T("pf_kpi_value"), f'{exp_total:,.0f}<small>{T("pkr")}</small>', f"{pick.iloc[0]['crop']} leads", PURPLE)
        kpi(k3, T("pf_kpi_profit"), f'{profit:+,.0f}<small>{T("pkr")}</small>',
            nm(pct_str(profit / budget * 100, 2)), GREEN if profit >= 0 else RED)
        kpi(k4, T("pf_kpi_down"), f'{pess_total - budget:+,.0f}<small>{T("pkr")}</small>', T("pf_kpi_down_sub"),
            RED if pess_total < budget else AMBER)
        st.write("")
        a, b2 = st.columns(2)
        with a:
            sec(T("pf_alloc"))
            fig = go.Figure(go.Bar(
                x=pick["invested"], y=pick["crop"], orientation="h",
                marker_color=[v.colors.get(c, BLUE) for c in pick["crop"]],
                text=[f"{x:,.0f}" for x in pick["invested"]], textposition="outside", cliponaxis=False,
                hovertemplate="%{y}: %{x:,.0f} PKR<extra></extra>"))
            style_plotly(fig, 320, legend=False, hover="closest")
            fig.update_xaxes(visible=False, range=[0, float(pick["invested"].max()) * 1.3 or 1.0])
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(margin=dict(l=10, r=40, t=30, b=10))
            show(fig, key="pf_alloc")
        with b2:
            sec(T("pf_range"))
            fig = go.Figure(go.Bar(
                x=pick["crop"], y=pick["net"], marker_color=[GREEN if x >= 0 else RED for x in pick["net"]],
                error_y=dict(type="data", symmetric=False, array=(pick["opt"] - pick["net"]).clip(lower=0),
                             arrayminus=(pick["net"] - pick["pess"]).clip(lower=0), color=INK, thickness=1.6),
                text=[f"{x:+.1f}%" for x in pick["net"]], textposition="outside", cliponaxis=False,
                hovertemplate="%{x}: %{y:+.1f}%<extra></extra>"))
            style_plotly(fig, 320, legend=False, hover="closest")
            fig.update_yaxes(title=dict(text="Net return %"))
            show(fig, key="pf_range")
        sec(T("pf_plan"))
        plan = pd.DataFrame({
            "Crop": pick["crop"], "Bags (40 kg)": pick["bags"], f"Price now ({T('unit')})": pick["now"].round(0),
            "Invested (PKR)": pick["invested"].round(0), "Expected value (PKR)": pick["exp_val"].round(0),
            "Pessimistic (PKR)": pick["pess_val"].round(0), "Optimistic (PKR)": pick["opt_val"].round(0),
            "Net return %": pick["net"].round(2), "Confidence": pick["conf"]})
        show_df(plan)
        note(T("pf_note"))
        st.download_button(T("pf_dl"), plan.to_csv(index=False).encode(),
                           file_name="mandimind_portfolio_plan.csv", mime="text/csv", key="pf_dl_btn")
        sec(T("pf_rank"))
        show_df(rank_view)
        st.caption(T("disclaimer_long"))


# ============================================================================
# v8.0 - KILLER FEATURE 2b: EXPORT CENTER (Pro)
# ============================================================================
def metrics_frame(metrics: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """
    Flatten the per-commodity XGBoost metrics into one table.

    Parameters
    ----------
    metrics : dict
        Content of ``metrics.json`` without the ``__meta__`` entry.

    Returns
    -------
    pd.DataFrame
        One row per commodity (empty if there are no metrics).
    """
    rows = []
    for crop, m in metrics.items():
        row: dict[str, Any] = {"commodity": crop}
        for k, val in m.items():
            if k == "top_features" and isinstance(val, list):
                row[k] = ", ".join(f'{f.get("feature")} ({float(f.get("importance", 0)):.2f})' for f in val)
            else:
                row[k] = val
        rows.append(row)
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Building forecasts for every commodity...")
def all_forecasts(horizon: int, d0_iso: str, d1_iso: str, districts: tuple[str, ...],
                  token: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Forecast and backtest leaderboard for every commodity under the current filters.

    Parameters
    ----------
    horizon : int
        Forecast horizon in periods.
    d0_iso, d1_iso : str
        Global date filter.
    districts : tuple of str
        Selected districts.
    token : str
        Data fingerprint.

    Returns
    -------
    tuple
        ``(forecasts, leaderboard)`` as DataFrames (empty when no crop has enough history).
    """
    df = filter_prices(load_core()["df"], pd.Timestamp(d0_iso), pd.Timestamp(d1_iso), list(districts))
    fcs, boards = [], []
    for crop in sorted(df["commodity_name"].dropna().unique()):
        b = forecast_bundle(crop, horizon, d0_iso, d1_iso, districts, token)
        if b is None:
            continue
        f = b["fc"].copy()
        f.insert(0, "commodity", crop)
        f["model"] = b["best"]
        fcs.append(f)
        bd = b["board"].copy()
        bd.insert(0, "commodity", crop)
        bd["winner"] = bd["Model"] == b["best"]
        boards.append(bd)
    return (pd.concat(fcs, ignore_index=True) if fcs else pd.DataFrame(),
            pd.concat(boards, ignore_index=True) if boards else pd.DataFrame())


def export_tables(v: View, horizon: int) -> dict[str, pd.DataFrame]:
    """
    Collect every table the app shows, keyed by file stem.

    Parameters
    ----------
    v : View
        Page view model (filters applied).
    horizon : int
        Forecast horizon for the forecast table.

    Returns
    -------
    dict
        Mapping file stem to DataFrame (tables that do not exist are omitted).
    """
    keep = [c for c in ("date", "commodity_name", "admin2_name", PRICE, "temperature_2m_max", "precipitation_sum")
            if c in v.df.columns]
    out: dict[str, pd.DataFrame] = {"prices": v.df[keep].sort_values(
        [c for c in ("commodity_name", "admin2_name", "date") if c in keep]).reset_index(drop=True)}
    if not v.an.empty:
        out["anomalies"] = v.an.drop(columns=["obs_date"], errors="ignore").reset_index(drop=True)
    if not v.profiles.empty:
        out["district_profiles"] = v.profiles.reset_index(drop=True)
    mf = metrics_frame(v.metrics)
    if not mf.empty:
        out["xgboost_metrics"] = mf
    fc, board = all_forecasts(horizon, v.d0.isoformat(), v.d1.isoformat(), v.districts, v.token)
    if not fc.empty:
        out["forecasts"] = fc
        out["backtest_leaderboard"] = board
    return out


@st.cache_data(show_spinner="Packing the data bundle...")
def build_bundle_zip(horizon: int, d0_iso: str, d1_iso: str, districts: tuple[str, ...], token: str,
                     a_token: tuple[float, int]) -> bytes:
    """
    Build a ZIP with every table as CSV plus the raw metrics JSON.

    Parameters
    ----------
    horizon : int
        Forecast horizon in periods.
    d0_iso, d1_iso : str
        Global date filter.
    districts : tuple of str
        Selected districts.
    token : str
        Data fingerprint.
    a_token : tuple
        Anomalies cache key.

    Returns
    -------
    bytes
        ZIP file content.
    """
    core = load_core()
    d0, d1, dl = pd.Timestamp(d0_iso), pd.Timestamp(d1_iso), list(districts)
    df = filter_prices(core["df"], d0, d1, dl)
    an = filter_anomalies(load_anomalies(a_token), d0, d1, dl)
    view = View(df=df, an=an, an_all=an, commodities=[], n_dist=0, d0=d0, d1=d1, districts=districts, token=token,
                a_token=a_token, colors={}, profiles=core["profiles"], metrics=core["metrics"], df_all=core["df"])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for stem, frame in export_tables(view, horizon).items():
            zf.writestr(f"{stem}.csv", frame.to_csv(index=False))
        zf.writestr("metrics.json", json.dumps({"per_commodity": core["metrics"], "meta": core["meta"]},
                                               indent=2, default=str))
        zf.writestr("README.txt", f"MandiMind v{APP_VERSION} data bundle\nWindow: {d0:%Y-%m-%d} to {d1:%Y-%m-%d}\n"
                                  f"Districts: {len(dl) or 'all'}\nForecast horizon: {horizon} periods\n"
                                  f"Generated: {datetime.now():%Y-%m-%d %H:%M}\n")
    return buf.getvalue()


@st.cache_data(show_spinner="Building the full workbook...")
def build_full_workbook(horizon: int, d0_iso: str, d1_iso: str, districts: tuple[str, ...], token: str,
                        a_token: tuple[float, int]) -> bytes:
    """
    Build one Excel workbook with every table on its own sheet.

    Parameters
    ----------
    horizon : int
        Forecast horizon in periods.
    d0_iso, d1_iso : str
        Global date filter.
    districts : tuple of str
        Selected districts.
    token : str
        Data fingerprint.
    a_token : tuple
        Anomalies cache key.

    Returns
    -------
    bytes
        The .xlsx file content.
    """
    core = load_core()
    d0, d1, dl = pd.Timestamp(d0_iso), pd.Timestamp(d1_iso), list(districts)
    df = filter_prices(core["df"], d0, d1, dl)
    an = filter_anomalies(load_anomalies(a_token), d0, d1, dl)
    view = View(df=df, an=an, an_all=an, commodities=[], n_dist=0, d0=d0, d1=d1, districts=districts, token=token,
                a_token=a_token, colors={}, profiles=core["profiles"], metrics=core["metrics"], df_all=core["df"])
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for stem, frame in export_tables(view, horizon).items():
            _or_placeholder(frame).to_excel(writer, sheet_name=stem[:31], index=False)
        _style_workbook(writer.book)
    return buf.getvalue()


def _fig_html(fig: go.Figure) -> bytes:
    """
    Serialise a Plotly figure to a standalone interactive HTML file.

    Parameters
    ----------
    fig : go.Figure
        Figure to export.

    Returns
    -------
    bytes
        UTF-8 HTML (Plotly.js is loaded from the CDN, so no extra package is needed).
    """
    return fig.to_html(include_plotlyjs="cdn", full_html=True, config=CHART_CONFIG).encode("utf-8")


def page_export(v: View) -> None:
    """
    Render the Export Center: every table, chart and metric, downloadable.

    Parameters
    ----------
    v : View
        Page view model.

    Returns
    -------
    None
    """
    note(T("ex_intro"))
    c1, c2, _ = st.columns([1.3, 1.3, 2.4])
    crop = c1.selectbox(T("ex_crop"), v.commodities, key="ex_crop")
    horizon = c2.slider(T("ex_horizon"), 1, 6, 3, key="ex_horizon")
    d0i, d1i = v.d0.isoformat(), v.d1.isoformat()
    tables = export_tables(v, horizon)

    sec(T("ex_tables"))
    labels = {"prices": "Prices", "anomalies": "Anomalies", "district_profiles": "District profiles",
              "xgboost_metrics": "XGBoost metrics", "forecasts": "Forecasts (all crops)",
              "backtest_leaderboard": "Backtest leaderboard"}
    cols = st.columns(3)
    for i, (stem, frame) in enumerate(tables.items()):
        cols[i % 3].download_button(f"⬇ {labels.get(stem, stem)} · CSV ({len(frame):,} rows)",
                                    frame.to_csv(index=False).encode(), file_name=f"mandimind_{stem}.csv",
                                    mime="text/csv", key=f"ex_csv_{stem}")
    st.write("")

    sec(T("ex_reports"))
    r1, r2, r3, r4 = st.columns(4)
    try:
        r1.download_button(f"⬇ {crop} workbook · Excel",
                           build_excel_report(crop, horizon, d0i, d1i, v.districts, v.token, v.a_token),
                           file_name=f"mandimind_analysis_{safe_filename(crop)}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="ex_xlsx_crop")
        r2.download_button("⬇ All tables · Excel",
                           build_full_workbook(horizon, d0i, d1i, v.districts, v.token, v.a_token),
                           file_name="mandimind_all_tables.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="ex_xlsx_all")
    except Exception as exc:
        st.warning(T("export_failed", err=str(exc)))
    try:
        pdf_bytes = make_advisory_pdf(crop, 100, min(horizon, 6), 0.8, d0i, d1i, v.districts, v.token)
    except Exception as exc:
        pdf_bytes = None
        st.warning(T("export_failed", err=str(exc)))
    if pdf_bytes:
        r3.download_button(f"⬇ {crop} advisory · PDF", pdf_bytes, file_name=f"mandimind_advisory_{safe_filename(crop)}.pdf",
                           mime="application/pdf", key="ex_pdf")
    elif not PDF_OK:
        r3.warning(T("pdf_unavailable"))
    try:
        r4.download_button("⬇ Everything · ZIP",
                           build_bundle_zip(horizon, d0i, d1i, v.districts, v.token, v.a_token),
                           file_name="mandimind_bundle.zip", mime="application/zip", key="ex_zip")
    except Exception as exc:
        st.warning(T("export_failed", err=str(exc)))
    st.write("")

    sec(T("ex_charts"))
    ch1, ch2, ch3 = st.columns(3)
    idx_fig = go.Figure()
    for c in v.commodities:
        s = v.df[v.df["commodity_name"] == c].groupby("date")[PRICE].mean().sort_index().dropna()
        if not s.empty and s.iloc[0] != 0:
            idx_fig.add_trace(go.Scatter(x=s.index, y=s / s.iloc[0] * 100, name=c, mode="lines",
                                         line=dict(color=v.colors.get(c, BLUE), width=2.4)))
    style_plotly(idx_fig, 420)
    idx_fig.update_layout(yaxis=dict(title=dict(text="Index (start = 100)")))
    ch1.download_button("⬇ Price index · HTML", _fig_html(idx_fig), file_name="mandimind_price_index.html",
                        mime="text/html", key="ex_chart_idx")
    g = v.df.groupby("commodity_name")[PRICE]
    cv = ((g.std() / g.mean()) * 100).replace([np.inf, -np.inf], np.nan).dropna().sort_values()
    vol_fig = go.Figure(go.Bar(x=cv.values, y=cv.index, orientation="h", marker_color=AMBER,
                               text=[f"{x:.1f}%" for x in cv.values], textposition="outside"))
    style_plotly(vol_fig, 420, legend=False, hover="closest")
    ch2.download_button("⬇ Volatility ranking · HTML", _fig_html(vol_fig), file_name="mandimind_volatility.html",
                        mime="text/html", key="ex_chart_vol")
    b = v.bundle(crop, horizon)
    if b is not None:
        y, fc = b["y"], b["fc"]
        fc_fig = go.Figure()
        fc_fig.add_trace(go.Scatter(x=y.index, y=y.values, name="Actual", line=dict(color=INK, width=2.6)))
        fc_fig.add_trace(go.Scatter(x=[y.index[-1]] + list(fc["date"]), y=[y.iloc[-1]] + list(fc["forecast"]),
                                    name=f"Forecast ({b['best']})", line=dict(color=GREEN, width=3, dash="dash")))
        style_plotly(fc_fig, 420)
        ch3.download_button(f"⬇ {crop} forecast · HTML", _fig_html(fc_fig),
                            file_name=f"mandimind_forecast_{safe_filename(crop)}.html", mime="text/html", key="ex_chart_fc")
    note(T("ex_png_hint"))

    sec(T("ex_metrics"))
    mf = tables.get("xgboost_metrics")
    m1, m2 = st.columns(2)
    if mf is not None:
        show_df(mf)
        m1.download_button("⬇ Metrics · CSV", mf.to_csv(index=False).encode(), file_name="mandimind_metrics.csv",
                           mime="text/csv", key="ex_metrics_csv")
    else:
        st.info(T("kpi_no_metrics"))
    m2.download_button("⬇ Metrics · JSON", json.dumps({"per_commodity": v.metrics}, indent=2, default=str).encode(),
                       file_name="mandimind_metrics.json", mime="application/json", key="ex_metrics_json")
    st.caption(T("disclaimer_long"))


# ============================================================================
# v8.0 - KILLER FEATURE 6: PERFORMANCE & ELEGANCE POLISH
# ============================================================================
def skeleton_html() -> str:
    """
    Grey shimmer placeholder shown while the data loads for the first time.

    Returns
    -------
    str
        HTML for a hero bar, four KPI boxes and two chart boxes.
    """
    return ('<div class="sk sk-hero"></div><div class="sk-grid">'
            + '<div class="sk sk-kpi"></div>' * 4
            + '<div class="sk sk-wide"></div><div class="sk sk-side"></div></div>')


def render_footer() -> None:
    """
    Draw the "Made with love in Pakistan" footer.

    Returns
    -------
    None
    """
    st.markdown(f'<div class="mm-footer">{T("footer_made")} · MandiMind v{APP_VERSION}</div>', unsafe_allow_html=True)


BACK_TO_TOP_JS = """
<script>
(function () {
  try {
    var d = window.parent.document;
    var old = d.getElementById('mm-top-btn');
    if (old) { old.remove(); }  // its handlers belong to a previous iframe
    var btn = d.createElement('button');
    btn.id = 'mm-top-btn'; btn.type = 'button'; btn.innerHTML = '&#8593;'; btn.title = 'Back to top';
    btn.style.cssText = 'position:fixed;left:24px;bottom:24px;z-index:9999;width:46px;height:46px;border-radius:50%;'
      + 'border:1px solid rgba(255,255,255,.7);background:#10B981;color:#fff;font-size:21px;font-weight:700;cursor:pointer;'
      + 'box-shadow:0 14px 28px -12px rgba(15,23,42,.65);opacity:0;pointer-events:none;transition:opacity .25s,transform .25s;';
    d.body.appendChild(btn);
    function scroller() {
      return d.querySelector('[data-testid="stMain"]') || d.querySelector('section.main') || d.scrollingElement;
    }
    function place() {
      var sb = d.querySelector('section[data-testid="stSidebar"]');
      var w = sb ? sb.getBoundingClientRect().width : 0;
      btn.style.left = (w + 24) + 'px';
      var s = scroller();
      var y = s ? s.scrollTop : 0;
      btn.style.opacity = y > 400 ? '1' : '0';
      btn.style.pointerEvents = y > 400 ? 'auto' : 'none';
    }
    btn.addEventListener('click', function () {
      var s = scroller();
      if (s && s.scrollTo) { s.scrollTo({ top: 0, behavior: 'smooth' }); }
    });
    var s0 = scroller();
    if (s0) { s0.addEventListener('scroll', place, { passive: true }); }
    window.parent.addEventListener('resize', place);
    setInterval(place, 800);
    place();
  } catch (e) { /* sandboxed: the button simply does not appear */ }
})();
</script>
"""


def render_back_to_top() -> None:
    """
    Add the "back to top" button (bottom-left, appears after scrolling).

    A tiny zero-height component injects the button into the parent page because Streamlit's
    scroll container cannot be reached with plain markdown. If components are unavailable,
    a static anchor link is drawn instead.

    Returns
    -------
    None
    """
    try:
        if hasattr(st, "iframe"):  # newer Streamlit: components.html is deprecated in favour of st.iframe
            st.iframe(BACK_TO_TOP_JS, height=1)
        else:
            import streamlit.components.v1 as components

            components.html(BACK_TO_TOP_JS, height=0)
    except Exception:
        st.markdown(f'<a class="to-top-fallback" href="#" title="{T("back_to_top")}">↑</a>', unsafe_allow_html=True)


CHANGELOG_HTML = (
    '<ul class="chlog">'
    "<li><b>Why This Matters</b> figures redesigned</li>"
    "<li><b>Ask MandiMind</b> moved into the sidebar</li>"
    "<li><b>Risk map</b>: separated bubbles and readable labels</li>"
    "<li><b>Slower, seamless</b> market ticker</li>"
    "<li><b>Glowing</b> HOLD and SELL NOW cards</li>"
    "<li><b>Gold PRO badge</b> and restyled password box</li>"
    "<li>Softer forecast band</li>"
    "</ul>"
)


def render_user_chip(slot: Any, pro: bool) -> None:
    """
    Draw the sidebar profile chip, with a gold PRO badge next to the name when unlocked.

    Parameters
    ----------
    slot : Any
        Placeholder reserved above the Pro toggle.
    pro : bool
        Whether Pro mode is unlocked.

    Returns
    -------
    None
    """
    name = str(st.session_state.get("user_name_saved") or "").strip() or DEFAULT_USER
    badge = '<span class="gold-pro">PRO</span>' if pro else ""
    slot.markdown(f'<div class="user-chip{" is-pro" if pro else ""}"><div class="av">{_html.escape(name[0].upper())}</div>'
                  f'<div class="who"><div class="row"><span class="name">{_html.escape(name)}</span>{badge}</div>'
                  f'<div class="plan">{T("user_pro" if pro else "user_free")}</div></div></div>',
                  unsafe_allow_html=True)


def save_user_name() -> None:
    """
    Callback: copy the typed display name into a key that survives the widget being hidden.

    Returns
    -------
    None
    """
    st.session_state["user_name_saved"] = str(st.session_state.get("user_name", "")).strip()


def render_sidebar_pro() -> bool:
    """
    Draw the Pro toggle and password prompt in the sidebar.

    Returns
    -------
    bool
        True when Pro mode is switched on and the password was accepted.
    """
    st.toggle(T("pro_toggle"), key="pro_on")
    if not st.session_state.get("pro_on"):
        st.session_state["pro_ok"] = False
        st.session_state.pop("pro_pwd", None)
        return False
    if not st.session_state.get("pro_ok"):
        pwd = st.text_input(T("pro_pwd"), type="password", key="pro_pwd")
        if pwd:
            if pwd == PRO_PASSWORD:
                st.session_state["pro_ok"] = True
                st.rerun()
            st.markdown(f'<div class="pro-err">{T("pro_wrong")}</div>', unsafe_allow_html=True)
        else:
            st.caption(T("pro_hint"))
        return False
    st.session_state["user_name"] = st.session_state.get("user_name_saved", "")
    st.text_input(T("user_name"), key="user_name", max_chars=28, placeholder=DEFAULT_USER, on_change=save_user_name)
    st.caption(T("pro_active"))
    return True


# ============================================================================
# MAIN
# ============================================================================
def route_page(page: str, v: View) -> None:
    """
    Draw the hero and the selected page (with a Pro gate on the paid pages).

    Parameters
    ----------
    page : str
        Page key from the navigation radio.
    v : View
        Page view model.

    Returns
    -------
    None
    """
    if page in PRO_PAGES and not st.session_state.get("pro_ok"):
        st.warning(T("pro_locked"))
        return
    if page == "home":
        page_home(v)
        return
    if v.df.empty or not v.commodities:
        st.warning(T("no_data_filters"))
        return
    title = T(f"nav_{page}").split(" ", 1)[-1]
    sub = T("hero_sub", d0=nm(f"{v.d0:%d %b %Y}"), d1=nm(f"{v.d1:%d %b %Y}"), nc=nm(len(v.commodities)),
            nd=nm(v.n_dist), nobs=nm(f"{len(v.df):,}"))
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{sub}</p></div>', unsafe_allow_html=True)
    {"overview": page_overview, "geo": page_geo, "forecast": page_forecast, "advisor": page_advisor,
     "whatif": page_whatif, "portfolio": page_portfolio, "export": page_export}.get(page, page_overview)(v)


def main() -> None:
    """
    App entry point: state defaults, skeleton, data loading, sidebar, filters, routing and widgets.

    Returns
    -------
    None
    """
    st.session_state.setdefault("lang", "en")
    st.session_state.setdefault("page", "home")
    st.session_state.setdefault("alerts_n", 0)
    st.session_state.setdefault("alerts_ts", datetime.now().strftime("%H:%M:%S"))
    inject_css()
    boot = st.empty()
    if not st.session_state.get("_booted"):
        boot.markdown(skeleton_html(), unsafe_allow_html=True)
    try:
        core = load_core()
    except Exception as exc:
        boot.empty()
        st.error(T("err_load", err=exc))
        st.info(T("err_run_pipeline"))
        st.stop()
        return
    boot.empty()
    st.session_state["_booted"] = True
    df_all: pd.DataFrame = core["df"]
    if df_all.empty:
        st.error(T("err_load", err="no usable rows after feature engineering"))
        st.info(T("err_run_pipeline"))
        st.stop()
        return
    token = data_token(df_all)
    a_token = alert_token()
    an_all = load_anomalies(a_token)
    colors = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(sorted(df_all["commodity_name"].dropna().unique()))}

    with st.sidebar:
        st.markdown("<h1 style='color:#10B981;margin-bottom:0'>🌾 MandiMind</h1>", unsafe_allow_html=True)
        st.caption(T("tagline"))
        st.radio(T("lang_label"), ["en", "ur"], format_func=lambda k: {"en": "English", "ur": "اردو"}[k],
                 horizontal=True, key="lang")
        st.divider()
        chip_slot = st.empty()
        with st.container():  # one fixed slot, so widgets below do not shift when Pro fields appear
            pro = render_sidebar_pro()
        render_user_chip(chip_slot, pro)
        pages = PAGES + PRO_PAGES if pro else PAGES
        sanitize_choice("page", pages)
        page = st.radio("NAVIGATION", pages, format_func=lambda k: T(f"nav_{k}"), key="page",
                        label_visibility="collapsed")
        render_alerts_sidebar(a_token)
        chat_slot = st.container()  # filled at the end of the run, once the page view model is ready
        st.divider()
        st.markdown(f"**{T('filters_title')}**")
        dmin, dmax = df_all["date"].min().date(), df_all["date"].max().date()
        dstart = max(dmin, (df_all["date"].max() - pd.DateOffset(years=3)).date())
        date_range = st.date_input(T("date_range"), (dstart, dmax), min_value=dmin, max_value=dmax, key="date_range")
        all_d = sorted(df_all["admin2_name"].dropna().unique()) if "admin2_name" in df_all else []
        sanitize_choice("districts", all_d, multi=True)
        if "districts" not in st.session_state:
            st.session_state["districts"] = list(all_d)
        sel_d = st.multiselect(T("districts"), all_d, key="districts")
        st.divider()
        st.markdown(f'<span class="ver-badge">VERSION {APP_VERSION}</span>', unsafe_allow_html=True)
        with st.expander(T("changelog_title")):
            st.markdown(CHANGELOG_HTML, unsafe_allow_html=True)
        st.caption(T("sidebar_footer"))

    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        d0, d1 = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    else:
        d0, d1 = pd.Timestamp(dmin), pd.Timestamp(dmax)
    districts = tuple(sel_d)
    df = filter_prices(df_all, d0, d1, list(districts))
    an = filter_anomalies(an_all, d0, d1, list(districts))
    commodities = sorted(df["commodity_name"].dropna().unique())
    n_dist = int(df["admin2_name"].nunique()) if "admin2_name" in df else 0
    v = View(df=df, an=an, an_all=an_all, commodities=commodities, n_dist=n_dist, d0=d0, d1=d1,
             districts=districts, token=token, a_token=a_token, colors=colors,
             profiles=core["profiles"], metrics=core["metrics"], df_all=df_all)

    render_ticker(token)
    route_page(page, v)
    render_footer()
    render_back_to_top()
    with chat_slot:
        render_chat(v)  # last, because a new message calls st.rerun()


main()