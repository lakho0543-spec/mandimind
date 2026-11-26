"""
app/main.py — Phase 4: Streamlit dashboard.

This is the UI for MandiMind. It reads the pre-computed parquet files
and the metrics JSON. It does NOT do any training live.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add the repo root to sys.path so we can import `pipeline`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# --- Page Config ---
st.set_page_config(
    page_title="MandiMind | Pakistan Crop Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

@st.cache_data
def load_data():
    """Load pre-computed data into memory. Cached so it's instant."""
    from pipeline.features import build_features
    
    # 1. Load the main analysis table AND build features (for the naive forecast)
    df_raw = pd.read_parquet(PROCESSED_DIR / "analysis_table.parquet")
    df = build_features(df_raw)
    
    # 2. Load the anomalies we detected in Phase 3B
    anomalies = pd.read_parquet(PROCESSED_DIR / "anomalies.parquet")
    
    # 3. Load the metrics JSON we just saved
    with open(MODELS_DIR / "metrics.json", "r") as f:
        metrics = json.load(f)
        
    return df, anomalies, metrics

# Load the data
try:
    df, anomalies, metrics = load_data()
except Exception as e:
    st.error(f"Error loading data. Did you run the pipeline scripts? Error: {e}")
    st.stop()

# --- SIDEBAR ---
st.sidebar.title("🌾 MandiMind")
st.sidebar.markdown("Pakistan Crop Price & Supply-Shock Intelligence")
st.sidebar.divider()

# Navigation
page = st.sidebar.radio(
    "Navigate", 
    ["Home (Price Pulse)", "Crop Forecast", "Anomaly Detective", "The Advisor", "What-If Lab"]
)

# ==========================================
# PAGE 1: HOME (PRICE PULSE)
# ==========================================
if page == "Home (Price Pulse)":
    st.title("National Price Pulse")
    st.markdown("Live tracking of crop prices and volatility across Pakistan.")
    
    # --- Top Level Metrics ---
    col1, col2, col3 = st.columns(3)
    
    # Average price of all crops
    avg_price = df["price_pkr_per_40kg"].mean()
    col1.metric("Average Price (PKR/40kg)", f"{avg_price:,.0f}")
    
    # Total anomalies detected
    col2.metric("Anomalies Detected", len(anomalies))
    
    # Best performing crop (by model improvement)
    best_crop = max(metrics.items(), key=lambda x: x[1]['improvement_pct'])
    col3.metric("Best ML Forecast", best_crop[0], f"{best_crop[1]['improvement_pct']:.1f}% improvement")

    st.divider()
    
    # --- Line Chart: Price Trend ---
    st.subheader("Price Trend Over Time")
    
    # Filter to recent data for better visibility
    plot_df = df[df["date"] >= "2023-01-01"]
    
    # Let user pick a crop to view
    selected_crop = st.selectbox("Select Crop", df["commodity_name"].unique())
    
    # Filter chart
    chart_df = plot_df[plot_df["commodity_name"] == selected_crop]
    
    fig = px.line(
        chart_df, x="date", y="price_pkr_per_40kg",
        title=f"{selected_crop} Price Trend",
        labels={"price_pkr_per_40kg": "Price (PKR per 40kg)", "date": "Date"}
    )
    st.plotly_chart(fig, use_container_width=True)


# ==========================================
# PAGE 2: CROP FORECAST
# ==========================================
elif page == "Crop Forecast":
    st.title("Crop Forecasts")
    st.markdown("View our machine learning predictions vs the Naive baseline.")
    
    # Select crop
    selected_crop = st.selectbox("Select a Commodity", df["commodity_name"].unique())
    
    # Get metrics for this crop
    crop_metrics = metrics.get(selected_crop, {})
    
    if not crop_metrics:
        st.warning("No model metrics available for this crop.")
    else:
        col1, col2 = st.columns(2)
        col1.metric("ML Model MAE", f"{crop_metrics['mae_xgb']:.2f} PKR")
        col2.metric("Naive Baseline MAE", f"{crop_metrics['mae_naive']:.2f} PKR")
        
        if crop_metrics['use_ml']:
            st.success(f"✅ Our ML model beats the baseline by {crop_metrics['improvement_pct']:.1f}%!")
        else:
            st.warning(f"⚠️ Our ML model underperforms the baseline. Naive forecast is recommended.")
            
        # Show price history
        df_crop = df[df["commodity_name"] == selected_crop].sort_values("date")
        
        fig = go.Figure()
        # Actual prices
        fig.add_trace(go.Scatter(x=df_crop["date"], y=df_crop["price_pkr_per_40kg"], mode="lines", name="Actual Price"))
        
        # We simulate the forecast using the lag_1 (Naive)
        fig.add_trace(go.Scatter(x=df_crop["date"], y=df_crop["price_lag_1"], mode="lines", name="Naive Forecast", line=dict(dash="dash")))
        
        fig.update_layout(title=f"{selected_crop} Price & Naive Forecast")
        st.plotly_chart(fig, use_container_width=True)


# ==========================================
# PAGE 3: ANOMALY DETECTIVE
# ==========================================
elif page == "Anomaly Detective":
    st.title("🕵️ Anomaly Detective")
    st.markdown("Historical price shocks and their drivers.")
    
    if len(anomalies) == 0:
        st.info("No anomalies detected in the dataset.")
    else:
        # Show anomaly table
        st.subheader("Detected Anomalies")
        
        # Let user pick which anomaly to investigate
        anomaly_options = anomalies.apply(lambda row: f"{row['date'].strftime('%Y-%m-%d')} - {row['commodity_name']} ({row['anomaly_reason']})", axis=1)
        selected_anomaly = st.selectbox("Select an anomaly", anomaly_options)
        
        # Show details
        idx = anomaly_options[anomaly_options == selected_anomaly].index[0]
        row = anomalies.loc[idx]
        
        st.info(f"**Why did this happen?** {row['anomaly_reason']}")
        col1, col2, col3 = st.columns(3)
        col1.write(f"**Date:** {row['date'].strftime('%Y-%m-%d')}")
        col2.write(f"**Price:** {row['price_pkr_per_40kg']:.2f} PKR/40kg")
        col3.write(f"**Change:** {row['price_change_pct']:.1f}%")


# ==========================================
# PAGE 4: THE ADVISOR (SELL / HOLD / STORE)
# ==========================================
elif page == "The Advisor":
    st.title("🤖 The Advisor")
    st.markdown("Get a simple Sell, Hold, or Store recommendation for your crop.")
    
    col1, col2 = st.columns(2)
    crop = col1.selectbox("Crop", df["commodity_name"].unique())
    quantity = col2.number_input("Quantity (in 40kg bags)", min_value=1, value=100)
    
    if st.button("Get Recommendation"):
        crop_metrics = metrics.get(crop, {})
        
        if not crop_metrics:
            st.error("No data available for this crop.")
        else:
            # Simulated logic: Look at the recent price trend
            df_crop = df[df["commodity_name"] == crop].sort_values("date")
            latest_price = df_crop["price_pkr_per_40kg"].iloc[-1]
            previous_price = df_crop["price_lag_1"].iloc[-1]
            
            pct_change = ((latest_price - previous_price) / previous_price) * 100
            
            st.subheader(f"Analysis for {quantity} bags of {crop}")
            st.write(f"**Current Price:** {latest_price:.2f} PKR per 40kg")
            st.write(f"**Recent Trend:** {pct_change:+.1f}%")
            
            # Decision Logic
            if crop_metrics['use_ml']:
                if pct_change > 2:
                    st.success("🟢 **HOLD**: Prices are trending up. Wait to sell.")
                    st.write("Our ML model is highly confident in this crop, and the price is rising.")
                else:
                    st.error("🔴 **SELL NOW**: Prices are falling or flat. Sell to avoid losses.")
                    st.write("Our ML model is highly confident, but the trend is down.")
            else:
                st.warning("🟡 **STORE**: Our ML model cannot predict this crop better than a coin flip (Naive baseline). If storage is cheap, hold. Otherwise, sell to avoid risk.")
                st.write("This is a high-volatility crop where historical patterns don't hold.")

# ==========================================
# PAGE 5: WHAT-IF LAB
# ==========================================
elif page == "What-If Lab":
    st.title("🔬 What-If Lab")
    st.markdown("Simulate how a shock (like heavy rain or a fuel hike) affects the forecast.")
    
    col1, col2 = st.columns(2)
    crop = col1.selectbox("Crop", df["commodity_name"].unique())
    
    st.subheader("Simulate Shocks")
    rain_increase = st.slider("Rainfall Increase (%)", 0, 200, 0)
    diesel_increase = st.slider("Diesel Price Increase (%)", 0, 100, 0)
    
    # Simulated Effect
    # (In a real app, we would feed these sliders into the XGBoost model. 
    # For now, we use a heuristic to demonstrate the UI.)
    base_price = df[df["commodity_name"] == crop]["price_pkr_per_40kg"].iloc[-1]
    
    # Logic: Heavy rain increases prices for most crops (supply shock)
    # Diesel increases transport costs
    impact_multiplier = 1.0 + (rain_increase / 100 * 0.1) + (diesel_increase / 100 * 0.05)
    new_price = base_price * impact_multiplier
    
    st.divider()
    st.metric("Base Current Price", f"{base_price:.2f} PKR")
    st.metric("Simulated Future Price", f"{new_price:.2f} PKR", delta=f"{(new_price-base_price):.2f} PKR")
    
    st.info("Note: This is a heuristic simulation. Full model integration is coming in Phase 5.")