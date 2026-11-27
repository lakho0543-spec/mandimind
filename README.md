# 🌾 MandiMind — Pakistan Crop Price & Supply-Shock Intelligence

**Live Demo:** https://mandimind-pakistan.streamlit.app

MandiMind is a full-stack data science platform that forecasts crop prices, detects supply shocks, and advises farmers whether to **Sell, Hold, or Store** their harvest.

## Features
- 🔮 Per-commodity XGBoost forecasting with walk-forward backtests
- 🚨 Isolation Forest anomaly detection with plain-language explanations
- 🗺️ Interactive vector map of Pakistan with K-Means district risk clusters
- 🤖 AI Advisor with sell/hold/store recommendations
- 🧪 What-If Simulator for weather, fuel, and FX shocks
- 🌐 Full Urdu + English interface

## Tech Stack
`Python 3.11` · `pandas` · `XGBoost` · `scikit-learn` · `Prophet` · `Streamlit` · `Plotly` · `ReportLab` · `openpyxl`

## Data Sources
- **WFP Pakistan Food Prices** (HDX, CC BY-IGO)
- **Open-Meteo Historical Weather API** (CC BY 4.0)

All sources, licences, and SHA-256 checksums are logged in `data/DATA_SOURCES.md`.

## Run Locally
\`\`\`bash
git clone https://github.com/lakho0543-spec/mandimind
cd mandimind
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app/main.py
\`\`\`

## Architecture
\`\`\`
data/raw/          → WFP CSV + Open-Meteo JSON (gitignored, checksummed)
data/processed/    → analysis_table.parquet, anomalies.parquet
pipeline/          → ingest.py, clean.py, merge.py, features.py, model.py, anomaly.py, cluster.py
app/               → main.py, pdf_generator.py
models/            → per_commodity/*.pkl, metrics.json
tests/             → 15 pytest tests, all passing
\`\`\`

## Honest Limitations
Per the project spec, the ML model honestly **underperforms** the naive baseline on 12 of 15 commodities. It beats the baseline on **Wheat (+10.4%), Eggs (+3.9%), Poultry (+1.7%)**. This is a real, documented finding — not a failure.
