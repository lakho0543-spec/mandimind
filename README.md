\# MandiMind — Pakistan Crop Price \& Supply-Shock Intelligence



Decision-support platform that forecasts crop prices, explains spikes,

clusters districts, and answers "sell, hold or store".



\## Status

Phase 0 — Foundation. Nothing runs yet.



\## Setup

1\. `python -m venv .venv`

2\. `.\\.venv\\Scripts\\Activate.ps1`

3\. `pip install -r requirements.txt`



\## Layout

\- data/raw, data/processed — never edit raw data

\- pipeline/ — ingest, clean, merge, features, model, precompute

\- models/ — saved models

\- app/ — Streamlit app

\- reports/ — generated PDFs

\- tests/ — unit tests

\- notebooks/ — exploration

