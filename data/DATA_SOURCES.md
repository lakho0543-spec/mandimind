\# MandiMind — Data Sources \& Licences



This file is the \*\*provenance ledger\*\* for every dataset used by MandiMind.

Rule (spec Section 4): \*All sources must be free; confirm each licence

before any commercial use.\* Update this file \*\*before\*\* any download, and

never edit raw files — raw is evidence.



\---



\## 1. Market prices — WFP Food Prices for Pakistan (HDX)



\- \*\*Source:\*\* UN World Food Programme (WFP), via the Humanitarian Data

&#x20; Exchange (HDX).

\- \*\*Dataset page:\*\* https://data.humdata.org/dataset/wfp-food-prices-for-pakistan

\- \*\*Licence:\*\* CC BY 3.0 IGO — free for commercial and non-commercial use

&#x20; with attribution.

\- \*\*Download:\*\* in section 5B. Target file: `data/raw/wfp\_food\_prices\_pakistan.csv`.

\- \*\*Downloaded on:\*\* 2026-11-26, 1,571,969 bytes

\- \*\*SHA-256:\*\* E6000F09E3469D7EB36AF5FBFD2342451658B6DB5D0B5CB604F4A2E8FFE80C6D

\- \*\*Used for:\*\* MandiMind features F1–F11 (prices, forecasts, anomalies,

&#x20; clustering, advisor). This is our primary price source for Phase 1 MVP.



\---



\## 2. Weather — Open-Meteo Historical Weather API



\- \*\*Source:\*\* Open-Meteo (open source, no API key required).

\- \*\*Docs:\*\* https://open-meteo.com/en/docs/historical-weather-api

- **Licence:** CC BY-IGO (Creative Commons Attribution for Intergovernmental Organisations) — free for commercial and non-commercial use with attribution.

\- \*\*Download:\*\* in section 5C. Target file: `data/raw/weather\_lahore\_2019\_2024.json`.

\- \*\*Downloaded on:\*\* \_(to be filled in 5C — date + byte size)\_

\- \*\*SHA-256:\*\* \_(to be filled in 5C — checksum)\_

\- \*\*Used for:\*\* MandiMind features F3, F7, F9, F10 — forecast features,

&#x20; what-if lab, and map risk scores.



\---



\## 3. Crop area \& production — \_pending\_



\_(Will be filled in when we add Pakistan Economic Survey / FAOSTAT /

provincial crop reporting. Not required to close Phase 1 MVP; the MVP

needs prices + weather first.)\_



\## 4. Macro drivers — \_pending\_



\_(State Bank of Pakistan exchange rate, diesel price, CPI food. Used in

F7 association rules and F9 what-if. Will be added in Phase 1.5.)\_



\## 5. Administrative boundaries — \_pending\_



\_(HDX Pakistan admin-0 and admin-1 GeoJSON. Needed for F10 map.

Will be added when we build the map in Phase 4.)\_

