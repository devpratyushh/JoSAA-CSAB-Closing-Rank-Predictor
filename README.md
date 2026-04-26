# JoSAA / CSAB College Admission Predictor

Predicts JoSAA and CSAB closing ranks for the upcoming counselling year using an ensemble of per-slot year-trend and round-progression models trained on historical data (2016–2025).

## Overview

Each unique seat slot (institute × programme × quota × seat type × gender) is modelled as a two-dimensional time series over years and rounds. The system outputs a full **R1–R6 closing-rank trajectory** for the target year, not just a single-round estimate.

Key finding: JoSAA closing ranks are **mean-reverting**, not trending. The historical median outperforms all linear extrapolation methods by ~23% on average. SVR with an RBF kernel achieves the best single-year MAE (3,406 on 2024) through kernel-induced mean reversion.

## Project Structure

```
JOSAA/
├── scrape_josaa.py        # Historical JoSAA scraper (2016–2024, Playwright)
├── scrape_josaa_2025.py   # JoSAA 2025 current-year scraper
├── scrape_csab.py         # Historical CSAB scraper (2021–2024)
├── scrape_csab_2025.py    # CSAB 2025 current-year scraper
├── predict_cli.py         # CLI: train / backtest / tune / predict
├── app.py                 # Streamlit web UI with interactive trajectory plot
├── compare_models.py      # Trend-model comparison framework
├── josaa_ranks.csv        # JoSAA dataset (~514k rows, 2016–2025)
├── csab_ranks.csv         # CSAB dataset (~47k rows, 2021–2025)
├── models/
│   ├── josaa_model.pkl    # Trained JoSAA slot models
│   └── csab_model.pkl     # Trained CSAB slot models
└── pipeline/
    ├── config.py          # Constants, hyperparameters, source configs
    ├── loader.py          # CSV loading, cleaning, quota normalisation
    ├── train.py           # SlotModel class and training loop
    ├── predict.py         # Eligibility filtering and per-round prediction
    └── evaluate.py        # Backtesting and ensemble weight tuning
```

## Installation

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install playwright scikit-learn pandas numpy streamlit plotly
playwright install chromium      # only needed for scrapers
```

## Quickstart

### 1. Collect data (skip if CSVs already present)

```bash
python scrape_josaa.py && python scrape_josaa_2025.py
python scrape_csab.py  && python scrape_csab_2025.py
```

Scraping takes ~4 hours for JoSAA and ~30 minutes for CSAB. The scrapers are resume-safe, a crash just requires a restart.

### 2. Train

```bash
python predict_cli.py train                  # JoSAA (default)
python predict_cli.py train --source csab
```

### 3. Backtest

```bash
python predict_cli.py backtest --year 2024
python predict_cli.py backtest --source csab
```

### 4. Tune ensemble weight (optional)

```bash
# Inspect the w vs MAE table
python predict_cli.py tune --source josaa --val-year 2024

# Write the optimal w into the model pickle
python predict_cli.py tune --source josaa --val-year 2024 --save
```

### 5. Predict (CLI)

```bash
python predict_cli.py predict \
    --rank 5000 --exam mains \
    --quota AI --seat-type OPEN \
    --gender Gender-Neutral
```

```bash
# CSAB (shows disclaimer automatically)
python predict_cli.py predict --source csab \
    --rank 8000 --exam mains \
    --quota AI --seat-type OPEN \
    --gender Gender-Neutral
```

### 6. Web UI

```bash
streamlit run app.py
```

Opens in the browser. Select source, exam type, rank, quota, seat type, and gender in the sidebar, then click **Predict**. Switch to the **Trajectory Plot** tab to compare predicted R1–R_max closing-rank trajectories for selected colleges.

## Prediction Categories

| Category | Condition |
|----------|-----------|
| Safe     | rank ≤ 0.80 × predicted close (JoSAA) / 0.60 × (CSAB) |
| Match    | 0.80 × pred < rank ≤ pred |
| Reach    | pred < rank ≤ 1.20 × pred (JoSAA) / 1.50 × (CSAB) |

CSAB thresholds are wider because CSAB MAE (~50,000) is ~10× higher than JoSAA MAE (~3,761), reflecting the inherent unpredictability of residual seat allocation.

## Trend Models

Pass `--trend-model <name>` to `train`, `backtest`, or `tune`. Default: `median`.

| Model | 2024 MAE | Notes |
|-------|----------|-------|
| SVR RBF | **3,406** | Best single-year; kernel decay → mean reversion |
| Median | 3,708 | Default; stable across 4 held-out years |
| Ridge | 4,173 | Shrinks slope; better than OLS, worse than Median |
| Weighted OLS | 4,984 | Recent years weighted ~10× more |
| Theil–Sen | 5,017 | Robust to outlier years |
| OLS | 5,028 | Baseline linear extrapolation |
| SVR Linear | 4,767 | Similar to Ridge in this regime |

## Backtesting Results (JoSAA, median model)

| Test year | Overall MAE |
|-----------|-------------|
| 2022 | 4,048 |
| 2023 | 3,636 |
| 2024 | 3,708 |
| 2025 | 3,761 |

CSAB overall MAE (2025, 4 training years): **49,869**.

## Data Sources

- JoSAA archive: https://josaa.admissions.nic.in/applicant/seatmatrix/openingclosingrankarchieve.aspx
- CSAB archive: https://csab.nic.in/

## Paper

A full write-up of the methodology, engineering decisions, and evaluation is in [`paper.tex`](paper.tex) / [`paper.pdf`](paper.pdf).
