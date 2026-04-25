"""
Central config: column names, exam-type detection, category constants.
"""

# Raw CSV columns
COL_YEAR        = "Year"
COL_ROUND       = "Round"
COL_INSTITUTE   = "Institute"
COL_PROGRAM     = "Academic Program Name"
COL_QUOTA       = "Quota"
COL_SEAT_TYPE   = "Seat Type"
COL_GENDER      = "Gender"
COL_OPEN_RANK   = "Opening Rank"
COL_CLOSE_RANK  = "Closing Rank"

# Derived column added during loading
COL_EXAM_TYPE   = "Exam Type"   # "advanced" | "mains"

# Exam-type detection (matched against institute name, case-insensitive)
# IITs use JEE Advanced ranks; everything else uses JEE Mains ranks.
IIT_KEYWORDS = ["indian institute of technology", "iit "]

# Prediction target year (set to next year before running predict) 
PREDICT_YEAR = 2025

# Minimum historical data points to fit a trend model per combination
MIN_YEARS_FOR_TREND = 2

# Ensemble weight for combining year-trend vs round-ratio predictions
# 0.0 = pure round-ratio scaling; 1.0 = pure per-round year trend
# 0.5 balances both signals equally.
# NOTE: When TREND_MODEL = "median", both signals are median-based so this
# weight has little practical effect.
ENSEMBLE_WEIGHT = 0.5

# Default trend model for the year-signal component
# Backtesting across 2024 and 2025 shows "median" outperforms all trend-based
# models by 25-33%: JoSAA closing ranks mean-revert to historical averages
# rather than following linear trends.
DEFAULT_TREND_MODEL = "median"

# Rounds to predict 
JOSAA_ROUNDS = [1, 2, 3, 4, 5, 6]
CSAB_ROUNDS  = [1, 2]          # CSAB typically has 2 special rounds
ALL_ROUNDS   = JOSAA_ROUNDS    # default; overridden per source in CLI

# CSAB quota name normalisation (current-year page uses full names)
# Historical archive uses abbreviations; current-year page uses full strings.
# Map everything to the abbreviation so all years share the same key space.
CSAB_QUOTA_NORM: dict[str, str] = {
    "all india":            "AI",
    "home state":           "HS",
    "other state":          "OS",
    "home state for goa":   "GO",
    "jammu & kashmir (ut)": "JK",
    "jammu and kashmir":    "JK",
    "ladakh (ut)":          "LA",
    "ladakh":               "LA",
}

# Data source configs
import os

SOURCES = {
    "josaa": {
        "csv":        "josaa_ranks.csv",
        "model":      "josaa_model.pkl",
        "round_col":  "Round",
        "rounds":     JOSAA_ROUNDS,
        # safe ≤ 0.80xpred, match ≤ 1.00xpred, reach ≤ 1.20xpred
        "safe_threshold":  0.80,
        "reach_threshold": 1.20,
        "disclaimer": None,
    },
    "csab": {
        "csv":        "csab_ranks.csv",
        "model":      "csab_model.pkl",
        "round_col":  "Special Round",
        "rounds":     CSAB_ROUNDS,
        # Widen thresholds - CSAB MAE (~50k) is ~10x JOSAA MAE (~5k),
        # so tight safe/reach bands would be misleading.
        "safe_threshold":  0.60,
        "reach_threshold": 1.50,
        "disclaimer": (
            "CSAB NOTICE: These predictions apply only to institutes with "
            "leftover seats after JOSAA counselling is complete. An institute "
            "will NOT appear in CSAB if all its seats were filled during JOSAA. "
            "CSAB closing ranks are highly variable year-to-year (model MAE ~50,000); "
            "treat all categories as rough guidance only."
        ),
    },
}

# Model artifact paths 
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH  = os.path.join(MODEL_DIR, "josaa_model.pkl")   # default (JOSAA)
