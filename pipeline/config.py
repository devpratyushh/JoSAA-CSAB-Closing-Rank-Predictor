"""
Central config: column names, exam-type detection, category constants.
"""

# ── Raw CSV columns ────────────────────────────────────────────────────────────
COL_YEAR        = "Year"
COL_ROUND       = "Round"
COL_INSTITUTE   = "Institute"
COL_PROGRAM     = "Academic Program Name"
COL_QUOTA       = "Quota"
COL_SEAT_TYPE   = "Seat Type"
COL_GENDER      = "Gender"
COL_OPEN_RANK   = "Opening Rank"
COL_CLOSE_RANK  = "Closing Rank"

# ── Derived column added during loading ───────────────────────────────────────
COL_EXAM_TYPE   = "Exam Type"   # "advanced" | "mains"

# ── Exam-type detection (matched against institute name, case-insensitive) ────
# IITs use JEE Advanced ranks; everything else uses JEE Mains ranks.
IIT_KEYWORDS = ["indian institute of technology", "iit "]

# ── Prediction target year (set to next year before running predict) ───────────
PREDICT_YEAR = 2025

# ── Minimum historical data points to fit a trend model per combination ────────
MIN_YEARS_FOR_TREND = 2

# ── Ensemble weight for combining year-trend vs round-ratio predictions ─────────
# 0.0 = pure round-ratio scaling; 1.0 = pure per-round year trend
# 0.5 balances both signals equally
ENSEMBLE_WEIGHT = 0.5

# ── Rounds to predict (even if some years had fewer) ──────────────────────────
ALL_ROUNDS = [1, 2, 3, 4, 5, 6]

# ── Model artifact paths ───────────────────────────────────────────────────────
import os
MODEL_DIR       = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH      = os.path.join(MODEL_DIR, "josaa_model.pkl")
