"""
Prediction engine.

Given a student profile, returns a ranked DataFrame of matching slots
with predicted closing ranks for every round (R1 … R6).

Output columns:
    Institute | Academic Program Name | Quota | Seat Type | Gender |
    R1 | R2 | R3 | R4 | R5 | R6 | Final Pred | Lower | Upper | Years | Seats | Category

Category (based on prediction interval vs student rank):
    safe   → rank ≤ lower bound of interval  (comfortably within range)
    match  → lower < rank ≤ Final Pred
    reach  → Final Pred < rank ≤ upper bound

Lower / Upper: per-slot prediction interval at the requested coverage level
    (default 90 %), derived from historical closing-rank variability.
    Slots with fewer than 2 data points fall back to ±20 % of Final Pred.

Seats column: current-year seat count from seat_matrix.csv (if available).
"""

import os
import pickle
import pandas as pd
from .config import MODEL_PATH, PREDICT_YEAR, ALL_ROUNDS

SEAT_MATRIX_PATH = os.path.join(os.path.dirname(__file__), "..", "seat_matrix.csv")


def load_model(path: str = MODEL_PATH) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_seat_matrix(path: str = SEAT_MATRIX_PATH) -> dict:
    """
    Load seat_matrix.csv and return a lookup dict:
        (institute, program, quota, seat_type, gender) -> seats (int)

    The seat matrix uses granular state names for NIT HS/OS quotas
    (e.g. "ANDHRA PRADESH" for home-state, "Other than ANDHRA PRADESH" for
    other-state). We aggregate these back to the coarse pipeline codes:
        AI              → "AI"
        <state name>    → "HS"   (home-state rows, not "Other than ...")
        Other than ...  → "OS"   (other-state rows)
        GO / JK / LA   → kept as-is
    Returns an empty dict if the file doesn't exist.
    """
    if not os.path.exists(path):
        return {}

    SPECIAL = {"AI", "GO", "JK", "LA"}

    df = pd.read_csv(path)
    df["_quota_norm"] = df["Quota"].apply(_coarse_quota)

    agg = (
        df.groupby(["Institute", "Program", "_quota_norm", "Seat Type", "Gender"],
                   sort=False)["Seats"]
        .sum()
        .reset_index()
    )

    return {
        (
            str(r["Institute"]).strip(),
            str(r["Program"]).strip(),
            str(r["_quota_norm"]).strip(),
            str(r["Seat Type"]).strip(),
            str(r["Gender"]).strip(),
        ): int(r["Seats"])
        for _, r in agg.iterrows()
    }


def _coarse_quota(raw: str) -> str:
    """Map fine-grained seat-matrix quota strings to pipeline quota codes."""
    s = str(raw).strip()
    upper = s.upper()
    if upper in {"AI", "GO", "JK", "LA"}:
        return upper
    if s.lower().startswith("other than"):
        return "OS"
    # Remaining rows are state-specific home-state quotas → HS
    return "HS"


def predict(
    rank:            int,
    exam_type:       str,       # "advanced" | "mains"
    quota:           str,       # "AI" | "HS" | "OS" | ...
    seat_type:       str,       # "OPEN" | "OBC-NCL" | "SC" | "ST" | "EWS" | ...
    gender:          str,       # "Gender-Neutral" | "Female-only (including Supernumerary)"
    model:           dict | None = None,
    year:            int = PREDICT_YEAR,
    rounds:          list[int] = ALL_ROUNDS,
    include_reach:   bool = True,
    safe_threshold:  float = 0.80,   # fallback when interval unavailable
    reach_threshold: float = 1.20,   # fallback when interval unavailable
    seat_matrix:     dict | None = None,  # (inst,prog,quota,st,gender) -> seats
    coverage:        float = 0.90,   # prediction interval coverage level
) -> pd.DataFrame:
    if model is None:
        model = load_model()

    # Load seat matrix lazily if not provided
    if seat_matrix is None:
        seat_matrix = load_seat_matrix()

    slots = model["slots"]

    results = []

    for key, slot_model in slots.items():
        inst, prog, q, st, g, et = key

        if et != exam_type or q != quota or st != seat_type or g != gender:
            continue

        # Predict all rounds (use per-model tuned weight if available)
        w = model.get("ensemble_weight")
        round_preds = slot_model.predict_all_rounds(year, rounds, w=w)

        # Final round = highest round this slot was seen in
        final_r = slot_model.max_round
        pred_final = round_preds.get(final_r) or round_preds[max(round_preds)]

        # Prediction interval for the final round
        has_intervals = hasattr(slot_model, "round_abs_deviations")
        if has_intervals:
            lower, upper = slot_model.predict_interval(final_r, year, coverage)
        else:
            # Old model pkl without interval support, fall back to fixed thresholds
            lower = safe_threshold * pred_final
            upper = reach_threshold * pred_final

        if rank <= lower:
            category = "safe"
        elif rank <= pred_final:
            category = "match"
        elif rank <= upper and include_reach:
            category = "reach"
        else:
            continue

        seats = seat_matrix.get((inst, prog, q, st, g))

        row = {
            "Institute":             inst,
            "Academic Program Name": prog,
            "Quota":                 q,
            "Seat Type":             st,
            "Gender":                g,
        }
        for r in rounds:
            row[f"R{r}"] = round_preds.get(r, "-")

        row["Final Pred"] = pred_final
        row["Lower"]      = int(round(lower))
        row["Upper"]      = int(round(upper))
        row["Years"]      = slot_model.n_years
        row["Seats"]      = seats
        row["Category"]   = category
        results.append(row)

    if not results:
        return pd.DataFrame()

    out = pd.DataFrame(results)
    cat_order = {"safe": 0, "match": 1, "reach": 2}
    out["_order"] = out["Category"].map(cat_order)
    out.sort_values(["_order", "Final Pred"], inplace=True)
    out.drop(columns=["_order"], inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out
