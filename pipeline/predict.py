"""
Prediction engine.

Given a student profile, returns a ranked DataFrame of matching slots
with predicted closing ranks for every round (R1 … R6).

Output columns:
    Institute | Academic Program Name | Quota | Seat Type | Gender |
    R1 | R2 | R3 | R4 | R5 | R6 | Final Pred | Category

Category (based on Final Pred vs student rank):
    safe   → rank ≤ 80 % of predicted final close
    match  → 80 % < rank ≤ 100 %
    reach  → 100 % < rank ≤ 120 %  (closing ranks shift year-to-year)
"""

import pickle
import pandas as pd
from .config import MODEL_PATH, PREDICT_YEAR, ALL_ROUNDS


def load_model(path: str = MODEL_PATH) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


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
    safe_threshold:  float = 0.80,
    reach_threshold: float = 1.20,
) -> pd.DataFrame:
    if model is None:
        model = load_model()

    slots     = model["slots"]
    slot_cols = model["slot_cols"]

    # slot_cols order: Institute, Program, Quota, SeatType, Gender, ExamType
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
        # Use highest available predicted round as the "final" benchmark
        pred_final = round_preds.get(final_r) or round_preds[max(round_preds)]

        ratio = rank / pred_final
        if ratio <= safe_threshold:
            category = "safe"
        elif ratio <= 1.0:
            category = "match"
        elif ratio <= reach_threshold and include_reach:
            category = "reach"
        else:
            continue

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
        row["Years"]      = slot_model.n_years
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
