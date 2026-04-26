"""
Streamlit web UI for the JoSAA / CSAB college admission predictor.

Prerequisites (install once):
    pip install streamlit plotly

Run:
    streamlit run app.py
"""

import os
import pickle

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pipeline.config import PREDICT_YEAR, SOURCES, MODEL_DIR
from pipeline.predict import predict

# Page config
st.set_page_config(
    page_title="JoSAA / CSAB Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constants
SEAT_TYPES = [
    "OPEN", "OPEN (PwD)",
    "EWS", "EWS (PwD)",
    "OBC-NCL", "OBC-NCL (PwD)",
    "SC", "SC (PwD)",
    "ST", "ST (PwD)",
]

QUOTAS = {
    "josaa": ["AI", "HS", "OS", "GO", "JK", "LA"],
    "csab":  ["AI", "HS", "OS", "JK", "LA"],
}

CAT_COLOR = {"safe": "#27ae60", "match": "#f39c12", "reach": "#e74c3c"}
CAT_ICON  = {"safe": "🟢", "match": "🟡", "reach": "🔴"}

# Model cache
@st.cache_resource(show_spinner="Loading model…")
def load_model_cached(source: str):
    cfg  = SOURCES[source]
    path = os.path.join(MODEL_DIR, cfg["model"])
    if not os.path.exists(path):
        # Auto-train from CSV if available (e.g. first deploy without pre-built pkl)
        csv_path = cfg["csv"]
        if os.path.exists(csv_path):
            os.makedirs(MODEL_DIR, exist_ok=True)
            from pipeline.train import train
            train(csv_path, model_path=path)
        else:
            return None, path
    with open(path, "rb") as f:
        return pickle.load(f), path


# Helpers
def _slot_label(row: pd.Series) -> str:
    inst  = row["Institute"].split("(")[0].strip()
    prog  = row["Academic Program Name"]
    prog  = prog if len(prog) <= 50 else prog[:47] + "…"
    return f"{inst} · {prog}"


def _build_trajectory_fig(
    df: pd.DataFrame,
    round_cols: list[str],
    student_rank: int,
    source: str,
) -> go.Figure:
    fig = go.Figure()

    # Student rank reference line
    fig.add_hline(
        y=student_rank,
        line_dash="dash",
        line_color="royalblue",
        line_width=2,
        annotation_text=f"  Your rank: {student_rank:,}",
        annotation_position="top right",
    )

    for _, row in df.iterrows():
        ys   = [row[r] for r in round_cols if isinstance(row.get(r), (int, float))]
        xs   = [r      for r in round_cols if isinstance(row.get(r), (int, float))]
        cat  = row["Category"]
        name = _slot_label(row)

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines+markers",
            name=name,
            line=dict(color=CAT_COLOR[cat], width=2),
            marker=dict(size=8),
            hovertemplate=(
                f"<b>{name}</b><br>"
                "Round: %{x}<br>"
                "Predicted closing rank: %{y:,}<br>"
                f"Category: {cat}"
                "<extra></extra>"
            ),
        ))

    _fg = "#1a1a1a"   # explicit dark text so it's readable on both light and dark Streamlit themes

    fig.update_layout(
        title=dict(
            text=f"Closing-rank trajectory: {source.upper()} {PREDICT_YEAR}",
            font=dict(size=16, color=_fg),
        ),
        font=dict(color=_fg),
        xaxis=dict(
            title=dict(text="Round", font=dict(color=_fg)),
            showgrid=True,
            gridcolor="#dddddd",
            tickmode="array",
            tickvals=round_cols,
            tickfont=dict(color=_fg),
            linecolor=_fg,
        ),
        yaxis=dict(
            title=dict(text="Predicted Closing Rank", font=dict(color=_fg)),
            autorange="reversed",
            showgrid=True,
            gridcolor="#dddddd",
            tickformat=",",
            tickfont=dict(color=_fg),
            linecolor=_fg,
        ),
        legend=dict(orientation="v", x=1.02, y=1, font=dict(size=11, color=_fg),
                    bgcolor="rgba(255,255,255,0.85)", bordercolor="#cccccc", borderwidth=1),
        height=540,
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


# Sidebar
with st.sidebar:
    st.title("College Predictor")
    st.caption(f"Predictions for **{PREDICT_YEAR}** counselling")
    st.markdown("---")

    source = st.radio("Counselling source", ["JoSAA", "CSAB"],
                      horizontal=True).lower()
    cfg = SOURCES[source]

    # CSAB has no IITs → only mains
    if source == "csab":
        exam_type = "mains"
        st.info("CSAB covers NITs / IIITs / GFTIs only (JEE Mains ranks).")
    else:
        exam_label = st.radio(
            "Exam",
            ["JEE Mains  →  NIT / IIIT / GFTI", "JEE Advanced  →  IIT"],
            help="JEE Advanced is exclusively for IIT admissions.",
        )
        exam_type = "advanced" if "Advanced" in exam_label else "mains"

    rank = st.number_input(
        "Your rank",
        min_value=1, max_value=1_000_000,
        value=10_000, step=100,
    )

    quota = st.selectbox("Quota", QUOTAS[source])
    seat_type = st.selectbox("Seat Type", SEAT_TYPES)

    gender_raw = st.radio("Gender", ["Gender-Neutral", "Female-only"])
    gender = (
        "Female-only (including Supernumerary)"
        if gender_raw == "Female-only"
        else "Gender-Neutral"
    )

    include_reach = st.checkbox("Include reach colleges", value=True)

    st.markdown("---")
    predict_btn = st.button("Predict", width="stretch", type="primary")

    # Clear cached results when any input changes
    current_inputs = (source, exam_type, rank, quota, seat_type, gender, include_reach)
    if st.session_state.get("last_inputs") != current_inputs:
        st.session_state.pop("results_df", None)
        st.session_state.pop("last_rank", None)


# Main area
st.title("JoSAA / CSAB Closing Rank Predictor")

if cfg.get("disclaimer"):
    st.warning(cfg["disclaimer"])

# Run prediction
if predict_btn:
    model, model_path = load_model_cached(source)
    if model is None:
        st.error(
            f"**Model and data not found.**\n\n"
            f"The app needs either:\n"
            f"- `models/{source}_model.pkl` (pre-trained, commit to the repo), **or**\n"
            f"- `{source}_ranks.csv` (raw data, app will auto-train on first load)\n\n"
            f"Run locally: `python predict_cli.py train --source {source}` "
            f"then commit `models/{source}_model.pkl`."
        )
        st.stop()

    with st.spinner("Computing predictions…"):
        df = predict(
            rank            = rank,
            exam_type       = exam_type,
            quota           = quota,
            seat_type       = seat_type,
            gender          = gender,
            model           = model,
            rounds          = cfg["rounds"],
            include_reach   = include_reach,
            safe_threshold  = cfg["safe_threshold"],
            reach_threshold = cfg["reach_threshold"],
        )

    st.session_state["results_df"]  = df
    st.session_state["last_rank"]   = rank
    st.session_state["last_inputs"] = current_inputs

# Display results
df = st.session_state.get("results_df")

if df is None:
    st.markdown(
        """
        ### How to use

        1. Choose your counselling source in the sidebar (**JoSAA** for main
           counselling, **CSAB** for the supplementary round).
        2. Fill in your exam type, rank, quota, seat type, and gender.
        3. Click **Predict** to see eligible colleges with predicted closing
           ranks for every round.
        4. Switch to the **Trajectory Plot** tab to compare how closing ranks
           evolve across rounds for your shortlisted colleges.
        """,
    )
    st.stop()

if df is None or df.empty:
    st.info("No matching colleges found for the given profile. Try widening your criteria.")
    st.stop()

round_cols = [c for c in df.columns if c.startswith("R") and c[1:].isdigit()]
student_rank = st.session_state.get("last_rank", rank)

# Summary metrics
counts = df["Category"].value_counts()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Safe",  counts.get("safe",  0))
c2.metric("Match", counts.get("match", 0))
c3.metric("Reach", counts.get("reach", 0))
c4.metric("Total", len(df))

tab_table, tab_plot = st.tabs(["Results Table", "Trajectory Plot"])

# Results table
with tab_table:
    display_cols = (
        ["Institute", "Academic Program Name"]
        + round_cols
        + ["Final Pred", "Years"]
    )
    col_cfg = {
        "Final Pred": st.column_config.NumberColumn("Final", format="%d"),
        "Years":      st.column_config.NumberColumn("Yrs",
                          help="Years of historical data for this slot"),
        **{r: st.column_config.NumberColumn(r, format="%d") for r in round_cols},
    }

    for cat in ["safe", "match", "reach"]:
        subset = df[df["Category"] == cat]
        if subset.empty:
            continue
        color = CAT_COLOR[cat]
        icon  = CAT_ICON[cat]
        st.markdown(
            f"<h4 style='color:{color};margin-bottom:6px'>"
            f"{icon} {cat.upper()} &nbsp; <small>({len(subset)} options)</small>"
            "</h4>",
            unsafe_allow_html=True,
        )
        st.dataframe(
            subset[display_cols].reset_index(drop=True),
            width="stretch",
            hide_index=True,
            column_config=col_cfg,
        )
        st.markdown("")

# Trajectory plot
with tab_plot:
    st.markdown(
        "Select colleges from your results to compare their predicted "
        "closing-rank trajectories across rounds. "
        "The dashed blue line marks **your rank**; traces below it indicate "
        "rounds where you would be eligible for that seat."
    )

    df["_label"] = df.apply(_slot_label, axis=1)

    # Default: top 6 safe+match options
    default_labels = (
        df[df["Category"].isin(["safe", "match"])]
        .head(6)["_label"]
        .tolist()
    )

    chosen_labels = st.multiselect(
        "Colleges to plot",
        options=df["_label"].tolist(),
        default=default_labels,
        help="You can select up to ~15 colleges; more than that becomes cluttered.",
    )

    if not chosen_labels:
        st.info("Select at least one college above to see the trajectory plot.")
    else:
        chosen_df = df[df["_label"].isin(chosen_labels)].copy()
        fig = _build_trajectory_fig(chosen_df, round_cols, student_rank, source)
        st.plotly_chart(fig, width="stretch")

        st.caption(
            "Y-axis is **inverted**: lower position on the chart = higher rank number "
            "= more accessible seat. Your rank line divides eligible seats (below) "
            "from too-competitive seats (above). Round R1 typically has the highest "
            "(most accessible) closing ranks; later rounds tighten as floating students "
            "fill seats."
        )
