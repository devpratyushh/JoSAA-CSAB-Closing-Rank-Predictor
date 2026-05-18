import os
import pickle
import re
import urllib.parse
import sqlite3


import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pipeline.config import PREDICT_YEAR, SOURCES, MODEL_DIR
from pipeline.predict import predict
from pipeline.institute_data import INSTITUTE_DOMAINS, GFTI_PERKS
from pipeline.gfti_fees_db import GFTI_FEE_DATA
from pipeline.choices import get_choices, toggle_choice

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


def _short_institute_name(inst: str) -> str:
    inst = inst.split("(")[0].strip()
    replacements = [
        (r"^Indian Institute of Technology, Design & Manufacturing\b", "IIITDM"),
        (r"^Indian Institute of Information Technology, Design & Manufacturing\b", "IIITDM"),
        (r"^Indian Institute of Information Technology\b", "IIIT"),
        (r"^International Institute of Information Technology\b", "IIIT"),
        (r"^National Institute of Technology\b", "NIT"),
        (r"^Indian Institute of Technology\b", "IIT"),
        (r"^Indian Institute of Engineering Science and Technology\b", "IIEST"),
        (r"^Indian Institute of Science Education and Research\b", "IISER"),
        (r"^Indian Institute of Science\b", "IISc"),
    ]
    for pattern, replacement in replacements:
        inst = re.sub(pattern, replacement, inst, count=1)
    inst = re.sub(r"\bUniversity\b", "Univ.", inst)
    inst = re.sub(r"\bInstitute\b", "Inst.", inst)
    inst = re.sub(r"\s+,\s+", ", ", inst)
    if len(inst) > 32:
        inst = inst[:29] + "…"
    return inst


def _short_program_name(prog: str) -> str:
    prog = prog.strip()
    replacements = [
        (r"Computer Science and Engineering", "CSE"),
        (r"Computer Science & Engineering", "CSE"),
        (r"Computer Engineering", "CSE"),
        (r"Information Technology", "IT"),
        (r"Electronics and Communication Engineering", "ECE"),
        (r"Electronics & Communication Engineering", "ECE"),
        (r"Electrical Engineering", "EE"),
        (r"Mechanical Engineering", "ME"),
        (r"Civil Engineering", "CE"),
        (r"Chemical Engineering", "CHE"),
        (r"Aerospace Engineering", "AE"),
        (r"Biotechnology", "BT"),
        (r"Metallurgical Engineering", "Met. Engg."),
        (r"Mathematics and Computing", "MnC"),
        (r"Artificial Intelligence", "AI"),
        (r"Data Science", "DS"),
        (r"Electronics and Electrical Engineering", "EEE"),
        (r"Electronics & Electrical Engineering", "EEE"),
    ]
    for pattern, replacement in replacements:
        prog = re.sub(pattern, replacement, prog, flags=re.IGNORECASE)

    prog = re.sub(r"\s*\(4 Years?, Bachelor of Technology\)", "", prog, flags=re.IGNORECASE)
    prog = re.sub(r"\s*\(4 Years?, Bachelor of Engineering\)", "", prog, flags=re.IGNORECASE)
    prog = re.sub(r"\s*\(5 Years?, Bachelor of Technology\)", "", prog, flags=re.IGNORECASE)

    if len(prog) > 28:
        prog = prog[:25] + "…"
    return prog


def _get_location(inst: str) -> str:
    """Extract city/location from institute name."""
    s = inst.split("(")[0].strip()
    if "," in s:
        loc = s.split(",")[-1].strip()
    else:
        # Fallback for names without commas like 'IIT Bombay'
        words = s.split()
        if len(words) > 2 and words[-2].lower() not in ["of", "and", "&"]:
            loc = " ".join(words[-2:])
        elif words:
            loc = words[-1]
        else:
            loc = "Unknown"
    return f"📍 {loc}"


# Institute type classifier
INST_TYPE_ORDER = [
    "IIT",
    "NIT",
    "IIIT",
    "IISc / IISER",
    "Central University",
    "State University",
    "GFTI / Other",
]

INST_TYPE_ICON = {
    "IIT":                "🏛️",
    "NIT":                "🎓",
    "IIIT":               "💻",
    "IISc / IISER":       "🔬",
    "Central University": "🏫",
    "State University":   "🏫",
    "GFTI / Other":       "🏢",
}


def _get_institute_type(inst: str) -> str:
    """Classify an institute name into a broad type group."""
    s = inst.split("(")[0].strip()
    sl = s.lower()

    # IITs
    if re.match(r"^indian institute of technology\b", s, re.I):
        return "IIT"
    if re.match(r"^indian school of mines\b", s, re.I):          # ISM Dhanbad
        return "IIT"
    if re.match(r"^indian institute of engineering science and technology\b", s, re.I):  # IIEST Shibpur
        return "IIT"

    # NITs — named-variant NITs first, then the generic prefix
    nit_prefixes = [
        r"^national institute of technology\b",
        r"^malaviya national institute of technology\b",
        r"^maulana azad national institute of technology\b",
        r"^motilal nehru national institute of technology\b",
        r"^sardar vallabhbhai national institute of technology\b",
        r"^visvesvaraya national institute of technology\b",
        r"^dr\.? b\.?r\.? ambedkar national institute of technology\b",
        r"^north eastern regional institute of science and technology\b",  # NERIST
    ]
    for p in nit_prefixes:
        if re.match(p, s, re.I):
            return "NIT"

    # IIITs
    iiit_prefixes = [
        r"^indian institute of information technology\b",
        r"^international institute of information technology\b",
        r"^atal bihari vajpayee indian institute of information technology\b",
        r"^pt\.? dwarka prasad mishra indian institute of information technology\b",
        r"^indian institute of information technology design",
        r"^indian institute of information technology, design",
    ]
    for p in iiit_prefixes:
        if re.match(p, s, re.I):
            return "IIIT"

    # IISc / IISERs
    if re.match(r"^indian institute of science\b", s, re.I):
        return "IISc / IISER"
    if re.match(r"^indian institute of science education and research\b", s, re.I):
        return "IISc / IISER"

    # Central Universities (explicit names + generic patterns)
    central_uni_names = [
        "cu jharkhand",
        "central university",
        "jawaharlal nehru university",
        "tezpur university",
        "university of hyderabad",
        "mizoram university",
        "assam university",
        "north-eastern hill university",
        "hnb garhwal university",
        "gurukula kangri vishwavidyalaya",
        "gati shakti vishwavidyalaya",
        "dr. h. s. gour university",
        "guru ghasidas vishwavidyalaya",
    ]
    for name in central_uni_names:
        if name in sl:
            return "Central University"

    # State / technical universities
    state_uni_patterns = [
        r"university",
        r"vishwavidyalaya",
    ]
    for p in state_uni_patterns:
        if re.search(p, sl):
            return "State University"

    # Everything else  →  GFTI / Other
    return "GFTI / Other"


def _slot_legend_label(row: pd.Series) -> str:
    inst = _short_institute_name(row["Institute"])
    prog = _short_program_name(row["Academic Program Name"])
    return f"{inst}<br>{prog}"


@st.cache_data(show_spinner=False, ttl=86400)
def _get_institute_info(inst: str) -> dict:
    """Resolve logo URL and official website from curated INSTITUTE_DOMAINS lookup.
    Falls back to ui-avatars initials logo if no domain is found."""
    search_term = inst.split('(')[0].strip()
    fallback_logo = (
        f"https://ui-avatars.com/api/?name={urllib.parse.quote_plus(search_term)}"
        f"&size=64&background=4f6ef5&color=fff&rounded=true&bold=true"
    )
    info = {"logo": fallback_logo, "url": None}

    # --- curated lookup (longest matching key wins) ---
    best_key  = ""
    best_domain = None
    for key, domain in INSTITUTE_DOMAINS.items():
        if key.lower() in inst.lower() and len(key) > len(best_key):
            best_key   = key
            best_domain = domain

    if best_domain:
        info["logo"] = f"https://logo.clearbit.com/{best_domain}?size=64"
        info["url"]  = f"https://www.{best_domain}"

    return info


def _get_gfti_perks(inst: str) -> list[str]:
    """Return perk badges for a GFTI institute (empty list if none defined)."""
    best_key   = ""
    best_perks: list[str] = []
    for key, perks in GFTI_PERKS.items():
        if key.lower() in inst.lower() and len(key) > len(best_key):
            best_key   = key
            best_perks = perks
    return best_perks


def _find_fee_data(inst: str) -> dict | None:
    """Look up fee data for an institute by matching substrings (longest match wins)."""
    inst_lower = inst.lower()
    best_key  = ""
    best_data = None
    for key, data in GFTI_FEE_DATA.items():
        if key in inst_lower and len(key) > len(best_key):
            best_key  = key
            best_data = data
    return best_data


def _build_trajectory_fig(
    df: pd.DataFrame,
    round_cols: list[str],
    student_rank: int,
    source: str,
) -> go.Figure:
    fig = go.Figure()

    fig.add_hline(
        y=student_rank,
        line_dash="dash",
        line_color="black",
        line_width=2,
        annotation_text=f"  Your rank: {student_rank:,}",
        annotation_position="top right",
    )

    for _, row in df.iterrows():
        ys   = [row[r] for r in round_cols if isinstance(row.get(r), (int, float))]
        xs   = [r      for r in round_cols if isinstance(row.get(r), (int, float))]
        cat  = row["Category"]
        name = _slot_label(row)
        legend_name = _slot_legend_label(row)

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines+markers",
            name=legend_name,
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

    # Calculate Y-axis range to keep bottom at exactly 500k by default
    all_ranks = [student_rank]
    for _, row in df.iterrows():
        ys = [row[r] for r in round_cols if isinstance(row.get(r), (int, float))]
        all_ranks.extend(ys)
    
    if all_ranks:
        min_rank = min(all_ranks)
        padding = 1000
        y_range = [500000, max(0, min_rank - padding)]
    else:
        y_range = [500000, 0]

    _fg = "#1a1a1a"

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
            range=y_range,
            showgrid=True,
            gridcolor="#dddddd",
            tickformat=",",
            tickfont=dict(color=_fg),
            linecolor=_fg,
        ),
        height=540,
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.20,
            xanchor="left",
            x=0,
            font=dict(size=11, color=_fg),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#cccccc",
            borderwidth=1,
        ),
        margin=dict(l=60, r=20, t=60, b=170),
    )
    return fig


# Sidebar
with st.sidebar:
    st.title("College Predictor")
    st.caption(f"Predictions for **{PREDICT_YEAR}** counselling")
    st.markdown("---")

    # For now, JoSAA is disabled as per user request
    source = "csab"
    st.info("Counselling source: **CSAB**")
    cfg = SOURCES[source]

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
        value=400_000, step=100,
        help=(
            "Enter the rank corresponding to your selected Seat Type. "
            "Use CRL for OPEN / OPEN (PwD), and category rank for "
            "OBC-NCL / SC / ST / EWS (including PwD variants)."
        ),
    )

    quota = st.selectbox(
        "Quota",
        QUOTAS[source],
        help=(
            "**AI**: All India (open to everyone). IITs and IIITs use this exclusively. "
            "NITs have very few AI seats and they're highly competitive.\n\n"
            "**HS**: Home State. NIT seats for students from the same state as the NIT. "
            "Largest share of NIT seats: pick this if you're from that state.\n\n"
            "**OS**: Other State. NIT seats for students from outside the NIT's state.\n\n"
            "**GO**: Goa. special quota at NIT Goa for students from Goa.\n\n"
            "**JK**: Jammu & Kashmir. reserved for students domiciled in J&K.\n\n"
            "**LA**: Ladakh. reserved for students domiciled in Ladakh.\n\n"
        ),
    )
    seat_type = st.selectbox("Seat Type", SEAT_TYPES, index=SEAT_TYPES.index("EWS"))

    gender_raw = st.radio("Gender", ["Gender-Neutral", "Female-only"])
    gender = (
        "Female-only (including Supernumerary)"
        if gender_raw == "Female-only"
        else "Gender-Neutral"
    )

    include_reach = st.checkbox("Include reach colleges", value=True)
    include_5yr   = st.checkbox("Include 5-year courses (Dual Degree / Integrated)", value=False)

    coverage = st.select_slider(
        "How safe should the prediction be?",
        options=[0.80, 0.85, 0.90, 0.95],
        value=0.90,
        format_func=lambda x: f"{int(x*100)}%",
        help=(
            "This controls how wide the prediction band is. "
            "Higher values make the band wider and safer. "
            "Lower values make it narrower and stricter."
        ),
    )

    st.markdown('<p style="font-size:0.85rem;font-weight:600;color:#475569;margin-bottom:4px;margin-top:12px;">🏢 GFTI Quick Filter</p>', unsafe_allow_html=True)
    GFTI_OPTIONS = [
        "Assam University, Silchar",
        "Birla Institute of Technology, Deoghar Off-Campus",
        "Birla Institute of Technology, Mesra, Ranchi",
        "Birla Institute of Technology, Patna Off-Campus",
        "CU Jharkhand",
        "Central University of Haryana",
        "Central University of Jammu",
        "Central University of Rajasthan, Rajasthan",
        "Central institute of Technology Kokrajar, Assam",
        "Chhattisgarh Swami Vivekanada Technical University, Bhilai (CSVTU Bhilai)",
        "Delhi Technological University, Delhi",
        "Gati Shakti Vishwavidyalaya, Vadodara",
        "Gautam Buddha University, Greater Noida",
        "Ghani Khan Choudhary Institute of Engineering and Technology, Malda, West Bengal",
        "Gurukula Kangri Vishwavidyalaya, Haridwar",
        "HNB Garhwal University Srinagar (Garhwal)",
        "Indian Institute of Carpet Technology, Bhadohi",
        "Indian Institute of Handloom Technology(IIHT), Varanasi",
        "Indian Institute of Handloom Technology, Salem",
        "Indian Maritime University - Visakhapatnam Campus",
        "Indira Gandhi Delhi Technical University for Women, New Delhi",
        "Institute of Chemical Technology, Mumbai: Indian Oil Odisha Campus, Bhubaneswar",
        "Institute of Engineering and Technology, Dr. H. S. Gour University. Sagar (A Central University)",
        "Institute of Infrastructure, Technology, Research and Management-Ahmedabad",
        "Institute of Technology, Guru Ghasidas Vishwavidyalaya (A Central University), Bilaspur, (C.G.)",
        "Islamic University of Science and Technology Kashmir",
        "J.K. Institute of Applied Physics & Technology, Department of Electronics & Communication, University of Allahabad- Allahabad",
        "Jawaharlal Nehru University, Delhi",
        "Manipal Institute of Technology, Manipal",
        "Mizoram University, Aizawl",
        "National Institute of Advanced Manufacturing Technology, Ranchi",
        "National Institute of Food Technology Entrepreneurship and Management, Kundli",
        "National Institute of Food Technology Entrepreneurship and Management, Sonepat, Haryana",
        "National Institute of Food Technology Entrepreneurship and Management, Thanjavur",
        "National Institute of Food Technology, Entrepreneurship and Management (NIFTEM) - Thanjavur",
        "National Institute of Foundry & Forge Technology, Hatia, Ranchi",
        "Netaji Subhas University of Technology, Delhi",
        "North Eastern Regional Institute of Science and Technology, Nirjuli-791109 (Itanagar),Arunachal Pradesh",
        "North-Eastern Hill University, Shillong",
        "Pandit Deendayal Energy University, Gandhinagar",
        "Pondicherry Engineering College, Puducherry",
        "Puducherry Technological University, Puducherry",
        "Punjab Engineering College, Chandigarh",
        "Rajiv Gandhi National Aviation University, Fursatganj, Amethi (UP)",
        "Sant Longowal Institute of Engineering and Technology",
        "School of Engineering, Tezpur University, Napaam, Tezpur",
        "School of Planning & Architecture, Bhopal",
        "School of Planning & Architecture, New Delhi",
        "School of Planning & Architecture: Vijayawada",
        "School of Studies of Engineering and Technology, Guru Ghasidas Vishwavidyalaya, Bilaspur",
        "Shri G. S. Institute of Technology and Science Indore",
        "Shri Mata Devi University, Katra, Jammu & Kashmir",
        "University of Hyderabad",
        "lndian Institute of Food Processing Technology, Thanjavur, Tamil Naidu."
    ]
    
    def on_gfti_select():
        selected = st.session_state["quick_gfti_select"]
        if selected != "-- Select GFTI --":
            st.session_state["filter_inst"] = selected
        else:
            st.session_state["filter_inst"] = ""
            
    st.selectbox(
        "Select GFTI to auto-filter",
        options=["-- Select GFTI --"] + GFTI_OPTIONS,
        label_visibility="collapsed",
        key="quick_gfti_select",
        on_change=on_gfti_select
    )

    st.markdown("---")
    predict_btn = st.button("Predict", width="stretch", type="primary")

    current_inputs = (source, exam_type, rank, quota, seat_type, gender, include_reach, include_5yr, coverage)
    if st.session_state.get("last_inputs") != current_inputs:
        st.session_state.pop("results_df", None)
        st.session_state.pop("last_rank", None)


# Main area
st.title("JoSAA / CSAB Closing Rank Predictor")

if cfg.get("disclaimer"):
    st.warning(cfg["disclaimer"])

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
            coverage        = coverage,
            include_5yr     = include_5yr,
        )

    df["Location"] = df["Institute"].apply(_get_location)
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

export_cols = [
    "Category", "Institute", "Location", "Academic Program Name", "Quota", "Seat Type", "Gender",
    *round_cols, "Final Pred", "Lower", "Upper", "Years", "Seats",
]
export_df = df[[c for c in export_cols if c in df.columns]].copy()
csv_data = export_df.to_csv(index=False).encode("utf-8")

with st.expander("Export results", expanded=False):
    selected_export_cats = st.multiselect(
        "Categories to include in filtered export",
        options=["safe", "match", "reach"],
        default=["safe", "match"],
        help="Choose one or more categories for the filtered CSV download.",
    )

    e1, e2 = st.columns(2)
    with e1:
        st.download_button(
            "Export all categories as CSV",
            data=csv_data,
            file_name=f"{source}_{exam_type}_rank_{student_rank}_predictions.csv",
            mime="text/csv",
            width="stretch",
        )

    with e2:
        filtered_export_df = export_df[export_df["Category"].isin(selected_export_cats)]
        filtered_csv_data = filtered_export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Export selected categories as CSV",
            data=filtered_csv_data,
            file_name=(
                f"{source}_{exam_type}_rank_{student_rank}_"
                f"{'-'.join(selected_export_cats) if selected_export_cats else 'none'}_predictions.csv"
            ),
            mime="text/csv",
            width="stretch",
            disabled=not selected_export_cats,
        )

tab_table, tab_plot = st.tabs(["Results Table", "Trajectory Plot"])

# Results table
with tab_table:
    f1, f2 = st.columns(2)

    with f1:
        prog_kw = st.text_input(
            "Filter by branch",
            key="filter_prog",
            placeholder="e.g. CSE, Mechanical, Data Science, AI",
            help="Matches any part of the program name. 'CSE' finds all CS variants at once.",
        )

    with f2:
        inst_kw = st.text_input(
            "Filter by institute",
            key="filter_inst",
            placeholder="e.g. NIT Trichy, IIT Bombay, IIIT Hyderabad",
            help="Matches any part of the institute name.",
        )

    starred_set = get_choices()
    table_df = df.copy()
    table_df["⭐"] = table_df.apply(lambda r: (r["Institute"], r["Academic Program Name"]) in starred_set, axis=1)
    if prog_kw:
        table_df = table_df[
            table_df["Academic Program Name"].str.contains(prog_kw, case=False, na=False)
        ]
    if inst_kw:
        def _match_inst(inst):
            inst_lower = inst.lower()
            query_parts = inst_kw.lower().split()
            
            gfti_filter = False
            nit_filter = False
            iiit_filter = False
            iit_filter = False
            
            clean_parts = []
            for part in query_parts:
                if part == "gfti":
                    gfti_filter = True
                elif part in ("nit", "nits"):
                    nit_filter = True
                elif part in ("iiit", "iiits"):
                    iiit_filter = True
                elif part in ("iit", "iits"):
                    iit_filter = True
                else:
                    clean_parts.append(part)
                    
            is_iit = "indian institute of technology" in inst_lower or "indian school of mines" in inst_lower or "shibpur" in inst_lower
            is_nit = "national institute of technology" in inst_lower or "visvesvaraya national" in inst_lower or "motilal nehru" in inst_lower or "malaviya national" in inst_lower or "maulana azad" in inst_lower or "sardar vallabhbhai" in inst_lower or "ambedkar national" in inst_lower
            is_iiit = "information technology" in inst_lower and not is_iit and not is_nit
            is_gfti = not (is_iit or is_nit or is_iiit)
            
            if gfti_filter and not is_gfti:
                return False
            if nit_filter and not is_nit:
                return False
            if iiit_filter and not is_iiit:
                return False
            if iit_filter and not is_iit:
                return False
                
            for part in clean_parts:
                if part not in inst_lower:
                    return False
            return True
            
        mask = table_df["Institute"].apply(_match_inst)
        table_df = table_df[mask]

    if table_df.empty:
        st.info("No rows match the selected program/institute filter.")
    else:
        # Fetch raw historical data to embed in cells
        try:
            with sqlite3.connect("historical.db") as conn:
                tbl = source
                rnd_col = "Round" if source == "josaa" else "Special Round"
                query = f"""
                SELECT "Institute", "Academic Program Name", "{rnd_col}" as rnd, "Year", "Closing Rank"
                FROM {tbl}
                WHERE "Quota"=? AND "Seat Type"=? AND "Gender"=?
                """
                hist_df = pd.read_sql(query, conn, params=(quota, seat_type, gender))
                
                # Convert to numeric for proper sorting and filling
                hist_df["rnd"] = pd.to_numeric(hist_df["rnd"], errors="coerce")
                hist_df["Closing Rank"] = pd.to_numeric(hist_df["Closing Rank"], errors="coerce")
                
                # Pivot, forward-fill across rounds, and melt back
                pivot_df = hist_df.pivot(
                    index=["Institute", "Academic Program Name", "Year"],
                    columns="rnd",
                    values="Closing Rank"
                )
                pivot_df = pivot_df.ffill(axis=1)
                hist_df = pivot_df.reset_index().melt(
                    id_vars=["Institute", "Academic Program Name", "Year"],
                    var_name="rnd",
                    value_name="Closing Rank"
                ).dropna(subset=["Closing Rank"])
                
                for r in round_cols:
                    try:
                        rnd_num = int(r.replace("R", ""))
                        subset_hist = hist_df[hist_df["rnd"] == rnd_num].sort_values("Year", ascending=False)
                        
                        lookup = {}
                        for _, row in subset_hist.iterrows():
                            key = (row["Institute"], row["Academic Program Name"])
                            if key not in lookup:
                                lookup[key] = []
                            if len(lookup[key]) < 4:
                                lookup[key].append(f"'{str(row['Year'])[-2:]}: {int(row['Closing Rank']):,}")
                        
                        def _format_cell(row):
                            val = row[r]
                            if val == "-":
                                return val
                            try:
                                val_str = f"{int(float(val)):,}"
                            except ValueError:
                                val_str = str(val)
                            key = (row["Institute"], row["Academic Program Name"])
                            if key in lookup and lookup[key]:
                                hist_str = ", ".join(lookup[key])
                                return f"{val_str} ({hist_str})"
                            return val_str
                            
                        table_df[r] = table_df.apply(_format_cell, axis=1)
                    except Exception:
                        pass
        except Exception:
            pass


    has_seats     = "Seats" in df.columns and df["Seats"].notna().any()
    has_intervals = "Lower" in df.columns and "Upper" in df.columns
    display_cols = (
        ["⭐", "Institute", "Location", "Academic Program Name"]
        + round_cols
        + ["Final Pred"]
        + (["Lower", "Upper"] if has_intervals else [])
        + ["Years"]
        + (["Seats"] if has_seats else [])
    )
    cov_pct = int(st.session_state.get("last_inputs", (None,)*9)[-1] * 100) if has_intervals else 90
    col_cfg = {
        "⭐":         st.column_config.CheckboxColumn("⭐", width="small", help="Star this branch to save it to your choices"),
        "Location":   st.column_config.TextColumn("Location", width="small"),
        "Final Pred": st.column_config.NumberColumn("Final", format="%,d"),
        "Lower":      st.column_config.NumberColumn(
                          f"Lower ({cov_pct}%)",
                          format="%,d",
                          help=f"Lower bound of the {cov_pct}% prediction interval. "
                               "Your rank below this means the slot is Safe."),
        "Upper":      st.column_config.NumberColumn(
                          f"Upper ({cov_pct}%)",
                          format="%,d",
                          help=f"Upper bound of the {cov_pct}% prediction interval. "
                               "Your rank above this means the slot is out of reach."),
        "Years":      st.column_config.NumberColumn("Yrs",
                          help="Years of historical data for this slot"),
        "Seats":      st.column_config.NumberColumn("Seats",
                          help="Total seats available in this slot for the prediction year"),
        **{r: st.column_config.TextColumn(r, width="large") for r in round_cols},
    }

    for cat in ["safe", "match", "reach"]:
        subset = table_df[table_df["Category"] == cat]
        if subset.empty:
            continue
        color = CAT_COLOR[cat]
        icon  = CAT_ICON[cat]
        st.markdown(
            f"<h4 style='color:{color};margin-bottom:12px;margin-top:24px;'>"
            f"{icon} {cat.upper()} &nbsp; <small>({len(subset)} options)</small>"
            "</h4>",
            unsafe_allow_html=True,
        )

        display_cols_no_inst = [c for c in display_cols if c != "Institute"]

        # Tag each row with its institute type
        subset = subset.copy()
        subset["_inst_type"] = subset["Institute"].apply(_get_institute_type)

        # Group by institute type, then by institute
        for inst_type in INST_TYPE_ORDER:
            type_subset = subset[subset["_inst_type"] == inst_type]
            if type_subset.empty:
                continue

            type_icon = INST_TYPE_ICON.get(inst_type, "🏢")
            st.markdown(
                f"<h5 style='margin-top:18px;margin-bottom:8px;color:#555;'>"
                f"{type_icon} {inst_type} &nbsp;<small style='font-weight:normal;'>({len(type_subset)} option{'s' if len(type_subset) != 1 else ''})</small>"
                "</h5>",
                unsafe_allow_html=True,
            )

            for inst, inst_df in type_subset.groupby("Institute", sort=True):
                # Auto-expand when there are few options overall or just one college in the type
                auto_expand = len(subset) <= 15 or len(type_subset.groupby("Institute")) == 1
                with st.expander(
                    f"**{inst}**  ({len(inst_df)} branch{'es' if len(inst_df) > 1 else ''})",
                    expanded=auto_expand,
                ):
                    # Premium dynamic logo header + Website + Fee Button
                    info = _get_institute_info(inst)
                    inst_url = info["url"] or f"https://www.google.com/search?q={urllib.parse.quote_plus(inst)}"
                    
                    header_col, btn_col = st.columns([3, 1])
                    with header_col:
                        st.markdown(
                            f'''<div style="display: flex; align-items: center; margin-bottom: 15px;">
                              <img src="{info["logo"]}" onerror="this.onerror=null; this.src='https://ui-avatars.com/api/?name={urllib.parse.quote_plus(inst.split('(')[0].strip())}&size=64&background=random&color=fff&rounded=true';" style="width: 48px; height: 48px; border-radius: 8px; margin-right: 15px; border: 1px solid #ddd; object-fit: contain; background: white;">
                              <h5 style="margin: 0; padding: 0; font-weight: 800;"><a href="{inst_url}" target="_blank" style="text-decoration: none; color: #1a3c6e; font-weight: 800;">{inst}</a></h5>
                            </div>''',
                            unsafe_allow_html=True,
                        )
                    with btn_col:
                        fee_data = _find_fee_data(inst)
                        if fee_data:
                            fee_btn_key = f"fee_{inst}_{cat}"
                            if st.button("💰 Fee Structure", key=fee_btn_key, use_container_width=True):
                                if st.session_state.get("show_fee_for") == inst:
                                    st.session_state.pop("show_fee_for", None)
                                else:
                                    st.session_state["show_fee_for"] = inst
                        else:
                            fee_search_url = f"https://www.google.com/search?q={urllib.parse.quote_plus(inst + ' fee structure 2024-25')}"
                            st.link_button("💰 Fee Structure", fee_search_url, use_container_width=True)

                    # ── Inline fee structure display ──────────────────────
                    if fee_data and st.session_state.get("show_fee_for") == inst:
                        cur = fee_data["currency"]
                        with st.container(border=True):
                            st.markdown(
                                f"<h6 style='margin:0 0 4px 0;'>"
                                f"💰 {fee_data['short_name']} — {fee_data['program']} "
                                f"Fee Structure ({fee_data['academic_year']})"
                                f"</h6>",
                                unsafe_allow_html=True,
                            )

                            fc1, fc2 = st.columns(2)

                            with fc1:
                                st.markdown("**One-Time Fees (at admission)**")
                                ot_rows = []
                                for f in fee_data["one_time_fees"]:
                                    ot_rows.append({"Component": f["item"], "Amount": f"{cur}{f['amount']:,}"})
                                ot_total = sum(f["amount"] for f in fee_data["one_time_fees"])
                                ot_rows.append({"Component": "Total One-Time", "Amount": f"{cur}{ot_total:,}"})
                                st.dataframe(
                                    pd.DataFrame(ot_rows),
                                    hide_index=True,
                                    use_container_width=True,
                                    column_config={
                                        "Component": st.column_config.TextColumn("Component"),
                                        "Amount": st.column_config.TextColumn("Amount"),
                                    },
                                )

                            with fc2:
                                st.markdown("**Semester Fees (per semester)**")
                                sem_rows = []
                                for f in fee_data["semester_fees"]:
                                    sem_rows.append({"Component": f["item"], "Amount": f"{cur}{f['amount']:,}"})
                                sem_total = sum(f["amount"] for f in fee_data["semester_fees"])
                                sem_rows.append({"Component": "Total / Semester", "Amount": f"{cur}{sem_total:,}"})
                                st.dataframe(
                                    pd.DataFrame(sem_rows),
                                    hide_index=True,
                                    use_container_width=True,
                                    column_config={
                                        "Component": st.column_config.TextColumn("Component"),
                                        "Amount": st.column_config.TextColumn("Amount"),
                                    },
                                )

                            if fee_data.get("hostel_fees"):
                                st.markdown("**Hostel Fees (per semester)**")
                                hostel_rows = []
                                for h in fee_data["hostel_fees"]:
                                    hostel_rows.append({"Occupancy": h["type"], "Fee / Semester": f"{cur}{h['amount_per_sem']:,}"})
                                st.dataframe(
                                    pd.DataFrame(hostel_rows),
                                    hide_index=True,
                                    use_container_width=True,
                                )

                            yr1_total = ot_total + (sem_total * 2)
                            total_4yr = ot_total + (sem_total * 8)
                            st.markdown(
                                f"<div style='padding:8px 12px;border-radius:8px;"
                                f"background:linear-gradient(135deg,#1a1a2e,#16213e);"
                                f"color:#e2e8f0;margin:6px 0;'>"
                                f"<b>Estimated 1st Year:</b> {cur}{yr1_total:,} &nbsp;·&nbsp; "
                                f"<b>Estimated 4-Year Total:</b> {cur}{total_4yr:,} "
                                f"<span style='font-size:0.8em;opacity:0.7;'>(tuition only, excl. hostel &amp; mess)</span>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                            if fee_data.get("additional_notes"):
                                with st.expander("📝 Important Notes", expanded=False):
                                    for note in fee_data["additional_notes"]:
                                        st.markdown(f"- {note}")

                            src_cols = st.columns(2)
                            with src_cols[0]:
                                if fee_data.get("source_url"):
                                    st.link_button("🌐 Official Fee Page", fee_data["source_url"], use_container_width=True)
                            with src_cols[1]:
                                if fee_data.get("source_pdf"):
                                    st.link_button("📄 Download PDF", fee_data["source_pdf"], use_container_width=True)


                    # GFTI perks – shown only in CSAB mode
                    if source == "csab" and inst_type == "GFTI / Other":
                        perks = _get_gfti_perks(inst)
                        if perks:
                            badges_html = " ".join(
                                f"<span style='"
                                f"display:inline-block;margin:3px 4px;"
                                f"padding:3px 10px;border-radius:999px;"
                                f"background:linear-gradient(135deg,#1a1a2e,#16213e);"
                                f"color:#e2e8f0;font-size:0.78rem;"
                                f"border:1px solid #4f6ef5;'"
                                f">{p}</span>"
                                for p in perks
                            )
                            st.markdown(
                                f"<div style='margin-bottom:10px;line-height:2;'>{badges_html}</div>",
                                unsafe_allow_html=True,
                            )

                    def on_editor_change(editor_key, df_subset):
                        state = st.session_state[editor_key]
                        if "edited_rows" in state:
                            for row_idx, edits in state["edited_rows"].items():
                                if "⭐" in edits:
                                    row = df_subset.iloc[int(row_idx)]
                                    toggle_choice(row["Institute"], row["Academic Program Name"], edits["⭐"])

                    editor_key = f"editor_{inst}_{cat}"
                    st.data_editor(
                        inst_df[display_cols_no_inst].reset_index(drop=True),
                        width="stretch",
                        hide_index=True,
                        column_config=col_cfg,
                        disabled=[c for c in display_cols_no_inst if c != "⭐"],
                        key=editor_key,
                        on_change=on_editor_change,
                        args=(editor_key, inst_df.reset_index(drop=True))
                    )

        st.markdown("")

# Trajectory plot
with tab_plot:
    st.markdown(
        """
        <div style="padding:0.75rem 1rem;border:1px solid #d0d0d0;border-radius:0.5rem;
                    background:rgba(255,255,255,0.04);margin-bottom:0.75rem;">
        <ul style="margin:0;padding-left:1.2rem;">
            <li>Select colleges from your results to compare their predicted closing-rank trajectories across rounds.</li>
            <li>The dashed blue line marks <strong>your rank</strong>. Traces below the line indicate rounds where you would be eligible for that seat.</li>
            <li>Use <strong>full-screen</strong> view for the cleanest chart when plotting many institutes.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df["_label"] = df.apply(_slot_label, axis=1)

    default_labels = (
        df[df["Category"].isin(["safe", "match"])]
        .head(3)["_label"]
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
