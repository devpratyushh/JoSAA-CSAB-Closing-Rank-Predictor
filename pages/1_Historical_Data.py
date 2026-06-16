"""
Historical closing-rank browser.
"""

import pandas as pd
import streamlit as st
import sqlite3
import os
import re
import plotly.graph_objects as go
import urllib.parse

from pipeline.config import IIT_KEYWORDS, PREDICT_YEAR, JOSAA_ROUNDS, CSAB_ROUNDS
from pipeline.gfti_fees_db import GFTI_FEE_DATA
from pipeline.choices import get_choices, toggle_choice


SEAT_TYPES = [
    "OPEN", "OPEN (PwD)",
    "EWS", "EWS (PwD)",
    "OBC-NCL", "OBC-NCL (PwD)",
    "SC", "SC (PwD)",
    "ST", "ST (PwD)",
]

QUOTA_NAMES = {
    "AI": "All India (AI)",
    "HS": "Home State (HS)",
    "OS": "Other State (OS)",
    "GO": "Goa (GO)",
    "JK": "Jammu & Kashmir (JK)",
    "LA": "Ladakh (LA)",
}

QUOTAS = {
    "josaa": ["AI", "HS", "OS", "GO", "JK", "LA"],
    "csab":  ["AI", "HS", "OS", "JK", "LA"],
}

DB_PATH = "historical.db"


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_years(table: str) -> list[int]:
    """Return sorted list of years by querying min and max (avoids loading all rows)."""
    if not os.path.exists(DB_PATH):
        return []
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(f'SELECT MIN(CAST("Year" AS INTEGER)), MAX(CAST("Year" AS INTEGER)) FROM {table}')
            row = cursor.fetchone()
            if row and row[0] is not None and row[1] is not None:
                return list(range(int(row[0]), int(row[1]) + 1))
    except Exception:
        pass
    return []


@st.cache_data(ttl=3600, show_spinner="Fetching data…")
def fetch_data(table: str, years: tuple[int, ...]) -> pd.DataFrame | None:
    """Returns None on a database error (e.g. statement timeout)."""
    if not os.path.exists(DB_PATH):
        st.error("Local database `historical.db` not found. Please run `python create_db.py` first.")
        return None
    try:
        with sqlite3.connect(DB_PATH) as conn:
            query = f'SELECT * FROM {table} WHERE CAST("Year" AS INTEGER) IN ({",".join("?" for _ in years)})'
            df = pd.read_sql(query, conn, params=years)
            return df
    except Exception as e:
        st.error(f"Couldn't load data right now. Please try again in a moment. ({e})")
        return None


def _exam_type(inst: str) -> str:
    inst_lower = inst.lower()
    for kw in IIT_KEYWORDS:
        if kw in inst_lower:
            return "advanced"
    return "mains"


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


@st.cache_resource(show_spinner="Loading model…")
def _load_model_cached(source: str):
    from pipeline.config import SOURCES, MODEL_DIR
    import pickle
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


def _build_historical_trend_fig(df: pd.DataFrame, metric: str = "Average") -> go.Figure | None:
    if df.empty:
        return None
    chart_df = df.copy()
    chart_df["Short Program"] = chart_df["Academic Program Name"].apply(_short_program_name)
    
    if metric == "Peak":
        grouped = chart_df.groupby(["Short Program", "Year"])["Closing Rank"].min().reset_index()
    elif metric == "Floor":
        grouped = chart_df.groupby(["Short Program", "Year"])["Closing Rank"].max().reset_index()
    else:
        grouped = chart_df.groupby(["Short Program", "Year"])["Closing Rank"].mean().reset_index()
    
    top_programs = chart_df["Short Program"].value_counts().head(8).index.tolist()
    grouped = grouped[grouped["Short Program"].isin(top_programs)]
    if grouped.empty:
        return None
        
    fig = go.Figure()
    years_order = sorted(grouped["Year"].unique(), key=lambda y: int(y))
    
    for prog in grouped["Short Program"].unique():
        prog_data = grouped[grouped["Short Program"] == prog].sort_values("Year")
        fig.add_trace(go.Scatter(
            x=prog_data["Year"],
            y=prog_data["Closing Rank"],
            mode="lines+markers",
            name=prog,
            line=dict(width=3, shape="spline", smoothing=0.8),
            marker=dict(size=8, symbol="circle", line=dict(width=2, color="white")),
            hovertemplate="<b>" + prog + "</b><br>Year: %{x}<br>" + metric + " Closing: %{y:,.0f}<extra></extra>"
        ))
        
    # Calculate Y-axis range to keep bottom at exactly 500k by default
    min_rank = grouped["Closing Rank"].min()
    if pd.isna(min_rank):
        y_range = [500000, 0]
    else:
        padding = 1000
        y_range = [500000, max(0, int(min_rank) - padding)]
    
    user_rank = st.session_state.get("last_rank", 400000)
    if user_rank:
        fig.add_hline(
            y=user_rank,
            line_dash="dash",
            line_color="black",
            line_width=2,
            annotation_text=f"  Your Rank: {user_rank:,}",
            annotation_position="top right",
        )

    fig.update_layout(
        xaxis=dict(
            title="",
            type="category",
            categoryorder="array",
            categoryarray=years_order,
            gridcolor="#f1f5f9",
            tickfont=dict(color="#64748b"),
            showline=False,
            zeroline=False,
        ),
        yaxis=dict(
            title=f"{metric} Closing Rank",
            range=y_range, # Inverted tightly bounded range
            gridcolor="#f1f5f9",
            tickformat=",",
            tickfont=dict(color="#64748b"),
            showline=False,
            zeroline=False,
        ),
        hovermode="x unified",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=50, r=20, t=10, b=10),
        height=320,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.35,
            xanchor="center",
            x=0.5,
            font=dict(size=11, color="#475569")
        )
    )
    return fig


# ── Sidebar: compact modern filters ───────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <style>
        /* Sidebar compact styling */
        [data-testid="stSidebar"] {
            background-color: #fafbfc;
            border-right: 1px solid #e8ecf1;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }
        /* Compress vertical spacing on sidebar widgets */
        [data-testid="stSidebar"] .stRadio > div { gap: 0.3rem !important; }
        [data-testid="stSidebar"] .stMultiSelect { margin-bottom: -8px; }
        .sb-section {
            font-size: 0.65rem;
            font-weight: 700;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-top: 12px;
            margin-bottom: 4px;
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        }
        .sb-divider {
            border: none;
            border-top: 1px solid #e8ecf1;
            margin: 10px 0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<p style="font-size:1.1rem;font-weight:700;color:#0f172a;margin-bottom:2px;">⚙️ Filters</p>', unsafe_allow_html=True)

    with st.expander("Source & Years", expanded=True):
        default_val = st.session_state.get("hist_source", "CSAB")
        default_idx = ["JoSAA", "CSAB"].index(default_val)
        source = st.radio("Source", ["JoSAA", "CSAB"], index=default_idx, horizontal=True, label_visibility="collapsed", key="hist_source").lower()
        table = source

        available_years = _fetch_years(table)
        if not available_years:
            st.error("No data found.")
            st.stop()

        selected_years = st.multiselect(
            "Year(s)",
            options=available_years,
            default=available_years,
            label_visibility="collapsed",
        )
        if not selected_years:
            st.warning("Pick at least one year.")
            st.stop()

    with st.expander("Exam & Quota", expanded=False):
        exam_filter = st.radio(
            "Exam type",
            ["All", "Mains (NIT/IIIT/GFTI)", "Advanced (IIT)"],
            horizontal=True,
            label_visibility="collapsed",
        )
        _exam_map = {"Mains (NIT/IIIT/GFTI)": "JEE Mains (NIT / IIIT / GFTI)", "Advanced (IIT)": "JEE Advanced (IIT)"}
        exam_filter = _exam_map.get(exam_filter, exam_filter)

        quota_filter = st.pills(
            "Quota", QUOTAS[source], default=["AI"],
            selection_mode="multi",
            format_func=lambda x: x,
            label_visibility="collapsed",
        )

    with st.expander("Seat & Gender", expanded=False):
        seat_filter = st.multiselect(
            "Seat Type", SEAT_TYPES, default=["EWS"],
            label_visibility="collapsed",
        )

        gender_filter = st.pills(
            "Gender",
            ["Gender-Neutral", "Female-only (including Supernumerary)"],
            default=["Gender-Neutral"],
            selection_mode="multi",
            format_func=lambda g: "Neutral" if g == "Gender-Neutral" else "Female",
            label_visibility="collapsed",
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
    
    def on_gfti_select_hist():
        selected = st.session_state["quick_gfti_select_hist"]
        if selected != "-- Select GFTI --":
            st.session_state["hist_inst"] = selected
        else:
            st.session_state["hist_inst"] = ""
            
    st.selectbox(
        "Select GFTI to auto-filter",
        options=["-- Select GFTI --"] + GFTI_OPTIONS,
        label_visibility="collapsed",
        key="quick_gfti_select_hist",
        on_change=on_gfti_select_hist
    )

rounds_available = JOSAA_ROUNDS if source == "josaa" else CSAB_ROUNDS
round_col = "Round" if source == "josaa" else "Special Round"

# Retrieve search keywords from session state if they exist
prog_kw = st.session_state.get("hist_prog", "")
inst_kw = st.session_state.get("hist_inst", "")

# ── Load + filter ─────────────────────────────────────────────────────────
df = fetch_data(table, tuple(sorted(selected_years)))

if df is None:
    st.stop()

if df.empty:
    st.warning("No data returned. Please try again later.")
    st.stop()

# Convert ranks to numeric for proper mathematical sorting and formatting
df["Opening Rank"] = pd.to_numeric(df["Opening Rank"], errors="coerce")
df["Closing Rank"] = pd.to_numeric(df["Closing Rank"], errors="coerce")

# Derive exam type for filtering
df["_et"] = df["Institute"].apply(_exam_type)
if exam_filter == "JEE Mains (NIT / IIIT / GFTI)":
    df = df[df["_et"] == "mains"]
elif exam_filter == "JEE Advanced (IIT)":
    df = df[df["_et"] == "advanced"]

# Apply filters
if quota_filter:
    df = df[df["Quota"].isin(quota_filter)]
if seat_filter:
    df = df[df["Seat Type"].isin(seat_filter)]
if gender_filter:
    df = df[df["Gender"].isin(gender_filter)]
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
        
    mask = df["Institute"].apply(_match_inst)
    df = df[mask]
if prog_kw:
    df = df[df["Academic Program Name"].str.contains(prog_kw, case=False, na=False)]

df = df.drop(columns=["_et", "id"], errors="ignore")

# ── 1. Global CSS ─────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Global font size bump + kill top margin */
    html, body, [data-testid="stAppViewContainer"] {
        font-size: 16px !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }
    .block-container { padding-top: 1rem !important; }
    header[data-testid="stHeader"] { height: 0px !important; min-height: 0px !important; }
    .dashboard-header {
        margin-bottom: 16px;
        font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .title-row {
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
    }
    .main-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0 !important;
        letter-spacing: -0.02em;
    }
    .count-badge {
        background-color: #eff6ff;
        color: #2563eb;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 9999px;
        border: 1px solid #dbeafe;
    }
    .sub-title {
        color: #64748b;
        font-size: 0.92rem;
        margin-top: 4px !important;
        margin-bottom: 0px !important;
    }
    .filter-pill-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 20px;
        align-items: center;
        font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        border-bottom: 1px solid #f1f5f9;
        padding-bottom: 16px;
    }
    .filter-label {
        font-size: 0.82rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-right: 4px;
    }
    .filter-pill {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 0.82rem;
        color: #334155;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    }
    .filter-pill strong {
        color: #0f172a;
        font-weight: 600;
    }
    
    @media print {
        @page { size: auto; margin: 12mm; }
        /* Hide Streamlit UI elements */
        header[data-testid="stHeader"], 
        [data-testid="stSidebar"], 
        .stButton,
        [data-testid="stToolbar"] { 
            display: none !important; 
        }
        
        /* Ensure main content handles page breaks and is visible */
        .stApp, .stMain, [data-testid="stAppViewContainer"], [data-testid="stAppViewBlockContainer"] {
            overflow: visible !important;
            height: auto !important;
            max-height: none !important;
            background: white !important;
        }

        /* Show the HTML table, hide the interactive dataframe */
        .print-table-wrapper { display: block !important; width: 100%; margin-bottom: 20px; }
        .print-table { width: 100%; border-collapse: collapse; font-size: 10pt; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        .print-table th, .print-table td { border: 1px solid #cbd5e1; padding: 6px 8px; text-align: left; }
        .print-table th { background-color: #f1f5f9; font-weight: 600; color: #0f172a; }
        .print-table tr:nth-child(even) { background-color: rgba(128,128,128,0.02); }
        
        div[data-testid="stDataFrame"], div[data-testid="stDataEditor"], .stDataFrame, .stDataEditor { display: none !important; }
        
        /* Ensure expanders are visible */
        details { display: block !important; }
    }
    @media screen {
        .print-table-wrapper { display: none !important; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ── Official Banner ───────────────────────────────────────────────────────
if source == "csab":
    banner_title = "CENTRAL SEAT ALLOCATION BOARD — CSAB"
    banner_hindi = "केन्द्रीय सीट आवंटन बोर्ड — सीएसएबी"
    banner_sub = f"Admission and eCounselling Services for Session {PREDICT_YEAR}"
    banner_color = "#00467F"
    logo_html = '<img src="https://cdnbbsr.s3waas.gov.in/s305a70454516ecd9194c293b0e415777f/uploads/2022/08/2022081238.png" style="width: 28px; height: 28px; object-fit: contain;">'
else:
    banner_title = "JOINT SEAT ALLOCATION AUTHORITY — JoSAA"
    banner_hindi = "संयुक्त सीट आवंटन प्राधिकरण — जोसा"
    banner_sub = f"Admission to IITs, NITs, IIITs & GFTIs — Session {PREDICT_YEAR}"
    banner_color = "#1a3c6e"
    logo_html = f'<span style="font-size: 0.9rem; font-weight: 800; color: {banner_color}; letter-spacing: -0.02em;">J</span>'

st.markdown(
    f"""
    <div style="
        background: linear-gradient(135deg, {banner_color} 0%, {banner_color}dd 60%, {banner_color}bb 100%);
        border-radius: 10px;
        padding: 14px 24px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.12);
    ">
        <div style="display: flex; align-items: center; gap: 14px;">
            <div style="
                width: 42px; height: 42px;
                background: white;
                border-radius: 50%;
                display: flex; align-items: center; justify-content: center;
                flex-shrink: 0;
                box-shadow: 0 1px 4px rgba(0,0,0,0.15);
            ">{logo_html}</div>
            <div>
                <p style="color: rgba(255,255,255,0.65); font-size: 0.68rem; margin: 0; font-family: sans-serif;">{banner_hindi}</p>
                <p style="color: #ffffff; font-size: 0.95rem; font-weight: 700; margin: 0; letter-spacing: 0.02em; font-family: -apple-system, BlinkMacSystemFont, sans-serif;">{banner_title}</p>
                <p style="color: rgba(255,255,255,0.75); font-size: 0.72rem; margin: 2px 0 0 0; font-family: sans-serif;">{banner_sub}</p>
            </div>
        </div>
        <div style="
            display: flex; align-items: center; gap: 8px;
            background: rgba(255,255,255,0.12);
            border-radius: 6px;
            padding: 6px 12px;
        ">
            <img src="https://cdnbbsr.s3waas.gov.in/s305a70454516ecd9194c293b0e415777f/uploads/2022/02/2022022452.png" style="width: 32px; height: 32px; object-fit: contain;">
            <div>
                <p style="color: #ffffff; font-size: 0.7rem; font-weight: 600; margin: 0;">Digital India</p>
                <p style="color: rgba(255,255,255,0.55); font-size: 0.55rem; margin: 0;">Power To Empower</p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── 4. Main Header Title & Count Badge ────────────────────────────────────
unique_institutes = df["Institute"].unique() if not df.empty else []
single_institute = len(unique_institutes) == 1

if not df.empty and single_institute:
    from pipeline.institute_data import append_nirf_rank
    title_text = append_nirf_rank(unique_institutes[0])
    sub_text = f"Manage and inspect actual multi-year cutoffs for {unique_institutes[0]}"
else:
    title_text = "Historical Closing Ranks"
    sub_text = f"Inspect multi-year college-wise and branch-wise cutoff trajectory ranks for {source.upper()}."

badge_html = f'<span class="count-badge">{len(df):,} items</span>' if not df.empty else '<span class="count-badge">0 items</span>'

st.markdown(
    f"""
    <div class="dashboard-header">
        <div class="title-row">
            <h1 class="main-title">{title_text}</h1>
            {badge_html}
        </div>
        <p class="sub-title">{sub_text}</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ── 5. Active Filters + Inline Search (single row) ────────────────────────
pill_items = []
if quota_filter:
    pill_items.append(f'<div class="filter-pill">Quota: <strong>{", ".join(quota_filter)}</strong></div>')
if seat_filter:
    pill_items.append(f'<div class="filter-pill">Seat: <strong>{", ".join(seat_filter)}</strong></div>')
if gender_filter:
    pill_items.append(f'<div class="filter-pill">Gender: <strong>{", ".join(gender_filter)}</strong></div>')

pills_html = "\n".join(pill_items) if pill_items else ""

fc1, fc2, fc3 = st.columns([3, 2, 2])
with fc1:
    if pills_html:
        st.markdown(
            f"""
            <div style="display:flex;flex-wrap:wrap;gap:6px;align-items:center;padding-top:6px;">
                <span class="filter-label">Filters:</span>
                {pills_html}
            </div>
            """,
            unsafe_allow_html=True
        )
with fc2:
    st.text_input(
        "Branch",
        value=prog_kw,
        key="hist_prog",
        placeholder="Search branch...",
        label_visibility="collapsed",
    )
with fc3:
    st.text_input(
        "Institute",
        value=inst_kw,
        key="hist_inst",
        placeholder="Search college...",
        label_visibility="collapsed",
    )

st.markdown('<div style="border-bottom: 1px solid #e2e8f0; margin-bottom: 16px;"></div>', unsafe_allow_html=True)

# Trend Chart
if len(selected_years) > 1 and not df.empty:
    th1, th2 = st.columns([3, 2])
    with th1:
        st.markdown('<h3 style="font-size: 1.05rem; font-weight: 600; color: #334155; margin-top: 4px; margin-bottom: 8px;">📈 Closing Rank Trends</h3>', unsafe_allow_html=True)
    with th2:
        trend_metric = st.pills(
            "Metric",
            options=["Peak", "Average", "Floor"],
            default="Floor",
            selection_mode="single",
            label_visibility="collapsed"
        )
        if not trend_metric:
            trend_metric = "Floor"
    
    fig = _build_historical_trend_fig(df, metric=trend_metric)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Not enough data to plot trend line chart.")
        
    valid_closing = df["Closing Rank"].dropna()
    avg_closing = int(valid_closing.mean()) if not valid_closing.empty else 0
    min_closing = int(valid_closing.min()) if not valid_closing.empty else 0
    max_closing = int(valid_closing.max()) if not valid_closing.empty else 0

    st.markdown(
        f"""
        <div style="display: flex; gap: 12px; margin-top: 8px; margin-bottom: 20px; font-family: -apple-system, BlinkMacSystemFont, sans-serif;">
            <div style="flex: 1; background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); border: 1px solid #bfdbfe; border-radius: 8px; padding: 10px 14px; box-shadow: 0 1px 3px rgba(37,99,235,0.05);">
                <p style="font-size: 0.68rem; color: #1e40af; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin: 0;">Peak Rank</p>
                <p style="font-size: 1.25rem; color: #1e3a8a; font-weight: 800; margin: 2px 0 0 0;">{min_closing:,}</p>
            </div>
            <div style="flex: 1; background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); border: 1px solid #bbf7d0; border-radius: 8px; padding: 10px 14px; box-shadow: 0 1px 3px rgba(22,163,74,0.05);">
                <p style="font-size: 0.68rem; color: #166534; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin: 0;">Average Rank</p>
                <p style="font-size: 1.25rem; color: #14532d; font-weight: 800; margin: 2px 0 0 0;">{avg_closing:,}</p>
            </div>
            <div style="flex: 1; background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%); border: 1px solid #fecaca; border-radius: 8px; padding: 10px 14px; box-shadow: 0 1px 3px rgba(220,38,38,0.05);">
                <p style="font-size: 0.68rem; color: #991b1b; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin: 0;">Floor Rank</p>
                <p style="font-size: 1.25rem; color: #7f1d1d; font-weight: 800; margin: 2px 0 0 0;">{max_closing:,}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Show fee data if exactly 1 institute is filtered
    if df["Institute"].nunique() == 1:
        single_inst = df["Institute"].iloc[0]
        fee_data = _find_fee_data(single_inst)
        if fee_data:
            with st.expander(f"💰 Fee Structure — {fee_data['short_name']}", expanded=False):
                cur = fee_data["currency"]
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
                    st.markdown("**📝 Important Notes**")
                    for note in fee_data["additional_notes"]:
                        st.markdown(f"- {note}")

                src_cols = st.columns(2)
                with src_cols[0]:
                    if fee_data.get("source_url"):
                        st.link_button("🌐 Official Fee Page", fee_data["source_url"], use_container_width=True)
                with src_cols[1]:
                    if fee_data.get("source_pdf"):
                        st.link_button("📄 Download PDF", fee_data["source_pdf"], use_container_width=True)
        else:
            fee_search_url = f"https://www.google.com/search?q={urllib.parse.quote_plus(single_inst + ' fee structure 2024-25')}"
            st.link_button(f"💰 Search Fee Structure for {single_inst.split('(')[0].strip()}", fee_search_url)

        # Show 2026 Predictions
        st.markdown('<div style="margin-top: 10px;"></div>', unsafe_allow_html=True)
        with st.expander("🔮 2026 Predicted Closing Ranks", expanded=False):
            # Let the user input or verify their rank
            default_rank = st.session_state.get("last_rank", 400000 if source == "csab" else 50000)
            user_rank = st.number_input(
                "Enter your JEE rank to see safety categories:",
                min_value=1, max_value=1_000_000,
                value=int(default_rank),
                key="hist_pred_rank"
            )
            st.session_state["last_rank"] = user_rank
            
            # Load the model using dynamic cache to avoid circular/illegal digit-prefix imports
            model, model_path = _load_model_cached(source)
            if model:
                slots = model["slots"]
                et = _exam_type(single_inst)
                
                # Active filters from the sidebar
                quota = quota_filter[0] if quota_filter else "AI"
                seat = seat_filter[0] if seat_filter else "EWS"
                gnd = gender_filter[0] if gender_filter else "Gender-Neutral"
                
                pred_rows = []
                for key, slot_model in slots.items():
                    inst, prog, q, st_val, g, exam = key
                    if inst == single_inst and q == quota and st_val == seat and g == gnd and exam == et:
                        w = model.get("ensemble_weight")
                        round_preds = slot_model.predict_all_rounds(2026, [1,2,3,4,5,6], w=w)
                        
                        final_r = slot_model.max_round
                        if round_preds.get(final_r):
                            interval_r = final_r
                            pred_final = round_preds[final_r]
                        else:
                            interval_r = max(round_preds) if round_preds else 1
                            pred_final = round_preds.get(interval_r, 0)
                            
                        has_intervals = hasattr(slot_model, "round_abs_deviations")
                        if has_intervals:
                            lower, upper = slot_model.predict_interval(interval_r, 2026, 0.90)
                        else:
                            lower = 0.80 * pred_final
                            upper = 1.20 * pred_final
                            
                        if user_rank <= lower:
                            cat = "Safe 🟢"
                        elif user_rank <= pred_final:
                            cat = "Match 🟡"
                        elif user_rank <= upper:
                            cat = "Reach 🔴"
                        else:
                            cat = "Out of Reach ❌"
                            
                        pred_rows.append({
                            "Branch": _short_program_name(prog),
                            "2026 Predicted Closing Rank": int(round(pred_final)) if pred_final else "-",
                            "Safety Category": cat
                        })
                        
                if pred_rows:
                    pred_df = pd.DataFrame(pred_rows).sort_values("2026 Predicted Closing Rank", na_position="last")
                    st.dataframe(
                        pred_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "2026 Predicted Closing Rank": st.column_config.NumberColumn("Predicted Closing Rank", format="%,d"),
                        }
                    )
                else:
                    st.info(f"No prediction model data found for {single_inst} under the selected filters (Quota: {quota}, Seat Type: {seat}, Gender: {gnd}).")
            else:
                st.warning("Prediction model not found. Please train the model first.")

# Stop if results are empty
if df.empty:
    st.info("No rows match the current filters. Try relaxing some selections.")
    st.stop()

# ── 7. Table Columns Setup ────────────────────────────────────────────────
# Detect which of Quota/Seat Type/Gender vary across the filtered data
# (constant values are already shown as filter pills above, no need in table)
FILTER_COLS = ["Quota", "Seat Type", "Gender"]
varying_cols = []
for col in FILTER_COLS:
    if col in df.columns and df[col].dropna().nunique() > 1:
        varying_cols.append(col)

BASE_TABLE_COLS = ["Academic Program Name", "Opening Rank", "Closing Rank"]

if not single_institute:
    BASE_TABLE_COLS.insert(BASE_TABLE_COLS.index("Academic Program Name"), "Institute")

for vc in varying_cols:
    idx = BASE_TABLE_COLS.index("Opening Rank")
    BASE_TABLE_COLS.insert(idx, vc)

display_cols_table = [c for c in BASE_TABLE_COLS if c in df.columns]

DISPLAY_COLS_CSV = [
    "Year", "Round", "Special Round", "Institute", "Academic Program Name",
    "Quota", "Seat Type", "Gender", "Opening Rank", "Closing Rank",
]
display_cols_csv = [c for c in DISPLAY_COLS_CSV if c in df.columns]

sort_round_col = round_col if round_col in df.columns else "Year"

# ── 8. Year Sections ──────────────────────────────────────────────────────
sorted_years = sorted(df["Year"].unique(), key=lambda y: int(y), reverse=True)
starred_set = get_choices()

for year_val in sorted_years:
    year_df = df[df["Year"] == year_val]
    if year_df.empty:
        continue

    # Detect active rounds
    actual_rounds = sorted(year_df[round_col].dropna().unique(), key=lambda x: int(x))

    # Grid columns to align Header on Left and Pills on Right horizontally
    hdr_col, rnd_col_ui = st.columns([2, 3])
    with hdr_col:
        st.markdown(f'<h3 style="font-size: 1.15rem; font-weight: 600; color: #0f172a; margin-top: 6px; margin-bottom: 0px; padding: 0px; font-family: -apple-system, BlinkMacSystemFont, sans-serif;">{year_val} Cutoffs</h3>', unsafe_allow_html=True)
        st.markdown(f'<p style="color: #94a3b8; font-size: 0.76rem; margin: 0px; padding: 0px;">{len(year_df):,} entries matched</p>', unsafe_allow_html=True)
    with rnd_col_ui:
        default_val = actual_rounds[-1] if actual_rounds else None
        year_round_filter = st.pills(
            "Round Selector",
            actual_rounds,
            default=default_val,
            selection_mode="single",
            format_func=lambda r: f"R{r}",
            key=f"rnd_toggle_{year_val}_{source}",
            label_visibility="collapsed"
        )

    # Apply inline round filter
    if year_round_filter is not None:
        year_df = year_df[year_df[round_col].astype(str) == str(year_round_filter)]
        if year_df.empty:
            st.caption("No rows matching the selected round.")
            st.markdown("---")
            continue

    # Sort descending (highest rank at the top)
    sorted_full_year_df = year_df.sort_values(
        ["Closing Rank", sort_round_col], ascending=[False, True]
    ).reset_index(drop=True)
    
    sorted_full_year_df["⭐"] = sorted_full_year_df.apply(
        lambda r: (r["Institute"], r["Academic Program Name"]) in starred_set, axis=1
    )
    
    display_cols = ["⭐"] + [c for c in display_cols_table if c != "⭐"]
    sorted_year_df = sorted_full_year_df[display_cols].copy()
    if "Institute" in sorted_year_df.columns:
        from pipeline.institute_data import append_nirf_rank
        sorted_year_df["Institute"] = sorted_year_df["Institute"].apply(append_nirf_rank)

    # Alternating row colors + bold rank fonts
    def _style_row(row):
        styles = []
        bg = 'background-color: rgba(128, 128, 128, 0.04)' if row.name % 2 != 0 else ''
        for col in row.index:
            parts = []
            if bg:
                parts.append(bg)
            if col in ("Opening Rank", "Closing Rank"):
                parts.append('font-weight: 700; color: #0f172a')
            styles.append('; '.join(parts))
        return styles

    styled_year_df = sorted_year_df.style.apply(_style_row, axis=1)

    editor_key = f"hist_editor_{year_val}_{source}"
    
    st.data_editor(
        styled_year_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "⭐": st.column_config.CheckboxColumn("⭐", width="small", help="Star this branch"),
            "Round":         st.column_config.NumberColumn("Round",         format="%d"),
            "Special Round": st.column_config.NumberColumn("Sp. Round",     format="%d"),
            "Opening Rank":  st.column_config.NumberColumn("Opening Rank",  format="%,d"),
            "Closing Rank":  st.column_config.NumberColumn("Closing Rank",  format="%,d"),
        },
        disabled=[c for c in display_cols if c != "⭐"],
        key=editor_key
    )

    state = st.session_state.get(editor_key, {})
    if "edited_rows" in state and state["edited_rows"]:
        changed = False
        for row_idx, edits in state["edited_rows"].items():
            if "⭐" in edits:
                row = sorted_full_year_df.iloc[int(row_idx)]
                toggle_choice(row["Institute"], row["Academic Program Name"], edits["⭐"])
                changed = True
        if changed:
            st.rerun()

    if single_institute:
        print_df = sorted_year_df.drop(columns=["⭐"], errors="ignore")
        if "Opening Rank" in print_df.columns:
            print_df["Opening Rank"] = print_df["Opening Rank"].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "")
        if "Closing Rank" in print_df.columns:
            print_df["Closing Rank"] = print_df["Closing Rank"].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "")
        html_table = print_df.to_html(index=False, classes="print-table")
        st.markdown(f'<div class="print-table-wrapper">{html_table}</div>', unsafe_allow_html=True)

    st.markdown('<div style="margin-top: 14px; margin-bottom: 14px; border-top: 1px solid #f1f5f9;"></div>', unsafe_allow_html=True)

# ── 9. Combined Bottom Summary Status Bar ─────────────────────────────────
csv_bytes = df[display_cols_csv].to_csv(index=False).encode("utf-8")
year_str = "-".join(str(y) for y in sorted(selected_years))

_, d2, _ = st.columns([1, 2, 1])
with d2:
    if single_institute:
        if st.button("🖨️ Download Page as PDF", use_container_width=True):
            import streamlit.components.v1 as components
            components.html(
                """
                <script>
                // Trigger print dialog in the parent window
                window.parent.print();
                </script>
                """,
                height=0,
            )
    else:
        st.download_button(
            "📥 Download Full CSV",
            data=csv_bytes,
            file_name=f"{source}_historical_{year_str}.csv",
            mime="text/csv",
            key="download_all_combined",
            use_container_width=True
        )

