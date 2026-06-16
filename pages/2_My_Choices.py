import streamlit as st
import pandas as pd
import sqlite3
import os
import plotly.graph_objects as go
from pipeline.choices import get_choices, toggle_choice
from pipeline.config import IIT_KEYWORDS

st.set_page_config(page_title="My Choices", page_icon="⭐", layout="wide")

# Custom print & screen styling to support unconstrained multi-page PDF generation
st.markdown(
    """
    <style>
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

def _short_program_name(prog: str) -> str:
    import re
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

def _get_exam_type(inst: str) -> str:
    inst_lower = inst.lower()
    for kw in IIT_KEYWORDS:
        if kw in inst_lower:
            return "advanced"
    return "mains"

def fetch_starred_ranks(starred_set, source, quota, seat_type, gender):
    if not starred_set:
        return pd.DataFrame()
    
    if not os.path.exists("historical.db"):
        return pd.DataFrame()
        
    with sqlite3.connect("historical.db") as conn:
        params = []
        where_pairs = []
        for inst, prog in starred_set:
            where_pairs.append('("Institute" = ? AND "Academic Program Name" = ?)')
            params.extend([inst, prog])
            
        where_clause = " OR ".join(where_pairs)
        rnd_col = "Round" if source == "josaa" else "Special Round"
        
        query = f"""
        SELECT "Institute", "Academic Program Name", "Year", "{rnd_col}" as rnd, "Opening Rank", "Closing Rank"
        FROM {source}
        WHERE ({where_clause})
          AND "Quota" = ?
          AND "Seat Type" = ?
          AND "Gender" = ?
        """
        params.extend([quota, seat_type, gender])
        
        df = pd.read_sql(query, conn, params=params)
        return df

def get_starred_predictions(starred_set, quota, seat_type, gender, model, rank, coverage=0.90):
    if not starred_set or model is None:
        return pd.DataFrame()
        
    slots = model["slots"]
    results = []
    
    for inst, prog in starred_set:
        exam_type = _get_exam_type(inst)
        key = (inst, prog, quota, seat_type, gender, exam_type)
        
        if key in slots:
            slot_model = slots[key]
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
                lower, upper = slot_model.predict_interval(interval_r, 2026, coverage)
            else:
                lower = 0.80 * pred_final
                upper = 1.20 * pred_final
                
            if rank <= lower:
                category = "Safe 🟢"
            elif rank <= pred_final:
                category = "Match 🟡"
            elif rank <= upper:
                category = "Reach 🔴"
            else:
                category = "Out of Reach ❌"
                
            results.append({
                "Institute": inst,
                "Academic Program Name": prog,
                "2026 Predicted Closing Rank": int(round(pred_final)) if pred_final else "-",
                "Category": category
            })
        else:
            results.append({
                "Institute": inst,
                "Academic Program Name": prog,
                "2026 Predicted Closing Rank": "-",
                "Category": "No Model Data"
            })
            
    return pd.DataFrame(results)

def get_historical_trends(df):
    if df.empty:
        return pd.DataFrame()
        
    df = df.copy()
    df["Closing Rank"] = pd.to_numeric(df["Closing Rank"], errors="coerce")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["rnd_num"] = pd.to_numeric(df["rnd"], errors="coerce")
    
    idx = df.groupby(["Institute", "Academic Program Name", "Year"])["rnd_num"].idxmax()
    trends = df.loc[idx]
    
    return trends[["Institute", "Academic Program Name", "Year", "Closing Rank"]]

st.title("⭐ My Preferred Choices")
st.markdown("Manage, track, and visualize 2026 closing-rank predictions for your saved choices.")

starred_set = get_choices()

if not starred_set:
    st.info("You haven't starred any branches yet. Go to the **Predictor** page to discover and star your preferred options!")
else:
    # ── Sidebar preferences ──────────────────────────────────────────────────
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

    with st.sidebar:
        st.markdown('<p style="font-size:1.1rem;font-weight:700;color:#0f172a;margin-bottom:2px;">⚙️ Preferences</p>', unsafe_allow_html=True)
        # Load default inputs from session state
        last_inputs = st.session_state.get("last_inputs", None)
        default_source = last_inputs[0].upper() if last_inputs else "CSAB"
        src_index = ["CSAB", "JoSAA"].index("CSAB" if default_source == "CSAB" else "JoSAA")
        
        source = st.radio("Source", ["CSAB", "JoSAA"], index=src_index).lower()
        
        default_rank = st.session_state.get("last_rank", 400000 if source == "csab" else 50000)
        default_quota = "AI"
        default_seat = "EWS"
        default_gender = "Gender-Neutral"
        
        if last_inputs:
            default_quota = last_inputs[3]
            default_seat = last_inputs[4]
            default_gender = last_inputs[5]
                
        rank = st.number_input("Your Rank", min_value=1, max_value=1_000_000, value=int(default_rank))
        st.session_state["last_rank"] = rank
        
        quota_options = QUOTAS[source]
        quota = st.selectbox("Quota", quota_options, index=quota_options.index(default_quota) if default_quota in quota_options else 0)
        
        seat_options = SEAT_TYPES
        seat_type = st.selectbox("Seat Type", seat_options, index=seat_options.index(default_seat) if default_seat in seat_options else 0)
        
        gender_options = ["Gender-Neutral", "Female-only (including Supernumerary)"]
        gender_format = lambda g: "Neutral" if g == "Gender-Neutral" else "Female"
        gender = st.selectbox("Gender", gender_options, index=gender_options.index(default_gender) if default_gender in gender_options else 0, format_func=gender_format)

    # ── Fetch Data & Predict ────────────────────────────────────────────────
    model, model_path = _load_model_cached(source)
    
    if model:
        df_preds = get_starred_predictions(starred_set, quota, seat_type, gender, model, rank)
    else:
        df_preds = pd.DataFrame()
        st.warning("Prediction model not found. Please train the model from the Predictor tab first.")

    df_raw = fetch_starred_ranks(starred_set, source, quota, seat_type, gender)
    
    base_data = [{"⭐": True, "Institute": inst, "Academic Program Name": prog} for inst, prog in starred_set]
    base_df = pd.DataFrame(base_data)
    
    if not df_preds.empty:
        merged_df = pd.merge(base_df, df_preds, on=["Institute", "Academic Program Name"], how="left")
    else:
        merged_df = base_df.copy()
        merged_df["2026 Predicted Closing Rank"] = pd.NA
        merged_df["Category"] = "No Model"

    # Sort
    merged_df["2026 Predicted Closing Rank"] = pd.to_numeric(merged_df["2026 Predicted Closing Rank"], errors="coerce")
    merged_df = merged_df.sort_values(["Institute", "2026 Predicted Closing Rank", "Academic Program Name"], na_position="last")

    # ── 1. Plot Trend Chart ──────────────────────────────────────────────────
    trends_df = get_historical_trends(df_raw)
    if not trends_df.empty:
        st.markdown('<h3 style="font-size:1.15rem;font-weight:600;color:#1e293b;margin-bottom:8px;">📈 Historical Closing Rank Trends</h3>', unsafe_allow_html=True)
        fig = go.Figure()
        
        for (inst, prog), group in trends_df.groupby(["Institute", "Academic Program Name"]):
            group = group.sort_values("Year")
            short_prog = _short_program_name(prog)
            short_inst = inst.split("(")[0].strip()
            label = f"{short_inst} — {short_prog}"
            
            fig.add_trace(go.Scatter(
                x=group["Year"],
                y=group["Closing Rank"],
                mode="lines+markers",
                name=label,
                line=dict(width=3, shape="spline", smoothing=0.8),
                marker=dict(size=8),
                hovertemplate="<b>" + label + "</b><br>Year: %{x}<br>Closing: %{y:,.0f}<extra></extra>"
            ))
            
        years_order = sorted(trends_df["Year"].unique(), key=lambda y: int(y))
        
        min_rank = trends_df["Closing Rank"].min()
        if pd.isna(min_rank):
            y_range = [500000, 0]
        else:
            padding = 1000
            y_range = [500000, max(0, int(min_rank) - padding)]

        fig.add_hline(
            y=rank,
            line_dash="dash",
            line_color="black",
            line_width=2,
            annotation_text=f"  Your Rank: {rank:,}",
            annotation_position="top right",
        )

        fig.update_layout(
            xaxis=dict(
                type="category",
                categoryorder="array",
                categoryarray=years_order,
                gridcolor="#f1f5f9",
                tickfont=dict(color="#64748b")
            ),
            yaxis=dict(
                title="Closing Rank",
                range=y_range,
                gridcolor="#f1f5f9",
                tickformat=",",
                tickfont=dict(color="#64748b")
            ),
            hovermode="x unified",
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            margin=dict(l=50, r=20, t=10, b=10),
            height=340,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.45,
                xanchor="center",
                x=0.5,
                font=dict(size=11, color="#475569")
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div style="margin-bottom: 24px;"></div>', unsafe_allow_html=True)

    # ── 2. Grouped College Categorization ────────────────────────────────────
    st.markdown('<h3 style="font-size:1.15rem;font-weight:600;color:#1e293b;margin-bottom:12px;">🏫 My Saved Branches by College</h3>', unsafe_allow_html=True)
    
    unique_colleges = sorted(merged_df["Institute"].unique())
    
    def on_editor_change(key, inst_df):
        state = st.session_state[key]
        if "edited_rows" in state:
            for row_idx, edits in state["edited_rows"].items():
                if "⭐" in edits:
                    row = inst_df.iloc[int(row_idx)]
                    toggle_choice(row["Institute"], row["Academic Program Name"], edits["⭐"])
                    st.rerun()

    for inst in unique_colleges:
        inst_df = merged_df[merged_df["Institute"] == inst].reset_index(drop=True)
        
        cols = st.columns([5, 1.2])
        with cols[0]:
            from pipeline.institute_data import append_nirf_rank
            inst_display = append_nirf_rank(inst)
            st.markdown(
                f"""
                <div style="background-color:#f8fafc; border-left:4px solid #4f6ef5; padding:8px 16px; border-radius:4px; margin-top:16px; margin-bottom:8px;">
                    <h4 style="margin:0; color:#1e293b; font-size:1.05rem;">🏛️ {inst_display}</h4>
                </div>
                """, 
                unsafe_allow_html=True
            )
        with cols[1]:
            st.markdown('<div style="margin-top: 16px;"></div>', unsafe_allow_html=True)
            if st.button("🔍 View College", key=f"view_btn_{inst.replace(' ', '_').replace('(', '').replace(')', '')}", use_container_width=True):
                st.session_state["hist_inst"] = inst
                st.session_state["hist_prog"] = ""
                st.session_state["hist_source"] = "CSAB" if source == "csab" else "JoSAA"
                st.switch_page("pages/1_Historical_Data.py")
        
        editor_key = f"editor_{inst.replace(' ', '_').replace('(', '').replace(')', '')}"
        
        st.data_editor(
            inst_df[["⭐", "Academic Program Name", "2026 Predicted Closing Rank", "Category"]],
            key=editor_key,
            use_container_width=True,
            hide_index=True,
            column_config={
                "⭐": st.column_config.CheckboxColumn("⭐", width="small", help="Uncheck to remove from saved choices"),
                "Academic Program Name": st.column_config.TextColumn("Branch", width="large"),
                "2026 Predicted Closing Rank": st.column_config.NumberColumn("2026 Predicted Closing Rank", format="%,d"),
                "Category": st.column_config.TextColumn("Safety Category"),
            },
            disabled=["Academic Program Name", "2026 Predicted Closing Rank", "Category"],
            on_change=on_editor_change,
            args=(editor_key, inst_df)
        )

        # Render static HTML table for clean PDF prints (hiding st.data_editor virtual canvas cutoffs)
        print_df = inst_df.drop(columns=["⭐"], errors="ignore")
        if "2026 Predicted Closing Rank" in print_df.columns:
            print_df["2026 Predicted Closing Rank"] = print_df["2026 Predicted Closing Rank"].apply(
                lambda x: f"{x:,.0f}" if pd.notnull(x) and x != "-" else x
            )
        html_table = print_df.to_html(index=False, classes="print-table")
        st.markdown(f'<div class="print-table-wrapper">{html_table}</div>', unsafe_allow_html=True)

    st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)
    st.caption("Tip: Uncheck the ⭐ checkbox in any table above to immediately remove that choice from your list.")

    # ── 3. Bottom Print Button ───────────────────────────────────────────────
    st.markdown('<div class="no-print" style="margin-top: 24px;"></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    
    with c1:
        if st.button("🖨️ Download Page as PDF", use_container_width=True, key="print_choices_pdf"):
            import streamlit.components.v1 as components
            components.html(
                """
                <script>
                window.parent.print();
                </script>
                """,
                height=0,
            )
            
    with c2:
        csv_bytes = merged_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download CSV",
            data=csv_bytes,
            file_name="my_saved_choices.csv",
            mime="text/csv",
            use_container_width=True,
            key="export_csv_my_choices"
        )
        
    with c3:
        import json
        export_dict = {
            "metadata": {
                "source": source.upper(),
                "student_rank": int(rank),
                "quota": quota,
                "seat_type": seat_type,
                "gender": gender,
                "export_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "saved_choices": []
        }
        for _, row in merged_df.iterrows():
            pred_rank = row.get("2026 Predicted Closing Rank")
            export_dict["saved_choices"].append({
                "institute": row["Institute"],
                "program": row["Academic Program Name"],
                "predicted_closing_rank_2026": int(pred_rank) if pd.notna(pred_rank) and pred_rank != "-" else None,
                "safety_category": row.get("Category", "Unknown")
            })
            
        json_bytes = json.dumps(export_dict, indent=2).encode("utf-8")
        
        st.download_button(
            label="💾 Export Choices as JSON",
            data=json_bytes,
            file_name=f"josaa_csab_saved_choices_{source}_{rank}.json",
            mime="application/json",
            use_container_width=True,
            key="export_choices_json"
        )
