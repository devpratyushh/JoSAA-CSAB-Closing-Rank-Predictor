import streamlit as st

st.set_page_config(
    page_title="JoSAA / CSAB Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

pg = st.navigation([
    st.Page("pages/0_Predictor.py",       title="Predictor",      icon="🎓"),
    st.Page("pages/1_Historical_Data.py", title="Historical Data", icon="📊"),
])
pg.run()
