# frontend/app.py — Streamlit multipage router (API-first)
import streamlit as st

st.set_page_config(page_title="Smart Meter Suite", layout="wide")

single = st.Page(
    "pages/single_api.py",
    title="Single Meter",
    icon=":material/bolt:",
    url_path="single",
    default=True,
)
compare = st.Page(
    "pages/compare_api.py",
    title="Compare Meters",
    icon=":material/compare_arrows:",
    url_path="compare",
)
bess = st.Page(
    "pages/bess_api.py",
    title="BESS Overview",
    icon=":material/battery_charging_full:",
    url_path="bess",
)

nav = st.navigation({"Dashboards": [single, compare, bess]})

with st.sidebar:
    st.markdown("### ⚡ Smart Meter Suite")
    st.caption("FastAPI + Parquet pyramids + LTTB")

nav.run()
