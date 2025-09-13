from __future__ import annotations
import datetime as _dt
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

API = st.secrets.get("API_URL", "http://localhost:8000")
st.title("🔀 Compare Two Systems")

@st.cache_data(ttl=60)
def _get_meters_classified():
    try:
        return requests.get(f"{API}/meters/classified", timeout=15).json()
    except:
        meters = requests.get(f"{API}/meters", timeout=15).json()
        return {"meters": meters, "bess": []}

@st.cache_data(ttl=60)
def _get_meter_info(meter: str):
    try:
        return requests.get(f"{API}/meters/{meter}/info", timeout=15).json()
    except:
        return None

try:
    classified = _get_meters_classified()
except Exception as e:
    st.error(f"Cannot reach API at {API}.\n\n{e}")
    st.stop()

all_meters = classified.get("meters", [])
all_bess = classified.get("bess", [])

if len(all_meters) + len(all_bess) < 2:
    st.warning("Need at least two systems to compare.")
    st.stop()

# System type selection
compare_type = st.radio("Compare:", ["Meters vs Meters", "BESS vs BESS", "Meter vs BESS"], horizontal=True)

c1, c2 = st.columns(2)

with c1:
    st.markdown("### System A")
    if compare_type == "Meters vs Meters":
        mA = st.selectbox("Select Meter A", all_meters, key="mA")
    elif compare_type == "BESS vs BESS":
        mA = st.selectbox("Select BESS A", all_bess, key="bA")
    else:
        mA = st.selectbox("Select Meter", all_meters, key="mxA")

with c2:
    st.markdown("### System B")
    if compare_type == "Meters vs Meters":
        mB = st.selectbox("Select Meter B", [m for m in all_meters if m != mA], key="mB")
    elif compare_type == "BESS vs BESS":
        mB = st.selectbox("Select BESS B", [b for b in all_bess if b != mA], key="bB")
    else:
        mB = st.selectbox("Select BESS", all_bess, key="bxB")

if mA == mB:
    st.info("Please select two different systems to compare.")
    st.stop()

# Get info for both meters
infoA = _get_meter_info(mA)
infoB = _get_meter_info(mB)

# Determine date range from both meters
with st.sidebar:
    st.markdown("### ⚙️ Comparison Settings")
    base_rule = st.selectbox("Resolution", ["5min","15min","30min","1h"], index=1)
    cumulative = st.checkbox("Energy as cumulative", value=True)
    
    # Calculate overlapping date range
    date_ranges = []
    for info in [infoA, infoB]:
        if info and info.get("date_range", {}).get("start"):
            date_ranges.append({
                "start": pd.Timestamp(info["date_range"]["start"]).date(),
                "end": pd.Timestamp(info["date_range"]["end"]).date()
            })
    
    if date_ranges:
        # Find overlapping range
        max_start = max(dr["start"] for dr in date_ranges)
        min_end = min(dr["end"] for dr in date_ranges)
        
        if max_start <= min_end:
            st.caption(f"Overlap: {max_start} to {min_end}")
            default_start = max(max_start, min_end - _dt.timedelta(days=7))
            dr = st.date_input(
                "Date range",
                (default_start, min_end),
                min_value=max_start,
                max_value=min_end
            )
        else:
            st.warning("No overlapping data range!")
            today = _dt.date.today()
            dr = st.date_input("Date range", (today - _dt.timedelta(days=7), today))
    else:
        today = _dt.date.today()
        dr = st.date_input("Date range", (today - _dt.timedelta(days=7), today))
    
    max_points = st.slider("Max points/trace", 2000, 20000, 6000, 1000)
    
    if infoA:
        st.caption(f"A: {infoA.get('signal_count', 0)} signals")
    if infoB:
        st.caption(f"B: {infoB.get('signal_count', 0)} signals")

# Signal selection based on comparison type
if compare_type == "Meters vs Meters":
    DEFAULT_SIGS = ["com_ap", "pf", "pos_ae", "neg_ae"]
elif compare_type == "BESS vs BESS":
    DEFAULT_SIGS = ["bms1_soc", "pcs1_ap", "aux_m_pos_ae", "aux_m_neg_ae"]
else:
    DEFAULT_SIGS = ["com_ap", "pf", "pcs1_ap", "bms1_soc"]

signals = st.multiselect(
    "Signals to compare",
    DEFAULT_SIGS + ["aux_m_ap", "bms1_soh", "bms1_v"],
    default=DEFAULT_SIGS[:3],
    help="Select signals present in both systems"
)

def _date_params():
    p = {}
    if len(dr) >= 1: p["start"] = pd.Timestamp(dr[0]).date().isoformat()
    if len(dr) == 2: p["end"] = pd.Timestamp(dr[1]).date().isoformat()
    return p

@st.cache_data(show_spinner=False, ttl=60)
def _bundle(meter: str, signals: tuple[str, ...], base_rule: str, cumulative: bool, dr: tuple[str|None, str|None], max_points: int):
    params = {
        "meter": meter,
        "signals": ",".join(signals),
        "rule": base_rule,
        "cumulative": str(cumulative).lower(),
        "max_points": max_points,
    }
    if dr[0]: params["start"] = dr[0]
    if dr[1]: params["end"] = dr[1]
    r = requests.get(f"{API}/bundle", params=params, timeout=180)
    r.raise_for_status()
    return r.json()

if st.button("🔄 Compare Systems", type="primary"):
    st.session_state.compare_data = True

if not st.session_state.get("compare_data"):
    st.info("Click 'Compare Systems' to load and visualize the data.")
    st.stop()

dates = (_date_params().get("start"), _date_params().get("end"))

try:
    with st.spinner("Loading comparison data..."):
        bA = _bundle(mA, tuple(signals), base_rule, cumulative, dates, max_points)
        bB = _bundle(mB, tuple(signals), base_rule, cumulative, dates, max_points)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

def df(js: dict|None):
    if not js or not js.get("timestamps"):
        return pd.DataFrame(columns=["t","v"])
    return pd.DataFrame({"t": pd.to_datetime(js["timestamps"]), "v": np.array(js["values"], dtype="float32")})

# KPI comparison
st.markdown("### 📊 KPI Comparison")
fmt = lambda v, d="–": (f"{v:,.2f}" if v is not None else d)

col1, col2 = st.columns(2)
for col, name, k in [(col1, mA.split('/')[-1], bA["kpis"]), (col2, mB.split('/')[-1], bB["kpis"])]:
    with col:
        st.markdown(f"**{name}**")
        x1,x2 = st.columns(2)
        x1.metric("Import kWh", fmt(k.get("total_import_kWh")))
        x2.metric("Export kWh", fmt(k.get("total_export_kWh")))
        y1,y2 = st.columns(2)
        y1.metric("Net kWh", fmt(k.get("net_kWh")))
        y2.metric("Peak kW", fmt(k.get("peak_kW")))
        st.caption(f"Avg PF: {k.get('avg_pf'):.3f}" if k.get("avg_pf") is not None else "Avg PF: –")

st.divider()

# Create comparison plots
st.markdown("### 📈 Signal Comparison")

# Determine which signals are available
available_plots = []
for sig in signals:
    dfA = df(bA["series"].get(sig))
    dfB = df(bB["series"].get(sig))
    if not dfA.empty or not dfB.empty:
        available_plots.append(sig)

if not available_plots:
    st.warning("No common signals with data found for the selected range.")
else:
    # Create tabs for each signal
    tabs = st.tabs([s.replace("_", " ").upper() for s in available_plots])
    
    for idx, sig in enumerate(available_plots):
        with tabs[idx]:
            dfA = df(bA["series"].get(sig))
            dfB = df(bB["series"].get(sig))
            
            fig = go.Figure()
            
            if not dfA.empty:
                fig.add_trace(go.Scatter(
                    x=dfA["t"], y=dfA["v"],
                    name=f"{mA.split('/')[-1]}",
                    mode='lines',
                    line=dict(width=2)
                ))
            
            if not dfB.empty:
                fig.add_trace(go.Scatter(
                    x=dfB["t"], y=dfB["v"],
                    name=f"{mB.split('/')[-1]}",
                    mode='lines',
                    line=dict(width=2, dash='dash')
                ))
            
            # Customize y-axis based on signal type
            y_title = sig.replace("_", " ").title()
            if "pf" in sig.lower():
                fig.update_yaxes(range=[0, 1.05])
                y_title = "Power Factor"
            elif "soc" in sig.lower() or "soh" in sig.lower():
                fig.update_yaxes(range=[0, 105])
                y_title = f"{sig.upper()} (%)"
            elif "_ap" in sig.lower():
                y_title = "Power (kW/kVA)"
            elif "_ae" in sig.lower():
                y_title = "Energy (kWh)"
            elif "_t" in sig.lower() or "temp" in sig.lower():
                y_title = "Temperature (°C)"
            elif "_v" in sig.lower():
                y_title = "Voltage (V)"
            elif "_c" in sig.lower():
                y_title = "Current (A)"
            
            fig.update_layout(
                title=f"{sig.replace('_', ' ').upper()} Comparison",
                xaxis_title="Time",
                yaxis_title=y_title,
                hovermode='x unified',
                margin=dict(l=50,r=20,t=40,b=40),
                height=400,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Daily energy comparison if energy signals selected
energy_signals = [s for s in signals if "_ae" in s.lower()]
if energy_signals:
    st.divider()
    st.markdown("### 📊 Daily Energy Comparison")
    
    def daily(df_in: pd.DataFrame, meter_label: str, signal_type: str) -> pd.DataFrame:
        if df_in.empty:
            return pd.DataFrame(columns=["Date","kWh","Signal","System"])
        d = df_in.set_index("t")["v"].resample("1d").sum(min_count=1).dropna()
        return pd.DataFrame({
            "Date": d.index,
            "kWh": d.values,
            "Signal": signal_type.replace("_", " ").title(),
            "System": meter_label.split('/')[-1]
        })
    
    rows = []
    for sig in energy_signals:
        dfA = df(bA["series"].get(sig))
        dfB = df(bB["series"].get(sig))
        if not dfA.empty:
            rows.append(daily(dfA, mA, sig))
        if not dfB.empty:
            rows.append(daily(dfB, mB, sig))
    
    if rows:
        combined = pd.concat(rows, ignore_index=True)
        
        fig = px.bar(
            combined.sort_values("Date"),
            x="Date",
            y="kWh",
            color="System",
            facet_col="Signal",
            barmode="group",
            height=400
        )
        
        fig.update_layout(
            margin=dict(l=50,r=20,t=60,b=40),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.divider()
st.caption(f"📍 Comparing: **{mA.split('/')[-1]}** vs **{mB.split('/')[-1]}** | 📊 Resolution: {base_rule} | 🔽 Max {max_points} points/series")