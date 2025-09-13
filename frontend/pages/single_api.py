from __future__ import annotations
import datetime as _dt
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

API = st.secrets.get("API_URL", "http://localhost:8000")
st.title("⚡ Single Meter Analysis")

# Fetch classified meters (cached within the session)
@st.cache_data(ttl=60)
def _get_meters_classified():
    try:
        return requests.get(f"{API}/meters/classified", timeout=15).json()
    except:
        # Fallback to simple list
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
    st.error(f"Cannot reach API at {API}. Set API_URL in secrets.toml.\n\n{e}")
    st.stop()

all_meters = classified.get("meters", [])
all_bess = classified.get("bess", [])

if not (all_meters or all_bess):
    st.warning("No meters/BESS folders found by backend. Check METER_ROOTS or POST /reload.")
    st.stop()

# Meter type selection
meter_type = st.radio("Select type:", ["Smart Meters", "BESS Systems"], horizontal=True)

if meter_type == "Smart Meters":
    available = all_meters
    default_signals = ["com_ap", "pf", "pos_ae", "neg_ae"]
else:
    available = all_bess
    default_signals = ["bms1_soc", "pcs1_ap", "aux_m_pos_ae", "aux_m_neg_ae"]

if not available:
    st.info(f"No {meter_type.lower()} found in the system.")
    st.stop()

meter = st.selectbox(f"Choose {meter_type[:-1].lower()}", available)

# Get meter info for date range
meter_info = _get_meter_info(meter)

# Sidebar controls with dynamic date range
with st.sidebar:
    st.markdown(f"### ⚙️ Settings")
    st.markdown(f"**Selected:** {meter.split('/')[-1]}")
    
    base_rule = st.selectbox("Base resample", ["5min","15min","30min","1h"], index=1)
    cumulative = st.checkbox("Energy files are cumulative", value=True)
    
    # Dynamic date range based on actual data
    if meter_info and meter_info.get("date_range", {}).get("start"):
        data_start = pd.Timestamp(meter_info["date_range"]["start"]).date()
        data_end = pd.Timestamp(meter_info["date_range"]["end"]).date()
        st.caption(f"Data available: {data_start} to {data_end}")
        
        # Default to last 7 days of available data
        default_start = max(data_start, data_end - _dt.timedelta(days=7))
        dr = st.date_input(
            "Date range", 
            (default_start, data_end),
            min_value=data_start,
            max_value=data_end
        )
    else:
        today = _dt.date.today()
        dr = st.date_input("Date range", (today - _dt.timedelta(days=7), today))
    
    max_points = st.slider("Max points/trace", 2000, 20000, 6000, 1000)
    
    if meter_info:
        st.caption(f"📊 {meter_info.get('signal_count', 0)} signals available")

# Signal selection based on meter type
if meter_type == "Smart Meters":
    AVAILABLE_SIGNALS = ["com_ap", "pf", "pos_ae", "neg_ae", "com_ae"]
else:
    AVAILABLE_SIGNALS = [
        "bms1_soc", "bms1_soh", "bms1_v", "bms1_c",
        "pcs1_ap", "pcs1_dcc", "pcs1_dcv",
        "aux_m_ap", "aux_m_pos_ae", "aux_m_neg_ae",
        "ac1_outside_t", "dh1_temp"
    ]

signals = st.multiselect(
    "Signals to display",
    AVAILABLE_SIGNALS,
    default=default_signals,
    help=f"Available signals for {meter_type.lower()}"
)

def _date_params():
    p = {}
    if len(dr) >= 1:
        p["start"] = pd.Timestamp(dr[0]).date().isoformat()
    if len(dr) == 2:
        p["end"] = pd.Timestamp(dr[1]).date().isoformat()
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

dates = (_date_params().get("start"), _date_params().get("end"))

if st.button("🔄 Load Data", type="primary"):
    st.session_state.load_data = True

if not st.session_state.get("load_data"):
    st.info("Click 'Load Data' to fetch and display the selected signals.")
    st.stop()

try:
    with st.spinner("Fetching data..."):
        bundle = _bundle(meter, tuple(signals), base_rule, cumulative, dates, max_points)
except Exception as e:
    st.error(f"Data request failed. Try shorter date range or fewer signals.\n\n{e}")
    st.stop()

# Display KPIs
kpis = bundle["kpis"]
fmt = lambda v, d="–": (f"{v:,.2f}" if v is not None else d)

st.markdown("### 📈 Key Performance Indicators")
c1,c2,c3,c4,c5 = st.columns(5)

if meter_type == "Smart Meters":
    c1.metric("Import kWh", fmt(kpis.get("total_import_kWh")))
    c2.metric("Export kWh", fmt(kpis.get("total_export_kWh")))
    c3.metric("Net kWh", fmt(kpis.get("net_kWh")))
    c4.metric("Peak kW", fmt(kpis.get("peak_kW")))
    c5.metric("Avg PF", f"{kpis.get('avg_pf'):.3f}" if kpis.get("avg_pf") is not None else "–")
else:
    # For BESS, show different KPIs if available
    c1.metric("Import kWh", fmt(kpis.get("total_import_kWh")))
    c2.metric("Export kWh", fmt(kpis.get("total_export_kWh")))
    c3.metric("Net kWh", fmt(kpis.get("net_kWh")))
    c4.metric("Peak kVA", fmt(kpis.get("peak_kW")))  # Actually kVA for BESS
    c5.metric("Efficiency", "–")  # Placeholder

def df_from_series(js: dict|None) -> pd.DataFrame:
    if not js or not js.get("timestamps"):
        return pd.DataFrame(columns=["t","v"])
    ts = pd.to_datetime(js["timestamps"])
    vals = np.array(js["values"], dtype="float32")
    return pd.DataFrame({"t": ts, "v": vals})

# Create tabs for different signal categories
if meter_type == "Smart Meters":
    tabs = st.tabs(["⚡ Power", "📊 Power Factor", "📈 Energy"])
    
    with tabs[0]:
        # Power plot
        if "com_ap" in signals:
            ap = df_from_series(bundle["series"].get("com_ap"))
            if not ap.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=ap["t"], y=ap["v"],
                    mode='lines',
                    name='Power (kW)',
                    line=dict(color='#1f77b4', width=1.5)
                ))
                fig.update_layout(
                    title="Active Power",
                    xaxis_title="Time",
                    yaxis_title="Power (kW)",
                    hovermode='x unified',
                    margin=dict(l=50,r=20,t=40,b=40),
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No power data available for selected range.")
        else:
            st.info("Select 'com_ap' signal to view power data.")
    
    with tabs[1]:
        # Power Factor plot
        if "pf" in signals:
            pf = df_from_series(bundle["series"].get("pf"))
            if not pf.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=pf["t"], y=pf["v"],
                    mode='lines',
                    name='Power Factor',
                    line=dict(color='#ff7f0e', width=1.5)
                ))
                fig.update_layout(
                    title="Power Factor",
                    xaxis_title="Time",
                    yaxis_title="PF",
                    yaxis=dict(range=[0, 1.05]),
                    hovermode='x unified',
                    margin=dict(l=50,r=20,t=40,b=40),
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No power factor data available.")
        else:
            st.info("Select 'pf' signal to view power factor data.")
    
    with tabs[2]:
        # Energy bar chart
        st.markdown("#### Daily Energy Consumption")
        daily = pd.DataFrame(index=pd.DatetimeIndex([]))
        
        series_map = {
            "pos_ae": "Import",
            "neg_ae": "Export",
            "com_ae": "Combined"
        }
        
        for k, lbl in series_map.items():
            if k in signals:
                s = df_from_series(bundle["series"].get(k))
                if not s.empty:
                    d = s.set_index("t")["v"].resample("1d").sum(min_count=1)
                    daily = daily.join(d.rename(lbl), how="outer")
        
        if not daily.empty:
            daily_reset = daily.dropna(how="all").reset_index(names="Date")
            melted = daily_reset.melt(id_vars=["Date"], var_name="Type", value_name="kWh")
            
            fig = go.Figure()
            for energy_type in melted["Type"].unique():
                data = melted[melted["Type"] == energy_type]
                fig.add_trace(go.Bar(
                    x=data["Date"],
                    y=data["kWh"],
                    name=f"{energy_type} kWh/day",
                    text=data["kWh"].round(1),
                    textposition='auto',
                ))
            
            fig.update_layout(
                title="Daily Energy",
                xaxis_title="Date",
                yaxis_title="Energy (kWh)",
                barmode='group',
                hovermode='x unified',
                margin=dict(l=50,r=20,t=40,b=40),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No energy data available for selected signals.")

else:  # BESS Systems
    tabs = st.tabs(["🔋 Battery Status", "⚡ Power", "🌡️ Temperature", "📊 Energy"])
    
    with tabs[0]:
        # SOC/SOH plots
        col1, col2 = st.columns(2)
        
        with col1:
            if "bms1_soc" in signals:
                soc = df_from_series(bundle["series"].get("bms1_soc"))
                if not soc.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=soc["t"], y=soc["v"],
                        mode='lines',
                        name='SOC (%)',
                        line=dict(color='#2ca02c', width=1.5),
                        fill='tozeroy',
                        fillcolor='rgba(44, 160, 44, 0.2)'
                    ))
                    fig.update_layout(
                        title="State of Charge",
                        xaxis_title="Time",
                        yaxis_title="SOC (%)",
                        yaxis=dict(range=[0, 105]),
                        margin=dict(l=50,r=20,t=40,b=40),
                        height=350
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No SOC data")
        
        with col2:
            if "bms1_soh" in signals:
                soh = df_from_series(bundle["series"].get("bms1_soh"))
                if not soh.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=soh["t"], y=soh["v"],
                        mode='lines',
                        name='SOH (%)',
                        line=dict(color='#d62728', width=1.5)
                    ))
                    fig.update_layout(
                        title="State of Health",
                        xaxis_title="Time",
                        yaxis_title="SOH (%)",
                        yaxis=dict(range=[0, 105]),
                        margin=dict(l=50,r=20,t=40,b=40),
                        height=350
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No SOH data")
    
    with tabs[1]:
        # PCS Power
        if "pcs1_ap" in signals:
            pcs = df_from_series(bundle["series"].get("pcs1_ap"))
            if not pcs.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=pcs["t"], y=pcs["v"],
                    mode='lines',
                    name='PCS Power (kVA)',
                    line=dict(color='#9467bd', width=1.5)
                ))
                fig.update_layout(
                    title="PCS Apparent Power",
                    xaxis_title="Time",
                    yaxis_title="Power (kVA)",
                    hovermode='x unified',
                    margin=dict(l=50,r=20,t=40,b=40),
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No PCS power data available.")
    
    with tabs[2]:
        # Temperature
        temp_signals = ["ac1_outside_t", "dh1_temp"]
        temp_data = []
        for sig in temp_signals:
            if sig in signals:
                data = df_from_series(bundle["series"].get(sig))
                if not data.empty:
                    temp_data.append((sig, data))
        
        if temp_data:
            fig = go.Figure()
            for sig, data in temp_data:
                fig.add_trace(go.Scatter(
                    x=data["t"], y=data["v"],
                    mode='lines',
                    name=sig.replace("_", " ").title(),
                    line=dict(width=1.5)
                ))
            fig.update_layout(
                title="Temperature Monitoring",
                xaxis_title="Time",
                yaxis_title="Temperature (°C)",
                hovermode='x unified',
                margin=dict(l=50,r=20,t=40,b=40),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No temperature data available.")
    
    with tabs[3]:
        # Auxiliary Energy
        st.markdown("#### Auxiliary Energy Consumption")
        daily = pd.DataFrame(index=pd.DatetimeIndex([]))
        
        series_map = {
            "aux_m_pos_ae": "Aux Import",
            "aux_m_neg_ae": "Aux Export"
        }
        
        for k, lbl in series_map.items():
            if k in signals:
                s = df_from_series(bundle["series"].get(k))
                if not s.empty:
                    d = s.set_index("t")["v"].resample("1d").sum(min_count=1)
                    daily = daily.join(d.rename(lbl), how="outer")
        
        if not daily.empty:
            daily_reset = daily.dropna(how="all").reset_index(names="Date")
            melted = daily_reset.melt(id_vars=["Date"], var_name="Type", value_name="kWh")
            
            fig = go.Figure()
            for energy_type in melted["Type"].unique():
                data = melted[melted["Type"] == energy_type]
                fig.add_trace(go.Bar(
                    x=data["Date"],
                    y=data["kWh"],
                    name=f"{energy_type} kWh/day",
                    text=data["kWh"].round(1),
                    textposition='auto',
                ))
            
            fig.update_layout(
                title="Daily Auxiliary Energy",
                xaxis_title="Date",
                yaxis_title="Energy (kWh)",
                barmode='group',
                hovermode='x unified',
                margin=dict(l=50,r=20,t=40,b=40),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No auxiliary energy data available.")

# Footer with data info
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption(f"📍 Data source: {meter}")
with col2:
    st.caption(f"📊 Resolution: {bundle.get('series', {}).get(signals[0] if signals else '', {}).get('rule', 'N/A')}")
with col3:
    st.caption(f"🔽 Points: {max_points} max per series")