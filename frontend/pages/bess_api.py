from __future__ import annotations
import datetime as _dt
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

API = st.secrets.get("API_URL", "http://localhost:8000")
st.title("🔋 BESS System Overview")

@st.cache_data(ttl=60)
def _get_meters_classified():
    try:
        return requests.get(f"{API}/meters/classified", timeout=15).json()
    except:
        meters = requests.get(f"{API}/meters", timeout=15).json()
        # Heuristic classification
        bess = [m for m in meters if "ZHPESS" in m or "/BESS/" in m or "bess" in m.lower()]
        return {"meters": [m for m in meters if m not in bess], "bess": bess}

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

bess_systems = classified.get("bess", [])

if not bess_systems:
    st.info("No BESS systems detected. BESS folders should contain 'ZHPESS' or be in a 'BESS' directory.")
    st.stop()

sel = st.selectbox("Select BESS System", bess_systems)

# Get system info
system_info = _get_meter_info(sel)

with st.sidebar:
    st.markdown("### ⚙️ BESS Settings")
    st.markdown(f"**System:** {sel.split('/')[-1]}")
    
    base_rule = st.selectbox("Data Resolution", ["15min","1h"], index=1)
    
    # Dynamic date range
    if system_info and system_info.get("date_range", {}).get("start"):
        data_start = pd.Timestamp(system_info["date_range"]["start"]).date()
        data_end = pd.Timestamp(system_info["date_range"]["end"]).date()
        st.caption(f"Data: {data_start} to {data_end}")
        
        default_start = max(data_start, data_end - _dt.timedelta(days=14))
        dr = st.date_input(
            "Analysis Period",
            (default_start, data_end),
            min_value=data_start,
            max_value=data_end
        )
    else:
        today = _dt.date.today()
        dr = st.date_input("Analysis Period", (today - _dt.timedelta(days=14), today))
    
    max_points = st.slider("Max points/series", 2000, 20000, 6000, 1000)
    
    if system_info:
        st.caption(f"📊 {system_info.get('signal_count', 0)} signals")
        st.caption(f"🏷️ Type: {system_info.get('type', 'unknown').upper()}")

def _date_params():
    p = {}
    if len(dr) >= 1: p["start"] = pd.Timestamp(dr[0]).date().isoformat()
    if len(dr) == 2: p["end"] = pd.Timestamp(dr[1]).date().isoformat()
    return p

params = {"meter": sel, "rule": base_rule} | _date_params()

if st.button("🔄 Load BESS Data", type="primary"):
    st.session_state.load_bess = True

if not st.session_state.get("load_bess"):
    st.info("Click 'Load BESS Data' to fetch system metrics and visualizations.")
    st.stop()

@st.cache_data(show_spinner=False, ttl=60)
def _bess_kpis(params: dict):
    r = requests.get(f"{API}/bess_kpis", params=params, timeout=120)
    r.raise_for_status()
    return r.json()

try:
    with st.spinner("Loading BESS metrics..."):
        k = _bess_kpis(params)
except Exception as e:
    st.error(f"Failed to load BESS KPIs: {e}")
    st.stop()

fmt = lambda v, d="–": (f"{v:,.2f}" if v is not None else d)

# Display KPIs
st.markdown("### 🎯 System Health Metrics")

# Battery status row
col1, col2, col3, col4 = st.columns(4)
col1.metric("SOC Average", f"{fmt(k.get('soc_avg'))}%", 
           delta=f"Min: {fmt(k.get('soc_min'))}%" if k.get('soc_min') else None)
col2.metric("SOH Average", f"{fmt(k.get('soh_avg'))}%")
col3.metric("Pack Voltage", f"{fmt(k.get('pack_v_avg'))} V")
col4.metric("Pack Current", f"{fmt(k.get('pack_c_avg'))} A")

# Performance row
col1, col2, col3, col4 = st.columns(4)
col1.metric("PCS Peak Power", f"{fmt(k.get('pcs_ap_peak'))} kVA")
col2.metric("Aux Power Avg", f"{fmt(k.get('aux_ap_avg'))} kW")
col3.metric("Cell ΔV Max", f"{fmt(k.get('cell_v_spread_max'))} V")
col4.metric("Cell Temp Avg", f"{fmt(k.get('cell_t_avg'))}°C")

# Environment & Alarms
col1, col2 = st.columns(2)
col1.metric("Environment Temp", f"{fmt(k.get('env_temp_avg'))}°C")
alarm_status = "⚠️ ACTIVE" if k.get("alarms_any") else "✅ None"
col2.metric("Alarm Status", alarm_status)

st.divider()

# Detailed visualizations
st.markdown("### 📊 Detailed Analysis")

@st.cache_data(show_spinner=False, ttl=60)
def fetch_series(signal: str, meter: str, params: dict):
    p = {"meter": meter, "signal": signal, "max_points": max_points} | params
    try:
        r = requests.get(f"{API}/series", params=p, timeout=180)
        if r.status_code != 200:
            return None
        js = r.json()
        if not js["timestamps"]:
            return None
        return pd.DataFrame({
            "t": pd.to_datetime(js["timestamps"]),
            "v": np.array(js["values"], dtype="float32")
        })
    except:
        return None

# Create tabs for different aspects
tabs = st.tabs(["🔋 Battery State", "⚡ Power Flow", "🌡️ Thermal", "📈 Energy", "⚠️ Diagnostics"])

with tabs[0]:
    # Battery State (SOC/SOH/Voltage/Current)
    col1, col2 = st.columns(2)
    
    with col1:
        soc = fetch_series("bms1_soc", sel, params)
        if soc is not None and not soc.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=soc["t"], y=soc["v"],
                mode='lines',
                name='SOC',
                line=dict(color='#2ca02c', width=2),
                fill='tozeroy',
                fillcolor='rgba(44, 160, 44, 0.1)'
            ))
            fig.update_layout(
                title="State of Charge (%)",
                xaxis_title="Time",
                yaxis_title="SOC (%)",
                yaxis=dict(range=[0, 105]),
                margin=dict(l=50,r=20,t=40,b=40),
                height=350,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No SOC data available")
    
    with col2:
        soh = fetch_series("bms1_soh", sel, params)
        if soh is not None and not soh.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=soh["t"], y=soh["v"],
                mode='lines',
                name='SOH',
                line=dict(color='#ff7f0e', width=2)
            ))
            fig.update_layout(
                title="State of Health (%)",
                xaxis_title="Time",
                yaxis_title="SOH (%)",
                yaxis=dict(range=[90, 105]),
                margin=dict(l=50,r=20,t=40,b=40),
                height=350,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No SOH data available")
    
    # Voltage and Current
    col1, col2 = st.columns(2)
    
    with col1:
        voltage = fetch_series("bms1_v", sel, params)
        if voltage is not None and not voltage.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=voltage["t"], y=voltage["v"],
                mode='lines',
                name='Pack Voltage',
                line=dict(color='#1f77b4', width=2)
            ))
            fig.update_layout(
                title="Pack Voltage",
                xaxis_title="Time",
                yaxis_title="Voltage (V)",
                margin=dict(l=50,r=20,t=40,b=40),
                height=350,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        current = fetch_series("bms1_c", sel, params)
        if current is not None and not current.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=current["t"], y=current["v"],
                mode='lines',
                name='Pack Current',
                line=dict(color='#d62728', width=2)
            ))
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig.update_layout(
                title="Pack Current (Charge +/Discharge -)",
                xaxis_title="Time",
                yaxis_title="Current (A)",
                margin=dict(l=50,r=20,t=40,b=40),
                height=350,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    # Power Flow
    st.markdown("#### PCS Apparent Power")
    pcs = fetch_series("pcs1_ap", sel, params)
    if pcs is not None and not pcs.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pcs["t"], y=pcs["v"],
            mode='lines',
            name='PCS Power',
            line=dict(color='#9467bd', width=2)
        ))
        fig.update_layout(
            title="PCS Apparent Power (kVA)",
            xaxis_title="Time",
            yaxis_title="Power (kVA)",
            margin=dict(l=50,r=20,t=40,b=40),
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No PCS power data available")
    
    # Auxiliary Power
    st.markdown("#### Auxiliary Systems Power")
    aux = fetch_series("aux_m_ap", sel, params)
    if aux is not None and not aux.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=aux["t"], y=aux["v"],
            mode='lines',
            name='Aux Power',
            line=dict(color='#e377c2', width=2)
        ))
        fig.update_layout(
            title="Auxiliary Power Consumption",
            xaxis_title="Time",
            yaxis_title="Power (kW)",
            margin=dict(l=50,r=20,t=40,b=40),
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    # Temperature monitoring
    st.markdown("#### Temperature Monitoring")
    
    temp_signals = {
        "bms1_cell_ave_t": "Cell Average Temp",
        "ac1_outside_t": "AC Outside Temp",
        "dh1_temp": "Environment Temp",
        "pcs1_t_igbt": "IGBT Temp",
        "pcs1_t_env": "PCS Environment"
    }
    
    temp_data = []
    for sig, label in temp_signals.items():
        data = fetch_series(sig, sel, params)
        if data is not None and not data.empty:
            temp_data.append((label, data))
    
    if temp_data:
        fig = go.Figure()
        colors = px.colors.qualitative.Plotly
        for idx, (label, data) in enumerate(temp_data):
            fig.add_trace(go.Scatter(
                x=data["t"], y=data["v"],
                mode='lines',
                name=label,
                line=dict(color=colors[idx % len(colors)], width=2)
            ))
        
        fig.update_layout(
            title="Temperature Trends",
            xaxis_title="Time",
            yaxis_title="Temperature (°C)",
            margin=dict(l=50,r=20,t=40,b=40),
            height=450,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No temperature data available")

with tabs[3]:
    # Energy accounting
    st.markdown("#### Daily Energy Flow")
    
    aux_imp = fetch_series("aux_m_pos_ae", sel, params)
    aux_exp = fetch_series("aux_m_neg_ae", sel, params)
    
    daily = pd.DataFrame(index=pd.DatetimeIndex([]))
    
    if aux_imp is not None and not aux_imp.empty:
        d = aux_imp.set_index("t")["v"].resample("1d").sum(min_count=1)
        daily = daily.join(d.rename("Import"), how="outer")
    
    if aux_exp is not None and not aux_exp.empty:
        d = aux_exp.set_index("t")["v"].resample("1d").sum(min_count=1)
        daily = daily.join(d.rename("Export"), how="outer")
    
    if not daily.empty:
        daily_reset = daily.dropna(how="all").reset_index(names="Date")
        melted = daily_reset.melt(id_vars=["Date"], var_name="Direction", value_name="kWh")
        
        fig = go.Figure()
        colors = {"Import": "#2ca02c", "Export": "#d62728"}
        for direction in melted["Direction"].unique():
            data = melted[melted["Direction"] == direction]
            fig.add_trace(go.Bar(
                x=data["Date"],
                y=data["kWh"],
                name=f"{direction} (kWh/day)",
                marker_color=colors.get(direction, "#1f77b4"),
                text=data["kWh"].round(1),
                textposition='auto',
            ))
        
        fig.update_layout(
            title="Daily Energy Flow",
            xaxis_title="Date",
            yaxis_title="Energy (kWh)",
            barmode='group',
            margin=dict(l=50,r=20,t=40,b=40),
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary stats
        col1, col2, col3 = st.columns(3)
        if "Import" in daily.columns:
            col1.metric("Total Import", f"{daily['Import'].sum():.1f} kWh")
        if "Export" in daily.columns:
            col2.metric("Total Export", f"{daily['Export'].sum():.1f} kWh")
        if "Import" in daily.columns and "Export" in daily.columns:
            col3.metric("Net Energy", f"{(daily['Import'].sum() - daily['Export'].sum()):.1f} kWh")
    else:
        st.info("No energy flow data available")

with tabs[4]:
    # Diagnostics and alarms
    st.markdown("#### System Diagnostics")
    
    # Cell voltage spread
    v_diff = fetch_series("bms1_cell_v_diff", sel, params)
    if v_diff is not None and not v_diff.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=v_diff["t"], y=v_diff["v"],
            mode='lines',
            name='Cell Voltage Difference',
            line=dict(color='#ff7f0e', width=2)
        ))
        # Add warning threshold
        fig.add_hline(y=0.05, line_dash="dash", line_color="orange", 
                     annotation_text="Warning Threshold", opacity=0.7)
        fig.update_layout(
            title="Cell Voltage Imbalance",
            xaxis_title="Time",
            yaxis_title="Voltage Difference (V)",
            margin=dict(l=50,r=20,t=40,b=40),
            height=350,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Temperature spread
    t_diff = fetch_series("bms1_cell_t_diff", sel, params)
    if t_diff is not None and not t_diff.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=t_diff["t"], y=t_diff["v"],
            mode='lines',
            name='Cell Temperature Difference',
            line=dict(color='#d62728', width=2)
        ))
        # Add warning threshold
        fig.add_hline(y=5, line_dash="dash", line_color="orange", 
                     annotation_text="Warning Threshold", opacity=0.7)
        fig.update_layout(
            title="Cell Temperature Imbalance",
            xaxis_title="Time",
            yaxis_title="Temperature Difference (°C)",
            margin=dict(l=50,r=20,t=40,b=40),
            height=350,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Alarm signals check
    st.markdown("#### Alarm History")
    alarm_signals = ["fa1_SmokeFlag", "fa1_ErrCode", "fa1_Level", "fa1_Co", "fa1_Voc"]
    alarm_found = False
    
    for sig in alarm_signals:
        data = fetch_series(sig, sel, params)
        if data is not None and not data.empty and (data["v"] > 0).any():
            alarm_found = True
            st.warning(f"⚠️ {sig}: Alarms detected in the selected period")
    
    if not alarm_found:
        st.success("✅ No alarms detected in the selected period")

# Footer
st.divider()
st.caption(f"📍 BESS System: **{sel.split('/')[-1]}** | 📊 Resolution: {base_rule} | 📅 Period: {dr[0] if dr else 'N/A'} to {dr[1] if len(dr) > 1 else 'N/A'}")