# frontend/pages/cell_analyzer.py
"""
BESS Cell Health Analyzer - Advanced Visualization

Features:
- Pack health comparison across all 5 packs
- 52-cell heatmaps for voltage, temperature, and degradation
- 3D visualization of cells over time
- Charging curve analysis and comparison
- Anomalous cell detection and ranking
- Cell-level degradation tracking
"""

import os
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any
import math

import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
st.set_page_config(page_title="🔋 BESS Cell Analyzer", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size: 16px;
    font-weight: bold;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 15px;
    border-radius: 10px;
    margin: 5px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.danger { border-left: 5px solid #ff4b4b; }
.warning { border-left: 5px solid #ffa500; }
.healthy { border-left: 5px solid #00cc00; }
</style>
""", unsafe_allow_html=True)

st.title("🔋 BESS Cell Health Analyzer")
st.caption("Advanced battery pack and cell-level analysis for 5 packs with 52 cells each")

# ---------------- HTTP Utilities ----------------
def _req(path: str, params: dict = None) -> dict:
    """Make API request with error handling"""
    try:
        response = requests.get(f"{API_URL}{path}", params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return {}

@st.cache_data(ttl=300)  # 5 minute cache
def get_bess_systems() -> dict:
    """Get available BESS systems"""
    classified = _req("/meters/classified")
    return classified.get("bess", {})

@st.cache_data(ttl=60)
def get_pack_health(system: str, pack_id: int, start: str = None, end: str = None) -> dict:
    """Get pack health analysis"""
    params = {}
    if start: params["start"] = start
    if end: params["end"] = end
    return _req(f"/cell/pack/{system}/{pack_id}/health", params)

@st.cache_data(ttl=60)
def get_pack_cells(system: str, pack_id: int, start: str = None, end: str = None) -> dict:
    """Get detailed cell metrics"""
    params = {}
    if start: params["start"] = start
    if end: params["end"] = end
    return _req(f"/cell/pack/{system}/{pack_id}/cells", params)

@st.cache_data(ttl=60)
def get_pack_comparison(system: str, start: str = None, end: str = None) -> dict:
    """Compare all packs in system"""
    params = {}
    if start: params["start"] = start
    if end: params["end"] = end
    return _req(f"/cell/system/{system}/comparison", params)

@st.cache_data(ttl=60)
def get_anomalous_cells(system: str, pack_id: int, start: str = None, end: str = None) -> dict:
    """Get anomalous cells"""
    params = {}
    if start: params["start"] = start
    if end: params["end"] = end
    return _req(f"/cell/pack/{system}/{pack_id}/anomalies", params)

@st.cache_data(ttl=60)
def get_heatmap_data(system: str, pack_id: int, metric: str = "voltage", start: str = None, end: str = None) -> dict:
    """Get heatmap data for visualization"""
    params = {"metric": metric}
    if start: params["start"] = start
    if end: params["end"] = end
    return _req(f"/cell/pack/{system}/{pack_id}/heatmap", params)

@st.cache_data(ttl=60)
def get_charging_curve_data(system: str, pack_id: int, cell_num: int, start: str = None, end: str = None) -> dict:
    """Get time series data for charging curve analysis"""
    params = {"max_points": 10000}
    if start: params["start"] = start
    if end: params["end"] = end
    return _req(f"/series", {"meter": system, "signal": f"bms1_p{pack_id}_v{cell_num}", **params})

# ---------------- Visualization Functions ----------------
def create_pack_comparison_chart(comparison_data: dict) -> go.Figure:
    """Create pack comparison radar chart"""
    if not comparison_data.get("packs"):
        return go.Figure()

    packs = list(comparison_data["packs"].keys())
    metrics = ["pack_soh", "voltage_imbalance", "avg_temperature", "healthy_cells"]

    fig = go.Figure()

    for pack_name, pack_data in comparison_data["packs"].items():
        pack_id = pack_data["pack_id"]

        # Normalize values for radar chart
        values = [
            pack_data["pack_soh"],  # 0-100
            (1 - pack_data["voltage_imbalance"]) * 100,  # Invert so higher is better
            max(0, 60 - pack_data["avg_temperature"]) * 100/60,  # Invert temp
            pack_data["healthy_cells"] / 52 * 100,  # Percentage of healthy cells
        ]

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=["SOH (%)", "Balance", "Thermal", "Healthy Cells (%)"],
            fill='toself',
            name=f"Pack {pack_id}",
            line_color=px.colors.qualitative.Set1[pack_id-1] if pack_id <= len(px.colors.qualitative.Set1) else "#888888"
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        title="Pack Health Comparison",
        showlegend=True,
        height=500
    )

    return fig

def create_cell_heatmap(heatmap_data: dict, pack_id: int) -> go.Figure:
    """Create 52-cell heatmap visualization"""
    if not heatmap_data.get("heatmap_data"):
        return go.Figure()

    data = heatmap_data["heatmap_data"]
    metric = heatmap_data["metric"]

    # Create 13x4 grid (52 cells)
    z_matrix = np.zeros((4, 13))
    cell_text = np.empty((4, 13), dtype=object)

    for cell_data in data:
        x, y = int(cell_data["x"]), int(cell_data["y"])
        z_matrix[y, x] = cell_data["value"]
        cell_text[y, x] = f"Cell {cell_data['cell_num']}<br>{cell_data['value']:.3f}"

    # Color scale based on metric
    if metric == "voltage":
        colorscale = "Viridis"
        title_suffix = "(V)"
    elif metric == "temperature":
        colorscale = "Hot"
        title_suffix = "(°C)"
    elif metric == "degradation":
        colorscale = "Reds"
        title_suffix = "(mV/month)"
    else:
        colorscale = "RdYlBu_r"
        title_suffix = "(V)"

    fig = go.Figure(data=go.Heatmap(
        z=z_matrix,
        text=cell_text,
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate="<b>%{text}</b><extra></extra>",
        colorscale=colorscale,
        showscale=True,
        colorbar=dict(title=f"{metric.title()} {title_suffix}")
    ))

    fig.update_layout(
        title=f"Pack {pack_id} - {metric.title()} Heatmap",
        xaxis=dict(title="Cell Column", dtick=1, range=[-0.5, 12.5]),
        yaxis=dict(title="Cell Row", dtick=1, range=[-0.5, 3.5]),
        height=300,
        width=800,
    )

    return fig

def create_3d_cell_visualization(system: str, pack_id: int, cells_to_show: List[int], date_range: tuple) -> go.Figure:
    """Create 3D visualization of selected cells over time"""
    fig = go.Figure()

    colors = px.colors.qualitative.Set3

    for i, cell_num in enumerate(cells_to_show[:10]):  # Limit to 10 cells for clarity
        try:
            # Get time series data for this cell
            start_date = date_range[0].strftime("%Y-%m-%d") if date_range[0] else None
            end_date = date_range[1].strftime("%Y-%m-%d") if date_range[1] else None

            cell_data = get_charging_curve_data(system, pack_id, cell_num, start_date, end_date)

            if cell_data and cell_data.get("timestamps") and cell_data.get("values"):
                timestamps = pd.to_datetime(cell_data["timestamps"])
                values = cell_data["values"]

                # Convert timestamp to numeric for 3D plot
                time_numeric = [(t - timestamps[0]).total_seconds() / 3600 for t in timestamps]  # Hours from start

                fig.add_trace(go.Scatter3d(
                    x=[cell_num] * len(time_numeric),  # Cell number
                    y=time_numeric,                    # Time (hours)
                    z=values,                          # Voltage (V)
                    mode='lines',
                    name=f'Cell {cell_num}',
                    line=dict(color=colors[i % len(colors)], width=4),
                    hovertemplate=f"<b>Cell {cell_num}</b><br>" +
                                 "Time: %{y:.1f} hours<br>" +
                                 "Voltage: %{z:.3f}V<extra></extra>"
                ))
        except Exception as e:
            st.warning(f"Could not load data for cell {cell_num}: {e}")

    fig.update_layout(
        title=f"3D Cell Voltage Evolution - Pack {pack_id}",
        scene=dict(
            xaxis_title="Cell Number",
            yaxis_title="Time (Hours)",
            zaxis_title="Voltage (V)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        height=600,
        showlegend=True
    )

    return fig

def create_charging_curves_comparison(system: str, pack_id: int, cell_nums: List[int], date_range: tuple) -> go.Figure:
    """Compare charging curves of multiple cells"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["Voltage vs Time", "Voltage Distribution"],
        vertical_spacing=0.1,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )

    colors = px.colors.qualitative.Set1

    for i, cell_num in enumerate(cell_nums):
        try:
            start_date = date_range[0].strftime("%Y-%m-%d") if date_range[0] else None
            end_date = date_range[1].strftime("%Y-%m-%d") if date_range[1] else None

            cell_data = get_charging_curve_data(system, pack_id, cell_num, start_date, end_date)

            if cell_data and cell_data.get("timestamps") and cell_data.get("values"):
                timestamps = pd.to_datetime(cell_data["timestamps"])
                values = cell_data["values"]

                color = colors[i % len(colors)]

                # Time series plot
                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=values,
                        mode='lines',
                        name=f'Cell {cell_num}',
                        line=dict(color=color, width=2),
                        hovertemplate=f"<b>Cell {cell_num}</b><br>" +
                                     "Time: %{x}<br>" +
                                     "Voltage: %{y:.3f}V<extra></extra>"
                    ),
                    row=1, col=1
                )

                # Voltage distribution
                fig.add_trace(
                    go.Histogram(
                        x=values,
                        name=f'Cell {cell_num}',
                        opacity=0.7,
                        marker_color=color,
                        nbinsx=50,
                        showlegend=False
                    ),
                    row=2, col=1
                )

        except Exception as e:
            st.warning(f"Could not load data for cell {cell_num}: {e}")

    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_yaxes(title_text="Voltage (V)", row=1, col=1)
    fig.update_xaxes(title_text="Voltage (V)", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)

    fig.update_layout(
        title=f"Charging Curve Analysis - Pack {pack_id}",
        height=800,
        showlegend=True
    )

    return fig

# ---------------- Main Interface ----------------
# Sidebar controls
st.sidebar.header("🔋 Cell Analyzer Controls")

# System selection
bess_systems = get_bess_systems()
if not bess_systems:
    st.error("No BESS systems found. Please check the API connection.")
    st.stop()

system_names = sorted(bess_systems.keys())
selected_system = st.sidebar.selectbox("BESS System", system_names)

# Date range selection
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=datetime.now().date() - timedelta(days=30))
with col2:
    end_date = st.date_input("End Date", value=datetime.now().date())

# Convert dates to strings
start_str = start_date.strftime("%Y-%m-%d") if start_date else None
end_str = end_date.strftime("%Y-%m-%d") if end_date else None

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Pack Overview",
    "🗺️ Cell Heatmaps",
    "⚠️ Anomaly Detection",
    "📈 Charging Curves",
    "🌐 3D Visualization"
])

with tab1:
    st.header("Pack Health Overview")

    # Load pack comparison data
    comparison_data = get_pack_comparison(selected_system, start_str, end_str)

    if comparison_data and comparison_data.get("packs"):
        # Pack comparison chart
        comparison_fig = create_pack_comparison_chart(comparison_data)
        st.plotly_chart(comparison_fig, use_container_width=True)

        # Pack statistics table
        st.subheader("Pack Statistics")

        pack_stats = []
        for pack_name, pack_data in comparison_data["packs"].items():
            pack_stats.append({
                "Pack": f"Pack {pack_data['pack_id']}",
                "SOH (%)": f"{pack_data['pack_soh']:.1f}%",
                "Avg Voltage (V)": f"{pack_data['average_voltage']:.3f}",
                "Imbalance (V)": f"{pack_data['voltage_imbalance']:.3f}",
                "Avg Temp (°C)": f"{pack_data['avg_temperature']:.1f}",
                "Degradation (V/month)": f"{pack_data['degradation_rate']:.4f}",
                "Cycles": pack_data['discharge_cycles'],
                "Usage": pack_data['usage_pattern'].title(),
                "Health Status": f"✅ {pack_data['healthy_cells']} / ⚠️ {pack_data['warning_cells']} / ❌ {pack_data['critical_cells']}"
            })

        pack_df = pd.DataFrame(pack_stats)
        st.dataframe(pack_df, use_container_width=True)

        # Health summary
        total_healthy = sum(p['healthy_cells'] for p in comparison_data["packs"].values())
        total_warning = sum(p['warning_cells'] for p in comparison_data["packs"].values())
        total_critical = sum(p['critical_cells'] for p in comparison_data["packs"].values())
        total_cells = total_healthy + total_warning + total_critical

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Cells", total_cells)
        with col2:
            st.metric("Healthy Cells", f"{total_healthy} ({total_healthy/total_cells*100:.1f}%)")
        with col3:
            st.metric("Warning Cells", f"{total_warning} ({total_warning/total_cells*100:.1f}%)")
        with col4:
            st.metric("Critical Cells", f"{total_critical} ({total_critical/total_cells*100:.1f}%)")

with tab2:
    st.header("Cell Heatmaps")

    # Pack and metric selection
    col1, col2 = st.columns(2)
    with col1:
        selected_pack = st.selectbox("Select Pack", range(1, 6))
    with col2:
        metric_type = st.selectbox("Metric", ["voltage", "temperature", "degradation"])

    # Load heatmap data
    heatmap_data = get_heatmap_data(selected_system, selected_pack, metric_type, start_str, end_str)

    if heatmap_data:
        # Create heatmap
        heatmap_fig = create_cell_heatmap(heatmap_data, selected_pack)
        st.plotly_chart(heatmap_fig, use_container_width=True)

        # Statistics
        if heatmap_data.get("heatmap_data"):
            values = [cell["value"] for cell in heatmap_data["heatmap_data"]]

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Average", f"{np.mean(values):.3f}")
            with col2:
                st.metric("Min", f"{np.min(values):.3f}")
            with col3:
                st.metric("Max", f"{np.max(values):.3f}")
            with col4:
                st.metric("Std Dev", f"{np.std(values):.3f}")

with tab3:
    st.header("Anomaly Detection")

    # Pack selection for anomaly detection
    anomaly_pack = st.selectbox("Select Pack for Analysis", range(1, 6), key="anomaly_pack")

    # Load anomalous cells
    anomalies_data = get_anomalous_cells(selected_system, anomaly_pack, start_str, end_str)

    if anomalies_data and anomalies_data.get("anomalies"):
        st.subheader(f"Found {anomalies_data['anomalous_cells_count']} Anomalous Cells in Pack {anomaly_pack}")

        # Display top anomalies
        for i, anomaly in enumerate(anomalies_data["anomalies"][:10], 1):
            cell = anomaly["cell"]
            reasons = anomaly.get("reasons", [])
            severity = anomaly["severity_score"]

            # Determine severity color
            if severity > 50:
                severity_class = "danger"
                severity_icon = "🚨"
            elif severity > 20:
                severity_class = "warning"
                severity_icon = "⚠️"
            else:
                severity_class = "healthy"
                severity_icon = "🔍"

            with st.container():
                st.markdown(f"""
                <div class="metric-card {severity_class}">
                <h4>{severity_icon} Cell {cell['cell_num']} - Severity Score: {severity:.1f}</h4>
                <p><strong>Voltage:</strong> {cell['voltage_mean']:.3f}V (min: {cell['voltage_min']:.3f}V, max: {cell['voltage_max']:.3f}V)</p>
                <p><strong>Degradation:</strong> {cell['degradation_rate']:.4f}V/month</p>
                <p><strong>Imbalance:</strong> {cell['imbalance_score']:.3f}V</p>
                <p><strong>Max Temp:</strong> {cell['temp_max']:.1f}°C</p>
                <p><strong>Data Quality:</strong> {cell['data_quality']:.1f}% ({cell['data_points']:,} points)</p>
                <p><strong>Issues:</strong> {', '.join(reasons) if reasons else 'Statistical anomaly detected'}</p>
                </div>
                """, unsafe_allow_html=True)

with tab4:
    st.header("Charging Curve Analysis")

    # Pack and cell selection
    col1, col2 = st.columns(2)
    with col1:
        curve_pack = st.selectbox("Select Pack", range(1, 6), key="curve_pack")
    with col2:
        # Multi-select for cells to compare
        available_cells = list(range(1, 53))
        selected_cells = st.multiselect(
            "Select Cells to Compare",
            available_cells,
            default=[1, 26, 52],  # First, middle, last cells
            max_selections=10
        )

    if selected_cells:
        # Create charging curves comparison
        curves_fig = create_charging_curves_comparison(
            selected_system, curve_pack, selected_cells, (start_date, end_date)
        )
        st.plotly_chart(curves_fig, use_container_width=True)

        # Analysis insights
        st.subheader("Charging Analysis Insights")
        st.info("""
        **Interpreting Charging Curves:**
        - **Healthy cells** show similar voltage patterns and distributions
        - **Degraded cells** may show lower peak voltages or irregular charging patterns
        - **Imbalanced cells** will have shifted voltage distributions
        - **Failing cells** often exhibit voltage drops or unusual spikes
        """)

with tab5:
    st.header("3D Visualization")

    # 3D visualization controls
    col1, col2 = st.columns(2)
    with col1:
        viz_pack = st.selectbox("Select Pack", range(1, 6), key="viz_pack")
    with col2:
        # Cell selection for 3D visualization
        viz_cells = st.multiselect(
            "Select Cells for 3D View",
            list(range(1, 53)),
            default=[1, 13, 26, 39, 52],  # Corner and center cells
            max_selections=8
        )

    if viz_cells:
        # Create 3D visualization
        viz_3d_fig = create_3d_cell_visualization(
            selected_system, viz_pack, viz_cells, (start_date, end_date)
        )
        st.plotly_chart(viz_3d_fig, use_container_width=True)

        st.subheader("3D Visualization Guide")
        st.info("""
        **Understanding the 3D Plot:**
        - **X-axis:** Cell number (1-52)
        - **Y-axis:** Time progression (hours from start date)
        - **Z-axis:** Cell voltage (V)
        - **Patterns:** Look for cells that deviate from the pack behavior
        - **Degradation:** Cells showing voltage drops over time
        - **Charging cycles:** Visible as voltage oscillations over time
        """)

# Footer with system info
st.markdown("---")
st.caption(f"📊 Analyzing {selected_system} | 🔋 5 packs × 52 cells = 260 cells total | 📅 {start_date} to {end_date}")

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "Analyze BESS pack data structure and cell patterns", "status": "completed", "activeForm": "Analyzed BESS pack data structure and cell patterns"}, {"content": "Create cell analyzer backend endpoint for pack health analysis", "status": "completed", "activeForm": "Created cell analyzer backend endpoint for pack health analysis"}, {"content": "Build frontend page for cell-level visualization and analysis", "status": "completed", "activeForm": "Built frontend page for cell-level visualization and analysis"}, {"content": "Implement cell imbalance detection algorithms", "status": "completed", "activeForm": "Implemented cell imbalance detection algorithms"}, {"content": "Add temporal analysis for cell degradation patterns", "status": "completed", "activeForm": "Added temporal analysis for cell degradation patterns"}, {"content": "Create heatmap visualization for 52 cells per pack", "status": "completed", "activeForm": "Created heatmap visualization for 52 cells per pack"}]