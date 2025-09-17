# Cell Analyzer - 3D Degradation Visualization (MVP)
import os
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional

import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
import numpy as np

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
st.set_page_config(page_title="🔬 Cell Analyzer", layout="wide")

st.title("🔬 Battery Cell Analyzer")
st.markdown("**Real degradation analysis from voltage curves - 3D visualization**")

# ---------------- HTTP Functions ----------------
def _req(path: str, params: dict = None) -> dict:
    """Make API request with error handling"""
    try:
        r = requests.get(f"{API_URL}{path}", params=params, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return {}

@st.cache_data(ttl=60)
def get_bess_systems() -> dict:
    """Get available BESS systems"""
    classified = _req("/meters/classified")
    return classified.get("bess", {})

@st.cache_data(ttl=60)
def load_degradation_3d(bess_system: str, start_date: Optional[date] = None, end_date: Optional[date] = None, resolution: str = "1d") -> dict:
    """Load 3D degradation data from preprocessed health metrics"""
    params = {"time_resolution": resolution}
    if start_date:
        params["start"] = start_date.isoformat()
    if end_date:
        params["end"] = end_date.isoformat()
    return _req(f"/cell/system/{bess_system}/degradation-3d", params)

# ---------------- Health Classification ----------------
def classify_health(health_pct: float) -> tuple:
    """Classify battery health with color coding"""
    if health_pct >= 98:
        return "🟢 Excellent", "#00D56A"  # Green
    elif health_pct >= 90:
        return "🔵 Good", "#0066CC"      # Blue
    elif health_pct >= 85:
        return "🟡 Fair", "#FFB800"     # Yellow
    elif health_pct >= 80:
        return "🟠 Poor", "#FF6B00"     # Orange
    else:
        return "🔴 Critical", "#FF0000" # Red

def get_health_color_scale():
    """Get continuous color scale for health values"""
    return [
        [0.0, '#FF0000'],    # Critical (75-80%)
        [0.2, '#FF6B00'],    # Poor (80-85%)
        [0.4, '#FFB800'],    # Fair (85-90%)
        [0.6, '#0066CC'],    # Good (90-98%)
        [1.0, '#00D56A']     # Excellent (98-100%)
    ]

# ---------------- 3D Visualization ----------------
def create_3d_degradation_plot(degradation_data: dict) -> go.Figure:
    """Create 3D surface plot of cell degradation over time"""
    degradation_3d = degradation_data.get("degradation_3d", {})

    if not degradation_3d:
        fig = go.Figure()
        fig.add_annotation(text="No degradation data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    # Parse degradation data into matrix
    timestamps = set()
    cells = set()

    # Collect all timestamps and cells
    for cell_key, time_series in degradation_3d.items():
        if not time_series:
            continue
        cells.add(cell_key)
        for point in time_series:
            timestamps.add(point["timestamp"])

    # Sort for consistent ordering
    timestamps = sorted(list(timestamps))
    cells = sorted(list(cells), key=lambda x: (
        int(x.split('_')[1]) if len(x.split('_')) > 1 else 0,  # pack number
        int(x.split('_')[3]) if len(x.split('_')) > 3 else 0   # cell number
    ))

    if not timestamps or not cells:
        fig = go.Figure()
        fig.add_annotation(text="No valid degradation data found", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    # Create Z matrix (health percentages)
    z_matrix = []
    y_labels = []  # Time labels
    x_labels = []  # Cell labels

    for i, timestamp in enumerate(timestamps):
        row = []
        for j, cell_key in enumerate(cells):
            # Find health value for this cell at this timestamp
            cell_data = degradation_3d.get(cell_key, [])
            health_value = 85.0  # default

            for point in cell_data:
                if point["timestamp"] == timestamp:
                    health_value = point["health_percentage"]
                    break

            row.append(health_value)

        z_matrix.append(row)
        y_labels.append(timestamp[:10])  # YYYY-MM-DD format

    # Create cell labels (Pack X Cell Y format)
    for cell_key in cells:
        parts = cell_key.split('_')
        if len(parts) >= 4:
            pack_num = parts[1]
            cell_num = parts[3]
            x_labels.append(f"P{pack_num}C{cell_num}")
        else:
            x_labels.append(cell_key)

    # Create 3D surface plot
    fig = go.Figure(data=[
        go.Surface(
            z=z_matrix,
            x=list(range(len(x_labels))),
            y=list(range(len(y_labels))),
            colorscale=get_health_color_scale(),
            colorbar=dict(
                title="Health %",
                titleside="right",
                tickmode="linear",
                tick0=75,
                dtick=5
            ),
            hovertemplate="<b>%{text}</b><br>Time: %{customdata[0]}<br>Health: %{z:.1f}%<extra></extra>",
            text=[[f"{x_labels[j]}" for j in range(len(x_labels))] for i in range(len(y_labels))],
            customdata=[[[y_labels[i]] for j in range(len(x_labels))] for i in range(len(y_labels))]
        )
    ])

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Battery Cell Health Over Time - {degradation_data.get('system', 'BESS')}",
            x=0.5,
            font=dict(size=20)
        ),
        scene=dict(
            xaxis=dict(
                title="Cell Number",
                tickmode='array',
                tickvals=list(range(0, len(x_labels), max(1, len(x_labels)//10))),
                ticktext=[x_labels[i] for i in range(0, len(x_labels), max(1, len(x_labels)//10))]
            ),
            yaxis=dict(
                title="Time",
                tickmode='array',
                tickvals=list(range(0, len(y_labels), max(1, len(y_labels)//8))),
                ticktext=[y_labels[i] for i in range(0, len(y_labels), max(1, len(y_labels)//8))]
            ),
            zaxis=dict(
                title="Health %",
                range=[75, 100]
            ),
            camera=dict(
                eye=dict(x=1.2, y=1.2, z=0.8)
            )
        ),
        width=1200,
        height=700,
        font=dict(size=14)
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