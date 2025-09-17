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
st.sidebar.header("🔧 Controls")

# System selection
bess_systems = get_bess_systems()
if not bess_systems:
    st.error("No BESS systems detected. Ensure preprocessing has run and systems are available.")
    st.stop()

selected_system = st.sidebar.selectbox("BESS System", sorted(bess_systems.keys()))

# Time resolution
resolution = st.sidebar.selectbox(
    "Time Resolution",
    ["1h", "1d", "1w"],
    index=1,
    help="Higher resolution = more detail but slower loading"
)

# Date range (optional)
use_custom_range = st.sidebar.checkbox("Custom Date Range", help="Limit analysis to specific period")
start_date, end_date = None, None

if use_custom_range:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start", value=datetime.now().date() - timedelta(days=30))
    with col2:
        end_date = st.date_input("End", value=datetime.now().date())

# Display system info
st.sidebar.markdown(f"**System Info:**")
st.sidebar.markdown(f"• Cells: 260 (5 packs × 52 cells)")
st.sidebar.markdown(f"• Data: Real voltage curves")

# Load degradation data
with st.spinner(f"Loading degradation data for {selected_system}..."):
    try:
        degradation_data = load_degradation_3d(selected_system, start_date, end_date, resolution)

        # Display metadata
        time_range = degradation_data.get("time_range", {})
        st.success(f"✅ Loaded health data: {degradation_data.get('total_cells', 0)} cells")

        if time_range:
            start_str = time_range.get("start", "Unknown")[:10]
            end_str = time_range.get("end", "Unknown")[:10]
            st.info(f"📅 Data range: {start_str} to {end_str}")

    except Exception as e:
        st.error(f"Failed to load degradation data: {str(e)}")
        st.stop()

# Main content tabs
tab_3d, tab_summary = st.tabs(["🌐 3D Visualization", "📋 Summary"])

with tab_3d:
    st.markdown("### 3D Battery Cell Health Visualization")
    st.markdown("**Interactive 3D surface showing health percentage across all cells over time**")

    if degradation_data.get("degradation_3d"):
        fig_3d = create_3d_degradation_plot(degradation_data)
        st.plotly_chart(fig_3d, use_container_width=True)

        st.markdown("**🎮 Navigation Tips:**")
        st.markdown("• **Drag** to rotate • **Scroll** to zoom • **Hover** for cell details")

        # Health classification legend
        st.markdown("**🏥 Health Classifications:**")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown("🟢 **Excellent** 98-100%")
        with col2:
            st.markdown("🔵 **Good** 90-97%")
        with col3:
            st.markdown("🟡 **Fair** 85-89%")
        with col4:
            st.markdown("🟠 **Poor** 80-84%")
        with col5:
            st.markdown("🔴 **Critical** <80%")
    else:
        st.warning("No 3D degradation data available. Ensure health metrics have been preprocessed.")

with tab_summary:
    st.markdown("### System Health Summary")

    degradation_3d = degradation_data.get("degradation_3d", {})
    if degradation_3d:
        # Calculate summary statistics
        all_health_values = []
        cell_count = 0
        pack_health = {}  # track health by pack

        for cell_key, time_series in degradation_3d.items():
            if not time_series:
                continue

            cell_count += 1
            # Get latest health value
            latest_health = time_series[-1]["health_percentage"] if time_series else 85.0
            all_health_values.append(latest_health)

            # Extract pack info
            parts = cell_key.split('_')
            if len(parts) >= 2:
                pack_num = parts[1]
                if pack_num not in pack_health:
                    pack_health[pack_num] = []
                pack_health[pack_num].append(latest_health)

        if all_health_values:
            # Overall statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Analyzed Cells", cell_count)
            with col2:
                st.metric("Average Health", f"{np.mean(all_health_values):.1f}%")
            with col3:
                st.metric("Min Health", f"{np.min(all_health_values):.1f}%")
            with col4:
                st.metric("Max Health", f"{np.max(all_health_values):.1f}%")

            # Pack-level summary
            if pack_health:
                st.markdown("#### Pack-Level Health Summary")
                pack_summary = []
                for pack_num in sorted(pack_health.keys()):
                    pack_healths = pack_health[pack_num]
                    avg_health = np.mean(pack_healths)
                    min_health = np.min(pack_healths)
                    max_health = np.max(pack_healths)

                    status, color = classify_health(avg_health)

                    pack_summary.append({
                        "Pack": f"Pack {pack_num}",
                        "Cells": len(pack_healths),
                        "Avg Health": f"{avg_health:.1f}%",
                        "Min Health": f"{min_health:.1f}%",
                        "Max Health": f"{max_health:.1f}%",
                        "Status": status
                    })

                st.dataframe(pd.DataFrame(pack_summary), use_container_width=True)

            # Health classification counts
            st.markdown("#### Health Classification Breakdown")
            classification_counts = {
                "🟢 Excellent": 0, "🔵 Good": 0, "🟡 Fair": 0,
                "🟠 Poor": 0, "🔴 Critical": 0
            }

            for health in all_health_values:
                status, _ = classify_health(health)
                classification_counts[status] += 1

            # Display as metrics
            cols = st.columns(5)
            for i, (status, count) in enumerate(classification_counts.items()):
                with cols[i]:
                    percentage = (count / len(all_health_values)) * 100 if all_health_values else 0
                    st.metric(status, f"{count} ({percentage:.1f}%)")
    else:
        st.warning("No summary data available.")

# Footer
st.markdown("---")
st.markdown("*Cell Analyzer uses real voltage curve data to calculate battery degradation metrics.*")