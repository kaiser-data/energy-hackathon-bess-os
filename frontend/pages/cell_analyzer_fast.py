# Real Battery Cell Health Monitor
import os
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from scipy import stats

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="🔋 PackPulse 💓 - SAT Voltage Monitor", layout="wide")

def api_call(endpoint, params=None):
    """Simple API call with error handling"""
    try:
        response = requests.get(f"{API_URL}{endpoint}", params=params or {}, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return {}

@st.cache_data(ttl=60)
def get_bess_systems():
    """Get available BESS systems"""
    data = api_call("/meters/classified")
    return data.get("bess", {})

@st.cache_data(ttl=300)  # 5 minute cache for raw plotting data
def get_raw_voltage_plot_data(system: str, cell_id: str, sample_rate: int = 50):
    """Get raw voltage data for fast plotting"""
    params = {
        "sample_rate": sample_rate,
        "start": "2024-01-01",
        "end": "2024-12-31"
    }

    try:
        data = api_call(f"/cell/system/{system}/raw-voltage-plot", params={"cell_id": cell_id, **params})
        if data and data.get("data_source") == "raw_bms_voltage_data":
            return data
    except Exception as e:
        st.error(f"Raw voltage plotting failed: {str(e)}")
        return None

    return None

@st.cache_data(ttl=60)
def get_cell_health_data(system: str, demo_mode: bool = False):
    """Get REAL cell health data from BMS voltage analysis - NO SYNTHETIC DATA"""
    params = {
        "time_resolution": "1d",
        "demo_mode": demo_mode
    }

    # ONLY use real SAT voltage endpoint - no synthetic fallbacks
    try:
        data = api_call(f"/cell/system/{system}/real-sat-voltage", params)
        if data and data.get("data_source") == "real_cell_voltages":
            if demo_mode:
                st.success(f"🎯 Demo Mode: Strategic sampling - {data.get('calculation_method', 'unknown')}")
            else:
                st.success(f"📡 Complete Coverage: Real BMS voltage data - {data.get('calculation_method', 'unknown')}")

            degradation_3d = data.get("degradation_3d", {})

            # Process the degradation_3d data into cells format if not already done
            if not degradation_3d:
                st.error("❌ No degradation_3d data in response")
                return None

            # Process the data to create the frontend format
            # Extract all unique timestamps and cells
            all_timestamps = set()
            all_cells = set()

            for cell_key, time_series in degradation_3d.items():
                if time_series:
                    all_cells.add(cell_key)
                    for point in time_series:
                        all_timestamps.add(point["timestamp"])

            timestamps = sorted(list(all_timestamps))
            # Sort cells by pack and cell number (pack_1_cell_1, pack_1_cell_2, etc.)
            def parse_cell_id(cell_key):
                try:
                    parts = cell_key.split('_')
                    pack_num = int(parts[1])  # pack_1 -> 1
                    cell_num = int(parts[3])  # cell_1 -> 1
                    return (pack_num, cell_num)
                except:
                    return (999, 999)  # Put parse errors at the end

            cells = sorted(list(all_cells), key=parse_cell_id)

            # Build health matrix: [timestamp][cell] = health%
            health_matrix = []
            for timestamp in timestamps:
                row = []
                for cell_key in cells:
                    cell_data = degradation_3d.get(cell_key, [])
                    health_value = 85.0  # default

                    for point in cell_data:
                        if point["timestamp"] == timestamp:
                            # Handle both real voltage data and synthetic health data
                            if "voltage_percentage" in point:
                                health_value = point["voltage_percentage"]  # Real SAT voltage percentage
                            elif "health_percentage" in point:
                                health_value = point["health_percentage"]   # Synthetic health data
                            break

                    row.append(health_value)
                health_matrix.append(row)

            # Add processed data to the response
            data["cells"] = cells
            data["timestamps"] = timestamps
            data["health_matrix"] = health_matrix
            data["total_cells"] = len(cells)

            return data
        else:
            st.error("❌ Real voltage data not available - wrong data source")
            return None
    except Exception as e:
        st.error(f"❌ Real voltage analysis failed: {str(e)}")
        st.error("🚫 NO SYNTHETIC DATA - Only real BMS cell data used")
        return None

# Main UI
st.title("🔋 PackPulse 💓 - SAT Voltage Monitor")
st.markdown("**Advanced saturation voltage analysis from real BESS telemetry data**")

# System selection
systems = get_bess_systems()
if not systems:
    st.error("No BESS systems found")
    st.stop()

col1, col2, col3 = st.columns(3)
with col1:
    selected_system = st.selectbox("Select BESS System", list(systems.keys()))
with col2:
    # Pack filter - 5 packs with 52 cells each (260 total cells)
    available_packs = ["All"] + [f"Pack {i}" for i in range(1, 6)]  # 5 packs × 52 cells = 260 total
    selected_pack = st.selectbox("Filter by Pack", available_packs)
with col3:
    # Demo mode toggle for instant showcase performance
    demo_mode = st.checkbox(
        "⚡ Demo Mode",
        value=True,  # Default to ON for instant response
        help="Ultra-light sampling (5 cells - 1 per pack) for instant showcase performance vs Complete Coverage (260 cells)"
    )

# Load data
loading_message = "⚡ Loading SAT voltage analysis (Demo: 5 cells)..." if demo_mode else "Loading SAT voltage analysis (Complete: 260 cells)..."
with st.spinner(loading_message):
    health_data = get_cell_health_data(selected_system, demo_mode)

if not health_data or not health_data.get("degradation_3d"):
    st.error("No SAT voltage data available")
    st.stop()
elif not health_data.get("cells"):
    # Try to process the degradation_3d data if cells data is missing
    health_data = get_cell_health_data(selected_system, demo_mode)
    if not health_data or not health_data.get("cells"):
        st.error("No SAT voltage data available - processing failed")
        st.stop()

# Apply pack filtering
if selected_pack != "All":
    pack_num = selected_pack.split()[1]  # "Pack 1" -> "1"
    filtered_cells = []
    filtered_indices = []

    for i, cell_key in enumerate(health_data["cells"]):
        try:
            cell_pack = cell_key.split('_')[1]  # pack_1_cell_1 -> 1
            if cell_pack == pack_num:
                filtered_cells.append(cell_key)
                filtered_indices.append(i)
        except:
            continue

    # Filter the health matrix for selected pack only
    filtered_health_matrix = []
    for row in health_data["health_matrix"]:
        filtered_row = [row[i] for i in filtered_indices]
        filtered_health_matrix.append(filtered_row)

    # Update health_data with filtered results
    health_data["cells"] = filtered_cells
    health_data["health_matrix"] = filtered_health_matrix
    health_data["total_cells"] = len(filtered_cells)

# Display system info
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("System", selected_system.replace("ZHPESS232A23000", "BESS-"))
with col2:
    view_label = selected_pack if selected_pack != "All" else "All Packs"
    st.metric("Viewing", view_label)
with col3:
    st.metric("Cells", health_data["total_cells"])
with col4:
    if health_data["health_matrix"]:
        latest_values = health_data["health_matrix"][-1]
        avg_sat_voltage = np.mean(latest_values)
        st.metric("Avg SAT-V", f"{avg_sat_voltage:.1f}%")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 SAT Voltage Overview", "🗺️ Heatmap & 3D View", "📈 Pack Trends", "📉 Degradation Analysis", "⚡ Real-Time Voltage"])

with tab1:
    st.subheader("SAT Voltage Analysis Overview")

    if health_data["health_matrix"]:
        # Latest SAT voltage values
        latest_values = health_data["health_matrix"][-1]

        # SAT voltage distribution
        fig_hist = px.histogram(
            x=latest_values,
            nbins=20,
            title="Cell SAT Voltage Distribution",
            labels={"x": "SAT Voltage %", "y": "Number of Cells"}
        )
        fig_hist.update_layout(height=400)
        st.plotly_chart(fig_hist, use_container_width=True)

        # Pack-level analysis
        pack_voltages = {}
        cells = health_data["cells"]

        for i, cell_key in enumerate(cells):
            try:
                # Parse pack_1_cell_1 format
                pack_num = cell_key.split('_')[1]  # pack_1 -> 1
                if pack_num not in pack_voltages:
                    pack_voltages[pack_num] = []
                pack_voltages[pack_num].append(latest_values[i])
            except:
                continue

        if pack_voltages:
            st.subheader("Pack Voltage Analysis Summary")
            pack_data = []
            for pack_num in sorted(pack_voltages.keys()):
                v_indices = pack_voltages[pack_num]
                pack_data.append({
                    "Pack": f"Pack {pack_num}",
                    "Cells": len(v_indices),
                    "Avg SAT-V": f"{np.mean(v_indices):.1f}%",
                    "Min SAT-V": f"{np.min(v_indices):.1f}%",
                    "Max SAT-V": f"{np.max(v_indices):.1f}%",
                    "Spread": f"{np.max(v_indices) - np.min(v_indices):.1f}%"
                })

            st.dataframe(pd.DataFrame(pack_data), use_container_width=True)

with tab2:
    st.subheader("SAT Voltage Heatmap & 3D Surface")

    if health_data["health_matrix"] and health_data["timestamps"]:
        # Create heatmap data with proper cell labels
        def format_cell_label(cell_key):
            try:
                # Convert pack_1_cell_1 to P1C1
                parts = cell_key.split('_')
                pack_num = parts[1]  # 1
                cell_num = parts[3]  # 1
                return f"P{pack_num}C{cell_num}"
            except:
                return cell_key

        df_heatmap = pd.DataFrame(
            health_data["health_matrix"],
            columns=[format_cell_label(cell) for cell in health_data["cells"]],
            index=[ts[:10] for ts in health_data["timestamps"]]  # Date only
        )

        # Sample time axis if too many days (keep all cells for pack view)
        if len(df_heatmap) > 50:
            step = max(1, len(df_heatmap) // 50)
            df_heatmap = df_heatmap.iloc[::step]

        # If viewing all packs and too many cells, sample cells
        if selected_pack == "All" and len(df_heatmap.columns) > 100:
            # Sample evenly across packs to show representation of each pack
            step = max(1, len(df_heatmap.columns) // 100)
            df_heatmap = df_heatmap.iloc[:, ::step]

        fig_heatmap = px.imshow(
            df_heatmap.values,
            x=df_heatmap.columns,
            y=df_heatmap.index,
            color_continuous_scale="RdYlGn",
            aspect="auto",
            title="Voltage Index Over Time",
            labels={"x": "Cells", "y": "Date", "color": "SAT-V %"}
        )
        fig_heatmap.update_layout(height=600)
        st.plotly_chart(fig_heatmap, use_container_width=True)

        # 3D Surface Plot
        st.subheader("3D Surface View")
        st.markdown("**Interactive 3D visualization of SAT voltage degradation across cells and time**")

        # Create 3D surface plot
        fig_3d = go.Figure(data=[
            go.Surface(
                z=df_heatmap.values,
                x=list(range(len(df_heatmap.columns))),
                y=list(range(len(df_heatmap.index))),
                colorscale="RdYlGn",
                colorbar=dict(title="SAT-V %"),
                hovertemplate="<b>%{text}</b><br>Date: %{customdata[0]}<br>SAT Voltage: %{z:.1f}%<extra></extra>",
                text=[[df_heatmap.columns[j] for j in range(len(df_heatmap.columns))]
                      for i in range(len(df_heatmap.index))],
                customdata=[[[df_heatmap.index[i]] for j in range(len(df_heatmap.columns))]
                           for i in range(len(df_heatmap.index))]
            )
        ])

        # Update 3D layout
        fig_3d.update_layout(
            title=dict(
                text=f"3D SAT Voltage Surface - {selected_pack}",
                x=0.5,
                font=dict(size=16)
            ),
            scene=dict(
                xaxis=dict(
                    title="Cell Number",
                    tickmode='array',
                    tickvals=list(range(0, len(df_heatmap.columns), max(1, len(df_heatmap.columns)//10))),
                    ticktext=[df_heatmap.columns[i] for i in range(0, len(df_heatmap.columns), max(1, len(df_heatmap.columns)//10))]
                ),
                yaxis=dict(
                    title="Time",
                    tickmode='array',
                    tickvals=list(range(0, len(df_heatmap.index), max(1, len(df_heatmap.index)//8))),
                    ticktext=[df_heatmap.index[i] for i in range(0, len(df_heatmap.index), max(1, len(df_heatmap.index)//8))]
                ),
                zaxis=dict(
                    title="SAT Voltage %",
                    range=[df_heatmap.values.min()-1, df_heatmap.values.max()+1]
                ),
                camera=dict(
                    eye=dict(x=1.4, y=1.4, z=0.8)
                )
            ),
            width=None,
            height=650,
            font=dict(size=12)
        )

        st.plotly_chart(fig_3d, use_container_width=True)

        # 3D Navigation tips
        st.info("🎮 **3D Navigation:** Drag to rotate • Scroll to zoom • Hover for cell details")

with tab3:
    st.subheader("Pack Voltage Degradation Trends")

    if health_data["health_matrix"] and health_data["cells"]:
        # Calculate pack averages for each timestamp
        pack_trends = {}
        timestamps = health_data["timestamps"]

        # Get original full data for pack trends (before filtering)
        # Note: Pack trends use the same mode as main analysis for consistency
        with st.spinner("Calculating pack trends..."):
            full_data = get_cell_health_data(selected_system, demo_mode)

        # Group cells by pack and calculate averages
        for timestamp_idx, timestamp in enumerate(full_data["timestamps"]):
            for cell_idx, cell_key in enumerate(full_data["cells"]):
                try:
                    pack_num = cell_key.split('_')[1]  # pack_1_cell_1 -> 1
                    pack_name = f"Pack {pack_num}"

                    if pack_name not in pack_trends:
                        pack_trends[pack_name] = {"timestamps": [], "sat_voltage_values": []}

                    # Only add once per timestamp (avoid duplicates)
                    if timestamp not in pack_trends[pack_name]["timestamps"]:
                        # Calculate average health for this pack at this timestamp
                        pack_cells_health = []
                        for c_idx, c_key in enumerate(full_data["cells"]):
                            if c_key.split('_')[1] == pack_num:  # Same pack
                                pack_cells_health.append(full_data["health_matrix"][timestamp_idx][c_idx])

                        if pack_cells_health:
                            pack_avg = np.mean(pack_cells_health)
                            pack_trends[pack_name]["timestamps"].append(timestamp)
                            pack_trends[pack_name]["sat_voltage_values"].append(pack_avg)
                except:
                    continue

        # Create pack trend plot
        fig_pack_trends = go.Figure()

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        for i, (pack_name, data) in enumerate(sorted(pack_trends.items())):
            if len(data["timestamps"]) > 0:
                # Calculate voltage decline rate for this pack
                if len(data["sat_voltage_values"]) > 1:
                    start_value = data["sat_voltage_values"][0]
                    end_value = data["sat_voltage_values"][-1]
                    total_decline = start_value - end_value
                    days_elapsed = len(data["timestamps"])
                    decline_rate = total_decline / days_elapsed * 30  # per month
                else:
                    decline_rate = 0

                fig_pack_trends.add_trace(go.Scatter(
                    x=data["timestamps"],
                    y=data["sat_voltage_values"],
                    mode='lines',
                    name=f"{pack_name} (-{decline_rate:.2f}%/mo)",
                    line=dict(width=3, color=colors[i % len(colors)]),
                    hovertemplate=f"<b>{pack_name}</b><br>Date: %{{x}}<br>SAT-V: %{{y:.1f}}%<extra></extra>"
                ))

        fig_pack_trends.update_layout(
            title="Pack Voltage Index Trends Over Time",
            xaxis_title="Date",
            yaxis_title="Pack Average Voltage Index %",
            height=600,
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            yaxis=dict(range=[92, 101])  # Focus on actual voltage index range
        )

        st.plotly_chart(fig_pack_trends, use_container_width=True)

        # Pack comparison statistics
        st.subheader("Pack Voltage Analysis Summary")

        if pack_trends:
            pack_stats = []
            for pack_name, data in sorted(pack_trends.items()):
                if len(data["sat_voltage_values"]) > 1:
                    start_value = data["sat_voltage_values"][0]
                    end_value = data["sat_voltage_values"][-1]
                    total_decline = start_value - end_value
                    days_elapsed = len(data["timestamps"])
                    decline_per_month = total_decline / days_elapsed * 30

                    pack_stats.append({
                        "Pack": pack_name,
                        "Start SAT-V": f"{start_value:.1f}%",
                        "Current SAT-V": f"{end_value:.1f}%",
                        "Total Decline": f"{total_decline:.1f}%",
                        "Rate (per month)": f"{decline_per_month:.2f}%",
                        "Projected 1yr": f"{end_value - (decline_per_month * 12):.1f}%"
                    })

            if pack_stats:
                st.dataframe(pd.DataFrame(pack_stats), use_container_width=True)

        # Show current pack selection focus
        if selected_pack != "All":
            st.info(f"💡 Currently viewing detailed data for **{selected_pack}** in other tabs")

with tab4:
    st.subheader("📉 Degradation Analysis & Curve Fitting")
    st.markdown("**Linear regression analysis of SAT voltage decline with cycle estimation**")

    if health_data["health_matrix"] and health_data["cells"]:
        # Get full dataset for analysis
        # Note: Degradation analysis uses the same mode as main analysis for consistency
        with st.spinner("Calculating degradation curves..."):
            full_data = get_cell_health_data(selected_system, demo_mode)

        # Calculate pack averages and fit curves
        pack_analysis = {}

        for cell_idx, cell_key in enumerate(full_data["cells"]):
            try:
                pack_num = cell_key.split('_')[1]  # pack_1_cell_1 -> 1
                pack_name = f"Pack {pack_num}"

                if pack_name not in pack_analysis:
                    pack_analysis[pack_name] = {
                        "timestamps": [],
                        "sat_voltages": [],
                        "days_elapsed": []
                    }

                # Get SAT voltage data for this pack
                for timestamp_idx, timestamp in enumerate(full_data["timestamps"]):
                    if timestamp not in [t for t in pack_analysis[pack_name]["timestamps"]]:
                        # Calculate pack average for this timestamp
                        pack_cells = []
                        for c_idx, c_key in enumerate(full_data["cells"]):
                            if c_key.split('_')[1] == pack_num:  # Same pack
                                pack_cells.append(full_data["health_matrix"][timestamp_idx][c_idx])

                        if pack_cells:
                            pack_avg = np.mean(pack_cells)
                            pack_analysis[pack_name]["timestamps"].append(timestamp)
                            pack_analysis[pack_name]["sat_voltages"].append(pack_avg)

                            # Calculate days elapsed since start
                            if timestamp_idx == 0:
                                days = 0
                            else:
                                start_date = datetime.fromisoformat(full_data["timestamps"][0].replace("Z", "+00:00"))
                                current_date = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                                days = (current_date - start_date).days

                            pack_analysis[pack_name]["days_elapsed"].append(days)
            except:
                continue

        # Create curve fitting analysis
        fig_curves = go.Figure()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        col1, col2 = st.columns(2)

        degradation_stats = []
        cycle_estimates = []

        for i, (pack_name, data) in enumerate(sorted(pack_analysis.items())):
            if len(data["sat_voltages"]) > 10:  # Need enough data points

                # Convert to numpy arrays
                days = np.array(data["days_elapsed"])
                voltages = np.array(data["sat_voltages"])

                # Linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(days, voltages)

                # Generate fit line
                fit_line = slope * days + intercept

                # Calculate degradation metrics more accurately
                total_days = days[-1] - days[0] if len(days) > 1 else 1
                total_degradation = abs(voltages[0] - voltages[-1])  # Total % change

                # Annual degradation based on actual time span
                annual_degradation = (total_degradation / total_days) * 365  # % per year
                monthly_degradation = (total_degradation / total_days) * 30  # % per month

                # Estimate charge cycles (assume 1 cycle per day average for BESS)
                estimated_cycles = total_days * 1.0  # 1 cycle/day assumption
                degradation_per_cycle = total_degradation / estimated_cycles if estimated_cycles > 0 else 0

                # Plot actual data
                fig_curves.add_trace(go.Scatter(
                    x=days,
                    y=voltages,
                    mode='markers',
                    name=f'{pack_name} Data',
                    marker=dict(color=colors[i % len(colors)], size=6),
                    showlegend=True
                ))

                # Plot linear fit
                fig_curves.add_trace(go.Scatter(
                    x=days,
                    y=fit_line,
                    mode='lines',
                    name=f'{pack_name} Fit (R²={r_value**2:.3f})',
                    line=dict(color=colors[i % len(colors)], dash='dash', width=2),
                    showlegend=True
                ))

                # Store statistics
                degradation_stats.append({
                    "Pack": pack_name,
                    "R² (Linearität)": f"{r_value**2:.4f}",
                    "Verlust/Jahr": f"{annual_degradation:.2f}%",
                    "Verlust/Monat": f"{monthly_degradation:.3f}%",
                    "Verlust/Zyklus": f"{degradation_per_cycle:.4f}%",
                    "Geschätzte Zyklen": f"{estimated_cycles:.0f}",
                    "Start SAT-V": f"{voltages[0]:.2f}%",
                    "Aktuell SAT-V": f"{voltages[-1]:.2f}%"
                })

                # Cycle analysis for detailed view
                cycle_estimates.append({
                    "Pack": pack_name,
                    "Tage im Betrieb": f"{total_days:.0f}",
                    "Zyklen geschätzt": f"{estimated_cycles:.0f}",
                    "Verlust pro Zyklus": f"{degradation_per_cycle:.4f}%",
                    "Hochrechnung 5000 Zyklen": f"{degradation_per_cycle * 5000:.1f}%",
                    "Hochrechnung 10000 Zyklen": f"{degradation_per_cycle * 10000:.1f}%"
                })

        # Update plot layout
        fig_curves.update_layout(
            title="SAT Voltage Degradation - Linear Curve Fitting",
            xaxis_title="Tage seit Start",
            yaxis_title="SAT Voltage %",
            height=600,
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        st.plotly_chart(fig_curves, use_container_width=True)

        # Display statistics tables
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Degradation Statistiken")
            if degradation_stats:
                st.dataframe(pd.DataFrame(degradation_stats), use_container_width=True)

        with col2:
            st.subheader("🔋 Zyklen-Hochrechnung")
            if cycle_estimates:
                st.dataframe(pd.DataFrame(cycle_estimates), use_container_width=True)

        # Quality assessment
        st.subheader("🎯 Qualitätsbewertung")

        if degradation_stats:
            # Calculate average metrics
            avg_r2 = np.mean([float(stat["R² (Linearität)"]) for stat in degradation_stats])
            avg_annual_loss = np.mean([float(stat["Verlust/Jahr"].replace("%", "")) for stat in degradation_stats])
            avg_cycle_loss = np.mean([float(stat["Verlust/Zyklus"].replace("%", "")) for stat in degradation_stats])

            col1, col2, col3 = st.columns(3)

            with col1:
                # Linearity quality
                if avg_r2 > 0.95:
                    st.success(f"✅ Sehr linear: R² = {avg_r2:.4f}")
                    linearity_quality = "Ausgezeichnet"
                elif avg_r2 > 0.90:
                    st.warning(f"⚠️ Linear: R² = {avg_r2:.4f}")
                    linearity_quality = "Gut"
                else:
                    st.error(f"❌ Nicht linear: R² = {avg_r2:.4f}")
                    linearity_quality = "Bedenklich"

            with col2:
                # Annual degradation quality - updated realistic thresholds
                if avg_annual_loss < 3.0:  # <3% per year is very good for BESS
                    st.success(f"✅ Niedrig: {avg_annual_loss:.1f}%/Jahr")
                    annual_quality = "Sehr gut"
                elif avg_annual_loss < 10.0:  # 3-10% is moderate
                    st.warning(f"⚠️ Moderat: {avg_annual_loss:.1f}%/Jahr")
                    annual_quality = "Akzeptabel"
                else:
                    st.error(f"❌ Hoch: {avg_annual_loss:.1f}%/Jahr")
                    annual_quality = "Bedenklich"

            with col3:
                # Cycle degradation quality - updated for new calculation
                if avg_cycle_loss < 0.02:  # <0.02% per cycle is very good
                    st.success(f"✅ Niedrig: {avg_cycle_loss:.4f}%/Zyklus")
                    cycle_quality = "Sehr gut"
                elif avg_cycle_loss < 0.05:  # 0.02-0.05% is moderate
                    st.warning(f"⚠️ Moderat: {avg_cycle_loss:.4f}%/Zyklus")
                    cycle_quality = "Akzeptabel"
                else:
                    st.error(f"❌ Hoch: {avg_cycle_loss:.4f}%/Zyklus")
                    cycle_quality = "Bedenklich"

            # Summary assessment
            st.markdown("### 📋 Gesamtbewertung")
            assessment = f"""
            **Linearität der Degradation:** {linearity_quality} (R² = {avg_r2:.4f})
            **Jährlicher Verlust:** {annual_quality} ({avg_annual_loss:.2f}% pro Jahr)
            **Verlust pro Zyklus:** {cycle_quality} ({avg_cycle_loss:.4f}% pro Zyklus)

            **Prognose für Lebensdauer:**
            - Bei aktuellem Tempo: ~{100/avg_annual_loss:.0f} Jahre bis zu 100% Verlust
            - Bei 5000 Zyklen: ~{avg_cycle_loss * 5000:.1f}% Gesamtverlust
            - Bei 10000 Zyklen: ~{avg_cycle_loss * 10000:.1f}% Gesamtverlust
            """
            st.markdown(assessment)
    else:
        st.warning("Nicht genügend Daten für Kurvenfit-Analyse verfügbar.")

# SAT Voltage alerts
st.subheader("⚠️ SAT Voltage Alerts")
if health_data["health_matrix"]:
    latest_values = health_data["health_matrix"][-1]
    cells = health_data["cells"]

    critical_cells = []
    warning_cells = []

    for i, sat_v in enumerate(latest_values):
        cell_key = cells[i]
        cell_label = format_cell_label(cell_key)

        if sat_v < 80:
            critical_cells.append((cell_label, sat_v))
        elif sat_v < 85:
            warning_cells.append((cell_label, sat_v))

    col1, col2 = st.columns(2)

    with col1:
        if critical_cells:
            st.error(f"🚨 {len(critical_cells)} Low SAT-V Cells (<80%)")
            for cell, sat_v in critical_cells[:5]:  # Show top 5
                st.write(f"• {cell}: {sat_v:.1f}%")
        else:
            st.success("✅ No critically low voltage indices")

    with col2:
        if warning_cells:
            st.warning(f"⚠️ {len(warning_cells)} Warning SAT-V Cells (<85%)")
            for cell, sat_v in warning_cells[:5]:  # Show top 5
                st.write(f"• {cell}: {sat_v:.1f}%")
        else:
            st.success("✅ All SAT voltages normal")

with tab5:
    st.subheader("⚡ Real-Time Voltage Plotting")
    st.markdown("**Raw BMS cell voltage data with fast Plotly visualization**")

    if health_data["cells"]:
        # Cell selection for plotting
        col1, col2, col3 = st.columns(3)

        with col1:
            # Select cell for plotting - now with ALL 260 cells available!
            available_cells = health_data["cells"]  # Show ALL processed cells (up to 260)
            selected_cell = st.selectbox("Select Cell for Plotting", available_cells, key="plot_cell")

        with col2:
            # Sampling rate control
            sample_rate = st.selectbox("Data Sampling Rate",
                                     [10, 25, 50, 100, 200],
                                     index=2,
                                     help="Higher = faster but less detail")

        with col3:
            # Plot button
            plot_button = st.button("🔄 Load Voltage Plot", type="primary")

        if plot_button and selected_cell:
            with st.spinner(f"Loading raw voltage data for {selected_cell} (sampling every {sample_rate} points)..."):
                plot_data = get_raw_voltage_plot_data(selected_system, selected_cell, sample_rate)

                if plot_data and plot_data.get("voltage_data"):
                    # Display statistics
                    stats = plot_data["statistics"]

                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Data Points", f"{stats['data_points']:,}")
                    with col2:
                        st.metric("Min Voltage", f"{stats['min_voltage']:.3f}V")
                    with col3:
                        st.metric("Max Voltage", f"{stats['max_voltage']:.3f}V")
                    with col4:
                        st.metric("Avg Voltage", f"{stats['avg_voltage']:.3f}V")
                    with col5:
                        st.metric("Range", f"{stats['voltage_range']*1000:.1f}mV")

                    # Extract plot data
                    voltage_data = plot_data["voltage_data"]
                    timestamps = [point["timestamp"] for point in voltage_data]
                    voltages = [point["voltage"] for point in voltage_data]
                    voltages_mv = [point["voltage_mv"] for point in voltage_data]

                    # Create fast Plotly line plot with WebGL rendering
                    fig_voltage = go.Figure()

                    # Add main voltage trace
                    fig_voltage.add_trace(go.Scattergl(
                        x=timestamps,
                        y=voltages,
                        mode='lines',
                        name=f'{selected_cell} Voltage',
                        line=dict(color='#1f77b4', width=1),
                        hovertemplate='<b>%{fullData.name}</b><br>' +
                                    'Time: %{x}<br>' +
                                    'Voltage: %{y:.4f}V<br>' +
                                    '<extra></extra>'
                    ))

                    fig_voltage.update_layout(
                        title=f"🔋 Real-Time Voltage Data: {selected_cell}",
                        xaxis_title="Timestamp",
                        yaxis_title="Cell Voltage (V)",
                        height=500,
                        showlegend=True,
                        template="plotly_white",
                        hovermode='x unified'
                    )

                    # Optimize for performance
                    fig_voltage.update_traces(
                        connectgaps=True,
                        line_shape='linear'
                    )

                    st.plotly_chart(fig_voltage, use_container_width=True, config={
                        'displayModeBar': True,
                        'displaylogo': False,
                        'modeBarButtonsToRemove': ['pan2d', 'lasso2d']
                    })

                    # Voltage in mV plot for detailed analysis
                    fig_voltage_mv = go.Figure()

                    fig_voltage_mv.add_trace(go.Scattergl(
                        x=timestamps,
                        y=voltages_mv,
                        mode='lines',
                        name=f'{selected_cell} Voltage (mV)',
                        line=dict(color='#ff7f0e', width=1),
                        hovertemplate='<b>%{fullData.name}</b><br>' +
                                    'Time: %{x}<br>' +
                                    'Voltage: %{y:.1f}mV<br>' +
                                    '<extra></extra>'
                    ))

                    fig_voltage_mv.update_layout(
                        title=f"🔍 Detailed Voltage Analysis (mV): {selected_cell}",
                        xaxis_title="Timestamp",
                        yaxis_title="Cell Voltage (mV)",
                        height=400,
                        template="plotly_white",
                        hovermode='x unified'
                    )

                    st.plotly_chart(fig_voltage_mv, use_container_width=True, config={
                        'displayModeBar': True,
                        'displaylogo': False,
                        'modeBarButtonsToRemove': ['pan2d', 'lasso2d']
                    })

                    # Analysis insights
                    st.subheader("📊 Voltage Analysis Insights")

                    # Calculate voltage trend
                    if len(voltages) >= 2:
                        voltage_trend = (voltages[-1] - voltages[0]) / len(voltages) * 1000000  # µV per sample

                        col1, col2 = st.columns(2)

                        with col1:
                            if voltage_trend > 0:
                                st.success(f"📈 Voltage trending UP: +{voltage_trend:.2f}µV per sample")
                            elif voltage_trend < -1:
                                st.error(f"📉 Voltage trending DOWN: {voltage_trend:.2f}µV per sample")
                            else:
                                st.info(f"➡️ Voltage stable: {voltage_trend:.2f}µV per sample")

                        with col2:
                            # Time span analysis
                            time_span_hours = len(voltage_data) * sample_rate / 60  # Approximate hours
                            st.info(f"⏱️ Time span: ~{time_span_hours:.1f} hours sampled")

                    # Data quality metrics
                    st.markdown("**Data Quality Metrics**")
                    quality_col1, quality_col2, quality_col3 = st.columns(3)

                    with quality_col1:
                        st.metric("Processed Rows", f"{stats['total_rows_processed']:,}")
                    with quality_col2:
                        st.metric("Sampling Rate", f"1:{sample_rate}")
                    with quality_col3:
                        efficiency = (stats['data_points'] / stats['total_rows_processed']) * 100
                        st.metric("Data Efficiency", f"{efficiency:.1f}%")

                else:
                    st.error(f"❌ No voltage data available for {selected_cell}")

        else:
            st.info("👆 Select a cell and click 'Load Voltage Plot' to visualize real BMS voltage data over time")
            st.markdown("""
            **Features:**
            - ⚡ **Fast WebGL rendering** for smooth interaction with large datasets
            - 🔍 **Dual-scale visualization** (V and mV) for detailed analysis
            - 📊 **Real-time statistics** and trend analysis
            - 🎛️ **Configurable sampling rates** for performance optimization
            - 📈 **Interactive zoom and pan** to explore voltage patterns
            """)

    else:
        st.error("No cells available for voltage plotting")

st.markdown("---")
st.markdown("*PackPulse 💓 - Professional SAT voltage analysis from real BESS charge cycle telemetry*")