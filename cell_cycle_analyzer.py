#!/usr/bin/env python3
"""
Cell-Level Charging Cycle Analyzer

Advanced analysis of individual cell charging patterns:
- Individual cell cycle detection and comparison
- 3D visualization (time × cell × voltage) for pack analysis
- Cycle chunking and comparison across time periods
- Charging curve aggregation and degradation tracking
- Cell-to-cell cycle variation analysis
"""

from __future__ import annotations
import os
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

@dataclass
class CellCycle:
    """Individual cell charging cycle"""
    cell_id: str
    pack_id: int
    cell_num: int
    cycle_id: int
    start_time: datetime
    end_time: datetime
    cycle_type: str  # 'charge', 'discharge'

    # Voltage metrics
    start_voltage: float
    end_voltage: float
    peak_voltage: float
    min_voltage: float
    voltage_delta: float
    voltage_efficiency: float  # end/start for charge cycles

    # Timing metrics
    duration_minutes: float
    charge_rate: float  # V/hour

    # Health indicators
    capacity_estimate: float
    internal_resistance: float
    degradation_score: float

@dataclass
class PackCycleComparison:
    """Pack-level cycle comparison"""
    pack_id: int
    cycle_id: int
    cycle_type: str
    start_time: datetime
    end_time: datetime

    # Cell variation metrics
    voltage_spread: float  # max - min cell voltage
    timing_sync: float    # how synchronized are cells
    efficiency_variance: float
    degradation_spread: float

    # Pack-level metrics
    total_capacity: float
    pack_efficiency: float
    imbalance_score: float

    # Individual cell cycles
    cell_cycles: List[CellCycle]

class CellCycleAnalyzer:
    """Advanced cell-level cycle analysis"""

    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or Path("backend/.meter_cache")

        # Critical thresholds - much more sensitive for stability
        self.voltage_threshold = 0.01   # V - even 10mV can be critical
        self.critical_imbalance = 0.02  # V - 20mV imbalance is concerning
        self.critical_degradation = 0.005  # V/cycle - very low threshold
        self.min_cycle_duration = 5     # minutes - shorter cycles

        # Neighbor analysis parameters
        self.neighbor_radius = 2        # cells to analyze around each cell
        self.thermal_influence_radius = 3  # thermal influence range

    def detect_cell_cycles(self, voltage_series: pd.Series,
                          cell_id: str, pack_id: int, cell_num: int) -> List[CellCycle]:
        """Detect charging/discharging cycles for a single cell"""

        cycles = []
        if len(voltage_series) < 100:  # Need minimum data
            return cycles

        # Smooth data to reduce noise
        voltage_smooth = voltage_series.rolling(window=5, center=True).mean().fillna(voltage_series)

        # Find voltage direction changes (charge/discharge transitions)
        voltage_diff = voltage_smooth.diff()

        # Use rolling statistics to identify significant changes
        rolling_std = voltage_diff.rolling(window=20).std()
        threshold = rolling_std.median() * 2

        # Find cycle boundaries
        significant_changes = np.abs(voltage_diff) > threshold
        change_points = voltage_series.index[significant_changes]

        if len(change_points) < 2:
            return cycles

        # Analyze segments between change points
        cycle_id = 0

        for i in range(len(change_points) - 1):
            start_idx = change_points[i]
            end_idx = change_points[i + 1]

            # Get segment data
            segment = voltage_smooth[start_idx:end_idx]
            if len(segment) < 10:  # Skip short segments
                continue

            # Determine cycle type and metrics
            start_v = segment.iloc[0]
            end_v = segment.iloc[-1]
            peak_v = segment.max()
            min_v = segment.min()
            voltage_delta = end_v - start_v

            # Duration check
            duration = (end_idx - start_idx).total_seconds() / 60  # minutes
            if duration < self.min_cycle_duration:
                continue

            # Classify cycle type
            if voltage_delta > self.voltage_threshold:
                cycle_type = 'charge'
                efficiency = end_v / start_v if start_v > 0 else 1.0
            elif voltage_delta < -self.voltage_threshold:
                cycle_type = 'discharge'
                efficiency = start_v / end_v if end_v > 0 else 1.0
            else:
                continue  # Skip rest periods

            # Calculate health metrics
            charge_rate = voltage_delta / (duration / 60)  # V/hour
            capacity_est = abs(voltage_delta) * 10  # Simple estimate
            internal_resistance = self._estimate_resistance(segment)
            degradation = self._calculate_degradation_score(segment, cycle_type)

            cycle = CellCycle(
                cell_id=cell_id,
                pack_id=pack_id,
                cell_num=cell_num,
                cycle_id=cycle_id,
                start_time=start_idx,
                end_time=end_idx,
                cycle_type=cycle_type,
                start_voltage=start_v,
                end_voltage=end_v,
                peak_voltage=peak_v,
                min_voltage=min_v,
                voltage_delta=voltage_delta,
                voltage_efficiency=efficiency,
                duration_minutes=duration,
                charge_rate=charge_rate,
                capacity_estimate=capacity_est,
                internal_resistance=internal_resistance,
                degradation_score=degradation
            )

            cycles.append(cycle)
            cycle_id += 1

        return cycles

    def _estimate_resistance(self, voltage_segment: pd.Series) -> float:
        """Estimate internal resistance from voltage curve"""
        if len(voltage_segment) < 10:
            return 0.0

        # Simple resistance estimation from voltage drop rate
        voltage_diff = voltage_segment.diff().dropna()
        if len(voltage_diff) == 0:
            return 0.0

        resistance_est = abs(voltage_diff.std() * 1000)  # Convert to mOhm
        return min(resistance_est, 100.0)  # Cap at reasonable value

    def _calculate_degradation_score(self, voltage_segment: pd.Series, cycle_type: str) -> float:
        """Calculate degradation score for cycle"""
        if len(voltage_segment) < 10:
            return 0.0

        # Degradation indicators
        voltage_noise = voltage_segment.diff().std()
        voltage_range = voltage_segment.max() - voltage_segment.min()

        # Higher noise and lower range indicate degradation
        noise_score = min(voltage_noise * 100, 1.0)
        range_score = max(0, 1.0 - voltage_range / 0.4)  # Normalize to expected range

        return (noise_score + range_score) / 2

    def analyze_pack_cycles(self, system_name: str, pack_id: int,
                           start_dt: datetime, end_dt: datetime) -> List[PackCycleComparison]:
        """Analyze all cells in a pack for cycle comparison"""

        # Load voltage data for all 52 cells
        pack_cycles = []

        # Load cell data in parallel
        cell_cycle_data = {}

        def load_and_analyze_cell(cell_num):
            try:
                from cell_analyzer import CellAnalyzer
                analyzer = CellAnalyzer()
                systems = analyzer._discover_bess_systems()
                if system_name not in systems:
                    return cell_num, []

                system_path = systems[system_name]
                voltage_series = analyzer._load_cell_data(
                    system_path, pack_id, cell_num, 'v', start_dt, end_dt
                )

                if len(voltage_series) > 100:
                    cell_id = f"p{pack_id}_c{cell_num:02d}"
                    cycles = self.detect_cell_cycles(voltage_series, cell_id, pack_id, cell_num)
                    return cell_num, cycles
                else:
                    return cell_num, []

            except Exception as e:
                print(f"Error analyzing cell {cell_num}: {e}")
                return cell_num, []

        # Process cells in parallel
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(load_and_analyze_cell, cell_num)
                      for cell_num in range(1, 53)]  # cells 1-52

            for future in futures:
                cell_num, cycles = future.result()
                if cycles:
                    cell_cycle_data[cell_num] = cycles

        # Group cycles by time periods for pack-level comparison
        all_cycles = []
        for cell_cycles in cell_cycle_data.values():
            all_cycles.extend(cell_cycles)

        if not all_cycles:
            return []

        # Group cycles by time windows (e.g., daily chunks)
        cycle_groups = self._group_cycles_by_time(all_cycles, window_hours=24)

        # Create pack comparisons
        pack_comparisons = []

        for group_id, group_cycles in cycle_groups.items():
            if len(group_cycles) < 10:  # Need minimum cycles for comparison
                continue

            # Calculate pack-level metrics
            voltages = [c.start_voltage for c in group_cycles]
            efficiencies = [c.voltage_efficiency for c in group_cycles]
            degradations = [c.degradation_score for c in group_cycles]

            comparison = PackCycleComparison(
                pack_id=pack_id,
                cycle_id=group_id,
                cycle_type=group_cycles[0].cycle_type,
                start_time=min(c.start_time for c in group_cycles),
                end_time=max(c.end_time for c in group_cycles),
                voltage_spread=max(voltages) - min(voltages),
                timing_sync=self._calculate_timing_sync(group_cycles),
                efficiency_variance=np.std(efficiencies),
                degradation_spread=max(degradations) - min(degradations),
                total_capacity=sum(c.capacity_estimate for c in group_cycles),
                pack_efficiency=np.mean(efficiencies),
                imbalance_score=self._calculate_imbalance_score(group_cycles),
                cell_cycles=group_cycles
            )

            pack_comparisons.append(comparison)

        return sorted(pack_comparisons, key=lambda x: x.start_time)

    def _group_cycles_by_time(self, cycles: List[CellCycle], window_hours: int = 24) -> Dict[int, List[CellCycle]]:
        """Group cycles into time windows for comparison"""
        if not cycles:
            return {}

        # Sort by start time
        cycles.sort(key=lambda x: x.start_time)

        # Create time windows
        start_time = cycles[0].start_time
        window_delta = timedelta(hours=window_hours)

        groups = {}
        current_group = 0
        current_window_start = start_time

        for cycle in cycles:
            # Check if cycle belongs to current window
            if cycle.start_time >= current_window_start + window_delta:
                current_group += 1
                current_window_start = cycle.start_time

            if current_group not in groups:
                groups[current_group] = []
            groups[current_group].append(cycle)

        return groups

    def _calculate_timing_sync(self, cycles: List[CellCycle]) -> float:
        """Calculate how synchronized cell cycles are"""
        if len(cycles) < 2:
            return 1.0

        start_times = [c.start_time.timestamp() for c in cycles]
        sync_score = 1.0 / (1.0 + np.std(start_times) / 3600)  # Normalize by hours
        return min(sync_score, 1.0)

    def _calculate_imbalance_score(self, cycles: List[CellCycle]) -> float:
        """Calculate pack imbalance from cycle data with critical sensitivity"""
        if len(cycles) < 2:
            return 0.0

        # Use voltage spread as primary imbalance indicator
        voltages = [c.start_voltage for c in cycles]
        voltage_spread = max(voltages) - min(voltages)

        # Much more sensitive scale - 20mV spread = high imbalance
        imbalance = min(voltage_spread / self.critical_imbalance, 1.0)
        return imbalance

    def analyze_neighbor_influence(self, pack_cycles: List[PackCycleComparison]) -> Dict[int, Dict]:
        """Analyze how neighboring cells influence each other"""

        if not pack_cycles:
            return {}

        # Collect all cell data by position
        cell_data = {}
        for pc in pack_cycles:
            for cell_cycle in pc.cell_cycles:
                cell_num = cell_cycle.cell_num
                if cell_num not in cell_data:
                    cell_data[cell_num] = []
                cell_data[cell_num].append({
                    'voltage': cell_cycle.start_voltage,
                    'degradation': cell_cycle.degradation_score,
                    'efficiency': cell_cycle.voltage_efficiency,
                    'time': cell_cycle.start_time
                })

        # Analyze neighbor correlations
        neighbor_analysis = {}

        for cell_num in cell_data:
            if len(cell_data[cell_num]) < 5:  # Need minimum data
                continue

            neighbors = self._get_cell_neighbors(cell_num)
            neighbor_correlations = {}
            thermal_influences = {}

            # Calculate correlations with each neighbor
            for neighbor_num in neighbors:
                if neighbor_num in cell_data and len(cell_data[neighbor_num]) >= 5:
                    correlation = self._calculate_neighbor_correlation(
                        cell_data[cell_num], cell_data[neighbor_num]
                    )
                    distance = self._get_cell_distance(cell_num, neighbor_num)

                    neighbor_correlations[neighbor_num] = {
                        'correlation': correlation,
                        'distance': distance,
                        'influence_score': correlation / max(distance, 1)
                    }

                    # Thermal influence (closer cells have stronger thermal coupling)
                    if distance <= self.thermal_influence_radius:
                        thermal_influences[neighbor_num] = correlation * (
                            1.0 - distance / self.thermal_influence_radius
                        )

            # Identify critical neighbor relationships
            critical_neighbors = []
            for neighbor_num, data in neighbor_correlations.items():
                if data['correlation'] > 0.7 and data['distance'] <= 2:
                    critical_neighbors.append({
                        'cell': neighbor_num,
                        'correlation': data['correlation'],
                        'risk_level': 'high' if data['correlation'] > 0.85 else 'medium'
                    })

            neighbor_analysis[cell_num] = {
                'total_neighbors': len(neighbors),
                'analyzed_neighbors': len(neighbor_correlations),
                'correlations': neighbor_correlations,
                'thermal_influences': thermal_influences,
                'critical_neighbors': critical_neighbors,
                'stability_risk': self._assess_neighbor_stability_risk(neighbor_correlations),
                'isolation_score': 1.0 - (len(critical_neighbors) / max(len(neighbors), 1))
            }

        return neighbor_analysis

    def _get_cell_neighbors(self, cell_num: int) -> List[int]:
        """Get neighboring cells within radius (52-cell pack layout)"""
        neighbors = []

        # Assume 52 cells arranged in a rectangular grid (e.g., 13x4)
        rows, cols = 4, 13
        row = (cell_num - 1) // cols
        col = (cell_num - 1) % cols

        for dr in range(-self.neighbor_radius, self.neighbor_radius + 1):
            for dc in range(-self.neighbor_radius, self.neighbor_radius + 1):
                if dr == 0 and dc == 0:
                    continue

                new_row = row + dr
                new_col = col + dc

                if 0 <= new_row < rows and 0 <= new_col < cols:
                    neighbor_num = new_row * cols + new_col + 1
                    neighbors.append(neighbor_num)

        return neighbors

    def _get_cell_distance(self, cell1: int, cell2: int) -> float:
        """Calculate physical distance between cells"""
        rows, cols = 4, 13

        row1, col1 = (cell1 - 1) // cols, (cell1 - 1) % cols
        row2, col2 = (cell2 - 1) // cols, (cell2 - 1) % cols

        return np.sqrt((row2 - row1)**2 + (col2 - col1)**2)

    def _calculate_neighbor_correlation(self, cell1_data: List[Dict], cell2_data: List[Dict]) -> float:
        """Calculate correlation between neighboring cells"""
        if len(cell1_data) < 3 or len(cell2_data) < 3:
            return 0.0

        # Align data by time
        cell1_voltages = [d['voltage'] for d in cell1_data[-10:]]  # Last 10 points
        cell2_voltages = [d['voltage'] for d in cell2_data[-10:]]

        if len(cell1_voltages) != len(cell2_voltages):
            min_len = min(len(cell1_voltages), len(cell2_voltages))
            cell1_voltages = cell1_voltages[-min_len:]
            cell2_voltages = cell2_voltages[-min_len:]

        if len(cell1_voltages) < 3:
            return 0.0

        # Calculate correlation coefficient
        correlation_matrix = np.corrcoef(cell1_voltages, cell2_voltages)
        return abs(correlation_matrix[0, 1]) if not np.isnan(correlation_matrix[0, 1]) else 0.0

    def _assess_neighbor_stability_risk(self, neighbor_correlations: Dict) -> str:
        """Assess stability risk based on neighbor correlations"""
        if not neighbor_correlations:
            return 'isolated'

        high_correlations = sum(1 for data in neighbor_correlations.values()
                               if data['correlation'] > 0.8)
        total_neighbors = len(neighbor_correlations)

        if high_correlations >= total_neighbors * 0.7:
            return 'critical'  # Too many highly correlated neighbors
        elif high_correlations >= total_neighbors * 0.4:
            return 'warning'
        else:
            return 'stable'

    def detect_critical_cells(self, pack_cycles: List[PackCycleComparison]) -> List[Dict]:
        """Detect cells that are critical for pack stability"""
        if not pack_cycles:
            return []

        # Analyze neighbor influences first
        neighbor_analysis = self.analyze_neighbor_influence(pack_cycles)

        critical_cells = []

        # Collect cell performance data
        cell_performance = {}
        for pc in pack_cycles:
            for cell_cycle in pc.cell_cycles:
                cell_num = cell_cycle.cell_num
                if cell_num not in cell_performance:
                    cell_performance[cell_num] = {
                        'voltages': [],
                        'degradations': [],
                        'efficiencies': []
                    }

                cell_performance[cell_num]['voltages'].append(cell_cycle.start_voltage)
                cell_performance[cell_num]['degradations'].append(cell_cycle.degradation_score)
                cell_performance[cell_num]['efficiencies'].append(cell_cycle.voltage_efficiency)

        # Evaluate each cell for criticality
        for cell_num, perf in cell_performance.items():
            if len(perf['voltages']) < 3:
                continue

            # Calculate critical metrics
            voltage_drift = np.std(perf['voltages'])
            avg_degradation = np.mean(perf['degradations'])
            voltage_trend = self._calculate_voltage_trend(perf['voltages'])

            # Critical conditions (much more sensitive)
            conditions = {
                'voltage_drift': voltage_drift > 0.01,  # 10mV drift
                'high_degradation': avg_degradation > self.critical_degradation,
                'voltage_decline': voltage_trend < -0.001,  # 1mV/cycle decline
                'low_efficiency': np.mean(perf['efficiencies']) < 0.95,
                'neighbor_risk': neighbor_analysis.get(cell_num, {}).get('stability_risk') in ['critical', 'warning'],
                'isolation_risk': neighbor_analysis.get(cell_num, {}).get('isolation_score', 0) > 0.8
            }

            # Count critical conditions
            critical_count = sum(conditions.values())

            if critical_count >= 2:  # 2+ critical conditions = critical cell
                risk_level = 'critical' if critical_count >= 4 else 'high' if critical_count >= 3 else 'medium'

                critical_cells.append({
                    'cell_num': cell_num,
                    'risk_level': risk_level,
                    'critical_conditions': [k for k, v in conditions.items() if v],
                    'voltage_drift': voltage_drift,
                    'avg_degradation': avg_degradation,
                    'voltage_trend': voltage_trend,
                    'neighbor_correlations': len(neighbor_analysis.get(cell_num, {}).get('critical_neighbors', [])),
                    'stability_impact': self._estimate_stability_impact(cell_num, neighbor_analysis)
                })

        # Sort by risk level and impact
        critical_cells.sort(key=lambda x: (
            {'critical': 3, 'high': 2, 'medium': 1}[x['risk_level']],
            x['stability_impact']
        ), reverse=True)

        return critical_cells

    def _calculate_voltage_trend(self, voltages: List[float]) -> float:
        """Calculate voltage trend (slope)"""
        if len(voltages) < 3:
            return 0.0

        x = np.arange(len(voltages))
        y = np.array(voltages)

        # Simple linear regression
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x * x)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        return slope

    def _estimate_stability_impact(self, cell_num: int, neighbor_analysis: Dict) -> float:
        """Estimate how much this cell impacts pack stability"""
        if cell_num not in neighbor_analysis:
            return 0.0

        analysis = neighbor_analysis[cell_num]

        # High correlation with many neighbors = high impact
        critical_neighbors = len(analysis.get('critical_neighbors', []))
        total_correlations = sum(data['correlation'] for data in analysis.get('correlations', {}).values())

        # Impact score based on network effects
        impact = (critical_neighbors * 0.3 + total_correlations * 0.7) / 10
        return min(impact, 1.0)

    def create_3d_pack_visualization(self, pack_cycles: List[PackCycleComparison],
                                   title: str = "Pack Voltage 3D Analysis") -> go.Figure:
        """Create 3D visualization: Time × Cell × Voltage"""

        if not pack_cycles:
            fig = go.Figure()
            fig.add_annotation(text="No cycle data available",
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig

        # Prepare data for 3D plotting
        times, cells, voltages, colors = [], [], [], []

        for pack_cycle in pack_cycles:
            for cell_cycle in pack_cycle.cell_cycles:
                # Use cycle start time as X-axis
                time_val = cell_cycle.start_time.timestamp()
                times.append(time_val)

                # Cell number as Y-axis
                cells.append(cell_cycle.cell_num)

                # Voltage as Z-axis
                voltages.append(cell_cycle.start_voltage)

                # Color by cycle type and degradation
                if cell_cycle.cycle_type == 'charge':
                    colors.append(cell_cycle.degradation_score + 1)  # 1-2 range
                else:
                    colors.append(cell_cycle.degradation_score)      # 0-1 range

        if not times:
            fig = go.Figure()
            fig.add_annotation(text="No valid cycle data",
                             xref="paper", yref="paper", x=0.5, y=0.5)
            return fig

        # Convert timestamps to readable dates
        time_labels = [datetime.fromtimestamp(t).strftime("%Y-%m-%d") for t in times]

        # Create 3D scatter plot
        fig = go.Figure(data=go.Scatter3d(
            x=times,
            y=cells,
            z=voltages,
            mode='markers',
            marker=dict(
                size=4,
                color=colors,
                colorscale='Viridis',
                colorbar=dict(title="Degradation Score"),
                opacity=0.7
            ),
            text=[f"Cell {c}<br>V: {v:.3f}V<br>Time: {t}"
                  for c, v, t in zip(cells, voltages, time_labels)],
            hovertemplate='<b>Cell %{y}</b><br>' +
                         'Voltage: %{z:.3f}V<br>' +
                         'Time: %{text}<br>' +
                         '<extra></extra>'
        ))

        # Update layout for 3D
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="Time",
                yaxis_title="Cell Number (1-52)",
                zaxis_title="Voltage (V)",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=800,
            height=600
        )

        return fig

    def create_cycle_comparison_plot(self, pack_cycles: List[PackCycleComparison]) -> go.Figure:
        """Create cycle comparison visualization with chunking controls"""

        if not pack_cycles:
            return go.Figure()

        # Create subplots for different metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Voltage Spread Over Time',
                'Pack Efficiency by Cycle',
                'Imbalance Score Evolution',
                'Cycle Duration Distribution'
            ],
            specs=[[{"secondary_y": True}, {"secondary_y": True}],
                   [{"secondary_y": True}, {"type": "histogram"}]]
        )

        # Extract metrics
        times = [pc.start_time for pc in pack_cycles]
        voltage_spreads = [pc.voltage_spread for pc in pack_cycles]
        efficiencies = [pc.pack_efficiency for pc in pack_cycles]
        imbalance_scores = [pc.imbalance_score for pc in pack_cycles]

        # Calculate durations
        durations = []
        for pc in pack_cycles:
            avg_duration = np.mean([cc.duration_minutes for cc in pc.cell_cycles])
            durations.append(avg_duration)

        # Plot 1: Voltage Spread
        fig.add_trace(go.Scatter(
            x=times, y=voltage_spreads,
            mode='lines+markers',
            name='Voltage Spread',
            line=dict(color='red')
        ), row=1, col=1)

        # Plot 2: Efficiency
        fig.add_trace(go.Scatter(
            x=times, y=efficiencies,
            mode='lines+markers',
            name='Pack Efficiency',
            line=dict(color='green')
        ), row=1, col=2)

        # Plot 3: Imbalance
        fig.add_trace(go.Scatter(
            x=times, y=imbalance_scores,
            mode='lines+markers',
            name='Imbalance Score',
            line=dict(color='orange')
        ), row=2, col=1)

        # Plot 4: Duration histogram
        fig.add_trace(go.Histogram(
            x=durations,
            name='Cycle Durations',
            nbinsx=20
        ), row=2, col=2)

        # Update layout
        fig.update_layout(
            title="Pack Cycle Analysis Dashboard",
            showlegend=False,
            height=600
        )

        return fig

    def get_cycle_aggregation_stats(self, pack_cycles: List[PackCycleComparison]) -> Dict[str, Any]:
        """Generate aggregated statistics for cycles"""

        if not pack_cycles:
            return {}

        # Collect all individual cell cycles
        all_cell_cycles = []
        for pc in pack_cycles:
            all_cell_cycles.extend(pc.cell_cycles)

        if not all_cell_cycles:
            return {}

        # Group by cell number for per-cell analysis
        cell_groups = {}
        for cycle in all_cell_cycles:
            if cycle.cell_num not in cell_groups:
                cell_groups[cycle.cell_num] = []
            cell_groups[cycle.cell_num].append(cycle)

        # Calculate per-cell statistics
        cell_stats = {}
        for cell_num, cycles in cell_groups.items():
            if len(cycles) > 0:
                voltages = [c.start_voltage for c in cycles]
                degradations = [c.degradation_score for c in cycles]

                cell_stats[f"cell_{cell_num:02d}"] = {
                    "total_cycles": len(cycles),
                    "avg_voltage": np.mean(voltages),
                    "voltage_std": np.std(voltages),
                    "avg_degradation": np.mean(degradations),
                    "max_degradation": max(degradations),
                    "cycle_types": {
                        "charge": len([c for c in cycles if c.cycle_type == 'charge']),
                        "discharge": len([c for c in cycles if c.cycle_type == 'discharge'])
                    }
                }

        # Pack-level aggregation
        pack_stats = {
            "total_cycles": len(pack_cycles),
            "avg_voltage_spread": np.mean([pc.voltage_spread for pc in pack_cycles]),
            "avg_efficiency": np.mean([pc.pack_efficiency for pc in pack_cycles]),
            "avg_imbalance": np.mean([pc.imbalance_score for pc in pack_cycles]),
            "time_span_days": (max(pc.end_time for pc in pack_cycles) -
                              min(pc.start_time for pc in pack_cycles)).days,
            "cells_analyzed": len(cell_stats)
        }

        return {
            "pack_summary": pack_stats,
            "cell_details": cell_stats,
            "analysis_timestamp": datetime.now().isoformat()
        }