#!/usr/bin/env python3
"""
BESS Cell Health Analyzer - Advanced battery pack and cell analysis

Features:
- Cell voltage and temperature analysis across 5 packs (52 cells each)
- Pack health degradation tracking over time
- Cell imbalance detection and ranking
- Thermal hotspot identification
- Capacity fade analysis
- Neighboring cell correlation analysis
- Statistical anomaly detection

Data Structure:
- 5 packs per BESS system (bms1_p1 to bms1_p5)
- 52 cells per pack (v1-v52 for voltage, t1-t52 for temperature)
- Minute-resolution data from Oct 2023 onwards
"""

from __future__ import annotations
import os
import hashlib
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# Simplified linear regression without scipy
def simple_linear_regression(x, y):
    """Simple linear regression without scipy dependency"""
    if len(x) != len(y) or len(x) < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Calculate slope and intercept
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    if abs(denominator) < 1e-10:
        return 0.0, y_mean, 0.0, 0.0, 0.0

    slope = numerator / denominator
    intercept = y_mean - slope * x_mean

    return slope, intercept, 0.0, 0.0, 0.0

# ---------------- Configuration ----------------
TIMEZONE = "Europe/Berlin"
CACHE_DIR = Path("backend/.meter_cache")

# Cell specifications (typical Li-ion)
NOMINAL_CELL_VOLTAGE = 3.7  # V
MIN_CELL_VOLTAGE = 3.0      # V
MAX_CELL_VOLTAGE = 4.2      # V
MAX_CELL_TEMP = 50          # °C
CRITICAL_TEMP = 60          # °C

# Analysis thresholds
IMBALANCE_THRESHOLD = 0.1   # V - significant cell imbalance
DEGRADATION_THRESHOLD = 0.05 # V/month - significant degradation
HOTSPOT_THRESHOLD = 5       # °C above pack average

@dataclass
class CellMetrics:
    """Comprehensive cell health metrics"""
    cell_id: str
    pack_id: int
    cell_num: int

    # Voltage metrics
    voltage_mean: float
    voltage_std: float
    voltage_min: float
    voltage_max: float
    voltage_range: float

    # Temperature metrics
    temp_mean: float
    temp_std: float
    temp_max: float
    temp_hotspot_events: int

    # Health indicators
    degradation_rate: float  # V/month
    imbalance_score: float   # vs pack average
    anomaly_score: float     # statistical anomaly
    neighbor_correlation: float  # correlation with adjacent cells

    # Time series info
    data_points: int
    start_date: datetime
    end_date: datetime
    data_quality: float  # percentage of non-null values

@dataclass
class PackHealthSummary:
    """Pack-level health analysis"""
    pack_id: int
    bess_system: str

    # Overall health
    pack_soh: float         # State of Health (0-100%)
    average_voltage: float
    voltage_imbalance: float # max - min cell voltage
    avg_temperature: float
    max_temperature: float

    # Degradation analysis
    degradation_rate: float # Average cell degradation
    worst_cell: str
    best_cell: str

    # Statistics
    healthy_cells: int      # cells within normal parameters
    warning_cells: int      # cells showing degradation signs
    critical_cells: int     # cells requiring attention

    # Temporal analysis
    discharge_cycles: int   # estimated from voltage patterns
    usage_pattern: str      # "heavy", "moderate", "light"

class CellAnalyzer:
    """Advanced BESS cell analysis engine"""

    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.executor = ThreadPoolExecutor(max_workers=4)

    def _parse_roots(self) -> List[Path]:
        """Parse BESS data roots"""
        raw = os.environ.get("METER_ROOTS", "").strip()
        if raw:
            roots = [Path(p.strip()) for p in raw.split(",") if p.strip()]
        else:
            roots = [Path("data/BESS"), Path("../data/BESS")]
        return [r for r in roots if r.exists()]

    def _discover_bess_systems(self) -> Dict[str, Path]:
        """Discover available BESS systems"""
        roots = self._parse_roots()
        systems = {}

        for root in roots:
            for system_dir in root.iterdir():
                if system_dir.is_dir() and "ZHPESS" in system_dir.name:
                    systems[system_dir.name] = system_dir

        return systems

    def _load_cell_data(self, system_path: Path, pack: int, cell: int,
                       metric: str, start_dt: Optional[datetime] = None,
                       end_dt: Optional[datetime] = None) -> pd.Series:
        """Load individual cell data (voltage or temperature)"""

        # Check for CSV file
        csv_file = system_path / f"bms1_p{pack}_{metric}{cell}.csv"
        if not csv_file.exists():
            return pd.Series(dtype='float32', name=f"p{pack}_{metric}{cell}")

        # Try cached parquet first for speed
        sig = hashlib.md5(f"{csv_file.resolve()}::{csv_file.stat().st_size}::{csv_file.stat().st_mtime}".encode()).hexdigest()

        # Choose appropriate LOD based on date range
        if start_dt and end_dt:
            span = end_dt - start_dt
            rule = "5min" if span <= timedelta(days=2) else "15min" if span <= timedelta(days=14) else "1h"
        else:
            rule = "1h"  # Default for analysis

        parquet_file = self.cache_dir / f"{sig}__{rule}.parquet"

        if parquet_file.exists():
            try:
                df = pd.read_parquet(parquet_file)
                series = df['value'].astype('float32')
                # Restore datetime index
                if not isinstance(series.index, pd.DatetimeIndex):
                    series.index = pd.to_datetime(series.index)
                series = series.sort_index()
            except Exception:
                # Fall back to CSV if parquet fails
                series = None
        else:
            series = None

        # Load from CSV if no parquet cache or parquet failed
        if series is None or len(series) == 0:
            try:
                df = pd.read_csv(csv_file)
                df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
                series = df.set_index(df.columns[0]).iloc[:, 0].astype('float32')
                series = series.sort_index()
            except Exception:
                series = pd.Series(dtype='float32')

        # Filter by date range if specified
        if start_dt:
            series = series[series.index >= start_dt]
        if end_dt:
            series = series[series.index <= end_dt]

        series.name = f"p{pack}_{metric}{cell}"
        return series

    def _load_pack_data(self, system_path: Path, pack: int, metric: str,
                       start_dt: Optional[datetime] = None,
                       end_dt: Optional[datetime] = None) -> pd.DataFrame:
        """Load all cells for a pack (voltage or temperature)"""

        print(f"Loading pack {pack} {metric} data...")

        # Load all 52 cells in parallel
        def load_cell(cell_num):
            return self._load_cell_data(system_path, pack, cell_num, metric, start_dt, end_dt)

        # Use ThreadPoolExecutor for parallel loading
        futures = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            for cell in range(1, 53):  # cells 1-52
                future = executor.submit(load_cell, cell)
                futures.append((cell, future))

        # Collect results
        cell_data = {}
        for cell, future in futures:
            try:
                series = future.result(timeout=30)
                if not series.empty:
                    cell_data[f"cell_{cell:02d}"] = series
            except Exception as e:
                print(f"Warning: Failed to load pack {pack} cell {cell} {metric}: {e}")
                continue

        if not cell_data:
            return pd.DataFrame()

        # Combine into DataFrame
        df = pd.DataFrame(cell_data)
        df = df.dropna(how='all')  # Remove rows where all cells are NaN

        print(f"Loaded pack {pack} {metric}: {len(df)} timestamps, {len(cell_data)} cells")
        return df

    def _calculate_cell_metrics(self, voltage_series: pd.Series, temp_series: pd.Series,
                               pack_avg_voltage: float) -> CellMetrics:
        """Calculate comprehensive metrics for a single cell"""

        if voltage_series.empty:
            # Return default metrics for missing data
            return CellMetrics(
                cell_id=voltage_series.name or "unknown",
                pack_id=0, cell_num=0,
                voltage_mean=0, voltage_std=0, voltage_min=0, voltage_max=0, voltage_range=0,
                temp_mean=0, temp_std=0, temp_max=0, temp_hotspot_events=0,
                degradation_rate=0, imbalance_score=0, anomaly_score=0, neighbor_correlation=0,
                data_points=0, start_date=datetime.now(), end_date=datetime.now(), data_quality=0
            )

        # Parse cell info from series name
        cell_name = voltage_series.name
        pack_id = int(cell_name.split('_')[0][1:]) if 'p' in cell_name else 0
        cell_num = int(cell_name.split('_')[1][1:]) if len(cell_name.split('_')) > 1 else 0

        # Voltage analysis
        v_clean = voltage_series.dropna()
        voltage_mean = float(v_clean.mean()) if len(v_clean) > 0 else 0
        voltage_std = float(v_clean.std()) if len(v_clean) > 1 else 0
        voltage_min = float(v_clean.min()) if len(v_clean) > 0 else 0
        voltage_max = float(v_clean.max()) if len(v_clean) > 0 else 0
        voltage_range = voltage_max - voltage_min

        # Temperature analysis
        if not temp_series.empty:
            t_clean = temp_series.dropna()
            temp_mean = float(t_clean.mean()) if len(t_clean) > 0 else 25
            temp_std = float(t_clean.std()) if len(t_clean) > 1 else 0
            temp_max = float(t_clean.max()) if len(t_clean) > 0 else 25
            temp_hotspot_events = int((t_clean > temp_mean + HOTSPOT_THRESHOLD).sum()) if len(t_clean) > 0 else 0
        else:
            temp_mean = temp_std = temp_max = 25
            temp_hotspot_events = 0

        # Degradation analysis (voltage trend over time)
        degradation_rate = 0
        if len(v_clean) > 100:  # Need sufficient data
            try:
                # Calculate monthly trend
                time_months = (v_clean.index - v_clean.index[0]).total_seconds() / (30 * 24 * 3600)
                slope, _, _, _, _ = simple_linear_regression(time_months, v_clean.values)
                degradation_rate = float(slope) # V/month
            except Exception:
                pass

        # Imbalance score (vs pack average)
        imbalance_score = abs(voltage_mean - pack_avg_voltage) if pack_avg_voltage > 0 else 0

        # Anomaly score using simple z-score calculation
        try:
            if len(v_clean) > 2:
                v_mean = v_clean.mean()
                v_std = v_clean.std()
                if v_std > 0:
                    z_scores = np.abs((v_clean - v_mean) / v_std)
                    anomaly_score = float(z_scores.mean())
                else:
                    anomaly_score = 0
            else:
                anomaly_score = 0
        except Exception:
            anomaly_score = 0

        # Data quality
        data_quality = len(v_clean) / len(voltage_series) * 100 if len(voltage_series) > 0 else 0

        return CellMetrics(
            cell_id=cell_name,
            pack_id=pack_id,
            cell_num=cell_num,
            voltage_mean=voltage_mean,
            voltage_std=voltage_std,
            voltage_min=voltage_min,
            voltage_max=voltage_max,
            voltage_range=voltage_range,
            temp_mean=temp_mean,
            temp_std=temp_std,
            temp_max=temp_max,
            temp_hotspot_events=temp_hotspot_events,
            degradation_rate=degradation_rate,
            imbalance_score=imbalance_score,
            anomaly_score=anomaly_score,
            neighbor_correlation=0,  # Will calculate separately
            data_points=len(v_clean),
            start_date=v_clean.index[0] if len(v_clean) > 0 else datetime.now(),
            end_date=v_clean.index[-1] if len(v_clean) > 0 else datetime.now(),
            data_quality=data_quality
        )

    def _estimate_cycles(self, voltage_df: pd.DataFrame) -> int:
        """Estimate discharge cycles from voltage patterns"""
        try:
            # Take pack average voltage
            pack_voltage = voltage_df.mean(axis=1).dropna()
            if len(pack_voltage) < 100:
                return 0

            # Simple cycle counting: count voltage drops below threshold
            # A cycle is roughly: charge (>3.8V) -> discharge (<3.3V) -> charge
            threshold_low = 3.3
            threshold_high = 3.8

            cycles = 0
            in_cycle = False

            for voltage in pack_voltage:
                if not in_cycle and voltage < threshold_low:
                    in_cycle = True
                elif in_cycle and voltage > threshold_high:
                    cycles += 1
                    in_cycle = False

            return cycles

        except Exception:
            return 0

    def _calculate_professional_soh(self, cell_metrics: List[CellMetrics],
                                   voltage_imbalance: float, cycles: int, timespan_days: float) -> float:
        """Calculate professional SOH using industry-standard battery degradation models"""

        if not cell_metrics or timespan_days <= 0:
            return 95.0  # Default starting point

        # Professional battery degradation factors
        age_years = timespan_days / 365.25

        # 1. Capacity fade from aging (calendar aging)
        # Typical Li-ion: 2-3% per year base degradation
        calendar_fade = min(age_years * 2.5, 15.0)  # Cap at 15% for calendar aging

        # 2. Cycle degradation
        if cycles > 0:
            cycle_rate = cycles / age_years if age_years > 0 else cycles
            # Industry standard: 0.02-0.05% per cycle depending on DOD
            cycle_fade = min(cycles * 0.035, 20.0)  # 3.5% per 100 cycles, cap at 20%
        else:
            cycle_fade = 0.0

        # 3. Voltage imbalance penalty (professional thresholds)
        # 20mV = concerning, 50mV = problematic, 100mV+ = critical
        if voltage_imbalance > 0.1:  # >100mV
            imbalance_penalty = 10.0
        elif voltage_imbalance > 0.05:  # 50-100mV
            imbalance_penalty = 5.0
        elif voltage_imbalance > 0.02:  # 20-50mV
            imbalance_penalty = 2.0
        else:
            imbalance_penalty = 0.0

        # 4. Cell degradation variance penalty
        if len(cell_metrics) > 1:
            degradation_rates = [abs(m.degradation_rate) for m in cell_metrics]
            degradation_std = np.std(degradation_rates)

            # High variance in degradation = pack issues
            variance_penalty = min(degradation_std * 100, 8.0)
        else:
            variance_penalty = 0.0

        # 5. Temperature stress (if available)
        temp_penalty = 0.0
        if cell_metrics:
            avg_max_temp = np.mean([m.temp_max for m in cell_metrics])
            if avg_max_temp > 40:  # >40°C
                temp_penalty = (avg_max_temp - 40) * 0.5  # 0.5% per degree above 40°C
                temp_penalty = min(temp_penalty, 10.0)

        # Calculate final SOH (start from 100%, deduct degradation)
        base_soh = 100.0
        total_degradation = calendar_fade + cycle_fade + imbalance_penalty + variance_penalty + temp_penalty

        # Professional SOH calculation
        professional_soh = max(base_soh - total_degradation, 65.0)  # Floor at 65% (EOL)

        # Apply realistic aging curve (non-linear)
        if age_years > 1.0:
            # Accelerated degradation after first year
            aging_acceleration = min((age_years - 1.0) * 1.5, 5.0)
            professional_soh -= aging_acceleration

        return max(professional_soh, 65.0)  # 65% = End of Life threshold

    def _classify_usage_pattern(self, voltage_df: pd.DataFrame, cycles: int, timespan_days: float) -> str:
        """Classify usage pattern based on cycling behavior"""
        if timespan_days <= 0:
            return "unknown"

        cycles_per_day = cycles / timespan_days

        if cycles_per_day > 2:
            return "heavy"
        elif cycles_per_day > 0.5:
            return "moderate"
        elif cycles_per_day > 0.1:
            return "light"
        else:
            return "minimal"

    def analyze_pack_health(self, system_name: str, pack: int,
                          start_dt: Optional[datetime] = None,
                          end_dt: Optional[datetime] = None) -> Tuple[PackHealthSummary, List[CellMetrics]]:
        """Comprehensive pack health analysis"""

        systems = self._discover_bess_systems()
        if system_name not in systems:
            raise ValueError(f"BESS system {system_name} not found")

        system_path = systems[system_name]

        # Load voltage and temperature data for all cells
        print(f"Analyzing {system_name} pack {pack}...")
        voltage_df = self._load_pack_data(system_path, pack, "v", start_dt, end_dt)
        temp_df = self._load_pack_data(system_path, pack, "t", start_dt, end_dt)

        if voltage_df.empty:
            raise ValueError(f"No voltage data found for pack {pack}")

        # Calculate pack-level statistics
        pack_avg_voltage = float(voltage_df.mean().mean())
        pack_voltage_std = float(voltage_df.mean().std())
        voltage_imbalance = float(voltage_df.mean(axis=0).max() - voltage_df.mean(axis=0).min())

        # Temperature analysis
        if not temp_df.empty:
            avg_temperature = float(temp_df.mean().mean())
            max_temperature = float(temp_df.max().max())
        else:
            avg_temperature = max_temperature = 25.0

        # Analyze each cell
        cell_metrics = []
        for col in voltage_df.columns:
            cell_num = int(col.split('_')[1])
            voltage_series = voltage_df[col].dropna()
            voltage_series.name = f"p{pack}_v{cell_num}"

            # Find corresponding temperature data
            temp_col = f"cell_{cell_num:02d}"
            temp_series = temp_df[temp_col].dropna() if temp_col in temp_df.columns else pd.Series(dtype='float32')

            metrics = self._calculate_cell_metrics(voltage_series, temp_series, pack_avg_voltage)
            cell_metrics.append(metrics)

        # Calculate neighbor correlations
        self._calculate_neighbor_correlations(cell_metrics, voltage_df)

        # Pack health summary
        healthy_cells = sum(1 for m in cell_metrics if m.imbalance_score < IMBALANCE_THRESHOLD and abs(m.degradation_rate) < DEGRADATION_THRESHOLD)
        warning_cells = sum(1 for m in cell_metrics if IMBALANCE_THRESHOLD <= m.imbalance_score < 2*IMBALANCE_THRESHOLD or DEGRADATION_THRESHOLD <= abs(m.degradation_rate) < 2*DEGRADATION_THRESHOLD)
        critical_cells = len(cell_metrics) - healthy_cells - warning_cells

        # Estimate cycles and usage
        cycles = self._estimate_cycles(voltage_df)
        timespan_days = (voltage_df.index[-1] - voltage_df.index[0]).total_seconds() / 86400 if len(voltage_df) > 0 else 1
        usage_pattern = self._classify_usage_pattern(voltage_df, cycles, timespan_days)

        # Professional SOH calculation using battery industry standards
        soh = self._calculate_professional_soh(cell_metrics, voltage_imbalance, cycles, timespan_days)

        # Find best/worst cells
        if cell_metrics:
            worst_cell = min(cell_metrics, key=lambda m: m.voltage_mean - abs(m.degradation_rate)).cell_id
            best_cell = max(cell_metrics, key=lambda m: m.voltage_mean - abs(m.degradation_rate)).cell_id
        else:
            worst_cell = best_cell = "unknown"

        pack_summary = PackHealthSummary(
            pack_id=pack,
            bess_system=system_name,
            pack_soh=float(soh),
            average_voltage=pack_avg_voltage,
            voltage_imbalance=voltage_imbalance,
            avg_temperature=avg_temperature,
            max_temperature=max_temperature,
            degradation_rate=float(np.mean([m.degradation_rate for m in cell_metrics])) if cell_metrics else 0,
            worst_cell=worst_cell,
            best_cell=best_cell,
            healthy_cells=healthy_cells,
            warning_cells=warning_cells,
            critical_cells=critical_cells,
            discharge_cycles=cycles,
            usage_pattern=usage_pattern
        )

        return pack_summary, cell_metrics

    def _calculate_neighbor_correlations(self, cell_metrics: List[CellMetrics], voltage_df: pd.DataFrame):
        """Calculate correlation with neighboring cells (assumes cells are physically adjacent by number)"""

        # Create mapping from cell number to voltage series
        cell_voltages = {}
        for col in voltage_df.columns:
            cell_num = int(col.split('_')[1])
            cell_voltages[cell_num] = voltage_df[col].dropna()

        # Calculate correlations for each cell
        for i, metrics in enumerate(cell_metrics):
            cell_num = metrics.cell_num

            if cell_num in cell_voltages:
                cell_series = cell_voltages[cell_num]

                # Find neighboring cells (adjacent by number)
                neighbors = []
                for neighbor_num in [cell_num - 1, cell_num + 1]:
                    if 1 <= neighbor_num <= 52 and neighbor_num in cell_voltages:
                        neighbors.append(cell_voltages[neighbor_num])

                # Calculate average correlation with neighbors
                if neighbors and len(cell_series) > 10:
                    try:
                        correlations = []
                        for neighbor in neighbors:
                            # Align series by index
                            aligned_self, aligned_neighbor = cell_series.align(neighbor, join='inner')
                            if len(aligned_self) > 10:
                                corr = aligned_self.corr(aligned_neighbor)
                                if not pd.isna(corr):
                                    correlations.append(corr)

                        metrics.neighbor_correlation = float(np.mean(correlations)) if correlations else 0
                    except Exception:
                        metrics.neighbor_correlation = 0

    def compare_packs_degradation(self, system_name: str,
                                start_dt: Optional[datetime] = None,
                                end_dt: Optional[datetime] = None) -> Dict[int, PackHealthSummary]:
        """Compare degradation across all 5 packs"""

        pack_summaries = {}

        for pack_id in range(1, 6):  # packs 1-5
            try:
                print(f"\nAnalyzing pack {pack_id}...")
                summary, _ = self.analyze_pack_health(system_name, pack_id, start_dt, end_dt)
                pack_summaries[pack_id] = summary
            except Exception as e:
                print(f"Warning: Could not analyze pack {pack_id}: {e}")
                continue

        return pack_summaries

    def detect_anomalous_cells(self, system_name: str, pack: int,
                             start_dt: Optional[datetime] = None,
                             end_dt: Optional[datetime] = None) -> List[CellMetrics]:
        """Detect cells with anomalous behavior using multiple criteria"""

        _, cell_metrics = self.analyze_pack_health(system_name, pack, start_dt, end_dt)

        # Define anomaly criteria
        anomalous_cells = []

        for cell in cell_metrics:
            is_anomalous = False
            reasons = []

            # Check imbalance
            if cell.imbalance_score > IMBALANCE_THRESHOLD:
                is_anomalous = True
                reasons.append(f"voltage_imbalance({cell.imbalance_score:.3f}V)")

            # Check degradation
            if abs(cell.degradation_rate) > DEGRADATION_THRESHOLD:
                is_anomalous = True
                reasons.append(f"degradation({cell.degradation_rate:.4f}V/month)")

            # Check temperature
            if cell.temp_max > MAX_CELL_TEMP:
                is_anomalous = True
                reasons.append(f"overheating({cell.temp_max:.1f}°C)")

            # Check statistical anomaly
            if cell.anomaly_score > 2.0:  # 2 sigma
                is_anomalous = True
                reasons.append(f"statistical_anomaly({cell.anomaly_score:.2f})")

            # Check correlation (low correlation might indicate problems)
            if cell.neighbor_correlation < 0.7 and cell.data_points > 1000:
                is_anomalous = True
                reasons.append(f"low_correlation({cell.neighbor_correlation:.3f})")

            if is_anomalous:
                # Add reasons to cell object (modify dataclass if needed)
                cell.anomaly_reasons = reasons
                anomalous_cells.append(cell)

        # Sort by severity (multiple criteria)
        anomalous_cells.sort(key=lambda c: (
            abs(c.degradation_rate) * 1000 +  # Degradation weight
            c.imbalance_score * 10 +           # Imbalance weight
            c.anomaly_score +                  # Statistical weight
            (1 - c.neighbor_correlation)       # Correlation weight
        ), reverse=True)

        return anomalous_cells

# ---------------- Usage Example ----------------
async def main():
    """Example usage of the cell analyzer"""

    analyzer = CellAnalyzer()

    # Discover available systems
    systems = analyzer._discover_bess_systems()
    print("Available BESS systems:", list(systems.keys()))

    if not systems:
        print("No BESS systems found. Check METER_ROOTS environment variable.")
        return

    system_name = list(systems.keys())[0]  # Use first system

    # Compare all packs
    print(f"\n=== Pack Comparison for {system_name} ===")
    pack_comparison = analyzer.compare_packs_degradation(system_name)

    for pack_id, summary in pack_comparison.items():
        print(f"\nPack {pack_id}:")
        print(f"  SOH: {summary.pack_soh:.1f}%")
        print(f"  Average Voltage: {summary.average_voltage:.3f}V")
        print(f"  Imbalance: {summary.voltage_imbalance:.3f}V")
        print(f"  Degradation: {summary.degradation_rate:.4f}V/month")
        print(f"  Usage: {summary.usage_pattern} ({summary.discharge_cycles} cycles)")
        print(f"  Health: {summary.healthy_cells} healthy, {summary.warning_cells} warning, {summary.critical_cells} critical")

    # Detailed analysis of most problematic pack
    worst_pack = min(pack_comparison.items(), key=lambda x: x[1].pack_soh)
    print(f"\n=== Detailed Analysis of Pack {worst_pack[0]} (Worst SOH: {worst_pack[1].pack_soh:.1f}%) ===")

    anomalous_cells = analyzer.detect_anomalous_cells(system_name, worst_pack[0])

    print(f"Found {len(anomalous_cells)} anomalous cells:")
    for i, cell in enumerate(anomalous_cells[:10], 1):  # Top 10
        print(f"{i:2d}. Cell {cell.cell_num:2d}: {cell.voltage_mean:.3f}V avg, "
              f"degradation: {cell.degradation_rate:.4f}V/month, "
              f"imbalance: {cell.imbalance_score:.3f}V")
        if hasattr(cell, 'anomaly_reasons'):
            print(f"     Issues: {', '.join(cell.anomaly_reasons)}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())