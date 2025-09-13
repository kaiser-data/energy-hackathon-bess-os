#!/usr/bin/env python3
"""
Charging Curve Analyzer for Irregular Battery Patterns

Handles non-periodic, irregular charging/discharging cycles by:
- Detecting charge/discharge events regardless of timing
- Creating statistical charging profiles
- Analyzing capacity fade over irregular cycles
- Comparing charging efficiency across different time periods
- Identifying degradation patterns in irregular usage
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
from scipy import stats, signal
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

@dataclass
class ChargingCycleEvent:
    """Single charge/discharge event detection"""
    start_time: datetime
    end_time: datetime
    start_voltage: float
    end_voltage: float
    peak_voltage: float
    min_voltage: float
    duration_hours: float
    voltage_delta: float
    event_type: str  # 'charge', 'discharge', 'rest'
    capacity_estimate: float  # Ah equivalent
    efficiency: float  # charge efficiency %

@dataclass
class ChargingProfile:
    """Statistical charging behavior profile"""
    cell_id: str
    pack_id: int
    cell_num: int

    # Event statistics
    total_cycles: int
    charge_events: int
    discharge_events: int
    rest_periods: int

    # Voltage patterns
    typical_charge_voltage: Tuple[float, float]  # (start, end)
    typical_discharge_voltage: Tuple[float, float]
    peak_voltage_trend: float  # V/month degradation
    min_voltage_trend: float

    # Timing patterns (irregular)
    avg_charge_duration: float  # hours
    avg_discharge_duration: float
    charge_duration_variability: float  # coefficient of variation
    discharge_duration_variability: float

    # Capacity and efficiency
    estimated_capacity_fade: float  # %/month
    charge_efficiency_trend: float  # %/month change
    voltage_recovery_time: float  # hours average

    # Pattern irregularity metrics
    cycle_regularity_score: float  # 0-1, 1=very regular
    usage_intensity_score: float   # cycles per day
    degradation_rate: float        # overall health decline

class ChargingCurveAnalyzer:
    """Analyze irregular charging/discharging patterns"""

    def __init__(self):
        self.cache_dir = Path("backend/.meter_cache")

        # Detection thresholds
        self.CHARGE_THRESHOLD = 0.05  # V minimum rise for charge detection
        self.DISCHARGE_THRESHOLD = 0.05  # V minimum drop for discharge detection
        self.MIN_EVENT_DURATION = 30  # minutes
        self.VOLTAGE_NOISE_FILTER = 0.01  # V noise filtering

    def _load_cell_timeseries(self, system_path: Path, pack: int, cell: int,
                             start_dt: Optional[datetime] = None,
                             end_dt: Optional[datetime] = None) -> pd.Series:
        """Load cell voltage time series data"""
        csv_file = system_path / f"bms1_p{pack}_v{cell}.csv"
        if not csv_file.exists():
            return pd.Series(dtype='float32')

        # Try cached parquet first
        sig = hashlib.md5(f"{csv_file.resolve()}::{csv_file.stat().st_size}::{csv_file.stat().st_mtime}".encode()).hexdigest()
        parquet_file = self.cache_dir / f"{sig}__5min.parquet"  # Use 5min for detailed analysis

        try:
            if parquet_file.exists():
                df = pd.read_parquet(parquet_file)
                series = df['value'].astype('float32')
                if not isinstance(series.index, pd.DatetimeIndex):
                    series.index = pd.to_datetime(series.index)
            else:
                # Fallback to CSV
                df = pd.read_csv(csv_file)
                df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
                series = df.set_index(df.columns[0]).iloc[:, 0].astype('float32')

            series = series.sort_index().dropna()

            # Filter by date range
            if start_dt:
                series = series[series.index >= start_dt]
            if end_dt:
                series = series[series.index <= end_dt]

            # Apply noise filtering
            series = self._smooth_voltage_data(series)

            return series

        except Exception:
            return pd.Series(dtype='float32')

    def _smooth_voltage_data(self, series: pd.Series) -> pd.Series:
        """Apply gentle smoothing to reduce measurement noise"""
        if len(series) < 10:
            return series

        # Use rolling median to preserve charge/discharge edges
        window_size = min(5, len(series) // 10)
        if window_size >= 3:
            smoothed = series.rolling(window=window_size, center=True).median()
            # Fill NaN values at edges
            smoothed = smoothed.fillna(series)
            return smoothed
        return series

    def _detect_charge_discharge_events(self, voltage_series: pd.Series) -> List[ChargingCycleEvent]:
        """Detect irregular charge/discharge events from voltage pattern"""
        if len(voltage_series) < 100:
            return []

        events = []

        # Calculate voltage derivatives to find trends
        voltage_diff = voltage_series.diff()
        time_diff = pd.Series(voltage_series.index).diff().dt.total_seconds() / 3600  # hours
        voltage_rate = voltage_diff / time_diff  # V/hour

        # State machine for event detection
        current_state = 'rest'
        event_start_idx = 0
        event_start_time = voltage_series.index[0]
        event_start_voltage = voltage_series.iloc[0]

        for i in range(1, len(voltage_series)):
            current_voltage = voltage_series.iloc[i]
            current_time = voltage_series.index[i]
            rate = voltage_rate.iloc[i] if not pd.isna(voltage_rate.iloc[i]) else 0

            # State transitions based on voltage rate
            if current_state == 'rest':
                if rate > 0.01:  # Starting to charge (V/hour)
                    current_state = 'charge'
                    event_start_idx = i
                    event_start_time = current_time
                    event_start_voltage = current_voltage
                elif rate < -0.01:  # Starting to discharge
                    current_state = 'discharge'
                    event_start_idx = i
                    event_start_time = current_time
                    event_start_voltage = current_voltage

            elif current_state == 'charge':
                if rate < -0.005:  # Charge stopping or reversing
                    # Create charge event
                    duration = (current_time - event_start_time).total_seconds() / 3600
                    if duration > self.MIN_EVENT_DURATION / 60:  # Minimum duration check
                        voltage_delta = current_voltage - event_start_voltage
                        if voltage_delta > self.CHARGE_THRESHOLD:

                            # Calculate event metrics
                            event_voltages = voltage_series.iloc[event_start_idx:i+1]
                            peak_voltage = event_voltages.max()
                            min_voltage = event_voltages.min()

                            # Rough capacity estimate (simplified)
                            capacity_estimate = voltage_delta * 10  # Ah (very rough)
                            efficiency = min(100, (voltage_delta / 0.8) * 100)  # % (simplified)

                            event = ChargingCycleEvent(
                                start_time=event_start_time,
                                end_time=current_time,
                                start_voltage=event_start_voltage,
                                end_voltage=current_voltage,
                                peak_voltage=peak_voltage,
                                min_voltage=min_voltage,
                                duration_hours=duration,
                                voltage_delta=voltage_delta,
                                event_type='charge',
                                capacity_estimate=capacity_estimate,
                                efficiency=efficiency
                            )
                            events.append(event)

                    current_state = 'discharge' if rate < -0.01 else 'rest'
                    if current_state == 'discharge':
                        event_start_idx = i
                        event_start_time = current_time
                        event_start_voltage = current_voltage

            elif current_state == 'discharge':
                if rate > 0.005:  # Discharge stopping or reversing
                    # Create discharge event
                    duration = (current_time - event_start_time).total_seconds() / 3600
                    if duration > self.MIN_EVENT_DURATION / 60:
                        voltage_delta = event_start_voltage - current_voltage  # Positive for discharge
                        if voltage_delta > self.DISCHARGE_THRESHOLD:

                            event_voltages = voltage_series.iloc[event_start_idx:i+1]
                            peak_voltage = event_voltages.max()
                            min_voltage = event_voltages.min()

                            capacity_estimate = voltage_delta * 10  # Ah (rough)
                            efficiency = 95.0  # Discharge efficiency typically high

                            event = ChargingCycleEvent(
                                start_time=event_start_time,
                                end_time=current_time,
                                start_voltage=event_start_voltage,
                                end_voltage=current_voltage,
                                peak_voltage=peak_voltage,
                                min_voltage=min_voltage,
                                duration_hours=duration,
                                voltage_delta=voltage_delta,
                                event_type='discharge',
                                capacity_estimate=capacity_estimate,
                                efficiency=efficiency
                            )
                            events.append(event)

                    current_state = 'charge' if rate > 0.01 else 'rest'
                    if current_state == 'charge':
                        event_start_idx = i
                        event_start_time = current_time
                        event_start_voltage = current_voltage

        return events

    def _calculate_degradation_trends(self, events: List[ChargingCycleEvent],
                                    total_days: float) -> Tuple[float, float, float]:
        """Calculate degradation trends from irregular events"""
        if len(events) < 5:
            return 0.0, 0.0, 0.0

        # Sort events by time
        events_sorted = sorted(events, key=lambda e: e.start_time)

        # Extract time-based trends
        times = [(e.start_time - events_sorted[0].start_time).total_seconds() / (30 * 24 * 3600) for e in events_sorted]  # months
        peak_voltages = [e.peak_voltage for e in events_sorted]
        min_voltages = [e.min_voltage for e in events_sorted]
        efficiencies = [e.efficiency for e in events_sorted if e.event_type == 'charge']

        # Calculate trends using linear regression
        peak_trend = 0.0
        min_trend = 0.0
        efficiency_trend = 0.0

        try:
            if len(peak_voltages) > 3:
                slope, _, _, _, _ = stats.linregress(times, peak_voltages)
                peak_trend = slope  # V/month

            if len(min_voltages) > 3:
                slope, _, _, _, _ = stats.linregress(times, min_voltages)
                min_trend = slope  # V/month

            if len(efficiencies) > 3:
                efficiency_times = [times[i] for i, e in enumerate(events_sorted) if e.event_type == 'charge']
                slope, _, _, _, _ = stats.linregress(efficiency_times, efficiencies)
                efficiency_trend = slope  # %/month

        except Exception:
            pass

        return peak_trend, min_trend, efficiency_trend

    def _calculate_irregularity_metrics(self, events: List[ChargingCycleEvent]) -> Tuple[float, float]:
        """Calculate how irregular the charging pattern is"""
        if len(events) < 3:
            return 0.0, 0.0

        # Time between events (irregularity)
        charge_events = [e for e in events if e.event_type == 'charge']
        discharge_events = [e for e in events if e.event_type == 'discharge']

        regularity_score = 0.0
        if len(charge_events) > 2:
            # Calculate coefficient of variation for time between charges
            charge_times = [e.start_time for e in sorted(charge_events, key=lambda x: x.start_time)]
            intervals = [(charge_times[i+1] - charge_times[i]).total_seconds() / 3600 for i in range(len(charge_times)-1)]

            if len(intervals) > 1:
                cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 1.0
                regularity_score = max(0, 1 - cv)  # 1 = very regular, 0 = very irregular

        # Usage intensity (events per day)
        if len(events) > 0:
            total_days = (events[-1].end_time - events[0].start_time).total_seconds() / (24 * 3600)
            usage_intensity = len(events) / total_days if total_days > 0 else 0
        else:
            usage_intensity = 0.0

        return regularity_score, usage_intensity

    def analyze_cell_charging_profile(self, system_path: Path, pack: int, cell: int,
                                    start_dt: Optional[datetime] = None,
                                    end_dt: Optional[datetime] = None) -> ChargingProfile:
        """Create comprehensive charging profile for irregular patterns"""

        # Load voltage data
        voltage_series = self._load_cell_timeseries(system_path, pack, cell, start_dt, end_dt)

        if voltage_series.empty:
            # Return default profile
            return ChargingProfile(
                cell_id=f"p{pack}_v{cell}",
                pack_id=pack,
                cell_num=cell,
                total_cycles=0, charge_events=0, discharge_events=0, rest_periods=0,
                typical_charge_voltage=(0, 0), typical_discharge_voltage=(0, 0),
                peak_voltage_trend=0, min_voltage_trend=0,
                avg_charge_duration=0, avg_discharge_duration=0,
                charge_duration_variability=0, discharge_duration_variability=0,
                estimated_capacity_fade=0, charge_efficiency_trend=0, voltage_recovery_time=0,
                cycle_regularity_score=0, usage_intensity_score=0, degradation_rate=0
            )

        # Detect charge/discharge events
        events = self._detect_charge_discharge_events(voltage_series)

        if not events:
            return ChargingProfile(
                cell_id=f"p{pack}_v{cell}",
                pack_id=pack, cell_num=cell,
                total_cycles=0, charge_events=0, discharge_events=0, rest_periods=0,
                typical_charge_voltage=(voltage_series.mean(), voltage_series.mean()),
                typical_discharge_voltage=(voltage_series.mean(), voltage_series.mean()),
                peak_voltage_trend=0, min_voltage_trend=0,
                avg_charge_duration=0, avg_discharge_duration=0,
                charge_duration_variability=0, discharge_duration_variability=0,
                estimated_capacity_fade=0, charge_efficiency_trend=0, voltage_recovery_time=0,
                cycle_regularity_score=0, usage_intensity_score=0, degradation_rate=0
            )

        # Analyze events
        charge_events = [e for e in events if e.event_type == 'charge']
        discharge_events = [e for e in events if e.event_type == 'discharge']

        # Calculate typical voltage ranges
        if charge_events:
            charge_starts = [e.start_voltage for e in charge_events]
            charge_ends = [e.end_voltage for e in charge_events]
            typical_charge_voltage = (np.mean(charge_starts), np.mean(charge_ends))
            avg_charge_duration = np.mean([e.duration_hours for e in charge_events])
            charge_duration_variability = np.std([e.duration_hours for e in charge_events]) / avg_charge_duration if avg_charge_duration > 0 else 0
        else:
            typical_charge_voltage = (voltage_series.mean(), voltage_series.mean())
            avg_charge_duration = 0
            charge_duration_variability = 0

        if discharge_events:
            discharge_starts = [e.start_voltage for e in discharge_events]
            discharge_ends = [e.end_voltage for e in discharge_events]
            typical_discharge_voltage = (np.mean(discharge_starts), np.mean(discharge_ends))
            avg_discharge_duration = np.mean([e.duration_hours for e in discharge_events])
            discharge_duration_variability = np.std([e.duration_hours for e in discharge_events]) / avg_discharge_duration if avg_discharge_duration > 0 else 0
        else:
            typical_discharge_voltage = (voltage_series.mean(), voltage_series.mean())
            avg_discharge_duration = 0
            discharge_duration_variability = 0

        # Calculate trends and degradation
        total_days = (voltage_series.index[-1] - voltage_series.index[0]).total_seconds() / (24 * 3600)
        peak_trend, min_trend, efficiency_trend = self._calculate_degradation_trends(events, total_days)

        # Irregularity metrics
        regularity_score, usage_intensity = self._calculate_irregularity_metrics(events)

        # Estimate capacity fade (simplified)
        capacity_fade = abs(peak_trend) * 12 * 100 / 4.2  # %/year assuming 4.2V nominal

        # Overall degradation rate
        degradation_rate = (abs(peak_trend) + abs(min_trend)) / 2

        return ChargingProfile(
            cell_id=f"p{pack}_v{cell}",
            pack_id=pack,
            cell_num=cell,
            total_cycles=len(charge_events),  # Count charge cycles
            charge_events=len(charge_events),
            discharge_events=len(discharge_events),
            rest_periods=len(events) - len(charge_events) - len(discharge_events),
            typical_charge_voltage=typical_charge_voltage,
            typical_discharge_voltage=typical_discharge_voltage,
            peak_voltage_trend=peak_trend,
            min_voltage_trend=min_trend,
            avg_charge_duration=avg_charge_duration,
            avg_discharge_duration=avg_discharge_duration,
            charge_duration_variability=charge_duration_variability,
            discharge_duration_variability=discharge_duration_variability,
            estimated_capacity_fade=capacity_fade,
            charge_efficiency_trend=efficiency_trend,
            voltage_recovery_time=avg_charge_duration,  # Simplified
            cycle_regularity_score=regularity_score,
            usage_intensity_score=usage_intensity,
            degradation_rate=degradation_rate
        )

    def compare_charging_profiles(self, system_path: Path, pack: int,
                                start_dt: Optional[datetime] = None,
                                end_dt: Optional[datetime] = None) -> List[ChargingProfile]:
        """Compare charging profiles across all cells in a pack"""

        profiles = []
        print(f"Analyzing charging profiles for pack {pack}...")

        for cell in range(1, 53):  # 52 cells
            try:
                profile = self.analyze_cell_charging_profile(system_path, pack, cell, start_dt, end_dt)
                profiles.append(profile)

                if cell % 10 == 0:
                    print(f"  Processed cell {cell}/52")

            except Exception as e:
                print(f"  Warning: Failed to analyze cell {cell}: {e}")
                continue

        return profiles

    def detect_charging_anomalies(self, profiles: List[ChargingProfile]) -> List[ChargingProfile]:
        """Detect cells with anomalous charging behavior"""

        if len(profiles) < 5:
            return []

        anomalous = []

        # Calculate pack statistics for comparison
        valid_profiles = [p for p in profiles if p.total_cycles > 0]
        if not valid_profiles:
            return []

        pack_avg_cycles = np.mean([p.total_cycles for p in valid_profiles])
        pack_avg_regularity = np.mean([p.cycle_regularity_score for p in valid_profiles])
        pack_avg_degradation = np.mean([p.degradation_rate for p in valid_profiles])

        for profile in valid_profiles:
            is_anomalous = False
            reasons = []

            # Check for significantly different cycle count
            if abs(profile.total_cycles - pack_avg_cycles) > pack_avg_cycles * 0.5:
                is_anomalous = True
                reasons.append(f"unusual_cycle_count({profile.total_cycles})")

            # Check for high irregularity
            if profile.cycle_regularity_score < pack_avg_regularity * 0.5:
                is_anomalous = True
                reasons.append(f"irregular_pattern({profile.cycle_regularity_score:.3f})")

            # Check for high degradation
            if profile.degradation_rate > pack_avg_degradation * 2:
                is_anomalous = True
                reasons.append(f"high_degradation({profile.degradation_rate:.4f}V/month)")

            # Check for capacity fade
            if profile.estimated_capacity_fade > 20:  # >20%/year
                is_anomalous = True
                reasons.append(f"capacity_fade({profile.estimated_capacity_fade:.1f}%/year)")

            # Check for voltage trend issues
            if abs(profile.peak_voltage_trend) > 0.01:  # >10mV/month
                is_anomalous = True
                reasons.append(f"voltage_trend({profile.peak_voltage_trend:.4f}V/month)")

            if is_anomalous:
                profile.anomaly_reasons = reasons
                anomalous.append(profile)

        # Sort by severity
        anomalous.sort(key=lambda p: abs(p.degradation_rate) + (1 - p.cycle_regularity_score), reverse=True)

        return anomalous

# Example usage
async def main():
    analyzer = ChargingCurveAnalyzer()

    # Discover systems
    roots = [Path("data/BESS")]
    systems = {}
    for root in roots:
        if root.exists():
            for system_dir in root.iterdir():
                if system_dir.is_dir() and "ZHPESS" in system_dir.name:
                    systems[system_dir.name] = system_dir

    if not systems:
        print("No BESS systems found")
        return

    system_name = list(systems.keys())[0]
    system_path = systems[system_name]

    print(f"Analyzing irregular charging patterns for {system_name}")

    # Analyze pack 1 as example
    profiles = analyzer.compare_charging_profiles(system_path, 1)

    print(f"\nCharging Profile Summary for Pack 1:")
    print(f"Total cells analyzed: {len(profiles)}")

    valid_profiles = [p for p in profiles if p.total_cycles > 0]
    if valid_profiles:
        avg_cycles = np.mean([p.total_cycles for p in valid_profiles])
        avg_regularity = np.mean([p.cycle_regularity_score for p in valid_profiles])
        avg_intensity = np.mean([p.usage_intensity_score for p in valid_profiles])

        print(f"Average cycles detected: {avg_cycles:.1f}")
        print(f"Average regularity score: {avg_regularity:.3f} (1.0 = very regular)")
        print(f"Average usage intensity: {avg_intensity:.2f} events/day")

        # Find anomalous charging behavior
        anomalies = analyzer.detect_charging_anomalies(profiles)
        print(f"\nAnomalous cells detected: {len(anomalies)}")

        for i, profile in enumerate(anomalies[:5], 1):
            print(f"{i}. Cell {profile.cell_num}:")
            print(f"   Cycles: {profile.total_cycles}, Regularity: {profile.cycle_regularity_score:.3f}")
            print(f"   Degradation: {profile.degradation_rate:.4f}V/month")
            print(f"   Capacity fade: {profile.estimated_capacity_fade:.1f}%/year")
            if hasattr(profile, 'anomaly_reasons'):
                print(f"   Issues: {', '.join(profile.anomaly_reasons)}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())