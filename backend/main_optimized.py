# backend/main_optimized.py
"""
Optimized Energy Analytics API

Key features:
- Recursive discovery of meters & BESS CSVs (via METER_ROOTS or sensible defaults)
- Reads precomputed Parquet pyramids from backend/.meter_cache
- Robust Parquet reading (restores DatetimeIndex; tolerant of index/value column variants)
- Date clamping: intersects requested range with available data; returns actual_start/actual_end
- Clear 404s (e.g., missing pyramid) instead of generic 500s
- Meters/BESS classification + info endpoints
- BESS KPIs endpoint (SOC/SOH/PCS/AUX + safety)
- /cache/validate to check preprocessing coverage
"""

from __future__ import annotations

import os
import time
import hashlib
import asyncio
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any
from datetime import datetime, timedelta
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
import uvloop

# Import cell analyzer
import sys
sys.path.append('.')
from cell_analyzer import CellAnalyzer, PackHealthSummary, CellMetrics

# ---------------- Runtime config ----------------
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

TIMEZONE = "Europe/Berlin"
CACHE_DIR = Path(__file__).resolve().parent / ".meter_cache"
CACHE_DIR.mkdir(exist_ok=True)

MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "4"))
CACHE_TTL = int(os.environ.get("CACHE_TTL", "300"))  # seconds
MAX_MEMORY_CACHE_SIZE = 128

thread_pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# Initialize cell analyzer
cell_analyzer = CellAnalyzer()


# ---------------- Small TTL caches ----------------
class TimedCache:
    def __init__(self, ttl_seconds: int = CACHE_TTL):
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._ttl = ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            v, ts = self._cache[key]
            if time.time() - ts < self._ttl:
                return v
            del self._cache[key]
        return None

    def put(self, key: str, value: Any):
        if len(self._cache) >= MAX_MEMORY_CACHE_SIZE:
            oldest = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest]
        self._cache[key] = (value, time.time())

    def clear(self):
        self._cache.clear()


series_cache = TimedCache(CACHE_TTL)
kpi_cache = TimedCache(CACHE_TTL * 2)


# ---------------- Utilities ----------------
def _parse_roots() -> List[Path]:
    raw = os.environ.get("METER_ROOTS", "").strip()
    if raw:
        return [Path(p.strip()) for p in raw.split(",") if p.strip()]
    candidates = [
        Path("data/meter"), Path("data/BESS"),
        Path("../data/meter"), Path("../data/BESS"),
        Path("./meter"), Path("./BESS"),
    ]
    return [p for p in candidates if p.exists()]


def is_bess_path(p: Path) -> bool:
    s = str(p).lower()
    return ("bess" in s) or ("zhpess" in s)


def detect_signal_generic(csv_path: Path) -> str:
    name = csv_path.stem.lower()
    if "com_ap" in name: return "com_ap"
    if "com_ae" in name: return "com_ae"
    if "pos_ae" in name: return "pos_ae"
    if "neg_ae" in name: return "neg_ae"
    if name.endswith("_pf") or name == "pf": return "pf"
    return csv_path.stem  # keep BESS stems as-is


def is_energy_signal_key(sig: str) -> bool:
    s = sig.lower()
    return s.endswith(("_pos_ae", "_neg_ae", "_com_ae")) or s in ("pos_ae", "neg_ae", "com_ae")


async def _run_in_thread(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(thread_pool, func, *args, **kwargs)


def _restore_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure datetime index exists. Accepts:
    - Parquet with index restored
    - Parquet with index in '__index_level_0__' or 'index'
    - Parquet with a timestamp column (timestamp/time/date/datetime/ts)
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        idx_col = None
        for cand in ["__index_level_0__", "index", "timestamp", "time", "date", "datetime", "ts"]:
            if cand in df.columns:
                idx_col = cand
                break
        if idx_col:
            df[idx_col] = pd.to_datetime(df[idx_col], utc=True, errors="coerce")
            df = df.set_index(idx_col)
        else:
            # last resort: try to_datetime on the existing index
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")

    df = df.sort_index()
    # localize/convert timezone
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT").tz_convert(TIMEZONE)
    else:
        df.index = df.index.tz_convert(TIMEZONE)
    return df


def _load_parquet_fast(path: Path) -> pd.Series:
    try:
        table = pq.read_table(str(path), memory_map=True)
        df = table.to_pandas()
    except Exception as e:
        raise HTTPException(500, f"Parquet read failed for {path.name}: {e}") from e

    df = _restore_dt_index(df)

    # robustly pick the value column
    if "value" not in df.columns:
        if df.shape[1] == 1:
            df = df.rename(columns={df.columns[0]: "value"})
        else:
            raise HTTPException(500, f"'value' column missing in {path.name} (cols={list(df.columns)})")

    try:
        s = pd.to_numeric(df["value"], errors="coerce").astype("float32").dropna()
    except Exception as e:
        raise HTTPException(500, f"Cannot coerce 'value' to float in {path.name}: {e}") from e

    return s.sort_index()


def _series_bounds(series: pd.Series) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    if series is None or len(series) == 0:
        return None, None
    return series.index.min(), series.index.max()


def _parse_client_dt(s: Optional[str]) -> Optional[pd.Timestamp]:
    """
    Parse client-provided ISO datetimes.
    - If naive (no timezone), assume TIMEZONE (Europe/Berlin).
    - If tz-aware, convert to TIMEZONE.
    Returns tz-aware pandas Timestamp or None.
    """
    if not s:
        return None
    ts = pd.to_datetime(s, errors="coerce")
    if pd.isna(ts):
        return None
    if getattr(ts, "tzinfo", None) is None:
        return ts.tz_localize(TIMEZONE, nonexistent="shift_forward", ambiguous="NaT")
    return ts.tz_convert(TIMEZONE)


# ---------------- Discovery (recursive) ----------------
@lru_cache(maxsize=1)
def _discover_all_meters() -> Dict[str, Dict[str, List[Path]]]:
    roots = _parse_roots()
    result: Dict[str, Dict[str, List[Path]]] = {}
    for root in roots:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in (".git", ".venv", "__pycache__", "zipped", ".meter_cache")]
            csvs = [f for f in filenames if f.lower().endswith(".csv")]
            if not csvs:
                continue
            folder = Path(dirpath)
            sysname = folder.name
            bucket = result.setdefault(sysname, {})
            for fn in csvs:
                csv_path = folder / fn
                sig = detect_signal_generic(csv_path)
                bucket.setdefault(sig, []).append(csv_path)
    return result


async def _get_meter_index() -> Dict[str, List[str]]:
    cache_key = "meter_index"
    cached = series_cache.get(cache_key)
    if cached:
        return cached
    allm = await _run_in_thread(_discover_all_meters)
    index = {name: sorted(sigs.keys()) for name, sigs in allm.items()}
    series_cache.put(cache_key, index)
    return index


# ---------------- Series loader ----------------
def _choose_rule(start_dt: Optional[datetime], end_dt: Optional[datetime]) -> str:
    if not start_dt or not end_dt:
        return "15min"
    span = end_dt - start_dt
    if span <= timedelta(hours=6): return "5min"
    if span <= timedelta(days=2):  return "15min"
    if span <= timedelta(days=14): return "1h"
    return "1d"


async def _load_series_with_lod(
    meter: str, signal: str,
    start_dt: Optional[datetime], end_dt: Optional[datetime],
    max_points: int
) -> Tuple[pd.Series, str]:
    ckey = f"{meter}:{signal}:{start_dt}:{end_dt}:{max_points}"
    cached = series_cache.get(ckey)
    if cached:
        return cached

    allm = await _run_in_thread(_discover_all_meters)
    if meter not in allm or signal not in allm[meter]:
        raise HTTPException(404, f"Signal '{signal}' not found in meter '{meter}'")

    # normalize tz (safety)
    if start_dt is not None and getattr(start_dt, "tzinfo", None) is not None:
        start_dt = start_dt.tz_convert(TIMEZONE)
    if end_dt is not None and getattr(end_dt, "tzinfo", None) is not None:
        end_dt = end_dt.tz_convert(TIMEZONE)

    rule = _choose_rule(start_dt, end_dt)
    series: Optional[pd.Series] = None

    # merge all same-signal files (if any)
    for csv_path in allm[meter][signal]:
        sig = hashlib.md5(f"{csv_path.resolve()}::{csv_path.stat().st_size}::{csv_path.stat().st_mtime}".encode()).hexdigest()
        pqt = CACHE_DIR / f"{sig}__{rule}.parquet"
        if not pqt.exists():
            raise HTTPException(404, f"Pyramid missing for {csv_path.name} [{rule}]. Re-run preprocessor.")
        s = await _run_in_thread(_load_parquet_fast, pqt)
        series = s if series is None else pd.concat([series, s]).sort_index()

    if series is None or len(series) == 0:
        raise HTTPException(404, f"No processed data for {meter}:{signal}")

    # Intersect requested range with available
    a_start, a_end = _series_bounds(series)
    req_start = start_dt or a_start
    req_end   = end_dt   or a_end

    if req_start and a_start and req_start < a_start:
        req_start = a_start
    if req_end and a_end and req_end > a_end:
        req_end = a_end

    if req_start and req_end and req_start > req_end:
        raise HTTPException(404, "Requested range has no overlap with available data")

    if req_start or req_end:
        series = series.loc[req_start:req_end]

    if len(series) == 0:
        raise HTTPException(404, "No data in specified time range")

    # Convert energy counters to intervals
    if is_energy_signal_key(signal):
        intervals = series.diff()
        intervals = intervals.where(intervals >= 0, 0)
        if len(intervals) > 10:
            cap = intervals.quantile(0.999)
            intervals = intervals.where(intervals <= cap, cap)
        series = intervals.fillna(0)

    # Downsample if needed
    if len(series) > max_points:
        step = max(1, len(series) // max_points)
        series = series.iloc[::step]

    result = (series, rule)
    series_cache.put(ckey, result)
    return result


# ---------------- KPI calculation ----------------
async def _calculate_kpis_parallel(
    meter: str, signals: List[str],
    start_dt: Optional[datetime], end_dt: Optional[datetime]
) -> Dict[str, Any]:
    ckey = f"kpis:{meter}:{','.join(sorted(signals))}:{start_dt}:{end_dt}"
    cached = kpi_cache.get(ckey)
    if cached:
        return cached

    async def one(sig: str) -> Tuple[str, Dict[str, float]]:
        try:
            s, _ = await _load_series_with_lod(meter, sig, start_dt, end_dt, max_points=50000)
            if s.empty:
                return sig, {}
            k = {
                "count": int(len(s)),
                "mean": float(s.mean()),
                "min": float(s.min()),
                "max": float(s.max()),
                "std": float(s.std()),
                "sum": float(s.sum()),
            }
            if not is_energy_signal_key(sig):
                k.update({
                    "p25": float(s.quantile(0.25)),
                    "p50": float(s.quantile(0.50)),
                    "p75": float(s.quantile(0.75)),
                    "p95": float(s.quantile(0.95)),
                })
            return sig, k
        except HTTPException:
            return sig, {}
        except Exception as e:
            print(f"[WARN] KPI calc failed for {sig}: {e}")
            return sig, {}

    results = await asyncio.gather(*[one(s) for s in signals], return_exceptions=False)
    data = {sig: k for sig, k in results if isinstance(k, dict)}
    kpi_cache.put(ckey, data)
    return data


# ---------------- FastAPI app ----------------
app = FastAPI(title="Energy Analytics API (Optimized)", version="2.4.0")

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


class SeriesResponse(BaseModel):
    timestamps: List[str]
    values: List[float]
    rule: str
    count: int
    meter: str
    signal: str
    actual_start: Optional[str] = None
    actual_end: Optional[str] = None

# Cell analyzer response models
class CellMetricsResponse(BaseModel):
    cell_id: str
    pack_id: int
    cell_num: int
    voltage_mean: float
    voltage_std: float
    voltage_min: float
    voltage_max: float
    degradation_rate: float
    imbalance_score: float
    temp_max: float
    data_points: int
    data_quality: float

class PackHealthResponse(BaseModel):
    pack_id: int
    bess_system: str
    pack_soh: float
    average_voltage: float
    voltage_imbalance: float
    avg_temperature: float
    degradation_rate: float
    worst_cell: str
    best_cell: str
    healthy_cells: int
    warning_cells: int
    critical_cells: int
    discharge_cycles: int
    usage_pattern: str


# ---------------- Endpoints ----------------
@app.get("/")
async def root():
    return {"service": "Energy Analytics API (Optimized)", "status": "running"}


@app.get("/meters")
async def meters_all() -> Dict[str, List[str]]:
    return await _get_meter_index()


@app.get("/meters/classified")
async def meters_classified():
    """Return two buckets to simplify frontend filtering."""
    allm = await _run_in_thread(_discover_all_meters)
    out = {"meters": {}, "bess": {}}
    for sysname, sigs in allm.items():
        sample = None
        for paths in sigs.values():
            if paths:
                sample = paths[0]
                break
        bucket = "bess" if (sample and is_bess_path(sample)) else "meters"
        out[bucket][sysname] = sorted(sigs.keys())
    return out


@app.get("/meters/{meter}/info")
async def meter_info(meter: str):
    allm = await _run_in_thread(_discover_all_meters)
    if meter not in allm:
        raise HTTPException(404, f"{meter} not found")
    sigs = sorted(allm[meter].keys())
    has_bess = any(s.lower().startswith(("bms1", "pcs1")) for s in sigs)
    return {"meter": meter, "signals": sigs, "signal_count": len(sigs), "has_bess": has_bess}


@app.get("/series", response_model=SeriesResponse)
async def series(
    meter: str,
    signal: str,
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
    max_points: int = Query(6000, ge=100, le=20000),
):
    # Parse in local project timezone (Berlin) for safe slicing
    start_dt = _parse_client_dt(start)
    end_dt = _parse_client_dt(end)

    try:
        s, rule = await _load_series_with_lod(meter, signal, start_dt, end_dt, max_points)
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(500, f"Failed to load series: {e}") from e

    return SeriesResponse(
        timestamps=[t.isoformat() for t in s.index],
        values=s.values.astype(float).tolist(),
        rule=rule,
        count=int(len(s)),
        meter=meter,
        signal=signal,
        actual_start=s.index.min().isoformat() if len(s) else None,
        actual_end=s.index.max().isoformat() if len(s) else None,
    )


@app.get("/bess_kpis")
async def bess_kpis(meter: str, rule: str, start: str, end: str):
    start_dt = _parse_client_dt(start)
    end_dt = _parse_client_dt(end)
    idx = await _get_meter_index()
    sigs = idx.get(meter)
    if not sigs:
        raise HTTPException(404, f"BESS {meter} not found")

    wanted = [
        "bms1_soc", "bms1_soh", "bms1_v", "bms1_c",
        "pcs1_ap", "pcs1_dcc", "pcs1_dcv", "pcs1_ia", "pcs1_ib", "pcs1_ic",
        "pcs1_t_env", "pcs1_t_a", "pcs1_t_igbt",
        "aux_m_ap", "aux_m_pf",
    ]
    available = [s for s in wanted if s in sigs]
    kpis = await _calculate_kpis_parallel(meter, available, start_dt, end_dt)

    safety = [s for s in sigs if s.lower().startswith("fa") or "flag" in s.lower() or "err" in s.lower()]
    if safety:
        kpis["safety"] = await _calculate_kpis_parallel(meter, safety, start_dt, end_dt)

    return {"meter": meter, "rule": rule,
            "start": start_dt.isoformat() if start_dt is not None else None,
            "end": end_dt.isoformat() if end_dt is not None else None,
            "kpis": kpis}


@app.post("/cache/clear")
async def cache_clear():
    series_cache.clear()
    kpi_cache.clear()
    _discover_all_meters.cache_clear()
    return {"status": "cleared"}


@app.get("/cache/validate")
async def cache_validate():
    """
    Check which CSV files have all pyramid levels and which are missing.
    Helps explain 404s like 'Pyramid missing for ...'.
    """
    levels = ["5min", "15min", "1h", "1d"]
    allm = await _run_in_thread(_discover_all_meters)
    missing: List[str] = []
    ok = 0
    total = 0
    for _, sigs in allm.items():
        for paths in sigs.values():
            for csv_path in paths:
                total += 1
                sig = hashlib.md5(f"{csv_path.resolve()}::{csv_path.stat().st_size}::{csv_path.stat().st_mtime}".encode()).hexdigest()
                if not all((CACHE_DIR / f"{sig}__{lvl}.parquet").exists() for lvl in levels):
                    missing.append(str(csv_path))
                else:
                    ok += 1
    return {"csv_total": total, "ok": ok, "missing": missing}


# ---------------- Cell Analyzer Endpoints ----------------
@app.get("/cell/pack/{bess_system}/{pack_id}/health", response_model=PackHealthResponse)
async def get_pack_health(
    bess_system: str,
    pack_id: int,
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
):
    """Get comprehensive health analysis for a specific pack"""
    start_dt = _parse_client_dt(start)
    end_dt = _parse_client_dt(end)

    try:
        summary, _ = await _run_in_thread(
            cell_analyzer.analyze_pack_health,
            bess_system, pack_id, start_dt, end_dt
        )
        return PackHealthResponse(
            pack_id=summary.pack_id,
            bess_system=summary.bess_system,
            pack_soh=summary.pack_soh,
            average_voltage=summary.average_voltage,
            voltage_imbalance=summary.voltage_imbalance,
            avg_temperature=summary.avg_temperature,
            degradation_rate=summary.degradation_rate,
            worst_cell=summary.worst_cell,
            best_cell=summary.best_cell,
            healthy_cells=summary.healthy_cells,
            warning_cells=summary.warning_cells,
            critical_cells=summary.critical_cells,
            discharge_cycles=summary.discharge_cycles,
            usage_pattern=summary.usage_pattern
        )
    except Exception as e:
        raise HTTPException(500, f"Pack health analysis failed: {e}")


@app.get("/cell/pack/{bess_system}/{pack_id}/cells")
async def get_pack_cells(
    bess_system: str,
    pack_id: int,
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
):
    """Get detailed metrics for all cells in a pack"""
    start_dt = _parse_client_dt(start)
    end_dt = _parse_client_dt(end)

    try:
        _, cell_metrics = await _run_in_thread(
            cell_analyzer.analyze_pack_health,
            bess_system, pack_id, start_dt, end_dt
        )

        cells = []
        for cell in cell_metrics:
            cells.append(CellMetricsResponse(
                cell_id=cell.cell_id,
                pack_id=cell.pack_id,
                cell_num=cell.cell_num,
                voltage_mean=cell.voltage_mean,
                voltage_std=cell.voltage_std,
                voltage_min=cell.voltage_min,
                voltage_max=cell.voltage_max,
                degradation_rate=cell.degradation_rate,
                imbalance_score=cell.imbalance_score,
                temp_max=cell.temp_max,
                data_points=cell.data_points,
                data_quality=cell.data_quality
            ))

        return {"bess_system": bess_system, "pack_id": pack_id, "cells": cells}
    except Exception as e:
        raise HTTPException(500, f"Cell analysis failed: {e}")


@app.get("/cell/system/{bess_system}/comparison")
async def get_pack_comparison(
    bess_system: str,
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
):
    """Compare health across all 5 packs in a BESS system"""
    start_dt = _parse_client_dt(start)
    end_dt = _parse_client_dt(end)

    try:
        pack_summaries = await _run_in_thread(
            cell_analyzer.compare_packs_degradation,
            bess_system, start_dt, end_dt
        )

        comparison = {}
        for pack_id, summary in pack_summaries.items():
            comparison[f"pack_{pack_id}"] = PackHealthResponse(
                pack_id=summary.pack_id,
                bess_system=summary.bess_system,
                pack_soh=summary.pack_soh,
                average_voltage=summary.average_voltage,
                voltage_imbalance=summary.voltage_imbalance,
                avg_temperature=summary.avg_temperature,
                degradation_rate=summary.degradation_rate,
                worst_cell=summary.worst_cell,
                best_cell=summary.best_cell,
                healthy_cells=summary.healthy_cells,
                warning_cells=summary.warning_cells,
                critical_cells=summary.critical_cells,
                discharge_cycles=summary.discharge_cycles,
                usage_pattern=summary.usage_pattern
            )

        return {"bess_system": bess_system, "packs": comparison}
    except Exception as e:
        raise HTTPException(500, f"Pack comparison failed: {e}")


@app.get("/cell/pack/{bess_system}/{pack_id}/anomalies")
async def get_anomalous_cells(
    bess_system: str,
    pack_id: int,
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
):
    """Detect cells with anomalous behavior"""
    print(f"[INFO] Anomaly detection request: {bess_system}, pack {pack_id}, {start}-{end}")
    start_dt = _parse_client_dt(start)
    end_dt = _parse_client_dt(end)
    print(f"[INFO] Parsed dates: {start_dt} to {end_dt}")

    try:
        anomalous_cells = await _run_in_thread(
            cell_analyzer.detect_anomalous_cells,
            bess_system, pack_id, start_dt, end_dt
        )
        print(f"[INFO] Found {len(anomalous_cells)} anomalous cells")

        anomalies = []
        for cell in anomalous_cells:
            cell_data = CellMetricsResponse(
                cell_id=cell.cell_id,
                pack_id=cell.pack_id,
                cell_num=cell.cell_num,
                voltage_mean=cell.voltage_mean,
                voltage_std=cell.voltage_std,
                voltage_min=cell.voltage_min,
                voltage_max=cell.voltage_max,
                degradation_rate=cell.degradation_rate,
                imbalance_score=cell.imbalance_score,
                temp_max=cell.temp_max,
                data_points=cell.data_points,
                data_quality=cell.data_quality
            )

            # Add anomaly reasons if available
            anomaly_info = {
                "cell": cell_data,
                "severity_score": abs(cell.degradation_rate) * 1000 + cell.imbalance_score * 10,
                "reasons": getattr(cell, 'anomaly_reasons', [])
            }
            anomalies.append(anomaly_info)

        return {
            "bess_system": bess_system,
            "pack_id": pack_id,
            "anomalous_cells_count": len(anomalies),
            "anomalies": anomalies
        }
    except Exception as e:
        raise HTTPException(500, f"Anomaly detection failed: {e}")


@app.get("/cell/pack/{bess_system}/{pack_id}/heatmap")
async def get_cell_heatmap_data(
    bess_system: str,
    pack_id: int,
    metric: str = Query("voltage", description="voltage, temperature, or degradation"),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
):
    """Get 52-cell heatmap data for visualization"""
    start_dt = _parse_client_dt(start)
    end_dt = _parse_client_dt(end)

    try:
        _, cell_metrics = await _run_in_thread(
            cell_analyzer.analyze_pack_health,
            bess_system, pack_id, start_dt, end_dt
        )

        # Create 2D array for heatmap (assume 13x4 grid layout)
        heatmap_data = []

        for cell in sorted(cell_metrics, key=lambda x: x.cell_num):
            if metric == "voltage":
                value = cell.voltage_mean
            elif metric == "temperature":
                value = cell.temp_max
            elif metric == "degradation":
                value = abs(cell.degradation_rate) * 1000  # Convert to mV/month
            else:
                value = cell.imbalance_score

            heatmap_data.append({
                "cell_num": cell.cell_num,
                "value": value,
                "x": (cell.cell_num - 1) % 13,  # 13 cells per row
                "y": (cell.cell_num - 1) // 13,  # 4 rows
                "cell_id": cell.cell_id
            })

        return {
            "bess_system": bess_system,
            "pack_id": pack_id,
            "metric": metric,
            "heatmap_data": heatmap_data,
            "dimensions": {"rows": 4, "cols": 13}
        }
    except Exception as e:
        raise HTTPException(500, f"Heatmap data generation failed: {e}")


# ---------------- Lifecycle ----------------
@app.on_event("startup")
async def on_start():
    print("[INFO] Energy API starting...")
    print(f"[INFO] Cache dir: {CACHE_DIR.resolve()}")
    try:
        idx = await _get_meter_index()
        print(f"[INFO] Discovered {len(idx)} systems")
    except Exception as e:
        print(f"[WARN] Discovery failed: {e}")


@app.on_event("shutdown")
async def on_stop():
    series_cache.clear()
    kpi_cache.clear()
    thread_pool.shutdown(wait=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_optimized:app", host="0.0.0.0", port=8000, workers=1, loop="uvloop", access_log=False)
