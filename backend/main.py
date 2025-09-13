# backend/main.py
# Run:
#   METER_ROOTS="data/meter,data/BESS" uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 2 --no-access-log

from __future__ import annotations
import os
import hashlib
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel

# ---------------- Config ----------------
TIMEZONE = "Europe/Berlin"
CACHE_DIR = Path("./.meter_cache")  # populated by preprocess_pyramids.py
CACHE_DIR.mkdir(exist_ok=True)
PYRAMID_RULES = ["5min", "15min", "1h", "1d"]  # lowercase

def _norm_rule(rule: str) -> str:
    return rule.lower().strip()

def _parse_roots() -> List[Path]:
    raw = os.environ.get("METER_ROOTS", "").strip()
    if raw:
        return [Path(p.strip()) for p in raw.split(",") if p.strip()]
    candidates = [Path("../data/meter"), Path("../data/BESS"), Path("./meter"), Path("./BESS")]
    return [p for p in candidates if p.exists()]

ROOTS: List[Path] = _parse_roots()

# ---------------- Generic signal helpers (meter + BESS) ----------------
def is_bess_path(p: Path) -> bool:
    s = str(p).lower()
    return "/bess/" in s or "zhpess" in s

def detect_signal_generic(csv_path: Path) -> str:
    """Normalize common meter names; otherwise keep file stem (for BESS etc.)."""
    name = csv_path.stem
    low = name.lower()
    if "com_ap" in low: return "com_ap"
    if "com_ae" in low: return "com_ae"
    if "pos_ae" in low: return "pos_ae"
    if "neg_ae" in low: return "neg_ae"
    if low.endswith("_pf") or low == "pf": return "pf"
    return name  # BESS or others: keep stem

def is_energy_signal_key(sig: str) -> bool:
    """True if this is a cumulative energy snapshot (needs diff to intervals)."""
    s = sig.lower()
    return s.endswith(("_pos_ae", "_neg_ae", "_com_ae")) or s in ("pos_ae", "neg_ae", "com_ae")

# ---------------- IO & caching ----------------
def _file_sig(p: Path) -> str:
    s = p.stat()
    return hashlib.md5(f"{p.resolve()}::{s.st_size}::{s.st_mtime}".encode()).hexdigest()

def _pyramid_path(sig: str, rule: str) -> Path:
    return CACHE_DIR / f"{sig}__{_norm_rule(rule)}.parquet"

def load_series_cached(csv_path: Path) -> pd.Series:
    """Fallback single-level cache if a pyramid file is missing (rare)."""
    sig = _file_sig(csv_path)
    pqt = CACHE_DIR / f"{sig}.parquet"
    if pqt.exists():
        df = pd.read_parquet(pqt)
        idx = pd.DatetimeIndex(pd.to_datetime(df.index))
        idx = idx.tz_convert(TIMEZONE) if idx.tz is not None else idx.tz_localize(TIMEZONE)
        df.index = idx
        s = pd.to_numeric(df["value"], errors="coerce").dropna().astype("float32")
        s.name = csv_path.stem
        return s

    # Minimal CSV parse (only if preprocessor wasn't run)
    df = pd.read_csv(csv_path)
    ts_cands = [c for c in df.columns if c.lower() in ("timestamp","time","date","datetime","ts")]
    idx = ts_cands[0] if ts_cands else df.columns[0]
    df[idx] = pd.to_datetime(df[idx], utc=True, errors="coerce")
    if df[idx].isna().all():
        df[idx] = pd.to_datetime(df[idx], errors="coerce")
    df = df.set_index(idx).sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize(TIMEZONE, ambiguous="NaT", nonexistent="shift_forward")
    else:
        df.index = df.index.tz_convert(TIMEZONE)
    # pick a numeric column
    val = None
    for c in df.columns:
        if c.lower() in ("timestamp","time","date","datetime","ts"): continue
        if pd.api.types.is_numeric_dtype(df[c]): val = c; break
    if val is None:
        for c in df.columns:
            if c.lower() in ("timestamp","time","date","datetime","ts"): continue
            try: pd.to_numeric(df[c], errors="raise"); val = c; break
            except Exception: pass
    if val is None: raise ValueError(f"No numeric column found in {csv_path.name}")
    s = pd.to_numeric(df[val], errors="coerce").dropna().astype("float32")
    s.name = csv_path.stem
    s.to_frame("value").to_parquet(pqt)
    return s

def _load_pyramid_series(csv_path: Path, rule: str) -> pd.Series:
    """Open precomputed Parquet for a given rule; fallback to on-the-fly if missing."""
    rule = _norm_rule(rule)
    sig = _file_sig(csv_path)
    pqt = _pyramid_path(sig, rule)
    if not pqt.exists():
        s = load_series_cached(csv_path)
        # fallback resample based on type
        how = "mean" if detect_signal_generic(csv_path) in ("com_ap","pf") else "last"
        s = s.resample(rule).mean() if how == "mean" else s.resample(rule).last()
        return s
    df = pd.read_parquet(pqt)
    idx = pd.DatetimeIndex(pd.to_datetime(df.index))
    idx = idx.tz_convert(TIMEZONE) if idx.tz is not None else idx.tz_localize(TIMEZONE)
    return pd.Series(pd.to_numeric(df["value"], errors="coerce").astype("float32"), index=idx, name=csv_path.stem)

def lod_rule_for_span(start: Optional[pd.Timestamp], end: Optional[pd.Timestamp], fallback: str = "15min") -> str:
    if start is None or end is None:
        return _norm_rule(fallback)
    span = (end - start).total_seconds()
    if span > 60*60*24*60:  # > 60 days
        return "1h"
    if span > 60*60*24*14:  # 14–60 days
        return "15min"
    return "5min"           # ≤ 14 days

def to_interval_energy_kwh(s: pd.Series) -> pd.Series:
    """Diff cumulative counter; protect resets and outliers."""
    d = s.diff()
    d = d.where(d >= 0)  # protect rollovers/resets
    if d.dropna().size > 10:
        cap = np.nanpercentile(d.dropna(), 99.9)
        if cap > 0:
            d = d.clip(upper=cap)
    return d

# LTTB downsampling for plotting payloads
def lttb_downsample(x: np.ndarray, y: np.ndarray, threshold: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(x)
    if threshold >= n or threshold < 3:
        return x, y
    bucket = (n - 2) / (threshold - 2)
    a = 0
    sx = [x[0]]; sy = [y[0]]
    for i in range(1, threshold - 1):
        start = int(np.floor((i - 1) * bucket)) + 1
        end = int(np.floor(i * bucket)) + 1
        end = min(end, n - 1)
        x_seg = x[start:end]; y_seg = y[start:end]
        nxt_end = int(np.floor((i + 1) * bucket)) + 1
        nxt_end = min(nxt_end, n)
        avg_x = x[end:nxt_end].mean() if end < nxt_end else x[-1]
        avg_y = y[end:nxt_end].mean() if end < nxt_end else y[-1]
        ax, ay = x[a], y[a]
        area = np.abs((ax - avg_x) * (y_seg - ay) - (ax - x_seg) * (avg_y - ay))
        idx = np.argmax(area)
        a = start + idx
        sx.append(x[a]); sy.append(y[a])
    sx.append(x[-1]); sy.append(y[-1])
    return np.array(sx), np.array(sy)

def downsample_for_json(s: pd.Series, max_points: int) -> pd.Series:
    s = s.dropna()
    n = len(s)
    if n <= max_points:
        return s
    x = s.index.view("int64").astype(np.float64)
    y = s.values.astype(np.float64)
    xs, ys = lttb_downsample(x, y, max_points)
    idx = pd.to_datetime(xs.astype(np.int64), utc=True).tz_convert(TIMEZONE)
    return pd.Series(ys.astype("float32"), index=idx, name=s.name).sort_index()

# ---------------- Discovery ----------------
Meters: List[Path] = []
SignalsByMeter: Dict[str, Dict[str, Path]] = {}

def discover_meters() -> List[Path]:
    out: List[Path] = []
    for r in ROOTS:
        if not r.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(r):
            dirnames[:] = [d for d in dirnames if d not in (".git",".venv","__pycache__","zipped")]
            if any(fn.endswith(".csv") for fn in filenames):
                out.append(Path(dirpath))
    return sorted(set(out))

def build_index() -> None:
    global Meters, SignalsByMeter
    Meters = discover_meters()
    SignalsByMeter = {}
    for fol in Meters:
        label = fol.as_posix().lstrip("./")
        SignalsByMeter[label] = {}
        for csv in sorted(fol.glob("*.csv")):
            key = detect_signal_generic(csv)
            SignalsByMeter[label][key] = csv

# ---------------- FastAPI app ----------------
app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class SeriesOut(BaseModel):
    meter: str
    signal: str
    timestamps: List[str]
    values: List[float]
    rule: str

class KpisOut(BaseModel):
    meter: str
    total_import_kWh: Optional[float]
    total_export_kWh: Optional[float]
    net_kWh: Optional[float]
    peak_kW: Optional[float]
    avg_pf: Optional[float]

class BundleOut(BaseModel):
    meter: str
    kpis: KpisOut
    series: Dict[str, SeriesOut]

class BessKpisOut(BaseModel):
    meter: str
    soc_avg: Optional[float] = None
    soc_min: Optional[float] = None
    soh_avg: Optional[float] = None
    pack_v_avg: Optional[float] = None
    pack_c_avg: Optional[float] = None
    cell_v_spread_max: Optional[float] = None
    cell_t_avg: Optional[float] = None
    pcs_ap_peak: Optional[float] = None
    aux_ap_avg: Optional[float] = None
    env_temp_avg: Optional[float] = None
    alarms_any: Optional[bool] = None

@app.on_event("startup")
def _startup():
    build_index()

@app.get("/")
def root():
    return {"service":"smart-meter-api","status":"ok",
            "roots":[str(r) for r in ROOTS],
            "meters_found": len(Meters),
            "endpoints":["/meters","/series","/kpis","/bundle","/bess_kpis","/reload","/prewarm","/health"]}

@app.get("/health")
def health():
    return {"ok": True, "roots": [str(r) for r in ROOTS], "meters": len(Meters)}

@app.post("/reload")
def reload_index():
    build_index()
    return {"ok": True, "meters": len(Meters)}

@app.post("/prewarm")
def prewarm():
    built = 0
    for _, signals in SignalsByMeter.items():
        for _, csv in signals.items():
            sig = _file_sig(csv)
            for rule in PYRAMID_RULES:
                pqt = _pyramid_path(sig, rule)
                if not pqt.exists():
                    s = load_series_cached(csv)
                    how = "mean" if detect_signal_generic(csv) in ("com_ap","pf") else "last"
                    s = s.resample(rule).mean() if how == "mean" else s.resample(rule).last()
                    s.to_frame("value").to_parquet(pqt)
                    built += 1
    return {"ok": True, "touched": built}

@app.get("/meters")
def list_meters():
    return sorted(SignalsByMeter.keys())

@app.get("/meters/classified")
def list_meters_classified():
    """Return meters classified as 'meter' or 'bess' type"""
    result = {"meters": [], "bess": []}
    for meter_path in sorted(SignalsByMeter.keys()):
        if is_bess_path(Path(meter_path)):
            result["bess"].append(meter_path)
        else:
            result["meters"].append(meter_path)
    return result

@app.get("/meters/{meter}/info")
def get_meter_info(meter: str):
    """Get meter info including available signals and date range"""
    if meter not in SignalsByMeter:
        raise HTTPException(404, f"Unknown meter: {meter}")
    
    signals = list(SignalsByMeter[meter].keys())
    
    # Find date range by checking a representative signal
    date_range = {"start": None, "end": None}
    for sig in ["com_ap", "pos_ae", "bms1_soc", "pcs1_ap"]:
        if sig in SignalsByMeter[meter]:
            try:
                s = load_series_cached(SignalsByMeter[meter][sig])
                if not s.empty:
                    date_range["start"] = s.index[0].isoformat()
                    date_range["end"] = s.index[-1].isoformat()
                    break
            except Exception:
                pass
    
    return {
        "meter": meter,
        "type": "bess" if is_bess_path(Path(meter)) else "meter",
        "signals": signals,
        "signal_count": len(signals),
        "date_range": date_range
    }

# ---------- core loaders ----------
def _load_series_lod(meter: str, signal: str, rule_query: str, cumulative: bool,
                     start: Optional[str], end: Optional[str]) -> Tuple[pd.Series, str]:
    if meter not in SignalsByMeter:
        raise HTTPException(404, f"Unknown meter: {meter}")
    csv_path = SignalsByMeter[meter].get(signal)
    if not csv_path:
        raise HTTPException(404, f"Signal {signal} not found for meter {meter}")

    st = pd.Timestamp(start).tz_localize(TIMEZONE) if start else None
    en = (pd.Timestamp(end).tz_localize(TIMEZONE) + pd.Timedelta(days=1)) if end else None
    rule_eff = _norm_rule(lod_rule_for_span(st, en, fallback=rule_query))

    s = _load_pyramid_series(csv_path, rule_eff)

    # Energy series: cumulative snapshots -> interval kWh if requested
    if is_energy_signal_key(signal):
        s = to_interval_energy_kwh(s if cumulative else s).fillna(0)

    if st is not None:
        s = s[s.index >= st]
    if en is not None:
        s = s[s.index <= en]
    return s, rule_eff

# ---------- endpoints ----------
@app.get("/series", response_model=SeriesOut)
def get_series(
    meter: str,
    signal: str = Query(...),  # allow any normalized name (incl. aux_m_*_ae)
    rule: str = "15min",
    cumulative: bool = True,
    start: Optional[str] = None,
    end: Optional[str] = None,
    max_points: int = Query(6000, ge=100, le=200000),
):
    s, rule_eff = _load_series_lod(meter, signal, rule, cumulative, start, end)
    s = downsample_for_json(s, max_points)
    return SeriesOut(
        meter=meter, signal=signal, rule=rule_eff,
        timestamps=[t.isoformat() for t in s.index],
        values=[float(x) for x in s.values],
    )

@app.get("/kpis", response_model=KpisOut)
def get_kpis(
    meter: str,
    rule: str = "15min",
    cumulative: bool = True,
    start: Optional[str] = None,
    end: Optional[str] = None,
):
    # totals computed from light LOD (1d if range set)
    rule_kpi = "1d" if (start or end) else "15min"

    def pick_energy(keys: List[str]) -> Optional[pd.Series]:
        for k in keys:
            p = SignalsByMeter.get(meter, {}).get(k)
            if p:
                return _load_pyramid_series(p, rule_kpi)
        return None

    # Try meter-style AE first, then BESS aux_m_* variants
    pos = pick_energy(["pos_ae", "aux_m_pos_ae"])
    neg = pick_energy(["neg_ae", "aux_m_neg_ae"])
    # com (combined) not strictly needed for totals but may be useful later
    # com = pick_energy(["com_ae", "aux_m_com_ae"])

    def total_cum(s: Optional[pd.Series]) -> Optional[float]:
        if s is None or s.empty:
            return None
        d = float(s.iloc[-1] - s.iloc[0])
        return d if d >= 0 else None

    def total_sum(s: Optional[pd.Series]) -> Optional[float]:
        if s is None or s.empty:
            return None
        return float(s.sum())

    total_import = total_cum(pos) if cumulative else total_sum(pos)
    total_export = total_cum(neg) if cumulative else total_sum(neg)
    net = (total_import - total_export) if (total_import is not None and total_export is not None) else None

    ap_p = SignalsByMeter.get(meter, {}).get("com_ap")
    pf_p = SignalsByMeter.get(meter, {}).get("pf")
    ap = _load_pyramid_series(ap_p, "15min") if ap_p else None
    pf = _load_pyramid_series(pf_p, "15min") if pf_p else None

    return KpisOut(
        meter=meter,
        total_import_kWh=total_import,
        total_export_kWh=total_export,
        net_kWh=net,
        peak_kW=float(ap.max()) if ap is not None and len(ap) else None,
        avg_pf=float(pf.mean()) if pf is not None and len(pf) else None,
    )

@app.get("/bundle", response_model=BundleOut)
def get_bundle(
    meter: str,
    signals: str = "com_ap,pf,pos_ae,neg_ae",
    rule: str = "15min",
    cumulative: bool = True,
    start: Optional[str] = None,
    end: Optional[str] = None,
    max_points: int = 6000,
):
    ks = get_kpis(meter, rule, cumulative, start, end)
    out = {}
    for sig in [s.strip() for s in signals.split(",") if s.strip()]:
        s, rule_eff = _load_series_lod(meter, sig, rule, cumulative, start, end)
        s = downsample_for_json(s, max_points)
        out[sig] = SeriesOut(
            meter=meter, signal=sig, rule=rule_eff,
            timestamps=[t.isoformat() for t in s.index],
            values=[float(x) for x in s.values],
        )
    return BundleOut(meter=meter, kpis=ks, series=out)

@app.get("/bess_kpis", response_model=BessKpisOut)
def bess_kpis(
    meter: str,
    rule: str = "15min",
    start: Optional[str] = None,
    end: Optional[str] = None,
):
    if meter not in SignalsByMeter:
        raise HTTPException(404, f"Unknown meter: {meter}")
    use_rule = "1h" if rule not in ("5min","15min") else rule

    def pick(keys: List[str]) -> Optional[pd.Series]:
        for k in keys:
            p = SignalsByMeter[meter].get(k)
            if p: return _load_pyramid_series(p, use_rule)
        return None

    # Common BESS signals (load if present)
    soc = pick(["bms1_soc","soc"])
    soh = pick(["bms1_soh","soh"])
    v   = pick(["bms1_v","pack_v","v"])
    c   = pick(["bms1_c","pack_c","c"])
    tavg= pick(["bms1_cell_ave_t","cell_avg_t","avg_t"])
    vmax= pick(["bms1_cell_max_v","cell_max_v"])
    vmin= pick(["bms1_cell_min_v","cell_min_v"])
    vdiff=pick(["bms1_cell_t_diff","bms1_cell_v_diff","cell_v_diff"])
    pcs_ap = pick(["pcs1_ap","pcs_ap","ap"])
    aux_ap = pick(["aux_m_ap","aux_ap"])
    env_t  = pick(["ac1_outside_t","dh1_temp","env_t"])
    alarm  = pick([
        "fa1_SmokeFlag","fa1_ErrCode","fa1_Level","fa1_Co","fa1_Voc",
        "fa2_SmokeFlag","fa2_ErrCode","fa2_Level","fa2_Co","fa2_Voc",
        "fa3_SmokeFlag","fa3_ErrCode","fa3_Level","fa3_Co","fa3_Voc",
        "fa4_SmokeFlag","fa4_ErrCode","fa4_Level","fa4_Co","fa4_Voc",
        "fa5_SmokeFlag","fa5_ErrCode","fa5_Level","fa5_Co","fa5_Voc",
        "fa1_alarm","fa1_flag"  # generic fallbacks
    ])

    st = pd.Timestamp(start).tz_localize(TIMEZONE) if start else None
    en = (pd.Timestamp(end).tz_localize(TIMEZONE) + pd.Timedelta(days=1)) if end else None

    def filt(s: Optional[pd.Series]):
        if s is None: return None
        if st is not None: s = s[s.index >= st]
        if en is not None: s = s[s.index <= en]
        return s

    soc, soh, v, c, tavg, vmax, vmin, vdiff, pcs_ap, aux_ap, env_t, alarm = map(
        filt, (soc, soh, v, c, tavg, vmax, vmin, vdiff, pcs_ap, aux_ap, env_t, alarm)
    )

    spread = None
    if vdiff is not None and len(vdiff):
        spread = float(vdiff.max())
    elif vmax is not None and vmin is not None and len(vmax) and len(vmin):
        spread = float((vmax - vmin).max())

    alarms_any = None
    if alarm is not None and len(alarm):
        alarms_any = bool((alarm.fillna(0) > 0).any())

    return BessKpisOut(
        meter=meter,
        soc_avg=float(soc.mean()) if soc is not None and len(soc) else None,
        soc_min=float(soc.min()) if soc is not None and len(soc) else None,
        soh_avg=float(soh.mean()) if soh is not None and len(soh) else None,
        pack_v_avg=float(v.mean()) if v is not None and len(v) else None,
        pack_c_avg=float(c.mean()) if c is not None and len(c) else None,
        cell_v_spread_max=spread,
        cell_t_avg=float(tavg.mean()) if tavg is not None and len(tavg) else None,
        pcs_ap_peak=float(pcs_ap.max()) if pcs_ap is not None and len(pcs_ap) else None,
        aux_ap_avg=float(aux_ap.mean()) if aux_ap is not None and len(aux_ap) else None,
        env_temp_avg=float(env_t.mean()) if env_t is not None and len(env_t) else None,
        alarms_any=alarms_any,
    )
