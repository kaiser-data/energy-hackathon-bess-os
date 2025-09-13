#!/usr/bin/env python3
"""
Precompute multi-resolution Parquet pyramids for ALL smart-meter & BESS folders.

ENV:
  METER_ROOTS="path1,path2"   # e.g. "data/meter,data/BESS"

Defaults (if METER_ROOTS not set):
  ../data/meter, ../data/BESS, ./meter, ./BESS

Output:
  backend/.meter_cache/<sig>__5min.parquet
  backend/.meter_cache/<sig>__15min.parquet
  backend/.meter_cache/<sig>__1h.parquet
  backend/.meter_cache/<sig>__1d.parquet

Aggregation rules:
- METERS:
    com_ap, pf      -> MEAN
    com_ae, pos_ae, neg_ae -> LAST snapshot (intervals computed later via diff)
- BESS:
    *_max_* or *_diff -> MAX (preserve worst-case)
    *_min_*           -> MIN
    flags/alarms      -> MAX (any activation)
    power/pf, volts, current, temps, soc/soh -> MEAN
"""

from __future__ import annotations
import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

# ---------------- Configuration ----------------
TIMEZONE = "Europe/Berlin"
PYRAMID_RULES_DEFAULT = ["5min", "15min", "1h", "1d"]  # lowercase to avoid FutureWarning
CACHE_DIR = Path("backend/.meter_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Helpers ----------------
def _norm_rule(rule: str) -> str:
    return rule.lower().strip()

def _parse_roots() -> List[Path]:
    raw = os.environ.get("METER_ROOTS", "").strip()
    if raw:
        roots = [Path(p.strip()) for p in raw.split(",") if p.strip()]
    else:
        roots = [Path("../data/meter"), Path("../data/BESS"), Path("./meter"), Path("./BESS")]
    return [r for r in roots if r.exists()]

def _file_sig(path: Path) -> str:
    st = path.stat()
    return hashlib.md5(f"{path.resolve()}::{st.st_size}::{st.st_mtime}".encode()).hexdigest()

def _maybe_parse_datetime_col(df: pd.DataFrame) -> pd.DataFrame:
    cand = [c for c in df.columns if c.lower() in ("timestamp", "time", "date", "datetime", "ts")]
    idx = cand[0] if cand else df.columns[0]
    df = df.copy()
    df[idx] = pd.to_datetime(df[idx], utc=True, errors="coerce")
    if df[idx].isna().all():
        df[idx] = pd.to_datetime(df[idx], errors="coerce")
    df = df.set_index(idx).sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize(TIMEZONE, ambiguous="NaT", nonexistent="shift_forward")
    else:
        df.index = df.index.tz_convert(TIMEZONE)
    return df

def _pick_value_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if c.lower() in ("timestamp", "time", "date", "datetime", "ts"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    for c in df.columns:
        if c.lower() in ("timestamp", "time", "date", "datetime", "ts"):
            continue
        try:
            pd.to_numeric(df[c], errors="raise")
            return c
        except Exception:
            pass
    raise ValueError("No numeric data column found in CSV")

def _load_csv_series(csv_path: Path) -> pd.Series:
    read_kwargs = {}
    try:
        import pyarrow  # noqa: F401
        read_kwargs["engine"] = "pyarrow"
    except Exception:
        pass
    df = pd.read_csv(csv_path, **read_kwargs)
    df = _maybe_parse_datetime_col(df)
    v = _pick_value_column(df)
    s = pd.to_numeric(df[v], errors="coerce").dropna().astype("float32")
    s.name = csv_path.stem
    return s

def _fix_dups(s: pd.Series, how: str) -> pd.Series:
    if not s.index.has_duplicates:
        return s.sort_index()
    if how == "mean":
        s = s.groupby(level=0).mean()
    elif how == "last":
        s = s.groupby(level=0).last()
    elif how == "sum":
        s = s.groupby(level=0).sum(min_count=1)
    elif how == "max":
        s = s.groupby(level=0).max()
    elif how == "min":
        s = s.groupby(level=0).min()
    return s.sort_index()

def _resample(s: pd.Series, rule: str, how: str) -> pd.Series:
    rule = _norm_rule(rule)
    s = s.sort_index()
    if   how == "sum":  return s.resample(rule).sum(min_count=1)
    elif how == "last": return s.resample(rule).last()
    elif how == "max":  return s.resample(rule).max()
    elif how == "min":  return s.resample(rule).min()
    else:               return s.resample(rule).mean()

def is_bess_path(p: Path) -> bool:
    s = str(p).lower()
    return "/bess/" in s or "zhpess" in s

def agg_for_bess_signal(sig: str) -> str:
    s = sig.lower()

    # --- ENERGY COUNTERS (cumulative snapshots) ---
    # e.g., aux_m_com_ae, aux_m_pos_ae, aux_m_neg_ae (and any *_com_ae etc.)
    if s.endswith("_com_ae") or s.endswith("_pos_ae") or s.endswith("_neg_ae"):
        return "last"

    # --- ALARMS / FLAGS / ERRORCODES ---
    # Your files include CamelCase like fa1_SmokeFlag, fa1_ErrCode, fa1_Level, fa1_Co, fa1_Voc
    # We'll match case-insensitively via lowercase string:
    if s.startswith("fa") or "smokeflag" in s or "errcode" in s or s.endswith("_flag") or "_alarm" in s:
        return "max"  # any activation should survive downsampling

    # --- EXTREMES / SPREADS ---
    if "_max_" in s or s.endswith("_max") or s.endswith("_diff") or "_t_diff" in s or "_v_diff" in s:
        return "max"
    if "_min_" in s or s.endswith("_min"):
        return "min"

    # --- ANALOG FAMILIES (means) ---
    # Apparent power / PF / currents / voltages / temps / SOC / SOH
    if s.endswith("_ap") or s.endswith("_pf"):
        return "mean"
    if s.endswith("_i"):   # currents like aux_m_i
        return "mean"
    if s.endswith("_v") or "_uab" in s or "_ubc" in s or "_uca" in s:  # pack/line voltages
        return "mean"
    if s.endswith("_c") or "dcc" in s:  # currents (pcs1_dcc)
        return "mean"
    if s.endswith("_t") or "temp" in s or "t_env" in s or "t_igbt" in s or "t_a" in s:
        return "mean"
    if "soc" in s or "soh" in s:
        return "mean"

    # Per-pack / per-cell files (bms1_pX_vN, bms1_pX_tN): treat as analogs → mean
    # (covered by _v / _t suffix above)

    # Default conservative choice for analog telemetry
    return "mean"


def detect_signal_generic(csv_path: Path) -> str:
    """Return a normalized signal key. For meters, map to known keys; for BESS, keep stem."""
    name = csv_path.stem
    low = name.lower()
    # meter knowns
    if "com_ap" in low: return "com_ap"
    if "com_ae" in low: return "com_ae"
    if "pos_ae" in low: return "pos_ae"
    if "neg_ae" in low: return "neg_ae"
    if low.endswith("_pf") or low == "pf": return "pf"
    # BESS & others: keep file stem as-is
    return name

def _pyramid_path(sig: str, rule: str) -> Path:
    return CACHE_DIR / f"{sig}__{_norm_rule(rule)}.parquet"

def _build_pyramid_for_file(csv_path: Path, rules: List[str]) -> Dict[str, str]:
    sig = _file_sig(csv_path)
    out: Dict[str, str] = {}
    if all(_pyramid_path(sig, r).exists() for r in rules):
        for r in rules:
            out[r] = str(_pyramid_path(sig, r))
        return out

    s = _load_csv_series(csv_path)
    is_bess = is_bess_path(csv_path)
    sig_name = detect_signal_generic(csv_path)

    # de-dup at ingest
    if not is_bess:
        s = _fix_dups(s, "mean" if sig_name in ("com_ap", "pf") else "last")
    else:
        how = agg_for_bess_signal(sig_name)
        s = _fix_dups(s, how if how in ("mean", "last", "sum", "max", "min") else "mean")

    # build pyramid levels
    for rule in rules:
        if not is_bess:
            res = _resample(s, rule, "mean" if sig_name in ("com_ap", "pf") else "last")
        else:
            how = agg_for_bess_signal(sig_name)
            res = _resample(s, rule, how)
        df = res.to_frame("value")
        df.to_parquet(_pyramid_path(sig, rule))
        out[rule] = str(_pyramid_path(sig, rule))
    return out

def _discover_meter_folders(roots: List[Path]) -> List[Path]:
    out = []
    for r in roots:
        for dirpath, dirnames, filenames in os.walk(r):
            dirnames[:] = [d for d in dirnames if d not in (".git", ".venv", "__pycache__", "zipped")]
            if any(f.endswith(".csv") for f in filenames):
                out.append(Path(dirpath))
    return sorted(set(out))

def _match_csvs(folder: Path) -> List[Path]:
    # keep every CSV — meters & BESS alike
    return sorted(folder.glob("*.csv"))

def _worker(args: Tuple[Path, List[str], str]) -> Dict:
    folder, rules, tz = args
    # Set module-level TZ for the worker process
    global TIMEZONE
    TIMEZONE = tz

    results = {"folder": str(folder), "processed": [], "errors": []}
    for csv_path in _match_csvs(folder):
        try:
            out = _build_pyramid_for_file(csv_path, rules)
            results["processed"].append({"csv": str(csv_path), "parquets": out})
        except Exception as e:
            results["errors"].append({"csv": str(csv_path), "error": repr(e)})
    return results

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Precompute Parquet pyramids for smart meter & BESS CSVs.")
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 1), help="Parallel workers")
    ap.add_argument("--rules", type=str, default="5min,15min,1h,1d", help="LOD rules (comma-separated; use lowercase)")
    ap.add_argument("--timezone", type=str, default="Europe/Berlin", help="IANA timezone (default Europe/Berlin)")
    args = ap.parse_args()

    # assign to module-level for helpers
    global TIMEZONE
    TIMEZONE = args.timezone

    rules = [_norm_rule(r) for r in args.rules.split(",") if r.strip()]
    roots = _parse_roots()
    if not roots:
        print("No roots found. Set METER_ROOTS or create ./meter ./BESS or ../data/*")
        sys.exit(2)

    print(f"[INFO] Roots: {', '.join(map(str, roots))}")
    print(f"[INFO] Rules: {rules}")
    print(f"[INFO] Cache dir: {CACHE_DIR.resolve()}")
    print(f"[INFO] Timezone: {TIMEZONE}")

    folders = _discover_meter_folders(roots)
    if not folders:
        print("[WARN] No folders containing CSVs.")
        sys.exit(0)
    print(f"[INFO] Discovered {len(folders)} folders.")

    tasks = [(f, rules, TIMEZONE) for f in folders]
    all_results: List[Dict] = []
    if args.workers == 1:
        for t in tasks:
            all_results.append(_worker(t))
    else:
        with Pool(processes=args.workers) as pool:
            for res in pool.imap_unordered(_worker, tasks):
                all_results.append(res)

    manifest = {
        "roots": [str(r) for r in roots],
        "rules": rules,
        "timezone": TIMEZONE,
        "folders": all_results,
    }
    (CACHE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    total_csv = sum(len(f["processed"]) + len(f["errors"]) for f in all_results)
    total_ok = sum(len(f["processed"]) for f in all_results)
    total_err = sum(len(f["errors"]) for f in all_results)
    print(f"[OK] Pyramid build complete. CSVs seen: {total_csv}, OK: {total_ok}, ERR: {total_err}")
    print(f"[OK] Manifest: {CACHE_DIR / 'manifest.json'}")

if __name__ == "__main__":
    main()
