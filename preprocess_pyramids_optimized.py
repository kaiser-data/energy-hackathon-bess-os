#!/usr/bin/env python3
"""
OPTIMIZED Parquet pyramid preprocessing for massive energy datasets.

Performance improvements:
- Chunked CSV reading to reduce memory usage
- Parallel Parquet compression with thread pools
- Memory pooling for DataFrames
- Optimized datetime parsing
- Streaming aggregation for large files
- SNAPPY compression for faster I/O

ENV:
  METER_ROOTS="path1,path2"   # e.g. "data/meter,data/BESS"
  CHUNK_SIZE=50000            # Rows per chunk (default: 50000)
  PARQUET_THREADS=4           # Compression threads (default: 4)
"""

from __future__ import annotations
import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Iterator
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------- Configuration ----------------
TIMEZONE = "Europe/Berlin"
PYRAMID_RULES_DEFAULT = ["5min", "15min", "1h", "1d"]
CACHE_DIR = Path("backend/.meter_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Performance tuning
DEFAULT_CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "50000"))
DEFAULT_PARQUET_THREADS = int(os.environ.get("PARQUET_THREADS", "4"))

# Parquet compression settings for speed
PARQUET_COMPRESSION = "snappy"  # Faster than gzip, good compression
PARQUET_ROW_GROUP_SIZE = 100000  # Larger row groups for better compression

# ---------------- Memory Pool Management ----------------
class DataFramePool:
    """Reuse DataFrame objects to reduce allocation overhead."""
    def __init__(self, max_size: int = 100):
        self._pool: List[pd.DataFrame] = []
        self._max_size = max_size

    def get_dataframe(self, index, columns) -> pd.DataFrame:
        if self._pool:
            df = self._pool.pop()
            # Reset the DataFrame
            df.index = index
            df.columns = columns
            return df
        return pd.DataFrame(index=index, columns=columns)

    def return_dataframe(self, df: pd.DataFrame):
        if len(self._pool) < self._max_size:
            df.iloc[:] = np.nan  # Clear data
            self._pool.append(df)

_df_pool = DataFramePool()

# ---------------- Optimized CSV Processing ----------------
def _parse_datetime_optimized(series: pd.Series) -> pd.Series:
    """Optimized datetime parsing with format inference."""
    # Try common formats first
    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S']:
        try:
            return pd.to_datetime(series, format=fmt, utc=True)
        except:
            continue

    # Fallback to flexible parsing
    try:
        return pd.to_datetime(series, utc=True, infer_datetime_format=True)
    except:
        return pd.to_datetime(series, errors='coerce')

def _load_csv_chunked(csv_path: Path, chunk_size: int = DEFAULT_CHUNK_SIZE) -> Iterator[pd.DataFrame]:
    """Load CSV in chunks to reduce memory usage."""
    # Try pyarrow engine first (faster)
    try:
        reader = pd.read_csv(
            csv_path,
            chunksize=chunk_size,
            engine='pyarrow',
            dtype_backend='pyarrow'  # Use PyArrow dtypes for better performance
        )
    except Exception:
        # Fallback to pandas engine
        reader = pd.read_csv(csv_path, chunksize=chunk_size, engine='python')

    for chunk in reader:
        yield chunk

def _process_chunk(chunk: pd.DataFrame, csv_path: Path) -> pd.Series:
    """Process a single chunk and return time series."""
    # Find datetime column
    ts_candidates = [c for c in chunk.columns
                    if c.lower() in ("timestamp", "time", "date", "datetime", "ts")]
    ts_col = ts_candidates[0] if ts_candidates else chunk.columns[0]

    # Parse datetime efficiently
    chunk[ts_col] = _parse_datetime_optimized(chunk[ts_col])

    # Set index and sort
    chunk = chunk.set_index(ts_col).sort_index()

    # Timezone handling
    if chunk.index.tz is None:
        chunk.index = chunk.index.tz_localize(TIMEZONE, ambiguous="NaT", nonexistent="shift_forward")
    else:
        chunk.index = chunk.index.tz_convert(TIMEZONE)

    # Find numeric column
    numeric_col = None
    for col in chunk.columns:
        if col.lower() in ("timestamp", "time", "date", "datetime", "ts"):
            continue
        try:
            pd.to_numeric(chunk[col], errors='raise')
            numeric_col = col
            break
        except:
            continue

    if numeric_col is None:
        raise ValueError(f"No numeric column found in {csv_path.name}")

    # Convert to numeric and return series
    series = pd.to_numeric(chunk[numeric_col], errors='coerce').dropna().astype('float32')
    series.name = csv_path.stem
    return series

def _load_csv_series_optimized(csv_path: Path) -> pd.Series:
    """Load CSV as time series with memory optimization."""
    file_size = csv_path.stat().st_size

    # For small files (<10MB), load directly
    if file_size < 10 * 1024 * 1024:
        try:
            df = pd.read_csv(csv_path, engine='pyarrow')
            return _process_chunk(df, csv_path)
        except Exception:
            pass

    # For large files, use chunked processing
    series_parts = []
    total_rows = 0

    for chunk in _load_csv_chunked(csv_path):
        try:
            series_chunk = _process_chunk(chunk, csv_path)
            series_parts.append(series_chunk)
            total_rows += len(series_chunk)

            # Memory management: if we have too many parts, combine them
            if len(series_parts) > 10:
                combined = pd.concat(series_parts, axis=0)
                series_parts = [combined]

        except Exception as e:
            print(f"[WARN] Chunk processing error in {csv_path.name}: {e}")
            continue

    if not series_parts:
        raise ValueError(f"No valid data found in {csv_path.name}")

    # Combine all parts
    final_series = pd.concat(series_parts, axis=0).sort_index()
    final_series.name = csv_path.stem

    print(f"[INFO] Loaded {csv_path.name}: {total_rows:,} rows")
    return final_series

# ---------------- Optimized Aggregation ----------------
def _streaming_resample(series: pd.Series, rule: str, how: str) -> pd.Series:
    """Memory-efficient resampling for large series."""
    rule = rule.lower().strip()

    # For very large series, use chunked resampling
    if len(series) > 1_000_000:
        return _chunked_resample(series, rule, how)

    # Standard resampling for smaller series
    series = series.sort_index()
    if how == "sum":
        return series.resample(rule).sum(min_count=1)
    elif how == "last":
        return series.resample(rule).last()
    elif how == "max":
        return series.resample(rule).max()
    elif how == "min":
        return series.resample(rule).min()
    else:
        return series.resample(rule).mean()

def _chunked_resample(series: pd.Series, rule: str, how: str, chunk_periods: int = 100) -> pd.Series:
    """Resample very large series in chunks to reduce memory usage."""
    # Determine chunk size based on rule
    if rule.endswith('min'):
        freq_minutes = int(rule.replace('min', ''))
        chunk_size = chunk_periods * freq_minutes * 60  # seconds
    elif rule.endswith('h'):
        freq_hours = int(rule.replace('h', ''))
        chunk_size = chunk_periods * freq_hours * 3600
    elif rule.endswith('d'):
        freq_days = int(rule.replace('d', ''))
        chunk_size = chunk_periods * freq_days * 86400
    else:
        chunk_size = 86400  # 1 day default

    # Split series into time chunks
    start_time = series.index.min()
    end_time = series.index.max()

    resampled_parts = []
    current_time = start_time

    while current_time < end_time:
        next_time = current_time + pd.Timedelta(seconds=chunk_size)
        chunk_mask = (series.index >= current_time) & (series.index < next_time)
        chunk = series[chunk_mask]

        if len(chunk) > 0:
            chunk_resampled = _streaming_resample(chunk, rule, how)
            resampled_parts.append(chunk_resampled)

        current_time = next_time

    return pd.concat(resampled_parts, axis=0).sort_index()

# ---------------- Optimized Parquet I/O ----------------
def _write_parquet_optimized(df: pd.DataFrame, path: Path, compression: str = PARQUET_COMPRESSION):
    """Write Parquet with optimized settings for speed and compression."""
    table = pa.Table.from_pandas(df, preserve_index=True)

    pq.write_table(
        table,
        path,
        compression=compression,
        row_group_size=PARQUET_ROW_GROUP_SIZE,
        use_dictionary=True,  # Better compression for repeated values
        write_statistics=True,  # Enable column statistics for faster queries
    )

# ---------------- Enhanced BESS Signal Detection ----------------
def agg_for_bess_signal(sig: str) -> str:
    """Optimized BESS signal aggregation detection."""
    s = sig.lower()

    # Energy counters (cumulative snapshots)
    if s.endswith(("_com_ae", "_pos_ae", "_neg_ae")):
        return "last"

    # Alarms/flags (case-insensitive patterns)
    alarm_patterns = ["fa", "smokeflag", "errcode", "_flag", "_alarm", "_err"]
    if any(pattern in s for pattern in alarm_patterns):
        return "max"

    # Extremes/spreads
    if any(pattern in s for pattern in ["_max_", "_max", "_diff", "_t_diff", "_v_diff"]):
        return "max"
    if any(pattern in s for pattern in ["_min_", "_min"]):
        return "min"

    # Analog telemetry (most common case)
    analog_patterns = ["_ap", "_pf", "_i", "_v", "_c", "_t", "temp", "soc", "soh",
                      "uab", "ubc", "uca", "dcc", "t_env", "t_igbt", "t_a"]
    if any(pattern in s for pattern in analog_patterns):
        return "mean"

    return "mean"  # Default for analog telemetry

# ---------------- Optimized File Processing ----------------
def _build_pyramid_for_file_optimized(csv_path: Path, rules: List[str]) -> Dict[str, str]:
    """Build pyramid with optimizations."""
    sig = hashlib.md5(f"{csv_path.resolve()}::{csv_path.stat().st_size}::{csv_path.stat().st_mtime}".encode()).hexdigest()
    out: Dict[str, str] = {}

    # Check if all pyramid files exist
    pyramid_paths = {rule: CACHE_DIR / f"{sig}__{rule.lower()}.parquet" for rule in rules}
    if all(path.exists() for path in pyramid_paths.values()):
        return {rule: str(path) for rule, path in pyramid_paths.items()}

    print(f"[INFO] Processing {csv_path.name}...")

    # Load series with optimization
    try:
        series = _load_csv_series_optimized(csv_path)
    except Exception as e:
        print(f"[ERROR] Failed to load {csv_path.name}: {e}")
        return {}

    # Detect signal type and aggregation method
    is_bess = "/bess/" in str(csv_path).lower() or "zhpess" in str(csv_path).lower()
    sig_name = csv_path.stem.lower()

    # Handle duplicates efficiently
    if series.index.has_duplicates:
        if is_bess:
            how = agg_for_bess_signal(sig_name)
        else:
            how = "mean" if sig_name in ("com_ap", "pf") else "last"

        if how == "mean":
            series = series.groupby(level=0).mean()
        elif how == "last":
            series = series.groupby(level=0).last()
        elif how == "max":
            series = series.groupby(level=0).max()
        elif how == "min":
            series = series.groupby(level=0).min()
        else:
            series = series.groupby(level=0).mean()

    # Build pyramid levels with parallel I/O
    def write_level(rule: str):
        if is_bess:
            how = agg_for_bess_signal(sig_name)
        else:
            how = "mean" if sig_name in ("com_ap", "pf") else "last"

        resampled = _streaming_resample(series, rule, how)
        df = resampled.to_frame("value")
        path = pyramid_paths[rule]
        _write_parquet_optimized(df, path)
        return rule, str(path)

    # Use thread pool for parallel Parquet writing
    with ThreadPoolExecutor(max_workers=DEFAULT_PARQUET_THREADS) as executor:
        results = list(executor.map(write_level, rules))

    out = dict(results)
    print(f"[INFO] Completed {csv_path.name}: {len(out)} levels")
    return out

# ---------------- Worker Function ----------------
def _worker_optimized(args: Tuple[Path, List[str], str]) -> Dict:
    """Optimized worker function with better error handling."""
    folder, rules, tz = args

    # Set timezone for worker
    global TIMEZONE
    TIMEZONE = tz

    results = {"folder": str(folder), "processed": [], "errors": []}
    csv_files = sorted(folder.glob("*.csv"))

    print(f"[INFO] Worker processing {folder.name}: {len(csv_files)} CSV files")

    for i, csv_path in enumerate(csv_files):
        try:
            print(f"[INFO] {folder.name}: {i+1}/{len(csv_files)} - {csv_path.name}")
            pyramids = _build_pyramid_for_file_optimized(csv_path, rules)

            if pyramids:
                results["processed"].append({
                    "csv": str(csv_path),
                    "parquets": pyramids
                })
            else:
                results["errors"].append({
                    "csv": str(csv_path),
                    "error": "No pyramids created"
                })

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"[ERROR] {csv_path.name}: {error_msg}")
            results["errors"].append({
                "csv": str(csv_path),
                "error": error_msg
            })

    print(f"[INFO] Worker {folder.name} complete: {len(results['processed'])} OK, {len(results['errors'])} errors")
    return results

# ---------------- Discovery Functions ----------------
def _discover_meter_folders(roots: List[Path]) -> List[Path]:
    """Discover folders containing CSV files."""
    folders = []
    excluded_dirs = {".git", ".venv", "__pycache__", "zipped", ".meter_cache"}

    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            # Filter out excluded directories
            dirnames[:] = [d for d in dirnames if d not in excluded_dirs]

            # Check if directory contains CSV files
            csv_count = sum(1 for f in filenames if f.endswith(".csv"))
            if csv_count > 0:
                folder_path = Path(dirpath)
                folders.append(folder_path)
                print(f"[INFO] Found folder: {folder_path} ({csv_count} CSV files)")

    return sorted(set(folders))

def _parse_roots() -> List[Path]:
    """Parse METER_ROOTS environment variable."""
    raw = os.environ.get("METER_ROOTS", "").strip()
    if raw:
        roots = [Path(p.strip()) for p in raw.split(",") if p.strip()]
    else:
        roots = [Path("../data/meter"), Path("../data/BESS"), Path("./meter"), Path("./BESS")]

    existing_roots = [r for r in roots if r.exists()]
    for root in existing_roots:
        print(f"[INFO] Data root: {root} ({sum(1 for _ in root.rglob('*.csv'))} CSV files)")

    return existing_roots

# ---------------- Main Function ----------------
def main():
    # Access global variables
    global TIMEZONE, DEFAULT_CHUNK_SIZE, DEFAULT_PARQUET_THREADS

    ap = argparse.ArgumentParser(description="OPTIMIZED Parquet pyramid preprocessing for energy data.")
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 1),
                   help="Parallel workers (default: CPU cores - 1)")
    ap.add_argument("--rules", type=str, default="5min,15min,1h,1d",
                   help="LOD rules (comma-separated, lowercase)")
    ap.add_argument("--timezone", type=str, default="Europe/Berlin",
                   help="IANA timezone")
    ap.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                   help="CSV chunk size for large files")
    ap.add_argument("--parquet-threads", type=int, default=DEFAULT_PARQUET_THREADS,
                   help="Parquet compression threads")
    ap.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = ap.parse_args()

    # Update global configuration
    TIMEZONE = args.timezone
    DEFAULT_CHUNK_SIZE = args.chunk_size
    DEFAULT_PARQUET_THREADS = args.parquet_threads

    # Parse rules and roots
    rules = [r.strip().lower() for r in args.rules.split(",") if r.strip()]
    roots = _parse_roots()

    if not roots:
        print("[ERROR] No data roots found. Set METER_ROOTS or create data directories.")
        sys.exit(2)

    print(f"\n[CONFIG] Settings:")
    print(f"  Roots: {', '.join(map(str, roots))}")
    print(f"  Rules: {rules}")
    print(f"  Workers: {args.workers}")
    print(f"  Chunk size: {args.chunk_size:,}")
    print(f"  Parquet threads: {args.parquet_threads}")
    print(f"  Cache dir: {CACHE_DIR.resolve()}")
    print(f"  Timezone: {TIMEZONE}")
    print(f"  Compression: {PARQUET_COMPRESSION}")

    # Discover folders
    folders = _discover_meter_folders(roots)
    if not folders:
        print("[WARN] No folders containing CSV files found.")
        sys.exit(0)

    total_csv_files = sum(len(list(folder.glob("*.csv"))) for folder in folders)
    print(f"\n[INFO] Processing {len(folders)} folders with {total_csv_files:,} total CSV files...")

    # Process folders
    tasks = [(folder, rules, TIMEZONE) for folder in folders]
    all_results: List[Dict] = []

    if args.workers == 1:
        # Single-threaded processing
        for task in tasks:
            result = _worker_optimized(task)
            all_results.append(result)
    else:
        # Multi-threaded processing
        with Pool(processes=args.workers) as pool:
            try:
                for i, result in enumerate(pool.imap_unordered(_worker_optimized, tasks)):
                    all_results.append(result)
                    progress = (i + 1) / len(tasks) * 100
                    print(f"[PROGRESS] {progress:.1f}% complete ({i+1}/{len(tasks)} folders)")
            except KeyboardInterrupt:
                print("\n[INFO] Interrupted by user. Saving partial results...")
                pool.terminate()
                pool.join()

    # Generate manifest
    manifest = {
        "roots": [str(r) for r in roots],
        "rules": rules,
        "timezone": TIMEZONE,
        "config": {
            "chunk_size": args.chunk_size,
            "parquet_threads": args.parquet_threads,
            "compression": PARQUET_COMPRESSION,
            "workers": args.workers
        },
        "folders": all_results,
        "summary": {
            "total_folders": len(folders),
            "total_csv_files": total_csv_files,
            "processed_files": sum(len(f["processed"]) for f in all_results),
            "error_files": sum(len(f["errors"]) for f in all_results)
        }
    }

    manifest_path = CACHE_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Summary
    total_processed = manifest["summary"]["processed_files"]
    total_errors = manifest["summary"]["error_files"]
    success_rate = (total_processed / total_csv_files * 100) if total_csv_files > 0 else 0

    print(f"\n[COMPLETE] Pyramid build finished!")
    print(f"  CSV files: {total_csv_files:,}")
    print(f"  Processed: {total_processed:,}")
    print(f"  Errors: {total_errors:,}")
    print(f"  Success rate: {success_rate:.1f}%")
    print(f"  Manifest: {manifest_path}")

    if total_errors > 0:
        print(f"\n[INFO] Check manifest.json for error details.")

if __name__ == "__main__":
    main()