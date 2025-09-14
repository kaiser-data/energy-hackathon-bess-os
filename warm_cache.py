#!/usr/bin/env python3
"""
Cache Warming Script for PackPulse
Pre-computes and caches both demo mode (5 cells) and complete (260 cells) data
for all BESS systems to ensure instant loading.
"""

import requests
import time
import json
from pathlib import Path

# API endpoint
API_BASE = "http://127.0.0.1:8000"

# BESS systems to warm
BESS_SYSTEMS = [
    "ZHPESS232A230002",
    "ZHPESS232A230003",
    "ZHPESS232A230007"
]

def warm_cache_for_system(system: str):
    """Warm cache for both demo and complete modes for a BESS system"""

    print(f"\n{'='*60}")
    print(f"Warming cache for {system}")
    print(f"{'='*60}")

    # Warm demo mode cache (5 cells - instant)
    print(f"⚡ Warming DEMO mode cache (5 cells)...")
    start_time = time.time()
    try:
        response = requests.get(
            f"{API_BASE}/cell/system/{system}/real-sat-voltage",
            params={"demo_mode": "true"},
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            cells = len(data.get("degradation_3d", {}))
            elapsed = time.time() - start_time
            print(f"✅ Demo cache warmed: {cells} cells in {elapsed:.2f}s")
        else:
            print(f"❌ Demo cache warming failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Demo cache warming error: {e}")

    # Warm complete mode cache (260 cells - slower)
    print(f"📊 Warming COMPLETE mode cache (260 cells)...")
    start_time = time.time()
    try:
        response = requests.get(
            f"{API_BASE}/cell/system/{system}/real-sat-voltage",
            params={"demo_mode": "false"},
            timeout=120  # Longer timeout for complete analysis
        )
        if response.status_code == 200:
            data = response.json()
            cells = len(data.get("degradation_3d", {}))
            elapsed = time.time() - start_time
            print(f"✅ Complete cache warmed: {cells} cells in {elapsed:.2f}s")
        else:
            print(f"❌ Complete cache warming failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Complete cache warming error: {e}")

def check_cache_status():
    """Check what's already cached"""
    cache_dir = Path("backend/.demo_cache")
    if not cache_dir.exists():
        print("No cache directory found")
        return

    print("\n📁 Current cache status:")
    cache_files = list(cache_dir.glob("*.json"))

    if not cache_files:
        print("  No cached files found")
        return

    for cache_file in sorted(cache_files):
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                mode = data.get('mode', 'unknown')
                cells = data.get('cells_processed', 0)
                days = data.get('days_analyzed', 0)
                print(f"  ✅ {cache_file.name}: {mode} mode, {cells} cells, {days} days")
        except:
            print(f"  ❌ {cache_file.name}: invalid or corrupted")

def main():
    print("🔥 PackPulse Cache Warming Script")
    print("=" * 60)

    # Check current status
    check_cache_status()

    # Check if API is running
    print("\n🔍 Checking API availability...")
    try:
        response = requests.get(f"{API_BASE}/", timeout=5)
        if response.status_code == 200:
            print("✅ API is running")
        else:
            print("❌ API returned unexpected status")
            return
    except:
        print("❌ API is not accessible. Please start the backend first.")
        print("   Run: uvicorn backend.main_optimized:app --host 127.0.0.1 --port 8000")
        return

    # Warm caches for all systems
    total_start = time.time()
    for system in BESS_SYSTEMS:
        warm_cache_for_system(system)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"✅ Cache warming complete in {total_elapsed:.2f}s")
    print(f"{'='*60}")

    # Show final status
    check_cache_status()

if __name__ == "__main__":
    main()