#!/usr/bin/env python3
"""
Performance benchmarking script for energy analytics pipeline.

Compares original vs optimized implementations across different data sizes.
"""

import time
import psutil
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

class PerformanceBenchmark:
    def __init__(self):
        self.results = {}
        self.cache_dir = Path("backend/.meter_cache")

    def measure_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def measure_preprocessing_performance(self, data_size: str) -> Dict:
        """Benchmark preprocessing performance."""
        print(f"\n🔄 Benchmarking preprocessing ({data_size})...")

        results = {}

        # Original implementation
        print("  Testing original preprocessing...")
        start_time = time.time()
        start_memory = self.measure_memory_usage()

        try:
            result = subprocess.run([
                "python", "preprocess_pyramids.py",
                "--workers", "4"
            ], capture_output=True, text=True, timeout=300)

            end_time = time.time()
            end_memory = self.measure_memory_usage()

            results["original"] = {
                "time": end_time - start_time,
                "memory_peak": end_memory - start_memory,
                "success": result.returncode == 0,
                "output": result.stdout[-500:] if result.stdout else "",
                "error": result.stderr[-500:] if result.stderr else ""
            }
        except subprocess.TimeoutExpired:
            results["original"] = {
                "time": 300,
                "memory_peak": 0,
                "success": False,
                "output": "",
                "error": "Timeout after 5 minutes"
            }

        # Clean cache
        if self.cache_dir.exists():
            subprocess.run(["rm", "-rf", str(self.cache_dir)])

        # Optimized implementation
        print("  Testing optimized preprocessing...")
        start_time = time.time()
        start_memory = self.measure_memory_usage()

        try:
            result = subprocess.run([
                "python", "preprocess_pyramids_optimized.py",
                "--workers", "4",
                "--chunk-size", "50000"
            ], capture_output=True, text=True, timeout=300)

            end_time = time.time()
            end_memory = self.measure_memory_usage()

            results["optimized"] = {
                "time": end_time - start_time,
                "memory_peak": end_memory - start_memory,
                "success": result.returncode == 0,
                "output": result.stdout[-500:] if result.stdout else "",
                "error": result.stderr[-500:] if result.stderr else ""
            }
        except subprocess.TimeoutExpired:
            results["optimized"] = {
                "time": 300,
                "memory_peak": 0,
                "success": False,
                "output": "",
                "error": "Timeout after 5 minutes"
            }

        return results

    def measure_api_performance(self) -> Dict:
        """Benchmark API response times."""
        print("\n🚀 Benchmarking API performance...")

        import requests
        import threading
        import subprocess

        results = {}

        # Test both APIs
        for api_type, port in [("original", 8000), ("optimized", 8001)]:
            print(f"  Testing {api_type} API...")

            # Start the API server
            if api_type == "original":
                server_cmd = ["uvicorn", "backend.main:app", "--port", str(port)]
            else:
                server_cmd = ["uvicorn", "backend.main_optimized:app", "--port", str(port)]

            server_process = subprocess.Popen(server_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(5)  # Wait for server to start

            try:
                # Test endpoints
                base_url = f"http://localhost:{port}"

                # Test /meters endpoint
                start_time = time.time()
                response = requests.get(f"{base_url}/meters", timeout=30)
                meters_time = time.time() - start_time
                meters_success = response.status_code == 200

                if meters_success:
                    meters_data = response.json()
                    first_meter = list(meters_data.keys())[0] if meters_data else None

                    if first_meter:
                        signals = meters_data[first_meter][:3]  # Test first 3 signals

                        # Test /bundle endpoint
                        start_time = time.time()
                        response = requests.get(
                            f"{base_url}/bundle",
                            params={
                                "meter": first_meter,
                                "signals": ",".join(signals),
                                "max_points": 1000
                            },
                            timeout=30
                        )
                        bundle_time = time.time() - start_time
                        bundle_success = response.status_code == 200
                        bundle_size = len(response.content) if response.content else 0
                    else:
                        bundle_time = 0
                        bundle_success = False
                        bundle_size = 0
                else:
                    bundle_time = 0
                    bundle_success = False
                    bundle_size = 0

                results[api_type] = {
                    "meters_time": meters_time,
                    "meters_success": meters_success,
                    "bundle_time": bundle_time,
                    "bundle_success": bundle_success,
                    "bundle_size_kb": bundle_size / 1024
                }

            except Exception as e:
                results[api_type] = {
                    "meters_time": 0,
                    "meters_success": False,
                    "bundle_time": 0,
                    "bundle_success": False,
                    "bundle_size_kb": 0,
                    "error": str(e)
                }

            finally:
                # Stop the server
                server_process.terminate()
                server_process.wait()

        return results

    def analyze_data_sizes(self) -> Dict:
        """Analyze data sizes and file counts."""
        print("\n📊 Analyzing data characteristics...")

        import os
        from pathlib import Path

        roots = []
        meter_roots = os.environ.get("METER_ROOTS", "")
        if meter_roots:
            roots = [Path(p.strip()) for p in meter_roots.split(",") if p.strip()]
        else:
            roots = [Path("data/meter"), Path("data/BESS")]

        total_files = 0
        total_size_mb = 0
        folders = []

        for root in roots:
            if not root.exists():
                continue

            for folder in root.iterdir():
                if not folder.is_dir():
                    continue

                csv_files = list(folder.glob("*.csv"))
                if not csv_files:
                    continue

                folder_size = sum(f.stat().st_size for f in csv_files)
                folders.append({
                    "name": folder.name,
                    "files": len(csv_files),
                    "size_mb": folder_size / 1024 / 1024
                })

                total_files += len(csv_files)
                total_size_mb += folder_size / 1024 / 1024

        return {
            "total_files": total_files,
            "total_size_mb": total_size_mb,
            "folders": folders,
            "avg_file_size_mb": total_size_mb / total_files if total_files > 0 else 0
        }

    def generate_synthetic_data(self, size: str) -> Path:
        """Generate synthetic data for testing."""
        print(f"  Generating synthetic data ({size})...")

        synthetic_dir = Path("data_synthetic") / size
        synthetic_dir.mkdir(parents=True, exist_ok=True)

        # Define sizes
        sizes = {
            "small": {"folders": 2, "files_per_folder": 3, "rows_per_file": 1000},
            "medium": {"folders": 5, "files_per_folder": 10, "rows_per_file": 10000},
            "large": {"folders": 10, "files_per_folder": 20, "rows_per_file": 100000}
        }

        config = sizes[size]

        for folder_i in range(config["folders"]):
            folder_path = synthetic_dir / f"meter_{folder_i+1}"
            folder_path.mkdir(exist_ok=True)

            for file_i in range(config["files_per_folder"]):
                # Generate time series data
                start_date = pd.Timestamp("2023-01-01", tz="Europe/Berlin")
                periods = config["rows_per_file"]
                freq = "15min"

                timestamps = pd.date_range(start_date, periods=periods, freq=freq)

                # Generate realistic power data
                base_power = 50 + np.random.normal(0, 10, periods)
                daily_pattern = 20 * np.sin(2 * np.pi * np.arange(periods) / (24 * 4))  # 4 samples per hour
                noise = np.random.normal(0, 5, periods)
                power = np.maximum(0, base_power + daily_pattern + noise)

                df = pd.DataFrame({
                    "timestamp": timestamps,
                    "com_ap": power
                })

                file_path = folder_path / f"com_ap_{file_i+1}.csv"
                df.to_csv(file_path, index=False)

        return synthetic_dir

    def run_comprehensive_benchmark(self):
        """Run complete performance benchmark."""
        print("🏁 Starting comprehensive performance benchmark...")

        benchmark_results = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "memory_gb": psutil.virtual_memory().total / 1024**3,
                "python_version": subprocess.run(["python", "--version"], capture_output=True, text=True).stdout.strip()
            }
        }

        # Analyze existing data
        benchmark_results["data_analysis"] = self.analyze_data_sizes()

        # Test preprocessing performance
        benchmark_results["preprocessing"] = self.measure_preprocessing_performance("existing")

        # Test API performance (if data exists)
        if benchmark_results["data_analysis"]["total_files"] > 0:
            benchmark_results["api"] = self.measure_api_performance()

        # Calculate improvements
        self.calculate_improvements(benchmark_results)

        # Save results
        results_file = Path("benchmark_results.json")
        with open(results_file, "w") as f:
            json.dump(benchmark_results, f, indent=2, default=str)

        print(f"\n✅ Benchmark complete! Results saved to {results_file}")
        self.print_summary(benchmark_results)

        return benchmark_results

    def calculate_improvements(self, results: Dict):
        """Calculate performance improvements."""
        if "preprocessing" in results:
            prep = results["preprocessing"]
            if "original" in prep and "optimized" in prep:
                orig = prep["original"]
                opt = prep["optimized"]

                if orig["success"] and opt["success"]:
                    time_improvement = (orig["time"] - opt["time"]) / orig["time"] * 100
                    memory_improvement = (orig["memory_peak"] - opt["memory_peak"]) / orig["memory_peak"] * 100 if orig["memory_peak"] > 0 else 0

                    prep["improvements"] = {
                        "time_improvement_percent": time_improvement,
                        "memory_improvement_percent": memory_improvement
                    }

        if "api" in results:
            api = results["api"]
            if "original" in api and "optimized" in api:
                orig = api["original"]
                opt = api["optimized"]

                if orig["bundle_success"] and opt["bundle_success"]:
                    time_improvement = (orig["bundle_time"] - opt["bundle_time"]) / orig["bundle_time"] * 100

                    api["improvements"] = {
                        "api_time_improvement_percent": time_improvement
                    }

    def print_summary(self, results: Dict):
        """Print benchmark summary."""
        print("\n" + "="*60)
        print("📈 PERFORMANCE BENCHMARK SUMMARY")
        print("="*60)

        # Data analysis
        data = results.get("data_analysis", {})
        print(f"📁 Data: {data.get('total_files', 0)} files, {data.get('total_size_mb', 0):.1f} MB")

        # Preprocessing results
        if "preprocessing" in results:
            prep = results["preprocessing"]
            print(f"\n⚙️  PREPROCESSING PERFORMANCE:")

            if "original" in prep:
                orig = prep["original"]
                status = "✅" if orig["success"] else "❌"
                print(f"   Original:  {status} {orig['time']:.1f}s, {orig['memory_peak']:.1f} MB peak")

            if "optimized" in prep:
                opt = prep["optimized"]
                status = "✅" if opt["success"] else "❌"
                print(f"   Optimized: {status} {opt['time']:.1f}s, {opt['memory_peak']:.1f} MB peak")

            if "improvements" in prep:
                imp = prep["improvements"]
                print(f"   💡 Improvements: {imp['time_improvement_percent']:.1f}% faster, {imp['memory_improvement_percent']:.1f}% less memory")

        # API results
        if "api" in results:
            api = results["api"]
            print(f"\n🚀 API PERFORMANCE:")

            for api_type in ["original", "optimized"]:
                if api_type in api:
                    data = api[api_type]
                    bundle_status = "✅" if data["bundle_success"] else "❌"
                    print(f"   {api_type.capitalize()}: {bundle_status} {data['bundle_time']:.2f}s, {data['bundle_size_kb']:.1f} KB")

            if "improvements" in api:
                imp = api["improvements"]
                print(f"   💡 API Improvement: {imp['api_time_improvement_percent']:.1f}% faster")

        print("="*60)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark energy analytics performance")
    parser.add_argument("--quick", action="store_true", help="Quick benchmark (preprocessing only)")
    parser.add_argument("--api-only", action="store_true", help="API benchmark only")
    args = parser.parse_args()

    benchmark = PerformanceBenchmark()

    if args.api_only:
        results = benchmark.measure_api_performance()
        print(json.dumps(results, indent=2))
    elif args.quick:
        results = benchmark.measure_preprocessing_performance("existing")
        benchmark.calculate_improvements({"preprocessing": results})
        print(json.dumps(results, indent=2))
    else:
        benchmark.run_comprehensive_benchmark()