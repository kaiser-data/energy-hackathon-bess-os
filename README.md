# Smart Meter + BESS Analytics

A Python-based analytics platform for smart meter and Battery Energy Storage System (BESS) data visualization and analysis.

## Quick Start

### Prerequisites
- Python 3.11+ (recommended: 3.12)
- Virtual environment support

### 1. Setup Environment

```bash
# Clone and navigate to project
cd energy_hackathon_data

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Data Sources

```bash
# Set data paths (run from project root)
export METER_ROOTS="data/meter,data/BESS"
```

### 3. Preprocess Data

```bash
# Standard preprocessing
python preprocess_pyramids.py --workers 4 --rules "5min,15min,1h,1d"

# OPTIMIZED preprocessing (recommended for large datasets)
python preprocess_pyramids_optimized.py --workers 4 --chunk-size 50000 --rules "5min,15min,1h,1d"
```

### 4. Start Services

#### Terminal 1 - Backend API
```bash
source .venv/bin/activate
export METER_ROOTS="data/meter,data/BESS"

# Standard API
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 2 --reload

# OPTIMIZED API (recommended for large datasets)
uvicorn backend.main_optimized:app --host 0.0.0.0 --port 8000 --loop uvloop --no-access-log
```

#### Terminal 2 - Frontend Dashboard
```bash
source .venv/bin/activate
cd frontend
streamlit run app.py
```

### 5. Access Application

- **Frontend Dashboard**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health

## Features

### Data Sources
- **Smart Meters**: Active power, power factor, energy counters (m1-m6)
- **BESS Systems**: SOC, SOH, thermal data, PCS metrics, auxiliary energy

### Dashboards
- **Single Meter**: Power analysis, daily energy consumption
- **Compare Meters**: Side-by-side meter comparison
- **BESS Overview**: Battery system KPIs and telemetry

### Performance Features
- **Multi-level caching** with Parquet pyramids (5min, 15min, 1h, 1d)
- **Automatic LOD selection** based on time range
- **Chunked CSV processing** for memory efficiency
- **Parallel Parquet compression** with SNAPPY
- **Async I/O operations** with uvloop
- **Memory-mapped file access** for fast loading
- **LTTB downsampling** for large datasets
- **Intelligent caching** with TTL
- **Server-side data processing** to minimize network transfer

## Data Structure

```
data/
├── meter/           # Smart meter folders (m1, m2, etc.)
│   └── m1/
│       ├── file1.csv
│       └── file2.csv
└── BESS/            # Battery system folders
    └── ZHPESS.../
        ├── data1.csv
        └── data2.csv
```

## Performance Optimization

### For Large Datasets (>1GB)

1. **Use optimized preprocessing**:
```bash
pip install -r requirements_optimized.txt
python preprocess_pyramids_optimized.py --workers 8 --chunk-size 100000
```

2. **Use optimized API backend**:
```bash
uvicorn backend.main_optimized:app --loop uvloop --workers 1
```

3. **Tune environment variables**:
```bash
export CHUNK_SIZE=100000          # Larger chunks for big files
export PARQUET_THREADS=8          # More compression threads
export CACHE_TTL=600              # Longer cache TTL
export MAX_WORKERS=8              # More I/O workers
```

### Performance Benchmarking
```bash
# Quick benchmark
python benchmark_performance.py --quick

# Full benchmark (includes API testing)
python benchmark_performance.py
```

### Expected Performance Improvements
- **Preprocessing**: 3-5x faster, 50% less memory usage
- **API Response**: 2-3x faster with intelligent caching
- **Memory Usage**: 60-80% reduction with chunked processing
- **File I/O**: 4-6x faster with parallel compression

## Development

### Code Quality
```bash
# Format code
ruff check --fix .
black .

# Type checking
mypy backend

# Run tests
pytest
```

### Adding New Data Sources
1. Place CSV files in appropriate `data/meter/` or `data/BESS/` subfolder
2. Run preprocessing: `python preprocess_pyramids.py`
3. Restart backend to reload data index

## Troubleshooting

**No data showing**: Verify `METER_ROOTS` environment variable and run preprocessing

**Slow performance**: Reduce date ranges, lower max_points in UI, ensure Parquet cache exists

**API errors**: Check backend logs, verify data folder structure matches expected format

## Architecture

- **Backend**: FastAPI with automatic LOD selection and LTTB downsampling
- **Frontend**: Streamlit multi-page application
- **Data**: Pandas/NumPy processing with Parquet caching
- **Timezone**: Europe/Berlin (all timestamps localized)

## License

[Add your license information here]