Here’s a complete, ready-to-drop-in **`code_cvlaude.md`** you can commit at the project root. It captures architecture, conventions, performance patterns, and “how to extend” checklists tailored to your Smart Meter + BESS stack.

---

# Code & Architecture Guidelines — Smart Meter + BESS Analytics

**Stack:** Python 3.12 • Pandas/NumPy • FastAPI • Uvicorn • Parquet (pyarrow) • Streamlit • Plotly
**Timezone:** Europe/Berlin
**Data Sources:** `data/meter/*`, `data/BESS/*` (CSV folders)

## 1) Project Layout

```
energy_hackathon_data/
├── backend/
│   └── main.py                 # FastAPI service (LOD, bundle endpoints, BESS KPIs)
├── frontend/
│   ├── app.py                  # Streamlit multipage router
│   └── pages/
│       ├── single_api.py       # Single meter dashboard
│       ├── compare_api.py      # Two-meters compare
│       └── bess_api.py         # BESS-specific KPIs/plots
├── preprocess_pyramids.py      # Precompute Parquet LOD pyramids
├── backend/.meter_cache/       # Parquet cache (generated)
├── data/
│   ├── meter/…                 # m1..m6 folders (CSV)
│   └── BESS/…                  # ZHPESS… folders (CSV)
├── code_cvlaude.md             # This document
├── requirements.txt / pyproject.toml
└── .venv/                      # Local virtualenv
```

## 2) Environment & Paths

* Set **roots** where CSV folders live:

  ```bash
  export METER_ROOTS="data/meter,data/BESS"
  ```
* Always run under **Europe/Berlin**. All code converts or localizes timestamps accordingly.
* Avoid relative path surprises. Run commands from repository root.

## 3) Data Model & Signal Semantics

### Smart meters

* `com_ap` — Active power (kW). **Mean** over time windows.
* `pf` — Power factor. **Mean**.
* `pos_ae`, `neg_ae`, `com_ae` — Energy counters (kWh). **Cumulative snapshots**. Plot/aggregate as **intervals** via `diff()` with reset protection.

### BESS

* `bms1_*`, `pcs1_*`, `ac1_*`, `dh1_*` etc. are **telemetry** → **Mean**.
* Extremes/spreads: `*_max_*`, `*_min_*`, `*_diff` → **Max/Min** as appropriate to preserve peaks.
* Alarms/flags: `fa*`, `*Flag`, `*ErrCode` → **Max** (any activation).
* Auxiliary energy: `aux_m_com_ae`, `aux_m_pos_ae`, `aux_m_neg_ae` are **cumulative** (same treatment as meter AE).

> Rule of thumb: When in doubt, analog → mean; counters → last snapshot; alarms → max; min-typed → min.

## 4) Preprocessing (Parquet Pyramids)

**Goal:** Transform raw CSVs into **Level-of-Detail (LOD)** Parquet series to keep UI/API fast.

* Default LODs (lowercase to avoid warnings): `["5min","15min","1h","1d"]`.
* Per-file **signature** = `md5(path :: size :: mtime)` → stable cache key.
* Output: `backend/.meter_cache/<sig>__{rule}.parquet` (single column `value`).
* De-duplication: group by timestamp using appropriate aggregator per signal kind.
* For cumulative energy, store **last** sample per bucket; compute intervals at query time.

**Run:**

```bash
source .venv/bin/activate
export METER_ROOTS="data/meter,data/BESS"
python preprocess_pyramids.py --workers 4 --rules "5min,15min,1h,1d"
```

**Gotchas:**

* Mixed headers: first column must parse into a datetime (`timestamp|time|date|datetime|ts|first_col`).
* Coerce non-numeric columns; drop NaNs post-parse.
* Timezones: localize/convert to Europe/Berlin uniformly.

## 5) Backend (FastAPI)

### Principles

* **LOD by span**: Choose `"5min"`, `"15min"`, or `"1h"` automatically via date range; default `"15min"`.
* **Bundle** endpoint to reduce HTTP roundtrips: KPIs + multiple series in one response.
* **Server-side downsampling** (LTTB) caps payload size (`max_points` default 6k).

### Key endpoints

* `GET /meters` → list of folders discovered under `METER_ROOTS`.
* `GET /bundle` → `{ kpis, series{signal→(t,v)}, rule }` for multiple signals.
* `GET /series` → one signal (LOD+downsampled).
* `GET /kpis` → totals & peaks from light LOD (`1d` if range specified).
* `GET /bess_kpis` → SOC/SOH/thermal/PCS/AUX/env/alarms.
* Utility: `/reload`, `/prewarm`, `/health`, root index.

### Energy intervals

* Any key that **ends with** `"_pos_ae"|" _neg_ae"|" _com_ae"` (including `aux_m_*_ae`) is treated as **cumulative**.
* Runtime conversion to intervals: `.diff()` + non-negativity + 99.9th percentile cap.

### Performance notes

* Prefer reading precomputed Parquet; avoid CSV at request time.
* LTTB downsampling done after LOD selection.
* Use `--workers 2`+ for Uvicorn in local dev; scale by CPU in prod.

## 6) Frontend (Streamlit + Plotly)

### Pages

* **Single Meter** — power, PF, daily energy; selectable signals (incl. `aux_m_*_ae`).
* **Compare** — 2 meters; overlay power/PF; grouped daily energy.
* **BESS Overview** — KPIs (SOC/SOH/PCS/AUX/environment/alarms) + quick series viewers.

### UX performance

* Single call to **/bundle** per page interaction.
* Use Plotly `render_mode="webgl"` for large lines.
* Streamlit `@st.cache_data(ttl=60)` around API calls.
* Limit `max_points` via UI (default 6000; range 2000–20000).

### Color/key legend conventions

* Power (kW): line plot.
* PF: line plot with fixed y-range `[0,1.05]`.
* Daily energy: grouped bars; Types:

  * `Import kWh/day` ← `pos_ae`
  * `Export kWh/day` ← `neg_ae`
  * `Aux Import kWh/day` ← `aux_m_pos_ae`
  * `Aux Export kWh/day` ← `aux_m_neg_ae`

## 7) Coding Conventions

* **Frequencies:** always lowercase (`"1h"`, `"1d"`, `"15min"`).
* **Timestamps:** store with tz; localize/convert to Europe/Berlin.
* **Parsing:** centralize datetime detection; fail early if no numeric data.
* **Signal detection:** normalize known meter keys; for BESS, keep stem and infer aggregators by suffix/pattern.
* **Logging:** concise `[INFO]/[WARN]/[ERR]` lines with folder/signal context.
* **Errors to user:** HTTP 404 for unknown meter/signal; 400/422 for bad params.

## 8) Testing Strategy

* **Unit** (pytest):

  * Parsing: datetime column auto-detect / fallbacks.
  * Aggregation: mean/last/max/min correctness.
  * Energy interval diffing: reset handling, outlier capping.
  * LTTB invariants (first/last preserved, size ≤ threshold).

* **Integration**:

  * Preprocess → API → Frontend flow with tiny sample datasets.
  * `/bundle` correctness across meters and BESS.

* **Golden files** (small): Expected CSV↔Parquet round-trip checks.

## 9) CI & Tooling

* **Lint/format**: ruff + black.
* **Type hints**: mypy in `--strict` for backend helpers.
* **Pre-commit**: run ruff/black/mypy/pytest on staged changes.
* **GitHub Actions**: matrix {3.11, 3.12} on push/PR.

## 10) Observability

* Add basic request logs (method, path, ms, status).
* Track cache hits/misses (optional).
* Expose `/health` (already present) and a lightweight `/metrics` if deploying to k8s.

## 11) Security & Hardening

* CORS limited to trusted origins if hosted publicly.
* Validate query params (bounds on `max_points`, allowed signals).
* Never serve raw CSV externally; always via curated endpoints.
* Sanitize path inputs (signals chosen from discovered index only).

## 12) Deployment Playbook

* **Precompute**: run `preprocess_pyramids.py` on the host with mounted data.
* **Backend**: `uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers <N>`.
* **Frontend**: `streamlit run frontend/app.py` behind reverse proxy.
* **Static caching**: persist `backend/.meter_cache` between releases.
* **Config**: pass `METER_ROOTS` and `API_URL` (frontend secrets).

## 13) Extending the System

**Add a new meter folder**

* Drop CSVs in `data/meter/new_folder/`.
* Run preprocess (or call `/prewarm`).
* `/reload` to rebuild index; it appears in the UI.

**Add a new BESS KPI**

* Identify CSV(s) → add picker keys in `/bess_kpis`.
* Choose aggregation (`mean/max/min/last`).
* Add to frontend KPI tiles.

**Add a new plot**

* Prefer adding to **bundle** to reduce calls.
* Use LOD+LTTB in server; avoid client resampling.

**Support new energy counter**

* Ensure filename ends with `*_pos_ae`, `*_neg_ae`, or `*_com_ae`.
* Preprocessor will store snapshots via **last**; backend will diff.

## 14) Troubleshooting

* **“No roots found”**: your CWD vs `METER_ROOTS` mismatch. From repo root:
  `export METER_ROOTS="data/meter,data/BESS"`.
* **Pandas `FutureWarning: 'H' deprecated`**: ensure frequencies are lowercase (`1h`, `1d`).
* **Slow UI**: shorten date range; reduce `max_points`; verify pyramids exist; check `/prewarm` count.
* **Missing series**: confirm CSV present; check `/meters`; signal key name in index; inspect `backend/.meter_cache/manifest.json` (preprocess output).

## 15) Makefile (optional)

```make
.PHONY: venv deps fmt lint test preprocess backend frontend

venv:
\tpython3 -m venv .venv && . .venv/bin/activate && python -m pip install --upgrade pip

deps:
\t. .venv/bin/activate && pip install -r requirements.txt

fmt:
\t. .venv/bin/activate && ruff check --fix . && black .

lint:
\t. .venv/bin/activate && ruff check . && mypy backend

test:
\t. .venv/bin/activate && pytest -q

preprocess:
\t. .venv/bin/activate && METER_ROOTS="data/meter,data/BESS" \\\n\t\tpython preprocess_pyramids.py --workers 4

backend:
\t. .venv/bin/activate && METER_ROOTS="data/meter,data/BESS" \\\n\t\tuvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 2 --no-access-log

frontend:
\t. .venv/bin/activate && cd frontend && streamlit run app.py
```

## 16) Requirements (minimal)

```
fastapi
uvicorn[standard]
pandas
numpy
pyarrow
plotly
streamlit
requests
```

> For dev: `black`, `ruff`, `mypy`, `pytest`.

---

### Quick “healthy run” sequence

```bash
# 0) Activate venv
source .venv/bin/activate

# 1) Precompute pyramids
export METER_ROOTS="data/meter,data/BESS"
python preprocess_pyramids.py --workers 4

# 2) Start backend
METER_ROOTS="data/meter,data/BESS" uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 2 --no-access-log

# 3) Start frontend
cd frontend
streamlit run app.py
```

---

**Keep PRs small.** Touch one layer at a time (preprocessor vs API vs UI). Update this doc when adding new signal families or KPIs.
