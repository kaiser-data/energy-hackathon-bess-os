# frontend/pages/bess_api.py
import os
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, DefaultDict, Tuple
from collections import defaultdict

import pandas as pd
import requests
import streamlit as st
import plotly.express as px

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
st.set_page_config(page_title="🔋 BESS Overview", layout="wide")
st.title("🔋 BESS Overview")

# ---------------- HTTP ----------------
def _req(path: str, params: dict | None = None) -> dict:
    r = requests.get(f"{API_URL}{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=60)
def meters_classified() -> dict:
    return _req("/meters/classified")

@st.cache_data(ttl=60)
def meter_info(meter: str) -> dict:
    return _req(f"/meters/{meter}/info")

def load_series(meter: str, signal: str, start: Optional[date], end: Optional[date], max_points=6000):
    params = {"meter": meter, "signal": signal, "max_points": max_points}
    if start: params["start"] = datetime.combine(start, datetime.min.time()).isoformat()
    if end:   params["end"]   = datetime.combine(end,   datetime.max.time()).isoformat()
    data = _req("/series", params=params)
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(pd.Series(data["timestamps"]), errors="coerce"),
        "value": pd.Series(data["values"], dtype="float32"),
        "signal": signal,
    }).dropna(subset=["timestamp"]).sort_values("timestamp")
    meta = {
        "rule": data.get("rule"), "count": data.get("count", 0),
        "actual_start": pd.to_datetime(data.get("actual_start")) if data.get("actual_start") else None,
        "actual_end":   pd.to_datetime(data.get("actual_end"))   if data.get("actual_end")   else None,
    }
    return df, meta

# ---------------- Groups & units ----------------
GROUPS = {
    "Battery (BMS)": [
        "bms1_soc", "bms1_soh", "bms1_v", "bms1_c",
        "bms1_cell_ave_v", "bms1_cell_ave_t",
        "bms1_cell_max_v", "bms1_cell_min_v", "bms1_cell_t_diff",
    ],
    "PCS (Inverter)": [
        "pcs1_ap", "pcs1_dcc", "pcs1_dcv", "pcs1_ia", "pcs1_ib", "pcs1_ic",
        "pcs1_uab", "pcs1_ubc", "pcs1_uca", "pcs1_t_env", "pcs1_t_a", "pcs1_t_igbt",
    ],
    "Aux/Thermal": [
        "aux_m_ap", "aux_m_pf", "ac1_outside_t", "ac1_outwater_t", "ac1_rtnwater_pre",
    ],
    "Environment/Safety": [
        "dh1_humi", "dh1_temp"
    ],
}
UNITS = {
    "bms1_soc": "%", "bms1_soh": "%", "bms1_v": "V", "bms1_c": "A",
    "bms1_cell_ave_v": "V", "bms1_cell_ave_t": "°C",
    "bms1_cell_max_v": "V", "bms1_cell_min_v": "V", "bms1_cell_t_diff": "°C",
    "pcs1_ap": "kW", "pcs1_dcc": "A", "pcs1_dcv": "V",
    "pcs1_ia": "A", "pcs1_ib": "A", "pcs1_ic": "A",
    "pcs1_uab": "V", "pcs1_ubc": "V", "pcs1_uca": "V",
    "pcs1_t_env": "°C", "pcs1_t_a": "°C", "pcs1_t_igbt": "°C",
    "aux_m_ap": "kW", "aux_m_pf": "–",
    "ac1_outside_t": "°C", "ac1_outwater_t": "°C", "ac1_rtnwater_pre": "bar",
    "dh1_humi": "%", "dh1_temp": "°C",
}

# ---------------- Scaling & formatting ----------------
def unit_scale(unit: str, series: pd.Series) -> Tuple[float, str]:
    if series.empty or unit in ("%", "°C", "bar", "–", ""):
        return 1.0, unit
    vmax = float(series.abs().max())
    if unit == "kW": return (1000.0, "MW") if vmax >= 1000 else (1.0, "kW")
    if unit == "V":  return (1000.0, "kV") if vmax >= 1000 else (1.0, "V")
    if unit == "A":  return (1000.0, "kA") if vmax >= 1000 else (1.0, "A")
    return 1.0, unit

def dynamic_decimals(series: pd.Series) -> int:
    if series.empty: return 2
    vmax = float(series.abs().max())
    if vmax >= 1000: return 0
    if vmax >= 100:  return 1
    if vmax >= 10:   return 1
    if vmax >= 1:    return 2
    return 3

def fmt_value(x: float, decimals: int, unit: str) -> str:
    if x is None or pd.isna(x): return "—"
    s = f"{x:,.{decimals}f}"
    return f"{s}{'' if unit in ('–','') else ' ' + unit}"

# ---------------- Plot helpers ----------------
PLOTLY_TEMPLATE = "plotly_white"
MODEBAR_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToAdd": [
        "zoom2d","pan2d","select2d","lasso2d","zoomIn2d","zoomOut2d","autoScale2d","resetScale2d"
    ],
}

def apply_axes(fig, y_dec: int, unit: str):
    tickfmt = f",.{y_dec}f"
    fig.update_xaxes(
        rangeslider=dict(visible=True),
        rangeselector=dict(buttons=[
            dict(count=1, label="1d", step="day", stepmode="backward"),
            dict(count=7, label="7d", step="day", stepmode="backward"),
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(step="all", label="All"),
        ]),
        tickformatstops=[
            dict(dtickrange=[None, 1000*60*60*24], value="%d %b %Y\n%H:%M"),
            dict(dtickrange=[1000*60*60*24, 1000*60*60*24*30], value="%d %b %Y"),
            dict(dtickrange=[1000*60*60*24*30, 1000*60*60*24*365], value="%b %Y"),
            dict(dtickrange=[1000*60*60*24*365, None], value="%Y"),
        ],
        showspikes=True, spikemode="across", spikesnap="cursor", showgrid=True,
    )
    fig.update_yaxes(
        fixedrange=False, showgrid=True,
        tickformat=tickfmt, ticksuffix=(" " + unit) if unit and unit != "–" else None,
    )
    fig.update_layout(
        hovermode="x unified", dragmode="zoom", template=PLOTLY_TEMPLATE,
        font=dict(size=16), margin=dict(t=60, r=20, b=60, l=70),
    )
    return fig

def multi_line_scaled(dfu: pd.DataFrame, unit: str, title: str):
    dec = dynamic_decimals(dfu["display_value"])
    tickfmt = f",.{dec}f"
    hover = "%{x}<br>%{fullData.name}: %{y:" + tickfmt + "}" + (f" {unit}" if unit and unit != "–" else "")
    fig = px.line(
        dfu, x="timestamp", y="display_value", color="signal",
        title=title, labels={"timestamp":"Time", "display_value": f"Value [{unit}]" if unit and unit != "–" else "Value"}
    )
    fig.update_traces(mode="lines", line=dict(width=3), hovertemplate=hover)
    return apply_axes(fig, dec, unit)

def summarize_formatted(dfu: pd.DataFrame, unit: str) -> pd.DataFrame:
    if dfu.empty:
        return pd.DataFrame(columns=["signal","count","mean","min","max","std","p50","p95"])
    g = dfu.groupby("signal")["display_value"]
    raw = pd.DataFrame({
        "count": g.count().astype(int),
        "mean": g.mean(),
        "min": g.min(),
        "max": g.max(),
        "std": g.std(),
        "p50": g.quantile(0.5),
        "p95": g.quantile(0.95),
    }).reset_index()
    dec = dynamic_decimals(dfu["display_value"])
    for c in ["mean","min","max","std","p50","p95"]:
        raw[c] = raw[c].map(lambda x: fmt_value(x, dec, unit))
    return raw[["signal","count","mean","min","max","std","p50","p95"]].sort_values("signal")

def values_table_formatted(dfu: pd.DataFrame, unit: str, max_rows: int = 500) -> pd.DataFrame:
    if dfu.empty:
        return pd.DataFrame(columns=["timestamp","signal","value"])
    dec = dynamic_decimals(dfu["display_value"])
    out = dfu[["timestamp","signal","display_value"]].copy().sort_values(["timestamp","signal"]).head(max_rows)
    out["value"] = out["display_value"].map(lambda x: fmt_value(x, dec, unit))
    return out.drop(columns=["display_value"])

# ---------------- Sidebar controls ----------------
classified = meters_classified()
bess_names = sorted(list(classified.get("bess", {}).keys()))
if not bess_names:
    st.warning("No BESS systems detected.")
    st.stop()

st.sidebar.header("Controls")
meter = st.sidebar.selectbox("BESS System", bess_names)
info = meter_info(meter)
available = set(info.get("signals", []))

# bounds by probing a robust signal
probe_sig = next((s for s in ["bms1_soc","pcs1_ap","bms1_v"] if s in available), (sorted(available)[0] if available else None))
probe_df, probe_meta = (load_series(meter, probe_sig, None, None, 1000) if probe_sig else (pd.DataFrame(), {}))
if not probe_df.empty:
    min_d = (probe_meta["actual_start"].date() if probe_meta.get("actual_start") is not None else probe_df["timestamp"].min().date())
    max_d = (probe_meta["actual_end"].date()   if probe_meta.get("actual_end")   is not None else probe_df["timestamp"].max().date())
else:
    today = datetime.now().date(); min_d, max_d = today - timedelta(days=7), today

preset = st.sidebar.selectbox("Range", ["Last 24h", "Last 7d", "Last 30d", "All available", "Custom"], index=1)
if preset == "Last 24h":
    start_sel, end_sel = max_d - timedelta(days=1), max_d
elif preset == "Last 7d":
    start_sel, end_sel = max_d - timedelta(days=7), max_d
elif preset == "Last 30d":
    start_sel, end_sel = max_d - timedelta(days=30), max_d
elif preset == "All available":
    start_sel, end_sel = min_d, max_d
else:
    dates = st.sidebar.date_input("Custom dates", (max(min_d, max_d - timedelta(days=7)), max_d),
                                  min_value=min_d, max_value=max_d, format="YYYY-MM-DD")
    start_sel, end_sel = (dates if isinstance(dates, tuple) else (dates, dates))
start_sel = max(min_d, min(start_sel, max_d))
end_sel   = max(min_d, min(end_sel,   max_d))

# group & signals in sidebar
groups_with_availability = {g: [s for s in sigs if s in available] for g, sigs in GROUPS.items()}
dyn_safety = [s for s in available if s.lower().startswith("fa") or "flag" in s.lower() or "err" in s.lower()]
if dyn_safety:
    groups_with_availability["Safety/Flags"] = sorted(dyn_safety)

group = st.sidebar.selectbox("Sensor group", [g for g, sigs in groups_with_availability.items() if sigs])
candidates = groups_with_availability[group]
default_pick = candidates[:3] if candidates else []
picked = st.sidebar.multiselect("Signals", candidates, default=default_pick, key=f"{meter}-{group}")

# Y-axis is always auto-scaled (clipping controls removed)

# ---------------- Content ----------------
st.caption("Pick system, range, and signals in the sidebar.")

if not picked:
    st.info("Select one or more signals to plot.")
else:
    frames: List[pd.DataFrame] = []
    for sig in picked:
        df, _ = load_series(meter, sig, start_sel, end_sel, 6000)
        if not df.empty:
            df["signal"] = sig
            df["unit"] = UNITS.get(sig, "")
            frames.append(df)

    if not frames:
        st.info("No data in this period for selected signals.")
    else:
        data = pd.concat(frames, ignore_index=True)

        # split by unit and render
        by_unit: DefaultDict[str, pd.DataFrame] = defaultdict(pd.DataFrame)
        for sig in data["signal"].unique():
            u = UNITS.get(sig, "")
            block = data[data["signal"] == sig]
            if u not in by_unit or by_unit[u].empty:
                by_unit[u] = block.copy()
            else:
                by_unit[u] = pd.concat([by_unit[u], block], ignore_index=True)

        st.subheader(f"{meter} · {group}")

        for unit, dfu in by_unit.items():
            # scale per-unit
            scale, disp_unit = unit_scale(unit, dfu["value"])
            dfu = dfu.copy()
            dfu["display_value"] = dfu["value"] / scale if scale != 1.0 else dfu["value"]

            # chart
            title = f"{group} ({disp_unit or 'various'})"
            fig = multi_line_scaled(dfu, disp_unit, title)

            # Y-axis is always auto-scaled (clipping removed)

            st.plotly_chart(fig, use_container_width=True, config=MODEBAR_CONFIG)

            # stats & formatted values below chart
            st.caption("Statistics (selected range)")
            st.dataframe(summarize_formatted(dfu, disp_unit), use_container_width=True)

            st.caption("Values (formatted)")
            st.dataframe(values_table_formatted(dfu, disp_unit), use_container_width=True)
