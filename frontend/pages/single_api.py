# frontend/pages/single_api.py
import os
from datetime import datetime, timedelta, date
from typing import Dict, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
import plotly.express as px

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
st.set_page_config(page_title="⚡ Single Meter Analysis", layout="wide")
st.title("⚡ Single Meter Analysis")

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

    # Force UTC-aware timestamps to avoid tz conversion errors later
    ts = pd.to_datetime(pd.Series(data["timestamps"]), errors="coerce", utc=True)

    df = pd.DataFrame({
        "timestamp": ts,
        "value": pd.to_numeric(pd.Series(data["values"]), errors="coerce"),
    }).dropna(subset=["timestamp"]).sort_values("timestamp")

    meta = {
        "rule": data.get("rule"),
        "count": data.get("count", 0),
        "actual_start": pd.to_datetime(data.get("actual_start"), utc=True) if data.get("actual_start") else None,
        "actual_end":   pd.to_datetime(data.get("actual_end"),   utc=True) if data.get("actual_end")   else None,
    }
    return df, meta

# ---------------- Units, labels & formatting ----------------
SIG_UNITS = {"com_ap": "kW", "pf": "–", "com_ae": "kWh", "pos_ae": "kWh", "neg_ae": "kWh"}
SIG_LABELS = {
    "com_ap": "Power",
    "pf": "Power Factor",
    "com_ae": "Energy (interval)",
    "pos_ae": "Energy Import (interval)",
    "neg_ae": "Energy Export (interval)",
}
ENERGY_SIGS = {"com_ae", "pos_ae", "neg_ae"}

def auto_scale(sig: str, series: pd.Series) -> Tuple[float, str]:
    unit = SIG_UNITS.get(sig, "")
    if unit == "–" or series.empty:
        return 1.0, unit
    vmax = float(pd.to_numeric(series, errors="coerce").abs().max() or 0.0)
    if unit == "kW":  return (1000.0, "MW") if vmax >= 1000 else (1.0, "kW")
    if unit == "kWh": return (1000.0, "MWh") if vmax >= 1000 else (1.0, "kWh")
    return 1.0, unit

def dynamic_decimals(series) -> int:
    """
    Accepts a pandas Series *or* DataFrame; returns sensible decimals.
    """
    if isinstance(series, pd.DataFrame):
        if "display_value" in series.columns:
            series = series["display_value"]
        else:
            series = series.iloc[:, 0]
    series = pd.to_numeric(series, errors="coerce")
    if series.dropna().empty:
        return 2
    vmax = float(series.abs().max())
    if vmax >= 1000: return 0
    if vmax >= 100:  return 1
    if vmax >= 10:   return 1
    if vmax >= 1:    return 2
    return 3

def fmt_number(x: float, decimals: int, unit: str) -> str:
    if x is None or pd.isna(x): return "—"
    s = f"{x:,.{decimals}f}"
    return (s if unit in ("–","") else f"{s} {unit}").strip()

def stats_for(series: pd.Series) -> Dict[str, float]:
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return {k: float("nan") for k in ["count","mean","min","max","std","p50","p95","sum"]}
    return {
        "count": int(s.count()),
        "mean": float(s.mean()),
        "min": float(s.min()),
        "max": float(s.max()),
        "std": float(s.std()),
        "p50": float(s.quantile(0.50)),
        "p95": float(s.quantile(0.95)),
        "sum": float(s.sum()),
    }

def y_title(sig: str, unit: str) -> str:
    base = SIG_LABELS.get(sig, sig)
    return f"{base} [{unit}]" if unit and unit != "–" else base

# ---------------- Plot helpers ----------------
PLOTLY_TEMPLATE = "plotly_white"
MODEBAR_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToAdd": [
        "zoom2d","pan2d","select2d","lasso2d","zoomIn2d","zoomOut2d","autoScale2d","resetScale2d"
    ],
}

def _apply_axes(fig, y_decimals: int, unit: str):
    tickfmt = f",.{y_decimals}f"
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
        title=y_title(current_signal, unit)
    )
    fig.update_layout(
        hovermode="x unified", dragmode="zoom", template=PLOTLY_TEMPLATE,
        font=dict(size=16), margin=dict(t=60, r=20, b=60, l=70),
    )
    return fig

def make_line(df_disp: pd.DataFrame, unit: str, title: str):
    dec = dynamic_decimals(df_disp["display_value"])
    tickfmt = f",.{dec}f"
    hover = "%{x}<br>%{y:" + tickfmt + "}" + (f" {unit}" if unit and unit != "–" else "")
    fig = px.line(
        df_disp, x="timestamp", y="display_value",
        title=title, labels={"timestamp":"Time", "display_value": y_title(current_signal, unit)}
    )
    fig.update_traces(mode="lines", line=dict(width=3), hovertemplate=hover)
    return _apply_axes(fig, dec, unit)

def make_area(df_disp: pd.DataFrame, unit: str, title: str):
    dec = dynamic_decimals(df_disp["display_value"])
    tickfmt = f",.{dec}f"
    hover = "%{x}<br>%{y:" + tickfmt + "}" + (f" {unit}" if unit and unit != "–" else "")
    fig = px.area(
        df_disp, x="timestamp", y="display_value",
        title=title, labels={"timestamp":"Time", "display_value": y_title(current_signal, unit)}
    )
    fig.update_traces(hovertemplate=hover)
    return _apply_axes(fig, dec, unit)

def make_bar(daily_disp: pd.DataFrame, unit: str, title: str):
    """
    Expects columns: ['timestamp', 'display_value'].
    """
    y_series = pd.to_numeric(daily_disp["display_value"], errors="coerce")
    dec = dynamic_decimals(y_series)
    tickfmt = f",.{dec}f"
    hover = "%{x}<br>%{y:" + tickfmt + "}" + (f" {unit}" if unit and unit != "–" else "")
    fig = px.bar(
        daily_disp, x="timestamp", y="display_value",
        title=title,
        labels={"timestamp": "Time",
                "display_value": f"Daily total [{unit}]" if unit and unit != "–" else "Daily total"},
    )
    fig.update_traces(hovertemplate=hover)
    return _apply_axes(fig, dec, unit)

# ---------------- Sidebar controls ----------------
classified = meters_classified()
meter_names = sorted(list(classified.get("meters", {}).keys()))
if not meter_names:
    st.warning("No classic meters found. (BESS systems are on the BESS page.)")
    st.stop()

st.sidebar.header("Controls")
meter = st.sidebar.selectbox("Meter", meter_names)
info = meter_info(meter)
prefer = [s for s in ("com_ap","pf","pos_ae","neg_ae","com_ae") if s in info.get("signals", [])]
signals = prefer or info.get("signals", [])
current_signal = st.sidebar.selectbox("Signal", sorted(signals))

# Probe available range
probe_df, probe_meta = load_series(meter, current_signal, None, None, 1000)
if not probe_df.empty:
    min_d = (probe_meta["actual_start"].date() if probe_meta["actual_start"] is not None else probe_df["timestamp"].min().date())
    max_d = (probe_meta["actual_end"].date()   if probe_meta["actual_end"]   is not None else probe_df["timestamp"].max().date())
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

y_mode = st.sidebar.radio("Y-axis", ["Auto", "Clip 1–99%", "Manual"], index=0, horizontal=False)
y_manual = None
if y_mode == "Manual":
    c1, c2 = st.sidebar.columns(2)
    y_min = c1.number_input("y min", value=0.0)
    y_max = c2.number_input("y max", value=1.0)
    if y_min < y_max:
        y_manual = [float(y_min), float(y_max)]

# ---------------- Content ----------------
df, meta = load_series(meter, current_signal, start_sel, end_sel, 6000)
st.caption(
    f"Rule: {meta['rule']} · Actual: "
    f"{meta['actual_start'].date() if meta['actual_start'] else '—'} → "
    f"{meta['actual_end'].date() if meta['actual_end'] else '—'} · Points: {meta['count']}"
)

if df.empty:
    st.info("No data in this period.")
else:
    # scale & stats
    scale, disp_unit = auto_scale(current_signal, df["value"])
    df_disp = df.copy()
    # Ensure numeric, scaled values
    df_disp["display_value"] = pd.to_numeric(df_disp["value"], errors="coerce") / (scale if scale != 0 else 1.0)

    stats = stats_for(df_disp["display_value"])
    dec = dynamic_decimals(df_disp["display_value"])

    a,b,c,d,e,f = st.columns(6)
    a.metric("Mean",   fmt_number(stats["mean"], dec, disp_unit))
    b.metric("Max",    fmt_number(stats["max"],  dec, disp_unit))
    c.metric("Min",    fmt_number(stats["min"],  dec, disp_unit))
    d.metric("Median", fmt_number(stats["p50"],  dec, disp_unit))
    e.metric("p95",    fmt_number(stats["p95"],  dec, disp_unit))
    f.metric("Sum" if current_signal in ENERGY_SIGS else "Std",
             fmt_number(stats["sum"] if current_signal in ENERGY_SIGS else stats["std"], dec, disp_unit))

    title = SIG_LABELS.get(current_signal, current_signal)
    fig = make_area(df_disp, disp_unit, title) if current_signal in ENERGY_SIGS else make_line(df_disp, disp_unit, title)
    if y_manual: fig.update_yaxes(range=y_manual)
    elif y_mode == "Clip 1–99%" and not df_disp["display_value"].empty:
        q1 = float(df_disp["display_value"].quantile(0.01)); q99 = float(df_disp["display_value"].quantile(0.99))
        if q1 < q99: fig.update_yaxes(range=[q1, q99])

    st.plotly_chart(fig, use_container_width=True, config=MODEBAR_CONFIG)

    # Daily totals for energy (scaled) — timezone-safe
    if current_signal in ENERGY_SIGS:
        # 'timestamp' is already tz-aware (UTC). Set as index directly.
        s_daily = (
            df.set_index("timestamp")["value"]
              .resample("1D").sum(min_count=1)
        )
        daily = s_daily.rename("value").reset_index()
        if not daily.empty:
            s2, u2 = auto_scale("com_ae", daily["value"])
            daily["display_value"] = pd.to_numeric(daily["value"], errors="coerce") / (s2 if s2 != 0 else 1.0)
            fig2 = make_bar(daily[["timestamp", "display_value"]], u2, "Daily Energy")
            st.plotly_chart(fig2, use_container_width=True, config=MODEBAR_CONFIG)

with st.expander("Raw (head 200)"):
    st.dataframe(df.head(200), use_container_width=True)
