# Create the ForecastIQ Streamlit app, requirements.txt, and README.md as downloadable files.

import os, textwrap, json, sys, pandas as pd  # pandas just to demo environment and for potential future edits
from pathlib import Path

base = Path("/mnt/data/forecastiq")
base.mkdir(parents=True, exist_ok=True)

app_py = r'''# ForecastIQ — AI-powered clarity for financial foresight
# Streamlit application implementing CSV/XLSX upload, column mapping, optional categorical filtering,
# forecasting via Prophet, AutoARIMA, or Simple Moving Average, plus narrative insights via OpenAI.
#
# Libraries used: streamlit, pandas, numpy, matplotlib, prophet (optional), pmdarima, openai
# To run locally: `pip install -r requirements.txt` then `streamlit run app.py`
#
# Notes:
# - Set your OpenAI API key as environment variable OPENAI_API_KEY for narrative insights.
# - If Prophet is unavailable on your platform, the Prophet option will be disabled gracefully.

import os
import io
import sys
import traceback
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime

# Optional imports with graceful fallbacks
HAVE_PROPHET = True
try:
    from prophet import Prophet
except Exception:  # ImportError or build issues
    HAVE_PROPHET = False

HAVE_PMDARIMA = True
try:
    from pmdarima import auto_arima
except Exception:
    HAVE_PMDARIMA = False

# OpenAI (optional)
HAVE_OPENAI = True
OPENAI_STYLE = "new"  # "new" for OpenAI() client, "legacy" for openai.ChatCompletion
try:
    from openai import OpenAI
    _ = OpenAI
except Exception:
    try:
        import openai  # legacy fallback
        OPENAI_STYLE = "legacy"
    except Exception:
        HAVE_OPENAI = False

st.set_page_config(page_title="ForecastIQ", layout="wide")

# ---------------------------
# Helpers
# ---------------------------

def read_input_file(uploaded_file: "UploadedFile") -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a .csv or .xlsx file.")

def coerce_to_datetime(df, col):
    out = pd.to_datetime(df[col], errors="coerce")
    return out

def coerce_to_numeric(df, col):
    out = pd.to_numeric(df[col], errors="coerce")
    return out

def ensure_monotonic_frequency(df, freq_code):
    """
    Ensure a regular time index at the chosen frequency by reindexing over a complete date_range.
    We fill missing values via forward-fill; if initial NaNs remain, back-fill as a last resort.
    We also consolidate duplicate timestamps by summing (generic-safe).
    """
    if df.empty:
        return df
    df = df.copy()
    df = df.groupby("ds", as_index=False)["y"].sum()  # consolidate duplicates safely
    df = df.set_index("ds").sort_index()
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq_code)
    df = df.reindex(full_idx)
    df.index.name = "ds"
    # Fill small gaps; if entire leading region is NaN, backfill that part as a last resort.
    df["y"] = df["y"].ffill().bfill()
    return df.reset_index().rename(columns={"index": "ds"})

def make_future_dates(last_ds: pd.Timestamp, periods: int, freq: str) -> pd.DatetimeIndex:
    if periods <= 0:
        return pd.DatetimeIndex([])
    # Start the next period after the last ds
    start = pd.date_range(last_ds, periods=2, freq=freq)[-1]
    return pd.date_range(start=start, periods=periods, freq=freq)

def forecast_prophet(df, horizon, freq_code):
    if not HAVE_PROPHET:
        raise RuntimeError("Prophet is not available in this environment.")
    # Prophet prefers strictly named ds, y and benefits from regular frequency
    model_df = ensure_monotonic_frequency(df[["ds","y"]], freq_code)
    m = Prophet()  # defaults; can be extended with seasonality controls
    m.fit(model_df)
    future = m.make_future_dataframe(periods=horizon, freq=freq_code)
    fcst = m.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    # Return only the forecast horizon rows (future)
    cutoff = model_df["ds"].max()
    return fcst[fcst["ds"] > cutoff].reset_index(drop=True)

def forecast_autoarima(df, horizon, freq_code):
    if not HAVE_PMDARIMA:
        raise RuntimeError("pmdarima (AutoARIMA) is not available in this environment.")
    model_df = ensure_monotonic_frequency(df[["ds","y"]], freq_code)
    series = model_df.set_index("ds")["y"]
    # Let auto_arima decide (seasonal can be autodetected by m)
    arima = auto_arima(series, seasonal=True, stepwise=True, suppress_warnings=True, error_action="ignore")
    preds, conf = arima.predict(n_periods=horizon, return_conf_int=True, alpha=0.05)
    future_ds = make_future_dates(series.index.max(), horizon, freq_code)
    out = pd.DataFrame({
        "ds": future_ds,
        "yhat": preds,
        "yhat_lower": conf[:,0],
        "yhat_upper": conf[:,1],
    })
    return out

def forecast_sma(df, horizon, freq_code):
    """
    Simple Moving Average forecast:
    - Rolling window based on frequency: D->7, W->4, M->3
    - Iterative forecasting so that future forecasts feed into subsequent windows.
    - Intervals from residual std of in-sample SMA vs actuals (Gaussian approx).
    """
    window_map = {"D": 7, "W": 4, "MS": 3}
    win = window_map.get(freq_code, 7)
    model_df = ensure_monotonic_frequency(df[["ds","y"]], freq_code).set_index("ds").sort_index()
    y = model_df["y"].astype(float).copy()

    # In-sample SMA and residuals
    sma = y.rolling(win, min_periods=max(1, win//2)).mean()
    resid = (y - sma).dropna()
    resid_std = float(resid.std()) if not resid.empty else 0.0

    future_ds = make_future_dates(y.index.max(), horizon, freq_code)
    history = y.tolist()  # use history + forecasts iteratively
    yhat = []
    for _ in range(horizon):
        if len(history) < win:
            current_win = history  # short start
        else:
            current_win = history[-win:]
        mean_val = float(np.mean(current_win))
        yhat.append(mean_val)
        history.append(mean_val)

    z = 1.96
    lower = [val - z*resid_std for val in yhat]
    upper = [val + z*resid_std for val in yhat]

    out = pd.DataFrame({
        "ds": future_ds,
        "yhat": yhat,
        "yhat_lower": lower,
        "yhat_upper": upper,
    })
    return out

def generate_narrative(openai_enabled: bool, forecast_df: pd.DataFrame, last_actual: float, freq_label: str, model_name: str) -> str:
    """
    Create a concise narrative summarizing forecast trend and uncertainty.
    If OPENAI is available and key is set, call the API. Otherwise, return a deterministic summary.
    """
    # Compute a quick slope on forecast yhat as a trend indicator
    yhat = forecast_df["yhat"].values
    x = np.arange(len(yhat))
    slope = float(np.polyfit(x, yhat, 1)[0]) if len(yhat) >= 2 else 0.0
    direction = "increasing" if slope > 0 else ("decreasing" if slope < 0 else "flat")
    avg = float(np.mean(yhat)) if len(yhat) else float("nan")
    start_val = float(yhat[0]) if len(yhat) else float("nan")
    end_val = float(yhat[-1]) if len(yhat) else float("nan")

    deterministic = (
        f"Outlook summary ({model_name}, {freq_label}): The forecast trajectory is {direction}. "
        f"Starting near {start_val:,.2f} and ending around {end_val:,.2f}, "
        f"the average level over the horizon is approximately {avg:,.2f}. "
        f"The most recent actual observed value before forecasting was {last_actual:,.2f}."
    )

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_enabled or not api_key:
        return deterministic

    prompt = f"""
You are a finance-savvy analyst. Write a crisp narrative (120–170 words) for a CFO
summarizing a time-series forecast. Avoid promises or guarantees. Use precise, factual language.

Context:
- Frequency: {freq_label}
- Model: {model_name}
- Last Actual: {last_actual:,.4f}
- Forecast Start: {start_val:,.4f}
- Forecast End: {end_val:,.4f}
- Direction: {direction}
- Forecast Summary Stats: average={avg:,.4f}, slope={slope:,.6f}

Data Sample (first 5 rows):
{forecast_df.head().to_string(index=False)}

Deliver:
- 2 short paragraphs max
- Mention uncertainty bands exist (yhat_lower, yhat_upper)
- Avoid the words "guarantee", "ensure", "will never".
"""

    try:
        if OPENAI_STYLE == "new":
            client = OpenAI()
            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "You are an expert financial analyst."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
        else:
            import openai
            openai.api_key = api_key
            resp = openai.ChatCompletion.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "You are an expert financial analyst."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return deterministic + " (Narrative via OpenAI unavailable; showing fallback.)"

def render_plot(hist_df, fcst_df, freq_label, model_name):
    fig, ax = plt.subplots(figsize=(10, 5))
    # Historical
    ax.plot(hist_df["ds"], hist_df["y"], label="Actual (y)")
    # Forecast
    ax.plot(fcst_df["ds"], fcst_df["yhat"], label="Forecast (yhat)")
    # Intervals
    ax.fill_between(fcst_df["ds"], fcst_df["yhat_lower"], fcst_df["yhat_upper"], alpha=0.2, label="Interval")
    ax.set_title(f"Forecast ({model_name}, {freq_label})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    st.pyplot(fig)

def to_excel_bytes(forecast_df: pd.DataFrame, meta: dict) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        forecast_df.to_excel(writer, sheet_name="Forecast", index=False)
        pd.DataFrame([meta]).T.rename(columns={0: "value"}).to_excel(writer, sheet_name="Meta")
    buffer.seek(0)
    return buffer.read()

# ---------------------------
# UI
# ---------------------------
st.title("ForecastIQ")
st.caption("AI-powered clarity for financial foresight")

with st.sidebar:
    st.header("1) Upload")
    uploaded = st.file_uploader("Upload a CSV or Excel (.xlsx) file", type=["csv", "xlsx", "xls"])

    st.header("2) Configure")
    freq_label = st.selectbox("Frequency", options=["Daily", "Weekly", "Monthly"], index=0)
    freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}
    freq_code = freq_map[freq_label]

    model_options = ["AutoARIMA", "Simple Moving Average"]
    if HAVE_PROPHET:
        model_options.insert(0, "Prophet")
    model_name = st.selectbox("Forecasting method", options=model_options, index=0)

    horizon = st.number_input("Forecast horizon (periods)", min_value=1, max_value=1000, value=90 if freq_code=="D" else (26 if freq_code=="W" else 12), step=1)

    st.markdown("---")
    st.header("3) Run")
    run_btn = st.button("Generate Forecast", type="primary")

st.markdown("### Upload")
if uploaded is None:
    st.info("Upload a dataset to begin. Accepted formats: CSV, XLSX.")
    st.stop()

# Load and basic validation
try:
    raw_df = read_input_file(uploaded)
except Exception as e:
    st.error(f"File could not be read: {e}")
    st.stop()

if raw_df.empty or raw_df.shape[1] < 2:
    st.error("The file seems empty or has too few columns.")
    st.stop()

st.success(f"Loaded data with {raw_df.shape[0]:,} rows and {raw_df.shape[1]:,} columns.")
with st.expander("Preview data (top 10 rows)"):
    st.dataframe(raw_df.head(10), use_container_width=True)

# Column selection
st.markdown("### Configure")
all_cols = list(raw_df.columns)
date_col = st.selectbox("Select date/time column (→ ds)", options=all_cols)
numeric_candidates = [c for c in all_cols if pd.api.types.is_numeric_dtype(raw_df[c]) or pd.api.types.is_bool_dtype(raw_df[c])]
target_col = st.selectbox("Select numeric target column (→ y)", options=numeric_candidates if numeric_candidates else all_cols)

category_col = st.selectbox("Optional categorical column to filter", options=["(None)"] + all_cols, index=0)
category_vals = []
if category_col != "(None)":
    unique_vals = pd.unique(raw_df[category_col].astype(str))
    category_vals = st.multiselect("Select one or more values to include", options=list(unique_vals))

st.markdown("### Forecast")
st.write("When ready, click **Generate Forecast** in the sidebar.")

if not run_btn:
    st.stop()

# ---------------------------
# Execution
# ---------------------------
with st.spinner("Preparing data..."):
    df = raw_df.copy()
    # Apply optional filter
    if category_col != "(None)" and category_vals:
        df = df[df[category_col].astype(str).isin(category_vals)].copy()

    # Parse date + numeric coercion
    df["ds"] = coerce_to_datetime(df, date_col)
    df["y"] = coerce_to_numeric(df, target_col)
    before = len(df)
    df = df.dropna(subset=["ds", "y"]).copy()
    dropped = before - len(df)
    if dropped > 0:
        st.warning(f"Dropped {dropped:,} rows due to invalid dates or non-numeric target.")

    df = df.sort_values("ds")
    if df.empty:
        st.error("No valid rows remain after cleaning.")
        st.stop()

    # Keep a copy of historical for plotting
    hist_df = df[["ds", "y"]].copy()

    # Basic size sanity
    if len(df) < max(10, min(50, horizon)):
        st.warning("Dataset is quite small; forecasts may be unstable.")

# Forecast
try:
    with st.spinner(f"Running {model_name}..."):
        if model_name == "Prophet":
            fcst = forecast_prophet(df, horizon=horizon, freq_code=freq_code)
        elif model_name == "AutoARIMA":
            fcst = forecast_autoarima(df, horizon=horizon, freq_code=freq_code)
        else:
            fcst = forecast_sma(df, horizon=horizon, freq_code=freq_code)
except Exception as e:
    st.error(f"Model failed: {e}")
    st.stop()

# Results
st.markdown("### Results")
col1, col2 = st.columns([2,1], gap="large")

with col1:
    try:
        render_plot(hist_df, fcst, freq_label, model_name)
    except Exception:
        st.error("Plotting failed.")
        st.exception(traceback.format_exc())

with col2:
    st.subheader("Narrative summary")
    last_actual = float(hist_df.iloc[-1]["y"])
    narrative = generate_narrative(HAVE_OPENAI, fcst, last_actual, freq_label, model_name)
    st.write(narrative)

st.subheader("Forecast table")
st.dataframe(fcst, use_container_width=True)

# Download
meta = {
    "generated_at": datetime.utcnow().isoformat() + "Z",
    "model": model_name,
    "horizon": horizon,
    "frequency": freq_label,
    "source_file": uploaded.name,
    "date_column": date_col,
    "target_column": target_col,
    "category_column": category_col if category_col != "(None)" else "",
    "category_values": ", ".join(category_vals) if category_vals else "",
}
excel_bytes = to_excel_bytes(fcst, meta)
st.download_button(
    label="Download Excel Output",
    data=excel_bytes,
    file_name="ForecastIQ_output.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.caption("Tip: Set the environment variable OPENAI_API_KEY to enable AI-generated narrative insights.")
'''

reqs = """streamlit==1.38.0
pandas==2.2.2
numpy==1.26.4
matplotlib==3.8.4
prophet==1.1.6
pmdarima==2.0.4
openai>=1.30.0
xlsxwriter==3.2.0
"""

readme = r'''# ForecastIQ — AI-powered clarity for financial foresight

A professional Streamlit app for quick time-series forecasting with CSV/XLSX upload, configurable columns, optional categorical filtering, three model choices (Prophet, AutoARIMA, Simple Moving Average), forecast intervals, narrative insights via OpenAI, and Excel export.

## Features

- **Data upload:** CSV or Excel
- **Column mapping:** Choose date/time (`ds`) and numeric target (`y`); optional categorical filter
- **Models:** Prophet, AutoARIMA, or Simple Moving Average
- **Controls:** Horizon length and frequency (Daily, Weekly, Monthly)
- **Outputs:** Matplotlib plot, forecast table (columns: `ds`, `yhat`, `yhat_lower`, `yhat_upper`), downloadable Excel
- **Narrative summary:** Optional, via OpenAI (set `OPENAI_API_KEY`)

## Local Setup

1. **Prerequisites**
   - Python 3.10 or newer (3.11 recommended)
   - (Optional) A virtual environment
   - (Optional) OpenAI key for narrative: set `OPENAI_API_KEY`

2. **Install**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
