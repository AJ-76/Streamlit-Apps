# ForecastIQ — AI-powered clarity for financial foresight
# Streamlit app: CSV/XLSX upload → column mapping → (optional) categorical filter →
# model (Prophet / AutoARIMA / Simple Moving Average) → forecast plot/table → Excel download → narrative (OpenAI).
#
# Run:
#   pip install -r requirements.txt
#   streamlit run app.py
#
# Notes:
# - Set OPENAI_API_KEY to enable AI narrative (optional).
# - If Prophet or pmdarima (AutoARIMA) is not installed, that option is hidden automatically.

import os
import io
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ---------- Optional libs with graceful fallbacks ----------
HAVE_PROPHET = True
try:
    from prophet import Prophet
except Exception:
    HAVE_PROPHET = False

HAVE_PMDARIMA = True
try:
    from pmdarima import auto_arima
except Exception:
    HAVE_PMDARIMA = False

HAVE_OPENAI = True
OPENAI_STYLE = "new"
try:
    from openai import OpenAI
    _ = OpenAI
except Exception:
    try:
        import openai  # legacy
        OPENAI_STYLE = "legacy"
    except Exception:
        HAVE_OPENAI = False

st.set_page_config(page_title="ForecastIQ", layout="wide")
st.title("ForecastIQ")
st.caption("AI-powered clarity for financial foresight")

# ---------- Utilities ----------

def read_input_file(uploaded_file) -> pd.DataFrame:
    """Read CSV/XLSX/XLS with explicit engines for reliability."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xlsm"):
        # requires openpyxl in requirements
        return pd.read_excel(uploaded_file, engine="openpyxl")
    if name.endswith(".xls"):
        # requires xlrd in requirements
        return pd.read_excel(uploaded_file, engine="xlrd")
    raise ValueError("Unsupported file type. Please upload .csv, .xlsx, or .xls.")

def coerce_to_datetime(df, col):
    # Ensure timezone-naive, coerce invalid to NaT
    ds = pd.to_datetime(df[col], errors="coerce")
    if hasattr(ds.dt, "tz_localize"):
        try:
            ds = ds.dt.tz_localize(None)
        except Exception:
            pass
    return ds

def coerce_to_numeric(df, col):
    """
    Make a best-effort numeric coercion:
    - Remove thousands separators
    - Handle ($1,234.56) → -1234.56
    - Strip common currency symbols
    """
    s = df[col].astype(str).str.strip()
    # parentheses for negatives
    s = s.str.replace("(", "-", regex=False).str.replace(")", "", regex=False)
    # remove commas and common symbols
    for sym in [",", "$", "₹", "€", "£"]:
        s = s.str.replace(sym, "", regex=False)
    return pd.to_numeric(s, errors="coerce")

def ensure_monotonic_frequency(df, freq_code):
    """
    Regularize the time index to the chosen frequency by reindexing over a complete date_range.
    - Consolidate duplicate timestamps by summing.
    - For weekly data, auto-anchor to the dominant weekday (W-MON ... W-SUN).
    - Fill gaps with forward-fill/back-fill for stability.
    """
    if df.empty:
        return df
    df = df.copy()
    df = df.groupby("ds", as_index=False)["y"].sum()
    df = df.set_index("ds").sort_index()

    # If weekly, align to dominant weekday anchor
    if freq_code.startswith("W"):
        wk = df.index.to_series().dt.weekday.mode()
        if not wk.empty:
            anchor = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"][int(wk.iat[0])]
            freq_code = f"W-{anchor}"

    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq_code)
    df = df.reindex(full_idx)
    df.index.name = "ds"
    df["y"] = df["y"].ffill().bfill()
    return df.reset_index().rename(columns={"index": "ds"})

def make_future_dates(last_ds, periods, freq):
    if periods <= 0:
        return pd.DatetimeIndex([])
    start = pd.date_range(last_ds, periods=2, freq=freq)[-1]
    return pd.date_range(start=start, periods=periods, freq=freq)

def forecast_prophet(df, horizon, freq_code):
    if not HAVE_PROPHET:
        raise RuntimeError("Prophet is not available in this environment.")
    model_df = ensure_monotonic_frequency(df[["ds","y"]], freq_code)
    m = Prophet()
    m.fit(model_df)
    future = m.make_future_dataframe(periods=horizon, freq=freq_code)
    fcst = m.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    cutoff = model_df["ds"].max()
    return fcst[fcst["ds"] > cutoff].reset_index(drop=True)

def forecast_autoarima(df, horizon, freq_code):
    if not HAVE_PMDARIMA:
        raise RuntimeError("pmdarima (AutoARIMA) is not available in this environment.")
    model_df = ensure_monotonic_frequency(df[["ds","y"]], freq_code)
    series = model_df.set_index("ds")["y"]
    arima = auto_arima(series, seasonal=True, stepwise=True, suppress_warnings=True, error_action="ignore")
    preds, conf = arima.predict(n_periods=horizon, return_conf_int=True, alpha=0.05)
    future_ds = make_future_dates(series.index.max(), horizon, freq_code)
    return pd.DataFrame({"ds": future_ds, "yhat": preds, "yhat_lower": conf[:,0], "yhat_upper": conf[:,1]})

def forecast_sma(df, horizon, freq_code):
    """
    Simple Moving Average forecast with iterative horizon.
    Window heuristic: D→7, W→4, MS→3. Intervals via in-sample residual std.
    """
    window_map = {"D": 7, "W": 4, "MS": 3}
    win = window_map.get(freq_code, 7)
    model_df = ensure_monotonic_frequency(df[["ds","y"]], freq_code).set_index("ds").sort_index()
    y = model_df["y"].astype(float).copy()

    sma = y.rolling(win, min_periods=max(1, win//2)).mean()
    resid = (y - sma).dropna()
    resid_std = float(resid.std()) if not resid.empty else 0.0

    future_ds = make_future_dates(y.index.max(), horizon, freq_code)
    history = y.tolist()
    yhat = []
    for _ in range(horizon):
        current_win = history[-win:] if len(history) >= win else history
        mean_val = float(np.mean(current_win)) if current_win else float("nan")
        yhat.append(mean_val)
        history.append(mean_val)

    z = 1.96
    lower = [val - z*resid_std for val in yhat]
    upper = [val + z*resid_std for val in yhat]
    return pd.DataFrame({"ds": future_ds, "yhat": yhat, "yhat_lower": lower, "yhat_upper": upper})

def generate_narrative(openai_enabled, forecast_df, last_actual, freq_label, model_name) -> str:
    """Optional AI narrative; deterministic fallback if OpenAI not available."""
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

    prompt = (
        "You are a finance-savvy analyst. Write a crisp narrative (120–170 words) for a CFO "
        "summarizing a time-series forecast. Avoid promises or guarantees. Use precise, factual language.\n\n"
        f"Frequency: {freq_label}\nModel: {model_name}\nLast Actual: {last_actual:,.4f}\n"
        f"Forecast Start: {start_val:,.4f}\nForecast End: {end_val:,.4f}\nDirection: {direction}\n"
        f"Average: {avg:,.4f}\n"
        "Mention that uncertainty bands (yhat_lower, yhat_upper) exist."
    )

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
    except Exception:
        return deterministic + " (Narrative via OpenAI unavailable; showing fallback.)"

def render_plot(hist_df, fcst_df, freq_label, model_name):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hist_df["ds"], hist_df["y"], label="Actual (y)")
    ax.plot(fcst_df["ds"], fcst_df["yhat"], label="Forecast (yhat)")
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

# ---------- Sidebar ----------
with st.sidebar:
    st.header("1) Upload")
    uploaded = st.file_uploader("Upload CSV or Excel (.xlsx, .xls)", type=["csv", "xlsx", "xls", "xlsm"])

    st.header("2) Configure")
    freq_label = st.selectbox("Frequency", options=["Daily", "Weekly", "Monthly"], index=0)
    # Use generic weekly 'W'; ensure_monotonic_frequency will auto-anchor to dominant weekday
    freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}
    freq_code = freq_map[freq_label]

    # Only show methods that are actually available
    model_options = []
    if HAVE_PROPHET:
        model_options.append("Prophet")
    if HAVE_PMDARIMA:
        model_options.append("AutoARIMA")
    model_options.append("Simple Moving Average")
    model_name = st.selectbox("Forecasting method", options=model_options, index=0)

    default_h = 90 if freq_code == "D" else (26 if freq_code == "W" else 12)
    horizon = st.number_input("Forecast horizon (periods)", min_value=1, max_value=1000, value=default_h, step=1)

    st.header("3) Run")
    run_btn = st.button("Generate Forecast", type="primary")

# ---------- Main flow ----------
st.markdown("### Upload")
if uploaded is None:
    st.info("Upload a dataset to begin. Accepted formats: CSV, XLSX, XLS.")
    st.stop()

# Read file
try:
    raw_df = read_input_file(uploaded)
except Exception as e:
    st.error(f"File could not be read: {e}")
    st.stop()

if raw_df.empty or raw_df.shape[1] < 2:
    st.error("The file seems empty or has too few columns.")
    st.stop()

st.success(f"Loaded data with {raw_df.shape[0]:,} rows and {raw_df.shape[1]:,} columns.")
with st.expander("Preview (first 10 rows)"):
    st.dataframe(raw_df.head(10), use_container_width=True)

# Column mapping
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
st.write("Click **Generate Forecast** in the sidebar when ready.")

if not run_btn:
    st.stop()

# Prep data
with st.spinner("Preparing data..."):
    df = raw_df.copy()
    if category_col != "(None)" and category_vals:
        df = df[df[category_col].astype(str).isin(category_vals)].copy()

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

    hist_df = df[["ds", "y"]].copy()
    if len(df) < max(10, min(50, horizon)):
        st.warning("Dataset is small; forecasts may be unstable.")

    # Guard: ensure enough valid numeric rows remain
    st.info(f"Rows after cleaning: {len(hist_df):,} (valid y: {hist_df['y'].notna().sum():,})")
    if hist_df["y"].notna().sum() < 2:
        st.error("After cleaning, fewer than 2 valid numeric rows remain. Check the date/target mapping, remove commas/currency symbols, or export to CSV.")
        st.stop()

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
    st.exception(traceback.format_exc())
    st.stop()

# Results
st.markdown("### Results")
c1, c2 = st.columns([2, 1], gap="large")
with c1:
    try:
        render_plot(hist_df, fcst, freq_label, model_name)
    except Exception:
        st.error("Plotting failed.")
        st.exception(traceback.format_exc())
with c2:
    st.subheader("Narrative summary")
    last_actual = float(hist_df.iloc[-1]["y"])
    narrative = generate_narrative(HAVE_OPENAI, fcst, last_actual, freq_label, model_name)
    st.write(narrative)

st.subheader("Forecast table")
st.dataframe(fcst, use_container_width=True)

meta = {
    "generated_at": datetime.utcnow().isoformat() + "Z",
    "model": model_name,
    "horizon": int(horizon),
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

st.caption("Tip: Add 'openpyxl' and 'xlrd' to requirements for Excel support, and set OPENAI_API_KEY to enable the narrative.")
