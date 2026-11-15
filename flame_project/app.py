"""UI:  streamlit run app.py
Change any setting in the sidebar and every graph redraws."""
import io
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from config import DEFAULTS
from adapters import ADAPTERS
from data_loader import load_data, detect_columns, default_labels
from preprocessing import prepare, to_binary
from plots import all_plots
from analysis import all_tables

st.set_page_config(page_title="OXV — Telemetry & Extinction Studio", layout="wide")

# DaVinci Resolve / Blender workstation aesthetic
st.markdown("""
<style>
    .stApp {
        background-color: #121418;
        color: #E2E8F0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    section[data-testid="stSidebar"] {
        background-color: #161920;
        border-right: 1px solid #272C37;
    }
    div[data-testid="metric-container"] {
        background-color: #181B22;
        border: 1px solid #272C37;
        border-radius: 3px;
        padding: 10px 14px;
    }
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        color: #F1F5F9;
    }
    div[data-testid="stMetricLabel"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #8C96A5;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 1px solid #272C37;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #8C96A5;
        border-radius: 2px 2px 0 0;
    }
    .stTabs [aria-selected="true"] {
        color: #F08C2E !important;
        border-bottom-color: #F08C2E !important;
        font-weight: 600;
    }
    button[kind="secondary"], .stButton>button {
        background-color: #1C2029;
        border: 1px solid #2D3342;
        color: #E2E8F0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.76rem;
        border-radius: 3px;
        letter-spacing: 0.04em;
    }
    button[kind="secondary"]:hover, .stButton>button:hover {
        background-color: #242A36;
        border-color: #F08C2E;
        color: #F08C2E;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<p style='font-family:monospace; font-size:0.7rem; letter-spacing:0.12em; color:#8C96A5; text-transform:uppercase;'>NASA OXV — ORBITAL EXPERIMENTATION VOLUMETRIC</p>", unsafe_allow_html=True)
st.title("Oxygen-Airflow Flame Outcome Analysis")

# ---- 1. data source -------------------------------------------------------
st.sidebar.header("1. Data")
up = st.sidebar.file_uploader("Upload CSV/Excel", type=["csv", "xlsx", "xls"])
local = sorted(p.name for p in Path("data").glob("*") if p.suffix in (".csv", ".xlsx", ".xls"))
choice = st.sidebar.selectbox("...or pick from data/ folder", ["-"] + local)
fmt = st.sidebar.selectbox("Dataset format", ["Auto", "Generic (already tidy)"] + list(ADAPTERS))
if up is not None:
    df = load_data(up, up.name, fmt)
elif choice != "-":
    df = load_data(Path("data") / choice, fmt=fmt)
else:
    st.info("Upload a file or put one in the data/ folder.")
    st.stop()
st.sidebar.caption(f"Format used: {df.attrs['format']}")
if df.attrs.get("notes"):
    st.info(df.attrs["notes"])

# ---- 2. columns -----------------------------------------------------------
st.sidebar.header("2. Columns")
guess = detect_columns(df)
cols_list = list(df.columns)
pick = lambda label, role: st.sidebar.selectbox(label, cols_list, cols_list.index(guess[role]))
cols = {"oxygen": pick("Oxygen", "oxygen"), "airflow": pick("Airflow", "airflow"),
        "outcome": pick("Outcome", "outcome")}

# ---- 3. outcome meaning ---------------------------------------------------
st.sidebar.header("3. Outcome meaning")
labels = sorted(df[cols["outcome"]].astype(str).unique())
if len(labels) > 30:
    st.sidebar.warning("Outcome has many unique values - is this the right column?")
sus0, ext0 = default_labels(df, cols["outcome"])
if sus0 is None:
    uniq = df[cols["outcome"]].astype(str).drop_duplicates()
    auto = to_binary(uniq); auto.index = uniq
    sus0 = [l for l in labels if auto.get(l) == 1]
    ext0 = [l for l in labels if auto.get(l) == 0]
sus = st.sidebar.multiselect("Sustained burning", labels, [l for l in sus0 if l in labels])
ext = st.sidebar.multiselect("Extinction / not sustained", labels, [l for l in ext0 if l in labels])

# ---- 4. analysis settings -------------------------------------------------
st.sidebar.header("4. Settings")
n_bins = st.sidebar.slider("Bins per axis", 3, 20, DEFAULTS["n_bins"])
method = st.sidebar.radio("Binning", ["quantile", "uniform"], horizontal=True,
                          help="quantile = equal counts per bin; better for skewed airflow")
min_n = st.sidebar.slider("Min samples per cell", 1, 20, DEFAULTS["min_samples"])

d = prepare(df, cols, sus, ext)
lab = dict(d.attrs)

# ---- 5. filters (ranges + any categorical column) -------------------------
st.sidebar.header("5. Filters")
for col, name in [("oxygen", "Oxygen"), ("airflow", "Airflow")]:
    lo, hi = float(d[col].min()), float(d[col].max())
    if lo < hi:
        r = st.sidebar.slider(f"{name} range", lo, hi, (lo, hi))
        d = d[d[col].between(*r)]
skip = {"oxygen", "airflow", "outcome_raw", "sustained", cols["outcome"]}
cat_cols = [c for c in d.columns if c not in skip and 1 < d[c].nunique() <= 12
            and not pd.api.types.is_float_dtype(d[c])]
with st.sidebar.expander("Filter by category (fuel, PI, ...)"):
    for c in cat_cols:
        opts = sorted(d[c].astype(str).unique())
        keep = st.multiselect(c, opts, opts, key=f"f_{c}")
        d = d[d[c].astype(str).isin(keep)]

d.attrs.update(lab)

# ---- main area ------------------------------------------------------------
if d.empty:
    st.error("No rows left after filtering."); st.stop()
m = st.columns(5)
m[0].metric("Rows used", len(d))
m[1].metric("Sustained", int((d.sustained == 1).sum()))
m[2].metric("Not sustained", int((d.sustained == 0).sum()))
m[3].metric("Unassigned", int(d.sustained.isna().sum()))
m[4].metric("Dropped (no O2/airflow)", d.attrs.get("dropped", 0))
if d.sustained.isna().any():
    st.warning("Some outcomes are unassigned - set them in the sidebar (section 3). Rate plots skip them.")

tab_plots, tab_tables, tab_data = st.tabs(["Graphs", "Tables", "Data"])
with tab_plots:
    items = list(all_plots(d, n_bins, min_n, method).items())
    for i in range(0, len(items), 2):
        row = st.columns(2)
        for col, (name, fig) in zip(row, items[i:i + 2]):
            col.pyplot(fig)
            buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            col.download_button("Download PNG", buf.getvalue(), f"{name}.png", "image/png", key=name)
            plt.close(fig)
with tab_tables:
    for name, t in all_tables(d, n_bins, method).items():
        st.subheader(name); st.dataframe(t)
with tab_data:
    st.dataframe(d)
    st.download_button("Download cleaned CSV", d.to_csv(index=False), "cleaned.csv")
