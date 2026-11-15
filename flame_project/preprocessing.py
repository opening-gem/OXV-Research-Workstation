"""Cleaning, binary outcome mapping, and binning helpers."""
import re
import numpy as np
import pandas as pd
try:
    from flame_project.config import POS_PREFIX, NEG_PREFIX
except ImportError:
    from config import POS_PREFIX, NEG_PREFIX


def to_binary(series, sustained_labels=None, extinct_labels=None):
    """Map outcomes to 1 (sustained) / 0 (extinct); others -> NaN.
    Explicit label lists win; otherwise guess from the text."""
    if sustained_labels is not None or extinct_labels is not None:
        pos = {str(x) for x in (sustained_labels or [])}
        neg = {str(x) for x in (extinct_labels or [])}
        return series.astype(str).map(lambda v: 1.0 if v in pos else (0.0 if v in neg else np.nan))

    if pd.api.types.is_numeric_dtype(series) and series.dropna().isin([0, 1]).all():
        return series.astype(float)

    def guess(v):
        if pd.isna(v):
            return np.nan
        val_str = str(v).strip().lower()
        if any(w in val_str for w in ("unknown", "unclear", "undetermined", "no data", "none", "n/a")):
            return np.nan
        toks = [t for t in re.split(r"[^a-z0-9]+", val_str) if t]
        if any(t in ("nominal", "normal") for t in toks) and any(t in ("burn", "flame", "ignit") for t in toks):
            return 1.0
        filtered_toks = [t for t in toks if t not in ("nominal", "normal", "outcome", "number")]
        if any(t.startswith(NEG_PREFIX) for t in filtered_toks):
            return 0.0
        if any(t.startswith(POS_PREFIX) for t in toks):
            return 1.0
        return np.nan
    return series.map(guess)


def prepare(df, cols, sustained_labels=None, extinct_labels=None):
    """Standardise to columns: oxygen, airflow, outcome_raw, sustained."""
    d = df.copy()   # keep every original column so the UI can filter on them
    d["oxygen"] = pd.to_numeric(df[cols["oxygen"]], errors="coerce")
    d["airflow"] = pd.to_numeric(df[cols["airflow"]], errors="coerce")
    d["outcome_raw"] = df[cols["outcome"]].astype(str)
    d["sustained"] = to_binary(df[cols["outcome"]], sustained_labels, extinct_labels)
    n0 = len(d)
    d = d.dropna(subset=["oxygen", "airflow"]).reset_index(drop=True)
    d.attrs["dropped"] = n0 - len(d)
    d.attrs["xlabel"], d.attrs["ylabel"] = cols["oxygen"], cols["airflow"]
    return d


def assign_bins(s, n_bins, method="uniform"):
    """Bin index + centres. Few distinct values -> one bin per value.
    method: 'uniform' (equal width) or 'quantile' (equal counts; better for skewed data)."""
    uniq = np.sort(s.unique())
    if len(uniq) <= n_bins:
        return s.map({v: i for i, v in enumerate(uniq)}).astype(int), uniq
    if method == "quantile":
        edges = np.unique(np.quantile(s, np.linspace(0, 1, n_bins + 1)))
        n_bins = len(edges) - 1
    else:
        edges = np.linspace(s.min(), s.max(), n_bins + 1)
    idx = np.clip(np.digitize(s, edges) - 1, 0, n_bins - 1)
    return pd.Series(idx, index=s.index), (edges[:-1] + edges[1:]) / 2


def grid_stats(d, n_bins, method="uniform"):
    """2D grids (airflow rows x oxygen cols): sustain rate, counts, centres."""
    xi, xc = assign_bins(d["oxygen"], n_bins, method)
    yi, yc = assign_bins(d["airflow"], n_bins, method)
    t = pd.DataFrame({"x": xi, "y": yi, "s": d["sustained"]})
    shape = dict(index=range(len(yc)), columns=range(len(xc)))
    # Sample count per bin matching the rows with valid evaluated outcomes:
    n = t.dropna(subset=["s"]).groupby(["y", "x"]).size().unstack(fill_value=0).reindex(**shape, fill_value=0)
    rate = t.dropna().groupby(["y", "x"])["s"].mean().unstack().reindex(**shape)
    return rate.values, n.values, xc, yc
