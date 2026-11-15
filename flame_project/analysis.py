"""Numeric summaries (tables) that back up the plots."""
import numpy as np
import pandas as pd
try:
    from flame_project.preprocessing import grid_stats
except ImportError:
    from preprocessing import grid_stats


def summary_by_outcome(d):
    return d.groupby("outcome_raw")[["oxygen", "airflow"]].describe().round(3)


def grid_table(d, n_bins, method="uniform"):
    rate, n, xc, yc = grid_stats(d, n_bins, method)
    rows = [dict(oxygen=xc[c], airflow=yc[r], n=n[r, c], sustain_rate=rate[r, c])
            for r in range(len(yc)) for c in range(len(xc)) if n[r, c] > 0]
    return pd.DataFrame(rows)


def correlations(d):
    x = d[["oxygen", "airflow", "sustained"]].dropna()
    return x.corr().round(3) if len(x) > 2 else pd.DataFrame()


def all_tables(d, n_bins, method="uniform"):
    return {"summary_by_outcome": summary_by_outcome(d),
            "grid_rates": grid_table(d, n_bins, method=method),
            "correlations": correlations(d)}
