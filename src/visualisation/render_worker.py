"""
Worker script to render scientific microgravity telemetry plots and numerical tables.
Called either directly or as a subprocess by the standalone server / CLI.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure temporary directory for matplotlib cache
os.environ.setdefault("MPLCONFIGDIR", "/tmp")

ROOT_DIR = Path(__file__).resolve().parents[2]
FLAME_PROJECT_DIR = ROOT_DIR / "flame_project"
sys.path.insert(0, str(FLAME_PROJECT_DIR))

from adapters import ADAPTERS
from data_loader import load_data
from preprocessing import prepare
from plots import all_plots, fig_to_base64_png
from analysis import all_tables

INTERPRETATIONS = {
    "01_scatter_outcomes": (
        "Maps individual microgravity test points in (Oxygen, Airflow) parameter space. "
        "Separation between sustained combustion, extinction, and blowoff regimes delineates "
        "the experimental boundary where convective cooling or oxygen starvation dominates."
    ),
    "02_sustain_rate_heatmap": (
        "Two-dimensional discretization showing fraction of sustained combustion per grid cell. "
        "Cells with sample count below minimum threshold are masked to prevent ungrounded extrapolation."
    ),
    "03_data_coverage": (
        "Flight test coverage density indicating total valid experimental runs per grid cell. "
        "High-density nodes represent well-tested operating conditions; zero-density zones "
        "highlight critical mission telemetry gaps."
    ),
    "04_rate_vs_oxygen": (
        "Combustion sustain probability as a function of oxygen concentration, stratified into "
        "discrete airflow bands. Demonstrates the shifting flammability threshold across flow regimes."
    ),
    "05_rate_vs_airflow": (
        "Combustion sustain probability versus forced convective airflow speed, stratified by "
        "oxygen concentration bands. Demonstrates non-monotonic extinction limits observed in orbit."
    ),
    "06_distributions": (
        "Box-and-whisker statistical distributions of key environmental parameters grouped by "
        "observed physical outcome, displaying medians, interquartile ranges, and outlier thresholds."
    ),
    "07_boundary": (
        "Empirical 50% sustain probability boundary curve across flow regimes. Delineates safe "
        "spacecraft operational envelopes from sustained flame propagation hazards."
    ),
    "08_burn_risk_surface": (
        "Continuous 2D parametric burn risk surface (R_burn in [0, 1]) mapped across the operational "
        "parameter envelope. Quantifies the escalation of flame propagation hazard as oxygen concentration "
        "increases or convective flow enters the peak burning regime, delineating safe spacecraft operational envelopes."
    ),
}

PLOT_TITLES = {
    "01_scatter_outcomes": "Experimental Observations Scatter",
    "02_sustain_rate_heatmap": "2D Sustained Burning Rate Matrix",
    "03_data_coverage": "Flight Test Coverage Density",
    "04_rate_vs_oxygen": "Sustain Rate vs Oxygen Concentration",
    "05_rate_vs_airflow": "Sustain Rate vs Airflow Velocity",
    "06_distributions": "Parameter Boxplot Distributions",
    "07_boundary": "50% Flammability Extinction Boundary",
    "08_burn_risk_surface": "Parametric Burn Risk Surface & Hazard Contours",
}


def load_dataset_bundle(dataset_name: str, n_bins: int = 8, min_samples: int = 2, method: str = "uniform", requested_plot: str = "all") -> dict:
    data_dir = ROOT_DIR / "data" / "raw" / "telemetry"
    if not data_dir.exists():
        data_dir = FLAME_PROJECT_DIR / "data"

    if dataset_name.upper().startswith("FLEX"):
        csv_file = data_dir / "PSI-69_Experimental_table_FLEX.csv"
        fmt = "FLEX (raw)"
        canonical_name = "FLEX (Droplet Combustion Experiment)"
        facility = "Combustion Integrated Rack (CIR)"
    else:
        csv_file = data_dir / "BASS-II.csv"
        fmt = "BASS-II (raw)"
        canonical_name = "BASS-II (Burning & Suppression of Solids - II)"
        facility = "Microgravity Science Glovebox (MSG)"

    if not csv_file.exists():
        raise FileNotFoundError(f"Telemetry file {csv_file} not found.")

    raw_df = load_data(csv_file, fmt=fmt)
    prepared_df = prepare(
        raw_df,
        raw_df.attrs["suggested"],
        raw_df.attrs.get("sustained_labels"),
        raw_df.attrs.get("extinct_labels")
    )

    total_rows = len(prepared_df)
    sustained_cnt = int((prepared_df["sustained"] == 1).sum())
    extinct_cnt = int((prepared_df["sustained"] == 0).sum())
    unassigned_cnt = int(prepared_df["sustained"].isna().sum())

    outcomes_breakdown = prepared_df["outcome_raw"].value_counts().to_dict()

    # Generate plots
    plot_figs = all_plots(prepared_df, n_bins, min_samples, method)
    plots_dict = {}

    for k, fig in plot_figs.items():
        if requested_plot != "all" and requested_plot != k:
            import matplotlib.pyplot as plt
            plt.close(fig)
            continue
        b64_img = fig_to_base64_png(fig)
        plots_dict[k] = {
            "key": k,
            "title": PLOT_TITLES.get(k, k),
            "image": f"data:image/png;base64,{b64_img}",
            "interpretation": INTERPRETATIONS.get(k, "")
        }

    # Generate tables
    tables_raw = all_tables(prepared_df, n_bins, method)
    summary_df = tables_raw.get("summary_by_outcome")
    grid_df = tables_raw.get("grid_rates")
    corr_df = tables_raw.get("correlations")

    summary_records = []
    if summary_df is not None:
        flat_summary = summary_df.copy()
        flat_summary.columns = [
            "_".join(str(part) for part in col if part).strip() if isinstance(col, tuple) else str(col)
            for col in flat_summary.columns
        ]
        flat_summary = flat_summary.reset_index()
        summary_records = json.loads(flat_summary.to_json(orient="records"))

    grid_records = []
    if grid_df is not None:
        grid_records = json.loads(grid_df.to_json(orient="records"))

    corr_records = []
    if corr_df is not None:
        corr_flat = corr_df.reset_index()
        corr_records = json.loads(corr_flat.to_json(orient="records"))

    # Parametric Burn Risk Summary (computed over valid, outcome-assigned rows)
    valid_sub = prepared_df.dropna(subset=["sustained"])
    valid_rows = len(valid_sub)
    overall_risk = float(sustained_cnt / valid_rows) if valid_rows > 0 else 0.0

    o2_slope = 0.0
    if len(valid_sub) >= 5 and float(valid_sub["oxygen"].std()) > 0:
        cov = float(np.cov(valid_sub["oxygen"].values, valid_sub["sustained"].values)[0, 1])
        var = float(np.var(valid_sub["oxygen"].values))
        if var > 0:
            o2_slope = (cov / var) * 100.0

    peak_flow_band = "N/A"
    if len(valid_sub) >= 5 and "airflow" in valid_sub:
        try:
            flow_groups = valid_sub.groupby(pd.qcut(valid_sub["airflow"], q=min(4, int(valid_sub["airflow"].nunique())), duplicates="drop"), observed=True)["sustained"].mean()
            if not flow_groups.empty:
                peak_interval = flow_groups.idxmax()
                peak_flow_band = f"{float(peak_interval.left):.1f} - {float(peak_interval.right):.1f}"
        except Exception:
            pass

    risk_tier = "CRITICAL HAZARD" if overall_risk >= 0.75 else "ELEVATED HAZARD" if overall_risk >= 0.50 else "MODERATE HAZARD" if overall_risk >= 0.25 else "LOW RISK / SAFE"

    risk_summary = {
        "overall_burn_risk": round(overall_risk, 3),
        "overall_burn_risk_pct": f"{overall_risk * 100:.1f}%",
        "risk_tier": risk_tier,
        "oxygen_risk_sensitivity": f"{o2_slope:+.2f}% risk per +1% O2",
        "peak_hazard_flow_band": peak_flow_band,
    }

    tables_dict = {
        "summary": summary_records,
        "grid_rates": grid_records,
        "correlations": corr_records,
    }

    return {
        "dataset_key": dataset_name,
        "dataset_name": canonical_name,
        "facility": facility,
        "notes": raw_df.attrs.get("notes", ""),
        "total_rows": total_rows,
        "sustained_count": sustained_cnt,
        "extinguished_count": extinct_cnt,
        "unassigned_count": unassigned_cnt,
        "outcomes_breakdown": outcomes_breakdown,
        "risk_summary": risk_summary,
        "n_bins": n_bins,
        "min_samples": min_samples,
        "method": method,
        "plots": plots_dict,
        "tables": tables_dict,
    }


def main():
    parser = argparse.ArgumentParser(description="Render microgravity telemetry plots.")
    parser.add_argument("--dataset", default="BASS-II", choices=["BASS-II", "FLEX"])
    parser.add_argument("--plot", default="all")
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--min_samples", type=int, default=2)
    parser.add_argument("--method", default="uniform", choices=["uniform", "quantile"])

    args = parser.parse_args()
    bundle = load_dataset_bundle(
        dataset_name=args.dataset,
        n_bins=args.bins,
        min_samples=args.min_samples,
        method=args.method,
        requested_plot=args.plot
    )
    print(json.dumps(bundle))


if __name__ == "__main__":
    main()
