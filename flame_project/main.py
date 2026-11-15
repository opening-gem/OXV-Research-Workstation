"""CLI:  python main.py --csv data/BASS-II.csv
Optional: --format "Generic (already tidy)" --oxygen COL --airflow COL --outcome COL
          --bins 8 --min-samples 3 --binning quantile"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from analysis import all_tables
from config import DEFAULTS
from data_loader import default_labels, detect_columns, load_data
from plots import all_plots
from preprocessing import prepare

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--format", default="Auto")
ap.add_argument("--oxygen")
ap.add_argument("--airflow")
ap.add_argument("--outcome")
ap.add_argument("--sustained", nargs="*", help="labels meaning sustained burning")
ap.add_argument("--extinct", nargs="*", help="labels meaning extinction")
ap.add_argument("--bins", type=int, default=DEFAULTS["n_bins"])
ap.add_argument("--binning", choices=["uniform", "quantile"], default="quantile")
ap.add_argument("--min-samples", type=int, default=DEFAULTS["min_samples"])
ap.add_argument("--out", default=DEFAULTS["output_dir"])
a = ap.parse_args()

df = load_data(a.csv, fmt=a.format)
print("Format:", df.attrs["format"])
if df.attrs.get("notes"):
    print("NOTE:", df.attrs["notes"])
cols = detect_columns(df, a.oxygen, a.airflow, a.outcome)
print("Using columns:", cols)
sus, ext = default_labels(df, cols["outcome"])
d = prepare(df, cols, a.sustained or sus, a.extinct or ext)
print(f"Rows used: {len(d)} (dropped {d.attrs['dropped']} with missing oxygen/airflow)")
print(d["outcome_raw"].value_counts().to_string())

out = Path(a.out)
out.mkdir(exist_ok=True, parents=True)
for name, fig in all_plots(d, a.bins, a.min_samples, a.binning).items():
    fig.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
for name, t in all_tables(d, a.bins, a.binning).items():
    t.to_csv(out / f"{name}.csv")
print("Saved to", out.resolve())
