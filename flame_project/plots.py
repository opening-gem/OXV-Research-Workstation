"""
Scientific plotting engine for microgravity combustion telemetry.
Generates publication-quality, dark-theme scientific figures with parametric burn risk modeling.
"""
import base64
import io
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
try:
    from flame_project.preprocessing import assign_bins, grid_stats
except ImportError:
    from preprocessing import assign_bins, grid_stats

# Scientific Telemetry Palette (Grey-Black Theme, Zero Emojis)
COLOR_SUSTAINED = "#10B981"     # Telemetry Emerald (Safe/Sustained Ignition)
COLOR_EXTINGUISHED = "#F59E0B"  # Telemetry Amber (Extinction / Boundary)
COLOR_BLOWOFF = "#F43F5E"       # Telemetry Rose (High-Flow Blowoff)
COLOR_NO_IGNITION = "#64748B"   # Muted Slate
COLOR_CYAN = "#38BDF8"          # Precision Cyan
COLOR_PURPLE = "#A855F7"        # Spectral Violet

OUTCOME_COLORS = {
    "Sustained": COLOR_SUSTAINED,
    "Completion": COLOR_SUSTAINED,
    "Extinguished": COLOR_EXTINGUISHED,
    "Extinction": COLOR_EXTINGUISHED,
    "Blowoff": COLOR_BLOWOFF,
    "No ignition": COLOR_NO_IGNITION,
    "Disruption": COLOR_PURPLE,
}

# Scientific Dark Colormaps
CMAP_HEATMAP = LinearSegmentedColormap.from_list(
    "scientific_telemetry",
    ["#0A0D14", "#0F213A", "#0369A1", "#0D9488", "#10B981", "#F59E0B"],
    N=256
)

CMAP_COVERAGE = LinearSegmentedColormap.from_list(
    "scientific_density",
    ["#080A0F", "#131C2E", "#1E3A8A", "#0284C7", "#38BDF8"],
    N=256
)

# Burn Risk Hazard Index Colormap (Dark Emerald -> Cyan -> Amber -> Crimson)
CMAP_RISK = LinearSegmentedColormap.from_list(
    "burn_risk_index",
    [
        "#081C15",  # Zero/Low Risk (Safe Quenching Zone)
        "#0D3B2E",  # Minimal Risk
        "#113E4D",  # Low-Moderate Transition
        "#1A425F",  # Moderate Flammability Boundary
        "#63340F",  # Elevated Hazard
        "#8B2516",  # High Hazard
        "#B91C1C",  # Critical Burn Hazard
    ],
    N=256
)


def apply_scientific_style():
    """Apply high-precision scientific instrument dark theme."""
    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.facecolor": "#090B0E",
        "axes.facecolor": "#11141C",
        "axes.edgecolor": "#232A3B",
        "axes.linewidth": 0.8,
        "axes.labelcolor": "#94A3B8",
        "axes.labelsize": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "600",
        "axes.titlecolor": "#F8FAFC",
        "xtick.color": "#64748B",
        "ytick.color": "#64748B",
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "grid.color": "#1A202C",
        "grid.linestyle": ":",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.8,
        "legend.facecolor": "#0D1017",
        "legend.edgecolor": "#232A3B",
        "legend.fontsize": 8.0,
        "legend.framealpha": 0.9,
        "font.family": "sans-serif",
    })


def X(d):
    return d.attrs.get("xlabel", "Oxygen concentration")


def Y(d):
    return d.attrs.get("ylabel", "Airflow speed")


def _fmt(v):
    return f"{v:.3g}"


def _axes_ticks(ax, xc, yc):
    ax.set_xticks(range(len(xc)))
    ax.set_xticklabels([_fmt(v) for v in xc], rotation=40, ha="right", fontfamily="monospace")
    ax.set_yticks(range(len(yc)))
    ax.set_yticklabels([_fmt(v) for v in yc], fontfamily="monospace")


def fig_to_base64_png(fig, dpi=160):
    """Render Matplotlib figure to Base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _compute_risk_surface(ds, grid_res=50):
    """Compute 2D continuous parametric burn risk factor grid via kernel regression."""
    x = ds["oxygen"].values
    y = ds["airflow"].values
    z = ds["sustained"].values

    x_min, x_max = x.min() - 0.5, x.max() + 0.5
    y_min, y_max = max(0.0, y.min() - 0.5), y.max() + 0.5

    gx = np.linspace(x_min, x_max, grid_res)
    gy = np.linspace(y_min, y_max, grid_res)
    GX, GY = np.meshgrid(gx, gy)

    sx = max(0.8, float(np.std(x)) * 0.35)
    sy = max(0.8, float(np.std(y)) * 0.35)

    dx = (GX[:, :, None] - x[None, None, :]) / sx
    dy = (GY[:, :, None] - y[None, None, :]) / sy
    weights = np.exp(-0.5 * (dx**2 + dy**2))
    denom = np.sum(weights, axis=-1) + 1e-6
    risk = np.sum(weights * z[None, None, :], axis=-1) / denom
    return gx, gy, GX, GY, risk


def scatter_outcomes(d):
    """Plot observations with background parametric burn risk isocontours."""
    apply_scientific_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ds = d.dropna(subset=["sustained"])

    # Background subtle risk contours
    if len(ds) >= 10:
        _, _, GX, GY, risk = _compute_risk_surface(ds, grid_res=40)
        c_risk = ax.contour(GX, GY, risk, levels=[0.25, 0.5, 0.75], colors=["#1E293B", "#38BDF8", "#EF4444"],
                            linewidths=[0.8, 1.2, 1.2], linestyles=[":", "--", "-"], alpha=0.6)
        ax.clabel(c_risk, inline=True, fmt={0.25: "Low Risk (0.25)", 0.5: "Boundary (0.50)", 0.75: "Crit Hazard (0.75)"},
                  fontsize=6.8)

    rng = np.random.default_rng(0)
    jx, jy = 0.012 * np.ptp(d["oxygen"]), 0.012 * np.ptp(d["airflow"])
    unique_outcomes = sorted(d["outcome_raw"].unique())
    fallback_cmap = plt.get_cmap("tab10")

    for i, c in enumerate(unique_outcomes):
        m = d["outcome_raw"] == c
        color = OUTCOME_COLORS.get(c, fallback_cmap(i % 10))
        ax.scatter(
            d.loc[m, "oxygen"] + rng.normal(0, jx, m.sum()),
            (d.loc[m, "airflow"] + rng.normal(0, jy, m.sum())).clip(lower=min(0, d["airflow"].min())),
            s=36,
            alpha=0.85,
            color=color,
            label=f"{c} (n={m.sum()})",
            edgecolor="#080A0D",
            linewidth=0.5,
        )

    ax.set(xlabel=X(d), ylabel=Y(d), title=f"OBSERVATIONS & BURN RISK ZONES: {X(d)} vs {Y(d)}")
    leg = ax.legend(loc="upper left")
    leg.get_frame().set_alpha(0.85)
    ax.grid(True)
    return fig


def rate_heatmap(d, n_bins, min_samples, method="uniform"):
    """Plot 2D sustain rate matrix with explicit burn risk classification tags per cell."""
    apply_scientific_style()
    rate, n, xc, yc = grid_stats(d, n_bins, method)
    masked = np.where(n >= min_samples, rate, np.nan)

    fig, ax = plt.subplots(figsize=(8.5, 6))
    im = ax.imshow(masked, origin="lower", aspect="auto", cmap=CMAP_HEATMAP, vmin=0, vmax=1)
    _axes_ticks(ax, xc, yc)

    for (r, c), v in np.ndenumerate(masked):
        if not np.isnan(v):
            txt_color = "#080A0D" if v > 0.6 else "#F8FAFC"
            tag = "CRIT" if v >= 0.75 else "ELEV" if v >= 0.50 else "MOD" if v >= 0.25 else "LOW"
            ax.text(c, r, f"{v:.0%}\n[{tag}]\nn={n[r, c]}", ha="center", va="center", fontsize=6.8,
                    fontfamily="monospace", color=txt_color, weight="600")

    cb = fig.colorbar(im, label="Burn Risk Factor (Fraction Sustained Burning)")
    cb.ax.yaxis.set_tick_params(color="#64748B", labelsize=8)
    ax.set(xlabel=X(d), ylabel=Y(d),
           title=f"SUSTAINED BURNING RATE MATRIX (Cells n<{min_samples} masked; [RISK TIERS])")
    return fig


def coverage_map(d, n_bins, method="uniform"):
    """Plot flight test density per parameter grid node."""
    apply_scientific_style()
    _, n, xc, yc = grid_stats(d, n_bins, method)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(n, origin="lower", aspect="auto", cmap=CMAP_COVERAGE)
    _axes_ticks(ax, xc, yc)

    cb = fig.colorbar(im, label="Flight Sample Count")
    cb.ax.yaxis.set_tick_params(color="#64748B", labelsize=8)
    ax.set(xlabel=X(d), ylabel=Y(d), title="DATA COVERAGE DENSITY (Flight Runs per Grid Cell)")
    return fig


def rate_by_band(d, x, n_bins, min_samples, method="uniform"):
    """Plot burn sustain rate vs parameter, with shaded risk factor tiers and secondary rating axis."""
    apply_scientific_style()
    band = "airflow" if x == "oxygen" else "oxygen"
    lab = {"oxygen": X(d), "airflow": Y(d)}
    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    # Risk factor zone backgrounds
    ax.axhspan(0.75, 1.05, facecolor="#EF4444", alpha=0.07, zorder=0, label="_nolegend_")
    ax.axhspan(0.50, 0.75, facecolor="#F59E0B", alpha=0.07, zorder=0, label="_nolegend_")
    ax.axhspan(0.25, 0.50, facecolor="#38BDF8", alpha=0.06, zorder=0, label="_nolegend_")
    ax.axhspan(-0.05, 0.25, facecolor="#10B981", alpha=0.06, zorder=0, label="_nolegend_")

    nb = min(4, d[band].nunique())
    bands = pd.qcut(d[band], nb, duplicates="drop") if d[band].nunique() > 4 else d[band]
    xi, xc = assign_bins(d[x], n_bins, method)

    band_colors = [COLOR_CYAN, COLOR_SUSTAINED, COLOR_EXTINGUISHED, COLOR_PURPLE]

    for idx, (name, g) in enumerate(d.assign(xb=xi, band=bands).dropna(subset=["sustained"]).groupby("band", observed=True)):
        s = g.groupby("xb")["sustained"].agg(["mean", "size"])
        s = s[s["size"] >= min_samples]
        if len(s):
            col = band_colors[idx % len(band_colors)]
            ax.plot(xc[s.index], s["mean"], marker="s", markersize=4.8, lw=1.8, color=col,
                    label=f"{lab[band]}: {name}")

    ax.set(xlabel=lab[x], ylabel="Fraction Sustained (Burn Risk Factor)", ylim=(-0.05, 1.05),
           title=f"PARAMETRIC BURN RISK vs {lab[x]} (Stratified by {lab[band]})")
    ax.legend(fontsize=7.8, loc="upper left")
    ax.grid(True, alpha=0.4)

    # Secondary twin axis showing Hazard Level
    ax2 = ax.twinx()
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_yticks([0.125, 0.375, 0.625, 0.875])
    ax2.set_yticklabels(["LOW RISK", "MODERATE", "ELEVATED", "CRITICAL"], fontsize=7.2, color="#94A3B8")
    ax2.set_ylabel("Burn Risk Hazard Classification", color="#94A3B8", fontsize=8.0)
    ax2.tick_params(colors="#64748B", length=0)
    return fig


def distributions(d):
    """Plot statistical distribution boxplots grouped by physical outcome."""
    apply_scientific_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    cats = sorted(d["outcome_raw"].unique())

    for ax, col in zip(axes, ["oxygen", "airflow"]):
        data = [d.loc[d["outcome_raw"] == c, col] for c in cats]
        ax.boxplot(data, tick_labels=cats, patch_artist=True, showmeans=True,
                   boxprops=dict(facecolor="#18202F", color="#38BDF8", linewidth=0.8),
                   whiskerprops=dict(color="#64748B", linewidth=0.8),
                   capprops=dict(color="#64748B", linewidth=0.8),
                   medianprops=dict(color="#F59E0B", linewidth=1.2),
                   meanprops=dict(marker="o", markeredgecolor="#10B981", markerfacecolor="#10B981", markersize=4))
        ax.set(title=f"DISTRIBUTION: {X(d) if col == 'oxygen' else Y(d)}",
               ylabel=X(d) if col == "oxygen" else Y(d))
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True)

    fig.tight_layout()
    return fig


def boundary_plot(d, n_bins, min_samples, method="uniform"):
    """Plot empirical 50% flammability boundary with risk regime shading."""
    apply_scientific_style()
    rate, n, xc, yc = grid_stats(d, n_bins, method)
    ok = (n >= min_samples) & (rate >= 0.5)
    pts = [(xc[np.argmax(row)], yc[i]) for i, row in enumerate(ok) if row.any()]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ds = d.dropna(subset=["sustained"])

    # Observations
    point_colors = [COLOR_SUSTAINED if s == 1 else COLOR_EXTINGUISHED for s in ds["sustained"]]
    ax.scatter(ds["oxygen"], ds["airflow"], c=point_colors, s=26, alpha=0.75,
               edgecolor="#080A0D", linewidth=0.4, label="Flight Runs (Green=Sustained, Amber=Extinct)")

    if pts:
        p = np.array(pts)
        ax.plot(p[:, 0], p[:, 1], color=COLOR_CYAN, linestyle="--", marker="o",
                markersize=6, lw=2.0, label="50% Flammability Risk Threshold")

    # Add text annotations for risk envelopes
    ax.text(0.98, 0.08, "SAFE / QUENCHING ENVELOPE\n(Burn Risk < 50%)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
            color="#10B981", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#091410", edgecolor="#10B981", alpha=0.8))

    ax.text(0.98, 0.92, "PROPAGATION HAZARD ZONE\n(Burn Risk >= 50%)",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            color="#F43F5E", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#1F0D12", edgecolor="#F43F5E", alpha=0.8))

    ax.set(xlabel=X(d), ylabel=Y(d),
           title=f"FLAMMABILITY EXTINCTION BOUNDARY & RISK ENVELOPES")
    ax.legend(loc="upper left", fontsize=7.8)
    ax.grid(True)
    return fig


def burn_risk_surface(d, n_bins=8, min_samples=2):
    """Plot continuous 2D parametric burn risk surface with hazard contours and sensitivity."""
    apply_scientific_style()
    ds = d.dropna(subset=["sustained"])
    x = ds["oxygen"].values
    y = ds["airflow"].values

    fig, ax = plt.subplots(figsize=(8.8, 6.0))

    gx, gy, GX, GY, risk = _compute_risk_surface(ds, grid_res=60)

    # 2D Filled Risk Contours
    levels = np.linspace(0.0, 1.0, 11)
    cf = ax.contourf(GX, GY, risk, levels=levels, cmap=CMAP_RISK, alpha=0.85)

    # Critical 50% Boundary Isoline
    c_line = ax.contour(GX, GY, risk, levels=[0.5], colors=["#F8FAFC"], linewidths=1.8, linestyles="--")
    ax.clabel(c_line, inline=True, fmt={0.5: "50% Boundary (R=0.50)"}, fontsize=7.8)

    # Secondary Hazard Isolines (0.25 Low, 0.75 High)
    c_sub = ax.contour(GX, GY, risk, levels=[0.25, 0.75], colors=["#94A3B8", "#EF4444"],
                       linewidths=[1.0, 1.2], linestyles=[":", "-."])
    ax.clabel(c_sub, inline=True, fmt={0.25: "Low Risk (0.25)", 0.75: "Crit Hazard (0.75)"}, fontsize=7.0)

    # Scatter observations
    point_colors = [COLOR_SUSTAINED if s == 1 else COLOR_EXTINGUISHED for s in ds["sustained"]]
    ax.scatter(x, y, c=point_colors, s=28, alpha=0.85, edgecolor="#090B0E", linewidth=0.5,
               label="Flight Runs (Green=Sustained, Amber=Extinct)")

    # Colorbar
    cb = fig.colorbar(cf, label="Calculated Parametric Burn Risk Factor R_burn [0.0 - 1.0]")
    cb.ax.yaxis.set_tick_params(color="#64748B", labelsize=8)

    ax.set(
        xlabel=X(d),
        ylabel=Y(d),
        title=f"PARAMETRIC BURN RISK FACTOR SURFACE & HAZARD CONTOURS"
    )
    ax.legend(loc="upper left", fontsize=7.8)
    ax.grid(True, alpha=0.35)
    return fig


def all_plots(d, n_bins, min_samples, method="uniform"):
    """name -> Figure, in display order."""
    return {
        "01_scatter_outcomes": scatter_outcomes(d),
        "02_sustain_rate_heatmap": rate_heatmap(d, n_bins, min_samples, method),
        "03_data_coverage": coverage_map(d, n_bins, method),
        "04_rate_vs_oxygen": rate_by_band(d, "oxygen", n_bins, min_samples, method),
        "05_rate_vs_airflow": rate_by_band(d, "airflow", n_bins, min_samples, method),
        "06_distributions": distributions(d),
        "07_boundary": boundary_plot(d, n_bins, min_samples, method),
        "08_burn_risk_surface": burn_risk_surface(d, n_bins, min_samples),
    }
