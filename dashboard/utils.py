"""
Flame in Freefall — Dashboard Utilities
========================================
Helper functions for data loading, color mapping, formatting,
and reusable styled components.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_CATALOG_PATH = PROJECT_ROOT / "data" / "processed" / "catalog.json"
SEED_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "psi" / "seed_investigations.json"

# ── NASA Theme Colors ─────────────────────────────────────────────────────────
COLORS = {
    "bg_primary": "#0B0D17",
    "bg_secondary": "#1B1D2A",
    "bg_card": "#232538",
    "fire_orange": "#FF6B35",
    "nasa_blue": "#0B3D91",
    "white": "#FFFFFF",
    "text_muted": "#A0A4B8",
    "success": "#28D89C",
    "warning": "#FFD166",
    "danger": "#EF476F",
    "cyan": "#06D6A0",
    "purple": "#9B5DE5",
}

# ── Category Colors ───────────────────────────────────────────────────────────
_CATEGORY_PALETTE: dict[str, str] = {
    "Flame Spread": "#FF6B35",
    "Soot Formation": "#FFD166",
    "Combustion Fundamentals": "#0B3D91",
    "Fire Safety": "#EF476F",
    "Fire Suppression": "#06D6A0",
    "Material Flammability": "#9B5DE5",
    "Fire Detection": "#FF9F1C",
    "Large-Scale Fire": "#E63946",
    "Smoke Dynamics": "#8D99AE",
    "Sensor Technology": "#2EC4B6",
    "Droplet Combustion": "#3A86FF",
    "Cool Flames": "#00B4D8",
    "Flame Extinguishment": "#F77F00",
    "Low-Temperature Chemistry": "#7209B7",
    "Spray Combustion": "#F94144",
    "Smoldering Combustion": "#E76F51",
    "Atmospheric Recovery": "#4CC9F0",
    "Gas-Jet Flames": "#F9C74F",
    "Flame Stability": "#43AA8B",
}

# ── Platform Colors ───────────────────────────────────────────────────────────
_PLATFORM_PALETTE: dict[str, str] = {
    "International Space Station (ISS)": "#0B3D91",
    "Cygnus Resupply Vehicle": "#FF6B35",
    "Parabolic Flight (Zero-G Aircraft)": "#9B5DE5",
    "Drop Tower (Bremen)": "#06D6A0",
    "Sounding Rocket (TEXUS)": "#FFD166",
}

# ── Status Colors ─────────────────────────────────────────────────────────────
STATUS_COLORS: dict[str, str] = {
    "Active": "#28D89C",
    "Completed": "#0B3D91",
    "Planned": "#FFD166",
}


def get_color_for_category(category: str) -> str:
    """Return a consistent hex color for a research category."""
    if category in _CATEGORY_PALETTE:
        return _CATEGORY_PALETTE[category]
    # Deterministic hash-based fallback
    h = hash(category) % 360
    return f"hsl({h}, 70%, 55%)"


def get_color_for_platform(platform: str) -> str:
    """Return a consistent hex color for a research platform."""
    return _PLATFORM_PALETTE.get(platform, "#A0A4B8")


# ── Data Loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_catalog() -> list[dict[str, Any]]:
    """Load investigation catalog from processed catalog, or fallback to raw seed data."""
    target_path = PROCESSED_CATALOG_PATH if PROCESSED_CATALOG_PATH.exists() else SEED_DATA_PATH
    if not target_path.exists():
        st.error(
            f"Investigation data not found at `{target_path}`. "
            "Run `python main.py ingest` to build the catalogue."
        )
        return []

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            data: list[dict[str, Any]] = json.load(f)
        return data
    except Exception as e:
        st.error(f"Error loading catalogue from {target_path}: {e}")
        return []


def get_all_categories(catalog: list[dict[str, Any]]) -> list[str]:
    """Extract and sort all unique categories from the catalog."""
    cats: set[str] = set()
    for inv in catalog:
        cats.update(inv.get("categories", []))
    return sorted(cats)


def get_all_platforms(catalog: list[dict[str, Any]]) -> list[str]:
    """Extract and sort all unique platforms from the catalog."""
    return sorted({inv.get("platform", "Unknown") for inv in catalog})


def get_all_statuses(catalog: list[dict[str, Any]]) -> list[str]:
    """Extract and sort all unique statuses from the catalog."""
    return sorted({inv.get("status", "Unknown") for inv in catalog})


# ── Formatting Helpers ────────────────────────────────────────────────────────

def format_date(date_str: str | None, fmt: str = "%b %d, %Y") -> str:
    """Format an ISO date string for display."""
    if not date_str:
        return "—"
    try:
        return datetime.fromisoformat(date_str).strftime(fmt)
    except (ValueError, TypeError):
        return date_str


def truncate_text(text: str, max_len: int = 150) -> str:
    """Truncate text to *max_len* characters with an ellipsis."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rsplit(" ", 1)[0] + "…"


# ── Plotly Theme ──────────────────────────────────────────────────────────────

PLOTLY_LAYOUT_DEFAULTS: dict[str, Any] = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": COLORS["white"], "family": "Inter, sans-serif"},
    "margin": {"l": 40, "r": 20, "t": 40, "b": 40},
    "colorway": [
        COLORS["fire_orange"],
        COLORS["nasa_blue"],
        COLORS["success"],
        COLORS["warning"],
        COLORS["purple"],
        COLORS["cyan"],
        COLORS["danger"],
        "#3A86FF",
        "#F9C74F",
        "#43AA8B",
    ],
}


def apply_nasa_theme(fig: Any) -> Any:
    """Apply the NASA dark theme to a Plotly figure."""
    fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
    fig.update_xaxes(gridcolor="rgba(160,164,184,0.15)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(160,164,184,0.15)", zeroline=False)
    return fig
