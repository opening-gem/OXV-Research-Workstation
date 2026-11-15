"""
02 — Investigation Explorer
============================
Searchable, filterable browser for the investigation catalog.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from dashboard.components import (
    category_badges,
    investigation_card,
    platform_badge,
    status_badge,
)
from dashboard.utils import (
    COLORS,
    get_all_categories,
    get_all_platforms,
    get_all_statuses,
    load_catalog,
    truncate_text,
)


def _matches_search(inv: dict[str, Any], query: str) -> bool:
    """Check if an investigation matches a search query."""
    if not query:
        return True
    q = query.lower()
    searchable = " ".join(
        [
            inv.get("title", ""),
            inv.get("description", ""),
            inv.get("id", ""),
            " ".join(inv.get("categories", [])),
            inv.get("platform", ""),
            " ".join(inv.get("objectives", [])),
        ]
    ).lower()
    return q in searchable


def render() -> None:
    """Render the Investigation Explorer page."""
    st.markdown(
        "<h1>Investigation Catalog Explorer</h1>"
        "<p style='color:#94A3B8;font-size:0.9rem;'>Query, filter, and inspect canonical microgravity combustion flight records</p>",
        unsafe_allow_html=True,
    )

    catalog = load_catalog()
    if not catalog:
        st.warning("No investigation data available.")
        return

    # ── Search Bar ────────────────────────────────────────────────────────────
    search_query = st.text_input(
        "Search investigations",
        placeholder="Type a keyword, fuel name, platform, or topic (e.g., PMMA, droplet, suppression)...",
        key="explorer_search",
    )

    # ── Filters ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Filters")

        selected_cats = st.multiselect(
            "Categories",
            options=get_all_categories(catalog),
            key="filter_cats",
        )
        selected_platforms = st.multiselect(
            "Platforms",
            options=get_all_platforms(catalog),
            key="filter_platforms",
        )
        selected_statuses = st.multiselect(
            "Status",
            options=get_all_statuses(catalog),
            key="filter_statuses",
        )

    # ── Apply Filters ─────────────────────────────────────────────────────────
    filtered = catalog.copy()

    # Search
    if search_query:
        filtered = [inv for inv in filtered if _matches_search(inv, search_query)]

    # Category filter
    if selected_cats:
        filtered = [
            inv
            for inv in filtered
            if any(c in selected_cats for c in inv.get("categories", []))
        ]

    # Platform filter
    if selected_platforms:
        filtered = [
            inv for inv in filtered if inv.get("platform") in selected_platforms
        ]

    # Status filter
    if selected_statuses:
        filtered = [
            inv for inv in filtered if inv.get("status") in selected_statuses
        ]

    # ── Results ───────────────────────────────────────────────────────────────
    st.markdown(
        f"<p style='color:{COLORS['text_muted']};'>Showing <strong>{len(filtered)}</strong> "
        f"of {len(catalog)} investigations</p>",
        unsafe_allow_html=True,
    )

    if not filtered:
        st.info("No investigations match your filters. Try broadening your search.")
        return

    for inv in filtered:
        investigation_card(inv, show_details=True)
        st.markdown("<div style='margin-bottom:16px;'></div>", unsafe_allow_html=True)
