"""
01 — Research Overview
======================
Hero metrics, interactive Plotly charts, and latest data updates
for the Flame in Freefall investigation catalog.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.components import metric_card
from dashboard.utils import (
    COLORS,
    apply_nasa_theme,
    format_date,
    get_all_categories,
    get_all_platforms,
    get_color_for_category,
    get_color_for_platform,
    load_catalog,
)


def render() -> None:
    """Render the Research Overview page."""
    st.markdown(
        "<h1 style='text-align:center;'>Research Overview</h1>"
        "<p style='text-align:center;color:#94A3B8;font-size:0.9rem;'>"
        "NASA Spaceflight Combustion Research Portfolio and Metric Summaries"
        "</p>",
        unsafe_allow_html=True,
    )

    catalog = load_catalog()
    if not catalog:
        st.warning("No investigation data available.")
        return

    # ── Hero Metrics ──────────────────────────────────────────────────────────
    all_cats = get_all_categories(catalog)
    all_platforms = get_all_platforms(catalog)
    total_versions = sum(len(inv.get("versions") or inv.get("data_versions") or [1]) for inv in catalog)

    cols = st.columns(4)
    with cols[0]:
        metric_card("Total Investigations", len(catalog))
    with cols[1]:
        metric_card("Research Categories", len(all_cats))
    with cols[2]:
        metric_card("Platforms", len(all_platforms))
    with cols[3]:
        metric_card("Data Versions", total_versions)

    st.markdown("---")

    # ── Charts Row 1 ──────────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    # Donut: Investigations by Category
    with col_left:
        st.subheader("Investigations by Category")
        cat_counts: Counter[str] = Counter()
        for inv in catalog:
            for cat in inv.get("categories", []):
                cat_counts[cat] += 1

        labels = list(cat_counts.keys())
        values = list(cat_counts.values())
        colors = [get_color_for_category(c) for c in labels]

        fig_donut = go.Figure(
            go.Pie(
                labels=labels,
                values=values,
                hole=0.5,
                marker={"colors": colors},
                textinfo="label+value",
                textposition="outside",
                textfont={"size": 11},
            )
        )
        fig_donut = apply_nasa_theme(fig_donut)
        fig_donut.update_layout(
            showlegend=False,
            height=420,
            margin={"l": 20, "r": 20, "t": 20, "b": 20},
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # Bar: Investigations by Platform
    with col_right:
        st.subheader("Investigations by Platform")
        plat_counts: Counter[str] = Counter()
        for inv in catalog:
            plat_counts[inv.get("platform", "Unknown")] += 1

        plat_labels = list(plat_counts.keys())
        plat_values = list(plat_counts.values())
        plat_colors = [get_color_for_platform(p) for p in plat_labels]

        fig_bar = go.Figure(
            go.Bar(
                x=plat_values,
                y=plat_labels,
                orientation="h",
                marker={"color": plat_colors},
                text=plat_values,
                textposition="outside",
            )
        )
        fig_bar = apply_nasa_theme(fig_bar)
        fig_bar.update_layout(
            height=420,
            yaxis={"categoryorder": "total ascending"},
            xaxis_title="Number of Investigations",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # ── Charts Row 2 ──────────────────────────────────────────────────────────
    col_left2, col_right2 = st.columns(2)

    # Timeline: Investigation start dates
    with col_left2:
        st.subheader("Investigation Timeline")
        timeline_data = []
        for inv in catalog:
            sd = inv.get("start_date")
            if sd:
                timeline_data.append(
                    {
                        "date": sd,
                        "title": inv["title"],
                        "status": inv.get("status", "Unknown"),
                        "id": inv["id"],
                    }
                )
        timeline_data.sort(key=lambda x: x["date"])

        if timeline_data:
            fig_timeline = go.Figure()
            status_grouped: dict[str, list] = {}
            for item in timeline_data:
                status_grouped.setdefault(item["status"], []).append(item)

            for status, items in status_grouped.items():
                color = {
                    "Active": COLORS["success"],
                    "Completed": COLORS["nasa_blue"],
                    "Planned": COLORS["warning"],
                }.get(status, COLORS["text_muted"])
                fig_timeline.add_trace(
                    go.Scatter(
                        x=[i["date"] for i in items],
                        y=[i["id"] for i in items],
                        mode="markers+text",
                        marker={"size": 14, "color": color, "symbol": "diamond"},
                        text=[i["id"] for i in items],
                        textposition="middle right",
                        textfont={"size": 10},
                        name=status,
                        hovertext=[i["title"] for i in items],
                        hoverinfo="text+x",
                    )
                )
            fig_timeline = apply_nasa_theme(fig_timeline)
            fig_timeline.update_layout(
                height=420,
                xaxis_title="Start Date",
                showlegend=True,
                legend={"orientation": "h", "y": -0.15},
            )
            st.plotly_chart(fig_timeline, use_container_width=True)

    # Stacked bar: Category distribution by Platform
    with col_right2:
        st.subheader("Categories by Platform")
        cross: dict[str, Counter[str]] = {}
        for inv in catalog:
            plat = inv.get("platform", "Unknown")
            for cat in inv.get("categories", []):
                cross.setdefault(plat, Counter())[cat] += 1

        all_unique_cats = sorted({cat for cnts in cross.values() for cat in cnts})
        platforms_sorted = sorted(cross.keys())

        fig_stacked = go.Figure()
        for cat in all_unique_cats:
            fig_stacked.add_trace(
                go.Bar(
                    name=cat,
                    x=platforms_sorted,
                    y=[cross.get(p, Counter()).get(cat, 0) for p in platforms_sorted],
                    marker_color=get_color_for_category(cat),
                )
            )
        fig_stacked = apply_nasa_theme(fig_stacked)
        fig_stacked.update_layout(
            barmode="stack",
            height=420,
            showlegend=True,
            legend={"font": {"size": 9}, "orientation": "h", "y": -0.35},
            xaxis={"tickangle": -25, "tickfont": {"size": 9}},
        )
        st.plotly_chart(fig_stacked, use_container_width=True)

    st.markdown("---")

    # ── Latest Data Updates ───────────────────────────────────────────────────
    st.subheader("Investigation Catalog Updates")

    all_versions = []
    for inv in catalog:
        for v in inv.get("data_versions", []):
            all_versions.append({**v, "investigation": inv["title"], "inv_id": inv["id"]})

    all_versions.sort(key=lambda x: x.get("date", ""), reverse=True)

    for v in all_versions[:8]:
        col_a, col_b, col_c = st.columns([3, 2, 5])
        with col_a:
            st.markdown(f"**{v['investigation']}**")
        with col_b:
            st.markdown(f"`v{v['version']}` — {format_date(v.get('date'))}")
        with col_c:
            st.markdown(f"_{v.get('notes', '')}_")
