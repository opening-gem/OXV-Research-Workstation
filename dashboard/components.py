"""
Flame in Freefall — Reusable Streamlit Components
===================================================
Styled components for investigation cards, category badges,
score bars, comparison tables, and citations.
"""

from __future__ import annotations

import html
from typing import Any

import streamlit as st

from dashboard.utils import (
    COLORS,
    STATUS_COLORS,
    format_date,
    get_color_for_category,
    get_color_for_platform,
    truncate_text,
)


# ── Metric Card ──────────────────────────────────────────────────────────────

def metric_card(label: str, value: str | int, icon: str | None = None, delta: str | None = None) -> None:
    """Render a styled metric card using Streamlit's native metric + custom wrapper."""
    delta_color = COLORS["success"] if delta and delta.startswith("+") else COLORS["text_muted"]
    esc_label = html.escape(str(label))
    esc_val = html.escape(str(value))
    esc_delta = html.escape(delta) if delta else ""
    icon_html = f'<div class="metric-icon">{icon}</div>' if icon else ""
    st.markdown(
        f"""
        <div class="metric-card">
            {icon_html}
            <div class="metric-value">{esc_val}</div>
            <div class="metric-label">{esc_label}</div>
            {"<div class='metric-delta' style='color:" + delta_color + "'>" + esc_delta + "</div>" if delta else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Category Badges ──────────────────────────────────────────────────────────

def category_badges(categories: list[str]) -> str:
    """Return HTML for a row of colored category badges."""
    badges = []
    for cat in categories:
        color = get_color_for_category(cat)
        esc_cat = html.escape(str(cat))
        badges.append(
            f'<span class="category-badge" style="background:{color}18;color:{color};'
            f'border:1px solid {color}40;border-radius:3px;font-family:monospace;font-size:0.75rem;padding:2px 6px;">{esc_cat}</span>'
        )
    return " ".join(badges)


def render_category_badges(categories: list[str]) -> None:
    """Render category badges into the Streamlit app."""
    st.markdown(category_badges(categories), unsafe_allow_html=True)


# ── Status Badge ─────────────────────────────────────────────────────────────

def status_badge(status: str) -> str:
    """Return HTML for a status badge."""
    color = STATUS_COLORS.get(status, COLORS["text_muted"])
    esc_status = html.escape(str(status))
    return (
        f'<span class="status-badge" style="background:{color}20;color:{color};'
        f'border:1px solid {color}50;border-radius:3px;font-family:monospace;font-size:0.72rem;padding:2px 6px;">[{esc_status.upper()}]</span>'
    )


def platform_badge(platform: str) -> str:
    """Return HTML for a platform badge."""
    color = get_color_for_platform(platform)
    esc_plat = html.escape(str(platform).upper())
    return (
        f'<span class="platform-badge" style="background:{color}18;color:{color};'
        f'border:1px solid {color}40;border-radius:3px;font-family:monospace;font-size:0.72rem;padding:2px 6px;">[{esc_plat}]</span>'
    )


# ── Investigation Card ───────────────────────────────────────────────────────

def investigation_card(inv: dict[str, Any], show_details: bool = True) -> None:
    """Render a full investigation card."""
    st.markdown(
        f"""
        <div class="inv-card">
            <div class="inv-card-header">
                <h3 style="margin:0;color:{COLORS['white']};font-size:1.05rem;">{inv['title']}</h3>
                <div style="display:flex;gap:6px;margin-top:6px;flex-wrap:wrap;">
                    {status_badge(inv.get('status', 'Unknown'))}
                    {platform_badge(inv.get('platform', 'Unknown'))}
                </div>
            </div>
            <p style="color:{COLORS['text_muted']};margin:8px 0;font-size:0.86rem;line-height:1.5;">
                {truncate_text(inv.get('description', ''), 200)}
            </p>
            <div style="margin-bottom:6px;">
                {category_badges(inv.get('categories', []))}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if show_details:
        with st.expander("View Specifications & Objectives"):
            _render_investigation_details(inv)


def _render_investigation_details(inv: dict[str, Any]) -> None:
    """Render the expanded details of an investigation."""
    st.markdown(f"**Description:** {inv.get('description', 'N/A')}")

    # Timeline
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Start Date:** {format_date(inv.get('start_date'))}")
    with col2:
        st.markdown(f"**End Date:** {format_date(inv.get('end_date'))}")

    # Objectives
    objectives = inv.get("objectives", [])
    if objectives:
        st.markdown("**Objectives:**")
        for obj in objectives:
            st.markdown(f"- {obj}")

    # Investigators
    investigators = inv.get("investigators", [])
    if investigators:
        st.markdown("**Investigators:**")
        for person in investigators:
            name = person.get("name", "Unknown")
            affil = person.get("affiliation") or person.get("institution", "NASA")
            role = f" ({person['role']})" if person.get("role") else ""
            st.markdown(f"- **{name}**{role} — {affil}")

    # Applications
    applications = inv.get("applications", [])
    if applications:
        st.markdown("**Applications:**")
        for app in applications:
            st.markdown(f"- {app}")

    # Data versions
    versions = inv.get("versions") or inv.get("data_versions") or []
    if versions:
        st.markdown("**Data Versions:**")
        for v in versions:
            if isinstance(v, dict):
                ver_num = v.get("version_number") or v.get("version", 1)
                doi_val = v.get("doi")
                doi_link = f"https://doi.org/{doi_val}" if doi_val else "#"
                v_date = format_date(v.get("release_date") or v.get("date"))
                v_desc = v.get("description") or v.get("notes", "")
                st.markdown(
                    f"- **v{ver_num}** ({v_date}) — [{doi_val or 'N/A'}]({doi_link}) — {v_desc}"
                )
            else:
                st.markdown(f"- **v{v}**")

    # Source
    PSI_DIRECT_URLS = {
        "BASS-II": "https://psi.nasa.gov/physci/repo/data/investigations/PSI-25",
        "BASS": "https://psi.nasa.gov/physci/repo/data/investigations/PSI-26",
        "ACME": "https://psi.nasa.gov/physci/repo/data/investigations/PSI-10",
        "ACME-BRE": "https://psi.nasa.gov/physci/repo/data/investigations/PSI-10",
        "ACME-CFI": "https://psi.nasa.gov/physci/repo/data/investigations/PSI-159",
        "ACME-E-FIELD": "https://psi.nasa.gov/physci/repo/data/investigations/PSI-10",
        "FLEX": "https://psi.nasa.gov/physci/repo/data/investigations/PSI-68",
        "FLEX-2": "https://psi.nasa.gov/physci/repo/data/investigations/PSI-69",
        "SAFFIRE-I": "https://psi.nasa.gov/physci/repo/data/investigations/PSI-98",
        "SAFFIRE-II": "https://psi.nasa.gov/physci/repo/data/investigations/PSI-99",
        "SAFFIRE-III": "https://psi.nasa.gov/physci/repo/data/investigations/PSI-100",
        "SAME": "https://psi.nasa.gov/physci/repo/data/investigations/PSI-102",
        "SPICE": "https://psi.nasa.gov/physci/repo/data/investigations/PSI-60",
        "CLD": "https://psi.nasa.gov/physci/repo/data/investigations/PSI-10",
        "SOFIE": "https://psi.nasa.gov/physci/repo/data/investigations/PSI-25",
        "SoFIE": "https://psi.nasa.gov/physci/repo/data/investigations/PSI-25",
        "FSDC": "https://www1.grc.nasa.gov/space/combustion-and-reacting-systems/",
    }
    inv_id = inv.get("id") or inv.get("investigation_id")
    source_url = PSI_DIRECT_URLS.get(inv_id) or inv.get("psi_url") or inv.get("source_url") or "https://psi.nasa.gov/physci/repo/"
    st.markdown(f"[NASA PSI Repository Record &nearr;]({source_url})")


# ── Score Visualization ──────────────────────────────────────────────────────

def score_bar(score: float, label: str = "", max_score: float = 1.0, color: str | None = None) -> str:
    """Return HTML for a horizontal score bar."""
    pct = min(score / max_score * 100, 100) if max_score > 0 else 0
    bar_color = color or COLORS["fire_orange"]
    return (
        f'<div class="score-bar-container">'
        f'<div class="score-bar-label">{label}</div>'
        f'<div class="score-bar-track">'
        f'<div class="score-bar-fill" style="width:{pct:.0f}%;background:{bar_color};"></div>'
        f'</div>'
        f'<div class="score-bar-value">{score:.2f}</div>'
        f'</div>'
    )


def render_score_breakdown(scores: dict[str, float]) -> None:
    """Render a set of score bars for ranking breakdown."""
    color_map = {
        "keyword": COLORS["fire_orange"],
        "semantic": COLORS["nasa_blue"],
        "metadata": COLORS["success"],
        "evidence": COLORS["purple"],
    }
    html_parts = []
    for key, val in scores.items():
        c = color_map.get(key, COLORS["text_muted"])
        html_parts.append(score_bar(val, label=key.capitalize(), color=c))
    st.markdown("".join(html_parts), unsafe_allow_html=True)


# ── Source Citation ──────────────────────────────────────────────────────────

def source_citation(inv: dict[str, Any], snippet: str = "") -> None:
    """Render a formatted source citation block."""
    st.markdown(
        f"""
        <div class="source-citation">
            <strong>{inv.get('title', 'Unknown')}</strong>
            <span style="color:{COLORS['text_muted']}"> — {inv.get('id', '')}</span>
            <br/>
            <span style="color:{COLORS['text_muted']};font-size:0.85em;">{snippet}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Comparison Table Helper ──────────────────────────────────────────────────

def comparison_row(label: str, val_a: str, val_b: str) -> str:
    """Return one row of a comparison table as HTML."""
    return (
        f"<tr>"
        f"<td style='color:{COLORS['fire_orange']};font-weight:600;padding:8px 12px;'>{label}</td>"
        f"<td style='padding:8px 12px;'>{val_a}</td>"
        f"<td style='padding:8px 12px;'>{val_b}</td>"
        f"</tr>"
    )
