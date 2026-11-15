"""
06 — Data Provenance and Updates
================================
Dataset versions, persistent DOIs, audit history, and traceability metrics.
"""
from __future__ import annotations

import json
from pathlib import Path
import streamlit as st
from dashboard.utils import load_catalog
from src.ingestion.version_tracker import VersionTracker


def render() -> None:
    st.markdown(
        "<h1 style='text-align:center;'>Data Provenance & Version Audit</h1>"
        "<p style='text-align:center;color:#94A3B8;font-size:0.9rem;'>"
        "Tracking canonical dataset versions, DOIs, and scientific attribution across the catalogue"
        "</p>",
        unsafe_allow_html=True,
    )

    catalog = load_catalog()
    tracker = VersionTracker()

    st.markdown("### NASA Provenance Standards")
    st.info(
        "NASA Physical Sciences Informatics (PSI) requires persistent identifiers (DOIs) for formal "
        "attribution and scientific repeatability. When newer data releases are published, active version "
        "pointers advance without destroying prior versions."
    )

    # Provenance metrics
    total_invs = len(catalog)
    with_doi = sum(1 for inv in catalog if ((inv.get("versions") or [{}])[0].get("doi") or inv.get("doi")))
    total_versions = sum(len(inv.get("versions") or [1]) for inv in catalog)

    c1, c2, c3 = st.columns(3)
    c1.metric("Indexed Records", total_invs)
    c2.metric("Tracked Versions", total_versions)
    c3.metric("DOI Attributed", f"{with_doi}/{total_invs}")

    st.markdown("### Canonical Investigation Version Register")
    rows = []
    for inv in catalog:
        active_ver_num = inv.get("active_version") or inv.get("version", 1)
        active_ver = (inv.get("versions") or [{}])[0]
        doi = active_ver.get("doi") or inv.get("doi") or "—"
        doi_link = f"https://doi.org/{doi}" if doi != "—" else inv.get("source_url", "#")
        rows.append(
            {
                "ID": inv.get("id"),
                "Title": inv.get("title"),
                "Active Version": f"v{active_ver_num}",
                "DOI": doi,
                "DOI Link": doi_link,
                "Platform": str(inv.get("platform", "")).upper(),
                "Status": str(inv.get("status", "")).capitalize(),
            }
        )

    # Render HTML table with sleek workstation styling
    table_html = """
    <table style='width:100%; border-collapse:collapse; background:#1B1E24; border:1px solid #2B313C; border-radius:4px; overflow:hidden;'>
        <thead>
            <tr style='background:#222630; border-bottom:1px solid #2B313C; color:#F8FAFC; text-align:left; font-family:monospace; font-size:0.75rem; letter-spacing:0.05em;'>
                <th style='padding:10px 12px;'>ID</th>
                <th style='padding:10px 12px;'>INVESTIGATION</th>
                <th style='padding:10px 12px;'>VERSION</th>
                <th style='padding:10px 12px;'>DOI / PERSISTENT LINK</th>
                <th style='padding:10px 12px;'>PLATFORM</th>
                <th style='padding:10px 12px;'>STATUS</th>
            </tr>
        </thead>
        <tbody>
    """
    for r in rows:
        doi_col = f"<a href='{r['DOI Link']}' target='_blank' style='color:#38BDF8; font-family:monospace; text-decoration:none;'>{r['DOI']}</a>" if r['DOI'] != "—" else "<span style='color:#64748B; font-family:monospace;'>Unassigned</span>"
        table_html += f"""
            <tr style='border-bottom:1px solid #232832;'>
                <td style='padding:10px 12px; font-weight:bold; font-family:monospace; color:#F8FAFC;'>{r['ID']}</td>
                <td style='padding:10px 12px; color:#E2E8F0; font-size:0.86rem;'>{r['Title']}</td>
                <td style='padding:10px 12px; font-family:monospace; color:#38BDF8;'>{r['Active Version']}</td>
                <td style='padding:10px 12px;'>{doi_col}</td>
                <td style='padding:10px 12px; font-family:monospace; color:#E2E8F0; font-size:0.8rem;'>{r['Platform']}</td>
                <td style='padding:10px 12px; font-family:monospace; color:#10B981; font-size:0.8rem;'>{r['Status']}</td>
            </tr>
        """
    table_html += "</tbody></table>"
    st.markdown(table_html, unsafe_allow_html=True)

    # Manifest viewer
    manifest_path = Path(__file__).resolve().parents[2] / "data" / "manifests" / "ingestion_manifest.json"
    if manifest_path.exists():
        st.markdown("### Latest Ingestion Manifest")
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        st.json(manifest_data)
