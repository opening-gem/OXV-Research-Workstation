"""
04 — Investigation Comparison
=============================
Side-by-side comparative analysis of NASA combustion research projects.
"""
from __future__ import annotations

import streamlit as st
from dashboard.utils import load_catalog


def render() -> None:
    st.markdown(
        "<h1 style='text-align:center;'>Investigation Comparison</h1>"
        "<p style='text-align:center;color:#94A3B8;font-size:0.9rem;'>"
        "Directly contrast experimental objectives, platforms, and technological implications"
        "</p>",
        unsafe_allow_html=True,
    )

    catalog = load_catalog()
    if len(catalog) < 2:
        st.error("Need at least two investigations in catalog to compare.")
        return

    inv_ids = [inv.get("id") for inv in catalog]
    inv_map = {inv.get("id"): inv for inv in catalog}

    col_a, col_b = st.columns(2)
    default_a = "BASS-II" if "BASS-II" in inv_ids else inv_ids[0]
    default_b = "ACME" if "ACME" in inv_ids else inv_ids[1]

    id_a = col_a.selectbox("Select Investigation A:", inv_ids, index=inv_ids.index(default_a))
    id_b = col_b.selectbox("Select Investigation B:", inv_ids, index=inv_ids.index(default_b))

    inv_a = inv_map[id_a]
    inv_b = inv_map[id_b]

    # Category sets
    cats_a = set(inv_a.get("categories", []))
    cats_b = set(inv_b.get("categories", []))
    shared_cats = cats_a.intersection(cats_b)
    unique_a = cats_a - cats_b
    unique_b = cats_b - cats_a

    st.markdown("### Category Overlap & Domain Focus")
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"**Unique to {id_a}:**\n" + ("\n".join(f"- `{c}`" for c in unique_a) or "None"))
    c2.markdown(f"**Shared Research Areas:**\n" + ("\n".join(f"- **`{c}`**" for c in shared_cats) or "None"))
    c3.markdown(f"**Unique to {id_b}:**\n" + ("\n".join(f"- `{c}`" for c in unique_b) or "None"))

    st.markdown("---")
    st.markdown("### Side-by-Side Specifications")
    comp_col1, comp_col2 = st.columns(2)

    with comp_col1:
        st.markdown(f"#### {inv_a.get('title')} [{id_a}]")
        st.write(f"**Full Title:** {inv_a.get('full_name', '')}")
        st.write(f"**Platform:** {str(inv_a.get('platform', '')).upper()}")
        st.write(f"**Facility:** {inv_a.get('facility', 'N/A')}")
        ver_num_a = inv_a.get('active_version') or inv_a.get('version', 1)
        st.write(f"**Active Version:** v{ver_num_a}")
        active_ver_a = (inv_a.get('versions') or [{}])[0]
        doi_a = active_ver_a.get('doi') or inv_a.get('doi') or 'Unassigned'
        st.write(f"**DOI:** `{doi_a}`")
        st.write("**Scientific Objectives:**")
        for obj in inv_a.get("objectives", []):
            st.markdown(f"- {obj}")
        st.write("**Practical Applications:**")
        for app in inv_a.get("applications", []):
            st.markdown(f"- {app}")

    with comp_col2:
        st.markdown(f"#### {inv_b.get('title')} [{id_b}]")
        st.write(f"**Full Title:** {inv_b.get('full_name', '')}")
        st.write(f"**Platform:** {str(inv_b.get('platform', '')).upper()}")
        st.write(f"**Facility:** {inv_b.get('facility', 'N/A')}")
        ver_num_b = inv_b.get('active_version') or inv_b.get('version', 1)
        st.write(f"**Active Version:** v{ver_num_b}")
        active_ver_b = (inv_b.get('versions') or [{}])[0]
        doi_b = active_ver_b.get('doi') or inv_b.get('doi') or 'Unassigned'
        st.write(f"**DOI:** `{doi_b}`")
        st.write("**Scientific Objectives:**")
        for obj in inv_b.get("objectives", []):
            st.markdown(f"- {obj}")
        st.write("**Practical Applications:**")
        for app in inv_b.get("applications", []):
            st.markdown(f"- {app}")
