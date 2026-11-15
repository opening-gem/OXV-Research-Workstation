"""
03 — AI Research Ranking
========================
Semantic & hybrid relevance ranking of NASA investigations based on user research questions.
"""
from __future__ import annotations

import streamlit as st
from dashboard.utils import load_catalog
from src.ai.embeddings import EmbeddingsManager
from src.ai.ranker import ResearchRanker


def render() -> None:
    st.markdown(
        "<h1 style='text-align:center;'>Research Relevance Ranking</h1>"
        "<p style='text-align:center;color:#94A3B8;font-size:0.9rem;'>"
        "Multi-factor scoring explaining relevance across keyword, semantic, and experimental evidence"
        "</p>",
        unsafe_allow_html=True,
    )

    catalog = load_catalog()
    if not catalog:
        st.error("Investigation catalog is empty.")
        return

    st.markdown("### Example Exploration Queries")
    col1, col2, col3, col4 = st.columns(4)
    query_input = ""
    if col1.button("Lunar Fire Detection"):
        query_input = "fire detection systems for lunar habitats"
    if col2.button("Solid Flame Spread"):
        query_input = "how do flames spread on materials in space"
    if col3.button("Droplet Extinction"):
        query_input = "droplet combustion and extinction dynamics in microgravity"
    if col4.button("Active Fire Suppression"):
        query_input = "fire suppression methods and extinguishant agents for spacecraft"

    query = st.text_input(
        "Enter your research query or space exploration question:",
        value=query_input or "fire detection and smoke sensors for space habitats",
    )

    if not query.strip():
        st.info("Enter a question above to rank investigations.")
        return

    # Create TF-IDF embeddings on the fly
    em = EmbeddingsManager()
    corpus = [inv.get("description", "") + " " + " ".join(inv.get("objectives", [])) for inv in catalog]
    em.fit(corpus)
    ranker = ResearchRanker(embeddings=em)

    ranked_results = ranker.rank(query, catalog, top_k=6)

    st.markdown(f"### Ranked Investigations: *\"{query}\"*")
    for idx, r in enumerate(ranked_results, start=1):
        with st.container():
            st.markdown(
                f"""
                <div style='background: #1B1E24; border: 1px solid #2B313C; border-left: 3px solid #38BDF8; border-radius: 4px; padding: 1rem; margin-bottom: 0.8rem;'>
                    <div style='display:flex; justify-content:space-between; align-items:center;'>
                        <h4 style='margin:0; color:#F8FAFC; font-size:1.02rem;'>#{idx} [{r.investigation_id}] {r.title}</h4>
                        <span style='background:#12151B; color:#38BDF8; border:1px solid #38BDF8; font-family:monospace; font-size:0.75rem; font-weight:600; padding:2px 8px; border-radius:3px;'>
                            RELEVANCE: {r.overall_score * 100:.1f}%
                        </span>
                    </div>
                    <p style='color:#94A3B8; margin-top:0.4rem; font-size:0.86rem; line-height:1.5;'><strong>Key Factor:</strong> {r.explanation}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Component score bars
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Keyword Match", f"{r.component_scores.get('keyword', 0)*100:.1f}%")
            c2.metric("Semantic Sim", f"{r.component_scores.get('semantic', 0)*100:.1f}%")
            c3.metric("Platform/Status", f"{r.component_scores.get('metadata', 0)*100:.1f}%")
            c4.metric("Evidence Quality", f"{r.component_scores.get('evidence', 0)*100:.1f}%")

            with st.expander("Inspect Supporting Objectives & Keywords"):
                if r.matching_keywords:
                    st.write(f"**Matching Query Terms:** {', '.join(r.matching_keywords)}")
                if r.relevant_objectives:
                    st.write("**Directly Relevant Objectives:**")
                    for obj in r.relevant_objectives:
                        st.markdown(f"- {obj}")
