"""
05 — Research Assistant (LLM + RAG)
===================================
Conversational research assistant powered by Ollama and NASA PSI retrieval.
Supports multi-turn dialogue, clickable source citations, and verified fallback.
"""
from __future__ import annotations

import streamlit as st
from dashboard.utils import load_catalog
from src.catalog.repository import ResearchCatalog
from src.rag.assistant import ResearchAssistant
from src.rag.chunker import DocumentChunker
from src.rag.retriever import InvestigationRetriever
from src.rag.ollama_client import OllamaClient


@st.cache_resource
def get_assistant() -> ResearchAssistant:
    """Initialize and cache the research assistant and retrieval index."""
    catalog = ResearchCatalog()
    catalog_data = [inv.model_dump() for inv in catalog.list_all()]
    chunker = DocumentChunker()
    chunks = chunker.chunk_all(catalog_data)
    retriever = InvestigationRetriever()
    retriever.build_index(chunks)
    client = OllamaClient()
    return ResearchAssistant(retriever=retriever, catalog=catalog, llm_client=client)


def render() -> None:
    st.markdown(
        "<h1 style='text-align:center;'>NASA Combustion Research Assistant</h1>"
        "<p style='text-align:center;color:#94A3B8;font-size:0.9rem;'>"
        "Conversational Evidence Synthesis Powered by Ollama and PSI Retrieval"
        "</p>",
        unsafe_allow_html=True,
    )

    assistant = get_assistant()
    health = assistant.llm_client.check_health() if assistant.llm_client else {"status": "unconfigured"}

    # Model & Connection status header
    if health.get("status") == "online":
        target_model = health.get("configured_model", "unknown")
        if health.get("model_available"):
            st.success(f"[STATUS: ONLINE] Ollama Connected — Model `{target_model}` active at `{health.get('base_url')}`")
        else:
            st.warning(
                f"[STATUS: WARNING] Ollama Reachable, but model `{target_model}` was not found. "
                f"Available models: {', '.join(health.get('available_models', [])) or 'None'}. "
                f"Run `ollama pull {target_model}` or switch model."
            )
    else:
        st.info(
            f"[STATUS: OFFLINE] Ollama Offline (`{health.get('base_url', 'http://127.0.0.1:11434')}`). "
            "Running in deterministic template synthesis mode. "
            "To activate local LLM conversational synthesis, launch your Ollama instance."
        )

    # Initialize chat history in session state
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "Welcome! I am the Orbital Experimentation Volumetric (OXV) Research Assistant. "
                    "Ask me anything regarding microgravity combustion experiments (e.g., BASS-II, ACME, FLEX, SAFFIRE), "
                    "fire safety detection, or terrestrial applications."
                ),
                "sources": [],
                "confidence": "high",
                "disclaimer": None,
                "model_name": None,
                "llm_generated": False,
            }
        ]

    # Pre-canned prompt buttons
    st.markdown("##### Suggested Queries")
    col1, col2, col3, col4 = st.columns(4)
    suggested_q = None
    if col1.button("Compare ACME & BASS-II"):
        suggested_q = "What is the difference between ACME and BASS-II?"
    if col2.button("Lunar Fire Detection"):
        suggested_q = "Which NASA investigations study smoke detection and fire safety for lunar habitats?"
    if col3.button("Droplet Cool Flames"):
        suggested_q = "What experiments investigate cool flames and low-temperature droplet combustion?"
    if col4.button("Clear Chat History"):
        st.session_state.chat_messages = []
        st.rerun()

    # Display conversational thread
    for msg in st.session_state.chat_messages:
        role = msg["role"]
        with st.chat_message(role):
            st.markdown(msg["content"])
            if msg.get("disclaimer"):
                st.caption(f"[NOTICE] {msg['disclaimer']}")

            sources = msg.get("sources", [])
            if sources:
                with st.expander(f"Verified Sources Cited ({len(sources)})"):
                    for s in sources:
                        doi_str = f" | DOI: `{s.doi}`" if s.doi else ""
                        ver_str = f" | v{s.active_version}" if s.active_version else ""
                        st.markdown(f"**[{s.investigation_id}] {s.title}**{doi_str}{ver_str}")
                        st.write(s.relevant_text)
                        if s.url:
                            st.markdown(f"[Open Official NASA PSI Record &nearr;]({s.url})")

    # User input
    user_prompt = st.chat_input("Ask a question about NASA combustion research...") or suggested_q

    if user_prompt:
        # Display user message
        st.chat_message("user").markdown(user_prompt)

        # Prepare history for RAG context
        history_for_rag = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.chat_messages
            if m.get("content")
        ]

        with st.chat_message("assistant"):
            with st.spinner("Retrieving NASA evidence & generating response..."):
                response = assistant.answer(
                    question=user_prompt,
                    conversation_history=history_for_rag,
                    use_llm=True,
                )

            st.markdown(response.answer)
            if response.disclaimer:
                st.caption(f"[NOTICE] {response.disclaimer}")

            if response.sources:
                with st.expander(f"Verified Sources Cited ({len(response.sources)})"):
                    for s in response.sources:
                        doi_str = f" | DOI: `{s.doi}`" if s.doi else ""
                        ver_str = f" | v{s.active_version}" if s.active_version else ""
                        st.markdown(f"**[{s.investigation_id}] {s.title}**{doi_str}{ver_str}")
                        st.write(s.relevant_text)
                        if s.url:
                            st.markdown(f"[Open Official NASA PSI Record &nearr;]({s.url})")
                            st.markdown(f"[Open Official NASA PSI Record]({s.url})")

        # Save to chat history
        st.session_state.chat_messages.append(
            {
                "role": "user",
                "content": user_prompt,
            }
        )
        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": response.answer,
                "sources": response.sources,
                "confidence": response.confidence,
                "disclaimer": response.disclaimer,
                "model_name": response.model_name,
                "llm_generated": response.llm_generated,
            }
        )
