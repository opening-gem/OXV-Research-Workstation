"""
Flame in Freefall — Research Intelligence Platform.
Streamlit dashboard entry point with dark scientific aesthetic.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


if STREAMLIT_AVAILABLE:
    logo_path = _ROOT / "assets" / "oxv_logo_cropped.png"

    st.set_page_config(
        page_title="Orbital Experimentation Volumetric (OXV)",
        page_icon=str(logo_path) if logo_path.exists() else None,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # DaVinci Resolve / Blender engineering workstation aesthetic
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #121418;
            color: #E2E8F0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }
        [data-testid="stSidebar"] {
            background-color: #161920;
            border-right: 1px solid #272C37;
        }
        h1, h2, h3, h4 {
            color: #F1F5F9 !important;
            font-weight: 600;
            letter-spacing: -0.01em;
        }
        .stButton>button {
            background-color: #1C2029;
            color: #E2E8F0;
            border: 1px solid #2D3342;
            border-radius: 3px;
            font-weight: 500;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            letter-spacing: 0.04em;
            transition: all 0.12s ease;
        }
        .stButton>button:hover {
            background-color: #242A36;
            border-color: #F08C2E;
            color: #F08C2E;
        }
        .stTextInput>div>div>input {
            background-color: #0E1014;
            border: 1px solid #272C37;
            border-radius: 3px;
            color: #F1F5F9;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
        }
        .stTextInput>div>div>input:focus {
            border-color: #F08C2E;
            box-shadow: none;
        }
        div[data-testid="metric-container"] {
            background-color: #181B22;
            border: 1px solid #272C37;
            border-radius: 3px;
            padding: 10px 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar branding with OXV logo
    if logo_path.exists():
        import base64
        b64_logo = base64.b64encode(logo_path.read_bytes()).decode("utf-8")
        st.sidebar.markdown(
            f"""
            <div style='text-align: center; padding: 0.75rem 0 1rem 0; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 1rem;'>
                <img src='data:image/png;base64,{b64_logo}' style='width: 60px; height: auto; margin-bottom: 8px; filter: drop-shadow(0 2px 6px rgba(234,88,12,0.3));'>
                <div style='font-family: \"JetBrains Mono\", monospace; font-size: 0.88rem; font-weight: 700; color: #F8FAFC; letter-spacing: 0.04em;'>ORBITAL EXPERIMENTATION VOLUMETRIC</div>
                <div style='font-family: \"JetBrains Mono\", monospace; font-size: 0.65rem; color: #94A3B8; letter-spacing: 0.1em; text-transform: uppercase;'>OXV // RESEARCH WORKSTATION</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            """
            <div style='text-align: center; padding: 0.75rem 0 1rem 0; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 1rem;'>
                <div style='font-family: \"JetBrains Mono\", monospace; font-size: 0.88rem; font-weight: 700; color: #F8FAFC;'>ORBITAL EXPERIMENTATION VOLUMETRIC</div>
                <div style='font-family: \"JetBrains Mono\", monospace; font-size: 0.65rem; color: #94A3B8; text-transform: uppercase;'>OXV // RESEARCH WORKSTATION</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Navigation options (clean, zero emojis)
    pages = {
        "01 Research Overview": "dashboard.pages.01_research_overview",
        "02 Investigation Explorer": "dashboard.pages.02_investigation_explorer",
        "03 AI Research Ranking": "dashboard.pages.03_ai_research_ranking",
        "04 Investigation Comparison": "dashboard.pages.04_investigation_comparison",
        "05 Research Assistant": "dashboard.pages.05_research_assistant",
        "06 Data Provenance & Audit": "dashboard.pages.06_data_provenance",
    }

    selected_page = st.sidebar.radio("Navigation Menu", list(pages.keys()))

    import importlib
    mod_name = pages.get(selected_page)
    if mod_name:
        try:
            mod = importlib.import_module(mod_name)
            mod.render()
        except Exception as e:
            st.error(f"Failed to load page {selected_page}: {e}")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style='font-size:0.72rem; color:#64748B; text-align:center; font-family: monospace;'>
            Orbital Experimentation Volumetric (OXV)<br>
            Spaceflight Microgravity Research Intelligence
        </div>
        """,
        unsafe_allow_html=True,
    )

else:
    # CLI fallback if launched directly without streamlit run
    from dashboard.standalone_server import run_standalone_dashboard
    run_standalone_dashboard()
