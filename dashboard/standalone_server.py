"""
Standalone Native Web Dashboard Server for Flame in Freefall.

Provides a rich, interactive, NASA-themed web application using Python's standard library
HTTP server and modern HTML5/CSS3/JavaScript. Enables instant local exploration without
external frontend or framework dependencies, while supporting all 6 platform pages.
"""
from __future__ import annotations

import json
import logging
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List

from src.catalog.repository import ResearchCatalog
from src.ai.embeddings import EmbeddingsManager
from src.ai.ranker import ResearchRanker
from src.rag.chunker import DocumentChunker
from src.rag.retriever import InvestigationRetriever
from src.rag.assistant import ResearchAssistant

logger = logging.getLogger("flame-standalone-server")

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Orbital Experimentation Volumetric (OXV) — Microgravity Flight Research</title>
    <link rel="icon" type="image/png" href="/assets/oxv_logo.png">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-canvas: #121418;
            --bg-topbar: #181B22;
            --bg-panel: #1A1D24;
            --bg-panel-header: #21252F;
            --bg-card: #1C1F27;
            --bg-card-hover: #232732;
            --bg-inset: #111317;
            --border-subtle: #272C37;
            --border-medium: #353C4C;
            --border-focus: #F08C2E;
            --border-active: #38BDF8;
            --text-primary: #F0F3F8;
            --text-secondary: #9AA4B2;
            --text-tertiary: #657082;
            --text-muted: #4B5565;
            --accent-orange: #F08C2E;
            --accent-cyan: #38BDF8;
            --accent-slate: #252A35;
            --status-green: #22C55E;
            --status-amber: #F59E0B;
            --status-red: #EF4444;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: var(--bg-canvas);
            color: var(--text-primary);
            display: flex;
            flex-direction: column;
            height: 100vh;
            overflow: hidden;
            -webkit-font-smoothing: antialiased;
        }

        /* Top Workstation Header Bar (DaVinci Resolve / Blender Style) */
        .studio-header {
            height: 48px;
            background: var(--bg-topbar);
            border-bottom: 1px solid var(--border-subtle);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 1.25rem;
            flex-shrink: 0;
            user-select: none;
        }

        .brand-block {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .brand-logo {
            width: 22px;
            height: 22px;
            object-fit: contain;
            filter: drop-shadow(0 1px 3px rgba(240, 140, 46, 0.4));
        }
        .brand-text {
            display: flex;
            flex-direction: column;
        }
        .brand-title {
            font-size: 0.85rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            color: var(--text-primary);
        }
        .brand-sub {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.58rem;
            letter-spacing: 0.1em;
            color: var(--text-tertiary);
            text-transform: uppercase;
        }

        /* Workspace Switcher Tabs */
        .workspace-tabs {
            display: flex;
            height: 100%;
            align-items: stretch;
            gap: 1px;
            background: var(--border-subtle);
            border-left: 1px solid var(--border-subtle);
            border-right: 1px solid var(--border-subtle);
        }
        .nav-item {
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 0 14px;
            background: var(--bg-topbar);
            color: var(--text-secondary);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            font-weight: 500;
            letter-spacing: 0.04em;
            cursor: pointer;
            transition: all 0.12s ease;
            position: relative;
        }
        .nav-item .nav-key {
            color: var(--text-muted);
            font-size: 0.65rem;
            opacity: 0.7;
        }
        .nav-item:hover {
            background: #1F232D;
            color: var(--text-primary);
        }
        .nav-item.active {
            background: #232732;
            color: #FFFFFF;
            font-weight: 600;
        }
        .nav-item.active::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: var(--accent-orange);
        }
        .nav-item.active .nav-key {
            color: var(--accent-orange);
            opacity: 1;
        }

        .studio-status-strip {
            display: flex;
            align-items: center;
            gap: 12px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            color: var(--text-tertiary);
        }
        .hardware-led {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 3px 8px;
            background: var(--bg-inset);
            border: 1px solid var(--border-subtle);
            border-radius: 2px;
            color: var(--text-secondary);
        }
        .led-pip {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: var(--status-green);
            box-shadow: 0 0 6px rgba(34, 197, 94, 0.5);
        }

        /* Main Workspace Canvas */
        main {
            flex-grow: 1;
            overflow-y: auto;
            padding: 1.5rem 2rem;
            background: var(--bg-canvas);
        }

        .kicker {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.68rem;
            letter-spacing: 0.12em;
            color: var(--accent-orange);
            text-transform: uppercase;
            margin-bottom: 2px;
        }
        .page-header {
            margin-bottom: 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            border-bottom: 1px solid var(--border-subtle);
            padding-bottom: 1rem;
        }
        .page-header h2 {
            font-size: 1.4rem;
            font-weight: 600;
            color: var(--text-primary);
            letter-spacing: -0.01em;
        }
        .page-header p {
            color: var(--text-secondary);
            font-size: 0.84rem;
            margin-top: 2px;
        }

        /* Scientific Metrics Grid */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 0.75rem;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 3px;
            padding: 1rem 1.1rem;
            position: relative;
        }
        .metric-title {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.65rem;
            color: var(--text-tertiary);
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .metric-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.7rem;
            font-weight: 600;
            color: var(--text-primary);
            margin: 0.25rem 0 0.15rem;
        }
        .metric-subtitle {
            font-size: 0.72rem;
            color: var(--text-secondary);
        }

        /* Panels & Cards (Sharp 3px corners, DaVinci styling) */
        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.25rem;
            margin-bottom: 1.5rem;
        }
        .panel {
            background: var(--bg-panel);
            border: 1px solid var(--border-subtle);
            border-radius: 3px;
            padding: 1.25rem;
            margin-bottom: 1.25rem;
        }
        .panel-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.85rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--border-subtle);
        }
        .panel-header h3 {
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--text-primary);
            letter-spacing: 0.02em;
            text-transform: uppercase;
            font-family: 'JetBrains Mono', monospace;
        }

        /* Monospace Data Badges (Sharp 2px corners) */
        .badge {
            font-family: 'JetBrains Mono', monospace;
            display: inline-block;
            padding: 2px 6px;
            border-radius: 2px;
            font-size: 0.7rem;
            font-weight: 500;
            text-transform: uppercase;
            border: 1px solid var(--border-medium);
            background: var(--bg-inset);
            color: var(--text-secondary);
        }
        .badge-cyan {
            border-color: rgba(56, 189, 248, 0.35);
            color: var(--accent-cyan);
            background: rgba(56, 189, 248, 0.08);
        }
        .badge-green {
            border-color: rgba(34, 197, 94, 0.35);
            color: var(--status-green);
            background: rgba(34, 197, 94, 0.08);
        }
        .badge-amber {
            border-color: rgba(245, 158, 11, 0.35);
            color: var(--status-amber);
            background: rgba(245, 158, 11, 0.08);
        }
        .badge-rose {
            border-color: rgba(239, 68, 68, 0.35);
            color: var(--status-red);
            background: rgba(239, 68, 68, 0.08);
        }

        /* Search & Form Elements (Blender Inset Well Style) */
        .search-bar {
            display: flex;
            gap: 8px;
            margin-bottom: 1rem;
        }
        .search-input {
            flex-grow: 1;
            background: var(--bg-inset);
            border: 1px solid var(--border-medium);
            border-radius: 3px;
            padding: 8px 12px;
            color: var(--text-primary);
            font-size: 0.85rem;
            font-family: inherit;
            transition: border-color 0.12s ease;
        }
        .search-input:focus {
            outline: none;
            border-color: var(--border-focus);
        }
        .btn {
            background: #232731;
            color: var(--text-primary);
            border: 1px solid var(--border-medium);
            border-radius: 3px;
            padding: 7px 14px;
            font-size: 0.78rem;
            font-weight: 500;
            font-family: 'JetBrains Mono', monospace;
            cursor: pointer;
            transition: all 0.12s ease;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        .btn:hover {
            background: #2C323F;
            border-color: var(--border-focus);
            color: #FFFFFF;
        }
        .btn.active {
            background: #2A313E;
            border-color: var(--accent-orange);
            color: #FFFFFF;
        }
        .btn-sm {
            padding: 4px 8px;
            font-size: 0.7rem;
        }

        /* Investigation Cards */
        .inv-card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 3px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.75rem;
            transition: border-color 0.12s ease;
        }
        .inv-card:hover {
            border-color: var(--border-medium);
            background: var(--bg-card-hover);
        }
        .inv-card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.4rem;
        }
        .inv-card h4 {
            font-size: 0.95rem;
            font-weight: 600;
            color: var(--text-primary);
        }
        .inv-card p {
            font-size: 0.83rem;
            color: var(--text-secondary);
            line-height: 1.5;
            margin-bottom: 0.65rem;
        }
        .inv-footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            color: var(--text-tertiary);
            padding-top: 0.5rem;
            border-top: 1px solid var(--border-subtle);
        }

        /* Scientific Data Tables (Console Inspector Style) */
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th {
            text-align: left;
            padding: 7px 10px;
            background: var(--bg-panel-header);
            color: var(--text-tertiary);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.68rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            border-bottom: 1px solid var(--border-medium);
        }
        td {
            padding: 8px 10px;
            border-bottom: 1px solid var(--border-subtle);
            font-size: 0.82rem;
            color: var(--text-secondary);
        }
        td strong {
            color: var(--text-primary);
        }

        /* Code / Data Blocks */
        pre {
            background: var(--bg-inset);
            border: 1px solid var(--border-subtle);
            padding: 0.8rem;
            border-radius: 2px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.76rem;
            overflow-x: auto;
            color: #CBD5E1;
            line-height: 1.5;
        }
        .pill-group {
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
            margin: 5px 0;
        }

        /* Status Dot Indicator */
        .status-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            display: inline-block;
        }
        .status-dot.green {
            background: var(--status-green);
            box-shadow: 0 0 6px rgba(34, 197, 94, 0.4);
        }
        .status-dot.amber {
            background: var(--status-amber);
            box-shadow: 0 0 6px rgba(245, 158, 11, 0.4);
        }
        .status-container {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 3px 8px;
            border-radius: 2px;
            background: var(--bg-inset);
            border: 1px solid var(--border-medium);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            color: var(--text-secondary);
        }
    </style>
</head>
<body>
    <header class="studio-header">
        <div class="brand-block">
            <img src="/assets/oxv_logo.png" alt="Orbital Experimentation Volumetric (OXV) Logo" class="brand-logo">
            <div class="brand-text">
                <span class="brand-title">ORBITAL EXPERIMENTATION VOLUMETRIC</span>
                <span class="brand-sub">OXV // RESEARCH WORKSTATION</span>
            </div>
        </div>
        <nav class="workspace-tabs">
            <div class="nav-item active" data-page="overview" onclick="switchPage('overview', this)">
                <span class="nav-key">1</span><span>OVERVIEW</span>
            </div>
            <div class="nav-item" data-page="explorer" onclick="switchPage('explorer', this)">
                <span class="nav-key">2</span><span>EXPLORER</span>
            </div>
            <div class="nav-item" data-page="ranking" onclick="switchPage('ranking', this)">
                <span class="nav-key">3</span><span>RANKING</span>
            </div>
            <div class="nav-item" data-page="comparison" onclick="switchPage('comparison', this)">
                <span class="nav-key">4</span><span>COMPARE</span>
            </div>
            <div class="nav-item" data-page="assistant" onclick="switchPage('assistant', this)">
                <span class="nav-key">5</span><span>ASSISTANT</span>
            </div>
            <div class="nav-item" data-page="provenance" onclick="switchPage('provenance', this)">
                <span class="nav-key">6</span><span>PROVENANCE</span>
            </div>
            <div class="nav-item" data-page="telemetry" onclick="switchPage('telemetry', this)">
                <span class="nav-key">7</span><span>TELEMETRY</span>
            </div>
        </nav>
        <div class="studio-status-strip">
            <span class="hardware-led"><span class="led-pip"></span> SYSTEM NOMINAL</span>
            <span>SPEC 4.0</span>
        </div>
    </header>

    <main id="main-content">
        <!-- Injected via JavaScript -->
    </main>

    <script>
        let currentTab = 'overview';
        let catalogData = [];

        const PSI_DIRECT_URLS = {
            'BASS-II': 'https://psi.nasa.gov/physci/repo/data/investigations/PSI-25',
            'BASS': 'https://psi.nasa.gov/physci/repo/data/investigations/PSI-26',
            'ACME': 'https://psi.nasa.gov/physci/repo/data/investigations/PSI-10',
            'ACME-BRE': 'https://psi.nasa.gov/physci/repo/data/investigations/PSI-10',
            'ACME-CFI': 'https://psi.nasa.gov/physci/repo/data/investigations/PSI-159',
            'ACME-E-FIELD': 'https://psi.nasa.gov/physci/repo/data/investigations/PSI-10',
            'FLEX': 'https://psi.nasa.gov/physci/repo/data/investigations/PSI-68',
            'FLEX-2': 'https://psi.nasa.gov/physci/repo/data/investigations/PSI-69',
            'SAFFIRE-I': 'https://psi.nasa.gov/physci/repo/data/investigations/PSI-98',
            'SAFFIRE-II': 'https://psi.nasa.gov/physci/repo/data/investigations/PSI-99',
            'SAFFIRE-III': 'https://psi.nasa.gov/physci/repo/data/investigations/PSI-100',
            'SAME': 'https://psi.nasa.gov/physci/repo/data/investigations/PSI-102',
            'SPICE': 'https://psi.nasa.gov/physci/repo/data/investigations/PSI-60',
            'CLD': 'https://psi.nasa.gov/physci/repo/data/investigations/PSI-10',
            'SOFIE': 'https://psi.nasa.gov/physci/repo/data/investigations/PSI-25',
            'SoFIE': 'https://psi.nasa.gov/physci/repo/data/investigations/PSI-25',
            'FSDC': 'https://www1.grc.nasa.gov/space/combustion-and-reacting-systems/'
        };

        function getInvestigationSourceUrl(inv) {
            if (!inv) return 'https://psi.nasa.gov/physci/repo/';
            const id = inv.investigation_id || inv.id;
            if (id && PSI_DIRECT_URLS[id]) return PSI_DIRECT_URLS[id];
            if (inv.source_url && inv.source_url.includes('/data/investigations/')) return inv.source_url;
            if (inv.psi_url && inv.psi_url.includes('/data/investigations/')) return inv.psi_url;
            if (inv.url && inv.url.includes('/data/investigations/')) return inv.url;
            return inv.source_url || inv.psi_url || inv.url || 'https://psi.nasa.gov/physci/repo/';
        }

        function esc(s) {
            if (s === null || s === undefined) return '';
            return String(s)
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#039;');
        }

        function formatContent(text) {
            if (!text) return '';
            return esc(text).split(String.fromCharCode(10)).join('<br>');
        }

        async function init() {
            try {
                const res = await fetch('/api/catalog');
                catalogData = await res.json();
                renderPage('overview');
            } catch (e) {
                console.error("Failed to load catalog:", e);
            }
        }

        function switchPage(page, el) {
            currentTab = page;
            document.querySelectorAll('.workspace-tabs .nav-item').forEach(item => {
                if (item.getAttribute('data-page') === page) {
                    item.classList.add('active');
                } else {
                    item.classList.remove('active');
                }
            });
            renderPage(page);
        }

        // Workstation workspace switcher shortcuts: 1-7 (DaVinci/Blender style)
        const PAGE_KEYS = { '1': 'overview', '2': 'explorer', '3': 'ranking', '4': 'comparison', '5': 'assistant', '6': 'provenance', '7': 'telemetry' };
        document.addEventListener('keydown', (e) => {
            const activeEl = document.activeElement;
            const isInput = activeEl && (activeEl.tagName === 'INPUT' || activeEl.tagName === 'TEXTAREA' || activeEl.tagName === 'SELECT');
            if (!isInput && PAGE_KEYS[e.key]) {
                switchPage(PAGE_KEYS[e.key]);
            }
        });

        function renderPage(page) {
            const main = document.getElementById('main-content');
            if (page === 'overview') renderOverview(main);
            else if (page === 'explorer') renderExplorer(main);
            else if (page === 'ranking') renderRanking(main);
            else if (page === 'comparison') renderComparison(main);
            else if (page === 'assistant') renderAssistant(main);
            else if (page === 'provenance') renderProvenance(main);
            else if (page === 'telemetry') renderTelemetry(main);
        }

        function renderOverview(container) {
            const total = catalogData.length;
            const doiCount = catalogData.filter(i => (i.versions && i.versions[0] && i.versions[0].doi)).length;
            const issCount = catalogData.filter(i => i.platform === 'iss').length;

            const cats = {};
            catalogData.forEach(i => (i.categories || []).forEach(c => cats[c] = (cats[c] || 0) + 1));

            container.innerHTML = `
                <div class="page-header">
                    <div>
                        <div class="kicker">NASA PSI // FLIGHT RESEARCH DATA</div>
                        <h2>Flight Research Catalogue</h2>
                        <p>Catalog covering spacecraft fire safety, droplet combustion dynamics, and low-gravity flammability</p>
                    </div>
                </div>

                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-title">Catalogue Investigations</div>
                        <div class="metric-value">${total}</div>
                        <div class="metric-subtitle">Active flight records indexed</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Orbital Flight Records</div>
                        <div class="metric-value">${issCount}</div>
                        <div class="metric-subtitle">International Space Station</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Persistent DOIs Tracked</div>
                        <div class="metric-value">${doiCount}</div>
                        <div class="metric-subtitle">Immutable repository pointers</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Research Categories</div>
                        <div class="metric-value">${Object.keys(cats).length}</div>
                        <div class="metric-subtitle">Domain classifications</div>
                    </div>
                </div>

                <div class="grid-2">
                    <div class="panel">
                        <div class="panel-header">
                            <h3>Research Domain Distribution</h3>
                        </div>
                        <table>
                            <thead><tr><th>Category</th><th>Count</th><th>Share</th></tr></thead>
                            <tbody>
                                ${Object.entries(cats).sort((a,b)=>b[1]-a[1]).map(([cat, cnt]) => `
                                    <tr>
                                        <td><strong>${cat.replace(/_/g, ' ')}</strong></td>
                                        <td><code>${cnt}</code></td>
                                        <td><span class="badge badge-cyan">${((cnt/total)*100).toFixed(0)}%</span></td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                    <div class="panel">
                        <div class="panel-header">
                            <h3>Canonical Research Benchmark</h3>
                        </div>
                        <p style="margin-bottom:1rem; color:var(--text-secondary); font-size:0.86rem; line-height:1.55;">
                            <strong>BASS-II (Burning and Suppression of Solids - II)</strong> establishes our baseline attribution record.
                            Conducted in the Microgravity Science Glovebox (MSG) aboard the ISS, it mapped non-monotonic extinction limits unobtainable in terrestrial 1-g environments.
                        </p>
                        <pre>INVESTIGATION_ID : BASS-II
DOI              : 10.60555/4qc4-de67
ACTIVE_VERSION   : 4
FLIGHT_PLATFORM  : International Space Station (ISS)
FACILITY         : Microgravity Science Glovebox (MSG)
INVESTIGATOR     : Dr. Paul Ferkul (NASA Glenn / USRA)</pre>
                        <div style="margin-top:1.1rem;">
                            <a href="https://psi.nasa.gov/physci/repo/data/investigations/PSI-25" target="_blank" class="btn btn-sm">OPEN REPOSITORY RECORD &nearr;</a>
                        </div>
                    </div>
                </div>
            `;
        }

        function renderExplorer(container) {
            container.innerHTML = `
                <div class="page-header">
                    <div>
                        <div class="kicker">METADATA EXPLORER</div>
                        <h2>Investigation Explorer</h2>
                        <p>Query, filter, and inspect canonical microgravity combustion experiments</p>
                    </div>
                </div>
                <div class="search-bar">
                    <input type="text" id="filter-input" class="search-input" placeholder="Search by title, fuel, keyword (e.g. PMMA, soot, lunar, suppression)..." oninput="filterExplorer()">
                </div>
                <div id="explorer-list"></div>
            `;
            filterExplorer();
        }

        function filterExplorer() {
            const query = (document.getElementById('filter-input')?.value || '').toLowerCase();
            const filtered = catalogData.filter(i => {
                const text = `${i.id} ${i.title} ${i.description} ${(i.categories||[]).join(' ')} ${(i.keywords||[]).join(' ')}`.toLowerCase();
                return text.includes(query);
            });
            const listEl = document.getElementById('explorer-list');
            if (!listEl) return;
            listEl.innerHTML = filtered.map(inv => `
                <div class="inv-card">
                    <div class="inv-card-header">
                        <h4>[${esc(inv.id)}] ${esc(inv.title)}</h4>
                        <div>
                            <span class="badge badge-cyan">${esc((inv.platform || 'ISS').toUpperCase())}</span>
                            <span class="badge badge-green">v${esc(inv.active_version || 1)}</span>
                        </div>
                    </div>
                    <p>${esc(inv.description)}</p>
                    <div class="pill-group">
                        ${(inv.categories || []).map(c => `<span class="badge">${esc(c.replace(/_/g, ' '))}</span>`).join('')}
                    </div>
                    <div class="inv-footer" style="margin-top: 10px;">
                        <span>DOI: <code>${esc((inv.versions && inv.versions[0] && inv.versions[0].doi) || '10.60555/UNASSIGNED')}</code></span>
                        <a href="${esc(getInvestigationSourceUrl(inv))}" target="_blank" style="color:var(--accent-cyan); text-decoration:none; font-weight:500;">REPOSITORY SOURCE &nearr;</a>
                    </div>
                </div>
            `).join('');
        }

        async function renderRanking(container) {
            container.innerHTML = `
                <div class="page-header">
                    <div>
                        <div class="kicker">STATISTICAL IR & HYBRID RANKING</div>
                        <h2>AI Relevance Ranking Engine</h2>
                        <p>Multi-factor hybrid ranking scoring BM25 term frequency, semantic cosine similarity, and provenance</p>
                    </div>
                </div>
                <div class="panel">
                    <div class="search-bar">
                        <input type="text" id="rank-input" class="search-input" value="fire detection and smoke sensor systems for lunar habitats" placeholder="Enter research query...">
                        <button class="btn" onclick="executeRanking()">EXECUTE RANKING</button>
                    </div>
                    <div style="display:flex; gap:8px; flex-wrap:wrap;">
                        <button class="btn btn-sm" onclick="setRankQuery('fire detection in lunar habitats')">LUNAR FIRE DETECTION</button>
                        <button class="btn btn-sm" onclick="setRankQuery('flame spread over solid spacecraft materials')">SOLID FLAME SPREAD</button>
                        <button class="btn btn-sm" onclick="setRankQuery('droplet burning and cool flames')">DROPLETS & COOL FLAMES</button>
                        <button class="btn btn-sm" onclick="setRankQuery('spacecraft fire suppression agents')">FIRE SUPPRESSION</button>
                    </div>
                </div>
                <div id="ranking-results"></div>
            `;
            executeRanking();
        }

        function setRankQuery(q) {
            document.getElementById('rank-input').value = q;
            executeRanking();
        }

        async function executeRanking() {
            const query = document.getElementById('rank-input')?.value || 'fire detection';
            const resEl = document.getElementById('ranking-results');
            if (!resEl) return;
            resEl.innerHTML = `<p style="color:var(--text-tertiary); font-family:'JetBrains Mono',monospace; font-size:0.8rem;">Computing hybrid BM25 + semantic similarity ranking...</p>`;
            try {
                const res = await fetch('/api/rank?q=' + encodeURIComponent(query));
                const ranked = await res.json();
                resEl.innerHTML = ranked.map((r, idx) => `
                    <div class="inv-card" style="border-left: 3px solid var(--accent-cyan);">
                        <div class="inv-card-header">
                            <h4>#${idx+1} [${esc(r.investigation_id)}] ${esc(r.title)}</h4>
                            <span class="badge badge-cyan" style="font-size:0.75rem;">SCORE: ${(r.overall_score * 100).toFixed(1)}%</span>
                        </div>
                        <p style="font-size:0.86rem; color:var(--text-secondary); line-height:1.5;">${esc(r.explanation)}</p>
                        <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:8px; background:var(--bg-inset); border:1px solid var(--border-subtle); padding:8px 12px; border-radius:2px; font-family:'JetBrains Mono',monospace; font-size:0.72rem; margin:8px 0; color:var(--text-tertiary);">
                            <div>KEYWORD: <strong style="color:var(--text-primary);">${(r.component_scores.keyword * 100).toFixed(0)}%</strong></div>
                            <div>SEMANTIC: <strong style="color:var(--text-primary);">${(r.component_scores.semantic * 100).toFixed(0)}%</strong></div>
                            <div>METADATA: <strong style="color:var(--text-primary);">${(r.component_scores.metadata * 100).toFixed(0)}%</strong></div>
                            <div>EVIDENCE: <strong style="color:var(--text-primary);">${(r.component_scores.evidence * 100).toFixed(0)}%</strong></div>
                        </div>
                        ${r.relevant_objectives && r.relevant_objectives.length ? `
                            <p style="font-size:0.78rem; color:var(--text-tertiary); margin-top:6px; font-family:'JetBrains Mono',monospace;">
                                MATCHING OBJECTIVE: ${esc(r.relevant_objectives[0])}
                            </p>
                        ` : ''}
                    </div>
                `).join('');
            } catch(e) {
                resEl.innerHTML = `<p style="color:#EF4444; font-family:'JetBrains Mono',monospace; font-size:0.8rem;">Ranking failed: ${esc(e)}</p>`;
            }
        }

        function renderComparison(container) {
            container.innerHTML = `
                <div class="page-header">
                    <div>
                        <div class="kicker">CROSS-EXPERIMENT ANALYSIS</div>
                        <h2>Investigation Comparison</h2>
                        <p>Side-by-side comparison of objectives, experimental hardware, and operational regimes</p>
                    </div>
                </div>
                <div class="panel" style="display:grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                    <div>
                        <label style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:var(--text-tertiary); display:block; margin-bottom:6px; text-transform:uppercase;">Investigation A:</label>
                        <select id="comp-a" class="search-input" onchange="runCompare()" style="width:100%;">
                            ${catalogData.map(i => `<option value="${i.id}" ${i.id==='BASS-II'?'selected':''}>${i.id}: ${i.title}</option>`).join('')}
                        </select>
                    </div>
                    <div>
                        <label style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:var(--text-tertiary); display:block; margin-bottom:6px; text-transform:uppercase;">Investigation B:</label>
                        <select id="comp-b" class="search-input" onchange="runCompare()" style="width:100%;">
                            ${catalogData.map(i => `<option value="${i.id}" ${i.id==='ACME'?'selected':''}>${i.id}: ${i.title}</option>`).join('')}
                        </select>
                    </div>
                </div>
                <div id="comparison-view"></div>
            `;
            runCompare();
        }

        function runCompare() {
            const idA = document.getElementById('comp-a')?.value || 'BASS-II';
            const idB = document.getElementById('comp-b')?.value || 'ACME';
            const invA = catalogData.find(i => i.id === idA) || catalogData[0];
            const invB = catalogData.find(i => i.id === idB) || catalogData[1];
            const view = document.getElementById('comparison-view');
            if (!view || !invA || !invB) return;

            view.innerHTML = `
                <div class="grid-2">
                    <div class="panel">
                        <div class="panel-header">
                            <h3>${esc(invA.title)} [${esc(invA.id)}]</h3>
                        </div>
                        <p style="margin-bottom:1rem; color:var(--text-secondary); font-size:0.86rem; line-height:1.5;">${esc(invA.description)}</p>
                        <p style="font-size:0.83rem; margin-bottom:4px;"><strong>PLATFORM:</strong> <span class="badge">${esc((invA.platform || 'ISS').toUpperCase())}</span></p>
                        <p style="font-size:0.83rem; margin-bottom:4px;"><strong>FACILITY:</strong> <span style="color:var(--text-secondary); font-family:'JetBrains Mono',monospace;">${esc(invA.facility || 'N/A')}</span></p>
                        <p style="font-size:0.83rem; margin-bottom:4px;"><strong>VERSION:</strong> <span class="badge badge-cyan">v${esc(invA.active_version || 1)}</span></p>
                        <div style="margin-top:1.2rem; border-top:1px solid var(--border-subtle); padding-top:0.8rem;">
                            <span style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:var(--text-tertiary); text-transform:uppercase;">Documented Objectives:</span>
                            <ul style="padding-left:16px; margin-top:6px; color:var(--text-secondary); font-size:0.82rem; line-height:1.5;">
                                ${(invA.objectives || []).map(o => `<li>${esc(o)}</li>`).join('')}
                            </ul>
                        </div>
                    </div>
                    <div class="panel">
                        <div class="panel-header">
                            <h3>${esc(invB.title)} [${esc(invB.id)}]</h3>
                        </div>
                        <p style="margin-bottom:1rem; color:var(--text-secondary); font-size:0.86rem; line-height:1.5;">${esc(invB.description)}</p>
                        <p style="font-size:0.83rem; margin-bottom:4px;"><strong>PLATFORM:</strong> <span class="badge">${esc((invB.platform || 'ISS').toUpperCase())}</span></p>
                        <p style="font-size:0.83rem; margin-bottom:4px;"><strong>FACILITY:</strong> <span style="color:var(--text-secondary); font-family:'JetBrains Mono',monospace;">${esc(invB.facility || 'N/A')}</span></p>
                        <p style="font-size:0.83rem; margin-bottom:4px;"><strong>VERSION:</strong> <span class="badge badge-cyan">v${esc(invB.active_version || 1)}</span></p>
                        <div style="margin-top:1.2rem; border-top:1px solid var(--border-subtle); padding-top:0.8rem;">
                            <span style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:var(--text-tertiary); text-transform:uppercase;">Documented Objectives:</span>
                            <ul style="padding-left:16px; margin-top:6px; color:var(--text-secondary); font-size:0.82rem; line-height:1.5;">
                                ${(invB.objectives || []).map(o => `<li>${esc(o)}</li>`).join('')}
                            </ul>
                        </div>
                    </div>
                </div>
            `;
        }

        let assistantChatHistory = [
            {
                role: 'assistant',
                content: 'Welcome to the Orbital Experimentation Volumetric (OXV) Research Assistant. Queries are grounded in microgravity combustion records with citation tracking. Ask about spaceflight investigations (e.g. BASS-II, ACME, FLEX, SAFFIRE), flame spread, or habitat fire safety.',
                sources: [],
                confidence: 'high'
            }
        ];

        async function renderAssistant(container) {
            container.innerHTML = `
                <div class="page-header">
                    <div>
                        <div class="kicker">EVIDENCE-GROUNDED REASONING</div>
                        <h2>Conversational Research Assistant</h2>
                        <p>Multi-turn Retrieval-Augmented Generation powered by Ollama with verified citations</p>
                    </div>
                    <div id="llm-status-pill">
                        <div class="status-container">
                            <span class="status-dot amber"></span>
                            <span>PROBING RUNTIME...</span>
                        </div>
                    </div>
                </div>

                <div class="panel">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                        <span style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:var(--text-tertiary); text-transform:uppercase; letter-spacing:0.08em;">GUIDED RESEARCH INQUIRIES</span>
                        <button class="btn btn-sm" onclick="clearAssistantChat()">CLEAR CHAT</button>
                    </div>
                    <div style="display:flex; gap:8px; flex-wrap:wrap;">
                        <button class="btn btn-sm" onclick="setAssistantQ('What is the difference between ACME and BASS-II?')">COMPARE ACME VS BASS-II</button>
                        <button class="btn btn-sm" onclick="setAssistantQ('Which investigations study smoke detection for lunar habitats?')">LUNAR SMOKE DETECTION</button>
                        <button class="btn btn-sm" onclick="setAssistantQ('What experiments investigate cool flames in microgravity?')">COOL FLAME KINETICS</button>
                        <button class="btn btn-sm" onclick="setAssistantQ('How do flames spread over solid materials on the space station?')">SOLID FLAME SPREAD (BASS-II)</button>
                    </div>
                </div>

                <div id="chat-stream-container" style="display:flex; flex-direction:column; gap:14px; margin-bottom:1.5rem;"></div>
                <div id="generating-indicator" style="display:none; padding:10px 14px; background:var(--bg-inset); border:1px solid var(--border-medium); border-radius:2px; font-family:'JetBrains Mono',monospace; color:var(--accent-cyan); font-size:0.78rem; margin-bottom:1rem;">
                    RETRIEVING ORBITAL EVIDENCE AND COMPUTING RESPONSE...
                </div>

                <div class="search-bar" style="position:sticky; bottom:0; background:var(--bg-canvas); padding-top:10px;">
                    <input type="text" id="assistant-input" class="search-input" placeholder="Enter research question or follow-up inquiry..." onkeydown="if(event.key==='Enter') askAssistant()">
                    <button class="btn" onclick="askAssistant()">SEND QUERY</button>
                </div>
            `;
            checkLLMStatus();
            renderChatHistory();
        }

        async function checkLLMStatus() {
            const pill = document.getElementById('llm-status-pill');
            if (!pill) return;
            try {
                const res = await fetch('/api/assistant/status');
                const data = await res.json();
                const modelName = esc((data.configured_model || 'llama3').toUpperCase());
                if (data.status === 'online' && data.model_available) {
                    pill.innerHTML = `
                        <div class="status-container">
                            <span class="status-dot green"></span>
                            <span>OLLAMA RUNTIME ACTIVE (${modelName})</span>
                        </div>
                    `;
                } else if (data.status === 'online') {
                    pill.innerHTML = `
                        <div class="status-container">
                            <span class="status-dot amber"></span>
                            <span>OLLAMA CONNECTED (${modelName} NOT LOADED — RUN: ollama pull ${data.configured_model || 'llama3'})</span>
                        </div>
                    `;
                } else {
                    pill.innerHTML = `
                        <div class="status-container">
                            <span class="status-dot amber"></span>
                            <span>OLLAMA OFFLINE (DETERMINISTIC FALLBACK ACTIVE)</span>
                        </div>
                    `;
                }
            } catch(e) {
                pill.innerHTML = `
                    <div class="status-container">
                        <span class="status-dot amber"></span>
                        <span>OLLAMA OFFLINE (DETERMINISTIC FALLBACK ACTIVE)</span>
                    </div>
                `;
            }
        }

        function clearAssistantChat() {
            assistantChatHistory = [
                {
                    role: 'assistant',
                    content: 'Chat session reset. How can I assist your microgravity combustion research today?',
                    sources: [],
                    confidence: 'high'
                }
            ];
            renderChatHistory();
        }

        function renderChatHistory() {
            const container = document.getElementById('chat-stream-container');
            if (!container) return;
            container.innerHTML = assistantChatHistory.map((msg, idx) => {
                if (msg.role === 'user') {
                    return `
                        <div style="align-self:flex-end; max-width:80%; background:var(--bg-card); border:1px solid var(--border-medium); color:#F1F5F9; padding:12px 18px; border-radius:3px; line-height:1.55; font-size:0.9rem;">
                            <div style="font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:var(--text-tertiary); margin-bottom:4px; text-transform:uppercase;">RESEARCH QUERY</div>
                            ${formatContent(msg.content)}
                        </div>
                    `;
                } else {
                    const sources = msg.sources || [];
                    const confColor = msg.confidence === 'high' ? 'badge-green' : msg.confidence === 'medium' ? 'badge-cyan' : 'badge-amber';
                    const modelTag = (msg.llm_generated && msg.model_name) ? `<span class="badge badge-cyan">MODEL: ${esc(msg.model_name.toUpperCase())}</span>` : `<span class="badge badge-amber">SYNTHESIS: DETERMINISTIC</span>`;
                    return `
                        <div style="align-self:flex-start; max-width:88%; background:var(--bg-card); border:1px solid var(--border-subtle); border-radius:3px; padding:16px 20px; line-height:1.6; font-size:0.9rem;">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; border-bottom:1px solid var(--border-subtle); padding-bottom:8px;">
                                <span style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; font-weight:600; color:var(--text-primary); letter-spacing:0.04em;">FLIGHT RESEARCH ASSISTANT</span>
                                <div style="display:flex; gap:6px;">
                                    ${modelTag}
                                    <span class="badge ${confColor}">${(msg.confidence || 'verified').toUpperCase()}</span>
                                </div>
                            </div>
                            <div style="color:var(--text-primary); font-size:0.9rem; line-height:1.6;">
                                ${formatContent(msg.content)}
                            </div>
                            ${msg.disclaimer ? `<div style="background:#1A1712; border:1px solid rgba(240,140,46,0.3); border-left:3px solid var(--accent-orange); padding:8px 12px; margin-top:12px; font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#F59E0B;">${msg.disclaimer}</div>` : ''}
                            ${sources.length ? `
                                <div style="margin-top:14px; border-top:1px solid var(--border-subtle); padding-top:10px;">
                                    <details>
                                        <summary style="cursor:pointer; color:var(--accent-cyan); font-family:'JetBrains Mono',monospace; font-size:0.75rem; font-weight:600; text-transform:uppercase;">VERIFIED RESEARCH CITATIONS (${sources.length})</summary>
                                        <div style="margin-top:8px; display:flex; flex-direction:column; gap:8px;">
                                            ${sources.map(s => `
                                                <div style="background:var(--bg-inset); border:1px solid var(--border-subtle); padding:10px; border-radius:2px; font-size:0.8rem;">
                                                    <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                                                        <strong style="color:var(--text-primary);">[${esc(s.investigation_id)}] ${esc(s.title)}</strong>
                                                        <span style="font-family:'JetBrains Mono',monospace; color:var(--text-tertiary);">${s.doi ? 'DOI: ' + esc(s.doi) : 'v' + esc(s.active_version || 1)}</span>
                                                    </div>
                                                    <p style="color:var(--text-secondary); margin:4px 0; font-size:0.78rem; line-height:1.45;">${esc(s.relevant_text)}</p>
                                                    ${s.url || s.investigation_id ? `<a href="${esc(getInvestigationSourceUrl(s))}" target="_blank" style="color:var(--accent-cyan); font-family:'JetBrains Mono',monospace; font-size:0.72rem; text-decoration:none; font-weight:500;">OPEN SOURCE RECORD &nearr;</a>` : ''}
                                                </div>
                                            `).join('')}
                                        </div>
                                    </details>
                                </div>
                            ` : ''}
                        </div>
                    `;
                }
            }).join('');
            container.scrollTop = container.scrollHeight;
        }

        function setAssistantQ(q) {
            const input = document.getElementById('assistant-input');
            if (input) {
                input.value = q;
                askAssistant();
            }
        }

        async function askAssistant() {
            const input = document.getElementById('assistant-input');
            const q = (input?.value || '').trim();
            if (!q) return;

            input.value = '';
            assistantChatHistory.push({ role: 'user', content: q });
            renderChatHistory();

            const ind = document.getElementById('generating-indicator');
            if (ind) ind.style.display = 'block';

            try {
                const historyPayload = assistantChatHistory.slice(0, -1).map(m => ({ role: m.role, content: m.content }));
                const res = await fetch('/api/assistant/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: q, history: historyPayload, use_llm: true })
                });
                const data = await res.json();
                assistantChatHistory.push({
                    role: 'assistant',
                    content: data.answer,
                    sources: data.sources || [],
                    confidence: data.confidence,
                    disclaimer: data.disclaimer,
                    model_name: data.model_name,
                    llm_generated: data.llm_generated
                });
            } catch(e) {
                assistantChatHistory.push({
                    role: 'assistant',
                    content: `Communication error: ${e}. Please verify your network and Ollama server status.`,
                    sources: [],
                    confidence: 'low',
                    disclaimer: 'Network failure'
                });
            } finally {
                if (ind) ind.style.display = 'none';
                renderChatHistory();
            }
        }

        function renderProvenance(container) {
            container.innerHTML = `
                <div class="page-header">
                    <div>
                        <div class="kicker">PROVENANCE & AUDIT TRAIL</div>
                        <h2>Data Provenance & Persistent Identifiers</h2>
                        <p>Audit trail of dataset versions, DOIs, and physical science attribution records</p>
                    </div>
                </div>
                <div class="panel">
                    <div class="panel-header">
                        <h3>Persistent Identifier (DOI) Registry</h3>
                    </div>
                    <table>
                        <thead>
                            <tr><th>ID</th><th>Investigation</th><th>Active Version</th><th>DOI Pointer</th><th>Status</th></tr>
                        </thead>
                        <tbody>
                            ${catalogData.map(i => {
                                const doi = (i.versions && i.versions[0] && i.versions[0].doi) || null;
                                return `
                                    <tr>
                                        <td><code><strong>${esc(i.id)}</strong></code></td>
                                        <td>${esc(i.title)}</td>
                                        <td><span class="badge badge-cyan">v${esc(i.active_version || 1)}</span></td>
                                        <td>${doi ? `<a href="https://doi.org/${encodeURIComponent(doi)}" target="_blank" style="color:var(--accent-cyan); font-family:monospace; text-decoration:none;">${esc(doi)}</a>` : '<span style="color:var(--text-muted); font-family:monospace;">UNASSIGNED</span>'}</td>
                                        <td><span class="badge badge-green">${esc((i.status || 'Completed').toUpperCase())}</span></td>
                                    </tr>
                                `;
                            }).join('')}
                        </tbody>
                    </table>
                </div>

                <div class="grid-2">
                    <div class="panel">
                        <div class="panel-header">
                            <h3>Historical Case Study: Apollo 1 (1967)</h3>
                        </div>
                        <p style="font-size:0.85rem; color:var(--text-secondary); line-height:1.5; margin-bottom:8px;">
                            High-pressure 100% pure oxygen cabin atmosphere accelerated combustion spread rates across nylon and velcro materials, establishing NASA's spacecraft material flammability standards.
                        </p>
                        <pre>INCIDENT         : Apollo 204 (Apollo 1)
ATMOSPHERE       : 100% O2 at 16.7 psia
CAUSE            : Electrical arcing near wire harness
DIRECTIVE        : NASA-STD-6001 Flammability Testing</pre>
                    </div>
                    <div class="panel">
                        <div class="panel-header">
                            <h3>Historical Case Study: Mir Space Station (1997)</h3>
                        </div>
                        <p style="font-size:0.85rem; color:var(--text-secondary); line-height:1.5; margin-bottom:8px;">
                            Solid-fuel oxygen generator (Vika / TGK) candle combustion in Kvant-1 module created dense aerosol soot, testing containment in microgravity enclosed environments.
                        </p>
                        <pre>INCIDENT         : Mir Kvant-1 Oxygen Candle Fire
DURATION         : ~14 minutes
ATMOSPHERE IMPACT: Heavy smoke aerosol generation
LESSON           : SAME (Smoke Aerosol Measurement Experiment)</pre>
                    </div>
                </div>
            `;
        }


        let currentTelemetryDataset = 'BASS-II';
        let currentTelemetryPlot = '01_scatter_outcomes';
        let currentTelemetryBins = 8;
        let currentTelemetryMinSamples = 2;
        let currentTelemetryMethod = 'uniform';
        let telemetryBundle = null;

        function updateBinsLabel(val) {
            const el = document.getElementById('bins-val');
            if (el) el.innerText = val;
        }

        function updateSamplesLabel(val) {
            const el = document.getElementById('samples-val');
            if (el) el.innerText = val;
        }

        function setTelemetryDataset(ds) {
            currentTelemetryDataset = ds;
            currentTelemetryPlot = '01_scatter_outcomes';
            const container = document.getElementById('main-content');
            if (container) renderTelemetry(container);
        }

        function selectTelemetryPlot(plotKey) {
            currentTelemetryPlot = plotKey;
            document.querySelectorAll('#plot-selector-group .btn').forEach(b => b.classList.remove('active'));
            if (window.event && window.event.target) {
                window.event.target.classList.add('active');
            }
            loadTelemetryBundle();
        }

        function applyTelemetryParams() {
            currentTelemetryBins = parseInt(document.getElementById('param-bins')?.value || '8', 10);
            currentTelemetryMinSamples = parseInt(document.getElementById('param-samples')?.value || '2', 10);
            currentTelemetryMethod = document.getElementById('param-method')?.value || 'uniform';
            loadTelemetryBundle();
        }

        async function loadTelemetryBundle() {
            const indicator = document.getElementById('telemetry-loading-indicator');
            if (indicator) indicator.innerText = 'COMPUTING...';

            try {
                const url = '/api/telemetry/data?dataset=' + encodeURIComponent(currentTelemetryDataset) +
                            '&plot=' + encodeURIComponent(currentTelemetryPlot) +
                            '&bins=' + currentTelemetryBins +
                            '&min_samples=' + currentTelemetryMinSamples +
                            '&method=' + encodeURIComponent(currentTelemetryMethod);
                const res = await fetch(url);
                telemetryBundle = await res.json();
                renderTelemetryDisplay();
            } catch (e) {
                const vp = document.getElementById('telemetry-viewport');
                if (vp) vp.innerHTML = '<p style="color:#EF4444; font-family:monospace;">Telemetry error: ' + e + '</p>';
            } finally {
                if (indicator) indicator.innerText = 'READY';
            }
        }

        function renderTelemetryDisplay() {
            const vp = document.getElementById('telemetry-viewport');
            if (!vp || !telemetryBundle) return;

            const data = telemetryBundle;
            const plots = data.plots || {};
            const riskSummary = data.risk_summary || {};

            const facilityTag = document.getElementById('telemetry-facility-tag');
            if (facilityTag) facilityTag.innerText = 'FACILITY: ' + (data.facility || 'N/A').toUpperCase();

            let plotHtml = '';
            const plotKeys = currentTelemetryPlot === 'all' ? Object.keys(plots) : [currentTelemetryPlot];

            plotKeys.forEach(k => {
                const p = plots[k];
                if (p) {
                    plotHtml += `
                        <div class="panel" style="margin-bottom:1.5rem;">
                            <div class="panel-header">
                                <h3>${p.title}</h3>
                                <span class="badge badge-cyan">${k}</span>
                            </div>
                            <div style="text-align:center; background:var(--bg-inset); border:1px solid var(--border-subtle); border-radius:2px; padding:12px; margin-bottom:12px;">
                                <img src="${p.image}" alt="${p.title}" style="max-width:100%; height:auto; border-radius:2px;">
                            </div>
                            ${p.interpretation ? `
                                <div style="background:var(--bg-inset); border:1px solid var(--border-subtle); border-left:3px solid var(--accent-cyan); padding:10px 14px; border-radius:2px; font-size:0.83rem; line-height:1.5; color:var(--text-secondary);">
                                    <strong style="color:var(--text-primary); font-family:monospace; text-transform:uppercase; font-size:0.75rem; display:block; margin-bottom:3px;">PHYSICAL SIGNIFICANCE & MISSION RELEVANCE:</strong>
                                    ${p.interpretation}
                                </div>
                            ` : ''}
                        </div>
                    `;
                }
            });

            const tier = riskSummary.risk_tier || 'MONITORING';
            const riskColor = tier.includes('CRITICAL') ? '#EF4444' : tier.includes('ELEVATED') ? '#F59E0B' : tier.includes('MODERATE') ? '#38BDF8' : '#10B981';
            const riskBadge = tier.includes('CRITICAL') ? 'badge-rose' : tier.includes('ELEVATED') ? 'badge-amber' : tier.includes('MODERATE') ? 'badge-cyan' : 'badge-green';

            const riskMonitorHtml = `
                <div class="panel" style="background:var(--bg-card); border:1px solid var(--border-medium); border-left:4px solid ${riskColor}; margin-bottom:1.5rem; padding:14px 18px; border-radius:3px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px; margin-bottom:12px;">
                        <span style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; font-weight:700; color:var(--text-primary); text-transform:uppercase; letter-spacing:0.06em;">PARAMETRIC BURN RISK MONITOR (FLIGHT HAZARD RATING)</span>
                        <span class="badge ${riskBadge}" style="font-weight:700; letter-spacing:0.05em;">${tier}</span>
                    </div>
                    <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap:12px;">
                        <div style="background:var(--bg-inset); border:1px solid var(--border-subtle); border-radius:2px; padding:10px 14px;">
                            <div style="font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:var(--text-tertiary); text-transform:uppercase;">MEAN SUSTAIN RISK (R_burn)</div>
                            <div style="font-size:1.4rem; font-weight:700; color:${riskColor}; font-family:'JetBrains Mono',monospace;">${riskSummary.overall_burn_risk_pct || '0.0%'}</div>
                            <div style="font-size:0.72rem; color:var(--text-muted);">Sustained flame propagation probability</div>
                        </div>
                        <div style="background:var(--bg-inset); border:1px solid var(--border-subtle); border-radius:2px; padding:10px 14px;">
                            <div style="font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:var(--text-tertiary); text-transform:uppercase;">OXYGEN RISK SENSITIVITY (dR/dO2)</div>
                            <div style="font-size:1.1rem; font-weight:600; color:var(--accent-cyan); font-family:'JetBrains Mono',monospace; margin-top:4px;">${riskSummary.oxygen_risk_sensitivity || 'N/A'}</div>
                            <div style="font-size:0.72rem; color:var(--text-muted);">Risk escalation per +1% O2</div>
                        </div>
                        <div style="background:var(--bg-inset); border:1px solid var(--border-subtle); border-radius:2px; padding:10px 14px;">
                            <div style="font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:var(--text-tertiary); text-transform:uppercase;">PEAK HAZARD FLOW REGIME</div>
                            <div style="font-size:1.1rem; font-weight:600; color:var(--text-primary); font-family:'JetBrains Mono',monospace; margin-top:4px;">${riskSummary.peak_hazard_flow_band || 'N/A'} cm/s</div>
                            <div style="font-size:0.72rem; color:var(--text-muted);">Velocity band of maximum burning</div>
                        </div>
                        <div style="background:var(--bg-inset); border:1px solid var(--border-subtle); border-radius:2px; padding:10px 14px;">
                            <div style="font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:var(--text-tertiary); text-transform:uppercase;">EXTINCTION SAFETY THRESHOLD</div>
                            <div style="font-size:1.1rem; font-weight:600; color:var(--status-amber); font-family:'JetBrains Mono',monospace; margin-top:4px;">R = 0.50 Boundary</div>
                            <div style="font-size:0.72rem; color:var(--text-muted);">Quenched below / Propagation above</div>
                        </div>
                    </div>
                </div>
            `;

            const breakdownHtml = Object.entries(data.outcomes_breakdown || {}).map(([cat, cnt]) => `
                <div style="background:var(--bg-inset); border:1px solid var(--border-subtle); border-radius:2px; padding:8px 12px; font-family:monospace; font-size:0.75rem;">
                    <span style="color:var(--text-tertiary);">${esc(cat)}:</span> <strong style="color:var(--text-primary);">${esc(cnt)}</strong>
                </div>
            `).join('');

            vp.innerHTML = `
                ${riskMonitorHtml}

                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-title">TOTAL FLIGHT RUNS</div>
                        <div class="metric-value">${data.total_rows}</div>
                        <div class="metric-subtitle">Valid microgravity test records</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">SUSTAINED COMBUSTION</div>
                        <div class="metric-value">${data.sustained_count}</div>
                        <div class="metric-subtitle">${((data.sustained_count/data.total_rows)*100).toFixed(1)}% of operational trials</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">EXTINCTION / BLOWOFF</div>
                        <div class="metric-value">${data.extinguished_count}</div>
                        <div class="metric-subtitle">Quenched flammability limits</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">ANALYSIS RESOLUTION</div>
                        <div class="metric-value">${data.n_bins}x${data.n_bins}</div>
                        <div class="metric-subtitle">Grid matrix discretization</div>
                    </div>
                </div>

                <div style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:1.5rem;">
                    ${breakdownHtml}
                </div>

                ${plotHtml}

                <div class="panel">
                    <div class="panel-header">
                        <h3>Numerical Telemetry Breakdown</h3>
                    </div>
                    <div style="overflow-x:auto;">
                        <table>
                            <thead><tr><th>Outcome Category</th><th>Runs (n)</th><th>O2 Mean</th><th>O2 Min - Max</th><th>Airflow Mean</th><th>Airflow Min - Max</th></tr></thead>
                            <tbody>
                                ${(data.tables?.summary || []).map(r => `
                                    <tr>
                                        <td><strong>${esc(r.outcome_raw || 'Unknown')}</strong></td>
                                        <td><code>${r.oxygen_count || '-'}</code></td>
                                        <td><code>${(r.oxygen_mean || 0).toFixed(2)}</code></td>
                                        <td><code>${(r.oxygen_min || 0).toFixed(1)} - ${(r.oxygen_max || 0).toFixed(1)}</code></td>
                                        <td><code>${(r.airflow_mean || 0).toFixed(2)}</code></td>
                                        <td><code>${(r.airflow_min || 0).toFixed(1)} - ${(r.airflow_max || 0).toFixed(1)}</code></td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                </div>
            `;
        }

        async function renderTelemetry(container) {
            container.innerHTML = `
                <div class="page-header">
                    <div>
                        <div class="kicker">FLIGHT TELEMETRY & EXTINCTION ANALYSIS</div>
                        <h2>Experimental Telemetry & Graph Studio</h2>
                        <p>High-precision parametric analysis of BASS-II and FLEX orbital combustion test matrices</p>
                    </div>
                </div>

                <div class="panel" style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px; margin-bottom:1.25rem;">
                    <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
                        <span style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:var(--text-tertiary); text-transform:uppercase;">FLIGHT EXPERIMENT:</span>
                        <button id="ds-btn-bass" class="btn btn-sm ${currentTelemetryDataset==='BASS-II'?'active':''}" onclick="setTelemetryDataset('BASS-II')" style="${currentTelemetryDataset==='BASS-II'?'border-color:var(--accent-cyan); color:#FFF; background:#18202F;':''}">BASS-II (SOLID FLAME SPREAD)</button>
                        <button id="ds-btn-flex" class="btn btn-sm ${currentTelemetryDataset==='FLEX'?'active':''}" onclick="setTelemetryDataset('FLEX')" style="${currentTelemetryDataset==='FLEX'?'border-color:var(--accent-cyan); color:#FFF; background:#18202F;':''}">FLEX (DROPLET COMBUSTION)</button>
                    </div>
                    <div style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:var(--text-tertiary);" id="telemetry-facility-tag">
                        FACILITY: MSG (MICROGRAVITY SCIENCE GLOVEBOX)
                    </div>
                </div>

                <div class="panel" style="margin-bottom:1.25rem;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; border-bottom:1px solid var(--border-subtle); padding-bottom:8px;">
                        <span style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:var(--text-tertiary); text-transform:uppercase; letter-spacing:0.08em;">TELEMETRY GRAPH PROJECTION</span>
                        <span style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:var(--accent-cyan);" id="telemetry-loading-indicator">READY</span>
                    </div>

                    <div style="display:flex; gap:6px; flex-wrap:wrap; margin-bottom:1rem;" id="plot-selector-group">
                        <button class="btn btn-sm ${currentTelemetryPlot==='01_scatter_outcomes'?'active':''}" onclick="selectTelemetryPlot('01_scatter_outcomes')">01. SCATTER OBSERVATIONS</button>
                        <button class="btn btn-sm ${currentTelemetryPlot==='02_sustain_rate_heatmap'?'active':''}" onclick="selectTelemetryPlot('02_sustain_rate_heatmap')">02. 2D SUSTAIN HEATMAP</button>
                        <button class="btn btn-sm ${currentTelemetryPlot==='03_data_coverage'?'active':''}" onclick="selectTelemetryPlot('03_data_coverage')">03. COVERAGE DENSITY</button>
                        <button class="btn btn-sm ${currentTelemetryPlot==='04_rate_vs_oxygen'?'active':''}" onclick="selectTelemetryPlot('04_rate_vs_oxygen')">04. RATE VS OXYGEN</button>
                        <button class="btn btn-sm ${currentTelemetryPlot==='05_rate_vs_airflow'?'active':''}" onclick="selectTelemetryPlot('05_rate_vs_airflow')">05. RATE VS AIRFLOW</button>
                        <button class="btn btn-sm ${currentTelemetryPlot==='06_distributions'?'active':''}" onclick="selectTelemetryPlot('06_distributions')">06. DISTRIBUTIONS</button>
                        <button class="btn btn-sm ${currentTelemetryPlot==='07_boundary'?'active':''}" onclick="selectTelemetryPlot('07_boundary')">07. EXTINCTION BOUNDARY</button>
                        <button class="btn btn-sm ${currentTelemetryPlot==='08_burn_risk_surface'?'active':''}" onclick="selectTelemetryPlot('08_burn_risk_surface')">08. BURN RISK SURFACE</button>
                        <button class="btn btn-sm ${currentTelemetryPlot==='all'?'active':''}" onclick="selectTelemetryPlot('all')">ALL PLOTS</button>
                    </div>

                    <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap:12px; background:var(--bg-inset); border:1px solid var(--border-subtle); border-radius:3px; padding:10px 14px; align-items:center;">
                        <div>
                            <label style="font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:var(--text-tertiary); display:block; margin-bottom:4px;">BINS PER AXIS: <span id="bins-val" style="color:var(--text-primary); font-weight:600;">8</span></label>
                            <input type="range" id="param-bins" min="3" max="16" value="${currentTelemetryBins}" oninput="updateBinsLabel(this.value)" style="width:100%;">
                        </div>
                        <div>
                            <label style="font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:var(--text-tertiary); display:block; margin-bottom:4px;">MIN SAMPLES / CELL: <span id="samples-val" style="color:var(--text-primary); font-weight:600;">2</span></label>
                            <input type="range" id="param-samples" min="1" max="10" value="${currentTelemetryMinSamples}" oninput="updateSamplesLabel(this.value)" style="width:100%;">
                        </div>
                        <div>
                            <label style="font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:var(--text-tertiary); display:block; margin-bottom:4px;">BINNING METHOD:</label>
                            <select id="param-method" class="search-input" style="padding:4px 8px; font-size:0.75rem; height:30px;">
                                <option value="uniform" ${currentTelemetryMethod==='uniform'?'selected':''}>UNIFORM (EQUAL SPAN)</option>
                                <option value="quantile" ${currentTelemetryMethod==='quantile'?'selected':''}>QUANTILE (EQUAL FREQ)</option>
                            </select>
                        </div>
                        <div style="display:flex; justify-content:flex-end;">
                            <button class="btn btn-sm" onclick="applyTelemetryParams()" style="background:#1E293B; border-color:var(--accent-cyan); color:var(--accent-cyan);">APPLY PARAMETERS</button>
                        </div>
                    </div>
                </div>

                <div id="telemetry-viewport">
                    <p style="color:var(--text-tertiary); font-family:'JetBrains Mono',monospace; font-size:0.8rem;">Loading microgravity telemetry bundle...</p>
                </div>
            `;
            loadTelemetryBundle();
        }

        init();
    </script>
</body>
</html>
"""


class StandaloneRequestHandler(SimpleHTTPRequestHandler):
    """HTTP Request Handler providing REST API and HTML dashboard."""

    catalog = ResearchCatalog()
    embeddings = None
    ranker = None
    assistant = None

    @classmethod
    def setup_engines(cls):
        invs = cls.catalog.list_all()
        corpus = [inv.to_search_text() for inv in invs]
        cls.embeddings = EmbeddingsManager()
        cls.embeddings.fit(corpus)
        cls.ranker = ResearchRanker(embeddings=cls.embeddings)

        raw_dicts = [inv.model_dump() for inv in invs]
        chunker = DocumentChunker()
        chunks = chunker.chunk_all(raw_dicts)
        retriever = InvestigationRetriever()
        retriever.build_index(chunks)
        cls.assistant = ResearchAssistant(retriever=retriever, catalog=cls.catalog)
        try:
            from src.visualisation.telemetry_studio import TelemetryStudio
            TelemetryStudio.warmup_cache()
        except Exception:
            pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query_params = urllib.parse.parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
            return

        if path in ("/favicon.ico", "/assets/oxv_logo.png"):
            logo_path = Path(__file__).resolve().parent.parent / "assets" / "oxv_logo_cropped.png"
            if logo_path.exists():
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Cache-Control", "public, max-age=86400")
                self.end_headers()
                self.wfile.write(logo_path.read_bytes())
                return
            self.send_response(204)
            self.end_headers()
            return

        if path == "/api/catalog":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            data = [inv.model_dump() for inv in self.catalog.list_all()]
            self.wfile.write(json.dumps(data, default=str).encode("utf-8"))
            return

        if path == "/api/rank":
            q = query_params.get("q", ["fire detection"])[0]
            invs = [inv.model_dump() for inv in self.catalog.list_all()]
            results = self.ranker.rank(q, invs, top_k=6)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            payload = [r.model_dump() for r in results]
            self.wfile.write(json.dumps(payload, default=str).encode("utf-8"))
            return

        if path == "/api/assistant":
            q = query_params.get("q", ["What is the difference between ACME and BASS-II?"])[0]
            resp = self.assistant.answer(q)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp.model_dump(), default=str).encode("utf-8"))
            return

        if path == "/api/assistant/status":
            health = self.assistant.llm_client.check_health() if self.assistant and self.assistant.llm_client else {"status": "unconfigured"}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(health, default=str).encode("utf-8"))
            return

        if path == "/api/telemetry/data":
            dataset = query_params.get("dataset", ["BASS-II"])[0]
            plot_req = query_params.get("plot", ["01_scatter_outcomes"])[0]
            n_bins = int(query_params.get("bins", ["8"])[0])
            min_samples = int(query_params.get("min_samples", ["2"])[0])
            method = query_params.get("method", ["uniform"])[0]

            from src.visualisation.telemetry_studio import TelemetryStudio
            try:
                bundle = TelemetryStudio.get_telemetry_bundle(
                    dataset=dataset,
                    plot=plot_req,
                    bins=n_bins,
                    min_samples=min_samples,
                    method=method,
                )
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(bundle, default=str).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        if path == "/api/telemetry/datasets":
            from src.visualisation.telemetry_studio import TelemetryStudio
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(TelemetryStudio.get_available_datasets()).encode("utf-8"))
            return

        self.send_error(404, "Endpoint not found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if path == "/api/assistant/chat":
            content_len = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_len) if content_len > 0 else b"{}"
            try:
                body = json.loads(body_bytes.decode("utf-8"))
            except Exception:
                body = {}
            question = body.get("question", "").strip()
            history = body.get("history", [])
            use_llm = body.get("use_llm", True)

            resp = self.assistant.answer(question, conversation_history=history, use_llm=use_llm)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp.model_dump(), default=str).encode("utf-8"))
            return

        self.send_error(404, "Endpoint not found")


class ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True


def run_standalone_dashboard(port: int = 8501) -> None:
    """Launch the native Flame in Freefall dashboard server."""
    StandaloneRequestHandler.setup_engines()

    target_port = port
    httpd = None
    for attempt_port in range(port, port + 10):
        try:
            server_address = ("127.0.0.1", attempt_port)
            httpd = ReusableHTTPServer(server_address, StandaloneRequestHandler)
            target_port = attempt_port
            break
        except OSError as e:
            if e.errno == 98:  # Address already in use
                continue
            raise

    if httpd is None:
        raise OSError(f"Could not bind to any port in range {port} to {port + 9}")

    if target_port != port:
        print(f"\n[PORT NOTICE] Port {port} is occupied by another process. Automatically shifted to port {target_port}.")

    print("\n" + "=" * 65)
    print("ORBITAL EXPERIMENTATION VOLUMETRIC (OXV) — RESEARCH WORKSTATION")
    print(f"Server live at: http://127.0.0.1:{target_port}/")
    print("Research pages, telemetry studio, and assistant ready.")
    print("Deterministic fallback synthesis active if local LLM is offline.")
    print("Press Ctrl+C to stop.")
    print("=" * 65 + "\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server...")
        httpd.server_close()
