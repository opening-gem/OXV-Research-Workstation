# Flame in Freefall

Flame in Freefall is a research intelligence platform designed to ingest, normalize, classify, and search NASA microgravity combustion experiments.

Decades of space combustion research—from drop towers and parabolic flights to glovebox experiments aboard the International Space Station (ISS)—contain vital data for spacecraft fire safety, lunar base design (Artemis), and terrestrial energy efficiency. However, flight logs and datasets across the NASA Physical Sciences Informatics (PSI) repository often have inconsistent field definitions, unassigned outcome labels, and fragmented dataset revisions.

This repository unifies those records into a version-tracked catalog paired with machine learning classification, hybrid retrieval ranking, and an evidence-grounded research assistant.

---

## Key Capabilities

- **Canonical Research Catalog & Provenance Tracking**: Ingests raw NASA PSI records into validated Pydantic models. Tracks dataset revisions, version bumps, and persistent DOIs (e.g., BASS-II v4: `10.60555/4qc4-de67`).
- **Multi-Label Research Classification**: Classifies experiments into combustion research categories (flame spread, droplet combustion, cool flames, spacecraft fire safety) using TF-IDF feature extraction and One-vs-Rest regularized Logistic Regression with per-feature attribution weights.
- **Explainable Hybrid Ranking**: Evaluates search queries using a 4-factor scoring model:
  $$\text{Score}(q, d) = 0.25 \cdot \text{BM25}(q, d) + 0.35 \cdot \cos(\vec{q}, \vec{d}) + 0.20 \cdot \text{Metadata}(q, d) + 0.20 \cdot \text{Evidence}(d)$$
- **Evidence-Grounded Research Assistant**: Combines multi-stage chunk retrieval with local/remote LLMs (via Ollama). Validates cited investigation IDs against retrieved source chunks to eliminate ungrounded citations, with a deterministic fallback mode when running offline.
- **Flight Telemetry Studio**: Analyzes experimental run tables (e.g., BASS-II solid burning and FLEX droplet combustion), plotting flammability boundaries across oxygen concentration and convective airflow velocity.

---

## Architecture Overview

```
                      NASA Research Sources
               (PSI Archive • Flight Telemetry)
                             │
                             ▼
                 Ingestion & Normalization
            (Metadata Cleaning • Version Audit)
                             │
                             ▼
                    Research Repository
               (Schema Validation • catalog.json)
                             │
             ┌───────────────┴───────────────┐
             ▼                               ▼
       AI Subsystems                  Flight Telemetry
   ┌───────────────────────┐       ┌───────────────────┐
   │ • Topic Classifier    │       │ • BASS-II / FLEX  │
   │ • Hybrid BM25 Ranker  │       │ • Flammability Map│
   │ • Ollama RAG Engine   │       │ • Critical Flow   │
   └──────────┬────────────┘       └─────────┬─────────┘
              └──────────────┬───────────────┘
                             ▼
                      User Interfaces
            (Streamlit Dashboard • Standalone Web UI • CLI)
```

---

## Quickstart

> **Prerequisites:** Python 3.11+ is required.

### 1. Installation

```bash
git clone https://github.com/opening-gem/orbital-experimentation-volumetric

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Ingest Data & Build the Catalog

Ingest the raw NASA PSI seed data, validate fields, and establish version records:

```bash
python main.py ingest
```

### 3. Train the Classifier & Run Evaluations

```bash
# Train research classifier and tag catalog
python main.py classify

# Run benchmark evaluation across classifier, ranker, and retrieval
python main.py evaluate
```

### 4. Launch the Dashboard

Launch the Streamlit dashboard:

```bash
python main.py dashboard --port 8501
```

*Note: If Streamlit is not installed, the CLI automatically starts the built-in standalone server on the same port.*

---

## CLI Usage

```text
flame-in-freefall CLI

Commands:
  ingest       Ingest NASA PSI records and generate version manifests
  classify     Train multi-label classifier and predict categories
  search       Perform hybrid search over investigations
  evaluate     Run benchmark evaluations (F1, NDCG@5, Citation validity)
  dashboard    Start the interactive mission dashboard
  llm-check    Test connection and model availability for Ollama
```

Example search:

```bash
python main.py search "smoke detection lunar habitat" --top-k 3
```

---

## Ollama LLM Assistant Configuration

The research assistant can run completely offline using a local or remote Ollama instance. It connects to the Ollama HTTP API (`POST /api/chat`) and formats retrieved PSI evidence into structured context blocks.

### Environment Variables

```bash
# Optional configuration (defaults to local Ollama on 11434)
export LLM_PROVIDER=ollama
export OLLAMA_BASE_URL=http://127.0.0.1:11434
export OLLAMA_MODEL=llama3
export OLLAMA_TIMEOUT_SECONDS=120
export OLLAMA_TEMPERATURE=0.2

# If connecting to an authenticated tunnel (e.g. Cloudflare / ngrok):
export OLLAMA_AUTH_HEADER="Bearer <token>"
```

Verify your connection anytime:

```bash
python main.py llm-check
```

If Ollama is offline or the target model is not pulled, the assistant automatically falls back to deterministic research synthesis so the dashboard remains functional.

---

## Repository Structure

```
flame-in-freefall/
├── main.py                  # Unified CLI entrypoint
├── app.py                   # Streamlit application entrypoint
├── requirements.txt         # Project dependencies
├── src/
│   ├── config.py            # Path resolution and runtime configuration
│   ├── catalog/             # Data models, validation, and JSON repository
│   │   ├── schema.py        # Pydantic v2 schemas
│   │   ├── repository.py    # Thread-safe catalog storage
│   │   └── normalizer.py    # Text cleaning and DOI validation
│   ├── ingestion/           # Seed ingestion, web scrapers, version tracker
│   ├── ai/                  # Classifier, embeddings, ranker, and evaluation
│   ├── rag/                 # Chunker, hybrid retriever, and Ollama assistant
│   └── visualisation/       # Telemetry studio and plot generation
├── dashboard/               # Streamlit multi-page UI & standalone web server
│   ├── pages/               # Dashboard pages (Overview, Ranking, Assistant, etc.)
│   └── standalone_server.py # Zero-dependency HTTP server with vanilla JS UI
├── flame_project/           # Flight telemetry parsers (BASS-II & FLEX adapters)
├── data/
│   ├── raw/                 # Raw seed JSON and flight CSVs
│   ├── processed/           # Processed catalog.json and training sets
│   └── manifests/           # Ingestion manifests and version history
├── tests/                   # Test suite (AI, catalog, RAG, telemetry)
└── docs/                    # Architecture and data dictionary specs
```

---

## Testing

Run the test suite:

```bash
python -m unittest discover tests/
```

Or with `pytest`:

```bash
pytest tests/ -v
```

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.
