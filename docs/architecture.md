# Flame in Freefall: System Architecture

## 1. Executive Summary

**Flame in Freefall** is an AI-powered scientific research intelligence platform for NASA microgravity combustion research. Rather than acting as a static dashboard displaying pre-aggregated tables, it functions as an autonomous, evidence-grounded research intelligence engine that structures, version-tracks, classifies, and ranks scientific investigations conducted aboard the International Space Station (ISS), drop towers, and parabolic flights.

The system is engineered around three demonstrable pillars:
1. **Canonical, Versioned Research Database**: Indexed from NASA Physical Sciences Informatics (PSI), maintaining persistent identifiers (DOIs) and tracking dataset revisions.
2. **Self-Built AI Subsystems**: Native TF-IDF vectorization, multi-label Logistic Regression with L2 regularization, explainable hybrid BM25 + semantic ranking, and Retrieval-Augmented Generation (RAG).
3. **Multi-Page Mission Dashboard**: Interactive intelligence turning raw space combustion data into operational insights for lunar (Artemis), Martian, and orbital habitat fire safety.

---

## 2. High-Level System Architecture

```
                       NASA KNOWLEDGE SOURCES
   ┌──────────────────────────────────────────────────────────────┐
   │  NASA PSI Repository  •  NASA Glenn Combustion Science       │
   │  Researcher's Guides  •  Historical Spaceflight Incident Logs │
   └──────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
                     DATA INGESTION & AUDIT LAYER
   ┌──────────────────────────────────────────────────────────────┐
   │  PSICatalogIngester  •  InvestigationNormalizer              │
   │  DocumentParser      •  VersionTracker & IngestionManifest   │
   └──────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
                      NORMALISATION & STORAGE
   ┌──────────────────────────────────────────────────────────────┐
   │  Pydantic v2 Schema  •  catalog.json (Locked Repository)     │
   │  Investigation, InvestigationVersion, Investigator Models    │
   └──────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
                        AI / ML INTELLIGENCE ENGINE
   ┌──────────────────────────────────────────────────────────────┐
   │  Model 1: ResearchClassifier (Multi-Label Logistic Reg)      │
   │  Model 2: ResearchRanker (BM25 + Semantic Cosine + Evidence) │
   │  Model 3: ResearchAssistant (Extractive & Cited RAG)         │
   │  Evaluation: AIEvaluator (Macro F1, NDCG@5, Precision@5)     │
   └──────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
                       APPLICATION & INTERFACE LAYER
   ┌──────────────────────────────────────────────────────────────┐
   │  Streamlit Multi-Page UI (app.py)                           │
   │  Native Standalone Web Server (standalone_server.py:8501)    │
   │  CLI Command Suite (main.py ingest|classify|search|evaluate) │
   └──────────────────────────────────────────────────────────────┘
```

---

## 3. Data Flow and Ingestion Pipeline

1. **Extraction**: Raw metadata records from NASA Physical Sciences Informatics (PSI) seed repository (`seed_investigations.json`) and NASA Science pages are ingested by `PSICatalogIngester`.
2. **Normalisation**: `InvestigationNormalizer` sanitizes Unicode strings, normalizes experimental platforms and status codes, and validates DOIs (matching `10.xxxx/...`).
3. **Version & Provenance Audit**: `VersionTracker` registers dataset revision histories, ensuring that newer releases advance active version pointers without erasing older versions.
4. **Catalogue Storage**: Thread-safe storage with OS-level advisory file locking (`fcntl.flock`) in `ResearchCatalog`, stored at `data/processed/catalog.json`.

---

## 4. AI Subsystem Design

### Model 1: Research Classification Engine
- **Representation**: Sublinear TF-IDF vectorizer ($1 + \log(TF)$) with unigram and bigram feature extraction ($ngram\_range=(1,2)$) and smooth IDF weighting.
- **Classifier**: Multi-label One-vs-Rest regularized Logistic Regression.
- **Explainability**: Computes linear attribution weights $w_j \cdot x_j$ for top contributing terms per assigned category.

### Model 2: Relevance Ranking Engine
- **Hybrid Scoring**: Combines four distinct operational signals:
  $$\text{Score}(q, d) = w_{\text{kw}} \cdot \text{BM25}(q, d) + w_{\text{sem}} \cdot \cos(\vec{q}, \vec{d}) + w_{\text{meta}} \cdot M(q, d) + w_{\text{ev}} \cdot E(d)$$
- **Default Weights**: $w_{\text{kw}}=0.25$, $w_{\text{sem}}=0.35$, $w_{\text{meta}}=0.20$, $w_{\text{ev}}=0.20$.
- **Explainability**: Highlights matching tokens, relevant objectives, and signal breakdowns.

### Model 3: Retrieval-Augmented Research Assistant & Ollama LLM
- **Chunking**: `DocumentChunker` splits investigations into distinct objective, description, and application chunks with parent metadata (`title`, `url`, `doi`, `active_version`).
- **Two-Stage Hybrid Retrieval**: Stage 1 candidate pool extraction; Stage 2 BM25 re-ranking with chunk-type weighting and normalized relevance scoring.
- **Reusable Ollama Client**: `OllamaClient` (`src/rag/ollama_client.py`) interfaces with Ollama via `POST /api/chat` and `GET /api/tags`. Fully parameterized via `LLMConfig` (`provider`, `base_url`, `model`, `timeout_seconds`, `temperature`, `auth_header`).
- **Structured System Prompt & Evidence Injection**: Injects formatted `[EVIDENCE i]` blocks into the LLM system prompt alongside NASA domain constraints:
  - Factual grounding in supplied evidence only;
  - Explicit instruction to abstain when evidence is insufficient;
  - Mandatory citation using bracketed investigation identifiers (e.g. `[BASS-II]`);
  - Rejection of fabricated DOIs, researchers, or mission claims.
- **Citation Extraction and Verification Guardrail**:
  - LLM bracketed citations are extracted and verified against the set of retrieved evidence IDs.
  - Legitimate citations are enriched with persistent DOIs, official NASA PSI URLs, and active version numbers.
  - Fabricated or ungrounded IDs are systematically rejected, stripped from the displayed source references, and reported in the response disclaimer.
- **Dual-Interface Shared Backend**:
  - Both Streamlit (`dashboard/pages/05_research_assistant.py`) and the native standalone web server (`dashboard/standalone_server.py`) execute identical RAG logic via `assistant.answer(...)`.
  - All LLM interactions occur strictly server-side; no network credentials, API keys, or raw endpoints are ever exposed to client-side JavaScript.
- **Deterministic Fallback**: If the Ollama server is unreachable, encounters a 404 model error, or times out, the assistant transparently falls back to deterministic template synthesis with an explanatory disclaimer.

---

## 5. Evaluation Framework
- **Evaluation Engine**: `AIEvaluator` (`src/ai/evaluation.py`) measures each subsystem with standard metrics:
  - Classification: Macro F1, Macro Precision, Macro Recall, per-category support.
  - Ranking: Precision@k, NDCG@k with graded relevance judgments.
  - Conversational RAG: Citation validity rate, evidence relevance score, answer groundedness rate, hallucination rate, and abstention accuracy on out-of-domain queries.
  - Provenance Audit: 100% verification of persistent identifiers (DOIs) and revision logs across the catalogue.

---

## 6. Provenance and Legacy BASS-II Preservation
- **BASS-II Record**: Version 4 canonical record linked to persistent DOI `10.60555/4qc4-de67`.
- **Legacy Module**: `legacy/bass_ii/adapters.py` and `legacy/bass_ii/analysis.py` maintain full reproducibility of historical ISS flame spread figures without entangling modern indexing.
