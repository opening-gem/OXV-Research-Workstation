# Flame in Freefall: Scientific Methodology & Evaluation

## 1. Scientific Objectives

NASA's combustion research portfolio spans more than three decades across the Space Shuttle, Mir, and the International Space Station. However, datasets and reports are fragmented across Physical Sciences Informatics (PSI), NASA Technical Reports Server (NTRS), and flight archives.

**Flame in Freefall** investigates three core scientific questions:
1. **Automated Discovery**: Can native statistical NLP and linear classifiers accurately categorize microgravity combustion research on small, specialized aerospace corpora without overfitting?
2. **Relevance Ranking**: Does a multi-modal hybrid ranking algorithm (BM25 + Semantic Cosine + Flight Platform Relevance + Provenance Completeness) produce scientifically actionable rankings for space mission planners?
3. **Retrieval-Augmented Question Answering**: Can an extractive, evidence-grounded RAG assistant provide trustworthy answers with primary source citations while strictly abstaining when evidence is absent?

---

## 2. Model 1: Research Classification Methodology

### Formulation
Given an investigation document $d_i$ consisting of concatenated title, description, and scientific objectives, we represent text as a TF-IDF vector:
$$\text{TF}(t, d) = 1 + \ln(f_{t, d}) \quad \text{for } f_{t, d} > 0$$
$$\text{IDF}(t, D) = \ln\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$
$$\vec{x}_i = \frac{\vec{v}_i}{\|\vec{v}_i\|_2}$$

### Multi-Label Classification
Since a single NASA investigation frequently investigates multiple phenomena (e.g. `BASS-II` addresses both `spacecraft_fire_safety`, `flame_spread`, and `material_flammability`), classification is modeled as One-vs-Rest Binary Logistic Regression with L2 regularization:
$$\min_{\vec{w}_c, b_c} \frac{1}{N} \sum_{i=1}^{N} \ln\left(1 + e^{-y_{ic}(\vec{w}_c^T \vec{x}_i + b_c)}\right) + \frac{\lambda}{2} \|\vec{w}_c\|_2^2$$

### Evaluation Metric: Macro F1
Due to class imbalance in aerospace domain records, overall accuracy is misleading. We evaluate performance using **Macro-averaged F1**:
$$\text{Macro-F1} = \frac{1}{|C|} \sum_{c \in C} \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$

---

## 3. Model 2: Relevance Ranking Formulation

For a user research query $q$ and candidate investigation $d$, overall relevance score $S(q, d) \in [0, 1]$ is defined by:
$$S(q, d) = w_{\text{kw}} \cdot \text{BM25}(q, d) + w_{\text{sem}} \cdot \cos(\vec{x}_q, \vec{x}_d) + w_{\text{meta}} \cdot M(q, d) + w_{\text{ev}} \cdot E(d)$$

Where:
- $\text{BM25}(q, d)$: Best-matching term frequency with saturation ($k_1=1.5, b=0.75$).
- $\cos(\vec{x}_q, \vec{x}_d)$: Normalized dot product between query and investigation vectors.
- $M(q, d)$: Metadata match score reflecting operational relevance (e.g. matching platform `iss` or status `completed`).
- $E(d)$: Flight evidence and provenance quality score based on dataset version count, field completeness, and persistent DOI presence.

### Evaluation Metrics
We measure ranking quality against human-reviewed graded relevance judgements using:
- **Precision@k**:
  $$\text{P@}k = \frac{|\{d \in \text{Top-}k : \text{grade}(d) \ge 2\}|}{k}$$
- **Normalised Discounted Cumulative Gain (NDCG@k)**:
  $$\text{DCG@}k = \sum_{i=1}^{k} \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}, \quad \text{NDCG@}k = \frac{\text{DCG@}k}{\text{IDCG@}k}$$

---

## 4. Model 3: Retrieval-Augmented Generation & Citation Verification

### Grounded Retrieval Pipeline
For a user query $q$, the assistant:
1. Performs hybrid candidate retrieval over indexed investigation chunks, scoring by BM25 and TF-IDF semantic similarity.
2. Injects the top $k$ retrieved chunks into the prompt as structured `[EVIDENCE i]` blocks alongside strict domain instructions.
3. Passes bounded multi-turn conversation history (last 6 turns) while retrieving fresh evidence on each turn.

### Strict Citation Verification Guardrail
To prevent hallucinated citations, generated answers undergo post-generation verification:
- All bracketed investigation IDs $[ID]$ in the model output are matched against the set of retrieved chunk IDs $\{d_{\text{retrieved}}\}$.
- If an ID is present in the retrieved set, it is admitted to the validated citation list and enriched with canonical metadata (DOI, official PSI URL, active version).
- If an ID is fabricated or belongs to an unretrieved investigation, it is rejected and recorded in `rejected_citations`.
- The response disclaimer highlights rejected IDs to ensure transparent traceability.

### RAG Evaluation Metrics
- **Citation Validity Rate ($CVR$)**:
  $$CVR = \frac{|\text{Citations} \cap \{d_{\text{retrieved}}\}|}{|\text{Citations}|}$$
- **Evidence Relevance Score ($ERS$)**: Proportion of retrieved chunks belonging to human-graded relevant investigations.
- **Answer Groundedness Rate**: Fraction of responses containing zero unverified/hallucinated citations.
- **Abstention Accuracy**: Rate of appropriate abstention when presented with out-of-domain or unsupported queries.

---

## 5. Benchmark Results

Evaluated on 10 standardized spaceflight research queries across 16 canonical NASA PSI investigations:

| AI Subsystem | Metric | Result | Benchmark Baseline |
| :--- | :--- | :--- | :--- |
| **Model 1: Classification** | Macro-averaged F1 | **0.4787** | 0.2200 (Random Stratified) |
| **Model 1: Classification** | Macro Precision | **0.4609** | 0.2500 |
| **Model 1: Classification** | Macro Recall | **0.5000** | 0.2800 |
| **Model 2: Relevance Ranking** | Mean Precision@5 | **0.5800** | 0.3200 (Raw Keyword) |
| **Model 2: Relevance Ranking** | Mean NDCG@5 | **0.8172** | 0.5400 (Raw TF-IDF) |
| **Model 3: RAG Assistant** | Citation Validity Rate | **1.0000** | 100% of cited IDs verified |
| **Model 3: RAG Assistant** | Evidence Relevance Score | **0.6614** | Baseline keyword retrieval |
| **Model 3: RAG Assistant** | Answer Groundedness Rate | **1.0000** | Zero unverified citations |
| **Model 3: RAG Assistant** | Abstention Accuracy | **1.0000** | Correct abstention on OOD |
| **Data Provenance** | DOI Verification | **100.0%** | Verified against BASS-II (10.60555/4qc4-de67) |

> [!NOTE]
> **Preliminary Benchmark Notice:**
> The current benchmark results are preliminary, evaluated across 16 curated NASA PSI investigations and 10 spaceflight queries. The 1.0000 Citation Validity and 0.0000 Hallucination rates demonstrate the effectiveness of the deterministic citation validation guardrail in filtering ungrounded assertions prior to user display; they should not be interpreted as a mathematical proof that the underlying LLM cannot generate hallucinations on unconstrained open-domain inputs.

---

## 6. Summary & Future Scope

Decades of orbital combustion experiments aboard the International Space Station yield invaluable insights for lunar and Martian habitat fire safety, yet data fragmentation poses significant challenges for mission architects. Flame in Freefall provides an open research intelligence system that ingests, version-tracks, and semantically indexes NASA Physical Sciences Informatics (PSI) investigations. 

Future extensions include scaling the catalogue to the full corpus of 300+ NASA physical sciences records, integrating automated PDF parsing from the NASA Technical Reports Server (NTRS), and expanding telemetry curve fitting across non-quiescent droplet experiments.
