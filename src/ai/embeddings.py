"""
Embeddings Manager for Flame in Freefall.

Manages text embeddings for semantic search and similarity across
NASA combustion research investigations.

Supports two modes:
1. TF-IDF embeddings — no external dependencies, works offline.
2. Sentence Transformer embeddings — higher quality, requires
   ``sentence-transformers`` package and model download.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import normalize
except ImportError:
    from src.ai.compat_ml import TfidfVectorizer, cosine_similarity, normalize

logger = logging.getLogger(__name__)

# Sentinel for lazy import of sentence_transformers
_SENTENCE_TRANSFORMERS_AVAILABLE: bool | None = None


def _check_sentence_transformers() -> bool:
    """Check if sentence-transformers is importable."""
    global _SENTENCE_TRANSFORMERS_AVAILABLE
    if _SENTENCE_TRANSFORMERS_AVAILABLE is None:
        try:
            import sentence_transformers  # noqa: F401

            _SENTENCE_TRANSFORMERS_AVAILABLE = True
        except ImportError:
            _SENTENCE_TRANSFORMERS_AVAILABLE = False
    return _SENTENCE_TRANSFORMERS_AVAILABLE


class EmbeddingsManager:
    """Manages text embeddings for semantic search and similarity.

    Supports two operational modes:

    * **tfidf** (default) — uses sklearn's ``TfidfVectorizer`` to
      produce sparse embeddings that are densified for cosine ops.
      Zero external dependencies beyond scikit-learn.
    * **transformer** — uses ``sentence-transformers`` to produce
      dense embeddings with a pre-trained model.  Falls back to
      TF-IDF automatically when the package is unavailable.

    Example:
        >>> em = EmbeddingsManager(mode="tfidf")
        >>> em.fit(["flame spread in microgravity", "droplet combustion"])
        >>> vec = em.encode("fire safety")
        >>> em.most_similar("fire", ["flame spread", "droplet burning"], top_k=1)
        [(0, 0.42)]
    """

    # Default transformer model — small, fast, good for scientific text
    DEFAULT_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        mode: str = "tfidf",
        transformer_model: str | None = None,
    ) -> None:
        if mode not in ("tfidf", "transformer"):
            raise ValueError(f"Unsupported mode '{mode}'. Use 'tfidf' or 'transformer'.")

        self._requested_mode = mode
        self._transformer_model_name = transformer_model or self.DEFAULT_TRANSFORMER_MODEL

        # Resolve effective mode
        if mode == "transformer" and not _check_sentence_transformers():
            logger.warning(
                "sentence-transformers not installed — falling back to TF-IDF mode"
            )
            self._effective_mode = "tfidf"
        else:
            self._effective_mode = mode

        # TF-IDF components
        self._tfidf_vectorizer: TfidfVectorizer | None = None
        self._tfidf_fitted: bool = False

        # Transformer components
        self._transformer_model: Any = None

        # Cached corpus index
        self._corpus_texts: list[str] = []
        self._corpus_embeddings: np.ndarray | None = None

        self._initialize()

    @property
    def mode(self) -> str:
        """The effective embedding mode in use."""
        return self._effective_mode

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the produced embeddings."""
        if self._effective_mode == "transformer" and self._transformer_model is not None:
            return int(self._transformer_model.get_sentence_embedding_dimension())
        if self._tfidf_fitted and self._tfidf_vectorizer is not None:
            return len(self._tfidf_vectorizer.get_feature_names_out())
        return 0

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        """Set up the selected backend."""
        if self._effective_mode == "tfidf":
            self._tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                stop_words="english",
                sublinear_tf=True,
            )
        elif self._effective_mode == "transformer":
            from sentence_transformers import SentenceTransformer

            self._transformer_model = SentenceTransformer(self._transformer_model_name)
            logger.info(
                "Loaded transformer model '%s'", self._transformer_model_name
            )

    # ------------------------------------------------------------------
    # Fitting (TF-IDF only — transformers are pre-trained)
    # ------------------------------------------------------------------

    def fit(self, texts: list[str]) -> None:
        """Fit the embedding model on a corpus of texts.

        For TF-IDF mode this fits the vectorizer vocabulary.
        For transformer mode this is a no-op (model is pre-trained)
        but the corpus is stored for index building.

        Args:
            texts: List of text strings to fit on.

        Raises:
            ValueError: If texts is empty.
        """
        if not texts:
            raise ValueError("Cannot fit on an empty text list")

        if self._effective_mode == "tfidf":
            assert self._tfidf_vectorizer is not None
            self._tfidf_vectorizer.fit(texts)
            self._tfidf_fitted = True
            logger.info(
                "TF-IDF vectorizer fitted on %d documents (%d features)",
                len(texts),
                len(self._tfidf_vectorizer.get_feature_names_out()),
            )

        # Cache corpus embeddings for similarity lookups
        self._corpus_texts = list(texts)
        self._corpus_embeddings = self.encode_batch(texts)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, text: str) -> np.ndarray:
        """Encode a single text string to a dense embedding vector.

        Args:
            text: Input text.

        Returns:
            1-D numpy array of shape ``(embedding_dim,)``.

        Raises:
            RuntimeError: If TF-IDF mode is used and ``fit`` has not
                been called.
        """
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode multiple texts to dense embedding vectors.

        Args:
            texts: List of input texts.

        Returns:
            2-D numpy array of shape ``(n_texts, embedding_dim)``.
        """
        if not texts:
            return np.empty((0, 0))

        if self._effective_mode == "transformer":
            embeddings = self._transformer_model.encode(
                texts, show_progress_bar=False, convert_to_numpy=True
            )
            return np.asarray(embeddings)

        # TF-IDF mode
        if not self._tfidf_fitted:
            raise RuntimeError(
                "TF-IDF vectorizer has not been fitted. Call fit() first."
            )
        assert self._tfidf_vectorizer is not None
        sparse = self._tfidf_vectorizer.transform(texts)
        # Densify and L2-normalize for cosine similarity
        dense = sparse.toarray() if hasattr(sparse, "toarray") else np.asarray(sparse)
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # avoid division by zero
        return dense / norms

    # ------------------------------------------------------------------
    # Similarity
    # ------------------------------------------------------------------

    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts.

        Args:
            text1: First text.
            text2: Second text.

        Returns:
            Cosine similarity in ``[-1, 1]``.
        """
        emb1 = self.encode(text1).reshape(1, -1)
        emb2 = self.encode(text2).reshape(1, -1)
        return float(cosine_similarity(emb1, emb2)[0, 0])

    def most_similar(
        self,
        query: str,
        corpus: list[str] | None = None,
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """Find the most similar texts to *query* in a corpus.

        If *corpus* is ``None``, the corpus passed to :meth:`fit` is used.

        Args:
            query: Query text.
            corpus: Optional list of candidate texts. If provided, they
                are encoded on-the-fly.
            top_k: Number of results to return.

        Returns:
            List of ``(index, similarity)`` tuples sorted by descending
            similarity.
        """
        query_emb = self.encode(query).reshape(1, -1)

        if corpus is not None:
            corpus_emb = self.encode_batch(corpus)
        elif self._corpus_embeddings is not None:
            corpus_emb = self._corpus_embeddings
        else:
            raise RuntimeError(
                "No corpus provided and no corpus has been fitted"
            )

        if corpus_emb.shape[0] == 0:
            return []

        sims = cosine_similarity(query_emb, corpus_emb).flatten()
        top_k = min(top_k, len(sims))
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(int(i), float(sims[i])) for i in top_indices]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_index(self, path: Path) -> None:
        """Save the fitted model and cached corpus embeddings.

        Args:
            path: Destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        bundle: dict[str, Any] = {
            "mode": self._effective_mode,
            "corpus_texts": self._corpus_texts,
            "corpus_embeddings": self._corpus_embeddings,
        }

        if self._effective_mode == "tfidf":
            bundle["tfidf_vectorizer"] = self._tfidf_vectorizer
            bundle["tfidf_fitted"] = self._tfidf_fitted

        # Note: transformer models are loaded by name, not persisted
        if self._effective_mode == "transformer":
            bundle["transformer_model_name"] = self._transformer_model_name

        with open(path, "wb") as f:
            pickle.dump(bundle, f)
        logger.info("Embeddings index saved to %s", path)

    def load_index(self, path: Path) -> None:
        """Load a previously saved model and corpus embeddings.

        Args:
            path: Source file path.

        Raises:
            FileNotFoundError: If path does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")

        with open(path, "rb") as f:
            bundle: dict[str, Any] = pickle.load(f)

        self._corpus_texts = bundle.get("corpus_texts", [])
        self._corpus_embeddings = bundle.get("corpus_embeddings", None)

        saved_mode = bundle.get("mode", "tfidf")

        if saved_mode == "tfidf":
            self._tfidf_vectorizer = bundle.get("tfidf_vectorizer")
            self._tfidf_fitted = bundle.get("tfidf_fitted", False)
            self._effective_mode = "tfidf"
        elif saved_mode == "transformer":
            model_name = bundle.get(
                "transformer_model_name", self.DEFAULT_TRANSFORMER_MODEL
            )
            self._transformer_model_name = model_name
            if _check_sentence_transformers():
                self._effective_mode = "transformer"
                from sentence_transformers import SentenceTransformer

                self._transformer_model = SentenceTransformer(model_name)
            else:
                logger.warning(
                    "Saved index used transformer mode but package is "
                    "unavailable — falling back to TF-IDF (corpus "
                    "embeddings still loaded)"
                )
                self._effective_mode = "tfidf"

        logger.info("Embeddings index loaded from %s", path)
