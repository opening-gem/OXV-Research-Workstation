"""
Document chunker for the Flame in Freefall RAG subsystem.

Splits NASA combustion research investigation documents into searchable
chunks with rich metadata, enabling fine-grained retrieval. Each investigation
is decomposed into typed chunks (description, objective, keyword_summary,
application, context) with configurable size and overlap.
"""

from __future__ import annotations

import hashlib
import logging
import re
import textwrap
from typing import Any

from src.catalog.compat_pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class DocumentChunk(BaseModel):
    """A searchable text fragment extracted from an investigation document."""

    chunk_id: str = Field(
        ...,
        description="Unique identifier: {investigation_id}_{chunk_type}_{index}",
    )
    investigation_id: str = Field(
        ..., description="Source investigation identifier."
    )
    chunk_type: str = Field(
        ...,
        description=(
            "Field the chunk was extracted from: description, objective, "
            "keyword_summary, application, or context."
        ),
    )
    text: str = Field(..., description="The chunk text content.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata carried alongside the chunk.",
    )
    word_count: int = Field(
        ..., description="Number of whitespace-delimited words in *text*."
    )


# ---------------------------------------------------------------------------
# Chunker implementation
# ---------------------------------------------------------------------------

# Mapping from investigation dict keys to our canonical chunk types.
_FIELD_TO_CHUNK_TYPE: dict[str, str] = {
    "description": "description",
    "detailed_description": "description",
    "summary": "description",
    "objective": "objective",
    "objectives": "objective",
    "research_objectives": "objective",
    "keywords": "keyword_summary",
    "keyword_summary": "keyword_summary",
    "tags": "keyword_summary",
    "application": "application",
    "applications": "application",
    "potential_applications": "application",
    "benefits": "application",
    "context": "context",
    "background": "context",
    "justification": "context",
    "rationale": "context",
    "results_summary": "context",
    "results": "context",
}

# Fields whose raw values are lists of strings (e.g. keywords, tags).
_LIST_FIELDS: set[str] = {"keywords", "tags", "applications", "potential_applications"}


class DocumentChunker:
    """
    Splits investigation documents into searchable chunks with metadata.

    Chunking strategy
    -----------------
    * Each investigation produces multiple chunks from different fields.
    * Chunk types: description, objective, keyword_summary, application, context.
    * Each chunk retains its source investigation ID and field type.
    * Long texts are split with word-level overlap for context continuity.

    Parameters
    ----------
    chunk_size : int
        Maximum number of **words** per chunk (default 300).
    overlap : int
        Number of overlapping words between consecutive chunks of the same
        field (default 50).
    """

    def __init__(self, chunk_size: int = 300, overlap: int = 50) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if overlap < 0:
            raise ValueError("overlap must be >= 0")
        if overlap >= chunk_size:
            raise ValueError("overlap must be < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_investigation(self, investigation: dict[str, Any]) -> list[DocumentChunk]:
        """Create searchable chunks from a single investigation dict.

        The investigation dict is expected to have at least an ``"id"`` key
        (or ``"investigation_id"``) and one or more text fields listed in
        ``_FIELD_TO_CHUNK_TYPE``.

        Returns a (possibly empty) list of :class:`DocumentChunk` objects.
        """
        inv_id = self._extract_id(investigation)
        if inv_id is None:
            logger.warning("Investigation missing id – skipping: %s", investigation)
            return []

        # Collect base metadata that every chunk inherits.
        base_meta = self._extract_base_metadata(investigation)

        chunks: list[DocumentChunk] = []
        seen_texts: set[str] = set()  # de-duplicate identical text across aliased fields

        for field_key, chunk_type in _FIELD_TO_CHUNK_TYPE.items():
            raw_value = investigation.get(field_key)
            if raw_value is None:
                continue

            text = self._normalise_value(raw_value, field_key)
            if not text or text in seen_texts:
                continue
            seen_texts.add(text)

            field_chunks = self._split_text(text)
            for idx, fragment in enumerate(field_chunks):
                word_count = len(fragment.split())
                chunk_id = f"{inv_id}_{chunk_type}_{idx}"
                meta = {
                    **base_meta,
                    "source_field": field_key,
                    "chunk_index": idx,
                    "total_field_chunks": len(field_chunks),
                }
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        investigation_id=inv_id,
                        chunk_type=chunk_type,
                        text=fragment,
                        metadata=meta,
                        word_count=word_count,
                    )
                )

        # If we extracted nothing from mapped fields, create a single
        # "context" chunk from the first non-trivial string value we find.
        if not chunks:
            fallback_text = self._fallback_text(investigation)
            if fallback_text:
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{inv_id}_context_0",
                        investigation_id=inv_id,
                        chunk_type="context",
                        text=fallback_text,
                        metadata={**base_meta, "source_field": "_fallback", "chunk_index": 0, "total_field_chunks": 1},
                        word_count=len(fallback_text.split()),
                    )
                )

        return chunks

    def chunk_all(self, investigations: list[dict[str, Any]]) -> list[DocumentChunk]:
        """Chunk every investigation in a list and return the flat collection."""
        all_chunks: list[DocumentChunk] = []
        for inv in investigations:
            all_chunks.extend(self.chunk_investigation(inv))
        logger.info(
            "Chunked %d investigations → %d chunks", len(investigations), len(all_chunks)
        )
        return all_chunks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_id(investigation: dict[str, Any]) -> str | None:
        """Return the investigation id, trying several key names."""
        for key in ("id", "investigation_id", "inv_id", "identifier"):
            val = investigation.get(key)
            if val is not None:
                return str(val)
        # Last resort: hash the dict
        try:
            raw = str(sorted(investigation.items()))
            return "auto_" + hashlib.md5(raw.encode()).hexdigest()[:12]
        except Exception:
            return None

    @staticmethod
    def _extract_base_metadata(investigation: dict[str, Any]) -> dict[str, Any]:
        """Pull out metadata fields useful for filtering / display."""
        meta: dict[str, Any] = {}
        for key in (
            "title",
            "category",
            "categories",
            "platform",
            "status",
            "start_date",
            "end_date",
            "url",
            "doi",
            "active_version",
        ):
            val = investigation.get(key)
            if val is not None:
                meta[key] = val

        # Ensure url is populated from psi_url or source_url
        if not meta.get("url"):
            meta["url"] = investigation.get("psi_url") or investigation.get("source_url")

        # Ensure doi is populated from active version if not on top-level
        if not meta.get("doi"):
            versions = investigation.get("versions") or []
            if versions:
                active_v = investigation.get("active_version")
                for v in versions:
                    if isinstance(v, dict):
                        if active_v is None or v.get("version_number") == active_v:
                            if v.get("doi"):
                                meta["doi"] = v.get("doi")
                                break
                    elif hasattr(v, "doi") and getattr(v, "doi"):
                        meta["doi"] = getattr(v, "doi")
                        break
        return meta

    @staticmethod
    def _normalise_value(value: Any, field_key: str) -> str:
        """Convert a raw field value into a single cleaned text string."""
        if isinstance(value, list):
            # Lists of keywords / tags → comma-joined sentence.
            parts = [str(v).strip() for v in value if v]
            return ", ".join(parts)
        if isinstance(value, dict):
            # Nested dicts → concatenate values.
            parts = [str(v).strip() for v in value.values() if v]
            return " ".join(parts)
        text = str(value).strip()
        # Collapse excessive whitespace.
        text = re.sub(r"\s+", " ", text)
        return text

    def _split_text(self, text: str) -> list[str]:
        """Split *text* into word-bounded chunks with overlap."""
        words = text.split()
        if len(words) <= self.chunk_size:
            return [text]

        chunks: list[str] = []
        step = self.chunk_size - self.overlap
        for start in range(0, len(words), step):
            end = start + self.chunk_size
            fragment = " ".join(words[start:end])
            chunks.append(fragment)
            if end >= len(words):
                break
        return chunks

    @staticmethod
    def _fallback_text(investigation: dict[str, Any]) -> str:
        """Build a minimal text from the investigation's title + any string values."""
        parts: list[str] = []
        title = investigation.get("title") or investigation.get("name")
        if title:
            parts.append(str(title))
        for key, val in investigation.items():
            if key in ("id", "investigation_id", "url"):
                continue
            if isinstance(val, str) and len(val) > 20:
                parts.append(val.strip())
                if len(parts) >= 3:
                    break
        return " | ".join(parts) if parts else ""
