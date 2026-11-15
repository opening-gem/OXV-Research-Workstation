"""
Document and Text Parser for Combustion Science Ingestion.

Extracts structured sections, scientific objectives, and key phrases from raw text
and prepares standardized chunks for semantic indexing and RAG retrieval.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from src.catalog.schema import Investigation


class DocumentParser:
    """Parses scientific text, extracts key concepts, and generates structured document chunks."""

    def __init__(self):
        self.stop_words = {
            "the", "and", "in", "of", "to", "a", "for", "on", "with", "as", "by", "is", "at", "from"
        }

    def extract_keywords(self, text: str, max_keywords: int = 8) -> List[str]:
        """Extract top frequent domain keywords from text."""
        words = re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", text.lower())
        meaningful = [w for w in words if w not in self.stop_words]
        counts: Dict[str, int] = {}
        for w in meaningful:
            counts[w] = counts.get(w, 0) + 1
        sorted_words = sorted(counts.items(), key=lambda x: -x[1])
        return [w for w, _ in sorted_words[:max_keywords]]

    def parse_objectives(self, text: str) -> List[str]:
        """Split bullet points or semicolon-separated text into distinct objectives."""
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        objectives = []
        for line in lines:
            cleaned = re.sub(r"^[\*\-\•\d+\.]\s*", "", line).strip()
            if cleaned:
                objectives.append(cleaned)
        return objectives

    def generate_investigation_document(self, investigation: Investigation) -> Dict[str, Any]:
        """Format an investigation into a standardized document representation."""
        full_text = investigation.to_search_text()
        return {
            "id": investigation.id,
            "title": investigation.title,
            "full_name": investigation.full_name,
            "text": full_text,
            "objectives": list(investigation.objectives),
            "categories": [c.value for c in investigation.categories],
            "platform": investigation.platform.value,
            "status": investigation.status.value,
            "keywords": list(investigation.keywords),
            "applications": list(investigation.applications),
            "doi": investigation.get_active_version().doi if investigation.get_active_version() else None,
        }
