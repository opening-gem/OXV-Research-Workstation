"""
Unified Search Engine for Flame in Freefall.

Provides faceted query filtering, keyword/semantic hybrid retrieval,
and autocompletion across the NASA combustion research catalogue.
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

from src.catalog.compat_pydantic import BaseModel, Field
from src.catalog.repository import ResearchCatalog
from src.catalog.schema import Investigation, ResearchCategory, SpacecraftPlatform


class SearchFilters(BaseModel):
    """Filters applicable to search queries."""
    categories: Optional[List[str]] = None
    platforms: Optional[List[str]] = None
    status: Optional[List[str]] = None
    has_doi: Optional[bool] = None


class SearchResult(BaseModel):
    """A single search match."""
    investigation_id: str
    title: str
    full_name: str
    description: str
    score: float
    categories: List[str]
    platform: str
    status: str
    doi: Optional[str] = None
    highlights: List[str] = Field(default_factory=list)


class SearchResults(BaseModel):
    """Container for search results and faceted metrics."""
    query: str
    total_results: int
    results: List[SearchResult]
    facets: Dict[str, Dict[str, int]]
    search_time_ms: float


class UnifiedSearchEngine:
    """
    Unified multi-modal search engine combining metadata filtering,
    keyword index matching, and relevance ranking.
    """

    def __init__(self, catalog: Optional[ResearchCatalog] = None):
        self.catalog = catalog or ResearchCatalog()

    def search(
        self,
        query: str = "",
        filters: Optional[SearchFilters] = None,
        sort_by: str = "relevance",
        top_k: int = 20,
    ) -> SearchResults:
        """Execute a faceted search query against the catalogue."""
        start_time = time.perf_counter()
        investigations = self.catalog.list_all()

        query_terms = [t.lower() for t in re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", query)]

        scored_results: List[tuple[float, Investigation, List[str]]] = []

        for inv in investigations:
            # 1. Apply strict filters
            if filters:
                if filters.categories:
                    inv_cats = {c.value for c in inv.categories}
                    if not any(fc in inv_cats for fc in filters.categories):
                        continue
                if filters.platforms:
                    if inv.platform.value not in filters.platforms:
                        continue
                if filters.status:
                    if inv.status.value not in filters.status:
                        continue
                if filters.has_doi is not None:
                    active_ver = inv.get_active_version()
                    has_d = bool(active_ver and active_ver.doi)
                    if has_d != filters.has_doi:
                        continue

            # 2. Compute search score
            search_text = inv.to_search_text().lower()
            score = 0.0
            highlights = []

            if not query_terms:
                score = 1.0  # Browsing without query
            else:
                for term in query_terms:
                    # Title match
                    if term in inv.title.lower():
                        score += 3.0
                        highlights.append(f"Title match: '{term}'")
                    # ID match
                    if term in inv.id.lower():
                        score += 5.0
                        highlights.append(f"Identifier match: '{inv.id}'")
                    # Category match
                    for cat in inv.categories:
                        if term in cat.value.lower():
                            score += 2.5
                    # Objectives match
                    for obj in inv.objectives:
                        if term in obj.lower():
                            score += 1.5
                            if len(highlights) < 3:
                                highlights.append(f"Objective: ...{obj[:70]}...")
                    # General description match
                    if term in inv.description.lower():
                        score += 1.0

            if query_terms and score <= 0.0:
                continue

            scored_results.append((score, inv, highlights))

        # Sort
        if sort_by == "relevance":
            scored_results.sort(key=lambda x: x[0], reverse=True)
        elif sort_by == "id":
            scored_results.sort(key=lambda x: x[1].id)
        elif sort_by == "title":
            scored_results.sort(key=lambda x: x[1].title)

        top_slice = scored_results[:top_k]
        results_list = []
        for s, inv, hl in top_slice:
            active_ver = inv.get_active_version()
            results_list.append(
                SearchResult(
                    investigation_id=inv.id,
                    title=inv.title,
                    full_name=inv.full_name,
                    description=inv.description,
                    score=round(s, 2),
                    categories=[c.value for c in inv.categories],
                    platform=inv.platform.value,
                    status=inv.status.value,
                    doi=active_ver.doi if active_ver else None,
                    highlights=hl,
                )
            )

        facets = self.get_facets()
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return SearchResults(
            query=query,
            total_results=len(scored_results),
            results=results_list,
            facets=facets,
            search_time_ms=round(elapsed_ms, 2),
        )

    def autocomplete(self, partial: str, limit: int = 5) -> List[str]:
        """Suggest completions based on investigation IDs, titles, and categories."""
        p = partial.strip().lower()
        if not p:
            return []
        suggestions = set()
        for inv in self.catalog.list_all():
            if p in inv.id.lower():
                suggestions.add(inv.id)
            if p in inv.title.lower():
                suggestions.add(inv.title)
            for c in inv.categories:
                if p in c.value.lower():
                    suggestions.add(c.value.replace("_", " "))
            for kw in inv.keywords:
                if p in kw.lower():
                    suggestions.add(kw)
        return sorted(list(suggestions))[:limit]

    def get_facets(self) -> Dict[str, Dict[str, int]]:
        """Calculate distribution facets for filtering."""
        category_counts: Dict[str, int] = {}
        platform_counts: Dict[str, int] = {}
        status_counts: Dict[str, int] = {}

        for inv in self.catalog.list_all():
            for c in inv.categories:
                category_counts[c.value] = category_counts.get(c.value, 0) + 1
            p = inv.platform.value
            platform_counts[p] = platform_counts.get(p, 0) + 1
            s = inv.status.value
            status_counts[s] = status_counts.get(s, 0) + 1

        return {
            "categories": category_counts,
            "platforms": platform_counts,
            "status": status_counts,
        }
