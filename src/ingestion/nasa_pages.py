"""
NASA Science and PSI Page Scraper / Context Ingester.

Fetches and caches web documentation from NASA Glenn Research Center and PSI repository.
Gracefully operates in offline environments by returning local cached snapshots.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class NASAPagesScraper:
    """Ingests scientific context from NASA websites with automatic offline fallback."""

    def __init__(self, cache_dir: Path | str | None = None):
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir
            else Path(__file__).resolve().parents[2] / "data" / "raw" / "context"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_page_context(self, url: str) -> Dict[str, Any]:
        """Fetch page text, falling back to cached context if offline."""
        cache_key = "".join(c if c.isalnum() else "_" for c in url)[:64]
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        # Attempt network fetch if requests is available and network exists
        try:
            import requests
            response = requests.get(url, timeout=4)
            if response.status_code == 200:
                data = {
                    "url": url,
                    "status": "online",
                    "content_length": len(response.text),
                    "summary": response.text[:1000],
                }
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                return data
        except Exception as e:
            logger.debug("Network fetch failed for %s: %s (using offline mode)", url, e)

        return {
            "url": url,
            "status": "offline_fallback",
            "content_length": 0,
            "summary": "NASA combustion science experimental reference documentation.",
        }
