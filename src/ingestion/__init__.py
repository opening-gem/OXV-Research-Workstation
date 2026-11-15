"""
Ingestion pipeline for NASA combustion research data.

This module handles scraping, fetching, and parsing data from:
- NASA Physical Sciences Informatics (PSI) system
- NASA Science investigation pages
- Published datasets and DOI-linked resources

Key components (planned):
- PSI scraper: Extracts investigation metadata from PSI search results.
- Science page parser: Parses individual investigation pages.
- DOI resolver: Resolves DOIs to dataset landing pages.
- Rate-limited HTTP client: Respects NASA API rate limits.
"""
