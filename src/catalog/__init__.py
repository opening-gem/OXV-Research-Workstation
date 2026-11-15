"""
Research catalogue — schemas, repository, and normalisation.

This sub-package is the backbone of the Flame in Freefall platform.  It defines
the canonical Pydantic v2 data models for NASA combustion investigations,
provides a JSON-backed CRUD repository with file-level locking and version
tracking, and includes normalisation utilities for cleaning ingested metadata.

Public API:
    - ``Investigation`` — central data model
    - ``ResearchCategory`` — topic taxonomy enum
    - ``ResearchCatalog`` — persistence layer
    - ``InvestigationNormalizer`` — metadata cleaner
"""

from src.catalog.schema import (
    Investigation,
    InvestigationVersion,
    Investigator,
    InvestigationStatus,
    ResearchCategory,
    SpacecraftPlatform,
)
from src.catalog.repository import ResearchCatalog
from src.catalog.normalizer import InvestigationNormalizer

__all__ = [
    "Investigation",
    "InvestigationVersion",
    "Investigator",
    "InvestigationStatus",
    "ResearchCategory",
    "SpacecraftPlatform",
    "ResearchCatalog",
    "InvestigationNormalizer",
]
