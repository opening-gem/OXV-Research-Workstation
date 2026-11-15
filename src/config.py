"""
Project-wide configuration for Flame in Freefall.

Centralises paths, API endpoints, model parameters, and dashboard settings.
Configuration is defined via plain ``dataclasses`` so that no external
dependency (e.g. ``pydantic-settings``) is required at import time.

The canonical way to access configuration::

    from src.config import ProjectConfig

    cfg = ProjectConfig()
    print(cfg.catalog_path)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _project_root() -> Path:
    """Return the repository root directory."""
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / "requirements.txt").exists():
            return parent
    return current.parent


@dataclass(frozen=True)
class PathConfig:
    """Repository filesystem paths."""

    root: Path = field(default_factory=_project_root)

    @property
    def data_dir(self) -> Path:
        env = os.environ.get("FIF_DATA_DIR")
        if env:
            return Path(env)
        return self.root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def embeddings_dir(self) -> Path:
        return self.data_dir / "embeddings"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def raw_psi_dir(self) -> Path:
        return self.raw_dir / "psi"

    @property
    def manifests_dir(self) -> Path:
        return self.data_dir / "manifests"

    @property
    def catalog_file(self) -> Path:
        return self.processed_dir / "catalog.json"


@dataclass(frozen=True)
class APIConfig:
    """External API endpoints."""

    nasa_psi_base_url: str = "https://psi.nasa.gov"
    nasa_psi_search_url: str = "https://psi.nasa.gov/search"
    nasa_science_base_url: str = "https://science.nasa.gov"
    nasa_techport_base_url: str = "https://techport.nasa.gov/api"
    request_timeout_seconds: int = 30
    request_retries: int = 3
    rate_limit_delay_seconds: float = 1.0


@dataclass(frozen=True)
class ModelConfig:
    """Machine-learning and embedding parameters."""

    # Sentence-transformer model for RAG embeddings.
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # Classification
    classifier_min_confidence: float = 0.3
    classifier_max_labels: int = 3

    # FAISS
    faiss_index_type: str = "Flat"  # "Flat", "IVF", "HNSW"
    faiss_nprobe: int = 10
    faiss_top_k: int = 10


@dataclass(frozen=True)
class DashboardConfig:
    """Streamlit dashboard settings."""

    page_title: str = "Orbital Experimentation Volumetric (OXV)"
    page_icon: str = "assets/oxv_logo_cropped.png"
    layout: str = "wide"
    sidebar_state: str = "expanded"
    default_page_size: int = 25



@dataclass(frozen=True)
class LLMConfig:
    """Runtime configuration for LLM connectivity."""

    provider: str = "ollama"
    base_url: str = "http://127.0.0.1:11434"
    model: str = "llama3"
    timeout_seconds: int = 120
    temperature: float = 0.2
    auth_header: str | None = None

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            provider=os.environ.get("LLM_PROVIDER", "ollama").strip().lower(),
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/"),
            model=os.environ.get("OLLAMA_MODEL", "llama3").strip(),
            timeout_seconds=int(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "120")),
            temperature=float(os.environ.get("OLLAMA_TEMPERATURE", "0.2")),
            auth_header=os.environ.get("OLLAMA_AUTH_HEADER"),
        )

@dataclass
class ProjectConfig:
    """Top-level configuration aggregating all sub-configs.

    Environment variable overrides:

    - ``FIF_DATA_DIR`` — override the default data directory.
    - ``FIF_CATALOG_PATH`` — override the default catalogue JSON path.
    """

    paths: PathConfig = field(default_factory=PathConfig)
    api: APIConfig = field(default_factory=APIConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    llm: LLMConfig = field(default_factory=LLMConfig.from_env)

    @property
    def catalog_path(self) -> Path:
        """Resolved path to the catalogue JSON file, respecting env vars."""
        env = os.environ.get("FIF_CATALOG_PATH")
        if env:
            return Path(env)
        return self.paths.catalog_file

    @property
    def raw_psi_dir(self) -> Path:
        return self.paths.raw_psi_dir

    @property
    def manifests_dir(self) -> Path:
        return self.paths.manifests_dir

    @property
    def processed_dir(self) -> Path:
        return self.paths.processed_dir

    @property
    def models_dir(self) -> Path:
        return self.paths.models_dir

    @property
    def data_dir(self) -> Path:
        """Resolved data directory, respecting env vars."""
        env = os.environ.get("FIF_DATA_DIR")
        if env:
            return Path(env)
        return self.paths.data_dir

    def ensure_directories(self) -> None:
        """Create all required data directories if they do not exist."""
        for d in (
            self.paths.raw_dir,
            self.paths.processed_dir,
            self.paths.embeddings_dir,
            self.paths.models_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
