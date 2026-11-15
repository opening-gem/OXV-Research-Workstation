"""
Telemetry & Scientific Graph Studio Manager for Flame in Freefall.
Connects the BASS-II and FLEX flight test datasets to the mission dashboard.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("telemetry-studio")

ROOT_DIR = Path(__file__).resolve().parents[2]
WORKER_SCRIPT = ROOT_DIR / "src" / "visualisation" / "render_worker.py"
VENV_PYTHON = ROOT_DIR / ".venv" / "bin" / "python"

_CACHE: Dict[str, Any] = {}


class TelemetryStudio:
    """Manages telemetry datasets, scientific graph generation, and table summaries."""

    DATASETS = ["BASS-II", "FLEX"]

    @classmethod
    def get_available_datasets(cls) -> List[str]:
        return cls.DATASETS

    @classmethod
    def get_telemetry_bundle(
        cls,
        dataset: str = "BASS-II",
        plot: str = "all",
        bins: int = 8,
        min_samples: int = 2,
        method: str = "uniform",
    ) -> Dict[str, Any]:
        """Fetch telemetry bundle with base64 graphs and numerical tables. Results are cached."""
        cache_key = f"{dataset}_{plot}_{bins}_{min_samples}_{method}"
        if cache_key in _CACHE:
            return _CACHE[cache_key]

        # Check if pandas/matplotlib can be imported in current process
        try:
            import matplotlib  # noqa
            import pandas  # noqa
            from src.visualisation.render_worker import load_dataset_bundle
            bundle = load_dataset_bundle(
                dataset_name=dataset,
                n_bins=bins,
                min_samples=min_samples,
                method=method,
                requested_plot=plot,
            )
            _CACHE[cache_key] = bundle
            return bundle
        except ImportError:
            # Fall back to flame_project venv python
            pass

        # Subprocess execution using venv python
        py_bin = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable
        env = dict(os.environ)
        env["MPLCONFIGDIR"] = "/tmp"

        cmd = [
            py_bin,
            str(WORKER_SCRIPT),
            "--dataset", dataset,
            "--plot", plot,
            "--bins", str(bins),
            "--min_samples", str(min_samples),
            "--method", method,
        ]

        logger.info("Executing telemetry worker: %s", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            logger.error("Telemetry worker failed: %s", proc.stderr)
            raise RuntimeError(f"Telemetry worker error: {proc.stderr}")

        try:
            bundle = json.loads(proc.stdout)
            _CACHE[cache_key] = bundle
            return bundle
        except json.JSONDecodeError as e:
            logger.error("Failed to decode telemetry JSON: %s (Output: %s)", e, proc.stdout[:200])
            raise RuntimeError(f"Failed to decode telemetry output: {e}")

    @classmethod
    def warmup_cache(cls):
        """Warm up cache for fast initial UI rendering."""
        try:
            cls.get_telemetry_bundle(dataset="BASS-II", plot="all", bins=8, min_samples=2, method="uniform")
            logger.info("Telemetry cache warmed up for BASS-II.")
        except Exception as e:
            logger.warning("Telemetry cache warmup failed (will compute on demand): %s", e)
