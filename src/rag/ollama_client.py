"""
Reusable Ollama LLM HTTP Client for Flame in Freefall.

Communicates with local or remote (e.g. Google Colab via tunnel) Ollama instances
using standard HTTP endpoints:
  - GET  /api/tags  (health check and model inventory)
  - POST /api/chat  (multi-turn chat completion)

Configuration is entirely driven by environment variables or an explicit LLMConfig.
Credentials and tunnel tokens are passed via headers and never hardcoded or logged.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMConfig:
    """Runtime configuration for LLM connectivity."""

    provider: str = "ollama"
    base_url: str = "http://127.0.0.1:11434"
    model: str = "llama3"
    timeout_seconds: int = 120
    temperature: float = 0.2
    auth_header: Optional[str] = None  # e.g., "Bearer <token>" for tunnels

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Instantiate configuration from environment variables."""
        return cls(
            provider=os.environ.get("LLM_PROVIDER", "ollama").strip().lower(),
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/"),
            model=os.environ.get("OLLAMA_MODEL", "llama3").strip(),
            timeout_seconds=int(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "120")),
            temperature=float(os.environ.get("OLLAMA_TEMPERATURE", "0.2")),
            auth_header=os.environ.get("OLLAMA_AUTH_HEADER"),
        )


class OllamaClientError(Exception):
    """Base exception for Ollama client failures."""
    pass


class OllamaConnectionError(OllamaClientError):
    """Raised when Ollama server cannot be reached."""
    pass


class OllamaModelNotFoundError(OllamaClientError):
    """Raised when the specified model is not pulled or available on the server."""
    pass


class OllamaTimeoutError(OllamaClientError):
    """Raised when a request exceeds the configured timeout."""
    pass


class OllamaClient:
    """
    Client for interacting with the Ollama HTTP API.
    Designed to work identically whether Ollama is running on localhost,
    within the same Google Colab container, or across an authenticated tunnel.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig.from_env()

    def _build_request(self, endpoint: str, data: Optional[bytes] = None, method: str = "GET") -> urllib.request.Request:
        url = f"{self.config.base_url}{endpoint}"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
        }
        if self.config.auth_header:
            headers["Authorization"] = self.config.auth_header

        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        return req

    def check_health(self) -> Dict[str, Any]:
        """
        Query GET /api/tags to verify server health and inspect available models.
        Returns a dict with 'status', 'available_models', 'model_found', and 'base_url'.
        """
        try:
            req = self._build_request("/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=min(self.config.timeout_seconds, 10)) as resp:
                if resp.status == 200:
                    body = json.loads(resp.read().decode("utf-8"))
                    models = [m.get("name", "") for m in body.get("models", [])]
                    # Also strip tag for loose matching (e.g., 'llama3:latest' matches 'llama3')
                    model_base_names = [m.split(":")[0] for m in models]
                    target_model = self.config.model
                    target_base = target_model.split(":")[0]
                    found = (target_model in models) or (target_base in model_base_names)
                    return {
                        "status": "online",
                        "base_url": self.config.base_url,
                        "configured_model": target_model,
                        "model_available": found,
                        "available_models": models,
                    }
                return {
                    "status": "error",
                    "http_status": resp.status,
                    "base_url": self.config.base_url,
                    "model_available": False,
                    "available_models": [],
                }
        except urllib.error.HTTPError as e:
            return {
                "status": "http_error",
                "code": e.code,
                "reason": str(e),
                "base_url": self.config.base_url,
                "model_available": False,
                "available_models": [],
            }
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            return {
                "status": "offline",
                "reason": str(e),
                "base_url": self.config.base_url,
                "model_available": False,
                "available_models": [],
            }

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send a multi-turn chat request to POST /api/chat.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Optional override for generation temperature.
            model: Optional override for model name.

        Returns:
            Dict containing 'message': {'role': 'assistant', 'content': '...'},
            'model', 'total_duration', and 'done'.
        """
        target_model = model or self.config.model
        temp = temperature if temperature is not None else self.config.temperature

        payload = {
            "model": target_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temp,
            },
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        req = self._build_request("/api/chat", data=data_bytes, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                if resp.status == 200:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    message = resp_data.get("message", {})
                    content = message.get("content", "").strip()
                    if not content:
                        logger.warning("Ollama returned an empty response content.")
                    return {
                        "message": message,
                        "content": content,
                        "model": resp_data.get("model", target_model),
                        "done": resp_data.get("done", True),
                        "total_duration": resp_data.get("total_duration"),
                    }
                raise OllamaClientError(f"Ollama returned HTTP status {resp.status}")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise OllamaModelNotFoundError(
                    f"Model '{target_model}' not found on Ollama server at {self.config.base_url}. "
                    f"Ensure you have run `ollama pull {target_model}`."
                ) from e
            raise OllamaClientError(f"HTTP {e.code}: {e.reason}") from e
        except TimeoutError as e:
            raise OllamaTimeoutError(
                f"Request to Ollama timed out after {self.config.timeout_seconds}s at {self.config.base_url}."
            ) from e
        except urllib.error.URLError as e:
            if isinstance(e.reason, TimeoutError):
                raise OllamaTimeoutError(
                    f"Request to Ollama timed out after {self.config.timeout_seconds}s."
                ) from e
            raise OllamaConnectionError(
                f"Could not connect to Ollama at {self.config.base_url}: {e.reason}."
            ) from e
        except Exception as e:
            raise OllamaClientError(f"Unexpected error communicating with Ollama: {e}") from e
