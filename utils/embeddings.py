from __future__ import annotations

import hashlib
import os
import re
from math import sqrt
from typing import List

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings


def get_provider() -> str:
    """Read model provider from environment."""
    return os.getenv("LLM_PROVIDER", "openai").strip().lower()


class LocalHashEmbeddings(Embeddings):
    """Deterministic local embeddings that work without external services."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def _embed_text(self, text: str) -> List[float]:
        tokens = re.findall(r"\w+", (text or "").lower())
        vector = [0.0] * self.dimensions
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if (digest[4] % 2 == 0) else -1.0
            vector[idx] += sign

        norm = sqrt(sum(v * v for v in vector))
        if norm == 0:
            return vector
        return [v / norm for v in vector]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)


def get_embedding_model(provider: str) -> Embeddings:
    """Return an embedding model for the selected provider."""
    if provider == "offline":
        return LocalHashEmbeddings()

    if provider == "ollama":
        model_name = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return OllamaEmbeddings(model=model_name, base_url=base_url)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is missing. Add it to .env, or set LLM_PROVIDER=ollama for local models."
        )

    model_name = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    return OpenAIEmbeddings(model=model_name, api_key=api_key)
