from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return (vecs / norms).astype(np.float32)


class Embedder:
    def embed(self, texts: List[str]) -> np.ndarray:  # (n, d)
        raise NotImplementedError


class OpenAIEmbedder(Embedder):
    def __init__(self, api_key: str, model: str = "text-embedding-3-large"):
        import openai

        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        out = self.client.embeddings.create(model=self.model, input=texts)
        vecs = np.array([d.embedding for d in out.data], dtype=np.float32)
        return _l2_normalize(vecs)


class LocalEmbedder(Embedder):
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Lazy import to avoid heavy import at module import time
        from sentence_transformers import SentenceTransformer

        self.model_name = model
        self.model = SentenceTransformer(model)

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return np.array(vecs, dtype=np.float32)


def get_embedder(
    backend: str,
    openai_key: Optional[str] = None,
    openai_model: str = "text-embedding-3-large",
    local_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Embedder:
    if backend == "openai":
        if not openai_key:
            raise ValueError("OpenAI key required for openai embed backend")
        return OpenAIEmbedder(api_key=openai_key, model=openai_model)
    return LocalEmbedder(model=local_model)

