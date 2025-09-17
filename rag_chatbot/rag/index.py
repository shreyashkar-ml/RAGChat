from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

import numpy as np

try:  # optional dependency – falls back to numpy search if unavailable
    import chromadb  # type: ignore
except Exception:  # pragma: no cover - chromadb optional
    chromadb = None


class VectorIndex:
    """Vector index backed by Chroma (local vector DB) with FAISS/numpy fallback."""

    def __init__(self, embedder, persist_path: Optional[str | Path] = None, batch_size: int = 64):
        self.embedder = embedder
        self.docs: List[Dict] = []
        self.vecs: Optional[np.ndarray] = None
        self._batch_size = max(1, batch_size)

        self._use_chroma = False
        self._collection = None
        self._chroma_client = None
        self._collection_name = None

        if chromadb is not None:
            try:
                path = Path(persist_path or Path.cwd() / ".rag_chroma").resolve()
                path.mkdir(parents=True, exist_ok=True)
                self._chroma_client = chromadb.PersistentClient(path=str(path))  # type: ignore[attr-defined]
                self._collection_name = f"rag_{uuid4().hex}"
                self._collection = self._chroma_client.get_or_create_collection(
                    name=self._collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
                self._use_chroma = True
            except Exception:
                self._chroma_client = None
                self._collection = None
                self._use_chroma = False

        self._faiss = None
        self._index = None
        if not self._use_chroma:
            try:  # optional acceleration when Chroma isn't available
                import faiss  # type: ignore

                self._faiss = faiss
            except Exception:
                self._faiss = None

    # Public API ---------------------------------------------------------
    def add_documents(self, docs: List[Dict]):
        if not docs:
            return
        if self._use_chroma:
            self._add_documents_chroma(docs)
        else:
            self._add_documents_numpy(docs)

    def search(self, query: str, top_k: int = 6) -> List[Dict]:
        if not self.docs:
            return []
        if self._use_chroma:
            return self._search_chroma(query, top_k)
        return self._search_numpy(query, top_k)

    def dump(self):
        """Return (docs, vecs) for persistence."""
        return self.docs, (self.vecs if self.vecs is not None else None)

    def load(self, docs: List[Dict], vecs: Optional[np.ndarray]):
        """Load docs and vectors, rehydrating the underlying search index."""
        self.docs = []
        self.vecs = None
        if self._use_chroma:
            self._reset_collection()
            if vecs is not None:
                self._add_documents_chroma(docs, embeddings=vecs)
            else:
                self._add_documents_chroma(docs)
        else:
            self._index = None
            if vecs is not None:
                self.vecs = vecs.astype("float32")
                self.docs = [dict(d) for d in docs]
                if self._faiss is not None:
                    dim = self.vecs.shape[1]
                    self._index = self._faiss.IndexFlatIP(dim)
                    self._index.add(self.vecs)
            else:
                self._add_documents_numpy(docs)

    # Internal helpers ---------------------------------------------------
    def _reset_collection(self):
        if not self._collection:
            return
        existing = self._collection.get()
        ids = existing.get("ids", [])
        if ids:
            self._collection.delete(ids=ids)

    def _ensure_doc_id(self, doc: Dict, fallback: int) -> str:
        doc_id = doc.get("id")
        if not doc_id:
            doc_id = f"doc-{len(self.docs) + fallback}"
        return str(doc_id)

    def _metadata_from_doc(self, doc: Dict, doc_id: str) -> Dict:
        meta = {k: v for k, v in doc.items() if k not in {"text", "score", "images"}}
        meta["doc_id"] = doc_id
        meta.setdefault("title", doc.get("title") or doc.get("url") or doc_id)
        return meta

    def _add_documents_chroma(self, docs: List[Dict], embeddings: Optional[np.ndarray] = None):
        if not self._collection:
            raise RuntimeError("Chroma collection not initialised")
        if embeddings is not None and len(embeddings) != len(docs):
            raise ValueError("Embeddings count must match docs count")

        new_vecs: List[np.ndarray] = []
        for offset in range(0, len(docs), self._batch_size):
            chunk = docs[offset : offset + self._batch_size]
            texts = [d["text"] for d in chunk]
            if embeddings is None:
                if self.embedder is None:
                    raise ValueError("Embedder required to build index")
                chunk_vecs = self.embedder.embed(texts).astype(np.float32)
            else:
                chunk_vecs = embeddings[offset : offset + len(chunk)].astype(np.float32)
            ids = [self._ensure_doc_id(d, offset + i) for i, d in enumerate(chunk)]
            metas = [self._metadata_from_doc(d, doc_id) for d, doc_id in zip(chunk, ids)]
            self._collection.add(
                ids=ids,
                embeddings=chunk_vecs.tolist(),
                documents=texts,
                metadatas=metas,
            )
            new_vecs.append(chunk_vecs)

        if new_vecs:
            combined = np.vstack(new_vecs)
            self.vecs = combined if self.vecs is None else np.vstack([self.vecs, combined])
        self.docs.extend([dict(d) for d in docs])

    def _add_documents_numpy(self, docs: List[Dict]):
        if self.embedder is None:
            raise ValueError("Embedder required to build index")
        new_vecs: List[np.ndarray] = []
        texts = [d["text"] for d in docs]
        for offset in range(0, len(texts), self._batch_size):
            chunk_texts = texts[offset : offset + self._batch_size]
            chunk_vecs = self.embedder.embed(chunk_texts).astype(np.float32)
            new_vecs.append(chunk_vecs)
            if self._faiss and self._index is not None:
                self._index.add(chunk_vecs)

        if new_vecs:
            vecs = np.vstack(new_vecs)
            self.vecs = vecs if self.vecs is None else np.vstack([self.vecs, vecs])
            if self._faiss and self._index is None and self.vecs is not None:
                dim = self.vecs.shape[1]
                self._index = self._faiss.IndexFlatIP(dim)
                self._index.add(self.vecs)

        self.docs.extend([dict(d) for d in docs])

    def _search_chroma(self, query: str, top_k: int) -> List[Dict]:
        if self.embedder is None:
            raise ValueError("Embedder required for querying")
        if not self._collection:
            return []
        q = self.embedder.embed([query]).astype(np.float32)
        limit = min(top_k, len(self.docs))
        res = self._collection.query(
            query_embeddings=[q[0].tolist()],
            n_results=limit,
            include=["documents", "metadatas", "distances", "ids"],
        )
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        ids = res.get("ids", [[]])[0]
        out: List[Dict] = []
        for idx, doc_text in enumerate(docs):
            meta = metas[idx] if idx < len(metas) else {}
            dist = dists[idx] if idx < len(dists) else None
            doc_id = meta.get("doc_id") or (ids[idx] if idx < len(ids) else str(idx))
            record = {
                "id": doc_id,
                "title": meta.get("title") or meta.get("url") or doc_id,
                "url": meta.get("url"),
                "text": doc_text,
                "score": float(1.0 - dist) if dist is not None else 0.0,
            }
            out.append(record)
        return out

    def _search_numpy(self, query: str, top_k: int) -> List[Dict]:
        if self.vecs is None or not len(self.docs):
            return []
        q = self.embedder.embed([query]).astype(np.float32)
        if self._faiss and self._index is not None:
            D, I = self._index.search(q, min(top_k, len(self.docs)))
            idxs = I[0].tolist()
            sims = D[0].tolist()
        else:
            sims = (self.vecs @ q[0]).tolist()
            idxs = np.argsort(sims)[::-1][:top_k].tolist()
        out: List[Dict] = []
        for rank, i in enumerate(idxs):
            d = dict(self.docs[i])
            d["score"] = float(sims[rank]) if sims else 0.0
            out.append(d)
        return out
