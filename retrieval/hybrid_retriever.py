from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import List, Sequence

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi

from retrieval.reranker import RerankedResult, rerank_results


@dataclass
class RetrievedChunk:
    document: Document
    score: float
    semantic_score: float
    keyword_score: float


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]{2,}", (text or "").lower())


def semantic_chunk_documents(
    documents: Sequence[Document],
    target_chars: int = 900,
    overlap_chars: int = 120,
) -> List[Document]:
    """Chunk documents by sentence boundaries to preserve meaning."""
    chunks: List[Document] = []

    for doc in documents:
        text = re.sub(r"\s+", " ", doc.page_content).strip()
        if not text:
            continue

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not sentences:
            continue

        current: List[str] = []
        current_len = 0

        for sentence in sentences:
            sentence_len = len(sentence)
            if current and current_len + sentence_len > target_chars:
                chunk_text = " ".join(current).strip()
                chunks.append(
                    Document(
                        page_content=chunk_text,
                        metadata={
                            **doc.metadata,
                            "chunk_hash": hashlib.sha1(chunk_text.encode("utf-8")).hexdigest()[:16],
                        },
                    )
                )

                # Overlap to preserve continuity across chunks.
                overlap_text = chunk_text[-overlap_chars:] if overlap_chars > 0 else ""
                current = [overlap_text, sentence] if overlap_text else [sentence]
                current_len = len(" ".join(current))
            else:
                current.append(sentence)
                current_len += sentence_len

        if current:
            chunk_text = " ".join(current).strip()
            chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        **doc.metadata,
                        "chunk_hash": hashlib.sha1(chunk_text.encode("utf-8")).hexdigest()[:16],
                    },
                )
            )

    return chunks


class HybridRetriever:
    """Hybrid retrieval using FAISS semantic search + BM25 keyword search."""

    def __init__(self, embeddings: Embeddings) -> None:
        self.embeddings = embeddings
        self.vector_store: FAISS | None = None
        self.chunks: List[Document] = []
        self._tokenized_chunks: List[List[str]] = []
        self._bm25: BM25Okapi | None = None

    @property
    def has_index(self) -> bool:
        return bool(self.chunks) and self.vector_store is not None and self._bm25 is not None

    def _rebuild_keyword_index(self) -> None:
        if not self.chunks:
            self._tokenized_chunks = []
            self._bm25 = None
            return
        self._tokenized_chunks = [_tokenize(doc.page_content) for doc in self.chunks]
        self._bm25 = BM25Okapi(self._tokenized_chunks)

    def add_documents(self, documents: Sequence[Document]) -> int:
        chunks = semantic_chunk_documents(documents)
        if not chunks:
            return 0

        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vector_store.add_documents(chunks)

        self.chunks.extend(chunks)
        self._rebuild_keyword_index()
        return len(chunks)

    def save_to_disk(self, folder_path: str) -> None:
        if self.vector_store is None:
            return
        os.makedirs(folder_path, exist_ok=True)
        self.vector_store.save_local(folder_path)

    def load_from_disk(self, folder_path: str) -> bool:
        index_file = os.path.join(folder_path, "index.faiss")
        if not os.path.exists(index_file):
            return False

        self.vector_store = FAISS.load_local(
            folder_path,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

        # Reconstruct in-memory chunk list from FAISS docstore.
        documents = list(getattr(self.vector_store.docstore, "_dict", {}).values())
        self.chunks = [doc for doc in documents if isinstance(doc, Document)]
        self._rebuild_keyword_index()
        return self.has_index

    def get_all_documents(self) -> List[Document]:
        return list(self.chunks)

    def retrieve(self, query: str, k: int = 4) -> List[RetrievedChunk]:
        if not self.has_index or self.vector_store is None or self._bm25 is None:
            return []

        semantic_hits = self.vector_store.similarity_search_with_score(query, k=max(k * 2, 8))
        query_tokens = _tokenize(query)
        bm25_scores = self._bm25.get_scores(query_tokens or ["study"])

        keyword_indices = sorted(
            range(len(bm25_scores)), key=lambda idx: bm25_scores[idx], reverse=True
        )[: max(k * 2, 8)]

        merged: dict[str, tuple[Document, float, float]] = {}

        for doc, distance in semantic_hits:
            key = str(doc.metadata.get("chunk_hash", hash(doc.page_content)))
            semantic_score = 1.0 / (1.0 + max(float(distance), 0.0))
            current = merged.get(key)
            if current is None:
                merged[key] = (doc, semantic_score, 0.0)
            else:
                merged[key] = (current[0], max(current[1], semantic_score), current[2])

        max_bm25 = max(float(s) for s in bm25_scores) if len(bm25_scores) else 1.0
        max_bm25 = max(max_bm25, 1.0)

        for idx in keyword_indices:
            doc = self.chunks[idx]
            key = str(doc.metadata.get("chunk_hash", hash(doc.page_content)))
            keyword_score = float(bm25_scores[idx]) / max_bm25
            current = merged.get(key)
            if current is None:
                merged[key] = (doc, 0.0, keyword_score)
            else:
                merged[key] = (current[0], current[1], max(current[2], keyword_score))

        candidates = list(merged.values())

        # Context filtering to reduce noisy chunks.
        filtered: List[tuple[Document, float, float]] = []
        for doc, semantic_score, keyword_score in candidates:
            content = doc.page_content.strip()
            if len(content) < 60:
                continue
            if semantic_score <= 0 and keyword_score <= 0:
                continue
            filtered.append((doc, semantic_score, keyword_score))

        reranked: List[RerankedResult] = rerank_results(query, filtered, top_k=k)

        return [
            RetrievedChunk(
                document=item.document,
                score=item.score,
                semantic_score=item.semantic_score,
                keyword_score=item.keyword_score,
            )
            for item in reranked
        ]
