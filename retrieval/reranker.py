from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List

from langchain_core.documents import Document


@dataclass
class RerankedResult:
    document: Document
    score: float
    semantic_score: float
    keyword_score: float


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]{2,}", (text or "").lower())


def rerank_results(
    query: str,
    candidates: Iterable[tuple[Document, float, float]],
    top_k: int = 4,
) -> List[RerankedResult]:
    """Rerank hybrid candidates using overlap and readability heuristics."""
    query_tokens = set(_tokenize(query))
    if not query_tokens:
        query_tokens = {"study"}

    ranked: List[RerankedResult] = []
    for document, semantic_score, keyword_score in candidates:
        content = document.page_content.strip()
        doc_tokens = set(_tokenize(content))

        overlap = len(query_tokens & doc_tokens) / max(len(query_tokens), 1)
        length_score = min(len(content) / 400.0, 1.0)

        # Weighted score to favor semantic relevance while promoting keyword grounding.
        final_score = (
            0.55 * semantic_score
            + 0.25 * keyword_score
            + 0.15 * overlap
            + 0.05 * length_score
        )

        ranked.append(
            RerankedResult(
                document=document,
                score=final_score,
                semantic_score=semantic_score,
                keyword_score=keyword_score,
            )
        )

    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked[:top_k]
