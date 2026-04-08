from __future__ import annotations

from typing import List, Sequence, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


def split_documents(
    documents: Sequence[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Document]:
    """Split source docs into semantically useful chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(list(documents))


def build_or_update_vector_store(
    documents: Sequence[Document],
    embeddings: Embeddings,
    existing_store: FAISS | None = None,
) -> tuple[FAISS, int]:
    """Create a FAISS store or append new chunks to an existing one."""
    chunks = split_documents(documents)
    if not chunks:
        raise ValueError("No valid text chunks were produced from the uploaded files.")

    if existing_store is None:
        store = FAISS.from_documents(chunks, embeddings)
    else:
        existing_store.add_documents(chunks)
        store = existing_store

    return store, len(chunks)


def retrieve_with_scores(
    store: FAISS,
    query: str,
    k: int = 4,
) -> List[Tuple[Document, float]]:
    """Return top-k similar chunks and raw FAISS distance scores."""
    return store.similarity_search_with_score(query, k=k)


def score_to_confidence(score: float) -> float:
    """Convert FAISS distance to an intuitive confidence value in [0, 1]."""
    return 1.0 / (1.0 + max(score, 0.0))
