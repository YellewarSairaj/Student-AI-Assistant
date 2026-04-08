from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List

from langchain_core.documents import Document
from pypdf import PdfReader


@dataclass(frozen=True)
class ProcessedPdf:
    """Container for extracted PDF text and per-page documents."""

    source_name: str
    full_text: str
    documents: List[Document]


def extract_pdf_documents(file_bytes: bytes, file_name: str) -> ProcessedPdf:
    """Extract page-level text documents from a single PDF byte stream."""
    reader = PdfReader(BytesIO(file_bytes))
    page_docs: List[Document] = []
    page_texts: List[str] = []

    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        cleaned_text = text.strip()
        if not cleaned_text:
            continue

        page_texts.append(cleaned_text)
        page_docs.append(
            Document(
                page_content=cleaned_text,
                metadata={
                    "source": file_name,
                    "page": page_index,
                },
            )
        )

    return ProcessedPdf(
        source_name=file_name,
        full_text="\n\n".join(page_texts),
        documents=page_docs,
    )


def merge_text_blobs(text_by_hash: Dict[str, str]) -> str:
    """Join all extracted text blocks for summarize/quiz prompts."""
    return "\n\n".join(text for text in text_by_hash.values() if text.strip())
