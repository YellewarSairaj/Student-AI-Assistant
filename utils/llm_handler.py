from __future__ import annotations

from collections import Counter
import os
import re
from typing import Iterable, List, Mapping, Optional

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


def get_chat_model(provider: str, temperature: float = 0.2) -> BaseChatModel:
    """Return a chat model instance for OpenAI or Ollama."""
    if provider == "ollama":
        model_name = os.getenv("OLLAMA_MODEL", "llama3.1")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=model_name, base_url=base_url, temperature=temperature)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is missing. Add it to .env, or set LLM_PROVIDER=ollama for local models."
        )

    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model_name, api_key=api_key, temperature=temperature)


def _fast_extractive_summary(text: str, bullet_count: int = 6) -> str:
    """Create a lightweight local summary without calling an LLM."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return "- No content available to summarize."

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
    if not sentences:
        return f"- {cleaned[:180]}"

    stop_words = {
        "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "is",
        "are", "was", "were", "be", "this", "that", "it", "as", "by", "from", "at",
    }
    words = re.findall(r"[a-zA-Z]{3,}", cleaned.lower())
    freq = Counter(w for w in words if w not in stop_words)

    scored: List[tuple[int, float, str]] = []
    for idx, sentence in enumerate(sentences):
        tokens = [w for w in re.findall(r"[a-zA-Z]{3,}", sentence.lower()) if w not in stop_words]
        if not tokens:
            continue
        score = sum(freq.get(t, 0) for t in tokens) / max(len(tokens), 1)
        scored.append((idx, score, sentence))

    if not scored:
        return "\n".join(f"- {s}" for s in sentences[:bullet_count])

    top = sorted(scored, key=lambda x: x[1], reverse=True)
    selected: List[tuple[int, float, str]] = []
    seen = set()
    for item in top:
        sentence_key = item[2].lower()
        if sentence_key in seen:
            continue
        seen.add(sentence_key)
        selected.append(item)
        if len(selected) >= bullet_count:
            break

    top_sorted = sorted(selected, key=lambda x: x[0])
    bullets = [f"- {s}" for _, _, s in top_sorted]
    return "\n".join(bullets)


def _format_history(history: List[Mapping[str, str]]) -> str:
    lines: List[str] = []
    for item in history[-8:]:
        role = item.get("role", "user").title()
        content = item.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _format_context(documents: Iterable[Document]) -> str:
    chunks: List[str] = []
    for idx, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        chunks.append(f"[Chunk {idx} | Source: {source} | Page: {page}]\n{doc.page_content}")
    return "\n\n".join(chunks)


def answer_question(
    llm: BaseChatModel,
    question: str,
    context_docs: List[Document],
    chat_history: List[Mapping[str, str]],
) -> str:
    """Generate an answer grounded in retrieved context and prior chat."""
    context = _format_context(context_docs)
    history = _format_history(chat_history)

    system_prompt = (
        "You are an AI study assistant. Use only the provided context to answer. "
        "If context is missing, say you are unsure and ask for a more specific question. "
        "Keep answers educational and structured."
    )

    user_prompt = (
        f"Conversation history:\n{history or 'No previous messages.'}\n\n"
        f"Retrieved context:\n{context or 'No context retrieved.'}\n\n"
        f"Question: {question}\n\n"
        "Answer with:\n"
        "1) concise answer\n"
        "2) short explanation\n"
        "3) key takeaway"
    )

    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    return str(response.content)


def summarize_text(llm: Optional[BaseChatModel], text: str) -> str:
    """Produce a concise summary from document text."""
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    summary_mode = os.getenv("SUMMARY_MODE", "fast").strip().lower()

    if provider == "ollama" and summary_mode != "llm":
        return _fast_extractive_summary(text, bullet_count=6)

    if llm is None:
        raise ValueError("LLM summary mode is enabled, but no chat model was provided.")

    max_chars = int(os.getenv("SUMMARY_MAX_CHARS", "7000"))
    cleaned_text = re.sub(r"\s+", " ", text).strip()

    if len(cleaned_text) <= max_chars:
        clipped_text = cleaned_text
    else:
        # Keep representative slices so long PDFs summarize faster while preserving coverage.
        part = max_chars // 3
        middle_start = max((len(cleaned_text) // 2) - (part // 2), 0)
        clipped_text = (
            f"{cleaned_text[:part]}\n\n[...content omitted for speed...]\n\n"
            f"{cleaned_text[middle_start:middle_start + part]}\n\n"
            f"[...content omitted for speed...]\n\n{cleaned_text[-part:]}"
        )

    prompt = (
        "Create a FAST study summary from the content below. "
        "Output exactly 6 bullet points, each one short and practical. "
        "Cover: main ideas, key definitions, and formulas only if they appear. "
        "Keep total output under 140 words.\n\n"
        f"Content:\n{clipped_text}"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return str(response.content)


def _fast_local_quiz(text: str, question_count: int = 5, mcq: bool = True) -> str:
    """Create a quick quiz from source text without LLM latency."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return "No content available to generate quiz."

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if len(s.strip()) > 35]
    if not sentences:
        sentences = [cleaned[:220]]

    stop_words = {
        "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "is",
        "are", "was", "were", "be", "this", "that", "it", "as", "by", "from", "at",
    }
    words = re.findall(r"[a-zA-Z]{4,}", cleaned.lower())
    top_terms = [w for w, _ in Counter(w for w in words if w not in stop_words).most_common(30)]

    selected: List[str] = []
    seen = set()
    for sentence in sentences:
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        selected.append(sentence)
        if len(selected) >= question_count:
            break

    lines: List[str] = []
    for idx, sentence in enumerate(selected, start=1):
        concept_match = re.search(r"\b([A-Za-z][A-Za-z0-9-]{3,})\b", sentence)
        concept = concept_match.group(1) if concept_match else f"Concept {idx}"

        if mcq:
            distractors = [t for t in top_terms if t.lower() != concept.lower()][:3]
            while len(distractors) < 3:
                distractors.append(f"term{len(distractors)+1}")
            options = [concept] + distractors
            options_text = [
                f"A) {options[0]}",
                f"B) {options[1]}",
                f"C) {options[2]}",
                f"D) {options[3]}",
            ]
            lines.append(f"{idx}. Which term best matches this idea?")
            lines.append(f"   \"{sentence[:180]}\"")
            lines.extend(f"   {opt}" for opt in options_text)
            lines.append("   Answer: A")
            lines.append(f"   Explanation: The sentence directly describes {concept}.")
        else:
            lines.append(f"{idx}. Briefly explain this statement:")
            lines.append(f"   \"{sentence[:180]}\"")
            lines.append(f"   Ideal answer: {sentence}")

    return "\n".join(lines)


def generate_quiz(
    llm: Optional[BaseChatModel],
    text: str,
    question_count: int = 5,
    mcq: bool = True,
) -> str:
    """Generate either MCQ or short-answer quiz items."""
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    quiz_mode = os.getenv("QUIZ_MODE", "fast").strip().lower()

    if provider == "ollama" and quiz_mode != "llm":
        return _fast_local_quiz(text, question_count=question_count, mcq=mcq)

    if llm is None:
        raise ValueError("LLM quiz mode is enabled, but no chat model was provided.")

    max_chars = int(os.getenv("QUIZ_MAX_CHARS", "6000"))
    clipped_text = re.sub(r"\s+", " ", text).strip()[:max_chars]
    if mcq:
        style = (
            "Generate multiple-choice questions with four options (A-D), "
            "mark the correct answer, and add a one-line explanation."
        )
    else:
        style = "Generate short-answer questions and include ideal answers."

    prompt = (
        f"You are a teaching assistant. Create exactly {question_count} quiz questions from the content below. "
        f"Keep wording concise and avoid long explanations. {style}\n\nContent:\n{clipped_text}"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return str(response.content)
