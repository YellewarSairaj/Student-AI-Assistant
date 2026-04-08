from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import json
import os
import re
from typing import Deque, Dict, List, Mapping

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


@dataclass
class LearningInsights:
    weak_topics: List[str]
    frequent_topics: List[str]
    interaction_count: int


class MemoryManager:
    """Short-term + long-term learner memory with lightweight personalization."""

    def __init__(self, short_term_window: int = 10) -> None:
        self.short_term_window = short_term_window
        self.short_term: Deque[Dict[str, str]] = deque(maxlen=short_term_window)
        self.long_term_store: FAISS | None = None
        self.topic_counter: Counter[str] = Counter()
        self.weak_topic_counter: Counter[str] = Counter()
        self.interaction_count = 0

    def _extract_topics(self, text: str, top_k: int = 4) -> List[str]:
        stop_words = {
            "this", "that", "with", "from", "have", "what", "when", "where", "which",
            "their", "there", "about", "into", "after", "before", "could", "would", "should",
            "study", "explain", "please", "question", "answer",
        }
        words = [
            w for w in re.findall(r"[a-zA-Z]{4,}", (text or "").lower()) if w not in stop_words
        ]
        return [topic for topic, _ in Counter(words).most_common(top_k)]

    def add_message(self, role: str, content: str) -> None:
        self.short_term.append({"role": role, "content": content})
        if role == "user":
            self.interaction_count += 1
            for topic in self._extract_topics(content):
                self.topic_counter[topic] += 1

    def add_long_term_memory(self, embeddings: Embeddings, role: str, content: str) -> None:
        text = content.strip()
        if not text:
            return

        doc = Document(page_content=f"{role}: {text}", metadata={"source": "conversation", "page": "-"})
        if self.long_term_store is None:
            self.long_term_store = FAISS.from_documents([doc], embeddings)
        else:
            self.long_term_store.add_documents([doc])

    def retrieve_relevant_memories(self, query: str, k: int = 3) -> List[Document]:
        if self.long_term_store is None:
            return []
        return self.long_term_store.similarity_search(query, k=k)

    def recent_history(self, limit: int = 8) -> List[Mapping[str, str]]:
        return list(self.short_term)[-limit:]

    def update_weak_topics(self, query: str, confidence: float) -> None:
        if confidence >= 0.5:
            return
        for topic in self._extract_topics(query, top_k=3):
            self.weak_topic_counter[topic] += 1

    def get_insights(self) -> LearningInsights:
        weak_topics = [topic for topic, _ in self.weak_topic_counter.most_common(5)]
        frequent_topics = [topic for topic, _ in self.topic_counter.most_common(5)]
        return LearningInsights(
            weak_topics=weak_topics,
            frequent_topics=frequent_topics,
            interaction_count=self.interaction_count,
        )

    def save_to_disk(self, folder_path: str) -> None:
        os.makedirs(folder_path, exist_ok=True)

        state_path = os.path.join(folder_path, "memory_state.json")
        payload = {
            "short_term_window": self.short_term_window,
            "short_term": list(self.short_term),
            "topic_counter": dict(self.topic_counter),
            "weak_topic_counter": dict(self.weak_topic_counter),
            "interaction_count": self.interaction_count,
        }
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

        if self.long_term_store is not None:
            self.long_term_store.save_local(os.path.join(folder_path, "long_term_faiss"))

    def load_from_disk(self, folder_path: str, embeddings: Embeddings) -> bool:
        loaded_any = False

        state_path = os.path.join(folder_path, "memory_state.json")
        if os.path.exists(state_path):
            with open(state_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            self.short_term_window = int(payload.get("short_term_window", self.short_term_window))
            self.short_term = deque(payload.get("short_term", []), maxlen=self.short_term_window)
            self.topic_counter = Counter(payload.get("topic_counter", {}))
            self.weak_topic_counter = Counter(payload.get("weak_topic_counter", {}))
            self.interaction_count = int(payload.get("interaction_count", 0))
            loaded_any = True

        faiss_dir = os.path.join(folder_path, "long_term_faiss")
        if os.path.exists(os.path.join(faiss_dir, "index.faiss")):
            self.long_term_store = FAISS.load_local(
                faiss_dir,
                embeddings,
                allow_dangerous_deserialization=True,
            )
            loaded_any = True

        return loaded_any
