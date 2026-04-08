from __future__ import annotations

import ast
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from memory.memory_manager import MemoryManager
from retrieval.hybrid_retriever import HybridRetriever
from utils.llm_handler import generate_quiz, summarize_text


@dataclass
class SourceAttribution:
    source: str
    page: str
    snippet: str
    confidence: float


@dataclass
class ToolResult:
    tool_name: str
    content: str
    sources: List[SourceAttribution]
    confidence: float
    observation: str


class AgentTools:
    def __init__(
        self,
        llm: BaseChatModel,
        retriever: HybridRetriever,
        memory: MemoryManager,
        state_provider: Optional[Callable[[], Dict]] = None,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.memory = memory
        self.state_provider = state_provider

    def _get_runtime_state(self) -> Dict:
        if not self.state_provider:
            return {}
        try:
            return self.state_provider() or {}
        except Exception:
            return {}

    def _backup_sources(self, state: Dict, summary_versions: List[Dict]) -> List[SourceAttribution]:
        sources: List[SourceAttribution] = []
        for _, path in list((state.get("stored_files") or {}).items())[:3]:
            sources.append(
                SourceAttribution(
                    source=os.path.basename(str(path)),
                    page="stored-file",
                    snippet=str(path),
                    confidence=0.8,
                )
            )

        if summary_versions:
            latest = summary_versions[-1]
            sources.append(
                SourceAttribution(
                    source="Latest PDF Summary",
                    page=f"v{latest.get('version', '?')}",
                    snippet=str(latest.get("timestamp", "unknown")),
                    confidence=0.85,
                )
            )
        return sources[:4]

    def _backup_frequency_label(self, activity_log: List[Dict]) -> str:
        if not activity_log:
            return "MONTHLY"

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=7)
        weekly_actions = 0
        for event in activity_log:
            ts_raw = str(event.get("timestamp", "")).replace("Z", "+00:00")
            try:
                ts = datetime.fromisoformat(ts_raw)
            except ValueError:
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts >= cutoff and event.get("action") in {"upload", "process", "summary"}:
                weekly_actions += 1

        if weekly_actions > 5:
            return "DAILY"
        if weekly_actions >= 2:
            return "WEEKLY"
        return "MONTHLY"

    def _should_enforce_latest_only(self, query: str) -> bool:
        normalized = (query or "").lower()
        explicit_old_markers = [
            "old file",
            "older file",
            "previous file",
            "previous document",
            "summarize old",
            "restore previous version",
            "version history",
        ]
        return not any(marker in normalized for marker in explicit_old_markers)

    def _latest_uploaded_file_name(self, state: Dict) -> Optional[str]:
        activity_log = state.get("activity_log") or []
        latest_name: Optional[str] = None
        latest_ts: Optional[datetime] = None

        for event in activity_log:
            if str(event.get("action", "")).lower() != "upload":
                continue
            file_name = str(event.get("file_name", "")).strip()
            if not file_name:
                continue

            ts_raw = str(event.get("timestamp", "")).replace("Z", "+00:00")
            try:
                ts = datetime.fromisoformat(ts_raw)
            except ValueError:
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            if latest_ts is None or ts > latest_ts:
                latest_ts = ts
                latest_name = file_name

        if latest_name:
            return latest_name

        db_store = state.get("db_store")
        if db_store and getattr(db_store, "enabled", False):
            try:
                resolved = db_store.resolve_file()
                if resolved and resolved.get("file_name"):
                    return str(resolved["file_name"])
            except Exception:
                pass

        file_versions = state.get("file_versions") or {}
        best_name: Optional[str] = None
        best_ts: Optional[datetime] = None
        for _, record in file_versions.items():
            file_name = str(record.get("file_name", "")).strip()
            for v in record.get("versions", []):
                ts_raw = str(v.get("timestamp", "")).replace("Z", "+00:00")
                try:
                    ts = datetime.fromisoformat(ts_raw)
                except ValueError:
                    continue
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if best_ts is None or ts > best_ts:
                    best_ts = ts
                    best_name = file_name

        return best_name

    def _same_source(self, source: str, target: str) -> bool:
        return source.strip().lower() == target.strip().lower()

    def _filter_docs_by_latest_source(self, docs: List[Document], latest_file: str) -> List[Document]:
        return [
            doc
            for doc in docs
            if self._same_source(str(doc.metadata.get("source", "")), latest_file)
        ]

    def _filter_retrieved_chunks_by_latest_source(self, items: List[Any], latest_file: str) -> List[Any]:
        filtered: List[Any] = []
        for item in items:
            doc = getattr(item, "document", None)
            source = str(getattr(doc, "metadata", {}).get("source", "")) if doc else ""
            if self._same_source(source, latest_file):
                filtered.append(item)
        return filtered

    def _confidence_label(self, score: float) -> str:
        if score >= 0.75:
            return "high"
        if score >= 0.5:
            return "medium"
        return "low"

    def _to_step_by_step(self, text: str, query: str) -> str:
        cleaned = (text or "").strip()
        if not cleaned:
            return cleaned

        normalized_query = (query or "").lower()
        wants_steps = any(
            k in normalized_query
            for k in ["explain", "summary", "summarize", "overview", "how", "steps"]
        )
        if not wants_steps:
            return cleaned

        if re.search(r"(?m)^\s*\d+\.\s+", cleaned):
            return cleaned

        lines = [line.strip(" -\t") for line in cleaned.splitlines() if line.strip()]
        if not lines:
            return cleaned

        max_points = min(6, len(lines))
        numbered = [f"{idx}. {line}" for idx, line in enumerate(lines[:max_points], start=1)]
        return "Step-by-step explanation:\n" + "\n".join(numbered)

    def _format_backup_response(
        self,
        final_answer: str,
        confidence_score: float,
        stored_files: Dict,
        summary_versions: List[Dict],
        backup_label: str,
    ) -> str:
        latest_version = summary_versions[-1].get("version") if summary_versions else "?"
        source_items: List[Dict[str, str]] = []
        for _, path in list(stored_files.items())[:3]:
            source_items.append(
                {
                    "document_or_file": os.path.basename(str(path)),
                    "version": "stored",
                    "timestamp": "local",
                    "snippet": str(path),
                }
            )
        if summary_versions:
            source_items.append(
                {
                    "document_or_file": "Latest PDF Summary",
                    "version": f"v{latest_version}",
                    "timestamp": str(summary_versions[-1].get("timestamp", "unknown")),
                    "snippet": "inferred summary snapshot",
                }
            )
        if not source_items:
            source_items.append(
                {
                    "document_or_file": "system",
                    "version": "-",
                    "timestamp": "-",
                    "snippet": "Not enough data available",
                }
            )

        payload = {
            "final_answer": (
                f"{final_answer} "
                f"(Stored Files={len(stored_files)}, Inferred Versions={len(summary_versions)}, Recommended Backup={backup_label})"
            ),
            "confidence": self._confidence_label(confidence_score),
            "sources": source_items,
        }
        return json.dumps(payload, ensure_ascii=True, indent=2)

    def backup_assistant_tool(self, query: str) -> ToolResult:
        state = self._get_runtime_state()
        stored_files = state.get("stored_files") or {}
        processed_hashes = state.get("processed_hashes") or {}
        file_versions = state.get("file_versions") or {}
        summary_versions = state.get("summary_versions") or []
        activity_log = state.get("activity_log") or []
        db_store = state.get("db_store")
        backup_label = self._backup_frequency_label(activity_log)

        normalized = query.lower()
        confidence = 0.82

        db_has_data = False
        if db_store and getattr(db_store, "enabled", False):
            try:
                db_has_data = bool(db_store.list_files(limit=1))
            except Exception:
                db_has_data = False

        if not stored_files and not summary_versions and not file_versions and not db_has_data:
            return ToolResult(
                tool_name="backup_assistant",
                content=self._format_backup_response(
                    final_answer="Not found in system.",
                    confidence_score=0.4,
                    stored_files=stored_files,
                    summary_versions=summary_versions,
                    backup_label=backup_label,
                ),
                sources=[
                    SourceAttribution(
                        source="backup-runtime-state",
                        page="-",
                        snippet="No stored files or summary versions yet",
                        confidence=0.4,
                    )
                ],
                confidence=0.4,
                observation="No backup-mode artifacts available.",
            )

        tool_result = self._route_backup_tool(query, state)
        content = tool_result.content
        confidence = tool_result.confidence
        observation = tool_result.observation

        return ToolResult(
            tool_name=tool_result.tool_name,
            content=self._format_backup_response(
                final_answer=content,
                confidence_score=confidence,
                stored_files=stored_files,
                summary_versions=summary_versions,
                backup_label=backup_label,
            ),
            sources=self._backup_sources(state, summary_versions),
            confidence=confidence,
            observation=observation,
        )

    def _route_backup_tool(self, query: str, state: Dict) -> ToolResult:
        normalized = query.lower()
        if "duplicate" in normalized:
            return self.duplicate_detection_tool(state)
        if "changed" in normalized or "difference" in normalized or "diff" in normalized:
            return self.file_diff_tool(state)
        if "restore" in normalized:
            return self.restore_version_tool(state, query)
        if "backup" in normalized or "frequency" in normalized:
            return self.backup_frequency_tool(state)
        if "timeline" in normalized:
            return self.timeline_tool(state)
        if "how many versions" in normalized or "number of versions" in normalized or "version" in normalized or "history" in normalized:
            return self.version_history_tool(state, query)
        if "upload" in normalized:
            return self.file_upload_tool(state)
        if "file" in normalized or "stored" in normalized or "list" in normalized:
            return self.file_list_tool(state)
        return ToolResult(
            tool_name="backup_assistant",
            content="I can help with uploads, file listing, version history, restore guidance, duplicate checks, diff analysis, timeline, and backup frequency.",
            sources=[],
            confidence=0.65,
            observation="Backup help response returned for broad request.",
        )

    def file_upload_tool(self, state: Dict) -> ToolResult:
        stored_files = state.get("stored_files") or {}
        if not stored_files:
            return ToolResult(
                tool_name="file_upload_tool",
                content="Use the sidebar Upload PDFs control, then click Process PDFs to register a file backup event.",
                sources=[],
                confidence=0.6,
                observation="Provided upload workflow guidance.",
            )
        return ToolResult(
            tool_name="file_upload_tool",
            content=f"Upload system ready. {len(stored_files)} file(s) are currently stored.",
            sources=[],
            confidence=0.75,
            observation="Reported current upload/store status.",
        )

    def file_list_tool(self, state: Dict) -> ToolResult:
        db_store = state.get("db_store")
        if db_store and getattr(db_store, "enabled", False):
            try:
                db_files = db_store.list_files(limit=12)
                if db_files:
                    return ToolResult(
                        tool_name="file_list_tool",
                        content="Saved uploaded files:\n" + "\n".join(
                            f"- {item['file_name']}" for item in db_files
                        ),
                        sources=[],
                        confidence=0.85,
                        observation="Listed files from MySQL store.",
                    )
            except Exception:
                pass

        stored_files = state.get("stored_files") or {}
        file_names = [os.path.basename(str(path)) for _, path in list(stored_files.items())[:12]]
        if not file_names:
            return ToolResult(
                tool_name="file_list_tool",
                content="Not found in system.",
                sources=[],
                confidence=0.45,
                observation="No saved uploaded files available.",
            )
        return ToolResult(
            tool_name="file_list_tool",
            content="Saved uploaded files:\n" + "\n".join(f"- {name}" for name in file_names),
            sources=[],
            confidence=0.8,
            observation=f"Listed {len(file_names)} saved files from local storage.",
        )

    def version_history_tool(self, state: Dict, query: str = "") -> ToolResult:
        db_store = state.get("db_store")
        if db_store and getattr(db_store, "enabled", False):
            try:
                file_hint = None
                query_lower = query.lower()
                if "for " in query_lower:
                    file_hint = query_lower.split("for ", 1)[1].strip()[:80]

                target_file = db_store.resolve_file(file_hint)
                if target_file:
                    versions = db_store.list_versions_for_file(target_file["id"])
                    lines = [
                        f"- {target_file['file_name']} v{item['version_number']} at {item['created_at']}"
                        for item in versions[-10:]
                    ]
                    return ToolResult(
                        tool_name="version_history_tool",
                        content="Version history:\n" + "\n".join(lines),
                        sources=[],
                        confidence=0.88,
                        observation="Built version history from MySQL file_versions.",
                    )
            except Exception:
                pass

        file_versions = state.get("file_versions") or {}
        summary_versions = state.get("summary_versions") or []
        if not file_versions and not summary_versions:
            return ToolResult(
                tool_name="version_history_tool",
                content="Not found in system.",
                sources=[],
                confidence=0.45,
                observation="No version artifacts found.",
            )

        lines: List[str] = []
        for _, rec in list(file_versions.items())[:6]:
            file_name = rec.get("file_name", "unknown")
            versions = rec.get("versions", [])
            lines.append(f"- {file_name}: {len(versions)} version(s)")

        if summary_versions:
            lines.append(f"- Inferred summary timeline: {len(summary_versions)} version(s)")

        return ToolResult(
            tool_name="version_history_tool",
            content="Version history:\n" + "\n".join(lines),
            sources=[],
            confidence=0.82,
            observation="Built version history from persistent local version metadata.",
        )

    def restore_version_tool(self, state: Dict, query: str = "") -> ToolResult:
        db_store = state.get("db_store")
        if db_store and getattr(db_store, "enabled", False):
            try:
                target_file = db_store.resolve_file()
                requested = db_store.extract_version_number(query)
                if target_file:
                    versions = db_store.list_versions_for_file(target_file["id"])
                    if not versions:
                        return ToolResult(
                            tool_name="restore_version_tool",
                            content="Not found in system.",
                            sources=[],
                            confidence=0.45,
                            observation="No DB versions available for restore guidance.",
                        )

                    max_version = versions[-1]["version_number"]
                    if requested is not None:
                        exists = any(v["version_number"] == requested for v in versions)
                        if not exists:
                            return ToolResult(
                                tool_name="restore_version_tool",
                                content=(
                                    f"Requested version v{requested} is not available for {target_file['file_name']}. "
                                    f"Available range: v1 to v{max_version}."
                                ),
                                sources=[],
                                confidence=0.55,
                                observation="Requested restore version missing in DB history.",
                            )
                        return ToolResult(
                            tool_name="restore_version_tool",
                            content=(
                                f"Restore guidance for {target_file['file_name']}: v{requested} exists and can be restored. "
                                "Warning: restoring an older version may overwrite newer content."
                            ),
                            sources=[],
                            confidence=0.88,
                            observation="Verified requested restore version in MySQL.",
                        )
            except Exception:
                pass

        summary_versions = state.get("summary_versions") or []
        if len(summary_versions) < 2:
            return ToolResult(
                tool_name="restore_version_tool",
                content="Restore guidance: only one inferred version exists, so there is nothing safer to revert to yet.",
                sources=[],
                confidence=0.55,
                observation="No earlier version available.",
            )
        current_v = summary_versions[-1].get("version", "?")
        suggested_v = summary_versions[-2].get("version", "?")
        return ToolResult(
            tool_name="restore_version_tool",
            content=(
                f"Current inferred version is v{current_v}. Safest rollback candidate is v{suggested_v}. "
                "Use 'Restore Previous Version' in Backup Insights. Warning: newer changes may be lost after restore."
            ),
            sources=[],
            confidence=0.8,
            observation="Generated restore guidance from inferred version stack.",
        )

    def duplicate_detection_tool(self, state: Dict) -> ToolResult:
        db_store = state.get("db_store")
        if db_store and getattr(db_store, "enabled", False):
            try:
                duplicate_name_groups = db_store.duplicate_name_candidates()
                if duplicate_name_groups:
                    return ToolResult(
                        tool_name="duplicate_detection_tool",
                        content=(
                            "Potential duplicates detected by repeated file names: "
                            + ", ".join(duplicate_name_groups)
                            + ". Suggestion: skip redundant uploads unless content changed."
                        ),
                        sources=[],
                        confidence=0.82,
                        observation="Duplicate check performed using MySQL files table.",
                    )
            except Exception:
                pass

        processed_hashes = state.get("processed_hashes") or {}
        names = list(processed_hashes.values())
        duplicate_name_groups = [name for name, count in Counter(names).items() if count > 1]
        if duplicate_name_groups:
            content = (
                "Potential duplicates detected by repeated file names: "
                + ", ".join(duplicate_name_groups)
                + ". Suggestion: skip redundant uploads unless content changed."
            )
        else:
            content = "No duplicate content hashes are stored right now."
        return ToolResult(
            tool_name="duplicate_detection_tool",
            content=content,
            sources=[],
            confidence=0.78,
            observation="Duplicate check performed on local hash and file-name history.",
        )

    def backup_frequency_tool(self, state: Dict) -> ToolResult:
        db_store = state.get("db_store")
        if db_store and getattr(db_store, "enabled", False):
            try:
                usage_count = db_store.usage_count_last_days(days=7)
                version_events = db_store.version_event_count_last_days(days=7)
                if usage_count > 5:
                    label = "DAILY"
                elif usage_count >= 2:
                    label = "WEEKLY"
                else:
                    label = "MONTHLY"

                risk = "HIGH RISK" if version_events > 5 and usage_count < 2 else "LOW RISK"
                return ToolResult(
                    tool_name="backup_frequency_tool",
                    content=(
                        f"Recommended backup frequency: {label}. "
                        f"Risk assessment: {risk}. "
                        "This is inferred from MySQL file_usage and file_versions activity in the last 7 days."
                    ),
                    sources=[],
                    confidence=0.84,
                    observation="Calculated backup frequency from MySQL file_usage.",
                )
            except Exception:
                pass

        activity_log = state.get("activity_log") or []
        label = self._backup_frequency_label(activity_log)
        return ToolResult(
            tool_name="backup_frequency_tool",
            content=(
                f"Recommended backup frequency: {label}. "
                "This is inferred from recent upload/process/summary activity in the last 7 days."
            ),
            sources=[],
            confidence=0.8,
            observation="Calculated backup frequency from local activity log.",
        )

    def file_diff_tool(self, state: Dict) -> ToolResult:
        summary_versions = state.get("summary_versions") or []
        if len(summary_versions) < 2:
            return ToolResult(
                tool_name="file_diff_tool",
                content="Not enough versions to compare yet. Generate another summary after processing to see changes.",
                sources=[],
                confidence=0.5,
                observation="Need at least two inferred versions for diff.",
            )

        prev = str(summary_versions[-2].get("summary", ""))
        latest = str(summary_versions[-1].get("summary", ""))
        prev_lines = {line.strip() for line in prev.splitlines() if line.strip()}
        latest_lines = {line.strip() for line in latest.splitlines() if line.strip()}
        added = list(latest_lines - prev_lines)[:4]
        removed = list(prev_lines - latest_lines)[:4]
        content = (
            "Summary change analysis between latest two inferred versions:\n"
            + ("Added:\n" + "\n".join(f"- {x}" for x in added) if added else "Added:\n- No major additions")
            + "\n"
            + ("Removed:\n" + "\n".join(f"- {x}" for x in removed) if removed else "Removed:\n- No major removals")
        )
        return ToolResult(
            tool_name="file_diff_tool",
            content=content,
            sources=[],
            confidence=0.78,
            observation="Compared previous and latest summary snapshots.",
        )

    def timeline_tool(self, state: Dict) -> ToolResult:
        db_store = state.get("db_store")
        if db_store and getattr(db_store, "enabled", False):
            try:
                rows = db_store.list_file_versions(limit=12)
                if rows:
                    lines = [
                        f"- {row['file_name']} v{row['version_number']} at {row['created_at']}"
                        for row in rows
                    ]
                    return ToolResult(
                        tool_name="timeline_tool",
                        content="Version timeline from MySQL:\n" + "\n".join(lines),
                        sources=[],
                        confidence=0.86,
                        observation="Built timeline from MySQL file_versions.",
                    )
            except Exception:
                pass

        summary_versions = state.get("summary_versions") or []
        if not summary_versions:
            return ToolResult(
                tool_name="timeline_tool",
                content="Not found in system.",
                sources=[],
                confidence=0.45,
                observation="No inferred summary versions found.",
            )
        lines = [
            f"- v{item.get('version', '?')} at {item.get('timestamp', 'unknown')} ({item.get('trigger', 'unknown')})"
            for item in summary_versions[-8:]
        ]
        return ToolResult(
            tool_name="timeline_tool",
            content="Inferred version timeline from summary updates:\n" + "\n".join(lines),
            sources=[],
            confidence=0.8,
            observation=f"Built timeline from {len(summary_versions)} inferred versions.",
        )

    def _build_sources_from_docs(self, docs: List[Document], confidence: float) -> List[SourceAttribution]:
        sources: List[SourceAttribution] = []
        for doc in docs:
            snippet = doc.page_content[:240].replace("\n", " ")
            sources.append(
                SourceAttribution(
                    source=str(doc.metadata.get("source", "unknown")),
                    page=str(doc.metadata.get("page", "?")),
                    snippet=snippet,
                    confidence=confidence,
                )
            )
        return sources

    def memory_retrieval_tool(self, query: str) -> ToolResult:
        memories = self.memory.retrieve_relevant_memories(query, k=3)
        if not memories:
            return ToolResult(
                tool_name="memory_retrieval",
                content="No relevant past memory found.",
                sources=[
                    SourceAttribution(
                        source="memory",
                        page="-",
                        snippet="No relevant memory found",
                        confidence=0.4,
                    )
                ],
                confidence=0.4,
                observation="Memory store empty or no close match.",
            )

        text = "\n".join(f"- {doc.page_content}" for doc in memories)
        return ToolResult(
            tool_name="memory_retrieval",
            content=f"Relevant memory:\n{text}",
            sources=self._build_sources_from_docs(memories, 0.6),
            confidence=0.6,
            observation=f"Retrieved {len(memories)} memory items.",
        )

    def calculator_tool(self, query: str) -> ToolResult:
        expression = query.lower().replace("calculate", "").replace("what is", "").strip()
        expression = expression.replace("^", "**")

        allowed = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Mod,
            ast.Pow,
            ast.USub,
            ast.UAdd,
            ast.Constant,
            ast.Load,
        )

        try:
            tree = ast.parse(expression, mode="eval")
            for node in ast.walk(tree):
                if not isinstance(node, allowed):
                    raise ValueError("Unsafe expression")
            result = eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, {})
            answer = f"Result: {result}"
            return ToolResult(
                tool_name="calculator",
                content=answer,
                sources=[
                    SourceAttribution(
                        source="calculator",
                        page="-",
                        snippet=expression,
                        confidence=1.0,
                    )
                ],
                confidence=1.0,
                observation="Math expression evaluated safely.",
            )
        except Exception as exc:
            return ToolResult(
                tool_name="calculator",
                content=f"I could not evaluate that expression safely: {exc}",
                sources=[
                    SourceAttribution(
                        source="calculator",
                        page="-",
                        snippet=expression,
                        confidence=0.2,
                    )
                ],
                confidence=0.2,
                observation="Calculator parsing failed.",
            )

    def summarization_tool(self, query: str) -> ToolResult:
        state = self._get_runtime_state()
        enforce_latest = self._should_enforce_latest_only(query)
        latest_file = self._latest_uploaded_file_name(state) if enforce_latest else None

        if enforce_latest and not latest_file:
            return ToolResult(
                tool_name="summarization",
                content="No recent uploaded document found",
                sources=[
                    SourceAttribution(
                        source="system",
                        page="-",
                        snippet="No recent uploaded document metadata found",
                        confidence=0.3,
                    )
                ],
                confidence=0.3,
                observation="Latest-file priority enforced but no recent upload found.",
            )

        docs = self.retriever.retrieve(query or "summarize document", k=6)
        if latest_file:
            docs = self._filter_retrieved_chunks_by_latest_source(docs, latest_file)
        merged = "\n\n".join(item.document.page_content for item in docs)

        if not merged:
            all_docs = self.retriever.get_all_documents()
            if latest_file:
                all_docs = self._filter_docs_by_latest_source(all_docs, latest_file)
            all_docs = all_docs[:8]
            merged = "\n\n".join(d.page_content for d in all_docs)
            source_docs = all_docs
            confidence = 0.45
        else:
            source_docs = [item.document for item in docs]
            confidence = sum(item.score for item in docs) / max(len(docs), 1)

        summary = summarize_text(self.llm, merged)
        summary = self._to_step_by_step(summary, query)

        if not source_docs:
            return ToolResult(
                tool_name="summarization",
                content="No recent uploaded document found",
                sources=[
                    SourceAttribution(
                        source="system",
                        page="-",
                        snippet="No chunks found for latest uploaded file",
                        confidence=0.3,
                    )
                ],
                confidence=0.3,
                observation="Latest-file priority enforced; no retrievable chunks found.",
            )

        return ToolResult(
            tool_name="summarization",
            content=summary,
            sources=self._build_sources_from_docs(source_docs[:4], min(confidence, 0.99)),
            confidence=min(confidence, 0.99),
            observation=(
                f"Summarized {len(source_docs)} chunks"
                + (f" from latest file: {latest_file}." if latest_file else ".")
            ),
        )

    def quiz_tool(self, query: str) -> ToolResult:
        state = self._get_runtime_state()
        enforce_latest = self._should_enforce_latest_only(query)
        latest_file = self._latest_uploaded_file_name(state) if enforce_latest else None

        if enforce_latest and not latest_file:
            return ToolResult(
                tool_name="quiz_generator",
                content="No recent uploaded document found",
                sources=[
                    SourceAttribution(
                        source="system",
                        page="-",
                        snippet="No recent uploaded document metadata found",
                        confidence=0.3,
                    )
                ],
                confidence=0.3,
                observation="Latest-file priority enforced but no recent upload found.",
            )

        docs = self.retriever.retrieve(query or "generate quiz", k=6)
        if latest_file:
            docs = self._filter_retrieved_chunks_by_latest_source(docs, latest_file)
        merged = "\n\n".join(item.document.page_content for item in docs)

        if not merged:
            all_docs = self.retriever.get_all_documents()
            if latest_file:
                all_docs = self._filter_docs_by_latest_source(all_docs, latest_file)
            all_docs = all_docs[:8]
            merged = "\n\n".join(d.page_content for d in all_docs)
            source_docs = all_docs
            confidence = 0.45
        else:
            source_docs = [item.document for item in docs]
            confidence = sum(item.score for item in docs) / max(len(docs), 1)

        if not source_docs:
            return ToolResult(
                tool_name="quiz_generator",
                content="No recent uploaded document found",
                sources=[
                    SourceAttribution(
                        source="system",
                        page="-",
                        snippet="No chunks found for latest uploaded file",
                        confidence=0.3,
                    )
                ],
                confidence=0.3,
                observation="Latest-file priority enforced; no retrievable chunks found.",
            )

        quiz = generate_quiz(self.llm, merged, question_count=5, mcq=True)
        return ToolResult(
            tool_name="quiz_generator",
            content=quiz,
            sources=self._build_sources_from_docs(source_docs[:4], min(confidence, 0.99)),
            confidence=min(confidence, 0.99),
            observation=(
                f"Generated quiz from {len(source_docs)} chunks"
                + (f" from latest file: {latest_file}." if latest_file else ".")
            ),
        )

    def document_qa_tool(self, query: str) -> ToolResult:
        state = self._get_runtime_state()
        enforce_latest = self._should_enforce_latest_only(query)
        latest_file = self._latest_uploaded_file_name(state) if enforce_latest else None

        if enforce_latest and not latest_file:
            return ToolResult(
                tool_name="document_qa",
                content="No recent uploaded document found",
                sources=[
                    SourceAttribution(
                        source="system",
                        page="-",
                        snippet="No recent uploaded document metadata found",
                        confidence=0.3,
                    )
                ],
                confidence=0.3,
                observation="Latest-file priority enforced but no recent upload found.",
            )

        retrieved = self.retriever.retrieve(query, k=4)
        if latest_file:
            retrieved = self._filter_retrieved_chunks_by_latest_source(retrieved, latest_file)
        if not retrieved:
            return ToolResult(
                tool_name="document_qa",
                content=("No recent uploaded document found" if latest_file else "Not found in document."),
                sources=[
                    SourceAttribution(
                        source="retriever",
                        page="-",
                        snippet=(
                            "No chunks retrieved for latest uploaded file"
                            if latest_file
                            else "No chunks retrieved"
                        ),
                        confidence=(0.3 if latest_file else 0.2),
                    )
                ],
                confidence=(0.3 if latest_file else 0.2),
                observation=(
                    f"No retrieval match for latest file: {latest_file}."
                    if latest_file
                    else "No retrieval match."
                ),
            )

        docs = [item.document for item in retrieved]
        context = "\n\n".join(
            f"[Source: {d.metadata.get('source', 'unknown')} | Page: {d.metadata.get('page', '?')}]\n{d.page_content}"
            for d in docs
        )

        memories = self.memory.retrieve_relevant_memories(query, k=2)
        memory_context = "\n".join(doc.page_content for doc in memories)

        system_prompt = (
            "You are an expert tutor agent. Use only provided context and memory. "
            "If the answer cannot be found in context, reply exactly: Not found in document."
        )

        user_prompt = (
            f"Memory context:\n{memory_context or 'No memory yet.'}\n\n"
            f"Document context:\n{context}\n\n"
            f"Question: {query}\n"
            "Provide a concise teaching answer in this format:\n"
            "1. Direct answer\n"
            "2. Step-by-step explanation\n"
            "3. Key takeaway"
        )

        response = self.llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        answer = self._to_step_by_step(str(response.content), query)

        confidence = sum(item.score for item in retrieved) / max(len(retrieved), 1)
        sources = [
            SourceAttribution(
                source=str(item.document.metadata.get("source", "unknown")),
                page=str(item.document.metadata.get("page", "?")),
                snippet=item.document.page_content[:240].replace("\n", " "),
                confidence=min(item.score, 0.99),
            )
            for item in retrieved
        ]

        return ToolResult(
            tool_name="document_qa",
            content=answer,
            sources=sources,
            confidence=min(confidence, 0.99),
            observation=(
                f"Retrieved and reranked {len(retrieved)} chunks"
                + (f" from latest file: {latest_file}." if latest_file else ".")
            ),
        )
