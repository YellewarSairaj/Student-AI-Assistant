from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List

from agent.tools import AgentTools, SourceAttribution, ToolResult
from memory.memory_manager import MemoryManager


@dataclass
class AgentStep:
    thought: str
    action: str
    observation: str
    reflection: str


@dataclass
class AgentResponse:
    final_answer: str
    tool_used: str
    confidence: float
    sources: List[SourceAttribution]
    steps: List[AgentStep]


class TutorAgentController:
    """Autonomous tutor agent with a lightweight Thought-Action-Observation loop."""

    def __init__(self, tools: AgentTools, memory: MemoryManager) -> None:
        self.tools = tools
        self.memory = memory

    def _confidence_label(self, score: float) -> str:
        if score >= 0.75:
            return "high"
        if score >= 0.5:
            return "medium"
        return "low"

    def _format_study_response(self, tool_result: ToolResult) -> str:
        content = str(tool_result.content or "").strip()

        # If a study tool accidentally returns JSON payload text, extract only the
        # human-facing final answer so chat output stays natural.
        if content.startswith("{") and "\"final_answer\"" in content:
            try:
                payload = json.loads(content)
                extracted = str(payload.get("final_answer", "")).strip()
                if extracted:
                    return extracted
            except Exception:
                pass

        return content

    def is_backup_query(self, query: str) -> bool:
        normalized = query.lower()
        strong_backup_keywords = [
            "backup",
            "restore",
            "duplicate",
            "timeline",
            "diff",
            "changed",
            "storage",
            "version history",
        ]
        if any(k in normalized for k in strong_backup_keywords):
            return True

        # Treat broad words like "file" and "version" as backup only when the query
        # clearly asks for backup-management style actions.
        has_file_or_version_term = any(k in normalized for k in ["file", "upload", "version", "stored", "saved"])
        has_backup_action = any(
            k in normalized
            for k in [
                "list",
                "show",
                "history",
                "how many",
                "restore",
                "duplicate",
                "timeline",
                "backup",
                "changed",
                "diff",
                "upload",
            ]
        )
        return has_file_or_version_term and has_backup_action

    def _is_explain_document_query(self, query: str) -> bool:
        normalized = query.lower()
        explain_markers = [
            "explain the pdf",
            "explain this pdf",
            "explain pdf",
            "explain the document",
            "explain this document",
            "explain the file",
            "explain this file",
        ]
        return any(marker in normalized for marker in explain_markers)

    def _classify_intent(self, query: str) -> str:
        normalized = query.lower()
        if any(k in normalized for k in ["quiz", "mcq", "test me", "practice question"]):
            return "quiz"
        if any(k in normalized for k in ["summarize", "summary", "brief", "overview"]):
            return "summary"
        if self._is_explain_document_query(query):
            return "summary"
        if any(k in normalized for k in ["calculate", "+", "-", "*", "/", "^"]):
            return "calculator"
        if any(k in normalized for k in ["remember", "what did i ask", "memory", "earlier"]):
            return "memory"
        if self.is_backup_query(query):
            return "backup"
        if not any(k in normalized for k in ["what", "why", "how", "explain", "question", "document"]):
            return "clarify"
        return "qa"

    def _select_tool(self, intent: str) -> str:
        mapping = {
            "qa": "document_qa",
            "summary": "summarization",
            "quiz": "quiz_generator",
            "calculator": "calculator",
            "memory": "memory_retrieval",
            "backup": "backup_assistant",
            "clarify": "clarification",
        }
        return mapping.get(intent, "document_qa")

    def _execute_tool(self, tool: str, query: str) -> ToolResult:
        if tool == "clarification":
            return ToolResult(
                tool_name="clarification",
                content=(
                    "Please clarify whether you want Study Mode (questions, summary, quiz) "
                    "or Backup Mode (files, versions, restore, duplicate, backup frequency)."
                ),
                sources=[],
                confidence=0.55,
                observation="Asked user for clarification due to ambiguous intent.",
            )
        if tool == "backup_assistant":
            return self.tools.backup_assistant_tool(query)
        if tool == "summarization":
            return self.tools.summarization_tool(query)
        if tool == "quiz_generator":
            return self.tools.quiz_tool(query)
        if tool == "calculator":
            return self.tools.calculator_tool(query)
        if tool == "memory_retrieval":
            return self.tools.memory_retrieval_tool(query)
        return self.tools.document_qa_tool(query)

    def run(self, query: str) -> AgentResponse:
        intent = self._classify_intent(query)
        tool = self._select_tool(intent)

        steps: List[AgentStep] = []
        thought = f"The user intent appears to be '{intent}'. I should choose the best tool."
        action = f"Call tool: {tool}"

        tool_result = self._execute_tool(tool, query)
        final_content = tool_result.content
        if intent != "backup":
            final_content = self._format_study_response(tool_result)

        reflection = (
            "Result looks grounded and complete. "
            "Return concise educational output with source attribution and confidence."
        )
        steps.append(
            AgentStep(
                thought=thought,
                action=action,
                observation=tool_result.observation,
                reflection=reflection,
            )
        )

        self.memory.add_message("user", query)
        self.memory.add_message("assistant", final_content)
        self.memory.update_weak_topics(query, tool_result.confidence)

        return AgentResponse(
            final_answer=final_content,
            tool_used=tool_result.tool_name,
            confidence=tool_result.confidence,
            sources=tool_result.sources,
            steps=steps,
        )
