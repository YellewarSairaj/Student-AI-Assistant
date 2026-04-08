from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.agent_controller import TutorAgentController
from agent.tools import AgentTools
from memory.memory_manager import MemoryManager
from retrieval.hybrid_retriever import HybridRetriever
from utils.db_store import BackupDbStore
from utils.embeddings import get_embedding_model
from utils.llm_handler import get_chat_model


def print_result(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    message = f"[{status}] {name}"
    if detail:
        message += f" - {detail}"
    print(message)


def check_env() -> bool:
    required = ["LLM_PROVIDER", "OLLAMA_BASE_URL", "OLLAMA_MODEL", "OLLAMA_EMBEDDING_MODEL"]
    missing = [key for key in required if not os.getenv(key)]
    ok = not missing
    print_result("Environment variables", ok, "missing: " + ", ".join(missing) if missing else "ok")
    return ok


def check_ollama() -> bool:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    url = f"{base_url}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            payload = response.read().decode("utf-8", errors="ignore")
        ok = '"models"' in payload
        print_result("Ollama API", ok, f"url={url}")
        return ok
    except (urllib.error.URLError, TimeoutError) as exc:
        print_result("Ollama API", False, f"url={url} error={exc}")
        return False


def check_local_state() -> bool:
    state_file = Path(".agent_data/app_state.json")
    uploads_dir = Path(".agent_data/uploads")

    ok = state_file.exists() and uploads_dir.exists()
    detail = f"state_file={state_file.exists()} uploads_dir={uploads_dir.exists()}"
    print_result("Local state artifacts", ok, detail)
    return ok


def check_vector_memory_load() -> bool:
    try:
        embeddings = get_embedding_model("ollama")
        retriever = HybridRetriever(embeddings)
        retriever.load_from_disk(str(Path(".agent_data/retriever")))

        memory = MemoryManager(short_term_window=10)
        memory.load_from_disk(str(Path(".agent_data/memory")), embeddings)

        ok = True
        detail = f"retriever_has_index={retriever.has_index}"
        print_result("Retriever + Memory load", ok, detail)
        return True
    except Exception as exc:
        print_result("Retriever + Memory load", False, str(exc))
        return False


def check_mysql() -> bool:
    db = BackupDbStore.from_env()
    if not db.enabled:
        print_result("MySQL connection", False, "DB_PROVIDER not mysql or connector/credentials missing")
        return False

    try:
        db.ensure_schema()
        files = db.list_files(limit=1)
        versions = db.list_file_versions(limit=1)
        print_result("MySQL connection", True, f"files={len(files)} versions={len(versions)}")
        return True
    except Exception as exc:
        print_result("MySQL connection", False, str(exc))
        return False


def check_agent_routing() -> bool:
    try:
        embeddings = get_embedding_model("ollama")
        retriever = HybridRetriever(embeddings)
        retriever.load_from_disk(str(Path(".agent_data/retriever")))

        memory = MemoryManager(short_term_window=10)
        memory.load_from_disk(str(Path(".agent_data/memory")), embeddings)

        llm = get_chat_model("ollama", temperature=0.25)
        db = BackupDbStore.from_env()

        tools = AgentTools(
            llm=llm,
            retriever=retriever,
            memory=memory,
            state_provider=lambda: {
                "stored_files": {},
                "processed_hashes": {},
                "file_versions": {},
                "summary_versions": [],
                "activity_log": [],
                "db_store": db,
            },
        )
        controller = TutorAgentController(tools=tools, memory=memory)

        backup_response = controller.run("show version history")
        study_response = controller.run("summarize this document")

        ok = bool(backup_response.tool_used) and bool(study_response.tool_used)
        detail = f"backup_tool={backup_response.tool_used}, study_tool={study_response.tool_used}"
        print_result("Agent routing", ok, detail)
        return ok
    except Exception as exc:
        print_result("Agent routing", False, str(exc))
        return False


def main() -> int:
    load_dotenv()

    print("=== AI Study & Backup Health Check ===")
    checks = [
        check_env(),
        check_ollama(),
        check_local_state(),
        check_vector_memory_load(),
        check_mysql(),
        check_agent_routing(),
    ]

    passed = sum(1 for item in checks if item)
    total = len(checks)
    print(f"=== Summary: {passed}/{total} checks passed ===")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
