from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Generator

import streamlit as st
from dotenv import load_dotenv

from agent.agent_controller import TutorAgentController
from agent.tools import AgentTools
from memory.memory_manager import MemoryManager
from retrieval.hybrid_retriever import HybridRetriever
from utils.backfill_mysql import backfill_from_state_path
from utils.db_store import BackupDbStore
from utils.embeddings import get_embedding_model
from utils.llm_handler import get_chat_model
from utils.pdf_loader import extract_pdf_documents


load_dotenv()

st.set_page_config(page_title="AI Study Assistant Agent", page_icon="📘", layout="wide")
st.title("📘 AI Study Assistant Agent")
st.caption("Autonomous offline tutor agent with reasoning, tools, memory, and hybrid retrieval.")

DATA_DIR = os.path.join(os.getcwd(), ".agent_data")
RETRIEVER_DIR = os.path.join(DATA_DIR, "retriever")
MEMORY_DIR = os.path.join(DATA_DIR, "memory")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
STATE_FILE = os.path.join(DATA_DIR, "app_state.json")


def save_app_state() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    payload = {
        "processed_hashes": st.session_state.get("processed_hashes", {}),
        "latest_pdf_summary": st.session_state.get("latest_pdf_summary", ""),
        "stored_files": st.session_state.get("stored_files", {}),
        "file_versions": st.session_state.get("file_versions", {}),
        "summary_versions": st.session_state.get("summary_versions", []),
        "activity_log": st.session_state.get("activity_log", []),
    }
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def load_app_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


@st.cache_resource
def get_cached_embeddings(provider: str):
    return get_embedding_model(provider)


@st.cache_resource
def get_cached_llm(provider: str, temperature: float):
    return get_chat_model(provider, temperature=temperature)


def init_session_state() -> None:
    app_state = load_app_state()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processed_hashes" not in st.session_state:
        st.session_state.processed_hashes = app_state.get("processed_hashes", {})
    if "latest_pdf_summary" not in st.session_state:
        st.session_state.latest_pdf_summary = app_state.get("latest_pdf_summary", "")
    if "stored_files" not in st.session_state:
        st.session_state.stored_files = app_state.get("stored_files", {})
    if "file_versions" not in st.session_state:
        st.session_state.file_versions = app_state.get("file_versions", {})
    if "summary_versions" not in st.session_state:
        st.session_state.summary_versions = app_state.get("summary_versions", [])
    if "activity_log" not in st.session_state:
        st.session_state.activity_log = app_state.get("activity_log", [])
    if "db_sync_error" not in st.session_state:
        st.session_state.db_sync_error = ""
    if "db_sync_notice" not in st.session_state:
        st.session_state.db_sync_notice = ""
    if "db_store" not in st.session_state:
        db_store = BackupDbStore.from_env()
        if db_store.enabled:
            try:
                db_store.ensure_schema()
                db_files = db_store.list_files(limit=1)
                if not db_files and app_state.get("stored_files"):
                    synced_files, synced_versions = backfill_from_state_path(db_store, Path(STATE_FILE))
                    st.session_state.db_sync_notice = (
                        f"MySQL backfill completed: files={synced_files}, versions={synced_versions}"
                    )
                st.session_state.db_sync_error = ""
            except Exception as exc:
                st.session_state.db_sync_error = f"MySQL sync disabled: {exc}"
        st.session_state.db_store = db_store
    if "retriever" not in st.session_state:
        st.session_state.retriever = HybridRetriever(get_cached_embeddings("ollama"))
        st.session_state.retriever.load_from_disk(RETRIEVER_DIR)
    if "memory_manager" not in st.session_state:
        st.session_state.memory_manager = MemoryManager(short_term_window=10)
        st.session_state.memory_manager.load_from_disk(MEMORY_DIR, get_cached_embeddings("ollama"))
    if "agent_controller" not in st.session_state:
        llm = get_cached_llm("ollama", temperature=0.25)
        tools = AgentTools(
            llm=llm,
            retriever=st.session_state.retriever,
            memory=st.session_state.memory_manager,
            state_provider=lambda: {
                "stored_files": st.session_state.get("stored_files", {}),
                "processed_hashes": st.session_state.get("processed_hashes", {}),
                "file_versions": st.session_state.get("file_versions", {}),
                "summary_versions": st.session_state.get("summary_versions", []),
                "activity_log": st.session_state.get("activity_log", []),
                "db_store": st.session_state.get("db_store"),
            },
        )
        st.session_state.agent_controller = TutorAgentController(tools=tools, memory=st.session_state.memory_manager)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_activity(action: str, file_name: str = "", file_hash: str = "") -> None:
    st.session_state.activity_log.append(
        {
            "timestamp": utc_now_iso(),
            "action": action,
            "file_name": file_name,
            "file_hash": file_hash,
        }
    )


def register_file_version(file_hash: str, file_name: str, file_path: str, event: str) -> None:
    versions_store = st.session_state.file_versions
    current = versions_store.get(file_hash)
    if not current:
        current = {"file_id": file_hash[:12], "file_name": file_name, "versions": []}

    next_version = len(current.get("versions", [])) + 1
    current["versions"].append(
        {
            "version_number": next_version,
            "timestamp": utc_now_iso(),
            "file_path": file_path,
            "event": event,
        }
    )
    current["file_name"] = file_name
    versions_store[file_hash] = current

    db_store = st.session_state.get("db_store")
    if not db_store or not db_store.enabled:
        return

    try:
        file_id = db_store.upsert_file(file_name=file_name, file_hash=file_hash)
        if file_id is None:
            return
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        db_store.add_file_version(
            file_id=file_id,
            version_number=next_version,
            file_path=file_path,
            file_size=file_size,
        )
        db_store.log_action(file_id=file_id, action=event)
    except Exception as exc:
        st.session_state.db_sync_error = f"MySQL sync error: {exc}"


def add_summary_version(summary_text: str, trigger: str) -> None:
    next_version = len(st.session_state.summary_versions) + 1
    st.session_state.summary_versions.append(
        {
            "version": next_version,
            "timestamp": utc_now_iso(),
            "trigger": trigger,
            "summary": summary_text,
        }
    )
    append_activity("summary")


def restore_previous_summary_version() -> tuple[bool, str]:
    versions = st.session_state.get("summary_versions", [])
    if len(versions) < 2:
        return (False, "Not enough inferred versions to restore.")

    target = versions[-2]
    st.session_state.latest_pdf_summary = str(target.get("summary", ""))

    next_version = len(versions) + 1
    st.session_state.summary_versions.append(
        {
            "version": next_version,
            "timestamp": utc_now_iso(),
            "trigger": "restore",
            "summary": st.session_state.latest_pdf_summary,
            "restored_from": target.get("version"),
        }
    )
    append_activity("restore")
    save_app_state()

    restored_from = target.get("version", "?")
    return (True, f"Restored summary from inferred version v{restored_from}.")


def infer_backup_frequency(activity_log: list[dict]) -> str:
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


def latest_summary_change(summary_versions: list[dict]) -> tuple[list[str], list[str]]:
    if len(summary_versions) < 2:
        return ([], [])

    previous = str(summary_versions[-2].get("summary", ""))
    latest = str(summary_versions[-1].get("summary", ""))
    previous_lines = {line.strip() for line in previous.splitlines() if line.strip()}
    latest_lines = {line.strip() for line in latest.splitlines() if line.strip()}

    added = list(latest_lines - previous_lines)[:3]
    removed = list(previous_lines - latest_lines)[:3]
    return (added, removed)


init_session_state()


with st.sidebar:
    st.header("Agent Controls")
    st.caption("Mode: Offline Ollama only")
    st.caption(f"Model: {os.getenv('OLLAMA_MODEL', 'llama3.1')}")
    st.caption(f"Embeddings: {os.getenv('OLLAMA_EMBEDDING_MODEL', 'nomic-embed-text')}")
    db_store = st.session_state.get("db_store")
    db_status = "MySQL sync enabled" if db_store and db_store.enabled else "MySQL sync disabled"
    st.caption(f"DB: {db_status}")
    if st.session_state.get("db_sync_error"):
        st.caption(st.session_state.db_sync_error)
    if st.session_state.get("db_sync_notice"):
        st.caption(st.session_state.db_sync_notice)

    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )
    process_clicked = st.button("Process PDFs", use_container_width=True)
    generate_summary_clicked = st.button("Generate PDF Summary", use_container_width=True)
    clear_chat_clicked = st.button("Clear Chat", use_container_width=True)

    if st.session_state.get("latest_pdf_summary"):
        with st.expander("Latest PDF Summary", expanded=False):
            st.markdown(st.session_state.latest_pdf_summary)

    stored_files = st.session_state.get("stored_files", {})
    with st.expander("Saved Uploaded Files", expanded=False):
        st.caption(f"Storage folder: {UPLOADS_DIR}")
        if stored_files:
            for file_hash, stored_path in stored_files.items():
                file_name = os.path.basename(stored_path)
                st.write(file_name)
                if os.path.exists(stored_path):
                    with open(stored_path, "rb") as file_handle:
                        file_bytes = file_handle.read()
                    st.download_button(
                        label=f"Download {file_name}",
                        data=file_bytes,
                        file_name=file_name,
                        mime="application/pdf",
                        key=f"download_{file_hash}",
                    )
                else:
                    st.caption(f"Missing on disk: {stored_path}")
        else:
            st.caption("No saved originals yet. Upload a PDF and click Process PDFs.")

    st.divider()
    st.subheader("Backup Insights")
    st.metric("Stored Files", len(st.session_state.get("stored_files", {})))
    st.metric("Inferred Versions", len(st.session_state.get("summary_versions", [])))
    st.metric("Recommended Backup", infer_backup_frequency(st.session_state.get("activity_log", [])))

    if st.button("Restore Previous Version", use_container_width=True):
        restored, message = restore_previous_summary_version()
        if restored:
            st.success(message)
        else:
            st.warning(message)

    versions = st.session_state.get("summary_versions", [])
    if versions:
        with st.expander("Backup Timeline", expanded=False):
            for item in versions[-6:]:
                st.write(
                    f"- v{item.get('version', '?')} | {item.get('trigger', 'unknown')} | {item.get('timestamp', 'unknown')}"
                )

    added_points, removed_points = latest_summary_change(versions)
    if added_points or removed_points:
        with st.expander("Latest Version Changes", expanded=False):
            if added_points:
                st.write("Added")
                for point in added_points:
                    st.write(f"- {point}")
            if removed_points:
                st.write("Removed")
                for point in removed_points:
                    st.write(f"- {point}")

    st.subheader("Learning Insights")
    insights = st.session_state.memory_manager.get_insights()
    st.metric("Interactions", insights.interaction_count)
    st.write("Frequent Topics")
    if insights.frequent_topics:
        for topic in insights.frequent_topics:
            st.write(f"- {topic}")
    else:
        st.caption("No topic pattern yet.")

    st.write("Weak Topics")
    if insights.weak_topics:
        for topic in insights.weak_topics:
            st.write(f"- {topic}")
    else:
        st.caption("No weak-topic signal yet.")

if clear_chat_clicked:
    st.session_state.messages = []
    st.success("Chat history cleared.")

if process_clicked:
    if not uploaded_files:
        st.warning("Upload at least one PDF before processing.")
    else:
        try:
            with st.spinner("Processing PDFs with adaptive chunking + hybrid index..."):
                all_new_docs = []
                new_files_count = 0

                for uploaded_file in uploaded_files:
                    file_bytes = uploaded_file.getvalue()
                    file_hash = hashlib.sha256(file_bytes).hexdigest()
                    os.makedirs(UPLOADS_DIR, exist_ok=True)
                    safe_name = os.path.basename(uploaded_file.name)
                    stored_name = f"{file_hash[:12]}_{safe_name}"
                    stored_path = os.path.join(UPLOADS_DIR, stored_name)

                    if file_hash not in st.session_state.stored_files:
                        with open(stored_path, "wb") as saved_file:
                            saved_file.write(file_bytes)
                        st.session_state.stored_files[file_hash] = stored_path
                    register_file_version(file_hash, uploaded_file.name, stored_path, "upload")
                    append_activity("upload", uploaded_file.name, file_hash)

                    if file_hash in st.session_state.processed_hashes:
                        continue

                    processed = extract_pdf_documents(file_bytes, uploaded_file.name)
                    if processed.documents:
                        all_new_docs.extend(processed.documents)
                        st.session_state.processed_hashes[file_hash] = uploaded_file.name
                        register_file_version(file_hash, uploaded_file.name, stored_path, "process")
                        new_files_count += 1

                if new_files_count == 0:
                    st.info("No new PDFs to process. Already-indexed files were skipped.")
                    if st.session_state.retriever.has_index:
                        summary_result = st.session_state.agent_controller.tools.summarization_tool(
                            "summarize uploaded pdfs"
                        )
                        st.session_state.latest_pdf_summary = summary_result.content
                        add_summary_version(summary_result.content, "process")
                        save_app_state()
                        st.success("Summary generated from your existing indexed PDFs.")
                        if st.session_state.get("stored_files"):
                            st.caption("Saved files are available in .agent_data/uploads.")
                else:
                    new_chunks = st.session_state.retriever.add_documents(all_new_docs)
                    append_activity("process")
                    st.session_state.retriever.save_to_disk(RETRIEVER_DIR)
                    summary_result = st.session_state.agent_controller.tools.summarization_tool(
                        "summarize uploaded pdfs"
                    )
                    st.session_state.latest_pdf_summary = summary_result.content
                    add_summary_version(summary_result.content, "process")
                    save_app_state()
                    st.success(
                        f"Indexed {new_files_count} new file(s) with {new_chunks} semantic chunks."
                    )
                    st.success("Generated a fresh PDF summary. Open 'Latest PDF Summary' in the sidebar.")
                    st.caption("Uploaded originals were saved in .agent_data/uploads.")
        except Exception as exc:
            st.error(f"Failed to process PDFs: {exc}")

if generate_summary_clicked:
    if not st.session_state.retriever.has_index:
        st.warning("Please upload and process PDFs first.")
    else:
        try:
            with st.spinner("Generating PDF summary from indexed chunks..."):
                summary_result = st.session_state.agent_controller.tools.summarization_tool(
                    "summarize uploaded pdfs"
                )
                st.session_state.latest_pdf_summary = summary_result.content
                add_summary_version(summary_result.content, "manual")
                save_app_state()
            st.success("PDF summary generated. Open 'Latest PDF Summary' in the sidebar.")
        except Exception as exc:
            st.error(f"Failed to generate summary: {exc}")

st.subheader("Study Chat")

if st.session_state.get("latest_pdf_summary"):
    st.markdown("### Latest PDF Summary")
    st.markdown(st.session_state.latest_pdf_summary)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("tool_used"):
            st.caption(f"Tool used: {message['tool_used']}")

        if message.get("confidence") is not None:
            st.caption(f"Confidence: {message['confidence'] * 100:.1f}%")

        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.markdown(
                        f"**{source['source']} (page {source['page']})**  \n"
                        f"confidence: {source['confidence'] * 100:.1f}%  \n"
                        f"> {source['snippet']}"
                    )

        if message.get("reasoning_steps"):
            with st.expander("Agent reasoning trace"):
                for idx, step in enumerate(message["reasoning_steps"], start=1):
                    st.markdown(f"**Step {idx}**")
                    st.markdown(f"- Thought: {step['thought']}")
                    st.markdown(f"- Action: {step['action']}")
                    st.markdown(f"- Observation: {step['observation']}")
                    st.markdown(f"- Reflection: {step['reflection']}")


user_query = st.chat_input("Ask me anything about your uploaded documents...")


def stream_text(text: str) -> Generator[str, None, None]:
    for token in text.split(" "):
        yield token + " "


if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    is_backup_query = st.session_state.agent_controller.is_backup_query(user_query)

    if not st.session_state.retriever.has_index and not is_backup_query:
        warning = "Please upload and process PDFs first."
        st.session_state.messages.append({"role": "assistant", "content": warning})
        with st.chat_message("assistant"):
            st.warning(warning)
    else:
        try:
            with st.spinner("Agent reasoning and selecting tools..."):
                response = st.session_state.agent_controller.run(user_query)

            with st.chat_message("assistant"):
                streamed_text = st.write_stream(stream_text(response.final_answer))
                st.caption(f"Tool used: {response.tool_used}")
                st.caption(f"Confidence: {response.confidence * 100:.1f}%")

                with st.expander("Sources"):
                    for source in response.sources:
                        st.markdown(
                            f"**{source.source} (page {source.page})**  \n"
                            f"confidence: {source.confidence * 100:.1f}%  \n"
                            f"> {source.snippet}"
                        )

                with st.expander("Agent reasoning trace"):
                    for idx, step in enumerate(response.steps, start=1):
                        st.markdown(f"**Step {idx}**")
                        st.markdown(f"- Thought: {step.thought}")
                        st.markdown(f"- Action: {step.action}")
                        st.markdown(f"- Observation: {step.observation}")
                        st.markdown(f"- Reflection: {step.reflection}")

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": streamed_text or response.final_answer,
                    "tool_used": response.tool_used,
                    "confidence": response.confidence,
                    "sources": [
                        {
                            "source": s.source,
                            "page": s.page,
                            "snippet": s.snippet,
                            "confidence": s.confidence,
                        }
                        for s in response.sources
                    ],
                    "reasoning_steps": [
                        {
                            "thought": step.thought,
                            "action": step.action,
                            "observation": step.observation,
                            "reflection": step.reflection,
                        }
                        for step in response.steps
                    ],
                }
            )

            embeddings = get_cached_embeddings("ollama")
            st.session_state.memory_manager.add_long_term_memory(embeddings, "user", user_query)
            st.session_state.memory_manager.add_long_term_memory(
                embeddings, "assistant", response.final_answer
            )
            st.session_state.memory_manager.save_to_disk(MEMORY_DIR)
        except Exception as exc:
            error_message = f"Agent failed to respond: {exc}"
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            with st.chat_message("assistant"):
                st.error(error_message)
