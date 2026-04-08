from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Optional

try:
    import mysql.connector
except Exception:  # pragma: no cover
    mysql = None


@dataclass
class DbConfig:
    provider: str
    host: str
    port: int
    database: str
    user: str
    password: str


class BackupDbStore:
    def __init__(self, config: DbConfig) -> None:
        self.config = config

    @classmethod
    def from_env(cls) -> "BackupDbStore":
        return cls(
            DbConfig(
                provider=os.getenv("DB_PROVIDER", "").strip().lower(),
                host=os.getenv("MYSQL_HOST", "localhost"),
                port=int(os.getenv("MYSQL_PORT", "3306")),
                database=os.getenv("MYSQL_DATABASE", "ai_study_backup"),
                user=os.getenv("MYSQL_USER", ""),
                password=os.getenv("MYSQL_PASSWORD", ""),
            )
        )

    @property
    def enabled(self) -> bool:
        return (
            self.config.provider == "mysql"
            and bool(self.config.user)
            and bool(self.config.password)
            and mysql is not None
        )

    def _connect(self, include_database: bool = True):
        if mysql is None:
            raise RuntimeError("mysql-connector-python is not installed.")

        kwargs = {
            "host": self.config.host,
            "port": self.config.port,
            "user": self.config.user,
            "password": self.config.password,
            "autocommit": True,
        }
        if include_database:
            kwargs["database"] = self.config.database
        return mysql.connector.connect(**kwargs)

    def ensure_schema(self) -> None:
        if not self.enabled:
            return

        with self._connect(include_database=False) as conn:
            with conn.cursor() as cur:
                cur.execute(f"CREATE DATABASE IF NOT EXISTS `{self.config.database}`")

        with self._connect(include_database=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS files (
                        id BIGINT AUTO_INCREMENT PRIMARY KEY,
                        file_name VARCHAR(255) NOT NULL,
                        file_hash CHAR(64) NOT NULL UNIQUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS file_versions (
                        id BIGINT AUTO_INCREMENT PRIMARY KEY,
                        file_id BIGINT NOT NULL,
                        version_number INT NOT NULL,
                        file_path TEXT NOT NULL,
                        file_size BIGINT NOT NULL DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE KEY uq_file_version (file_id, version_number),
                        CONSTRAINT fk_file_versions_file
                            FOREIGN KEY (file_id) REFERENCES files(id)
                            ON DELETE CASCADE
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS backup_logs (
                        id BIGINT AUTO_INCREMENT PRIMARY KEY,
                        file_id BIGINT NOT NULL,
                        action VARCHAR(50) NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        CONSTRAINT fk_backup_logs_file
                            FOREIGN KEY (file_id) REFERENCES files(id)
                            ON DELETE CASCADE
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS file_usage (
                        id BIGINT AUTO_INCREMENT PRIMARY KEY,
                        file_id BIGINT NOT NULL,
                        access_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        CONSTRAINT fk_file_usage_file
                            FOREIGN KEY (file_id) REFERENCES files(id)
                            ON DELETE CASCADE
                    )
                    """
                )

    def upsert_file(self, file_name: str, file_hash: str) -> Optional[int]:
        if not self.enabled:
            return None

        with self._connect(include_database=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO files (file_name, file_hash)
                    VALUES (%s, %s)
                    ON DUPLICATE KEY UPDATE file_name = VALUES(file_name)
                    """,
                    (file_name, file_hash),
                )
                cur.execute("SELECT id FROM files WHERE file_hash = %s", (file_hash,))
                row = cur.fetchone()
                return int(row[0]) if row else None

    def add_file_version(
        self,
        file_id: int,
        version_number: int,
        file_path: str,
        file_size: int,
    ) -> None:
        if not self.enabled:
            return

        with self._connect(include_database=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO file_versions (file_id, version_number, file_path, file_size)
                    VALUES (%s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        file_path = VALUES(file_path),
                        file_size = VALUES(file_size)
                    """,
                    (file_id, version_number, file_path, file_size),
                )

    def log_action(self, file_id: int, action: str) -> None:
        if not self.enabled:
            return

        with self._connect(include_database=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO backup_logs (file_id, action) VALUES (%s, %s)",
                    (file_id, action),
                )

    def log_usage(self, file_id: int) -> None:
        if not self.enabled:
            return

        with self._connect(include_database=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO file_usage (file_id) VALUES (%s)",
                    (file_id,),
                )

    def list_files(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self.enabled:
            return []

        with self._connect(include_database=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, file_name, file_hash, created_at
                    FROM files
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cur.fetchall()

        return [
            {
                "id": int(row[0]),
                "file_name": str(row[1]),
                "file_hash": str(row[2]),
                "created_at": str(row[3]),
            }
            for row in rows
        ]

    def list_file_versions(self, limit: int = 100) -> list[dict[str, Any]]:
        if not self.enabled:
            return []

        with self._connect(include_database=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT f.file_name, f.file_hash, fv.version_number, fv.file_path, fv.file_size, fv.created_at
                    FROM file_versions fv
                    JOIN files f ON f.id = fv.file_id
                    ORDER BY fv.created_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cur.fetchall()

        return [
            {
                "file_name": str(row[0]),
                "file_hash": str(row[1]),
                "version_number": int(row[2]),
                "file_path": str(row[3]),
                "file_size": int(row[4]),
                "created_at": str(row[5]),
            }
            for row in rows
        ]

    def duplicate_name_candidates(self) -> list[str]:
        if not self.enabled:
            return []

        with self._connect(include_database=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT file_name
                    FROM files
                    GROUP BY file_name
                    HAVING COUNT(*) > 1
                    ORDER BY file_name ASC
                    """
                )
                rows = cur.fetchall()

        return [str(row[0]) for row in rows]

    def usage_count_last_days(self, days: int = 7) -> int:
        if not self.enabled:
            return 0

        with self._connect(include_database=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COUNT(*)
                    FROM file_usage
                    WHERE access_time >= (UTC_TIMESTAMP() - INTERVAL %s DAY)
                    """,
                    (days,),
                )
                row = cur.fetchone()

        return int(row[0]) if row and row[0] is not None else 0

    def resolve_file(self, hint: str | None = None) -> Optional[dict[str, Any]]:
        if not self.enabled:
            return None

        with self._connect(include_database=True) as conn:
            with conn.cursor() as cur:
                if hint:
                    cur.execute(
                        """
                        SELECT id, file_name, file_hash, created_at
                        FROM files
                        WHERE LOWER(file_name) LIKE LOWER(%s)
                        ORDER BY created_at DESC
                        LIMIT 1
                        """,
                        (f"%{hint}%",),
                    )
                else:
                    cur.execute(
                        """
                        SELECT id, file_name, file_hash, created_at
                        FROM files
                        ORDER BY created_at DESC
                        LIMIT 1
                        """
                    )
                row = cur.fetchone()

        if not row:
            return None
        return {
            "id": int(row[0]),
            "file_name": str(row[1]),
            "file_hash": str(row[2]),
            "created_at": str(row[3]),
        }

    def list_versions_for_file(self, file_id: int) -> list[dict[str, Any]]:
        if not self.enabled:
            return []

        with self._connect(include_database=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT version_number, file_path, file_size, created_at
                    FROM file_versions
                    WHERE file_id = %s
                    ORDER BY version_number ASC
                    """,
                    (file_id,),
                )
                rows = cur.fetchall()

        return [
            {
                "version_number": int(row[0]),
                "file_path": str(row[1]),
                "file_size": int(row[2]),
                "created_at": str(row[3]),
            }
            for row in rows
        ]

    def version_event_count_last_days(self, days: int = 7) -> int:
        if not self.enabled:
            return 0

        with self._connect(include_database=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COUNT(*)
                    FROM file_versions
                    WHERE created_at >= (UTC_TIMESTAMP() - INTERVAL %s DAY)
                    """,
                    (days,),
                )
                row = cur.fetchone()

        return int(row[0]) if row and row[0] is not None else 0

    @staticmethod
    def extract_version_number(query: str) -> Optional[int]:
        match = re.search(r"\bv\s*(\d+)\b|\bversion\s+(\d+)\b", query.lower())
        if not match:
            return None
        candidate = match.group(1) or match.group(2)
        try:
            return int(candidate)
        except (TypeError, ValueError):
            return None
