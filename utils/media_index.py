"""SQLite-backed media library index used by DLNA Browse.

The index keeps expensive ffprobe metadata across process restarts while each
Browse still performs a light stat pass over the requested directory. That pass
detects files copied into, removed from, or modified inside an already-open
folder and only re-probes changed video files.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import config
from utils.logger import get
from utils.mkv_cues import MkvCuesInfo, probe_mkv_cues
from utils.offline_outputs import has_offline_passthrough_output, is_offline_passthrough_output_name
from utils.video_metadata import probe_video_metadata, select_backend

log = get("media_index")

SCHEMA_VERSION = 2
PROBE_ERROR_RETRY_SEC = 60.0
SQLITE_TIMEOUT_SEC = 10.0
LARGE_DIRECTORY_WARN_CHILDREN = 5000
NEW_FILE_PROBE_GRACE_SEC = 2.0
PENDING_PROBE_RETRY_SEC = 2.0
PROBE_FAILURE_LOG_LIMIT_PER_SCAN = 5
SI_SIDECAR_SUFFIX = ".si.wav"
SI_SIDECAR_SOURCE_EXT = ".mp4"


class MediaIndexSchemaError(RuntimeError):
    """Raised when the on-disk index cannot be used by this code version."""


@dataclass(frozen=True)
class IndexedVideo:
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    codec_name: str = ""
    pix_fmt: str = ""
    backend_verdict: str = ""
    backend_reason: str = ""
    probe_error: str = ""
    mkv_cues_status: str = ""
    mkv_cues_position: int = -1
    mkv_cues_reason: str = ""

    @property
    def mkv_needs_fix(self) -> bool:
        return self.mkv_cues_status in {"tail", "missing", "unknown"}

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}" if self.width > 0 and self.height > 0 else ""


@dataclass(frozen=True)
class IndexedChild:
    path: Path
    key: str
    parent_key: str
    name: str
    is_dir: bool
    size: int
    mtime_ns: int
    video: IndexedVideo | None = None


@dataclass(frozen=True)
class DirectorySnapshot:
    path: Path
    key: str
    signature: str
    children: tuple[IndexedChild, ...]


def _now() -> float:
    return time.time()


def _dir_key(path: Path) -> str:
    return config.MEDIA_LIBRARY.path_to_key(path)


def _signature(parts: list[str]) -> str:
    raw = "\n".join(parts).encode("utf-8", "surrogatepass")
    return hashlib.sha1(raw).hexdigest()


def _si_sidecar_source_name(name: str) -> str:
    if not name.lower().endswith(SI_SIDECAR_SUFFIX):
        return ""
    return f"{name[:-len(SI_SIDECAR_SUFFIX)]}{SI_SIDECAR_SOURCE_EXT}"


class MediaIndex:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = (db_path or config.LIBRARY_INDEX_DB).resolve()
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._scan_probe_failures = 0

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                self._conn = self._open_connection()
                self._ensure_schema(self._conn)
            except (sqlite3.DatabaseError, MediaIndexSchemaError) as e:
                log.warning("library index unusable, rebuilding %s: %s", self.db_path, e)
                self._rebuild_database(e)
                self._conn = self._open_connection()
                self._ensure_schema(self._conn)
        return self._conn

    def _open_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=SQLITE_TIMEOUT_SEC, check_same_thread=False)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(f"PRAGMA busy_timeout={int(SQLITE_TIMEOUT_SEC * 1000)}")
            return conn
        except sqlite3.DatabaseError:
            conn.close()
            raise

    def _rebuild_database(self, reason: Exception) -> None:
        self.close()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for path in (self.db_path, Path(f"{self.db_path}-wal"), Path(f"{self.db_path}-shm")):
            if not path.exists():
                continue
            backup = path.with_name(f"{path.name}.corrupt_{stamp}")
            try:
                os.replace(path, backup)
            except OSError as e:
                log.warning("failed to move unusable index artifact %s: %s", path, e)
                try:
                    path.unlink()
                except OSError:
                    pass
        log.warning("library index rebuilt after unusable database: %s", reason)

    def close(self) -> None:
        if self._conn is None:
            return
        try:
            self._conn.close()
        except sqlite3.DatabaseError:
            pass
        finally:
            self._conn = None

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        current_version = self._schema_version(conn)
        if current_version > SCHEMA_VERSION:
            raise MediaIndexSchemaError(
                f"index schema version {current_version} is newer than supported {SCHEMA_VERSION}"
            )
        if current_version < SCHEMA_VERSION:
            self._migrate_schema(conn, current_version, SCHEMA_VERSION)
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS entries (
                path_key TEXT PRIMARY KEY,
                parent_key TEXT NOT NULL,
                name TEXT NOT NULL,
                is_dir INTEGER NOT NULL,
                suffix TEXT NOT NULL,
                size INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_entries_parent_name
                ON entries(parent_key, is_dir DESC, name COLLATE NOCASE);
            CREATE TABLE IF NOT EXISTS video_metadata (
                path_key TEXT PRIMARY KEY,
                size INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL,
                duration REAL NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                fps REAL NOT NULL,
                codec_name TEXT NOT NULL,
                pix_fmt TEXT NOT NULL,
                backend_verdict TEXT NOT NULL,
                backend_reason TEXT NOT NULL,
                probe_error TEXT NOT NULL,
                mkv_cues_status TEXT NOT NULL DEFAULT '',
                mkv_cues_position INTEGER NOT NULL DEFAULT -1,
                mkv_cues_reason TEXT NOT NULL DEFAULT '',
                probed_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS directory_state (
                path_key TEXT PRIMARY KEY,
                signature TEXT NOT NULL,
                child_count INTEGER NOT NULL,
                scanned_at REAL NOT NULL
            );
            """
        )
        self._validate_schema(conn)
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        conn.commit()

    def _migrate_schema(self, conn: sqlite3.Connection, from_version: int, to_version: int) -> None:
        """Migrate older cache schemas in place when possible.

        Keep cache migrations additive so deployed libraries do not need to be
        rebuilt for metadata fields that can be recomputed lazily.
        """
        if from_version <= 0:
            return
        if from_version < 2 <= to_version:
            columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(video_metadata)")}
            if "mkv_cues_status" not in columns:
                conn.execute("ALTER TABLE video_metadata ADD COLUMN mkv_cues_status TEXT NOT NULL DEFAULT ''")
            if "mkv_cues_position" not in columns:
                conn.execute("ALTER TABLE video_metadata ADD COLUMN mkv_cues_position INTEGER NOT NULL DEFAULT -1")
            if "mkv_cues_reason" not in columns:
                conn.execute("ALTER TABLE video_metadata ADD COLUMN mkv_cues_reason TEXT NOT NULL DEFAULT ''")
        if from_version == to_version:
            return
        if to_version <= 2:
            return
        raise MediaIndexSchemaError(f"no migration path from schema {from_version} to {to_version}")

    def _schema_version(self, conn: sqlite3.Connection) -> int:
        user_version = 0
        try:
            user_version = int(conn.execute("PRAGMA user_version").fetchone()[0] or 0)
        except (sqlite3.DatabaseError, TypeError, ValueError):
            user_version = 0
        try:
            row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        except sqlite3.DatabaseError:
            row = None
        meta_version = 0
        if row is not None:
            try:
                meta_version = int(row["value"])
            except (TypeError, ValueError, KeyError, IndexError):
                meta_version = 0
        return max(user_version, meta_version)

    def _validate_schema(self, conn: sqlite3.Connection) -> None:
        required = {
            "meta": {"key", "value"},
            "entries": {"path_key", "parent_key", "name", "is_dir", "suffix", "size", "mtime_ns", "updated_at"},
            "video_metadata": {
                "path_key",
                "size",
                "mtime_ns",
                "duration",
                "width",
                "height",
                "fps",
                "codec_name",
                "pix_fmt",
                "backend_verdict",
                "backend_reason",
                "probe_error",
                "mkv_cues_status",
                "mkv_cues_position",
                "mkv_cues_reason",
                "probed_at",
            },
            "directory_state": {"path_key", "signature", "child_count", "scanned_at"},
        }
        for table, columns in required.items():
            found = {str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})")}
            missing = columns - found
            if missing:
                raise MediaIndexSchemaError(f"table {table} missing columns: {', '.join(sorted(missing))}")

    def list_directory(self, directory: Path) -> DirectorySnapshot:
        try:
            return self._list_directory_once(directory)
        except (sqlite3.DatabaseError, MediaIndexSchemaError) as e:
            log.warning("library index operation failed, rebuilding and retrying: %s", e)
            with self._lock:
                self._rebuild_database(e)
            return self._list_directory_once(directory)

    def _list_directory_once(self, directory: Path) -> DirectorySnapshot:
        directory = directory.resolve()
        try:
            key = _dir_key(directory)
        except ValueError as e:
            log.warning("index rejected directory outside media roots: %s", e)
            return DirectorySnapshot(directory, "", "outside-root", ())
        rows: list[tuple[Path, str, str, bool, int, int]] = []
        signature_parts: list[str] = []
        try:
            child_iter = list(directory.iterdir())
            children = sorted(child_iter, key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError as e:
            log.warning("index list %s failed: %s", directory, e)
            return DirectorySnapshot(directory, key, "error", ())
        if len(children) >= LARGE_DIRECTORY_WARN_CHILDREN:
            log.info("large media directory scan: path=%s entries=%d", directory, len(children))
        child_names = {child.name.lower() for child in children}

        for child in children:
            try:
                is_dir = child.is_dir()
                suffix = child.suffix.lower()
                is_file = child.is_file()
                is_video = is_file and suffix in config.VIDEO_EXTS
                is_image = is_file and config.DLNA_IMAGE_ENABLED and suffix in config.IMAGE_EXTS
                si_source_name = _si_sidecar_source_name(child.name).lower()
                is_si_sidecar = is_file and bool(si_source_name) and si_source_name in child_names
                if not is_dir and not is_video and not is_image and not is_si_sidecar:
                    continue
                st = child.stat()
                if is_si_sidecar and not is_video and not is_image:
                    signature_parts.append(f"{child.name}|si-sidecar|{int(st.st_size)}|{int(st.st_mtime_ns)}")
                    continue
                child_key = _dir_key(child)
            except (OSError, ValueError) as e:
                log.debug("index skip %s: %s", child, e)
                continue
            size = int(st.st_size if not is_dir else 0)
            mtime_ns = int(st.st_mtime_ns)
            signature_parts.append(f"{child.name}|{1 if is_dir else 0}|{size}|{mtime_ns}")
            rows.append((child, child_key, child.suffix.lower(), is_dir, size, mtime_ns))

        sig = _signature(signature_parts)
        with self._lock:
            conn = self._connect()
            now = _now()
            self._scan_probe_failures = 0
            old = conn.execute("SELECT signature FROM directory_state WHERE path_key=?", (key,)).fetchone()
            if old is not None and old["signature"] == sig:
                cached = self._children_from_db(conn, rows, key)
                if cached is not None:
                    return DirectorySnapshot(directory, key, sig, tuple(cached))

            children_out: list[IndexedChild] = []
            current_keys = [row[1] for row in rows]
            self._delete_stale_entries(conn, key, current_keys)

            for child, child_key, suffix, is_dir, size, mtime_ns in rows:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO entries
                    (path_key, parent_key, name, is_dir, suffix, size, mtime_ns, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (child_key, key, child.name, 1 if is_dir else 0, suffix, size, mtime_ns, now),
                )
                video = self._video_for(conn, child, child_key, size, mtime_ns) if (not is_dir and suffix in config.VIDEO_EXTS) else None
                children_out.append(
                    IndexedChild(
                        path=child,
                        key=child_key,
                        parent_key=key,
                        name=child.name,
                        is_dir=is_dir,
                        size=size,
                        mtime_ns=mtime_ns,
                        video=video,
                    )
                )

            conn.execute(
                """
                INSERT OR REPLACE INTO directory_state(path_key, signature, child_count, scanned_at)
                VALUES (?, ?, ?, ?)
                """,
                (key, sig, len(children_out), now),
            )
            conn.commit()
            self._scan_probe_failures = 0
            return DirectorySnapshot(directory, key, sig, tuple(children_out))

    def _delete_stale_entries(self, conn: sqlite3.Connection, parent_key: str, current_keys: list[str]) -> None:
        conn.execute("CREATE TEMP TABLE IF NOT EXISTS current_directory_keys(path_key TEXT PRIMARY KEY)")
        conn.execute("DELETE FROM current_directory_keys")
        if current_keys:
            conn.executemany(
                "INSERT OR IGNORE INTO current_directory_keys(path_key) VALUES (?)",
                ((path_key,) for path_key in current_keys),
            )
        conn.execute(
            """
            DELETE FROM entries
            WHERE parent_key=?
              AND NOT EXISTS (
                  SELECT 1 FROM current_directory_keys
                  WHERE current_directory_keys.path_key = entries.path_key
              )
            """,
            (parent_key,),
        )

    def _children_from_db(
        self,
        conn: sqlite3.Connection,
        rows: list[tuple[Path, str, str, bool, int, int]],
        parent_key: str,
    ) -> list[IndexedChild] | None:
        out: list[IndexedChild] = []
        for child, child_key, suffix, is_dir, size, mtime_ns in rows:
            entry = conn.execute(
                """
                SELECT size, mtime_ns FROM entries
                WHERE path_key=? AND parent_key=? AND is_dir=?
                """,
                (child_key, parent_key, 1 if is_dir else 0),
            ).fetchone()
            if entry is None or int(entry["size"]) != size or int(entry["mtime_ns"]) != mtime_ns:
                return None
            video = None
            if not is_dir and suffix in config.VIDEO_EXTS:
                video = self._video_from_db(conn, child_key, size, mtime_ns)
                if video is None:
                    return None
                if child.suffix.lower() == ".mkv" and not video.mkv_cues_status:
                    return None
            out.append(
                IndexedChild(
                    path=child,
                    key=child_key,
                    parent_key=parent_key,
                    name=child.name,
                    is_dir=is_dir,
                    size=size,
                    mtime_ns=mtime_ns,
                    video=video,
                )
            )
        return out

    def _video_from_db(
        self,
        conn: sqlite3.Connection,
        key: str,
        size: int,
        mtime_ns: int,
    ) -> IndexedVideo | None:
        row = conn.execute(
            "SELECT * FROM video_metadata WHERE path_key=? AND size=? AND mtime_ns=?",
            (key, size, mtime_ns),
        ).fetchone()
        if row is None:
            return None
        probe_error = str(row["probe_error"] or "")
        if probe_error:
            age = _now() - float(row["probed_at"] or 0.0)
            retry_sec = PENDING_PROBE_RETRY_SEC if probe_error.startswith("pending:") else PROBE_ERROR_RETRY_SEC
            if age >= retry_sec:
                return None
        return IndexedVideo(
            duration=float(row["duration"]),
            width=int(row["width"]),
            height=int(row["height"]),
            fps=float(row["fps"]),
            codec_name=str(row["codec_name"] or ""),
            pix_fmt=str(row["pix_fmt"] or ""),
            backend_verdict=str(row["backend_verdict"] or ""),
            backend_reason=str(row["backend_reason"] or ""),
            probe_error=probe_error,
            mkv_cues_status=str(row["mkv_cues_status"] or ""),
            mkv_cues_position=int(row["mkv_cues_position"] or -1),
            mkv_cues_reason=str(row["mkv_cues_reason"] or ""),
        )

    def _video_for(
        self,
        conn: sqlite3.Connection,
        path: Path,
        key: str,
        size: int,
        mtime_ns: int,
    ) -> IndexedVideo:
        cached = self._video_from_db(conn, key, size, mtime_ns)
        if cached is not None:
            return cached

        now = _now()
        age = max(0.0, now - (mtime_ns / 1_000_000_000.0))
        if age < NEW_FILE_PROBE_GRACE_SEC:
            return self._store_probe_error(
                conn,
                key,
                size,
                mtime_ns,
                f"pending: file modified {age:.2f}s ago",
                now,
            )
        try:
            meta = probe_video_metadata(path)
            backend = select_backend(meta.timing, meta.codec, meta.color)
            mkv_cues = probe_mkv_cues(path)
            video = IndexedVideo(
                duration=float(meta.timing.duration),
                width=int(meta.codec.width),
                height=int(meta.codec.height),
                fps=float(meta.timing.source_fps),
                codec_name=str(meta.codec.codec_name or ""),
                pix_fmt=str(meta.codec.pix_fmt or ""),
                backend_verdict=backend.verdict,
                backend_reason=backend.reason,
                probe_error="",
                mkv_cues_status=mkv_cues.status,
                mkv_cues_position=mkv_cues.position,
                mkv_cues_reason=mkv_cues.reason,
            )
        except Exception as e:
            if self._scan_probe_failures < PROBE_FAILURE_LOG_LIMIT_PER_SCAN:
                log.warning("index probe %s failed: %s", key, e)
            elif self._scan_probe_failures == PROBE_FAILURE_LOG_LIMIT_PER_SCAN:
                log.warning("additional index probe failures suppressed for this directory scan")
            self._scan_probe_failures += 1
            return self._store_probe_error(
                conn,
                key,
                size,
                mtime_ns,
                f"{type(e).__name__}: {str(e).splitlines()[0][:300]}",
                now,
            )

        conn.execute(
            """
            INSERT OR REPLACE INTO video_metadata
            (path_key, size, mtime_ns, duration, width, height, fps, codec_name,
             pix_fmt, backend_verdict, backend_reason, probe_error,
             mkv_cues_status, mkv_cues_position, mkv_cues_reason, probed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                key,
                size,
                mtime_ns,
                video.duration,
                video.width,
                video.height,
                video.fps,
                video.codec_name,
                video.pix_fmt,
                video.backend_verdict,
                video.backend_reason,
                video.probe_error,
                video.mkv_cues_status,
                video.mkv_cues_position,
                video.mkv_cues_reason,
                now,
            ),
        )
        return video

    def _store_probe_error(
        self,
        conn: sqlite3.Connection,
        key: str,
        size: int,
        mtime_ns: int,
        error: str,
        now: float,
    ) -> IndexedVideo:
        video = IndexedVideo(probe_error=error)
        conn.execute(
            """
            INSERT OR REPLACE INTO video_metadata
            (path_key, size, mtime_ns, duration, width, height, fps, codec_name,
             pix_fmt, backend_verdict, backend_reason, probe_error,
             mkv_cues_status, mkv_cues_position, mkv_cues_reason, probed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (key, size, mtime_ns, 0.0, 0, 0, 0.0, "", "", "", "", video.probe_error, "", -1, "", now),
        )
        return video

    def child_count(self, directory: Path) -> int:
        count = 0
        try:
            children = list(directory.iterdir())
        except OSError as e:
            log.warning("index child count %s failed: %s", directory, e)
            return 0
        for child in children:
            try:
                if child.is_dir():
                    count += 1
                elif child.is_file() and child.suffix.lower() in config.VIDEO_EXTS:
                    if (
                        is_offline_passthrough_output_name(child.name)
                        or config.PASSTHROUGH_OUTPUT_MODE == "none"
                        or has_offline_passthrough_output(child, children)
                        or self._hide_passthrough_for_path(child)
                    ):
                        count += 1
                    else:
                        count += 3 if config.PASSTHROUGH_OUTPUT_MODE == "all" else 2
                elif child.is_file() and config.DLNA_IMAGE_ENABLED and child.suffix.lower() in config.IMAGE_EXTS:
                    count += 1
            except OSError:
                continue
        return count

    def _hide_passthrough_for_path(self, path: Path) -> bool:
        if path.suffix.lower() != ".mkv":
            return False
        policy = config.PASSTHROUGH_MKV_LIVE_POLICY
        if policy == "block":
            return True
        if policy == "allow":
            return False
        try:
            st = path.stat()
            key = _dir_key(path)
            size = int(st.st_size)
            mtime_ns = int(st.st_mtime_ns)
        except (OSError, ValueError):
            return False
        with self._lock:
            conn = self._connect()
            cached = self._video_from_db(conn, key, size, mtime_ns)
            if cached is not None and cached.mkv_cues_status:
                return cached.mkv_needs_fix
            info = probe_mkv_cues(path)
            self._store_mkv_cues_info(conn, key, size, mtime_ns, info)
            conn.commit()
            return info.needs_fix

    def _store_mkv_cues_info(
        self,
        conn: sqlite3.Connection,
        key: str,
        size: int,
        mtime_ns: int,
        info: MkvCuesInfo,
    ) -> None:
        row = conn.execute(
            "SELECT path_key FROM video_metadata WHERE path_key=? AND size=? AND mtime_ns=?",
            (key, size, mtime_ns),
        ).fetchone()
        if row is None:
            return
        conn.execute(
            """
            UPDATE video_metadata
            SET mkv_cues_status=?, mkv_cues_position=?, mkv_cues_reason=?
            WHERE path_key=? AND size=? AND mtime_ns=?
            """,
            (info.status, info.position, info.reason, key, size, mtime_ns),
        )


_INDEX = MediaIndex()


def get_media_index() -> MediaIndex:
    return _INDEX
