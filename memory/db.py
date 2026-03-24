"""
memory/db.py — DatabaseClient: Fast SQLite WAL storage for history and memory items.

Source of truth for all memory data. Zvec is a secondary index.
"""

import logging
import asyncio
import sqlite3
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Paths
MEMORY_DIR = Path(__file__).parent.parent / "data"
SQLITE_PATH = str(MEMORY_DIR / "agent_session.db")

MEMORY_DIR.mkdir(parents=True, exist_ok=True)


class DatabaseClient:
    """Handles fast, highly-concurrent SQLite storage for history and memory items."""

    def __init__(self, db_path: str = SQLITE_PATH):
        self.db_path = db_path
        self._lock = asyncio.Lock()
        self.initialize()

    def get_fast_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA temp_store = MEMORY;")
        conn.execute("PRAGMA mmap_size = 3000000000;")
        return conn

    def initialize(self):
        """Create tables if they don't exist."""
        with self.get_fast_connection() as conn:
            # Table for chat history
            conn.execute("""
                CREATE TABLE IF NOT EXISTS thread_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_thread_id ON thread_history(thread_id)")

            # Rich memory_items table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_items (
                    id TEXT PRIMARY KEY,
                    kind TEXT CHECK(kind IN ('pref','fact','rule')) NOT NULL,
                    text TEXT NOT NULL,
                    created_ts TEXT NOT NULL,
                    updated_ts TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    source_thread_id TEXT,
                    status TEXT CHECK(status IN ('active','tombstoned')) DEFAULT 'active',
                    supersedes_id TEXT,
                    indexed INTEGER DEFAULT 0
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mi_status ON memory_items(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mi_pending ON memory_items(indexed) WHERE indexed = 0")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mi_kind ON memory_items(kind)")

            # FTS5 virtual table for BM25 keyword search
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                    text, kind,
                    content='memory_items',
                    content_rowid='rowid'
                )
            """)

            # Verify FTS integrity on startup — rebuild if out of sync
            self._ensure_fts_integrity(conn)

    def _ensure_fts_integrity(self, conn):
        """Rebuild FTS index if row counts diverge or table is corrupted."""
        try:
            mi_count = conn.execute("SELECT COUNT(*) FROM memory_items WHERE status = 'active'").fetchone()[0]
            fts_count = conn.execute("SELECT COUNT(*) FROM memory_fts").fetchone()[0]
            if mi_count != fts_count:
                logger.warning(f"⚠️  FTS out of sync ({fts_count} vs {mi_count} active items) — rebuilding...")
                self._rebuild_fts(conn, force_recreate=True)
        except Exception as e:
            logger.warning(f"⚠️  FTS integrity check failed ({e}) — dropping and recreating...")
            self._rebuild_fts(conn, force_recreate=True)

    def _rebuild_fts(self, conn, force_recreate: bool = False):
        """Reindex FTS5 from the memory_items table. Handles corruption via drop+recreate."""
        try:
            if force_recreate:
                conn.execute("DROP TABLE IF EXISTS memory_fts")
                conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                        text, kind,
                        content='memory_items',
                        content_rowid='rowid'
                    )
                """)
            else:
                conn.execute("DELETE FROM memory_fts")
            conn.execute("""
                INSERT INTO memory_fts (rowid, text, kind)
                SELECT rowid, text, kind FROM memory_items WHERE status = 'active'
            """)
            logger.info("✅ FTS5 index rebuilt from memory_items")
        except Exception as e:
            logger.error(f"FTS rebuild failed: {e} — keyword search will be degraded")

    # ── History methods ────────────────────────────────────────

    async def add_history(self, thread_id: str, role: str, content: str):
        async with self._lock:
            def _insert():
                with self.get_fast_connection() as conn:
                    conn.execute(
                        "INSERT INTO thread_history (thread_id, timestamp, role, content) VALUES (?, ?, ?, ?)",
                        (thread_id, datetime.now(timezone.utc).isoformat(), role, content[:1500])
                    )
            await asyncio.to_thread(_insert)

    async def get_recent_history(self, thread_id: str, limit: int = 10) -> List[Dict]:
        def _fetch():
            with self.get_fast_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT role, content FROM thread_history WHERE thread_id = ? ORDER BY timestamp DESC LIMIT ?",
                    (thread_id, limit)
                )
                rows = [dict(row) for row in cursor.fetchall()]
                return list(reversed(rows))
        return await asyncio.to_thread(_fetch)

    async def clear_history(self, thread_id: str):
        """Clear the history for a given thread."""
        async with self._lock:
            def _clear():
                with self.get_fast_connection() as conn:
                    conn.execute("DELETE FROM thread_history WHERE thread_id = ?", (thread_id,))
            await asyncio.to_thread(_clear)

    # ── Memory item methods ─────────────────────────────────────

    async def item_exists_by_text(self, text: str) -> Optional[str]:
        """Check if an active memory item with this exact text exists."""
        def _fetch():
            with self.get_fast_connection() as conn:
                cursor = conn.execute(
                    "SELECT id FROM memory_items WHERE text = ? AND status = 'active'",
                    (text,)
                )
                row = cursor.fetchone()
                return row[0] if row else None
        return await asyncio.to_thread(_fetch)

    async def get_active_items_by_ids(self, item_ids: List[str]) -> List[Dict]:
        """Retrieve active memory items by their IDs, including timestamps for time-decay."""
        if not item_ids:
            return []
        def _fetch():
            with self.get_fast_connection() as conn:
                conn.row_factory = sqlite3.Row
                placeholders = ",".join("?" * len(item_ids))
                cursor = conn.execute(
                    f"SELECT id, kind, text, updated_ts FROM memory_items WHERE id IN ({placeholders}) AND status = 'active'",
                    item_ids
                )
                return [dict(r) for r in cursor.fetchall()]
        return await asyncio.to_thread(_fetch)

    async def tombstone_by_content(self, contents: List[str]):
        """Tombstone (soft-delete) memory items matching content."""
        if not contents:
            return
        async with self._lock:
            def _tombstone():
                now = datetime.now(timezone.utc).isoformat()
                with self.get_fast_connection() as conn:
                    for content in contents:
                        conn.execute(
                            "UPDATE memory_items SET status = 'tombstoned', updated_ts = ? WHERE text = ? AND status = 'active'",
                            (now, content)
                        )
            await asyncio.to_thread(_tombstone)

    async def touch_item(self, item_id: str):
        """Update updated_ts of an existing item."""
        async with self._lock:
            def _touch():
                now = datetime.now(timezone.utc).isoformat()
                with self.get_fast_connection() as conn:
                    conn.execute("UPDATE memory_items SET updated_ts = ? WHERE id = ?", (now, item_id))
            await asyncio.to_thread(_touch)

    async def update_item_text(self, item_id: str, new_text: str):
        """Update both the text and updated_ts of an existing memory item (merge/evolve)."""
        async with self._lock:
            def _update():
                now = datetime.now(timezone.utc).isoformat()
                with self.get_fast_connection() as conn:
                    conn.execute(
                        "UPDATE memory_items SET text = ?, updated_ts = ?, indexed = 0 WHERE id = ?",
                        (new_text, now, item_id)
                    )
                    # Re-sync FTS for this item
                    conn.execute(
                        "DELETE FROM memory_fts WHERE rowid = (SELECT rowid FROM memory_items WHERE id = ?)",
                        (item_id,)
                    )
                    conn.execute(
                        "INSERT INTO memory_fts (rowid, text, kind) SELECT rowid, text, kind FROM memory_items WHERE id = ?",
                        (item_id,)
                    )
            await asyncio.to_thread(_update)

    async def insert_memory_item(self, item_id: str, kind: str, text: str, source_thread_id: str = None):
        """Insert a new memory item and sync FTS."""
        async with self._lock:
            def _insert():
                now = datetime.now(timezone.utc).isoformat()
                with self.get_fast_connection() as conn:
                    conn.execute(
                        "INSERT OR IGNORE INTO memory_items (id, kind, text, created_ts, updated_ts, source_thread_id, indexed) VALUES (?, ?, ?, ?, ?, ?, 0)",
                        (item_id, kind, text, now, now, source_thread_id)
                    )
                    # Sync FTS index (safe: only inserts if item exists)
                    conn.execute(
                        "INSERT INTO memory_fts (rowid, text, kind) SELECT rowid, text, kind FROM memory_items WHERE id = ?",
                        (item_id,)
                    )
            await asyncio.to_thread(_insert)

    async def search_fts(self, query: str, limit: int = 20) -> List[Dict]:
        """BM25 keyword search over memory_items via FTS5."""
        def _search():
            with self.get_fast_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT mi.id, mi.kind, mi.text, mi.updated_ts, rank
                    FROM memory_fts fts
                    JOIN memory_items mi ON mi.rowid = fts.rowid
                    WHERE memory_fts MATCH ?
                    AND mi.status = 'active'
                    ORDER BY rank
                    LIMIT ?
                """, (query, limit))
                return [dict(r) for r in cursor.fetchall()]
        try:
            return await asyncio.to_thread(_search)
        except Exception as e:
            logger.debug(f"FTS search failed (likely empty or bad query): {e}")
            return []

    async def fetch_pending_items(self, batch_size: int = 256) -> List[Dict]:
        """Fetch memory items not yet indexed in Zvec."""
        def _fetch():
            with self.get_fast_connection() as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT id, text FROM memory_items WHERE indexed = 0 AND status = 'active' LIMIT ?",
                    (batch_size,)
                ).fetchall()
                return [dict(r) for r in rows]
        return await asyncio.to_thread(_fetch)

    async def mark_items_indexed(self, item_ids: List[str]):
        """Mark memory items as indexed in Zvec."""
        if not item_ids:
            return
        def _mark():
            with self.get_fast_connection() as conn:
                placeholders = ",".join("?" * len(item_ids))
                conn.execute(f"UPDATE memory_items SET indexed = 1 WHERE id IN ({placeholders})", item_ids)
        await asyncio.to_thread(_mark)

    # ── Introspection methods ─────────────────────────────────

    async def get_all_active_items(self) -> List[Dict]:
        """Return all active memory items for debugging/admin."""
        def _fetch():
            with self.get_fast_connection() as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT id, kind, text, created_ts, updated_ts, confidence FROM memory_items WHERE status = 'active' ORDER BY updated_ts DESC"
                ).fetchall()
                return [dict(r) for r in rows]
        return await asyncio.to_thread(_fetch)

    async def get_stats(self) -> Dict:
        """Return memory statistics: counts by kind, status, indexed state."""
        def _fetch():
            with self.get_fast_connection() as conn:
                stats = {}
                # By kind
                for kind in ('pref', 'fact', 'rule'):
                    count = conn.execute(
                        "SELECT COUNT(*) FROM memory_items WHERE kind = ? AND status = 'active'", (kind,)
                    ).fetchone()[0]
                    stats[f"active_{kind}"] = count
                # Totals
                stats["total_active"] = conn.execute(
                    "SELECT COUNT(*) FROM memory_items WHERE status = 'active'"
                ).fetchone()[0]
                stats["total_tombstoned"] = conn.execute(
                    "SELECT COUNT(*) FROM memory_items WHERE status = 'tombstoned'"
                ).fetchone()[0]
                stats["pending_index"] = conn.execute(
                    "SELECT COUNT(*) FROM memory_items WHERE indexed = 0 AND status = 'active'"
                ).fetchone()[0]
                return stats
        return await asyncio.to_thread(_fetch)
