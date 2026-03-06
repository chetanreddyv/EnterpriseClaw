"""
memory.py — MemoryGate background worker with Zvec and SQLite.

Production Upgrades: Async I/O, context-aware deduplication,
fast SQLite WAL session storage, and semantic retrieval 
using Zvec (in-process vector DB).
"""

import logging
import asyncio
import uuid
import sqlite3
from typing import List, Dict
from pathlib import Path
from datetime import datetime, timezone

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
import zvec

logger = logging.getLogger(__name__)

# Paths
MEMORY_DIR = Path(__file__).parent / "data"
ZVEC_PATH = str(MEMORY_DIR / "zvec_index")
ZVEC_SKILLS_PATH = str(MEMORY_DIR / "zvec_skills")
SQLITE_PATH = str(MEMORY_DIR / "agent_session.db")
SKILLS_DIR = Path(__file__).parent / "skills"

MEMORY_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# 1. Extraction Schema & Agent (Gemini)
# ==========================================================

class ExtractedMemory(BaseModel):
    preferences: List[str] = Field(default_factory=list, description="Writing style, tone, format rules. These are 'pref' kind.")
    facts: List[str] = Field(default_factory=list, description="Hard facts: names, roles, constraints, locations. These are 'fact' kind.")
    corrections: List[str] = Field(default_factory=list, description="Explicit rules where user corrected the AI. These are 'rule' kind.")
    obsolete_items: List[str] = Field(default_factory=list, description="Existing memories that are no longer true — will be tombstoned.")
    important: bool = Field(default=False, description="True ONLY if new, actionable information was discovered.")


EXTRACTION_PROMPT = """You are an advanced memory extraction system. 
Analyze the conversation between the user and the AI.

Your task: Extract NEW, notable, and actionable information about the user.

**Existing Relevant Memory Context (DO NOT EXTRACT AGAIN):**
{existing_memory}

**Rules:**
1. DO NOT extract information that is already in the Existing Memory.
2. If the user contradicts an Existing Memory (e.g., they moved to a new city), extract the new fact AND list the old fact in `obsolete_items`.
3. If the conversation is casual, routine, or contains no new long-term value, set `important=false` and leave lists empty.
4. Keep entries to a single, concise sentence.
"""

memory_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash").with_structured_output(ExtractedMemory)


# ==========================================================
# 2. DatabaseClient (Fast SQLite WAL)
# ==========================================================

class DatabaseClient:
    """Handles fast, highly-concurrent SQLite storage for history and docs."""

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
        """Create tables if they don't exist. Runs migrations safely."""
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

            # ── Legacy table (kept for migration) ──
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_docs (
                    doc_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)

            # ── Rich memory_items table (Phase 2) ──
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

            # ── FTS5 virtual table for BM25 keyword search ──
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                    text, kind,
                    content='memory_items',
                    content_rowid='rowid'
                )
            """)

            # ── Auto-migrate legacy memory_docs → memory_items ──
            cursor = conn.execute("SELECT COUNT(*) FROM memory_docs")
            legacy_count = cursor.fetchone()[0]
            if legacy_count > 0:
                logger.info(f"🔄 Migrating {legacy_count} legacy memory_docs → memory_items...")
                now = datetime.now(timezone.utc).isoformat()
                conn.execute("""
                    INSERT OR IGNORE INTO memory_items (id, kind, text, created_ts, updated_ts, indexed)
                    SELECT doc_id, 'fact', content, created_at, ?, 0
                    FROM memory_docs
                """, (now,))
                # Clear legacy table after migration
                conn.execute("DELETE FROM memory_docs")
                logger.info("✅ Migration complete — legacy memory_docs cleared")

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

    # ── Memory item methods (Phase 2) ─────────────────────────

    async def item_exists_by_text(self, text: str) -> str | None:
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
        """Retrieve active memory items by their IDs."""
        if not item_ids:
            return []
        def _fetch():
            with self.get_fast_connection() as conn:
                conn.row_factory = sqlite3.Row
                placeholders = ",".join("?" * len(item_ids))
                cursor = conn.execute(
                    f"SELECT id, kind, text FROM memory_items WHERE id IN ({placeholders}) AND status = 'active'",
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
                    # Sync FTS index
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
                # FTS5 MATCH query with BM25 ranking
                cursor = conn.execute("""
                    SELECT mi.id, mi.kind, mi.text, rank
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


from fastembed import TextEmbedding
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

HEALTH_ID = "__health_check__"

# ==========================================================
# 3. Zvec Vector Store
# ==========================================================

class ZvecMemoryStore:
    """In-process Semantic Memory DB backed by Zvec + SQLite doc store."""

    def __init__(self, db_client: DatabaseClient):
        self.db = db_client
        self.collection = None
        self._needs_rebuild = False  # set True when corruption forces a fresh index
        self._zvec_write_lock = asyncio.Lock()

        # FastEmbed Client (Lightweight local BGE model by default)
        self.embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        # bge-small-en-v1.5 embedding dimension is 384
        self.dim = 384
        self.health_vector = [1.0] + [0.0] * (self.dim - 1)

        self.skill_collection = None

    def _wipe_and_recreate(self, path: str, schema: zvec.CollectionSchema):
        """Delete a corrupt Zvec directory and return a fresh collection."""
        import shutil
        logger.warning(f"⚠️  Corrupt Zvec index at {path} — wiping and recreating...")
        shutil.rmtree(path, ignore_errors=True)
        return zvec.create_and_open(path=path, schema=schema)

    def _ensure_health_doc(self, coll: zvec.Collection):
        got = coll.fetch(HEALTH_ID)
        if HEALTH_ID in got:
            return
        coll.upsert([zvec.Doc(id=HEALTH_ID, vectors={"embedding": self.health_vector})])
        coll.flush()

    def _probe_integrity(self, coll: zvec.Collection) -> bool:
        try:
            got = coll.fetch(HEALTH_ID)
            if HEALTH_ID not in got:
                return False
            res = coll.query(vectors=zvec.VectorQuery("embedding", vector=self.health_vector), topk=1)
            return bool(res) and res[0].id == HEALTH_ID
        except Exception:
            return False

    def initialize(self):
        """Must be called on startup to open the Zvec indexes."""
        mem_schema = zvec.CollectionSchema(
            name="agent_memory",
            vectors=zvec.VectorSchema("embedding", zvec.DataType.VECTOR_FP32, self.dim),
        )

        # --- Memory index ---
        try:
            self.collection = zvec.open(path=ZVEC_PATH)
            self._ensure_health_doc(self.collection)
            if not self._probe_integrity(self.collection):
                raise RuntimeError("Memory Zvec integrity probe failed")
        except Exception as e:
            logger.warning(f"⚠️  zvec_index open/probe failed ({e}), rebuilding from SQLite...")
            self.collection = self._wipe_and_recreate(ZVEC_PATH, mem_schema)
            self._ensure_health_doc(self.collection)
            self._needs_rebuild = True  # will re-embed from SQLite after init

        # --- Skill index ---
        skill_schema = zvec.CollectionSchema(
            name="skill_memory",
            vectors=zvec.VectorSchema("embedding", zvec.DataType.VECTOR_FP32, self.dim),
        )
        try:
            self.skill_collection = zvec.open(path=ZVEC_SKILLS_PATH)
            self._ensure_health_doc(self.skill_collection)
            if not self._probe_integrity(self.skill_collection):
                raise RuntimeError("Skill Zvec integrity probe failed")
        except Exception as e:
            logger.warning(f"⚠️  zvec_skills open failed ({e}), recreating...")
            self.skill_collection = self._wipe_and_recreate(ZVEC_SKILLS_PATH, skill_schema)
            self._ensure_health_doc(self.skill_collection)

    async def _embed(self, texts: List[str]) -> List[List[float]]:
        """Generate local vector embeddings using FastEmbed."""
        if not texts:
            return []
        
        def _get_embeddings():
            try:
                # FastEmbed returns a generator of numpy arrays, convert to standard lists
                embeddings_generator = self.embedding_model.embed(texts)
                return [emb.tolist() for emb in embeddings_generator]
            except Exception as e:
                logger.error(f"FastEmbed failed: {e}")
                return []
                
        return await asyncio.to_thread(_get_embeddings)

    async def get_relevant_context(self, query: str, top_k: int = 5) -> str:
        """Hybrid retrieval: Zvec vector search + FTS5 BM25, merged and deduped."""
        if not self.collection:
            return ""
            
        vector = (await self._embed([query]))[0]
        
        # 1. Vector search (Zvec)
        zvec_ranked = []
        try:
            results = self.collection.query(
                zvec.VectorQuery("embedding", vector=vector),
                topk=top_k * 2
            )
            zvec_ranked = [res.id for res in results if res.id != HEALTH_ID]
        except Exception as e:
            logger.error(f"Zvec query failed: {e}")
        
        # 2. BM25 keyword search (FTS5)
        fts_results = await self.db.search_fts(query, limit=top_k * 2)
        fts_ranked = [r["id"] for r in fts_results]
        
        # 3. Merge + deduplicate via Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        for rank, item_id in enumerate(zvec_ranked):
            rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + 1.0 / (60 + rank + 1)
            
        for rank, item_id in enumerate(fts_ranked):
            rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + 1.0 / (60 + rank + 1)

        merged_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
        
        if not merged_ids:
            return ""
            
        # 4. Retrieve active items from SQLite
        items = await self.db.get_active_items_by_ids(merged_ids)
        if not items:
            return ""
        return "\n".join(f"- [{item['kind']}] {item['text']}" for item in items)

    async def get_relevant_skills(self, query: str, top_k: int = 2, threshold: float = 0.65) -> str:
        """Search Zvec for skills relevant to the query. Only returns skills above score threshold."""
        if not self.skill_collection:
            return ""
            
        vector = (await self._embed([query]))[0]
        
        try:
            results = self.skill_collection.query(
                zvec.VectorQuery("embedding", vector=vector),
                topk=top_k * 5  # get a few extra to ensure unique skills
            )
            
            # Filter by threshold + deduplicate to unique skill names
            skill_names = []
            for res in results:
                if res.id == HEALTH_ID:
                    continue
                
                logger.debug(f"  -> Found skill {res.id} with score {res.score:.3f}")
                
                if res.score < threshold:
                    logger.debug(f"  -> Skipping skill {res.id} (below threshold {threshold})")
                    continue
                    
                skill_id = res.id
                if skill_id not in skill_names:
                    skill_names.append(skill_id)
                    logger.info(f"  -> Skill accepted: {skill_id} (score={res.score:.3f})")
            
            skill_names = skill_names[:top_k]
            
            if not skill_names:
                logger.debug("  -> No skills above threshold — using general fallback")
                return "You are a helpful personal assistant. Be concise and accurate."
                
            prompts = []
            for skill_name in skill_names:
                skill_dir = SKILLS_DIR / skill_name
                skill_file = None
                
                # Case-insensitive search for skill.md
                if skill_dir.exists():
                    for f in skill_dir.iterdir():
                        if f.name.lower() == "skill.md":
                            skill_file = f
                            break
                            
                if skill_file and skill_file.exists():
                    content = skill_file.read_text()
                    prompts.append(f"## {skill_name.replace('_', ' ').title()}\n{content}")
                    logger.info(f"  -> Retrieving dynamic skill prompt: {skill_file.name}")
                    
            if not prompts:
                return "You are a helpful personal assistant. Be concise and accurate."
                
            return "\n\n".join(prompts)
            
        except Exception as e:
            logger.error(f"Zvec skills query failed: {e}")
            return "You are a helpful personal assistant. Be concise and accurate."

    async def apply_updates(self, memory: ExtractedMemory, source_thread_id: str = None):
        """Write to SQLite memory_items only. Zvec is synced deferred."""
        
        # 1. Tombstone obsolete items (soft delete)
        if memory.obsolete_items:
            await self.db.tombstone_by_content(memory.obsolete_items)
            logger.info(f"  -> Tombstoned {len(memory.obsolete_items)} obsolete memories")

        # 2. Ingest new items with kind classification
        kind_map = [
            (memory.preferences, 'pref'),
            (memory.facts, 'fact'),
            (memory.corrections, 'rule'),
        ]
        
        inserted = 0
        for items, kind in kind_map:
            for text in items:
                existing_id = await self.db.item_exists_by_text(text)
                if existing_id:
                    await self.db.touch_item(existing_id)
                    continue
                    
                # Semantic similarity check for deduplication
                is_duplicate = False
                if self.collection:
                    try:
                        vector = (await self._embed([text]))[0]
                        results = self.collection.query(
                            zvec.VectorQuery("embedding", vector=vector),
                            topk=3
                        )
                        for res in results:
                            if res.id != HEALTH_ID and res.score > 0.92:
                                await self.db.touch_item(res.id)
                                logger.info(f"  -> Semantic dedup (>0.92): updated existing item {res.id}")
                                is_duplicate = True
                                break
                    except Exception as e:
                        logger.error(f"Semantic dedup check failed: {e}")
                        
                if is_duplicate:
                    continue
                    
                item_id = f"mem_{uuid.uuid4().hex[:12]}"
                await self.db.insert_memory_item(item_id, kind, text, source_thread_id)
                inserted += 1
        
        if inserted:
            logger.info(f"  -> Inserted {inserted} new memory items into SQLite")

    async def sync_pending_memories(self, batch_size: int = 256):
        """Batch embed and sync memory_items -> Zvec in one fast locked operation."""
        if not self.collection:
            return

        pending = await self.db.fetch_pending_items(batch_size)
        if not pending:
            return

        texts = [r["text"] for r in pending]
        ids = [r["id"] for r in pending]
        
        # Embed outside the lock (expensive)
        vectors = await self._embed(texts)
        if not vectors:
            return

        docs = [zvec.Doc(id=i, vectors={"embedding": v}) for i, v in zip(ids, vectors)]

        # Single locked write batch into Zvec + flush
        async with self._zvec_write_lock:
            try:
                self._ensure_health_doc(self.collection)
                self.collection.upsert(docs)
                self.collection.flush()
            except Exception as e:
                logger.error(f"Zvec deferred sync failed: {e}")
                return

        # Mark as indexed in SQLite (after Zvec flush succeeds)
        await self.db.mark_items_indexed(ids)
        logger.info(f"✅ Synced {len(ids)} new memories into Zvec index")

    async def rebuild_from_sqlite(self):
        """
        Re-populate zvec_index from the durable memory_items table.

        Called after corruption forces a fresh index so no user memory is lost.
        The text lives in SQLite; we just need to re-embed it.
        """
        logger.info("🔄 Rebuilding zvec_index from SQLite memory_items...")

        def _fetch_all():
            with self.db.get_fast_connection() as conn:
                cursor = conn.execute("SELECT id, text FROM memory_items WHERE status = 'active'")
                return cursor.fetchall()

        rows = await asyncio.to_thread(_fetch_all)

        if not rows:
            logger.info("  -> No active memory_items found — fresh start.")
            self._needs_rebuild = False
            return

        item_ids = [r[0] for r in rows]
        texts    = [r[1] for r in rows]

        embeddings = await self._embed(texts)
        docs_to_zvec = [
            zvec.Doc(id=item_id, vectors={"embedding": vector})
            for item_id, vector in zip(item_ids, embeddings)
        ]

        try:
            async with self._zvec_write_lock:
                self.collection.upsert(docs_to_zvec)
                self.collection.flush()
            logger.info(f"✅ Rebuilt zvec_index with {len(docs_to_zvec)} memory items from SQLite")
        except Exception as e:
            logger.error(f"Zvec rebuild insert failed: {e}")

        # Mark all as indexed
        await self.db.mark_items_indexed(item_ids)
        self._needs_rebuild = False

    async def initialize_skills(self):
        """Called on startup to embed skills/*.md so Zvec can retrieve them."""
        if not SKILLS_DIR.exists() or not self.skill_collection:
            return

        import re
        skills_to_embed = []
        skill_ids = []

        # Load all .md files (case-insensitive for skill.md)
        for skill_file in SKILLS_DIR.rglob("*"):
            if skill_file.name.lower() != "skill.md":
                continue
                
            skill_id = skill_file.parent.name
            if skill_id == "identity":
                continue # identity is loaded separately into the system prompt

            content = skill_file.read_text()
            
            name = skill_id
            description = "Detailed documentation for the skill."
            
            # Simple YAML frontmatter parser
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    frontmatter = parts[1]
                    name_match = re.search(r"^name:\s*(.+)$", frontmatter, re.MULTILINE)
                    if name_match:
                        name = name_match.group(1).strip()
                    desc_match = re.search(r"^description:\s*(.+)$", frontmatter, re.MULTILINE)
                    if desc_match:
                        description = desc_match.group(1).strip()

            summary = f"Skill: {name}\nDescription: {description}"
            skills_to_embed.append(summary)
            skill_ids.append(skill_id)

        if not skills_to_embed:
            return

        embeddings = await self._embed(skills_to_embed)

        docs_to_zvec = []
        for s_id, vector in zip(skill_ids, embeddings):
            docs_to_zvec.append(zvec.Doc(id=s_id, vectors={"embedding": vector}))

        try:
            async with self._zvec_write_lock:
                self.skill_collection.upsert(docs_to_zvec)
                self.skill_collection.flush()
            logger.info(f"✅ Embedded {len(docs_to_zvec)} skill summaries for Progressive Disclosure")
        except Exception as e:
            logger.error(f"Zvec skills insert failed: {e}")

    async def close(self):
        """Gracefully flush and close Zvec collections on shutdown."""
        if self.collection:
            try:
                self.collection.flush()
            except Exception as e:
                logger.error(f"Error flushing memory index: {e}")
            finally:
                self.collection = None
                
        if self.skill_collection:
            try:
                self.skill_collection.flush()
            except Exception as e:
                logger.error(f"Error flushing skill index: {e}")
            finally:
                self.skill_collection = None

# ==========================================================
# 4. MemoryGate Interface
# ==========================================================

class MemoryGate:
    """Background semantic extraction and retrieval."""

    def __init__(self):
        self.db = DatabaseClient()
        self.store = ZvecMemoryStore(db_client=self.db)

    async def initialize(self):
        """Must be called at application startup."""
        self.db.initialize()
        self.store.initialize()
        # If corruption was detected, rebuild the vector index from durable SQLite
        if self.store._needs_rebuild:
            await self.store.rebuild_from_sqlite()
        await self.store.initialize_skills()

    async def process(self, thread_id: str, user_input: str, agent_response: str):
        """Background extraction triggered after a turn."""
        logger.info(f"--- [MemoryGate: Processing Thread {thread_id}] ---")
        
        # 1. Fast History Write
        await self.db.add_history(thread_id, "user", user_input)
        await self.db.add_history(thread_id, "assistant", agent_response)

        conversation = f"User: {user_input}\nAssistant: {agent_response}"
        
        # 2. Fetch relevant memories (hybrid: vector + BM25) for dedup context
        relevant_context = await self.store.get_relevant_context(conversation, top_k=10)

        # 3. Extract structured memories (passing current memories as context)
        try:
            system_msg = EXTRACTION_PROMPT.format(existing_memory=relevant_context if relevant_context else "No existing memory.")
            memory = await memory_llm.ainvoke([
                SystemMessage(content=system_msg),
                HumanMessage(content=conversation)
            ])

            if memory.important:
                await self.store.apply_updates(memory, source_thread_id=thread_id)
                logger.info(
                    f" -> Memory updated: +{len(memory.preferences)}P "
                    f"+{len(memory.facts)}F +{len(memory.corrections)}R "
                    f"-{len(memory.obsolete_items)} tombstoned"
                )
            else:
                logger.info(" -> No notable memory from this turn")

        except Exception as e:
            logger.error(f" -> Memory extraction failed: {e}", exc_info=True)

    async def get_context(self, thread_id: str) -> str:
        """Retrieve relevant context for the agent node before generation."""
        # 1. Fetch recent fast history
        recent_rows = await self.db.get_recent_history(thread_id, limit=6)
        
        parts = []
        if recent_rows:
            history_str = "\n".join(f"- **{row['role'].title()}**: {row['content']}" for row in recent_rows)
            # 2. Hybrid semantic+BM25 context based on user's latest message
            latest_user = next((row['content'] for row in reversed(recent_rows) if row['role'] == 'user'), "")
            
            semantic_context = ""
            if latest_user:
                semantic_context = await self.store.get_relevant_context(latest_user, top_k=10)
            
            if semantic_context:
                parts.append("## Semantic Memory Context")
                parts.append(semantic_context)
                
            parts.append("## Recent Conversation History")
            parts.append(history_str)
            
        return "\n".join(parts) if parts else "No established context."

    async def get_relevant_skills(self, user_input: str) -> str:
        """Fetch dynamic skill prompts to inject into system prompt."""
        return await self.store.get_relevant_skills(user_input, top_k=2)


# Singleton
memorygate = MemoryGate()
