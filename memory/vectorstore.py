"""
memory/vectorstore.py — ZvecMemoryStore: Semantic vector index backed by Zvec + FastEmbed.

Handles embedding, hybrid retrieval (vector + BM25 via RRF), 
semantic deduplication, time-decay scoring, and skill matching.
"""

import logging
import asyncio
import math
import os
import uuid
import shutil
import re
from typing import Any, Dict, List, Tuple
from pathlib import Path
from datetime import datetime, timezone

import zvec
from fastembed import TextEmbedding

logger = logging.getLogger(__name__)

# Paths
MEMORY_DIR = Path(__file__).parent.parent / "data"
ZVEC_PATH = str(MEMORY_DIR / "zvec_index")
ZVEC_SKILLS_PATH = str(MEMORY_DIR / "zvec_skills")
SKILLS_DIR = Path(__file__).parent.parent / "skills"

HEALTH_ID = "__health_check__"

# Time-decay parameter: half-life of ~70 days (lambda = ln(2) / 70)
TIME_DECAY_LAMBDA = 0.0099
MEMORY_SIMILARITY_THRESHOLD = float(os.getenv("MEMORY_SIMILARITY_THRESHOLD", "0.60"))

# Pre-compiled Regex Patterns (Performance optimization)
TRIGGER_PATTERN = re.compile(r'### TRIGGER_EXAMPLES(.*?)### END_TRIGGER_EXAMPLES', re.DOTALL)
NAME_PATTERN = re.compile(r"^name:\s*(.+)$", re.MULTILINE)
DESC_PATTERN = re.compile(r"^description:\s*(.+)$", re.MULTILINE)


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        item = str(value).strip()
        if item and item not in normalized:
            normalized.append(item)
    return normalized


def _parse_tools_from_frontmatter(frontmatter: str) -> list[str]:
    """Parse tools from either inline CSV or YAML list syntax."""
    lines = frontmatter.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("tools:"):
            continue

        inline_value = stripped[len("tools:"):].strip()
        if inline_value:
            return _dedupe_preserve_order(inline_value.split(","))

        parsed: list[str] = []
        for follow in lines[i + 1:]:
            follow_stripped = follow.strip()
            if not follow_stripped:
                continue
            if follow_stripped.startswith("#"):
                continue
            if follow_stripped.startswith("- "):
                parsed.append(follow_stripped[2:].strip())
                continue

            # End of tools block once we leave list entries.
            if len(follow) - len(follow.lstrip()) == 0:
                break
            break

        return _dedupe_preserve_order(parsed)

    return []


def chunked_iterable(iterable, size):
    """Yield successive n-sized chunks from an iterable."""
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]


class ZvecMemoryStore:
    """In-process Semantic Memory DB backed by Zvec + SQLite doc store."""

    def __init__(self, db_client):
        self.db = db_client
        self.collection = None
        self._needs_rebuild = False
        self._zvec_write_lock = asyncio.Lock()

        # bge-small-en-v1.5 embedding dimension
        self.dim = 384  
        self.health_vector = [1.0] + [0.0] * (self.dim - 1)

        self.skill_collection = None
        self._embedding_model = None  # Lazy init
        
        # RAM Cache to prevent Disk I/O during retrievals
        self._skill_cache: Dict[str, Any] = {}

    # ─── Async Zvec Wrappers (Prevents Event Loop Blocking) ─────────────

    async def _async_query(self, collection, query_obj, topk: int):
        return await asyncio.to_thread(collection.query, query_obj, topk=topk)

    async def _async_upsert_and_flush(self, collection, docs):
        async with self._zvec_write_lock:
            await asyncio.to_thread(collection.upsert, docs)
            await asyncio.to_thread(collection.flush)

    # ─── Lifecycle & Initialization ─────────────────────────────────────

    def _wipe_and_recreate(self, path: str, schema: zvec.CollectionSchema):
        """Delete a Zvec directory and return a fresh collection."""
        logger.warning(f"⚠️ Wiping and recreating Zvec index at {path}...")
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
            logger.warning(f"⚠️ zvec_index open/probe failed ({e}), rebuilding from SQLite...")
            self.collection = self._wipe_and_recreate(ZVEC_PATH, mem_schema)
            self._ensure_health_doc(self.collection)
            self._needs_rebuild = True

        # --- Skill index ---
        skill_schema = zvec.CollectionSchema(
            name="skill_memory",
            vectors=zvec.VectorSchema("embedding", zvec.DataType.VECTOR_FP32, self.dim),
        )
        # Always wipe skills on startup to prevent "Ghost Triggers" from deleted markdown files
        self.skill_collection = self._wipe_and_recreate(ZVEC_SKILLS_PATH, skill_schema)
        self._ensure_health_doc(self.skill_collection)

    @property
    def embedding_model(self):
        """Lazy loader for the FastEmbed model."""
        if self._embedding_model is None:
            logger.info("📡 Loading FastEmbed model (BAAI/bge-small-en-v1.5)...")
            try:
                self._embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
            except Exception as e:
                logger.error(f"❌ Failed to load FastEmbed model: {e}")
        return self._embedding_model

    async def _embed(self, texts: List[str]) -> List[List[float]]:
        """Generate local vector embeddings using FastEmbed (Batched)."""
        if not texts:
            return []

        if not self.embedding_model:
            logger.error("_embed: Model not loaded, cannot embed.")
            return []

        def _get_embeddings_sync():
            try:
                embeddings_generator = self.embedding_model.embed(texts)
                return [emb.tolist() for emb in embeddings_generator]
            except Exception as e:
                logger.error(f"FastEmbed failed: {e}")
                return []

        return await asyncio.to_thread(_get_embeddings_sync)

    @staticmethod
    def _time_decay(updated_ts: str) -> float:
        try:
            updated = datetime.fromisoformat(updated_ts)
            now = datetime.now(timezone.utc)
            days_old = max((now - updated).total_seconds() / 86400, 0)
            return math.exp(-TIME_DECAY_LAMBDA * days_old)
        except Exception:
            return 0.5  

    # ─── Core Retrieval ─────────────────────────────────────────────────

    async def get_relevant_context(self, query: str, top_k: int = 5) -> str:
        if not self.collection:
            return ""

        vectors = await self._embed([query])
        if not vectors:
            return ""

        # 1. Async Vector search (Zvec)
        zvec_ranked = []
        try:
            results = await self._async_query(
                self.collection, 
                zvec.VectorQuery("embedding", vector=vectors[0]), 
                top_k * 2
            )
            zvec_ranked = [res.id for res in results if res.id != HEALTH_ID and res.score >= MEMORY_SIMILARITY_THRESHOLD]
        except Exception as e:
            logger.error(f"Zvec query failed: {e}")

        # 2. BM25 keyword search (FTS5)
        fts_results = await self.db.search_fts(query, limit=top_k * 2)
        fts_ranked = [r["id"] for r in fts_results]
        fts_timestamps = {r["id"]: r.get("updated_ts", "") for r in fts_results}

        # 3. Merge via RRF
        rrf_scores = {}
        for rank, item_id in enumerate(zvec_ranked):
            rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + 1.0 / (60 + rank + 1)
        for rank, item_id in enumerate(fts_ranked):
            rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + 1.0 / (60 + rank + 1)

        merged_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k * 2]

        if not merged_ids:
            return ""

        items = await self.db.get_active_items_by_ids(merged_ids)
        if not items:
            return ""

        # 4. Apply time-decay weighting
        scored_items = []
        for item in items:
            base_rrf = rrf_scores.get(item["id"], 0.0)
            decay = self._time_decay(item.get("updated_ts", ""))
            final_score = 0.7 * base_rrf + 0.3 * (base_rrf * decay)
            scored_items.append((final_score, item))

        scored_items.sort(key=lambda x: x[0], reverse=True)
        top_items = [item for _, item in scored_items[:top_k]]

        return "\n".join(f"- [{item['kind']}] {item['text']}" for item in top_items)

    async def get_relevant_skills(self, query: str, top_k: int = 2) -> tuple[str, list[str]]:
        prompts, skill_names, _ = await self.get_relevant_skills_with_metadata(query, top_k=top_k)
        return prompts, skill_names

    async def get_relevant_skills_with_metadata(
        self,
        query: str,
        top_k: int = 2,
    ) -> tuple[str, list[str], dict[str, list[str]]]:
        fallback = ("You are a helpful personal assistant. Be concise and accurate.", [], {})
        if not self.skill_collection:
            return fallback

        vectors = await self._embed([query])
        if not vectors:
            return fallback

        try:
            results = await self._async_query(
                self.skill_collection,
                zvec.VectorQuery("embedding", vector=vectors[0]),
                topk=top_k * 5
            )

            skill_names = []
            if results:
                logger.info(f"get_relevant_skills: Zvec query returned {len(results)} raw results (top result score: {results[0].score:.4f} id: {results[0].id})")
            else:
                logger.info("get_relevant_skills: Zvec query returned 0 raw results")
            
            query_lower = query.lower()

            for res in results:
                if res.id == HEALTH_ID:
                    continue

                skill_id = res.id.split("--")[-1]
                
                # Keyword fallback
                id_words = skill_id.lower().replace("_", " ").split()
                keyword_match = any(w in query_lower for w in id_words if len(w) > 2)

                if res.score < MEMORY_SIMILARITY_THRESHOLD and not keyword_match:
                    continue

                if skill_id not in skill_names:
                    skill_names.append(skill_id)

            skill_names = skill_names[:top_k]
            if not skill_names:
                return fallback

            prompts = []
            matched_skill_names = []
            skill_tools: dict[str, list[str]] = {}
            for skill_name in skill_names:
                # 💥 NO DISK I/O HERE: Pull straight from RAM cache
                cache_entry = self._skill_cache.get(skill_name)

                cached_content = ""
                declared_tools_raw: list[str] = []
                if isinstance(cache_entry, dict):
                    cached_content = str(cache_entry.get("content", "")).strip()
                    raw_tools = cache_entry.get("tools") or []
                    if isinstance(raw_tools, list):
                        declared_tools_raw = raw_tools
                elif cache_entry:
                    # Backward-compat path for existing tests/cache stubs.
                    cached_content = str(cache_entry).strip()

                if cached_content:
                    normalized_declared_tools: list[str] = []
                    for tool_name in declared_tools_raw:
                        if isinstance(tool_name, str):
                            value = tool_name.strip()
                            if value and value not in normalized_declared_tools:
                                normalized_declared_tools.append(value)

                    if normalized_declared_tools:
                        skill_tools[skill_name] = normalized_declared_tools

                    tools_preview = ", ".join(normalized_declared_tools) if normalized_declared_tools else "None declared"
                    prompts.append(
                        f"### Skill Profile: {skill_name.replace('_', ' ').title()}\n"
                        f"- Declared tools: {tools_preview}\n"
                        "--- BEGIN SKILL ---\n"
                        f"{cached_content}\n"
                        "--- END SKILL ---"
                    )
                    matched_skill_names.append(skill_name)

            if not prompts:
                return fallback

            return "\n\n".join(prompts), matched_skill_names, skill_tools

        except Exception as e:
            logger.error(f"Zvec skills query failed: {e}")
            return fallback

    # ─── Data Ingestion & Sync ──────────────────────────────────────────

    async def apply_updates(self, memory, source_thread_id: str = None):
        """Batch-optimized ingestion. Eliminates 'Double Embedding' penalty."""
        if memory.obsolete_items:
            await self.db.tombstone_by_content(memory.obsolete_items)

        if hasattr(memory, 'updates') and memory.updates:
            for update_str in memory.updates:
                if "|||" in update_str:
                    old_text, new_text = [s.strip() for s in update_str.split("|||", 1)]
                    existing_id = await self.db.item_exists_by_text(old_text)
                    if existing_id:
                        await self.db.update_item_text(existing_id, new_text)
                    else:
                        item_id = f"mem_{uuid.uuid4().hex[:12]}"
                        await self.db.insert_memory_item(item_id, 'fact', new_text, source_thread_id)

        # 1. Gather all novel text to leverage FastEmbed batching
        kind_map = [
            (memory.preferences, 'pref'),
            (memory.facts, 'fact'),
            (memory.corrections, 'rule'),
        ]

        texts_to_check = []
        for items, kind in kind_map:
            for text in items:
                existing_id = await self.db.item_exists_by_text(text)
                if existing_id:
                    await self.db.touch_item(existing_id)
                else:
                    texts_to_check.append((text, kind))

        if not texts_to_check or not self.collection:
            return

        # 2. Batch Embed once
        raw_texts = [t[0] for t in texts_to_check]
        vectors = await self._embed(raw_texts)

        # 3. Zvec Semantic Deduplication + Immediate Upsert
        docs_to_upsert = []
        item_ids_to_mark = []

        for (text, kind), vector in zip(texts_to_check, vectors):
            # Async check against index
            results = await self._async_query(
                self.collection, 
                zvec.VectorQuery("embedding", vector=vector), 
                topk=1
            )
            
            is_dup = False
            for res in results:
                if res.id != HEALTH_ID and res.score > 0.92:
                    await self.db.touch_item(res.id)
                    is_dup = True
                    break
            
            if not is_dup:
                item_id = f"mem_{uuid.uuid4().hex[:12]}"
                # Insert to SQLite and immediately mark as indexed since we have the vector
                await self.db.insert_memory_item(item_id, kind, text, source_thread_id)
                item_ids_to_mark.append(item_id)
                docs_to_upsert.append(zvec.Doc(id=item_id, vectors={"embedding": vector}))

        # 4. Upsert directly to Zvec (Skips background sync for these items)
        if docs_to_upsert:
            await self._async_upsert_and_flush(self.collection, docs_to_upsert)
            await self.db.mark_items_indexed(item_ids_to_mark)


    async def sync_pending_memories(self, batch_size: int = 256):
        """Background worker for items that missed direct indexing."""
        if not self.collection:
            return

        pending = await self.db.fetch_pending_items(batch_size)
        if not pending:
            return

        texts = [r["text"] for r in pending]
        ids = [r["id"] for r in pending]

        vectors = await self._embed(texts)
        if not vectors:
            return

        # Prevent silent corruption if embedding model drops a vector
        assert len(ids) == len(vectors), "Fatal: Vector count mismatch during batch sync."

        docs = [zvec.Doc(id=i, vectors={"embedding": v}) for i, v in zip(ids, vectors)]
        await self._async_upsert_and_flush(self.collection, docs)
        await self.db.mark_items_indexed(ids)
        logger.info(f"✅ Synced {len(ids)} pending memories into Zvec index")


    async def rebuild_from_sqlite(self):
        """Paginated re-population of zvec_index to prevent OOM."""
        logger.info("🔄 Rebuilding zvec_index from SQLite memory_items...")

        def _fetch_all():
            with self.db.get_fast_connection() as conn:
                cursor = conn.execute("SELECT id, text FROM memory_items WHERE status = 'active'")
                return cursor.fetchall()

        rows = await asyncio.to_thread(_fetch_all)
        if not rows:
            self._needs_rebuild = False
            return

        total_upserted = 0
        # Process in chunks of 512 to protect RAM
        for chunk in chunked_iterable(rows, 512):
            item_ids = [r[0] for r in chunk]
            texts    = [r[1] for r in chunk]

            embeddings = await self._embed(texts)
            if not embeddings:
                continue

            docs = [zvec.Doc(id=i, vectors={"embedding": v}) for i, v in zip(item_ids, embeddings)]
            await self._async_upsert_and_flush(self.collection, docs)
            await self.db.mark_items_indexed(item_ids)
            total_upserted += len(docs)

        logger.info(f"✅ Rebuilt zvec_index with {total_upserted} memory items")
        self._needs_rebuild = False


    async def initialize_skills(self):
        """Called on startup. Parses, RAM-caches, and embeds skills."""
        if not self.skill_collection:
             return
        
        logger.info(f"🔄 Refreshing skills index and RAM cache...")
        
        skills_to_embed = []
        doc_ids = []
        self._skill_cache.clear()

        for skill_dir in sorted(SKILLS_DIR.iterdir()):
            if not skill_dir.is_dir() or skill_dir.name == "identity":
                continue
                
            skill_id = skill_dir.name
            skill_file = skill_dir / "SKILL.md" if (skill_dir / "SKILL.md").exists() else skill_dir / "skill.md"
            
            if not skill_file.exists():
                continue

            content = skill_file.read_text()
            name = skill_id
            description = ""
            declared_tools: list[str] = []

            # Extract Frontmatter via regex
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    frontmatter = parts[1]
                    name_match = NAME_PATTERN.search(frontmatter)
                    if name_match: name = name_match.group(1).strip()
                    desc_match = DESC_PATTERN.search(frontmatter)
                    if desc_match: description = desc_match.group(1).strip()
                    declared_tools = _parse_tools_from_frontmatter(frontmatter)

            # Extract Trigger Examples
            trigger_match = TRIGGER_PATTERN.search(content)
            if trigger_match:
                raw_examples = trigger_match.group(1).strip().split('\n')
                for i, ex in enumerate(raw_examples):
                    clean_ex = ex.replace('-', '').replace('"', '').replace('*', '').strip()
                    if clean_ex:
                        skills_to_embed.append(clean_ex)
                        doc_ids.append(f"ex_{i}--{skill_id}")
            
            # Embed the base summary as a fallback
            summary = f"Skill: {name}. {description}".strip()
            skills_to_embed.append(summary)
            doc_ids.append(f"sum--{skill_id}")

            # RAM-Cache the clean content for zero-IO retrievals
            clean_content = TRIGGER_PATTERN.sub('', content).strip()
            self._skill_cache[skill_id] = {
                "content": clean_content,
                "tools": declared_tools,
            }
            logger.info("SkillCache: loaded '%s' with %d declared tools.", skill_id, len(declared_tools))

        if not skills_to_embed:
            return

        embeddings = await self._embed(skills_to_embed)
        docs_to_zvec = [zvec.Doc(id=d, vectors={"embedding": v}) for d, v in zip(doc_ids, embeddings)]

        await self._async_upsert_and_flush(self.skill_collection, docs_to_zvec)
        logger.info(f"✅ Re-Embedded {len(docs_to_zvec)} skill vectors. Cache populated.")


    async def close(self):
        """Gracefully flush and close Zvec collections on shutdown."""
        if self.collection:
            try:
                await asyncio.to_thread(self.collection.flush)
            except Exception as e:
                logger.error(f"Error flushing memory index: {e}")
            finally:
                self.collection = None

        if self.skill_collection:
            try:
                await asyncio.to_thread(self.skill_collection.flush)
            except Exception as e:
                logger.error(f"Error flushing skill index: {e}")
            finally:
                self.skill_collection = None