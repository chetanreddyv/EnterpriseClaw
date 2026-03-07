"""
memory/vectorstore.py — ZvecMemoryStore: Semantic vector index backed by Zvec + FastEmbed.

Handles embedding, hybrid retrieval (vector + BM25 via RRF), 
semantic deduplication, time-decay scoring, and skill matching.
"""

import logging
import asyncio
import math
import uuid
from typing import List
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


class ZvecMemoryStore:
    """In-process Semantic Memory DB backed by Zvec + SQLite doc store."""

    def __init__(self, db_client):
        self.db = db_client
        self.collection = None
        self._needs_rebuild = False
        self._zvec_write_lock = asyncio.Lock()

        # FastEmbed Client (Lightweight local BGE model)
        self.embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.dim = 384  # bge-small-en-v1.5 embedding dimension
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
            self._needs_rebuild = True

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
                embeddings_generator = self.embedding_model.embed(texts)
                return [emb.tolist() for emb in embeddings_generator]
            except Exception as e:
                logger.error(f"FastEmbed failed: {e}")
                return []

        return await asyncio.to_thread(_get_embeddings)

    @staticmethod
    def _time_decay(updated_ts: str) -> float:
        """Exponential decay multiplier based on age. Returns 0.0–1.0."""
        try:
            updated = datetime.fromisoformat(updated_ts)
            now = datetime.now(timezone.utc)
            days_old = max((now - updated).total_seconds() / 86400, 0)
            return math.exp(-TIME_DECAY_LAMBDA * days_old)
        except Exception:
            return 0.5  # fallback for unparseable timestamps

    async def get_relevant_context(self, query: str, top_k: int = 5) -> str:
        """Hybrid retrieval: Zvec vector search + FTS5 BM25, merged via time-weighted RRF."""
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

        # Collect timestamps from FTS results for time-decay
        fts_timestamps = {r["id"]: r.get("updated_ts", "") for r in fts_results}

        # 3. Merge via Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        for rank, item_id in enumerate(zvec_ranked):
            rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + 1.0 / (60 + rank + 1)

        for rank, item_id in enumerate(fts_ranked):
            rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + 1.0 / (60 + rank + 1)

        merged_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k * 2]

        if not merged_ids:
            return ""

        # 4. Retrieve active items from SQLite (with timestamps)
        items = await self.db.get_active_items_by_ids(merged_ids)
        if not items:
            return ""

        # 5. Apply time-decay weighting to final ranking
        scored_items = []
        for item in items:
            base_rrf = rrf_scores.get(item["id"], 0.0)
            decay = self._time_decay(item.get("updated_ts", ""))
            # Blend: 70% relevance, 30% recency
            final_score = 0.7 * base_rrf + 0.3 * (base_rrf * decay)
            scored_items.append((final_score, item))

        scored_items.sort(key=lambda x: x[0], reverse=True)
        top_items = [item for _, item in scored_items[:top_k]]

        return "\n".join(f"- [{item['kind']}] {item['text']}" for item in top_items)

    async def get_relevant_skills(self, query: str, top_k: int = 2, threshold: float = 0.65) -> str:
        """Search Zvec for skills relevant to the query. Only returns skills above score threshold."""
        if not self.skill_collection:
            return ""

        vector = (await self._embed([query]))[0]

        try:
            results = self.skill_collection.query(
                zvec.VectorQuery("embedding", vector=vector),
                topk=top_k * 5
            )

            skill_names = []
            for res in results:
                if res.id == HEALTH_ID:
                    continue

                logger.debug(f"  -> Found skill {res.id} with score {res.score:.3f}")

                # Keyword fallback
                query_lower = query.lower()
                id_words = res.id.lower().replace("_", " ").split()
                keyword_match = any(w in query_lower for w in id_words if len(w) > 2)

                if res.score < threshold and not keyword_match:
                    logger.debug(f"  -> Skipping skill {res.id} (below threshold {threshold} and no keyword match)")
                    continue

                skill_id = res.id
                if skill_id not in skill_names:
                    skill_names.append(skill_id)
                    reason = f"score={res.score:.3f}" if res.score >= threshold else "keyword match"
                    logger.info(f"  -> Skill accepted: {skill_id} ({reason})")

            skill_names = skill_names[:top_k]

            if not skill_names:
                logger.debug("  -> No skills above threshold — using general fallback")
                return "You are a helpful personal assistant. Be concise and accurate."

            prompts = []
            for skill_name in skill_names:
                skill_dir = SKILLS_DIR / skill_name
                skill_file = None

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

    async def apply_updates(self, memory, source_thread_id: str = None):
        """Write to SQLite memory_items only. Zvec is synced deferred."""
        from .extraction import ExtractedMemory

        # 1. Tombstone obsolete items (soft delete)
        if memory.obsolete_items:
            await self.db.tombstone_by_content(memory.obsolete_items)
            logger.info(f"  -> Tombstoned {len(memory.obsolete_items)} obsolete memories")

        # 2. Apply updates (evolve existing memories with new info)
        if hasattr(memory, 'updates') and memory.updates:
            for update_str in memory.updates:
                if "|||" in update_str:
                    old_text, new_text = [s.strip() for s in update_str.split("|||", 1)]
                    existing_id = await self.db.item_exists_by_text(old_text)
                    if existing_id:
                        await self.db.update_item_text(existing_id, new_text)
                        logger.info(f"  -> Updated memory {existing_id}: '{old_text[:40]}...' → '{new_text[:40]}...'")
                    else:
                        logger.debug(f"  -> Update target not found, treating as new fact: '{new_text[:60]}'")
                        item_id = f"mem_{uuid.uuid4().hex[:12]}"
                        await self.db.insert_memory_item(item_id, 'fact', new_text, source_thread_id)

        # 3. Ingest new items with kind classification
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
        """Re-populate zvec_index from the durable memory_items table."""
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

        await self.db.mark_items_indexed(item_ids)
        self._needs_rebuild = False

    async def initialize_skills(self):
        """Called on startup to embed skills/*.md so Zvec can retrieve them."""
        if not SKILLS_DIR.exists() or not self.skill_collection:
            return

        import re
        skills_to_embed = []
        skill_ids = []

        for skill_file in SKILLS_DIR.rglob("*"):
            if skill_file.name.lower() != "skill.md":
                continue

            skill_id = skill_file.parent.name
            if skill_id == "identity":
                continue

            content = skill_file.read_text()

            name = skill_id
            description = "Detailed documentation for the skill."

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
