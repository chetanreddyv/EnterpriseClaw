"""
memory/vectorstore.py — ZvecMemoryStore: Semantic vector index backed by Zvec + FastEmbed.

Production-grade implementation featuring:
- Thread-safe optimistic caching (Lock Striping)
- Bounded ONNX inference queues (Semaphore backpressure)
- Asynchronous File I/O
- Reciprocal Rank Fusion (RRF) & Time-Decay Scoring
- Event-loop non-blocking initialization
"""

import logging
import asyncio
import math
import os
import uuid
import shutil
import re
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, timezone

import zvec
from fastembed import TextEmbedding

logger = logging.getLogger(__name__)

# --- Paths & Constants ---
MEMORY_DIR = Path(__file__).parent.parent / "data"
ZVEC_PATH = str(MEMORY_DIR / "zvec_index")
ZVEC_SKILLS_PATH = str(MEMORY_DIR / "zvec_skills")
SKILLS_DIR = Path(__file__).parent.parent / "skills"
HEALTH_ID = "__health_check__"

# --- Hyperparameters ---
TIME_DECAY_LAMBDA = 0.0099
MEMORY_SIMILARITY_THRESHOLD = float(os.getenv("MEMORY_SIMILARITY_THRESHOLD", "0.60"))
DEDUPLICATION_THRESHOLD = 0.92
RRF_CONSTANT = 60
DECAY_WEIGHT_BASE = 0.7
DECAY_WEIGHT_TIME = 0.3

# --- Pre-compiled Patterns ---
TRIGGER_PATTERN = re.compile(r'### TRIGGER_EXAMPLES(.*?)### END_TRIGGER_EXAMPLES', re.DOTALL)
NAME_PATTERN = re.compile(r"^name:\s*(.+)$", re.MULTILINE)
DESC_PATTERN = re.compile(r"^description:\s*(.+)$", re.MULTILINE)

NAVIGATION_INTENT_PREFIXES = ("go to ", "visit ", "open ", "navigate to ", "browse ")
NAVIGATION_INTENT_PATTERN = re.compile(r"\b(go\s+to|visit|open|navigate\s+to|browse)\b")


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        item = str(value).strip()
        if item and item not in normalized:
            normalized.append(item)
    return normalized

def chunked_iterable(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

def _looks_like_navigation_intent(query_lower: str) -> bool:
    normalized = str(query_lower or "").strip().lower()
    if not normalized: return False
    if any(normalized.startswith(prefix) for prefix in NAVIGATION_INTENT_PREFIXES): return True
    trimmed = normalized.lstrip("\"'`([{<:;,.!?- ")
    if any(trimmed.startswith(prefix) for prefix in NAVIGATION_INTENT_PREFIXES): return True
    return bool(NAVIGATION_INTENT_PATTERN.search(normalized))


class ZvecMemoryStore:
    """Production-hardened Semantic Memory DB."""

    # --- Hyperparameters ---
    TIME_DECAY_LAMBDA = TIME_DECAY_LAMBDA
    DEFAULT_SIMILARITY_THRESHOLD = MEMORY_SIMILARITY_THRESHOLD
    DEDUPLICATION_THRESHOLD = DEDUPLICATION_THRESHOLD
    RRF_CONSTANT = RRF_CONSTANT
    DECAY_WEIGHT_BASE = DECAY_WEIGHT_BASE
    DECAY_WEIGHT_TIME = DECAY_WEIGHT_TIME

    def __init__(self, db_client):
        self.db = db_client
        self.collection = None
        self.skill_collection = None
        self._needs_rebuild = False
        
        self.dim = 384  
        self.health_vector = [1.0] + [0.0] * (self.dim - 1)

        self._embedding_model = None
        self._zvec_write_lock = asyncio.Lock()
        
        # Concurrency Controls
        self._skill_cache: Dict[str, Any] = {}
        self._skill_locks: Dict[str, asyncio.Lock] = {} 
        self._embedding_semaphore = asyncio.Semaphore(4) # Bounded ML inference

    # ─── Bounded ML Inference ──────────────────────────────────────────

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            logger.info("📡 Loading FastEmbed model (BAAI/bge-small-en-v1.5)...")
            try:
                self._embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", threads=1)
            except Exception as e:
                logger.error(f"❌ Failed to load FastEmbed model: {e}")
        return self._embedding_model

    async def _embed(self, texts: List[str]) -> List[List[float]]:
        if not texts: return []

        async with self._embedding_semaphore:
            def _get_embeddings_sync():
                if not self.embedding_model: return []
                try:
                    return [emb.tolist() for emb in self.embedding_model.embed(texts)]
                except Exception as e:
                    logger.error(f"FastEmbed failed: {e}")
                    return []
            return await asyncio.to_thread(_get_embeddings_sync)

    # ─── Thread-Safe Optimistic Caching & Parsing ──────────────────────

    def _get_skill_lock(self, skill_id: str) -> asyncio.Lock:
        if skill_id not in self._skill_locks:
            self._skill_locks[skill_id] = asyncio.Lock()
        return self._skill_locks[skill_id]

    async def _get_cached_skill(self, skill_id: str) -> Optional[Dict[str, Any]]:
        """Optimistic read: Check RAM first, fallback to locking disk read."""
        if skill_id in self._skill_cache:
            return self._skill_cache[skill_id]

        async with self._get_skill_lock(skill_id):
            if skill_id in self._skill_cache:
                return self._skill_cache[skill_id]
            
            parsed_data = await self._async_read_and_parse_skill(skill_id)
            if parsed_data:
                self._skill_cache[skill_id] = parsed_data
            return parsed_data

    async def _async_read_and_parse_skill(self, skill_id: str) -> Optional[Dict[str, Any]]:
        clean_skill_id = str(skill_id or "").strip()
        if not clean_skill_id: return None

        skill_dir = SKILLS_DIR / clean_skill_id
        
        def _read():
            if not skill_dir.exists() or not skill_dir.is_dir(): return None
            for filename in ["SKILL.md", "skill.md"]:
                path = skill_dir / filename
                if path.exists(): return path.read_text()
            return None

        content = await asyncio.to_thread(_read)
        if not content: return None

        name = clean_skill_id
        description = ""
        declared_tools: List[str] = []
        triggers: List[str] = []

        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                frontmatter = parts[1]
                name_match = NAME_PATTERN.search(frontmatter)
                if name_match: name = name_match.group(1).strip()
                desc_match = DESC_PATTERN.search(frontmatter)
                if desc_match: description = desc_match.group(1).strip()
                declared_tools = self._parse_tools_from_frontmatter(frontmatter)

        trigger_match = TRIGGER_PATTERN.search(content)
        if trigger_match:
            raw_examples = trigger_match.group(1).strip().split('\n')
            triggers = [ex.replace('-', '').replace('"', '').replace('*', '').strip() 
                       for ex in raw_examples if ex.strip()]

        clean_content = TRIGGER_PATTERN.sub('', content).strip()
        if not clean_content: return None

        return {
            "name": name,
            "content": clean_content,
            "tools": declared_tools,
            "description": description,
            "triggers": triggers
        }

    def _parse_tools_from_frontmatter(self, frontmatter: str) -> list[str]:
        lines = frontmatter.splitlines()
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped.startswith("tools:"): continue

            inline_value = stripped[len("tools:"):].strip()
            if inline_value: return _dedupe_preserve_order(inline_value.split(","))

            parsed: list[str] = []
            for follow in lines[i + 1:]:
                follow_stripped = follow.strip()
                if not follow_stripped or follow_stripped.startswith("#"): continue
                if follow_stripped.startswith("- "):
                    parsed.append(follow_stripped[2:].strip())
                    continue
                if len(follow) - len(follow.lstrip()) == 0: break
                break
            return _dedupe_preserve_order(parsed)
        return []

    # ─── Lifecycle & Initialization ─────────────────────────────────────

    async def _async_query(self, collection, query_obj, topk: int):
        return await asyncio.to_thread(collection.query, query_obj, topk=topk)

    async def _async_upsert_and_flush(self, collection, docs):
        async with self._zvec_write_lock:
            await asyncio.to_thread(collection.upsert, docs)
            await asyncio.to_thread(collection.flush)

    def _ensure_health_doc(self, coll: zvec.Collection):
        got = coll.fetch(HEALTH_ID)
        if HEALTH_ID not in got:
            coll.upsert([zvec.Doc(id=HEALTH_ID, vectors={"embedding": self.health_vector})])
            coll.flush()

    def _probe_integrity(self, coll: zvec.Collection) -> bool:
        try:
            got = coll.fetch(HEALTH_ID)
            if HEALTH_ID not in got: return False
            res = coll.query(vectors=zvec.VectorQuery("embedding", vector=self.health_vector), topk=1)
            return bool(res) and res[0].id == HEALTH_ID
        except Exception:
            return False

    async def _wipe_and_recreate(self, path: str, schema: zvec.CollectionSchema):
        logger.warning(f"⚠️ Wiping and recreating Zvec index at {path}...")
        await asyncio.to_thread(shutil.rmtree, path, ignore_errors=True)
        return await asyncio.to_thread(zvec.create_and_open, path=path, schema=schema)

    async def initialize(self):
        """Async initialization to prevent blocking the event loop on startup."""
        mem_schema = zvec.CollectionSchema(
            name="agent_memory",
            vectors=zvec.VectorSchema("embedding", zvec.DataType.VECTOR_FP32, self.dim),
        )

        try:
            self.collection = await asyncio.to_thread(zvec.open, path=ZVEC_PATH)
            await asyncio.to_thread(self._ensure_health_doc, self.collection)
            if not await asyncio.to_thread(self._probe_integrity, self.collection):
                raise RuntimeError("Memory Zvec integrity probe failed")
        except Exception as e:
            logger.warning(f"⚠️ zvec_index probe failed ({e}), rebuilding from SQLite...")
            self.collection = await self._wipe_and_recreate(ZVEC_PATH, mem_schema)
            await asyncio.to_thread(self._ensure_health_doc, self.collection)
            self._needs_rebuild = True

        skill_schema = zvec.CollectionSchema(
            name="skill_memory",
            vectors=zvec.VectorSchema("embedding", zvec.DataType.VECTOR_FP32, self.dim),
        )
        self.skill_collection = await self._wipe_and_recreate(ZVEC_SKILLS_PATH, skill_schema)
        await asyncio.to_thread(self._ensure_health_doc, self.skill_collection)

    @staticmethod
    def _time_decay(updated_ts: str) -> float:
        try:
            updated = datetime.fromisoformat(updated_ts)
            now = datetime.now(timezone.utc)
            days_old = max((now - updated).total_seconds() / 86400, 0)
            return math.exp(-ZvecMemoryStore.TIME_DECAY_LAMBDA * days_old)
        except Exception:
            return 0.5  

    # ─── Core Retrieval ─────────────────────────────────────────────────

    async def get_relevant_context(self, query: str, top_k: int = 5) -> str:
        if not self.collection: return ""
        vectors = await self._embed([query])
        if not vectors: return ""

        zvec_ranked = []
        try:
            results = await self._async_query(
                self.collection, zvec.VectorQuery("embedding", vector=vectors[0]), top_k * 2
            )
            zvec_ranked = [res.id for res in results if res.id != HEALTH_ID and res.score >= self.DEFAULT_SIMILARITY_THRESHOLD]
        except Exception as e:
            logger.error(f"Zvec query failed: {e}")

        fts_results = await self.db.search_fts(query, limit=top_k * 2)
        fts_ranked = [r["id"] for r in fts_results]

        rrf_scores = {}
        for rank, item_id in enumerate(zvec_ranked):
            rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + 1.0 / (self.RRF_CONSTANT + rank + 1)
        for rank, item_id in enumerate(fts_ranked):
            rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + 1.0 / (self.RRF_CONSTANT + rank + 1)

        merged_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k * 2]
        if not merged_ids: return ""

        items = await self.db.get_active_items_by_ids(merged_ids)
        if not items: return ""

        scored_items = []
        for item in items:
            base_rrf = rrf_scores.get(item["id"], 0.0)
            decay = self._time_decay(item.get("updated_ts", ""))
            final_score = self.DECAY_WEIGHT_BASE * base_rrf + self.DECAY_WEIGHT_TIME * (base_rrf * decay)
            scored_items.append((final_score, item))

        scored_items.sort(key=lambda x: x[0], reverse=True)
        top_items = [item for _, item in scored_items[:top_k]]

        return "\n".join(f"- [{item['kind']}] {item['text']}" for item in top_items)

    async def get_relevant_skills(self, query: str, top_k: int = 2) -> tuple[str, list[str]]:
        prompts, skill_names, _ = await self.get_relevant_skills_with_metadata(query, top_k=top_k)
        return prompts, skill_names

    async def get_relevant_skills_with_metadata(
        self, query: str, top_k: int = 2
    ) -> tuple[str, list[str], dict[str, list[str]]]:
        fallback = ("You are a helpful personal assistant. Be concise and accurate.", [], {})
        if not self.skill_collection: return fallback

        vectors = await self._embed([query])
        if not vectors: return fallback

        try:
            results = await self._async_query(
                self.skill_collection, zvec.VectorQuery("embedding", vector=vectors[0]), topk=top_k * 5
            )

            skill_names = []
            ranked_skill_candidates: list[tuple[str, float]] = []
            seen_skill_candidates: set[str] = set()
            
            if results:
                logger.info(f"get_relevant_skills: Zvec query returned {len(results)} raw results (top result score: {results[0].score:.4f} id: {results[0].id})")
            else:
                logger.info("get_relevant_skills: Zvec query returned 0 raw results")
            
            query_lower = query.lower()
            is_navigation_intent = _looks_like_navigation_intent(query_lower)
            score_threshold = min(self.DEFAULT_SIMILARITY_THRESHOLD, 0.50) if is_navigation_intent else self.DEFAULT_SIMILARITY_THRESHOLD

            for res in results or []:
                if res.id == HEALTH_ID: continue

                skill_id = res.id.split("--")[-1]
                if not skill_id: continue

                if skill_id not in seen_skill_candidates:
                    ranked_skill_candidates.append((skill_id, float(getattr(res, "score", 0.0))))
                    seen_skill_candidates.add(skill_id)
                
                # Keyword matching
                id_words = skill_id.lower().replace("_", " ").split()
                keyword_match = any(w in query_lower for w in id_words if len(w) > 2)

                cache_entry = await self._get_cached_skill(skill_id)

                if not keyword_match and cache_entry:
                    description_text = str(cache_entry.get("description", "")).lower()
                    description_terms = {token for token in re.findall(r"[a-z0-9]+", description_text) if len(token) > 3}
                    if description_terms:
                        keyword_match = any(term in query_lower for term in description_terms)

                if not keyword_match and is_navigation_intent and cache_entry:
                    declared_tools = cache_entry.get("tools", [])
                    keyword_match = any(t.strip().lower().startswith("browser_") for t in declared_tools)

                if res.score < score_threshold and not keyword_match: continue
                if skill_id not in skill_names: skill_names.append(skill_id)

            skill_names = skill_names[:top_k]
            
            if not skill_names:
                forced_skill = ""
                for candidate_skill, _ in ranked_skill_candidates:
                    if candidate_skill.replace("_", " ").lower() in query_lower:
                        forced_skill = candidate_skill
                        break

                if not forced_skill and is_navigation_intent:
                    for candidate_skill, _ in ranked_skill_candidates:
                        cache_entry = await self._get_cached_skill(candidate_skill)
                        if cache_entry and any(t.strip().lower().startswith("browser_") for t in cache_entry.get("tools", [])):
                            forced_skill = candidate_skill
                            break

                if not forced_skill and ranked_skill_candidates:
                    candidate_skill, candidate_score = ranked_skill_candidates[0]
                    if candidate_score >= max(0.35, score_threshold * 0.85):
                        forced_skill = candidate_skill

                if not forced_skill: return fallback
                skill_names = [forced_skill]

            prompts = []
            skill_tools: dict[str, list[str]] = {}
            
            for skill_name in skill_names:
                cache_entry = await self._get_cached_skill(skill_name)
                if not cache_entry: continue
                
                cached_content = cache_entry.get("content", "")
                if cached_content:
                    declared_tools = cache_entry.get("tools", [])
                    if declared_tools:
                        skill_tools[skill_name] = declared_tools

                    tools_preview = ", ".join(declared_tools) if declared_tools else "None declared"
                    prompts.append(
                        f"### Skill Profile: {skill_name.replace('_', ' ').title()}\n"
                        f"- Declared tools: {tools_preview}\n"
                        "--- BEGIN SKILL ---\n"
                        f"{cached_content}\n"
                        "--- END SKILL ---"
                    )

            if not prompts: return fallback
            return "\n\n".join(prompts), skill_names, skill_tools

        except Exception as e:
            logger.error(f"Zvec skills query failed: {e}")
            return fallback

    # ─── Data Ingestion & Sync ──────────────────────────────────────────

    async def apply_updates(self, memory, source_thread_id: str = None):
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

        kind_map = [(memory.preferences, 'pref'), (memory.facts, 'fact'), (memory.corrections, 'rule')]
        texts_to_check = []
        
        for items, kind in kind_map:
            for text in items:
                existing_id = await self.db.item_exists_by_text(text)
                if existing_id: await self.db.touch_item(existing_id)
                else: texts_to_check.append((text, kind))

        if not texts_to_check or not self.collection: return

        raw_texts = [t[0] for t in texts_to_check]
        vectors = await self._embed(raw_texts)

        docs_to_upsert = []
        item_ids_to_mark = []

        for (text, kind), vector in zip(texts_to_check, vectors):
            results = await self._async_query(self.collection, zvec.VectorQuery("embedding", vector=vector), topk=1)
            is_dup = any(res.id != HEALTH_ID and res.score > self.DEDUPLICATION_THRESHOLD for res in results or [])
            
            if not is_dup:
                item_id = f"mem_{uuid.uuid4().hex[:12]}"
                await self.db.insert_memory_item(item_id, kind, text, source_thread_id)
                item_ids_to_mark.append(item_id)
                docs_to_upsert.append(zvec.Doc(id=item_id, vectors={"embedding": vector}))

        if docs_to_upsert:
            await self._async_upsert_and_flush(self.collection, docs_to_upsert)
            await self.db.mark_items_indexed(item_ids_to_mark)


    async def sync_pending_memories(self, batch_size: int = 256):
        if not self.collection: return
        pending = await self.db.fetch_pending_items(batch_size)
        if not pending: return

        texts = [r["text"] for r in pending]
        ids = [r["id"] for r in pending]
        vectors = await self._embed(texts)
        if not vectors: return

        assert len(ids) == len(vectors), "Fatal: Vector count mismatch during batch sync."
        docs = [zvec.Doc(id=i, vectors={"embedding": v}) for i, v in zip(ids, vectors)]
        await self._async_upsert_and_flush(self.collection, docs)
        await self.db.mark_items_indexed(ids)
        logger.info(f"✅ Synced {len(ids)} pending memories into Zvec index")


    async def rebuild_from_sqlite(self):
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
        for chunk in chunked_iterable(rows, 512):
            item_ids = [r[0] for r in chunk]
            texts = [r[1] for r in chunk]
            embeddings = await self._embed(texts)
            if not embeddings: continue

            docs = [zvec.Doc(id=i, vectors={"embedding": v}) for i, v in zip(item_ids, embeddings)]
            await self._async_upsert_and_flush(self.collection, docs)
            await self.db.mark_items_indexed(item_ids)
            total_upserted += len(docs)

        logger.info(f"✅ Rebuilt zvec_index with {total_upserted} memory items")
        self._needs_rebuild = False


    async def initialize_skills(self):
        if not self.skill_collection: return
        logger.info("🔄 Refreshing skills index and RAM cache...")
        
        skills_to_embed = []
        doc_ids = []
        
        async with asyncio.Lock(): # Guard global clear during rebuild
            self._skill_cache.clear()
            dirs = await asyncio.to_thread(lambda: [d for d in SKILLS_DIR.iterdir() if d.is_dir() and d.name != "identity"])

            for skill_dir in sorted(dirs):
                skill_id = skill_dir.name
                cache_entry = await self._async_read_and_parse_skill(skill_id)
                if not cache_entry: continue
                
                self._skill_cache[skill_id] = cache_entry

                for i, ex in enumerate(cache_entry.get("triggers", [])):
                    skills_to_embed.append(ex)
                    doc_ids.append(f"ex_{i}--{skill_id}")
                
                summary = f"Skill: {cache_entry.get('name')}. {cache_entry.get('description')}".strip()
                skills_to_embed.append(summary)
                doc_ids.append(f"sum--{skill_id}")

        if not skills_to_embed: return

        embeddings = await self._embed(skills_to_embed)
        docs_to_zvec = [zvec.Doc(id=d, vectors={"embedding": v}) for d, v in zip(doc_ids, embeddings)]

        await self._async_upsert_and_flush(self.skill_collection, docs_to_zvec)
        logger.info(f"✅ Re-Embedded {len(docs_to_zvec)} skill vectors. Cache populated.")


    async def close(self):
        if self.collection:
            try: await asyncio.to_thread(self.collection.flush)
            except Exception as e: logger.error(f"Error flushing memory index: {e}")
            finally: self.collection = None

        if self.skill_collection:
            try: await asyncio.to_thread(self.skill_collection.flush)
            except Exception as e: logger.error(f"Error flushing skill index: {e}")
            finally: self.skill_collection = None
