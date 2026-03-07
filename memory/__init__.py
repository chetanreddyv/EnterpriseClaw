"""
memory/ — EnterpriseClaw Memory System.

Triple-tiered architecture: SQLite WAL (source of truth), 
Zvec (semantic vector index), FTS5 (BM25 keyword search).
Hybrid retrieval via Reciprocal Rank Fusion with time-decay weighting.
"""
