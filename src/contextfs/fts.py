"""
SQLite FTS5 Full-Text Search Backend.

Provides BM25-ranked keyword search as a lightweight alternative to RAG.
Can be used standalone or alongside the RAG backend for hybrid search.
"""

import sqlite3
import json
from pathlib import Path
from typing import Optional
from datetime import datetime

from contextfs.schemas import Memory, MemoryType, SearchResult


class FTSBackend:
    """
    SQLite FTS5 full-text search backend.

    Features:
    - BM25 ranking for relevance scoring
    - Phrase search support
    - Prefix matching
    - Column weighting (content > summary > tags)
    - Fast keyword search without ML dependencies
    """

    def __init__(self, db_path: Path):
        """
        Initialize FTS backend.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self._init_fts()

    def _init_fts(self) -> None:
        """Initialize FTS5 tables and triggers."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create FTS5 virtual table with BM25 ranking
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                id UNINDEXED,
                content,
                summary,
                tags,
                type UNINDEXED,
                namespace_id UNINDEXED,
                content='memories',
                content_rowid='rowid',
                tokenize='porter unicode61'
            )
        """)

        # Triggers to keep FTS in sync with memories table
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, id, content, summary, tags, type, namespace_id)
                VALUES (NEW.rowid, NEW.id, NEW.content, NEW.summary, NEW.tags, NEW.type, NEW.namespace_id);
            END
        """)

        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, id, content, summary, tags, type, namespace_id)
                VALUES ('delete', OLD.rowid, OLD.id, OLD.content, OLD.summary, OLD.tags, OLD.type, OLD.namespace_id);
            END
        """)

        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, id, content, summary, tags, type, namespace_id)
                VALUES ('delete', OLD.rowid, OLD.id, OLD.content, OLD.summary, OLD.tags, OLD.type, OLD.namespace_id);
                INSERT INTO memories_fts(rowid, id, content, summary, tags, type, namespace_id)
                VALUES (NEW.rowid, NEW.id, NEW.content, NEW.summary, NEW.tags, NEW.type, NEW.namespace_id);
            END
        """)

        conn.commit()
        conn.close()

    def rebuild_index(self) -> None:
        """Rebuild FTS index from memories table."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Clear existing FTS data
        cursor.execute("DELETE FROM memories_fts")

        # Rebuild from memories table
        cursor.execute("""
            INSERT INTO memories_fts(rowid, id, content, summary, tags, type, namespace_id)
            SELECT rowid, id, content, summary, tags, type, namespace_id FROM memories
        """)

        # Optimize the index
        cursor.execute("INSERT INTO memories_fts(memories_fts) VALUES('optimize')")

        conn.commit()
        conn.close()

    def search(
        self,
        query: str,
        limit: int = 10,
        type: Optional[MemoryType] = None,
        tags: Optional[list[str]] = None,
        namespace_id: Optional[str] = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """
        Search memories using FTS5.

        Args:
            query: Search query (supports FTS5 syntax)
            limit: Maximum results
            type: Filter by memory type
            tags: Filter by tags
            namespace_id: Filter by namespace
            min_score: Minimum BM25 score (default 0)

        Returns:
            List of SearchResult objects sorted by relevance
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Build search query with BM25 ranking
        # Weight: content=10, summary=5, tags=2
        sql = """
            SELECT
                m.*,
                bm25(memories_fts, 0, 10.0, 5.0, 2.0, 0, 0) as rank
            FROM memories m
            JOIN memories_fts fts ON m.id = fts.id
            WHERE memories_fts MATCH ?
        """
        params = [self._prepare_query(query)]

        if namespace_id:
            sql += " AND fts.namespace_id = ?"
            params.append(namespace_id)

        if type:
            sql += " AND fts.type = ?"
            params.append(type.value)

        sql += " ORDER BY rank LIMIT ?"
        params.append(limit * 2)  # Get extra for tag filtering

        try:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
        except sqlite3.OperationalError:
            # Invalid FTS query syntax, try simple search
            return self._simple_search(query, limit, type, namespace_id)
        finally:
            conn.close()

        # Process results
        results = []
        for row in rows:
            memory = self._row_to_memory(row[:-1])  # Exclude rank column
            rank = row[-1]

            # Normalize BM25 score to 0-1 range (BM25 is negative, lower = better)
            score = 1.0 / (1.0 + abs(rank))

            if score < min_score:
                continue

            # Filter by tags if specified
            if tags and not any(t in memory.tags for t in tags):
                continue

            results.append(SearchResult(
                memory=memory,
                score=score,
                highlights=self._get_highlights(memory.content, query),
            ))

            if len(results) >= limit:
                break

        return results

    def _prepare_query(self, query: str) -> str:
        """
        Prepare query for FTS5.

        Handles:
        - Phrase queries: "exact phrase"
        - Prefix queries: word*
        - Boolean: AND, OR, NOT
        - Column targeting: content:word
        """
        # Escape special characters if not using advanced syntax
        if not any(c in query for c in ['"', '*', 'AND', 'OR', 'NOT', ':']):
            # Split into words and add prefix matching for better recall
            words = query.strip().split()
            if len(words) == 1:
                return f"{words[0]}*"
            else:
                # Match any word with prefix
                return " OR ".join(f"{w}*" for w in words)

        return query

    def _simple_search(
        self,
        query: str,
        limit: int,
        type: Optional[MemoryType],
        namespace_id: Optional[str],
    ) -> list[SearchResult]:
        """Fallback LIKE-based search."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        sql = "SELECT * FROM memories WHERE (content LIKE ? OR summary LIKE ?)"
        params = [f"%{query}%", f"%{query}%"]

        if namespace_id:
            sql += " AND namespace_id = ?"
            params.append(namespace_id)

        if type:
            sql += " AND type = ?"
            params.append(type.value)

        sql += f" ORDER BY created_at DESC LIMIT {limit}"

        cursor.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()

        return [
            SearchResult(memory=self._row_to_memory(row), score=0.5)
            for row in rows
        ]

    def _get_highlights(self, content: str, query: str, context: int = 50) -> list[str]:
        """Extract highlighted snippets around query matches."""
        highlights = []
        words = query.lower().replace('"', '').split()
        content_lower = content.lower()

        for word in words:
            word_clean = word.rstrip('*')
            idx = content_lower.find(word_clean)
            if idx >= 0:
                start = max(0, idx - context)
                end = min(len(content), idx + len(word_clean) + context)
                snippet = content[start:end]
                if start > 0:
                    snippet = "..." + snippet
                if end < len(content):
                    snippet = snippet + "..."
                highlights.append(snippet)

        return highlights[:3]  # Max 3 highlights

    def _row_to_memory(self, row) -> Memory:
        """Convert database row to Memory object."""
        return Memory(
            id=row[0],
            content=row[1],
            type=MemoryType(row[2]),
            tags=json.loads(row[3]) if row[3] else [],
            summary=row[4],
            namespace_id=row[5],
            source_file=row[6],
            source_repo=row[7],
            session_id=row[8],
            created_at=datetime.fromisoformat(row[9]),
            updated_at=datetime.fromisoformat(row[10]),
            metadata=json.loads(row[11]) if row[11] else {},
        )

    def get_stats(self) -> dict:
        """Get FTS index statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM memories_fts")
        count = cursor.fetchone()[0]

        conn.close()

        return {
            "indexed_memories": count,
            "tokenizer": "porter unicode61",
        }


class HybridSearch:
    """
    Combine FTS and RAG search for best results.

    Uses FTS for fast keyword matching and RAG for semantic understanding.
    Results are merged using Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, fts_backend: FTSBackend, rag_backend=None):
        """
        Initialize hybrid search.

        Args:
            fts_backend: FTS5 search backend
            rag_backend: RAG search backend (optional)
        """
        self.fts = fts_backend
        self.rag = rag_backend

    def search(
        self,
        query: str,
        limit: int = 10,
        type: Optional[MemoryType] = None,
        tags: Optional[list[str]] = None,
        namespace_id: Optional[str] = None,
        fts_weight: float = 0.4,
        rag_weight: float = 0.6,
    ) -> list[SearchResult]:
        """
        Hybrid search combining FTS and RAG.

        Args:
            query: Search query
            limit: Maximum results
            type: Filter by memory type
            tags: Filter by tags
            namespace_id: Filter by namespace
            fts_weight: Weight for FTS results (0-1)
            rag_weight: Weight for RAG results (0-1)

        Returns:
            Merged list of SearchResult objects
        """
        # Get FTS results
        fts_results = self.fts.search(
            query=query,
            limit=limit * 2,
            type=type,
            tags=tags,
            namespace_id=namespace_id,
        )

        # If no RAG backend, return FTS results
        if self.rag is None:
            return fts_results[:limit]

        # Get RAG results
        rag_results = self.rag.search(
            query=query,
            limit=limit * 2,
            type=type,
            tags=tags,
            namespace_id=namespace_id,
        )

        # Merge using Reciprocal Rank Fusion
        return self._rrf_merge(fts_results, rag_results, fts_weight, rag_weight, limit)

    def _rrf_merge(
        self,
        fts_results: list[SearchResult],
        rag_results: list[SearchResult],
        fts_weight: float,
        rag_weight: float,
        limit: int,
        k: int = 60,
    ) -> list[SearchResult]:
        """
        Merge results using Reciprocal Rank Fusion.

        RRF score = sum(weight / (k + rank))
        """
        scores: dict[str, float] = {}
        memories: dict[str, Memory] = {}
        highlights: dict[str, list[str]] = {}

        # Score FTS results
        for rank, result in enumerate(fts_results):
            memory_id = result.memory.id
            scores[memory_id] = scores.get(memory_id, 0) + fts_weight / (k + rank + 1)
            memories[memory_id] = result.memory
            highlights[memory_id] = result.highlights

        # Score RAG results
        for rank, result in enumerate(rag_results):
            memory_id = result.memory.id
            scores[memory_id] = scores.get(memory_id, 0) + rag_weight / (k + rank + 1)
            if memory_id not in memories:
                memories[memory_id] = result.memory
                highlights[memory_id] = result.highlights

        # Sort by combined score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # Build final results
        results = []
        for memory_id in sorted_ids[:limit]:
            # Normalize score to 0-1
            max_score = (fts_weight + rag_weight) / (k + 1)
            normalized_score = min(1.0, scores[memory_id] / max_score)

            results.append(SearchResult(
                memory=memories[memory_id],
                score=normalized_score,
                highlights=highlights.get(memory_id, []),
            ))

        return results
