"""
Integration tests for RAG backend.
"""

from pathlib import Path

import pytest


class TestRAGBackend:
    """Tests for RAG backend."""

    @pytest.mark.slow
    def test_add_and_search_memory(self, rag_backend, temp_dir: Path):
        """Test adding memory and searching."""
        from contextfs.schemas import Memory, MemoryType

        # Add memories
        memories = [
            Memory(
                content="Python is a programming language known for its simplicity.",
                type=MemoryType.FACT,
                tags=["python", "programming"],
            ),
            Memory(
                content="Rust is a systems programming language focused on safety.",
                type=MemoryType.FACT,
                tags=["rust", "programming"],
            ),
            Memory(
                content="Docker containers provide application isolation.",
                type=MemoryType.FACT,
                tags=["docker", "devops"],
            ),
        ]

        for memory in memories:
            rag_backend.add_memory(memory)

        # Search
        results = rag_backend.search("programming languages", limit=5)

        assert len(results) >= 2
        # Should find Python and Rust related content
        content = " ".join(r.memory.content for r in results)
        assert "Python" in content or "Rust" in content

    @pytest.mark.slow
    def test_search_with_filters(self, rag_backend, temp_dir: Path):
        """Test searching with tag filters."""
        from contextfs.schemas import Memory, MemoryType

        # Add memories
        rag_backend.add_memory(
            Memory(
                content="Python web frameworks include Django and Flask.",
                type=MemoryType.FACT,
                tags=["python", "web"],
            )
        )
        rag_backend.add_memory(
            Memory(
                content="JavaScript frameworks include React and Vue.",
                type=MemoryType.FACT,
                tags=["javascript", "web"],
            )
        )

        # Search with tag filter
        results = rag_backend.search(
            "web frameworks",
            tags=["python"],
            limit=5,
        )

        assert len(results) >= 1
        assert any("Python" in r.memory.content for r in results)

    @pytest.mark.slow
    def test_update_memory(self, rag_backend, temp_dir: Path):
        """Test updating memory."""
        from contextfs.schemas import Memory, MemoryType

        memory = Memory(
            content="Original content",
            type=MemoryType.FACT,
        )

        rag_backend.add_memory(memory)

        # Update
        memory.content = "Updated content"
        rag_backend.update_memory(memory)

        # Search should find updated content
        results = rag_backend.search("updated", limit=5)
        assert any("Updated" in r.memory.content for r in results)

    @pytest.mark.slow
    def test_remove_memory(self, rag_backend, temp_dir: Path):
        """Test removing memory."""
        from contextfs.schemas import Memory, MemoryType

        memory = Memory(
            content="This memory will be deleted",
            type=MemoryType.FACT,
        )

        rag_backend.add_memory(memory)
        rag_backend.remove_memory(memory.id)

        # Should not find deleted memory
        results = rag_backend.search("deleted", limit=5)
        assert not any(r.memory.id == memory.id for r in results)

    @pytest.mark.slow
    def test_get_stats(self, rag_backend, temp_dir: Path):
        """Test getting backend stats."""
        from contextfs.schemas import Memory, MemoryType

        # Add some memories
        for i in range(3):
            rag_backend.add_memory(
                Memory(
                    content=f"Test memory {i}",
                    type=MemoryType.FACT,
                )
            )

        stats = rag_backend.get_stats()

        assert "total_memories" in stats
        assert stats["total_memories"] >= 3


class TestFTSBackend:
    """Tests for FTS backend."""

    def test_add_and_search(self, fts_backend, temp_dir: Path):
        """Test adding content and searching."""
        from contextfs.schemas import Memory, MemoryType

        # Add memories
        memories = [
            Memory(
                content="Python decorators are powerful metaprogramming tools.",
                type=MemoryType.FACT,
                tags=["python"],
            ),
            Memory(
                content="JavaScript async/await simplifies asynchronous code.",
                type=MemoryType.FACT,
                tags=["javascript"],
            ),
        ]

        for memory in memories:
            fts_backend.add_memory(memory)

        # Search
        results = fts_backend.search("Python decorators", limit=5)

        assert len(results) >= 1
        assert any("Python" in r.memory.content for r in results)

    def test_bm25_ranking(self, fts_backend, temp_dir: Path):
        """Test BM25 ranking."""
        from contextfs.schemas import Memory, MemoryType

        # Add memories with varying relevance
        fts_backend.add_memory(
            Memory(
                content="Python Python Python - very relevant",
                type=MemoryType.FACT,
            )
        )
        fts_backend.add_memory(
            Memory(
                content="Python is mentioned once",
                type=MemoryType.FACT,
            )
        )
        fts_backend.add_memory(
            Memory(
                content="No match here",
                type=MemoryType.FACT,
            )
        )

        results = fts_backend.search("Python", limit=5)

        # More relevant result should rank higher
        assert len(results) >= 2
        # First result should have higher score
        if len(results) >= 2:
            assert results[0].score >= results[1].score


class TestHybridSearch:
    """Tests for hybrid RAG + FTS search."""

    @pytest.mark.slow
    def test_hybrid_search(self, rag_backend, fts_backend, temp_dir: Path):
        """Test hybrid search combining RAG and FTS."""
        from contextfs.fts import HybridSearch
        from contextfs.schemas import Memory, MemoryType

        hybrid = HybridSearch(rag=rag_backend, fts=fts_backend)

        # Add memories to both backends
        memories = [
            Memory(
                content="Machine learning models require training data.",
                type=MemoryType.FACT,
            ),
            Memory(
                content="Deep learning uses neural networks with many layers.",
                type=MemoryType.FACT,
            ),
            Memory(
                content="Supervised learning needs labeled datasets.",
                type=MemoryType.FACT,
            ),
        ]

        for memory in memories:
            rag_backend.add_memory(memory)
            fts_backend.add_memory(memory)

        # Search
        results = hybrid.search("machine learning training", limit=5)

        assert len(results) >= 1


class TestSmartDocumentProcessor:
    """Tests for smart document processor."""

    def test_process_python_file(self, temp_dir: Path, sample_python_code: str):
        """Test processing Python file."""
        from contextfs.filetypes.integration import SmartDocumentProcessor

        # Create test file
        test_file = temp_dir / "test.py"
        test_file.write_text(sample_python_code)

        processor = SmartDocumentProcessor()
        chunks = processor.process_file(test_file)

        assert len(chunks) > 0
        assert all("embedding_text" in c for c in chunks)
        assert all(c["metadata"]["file_type"] == "python" for c in chunks)

    def test_process_directory(
        self, temp_dir: Path, sample_python_code: str, sample_typescript_code: str
    ):
        """Test processing directory."""
        from contextfs.filetypes.integration import SmartDocumentProcessor

        # Create test files
        (temp_dir / "app.py").write_text(sample_python_code)
        (temp_dir / "service.ts").write_text(sample_typescript_code)

        processor = SmartDocumentProcessor()
        chunks = processor.process_directory(temp_dir, extensions=[".py", ".ts"])

        assert len(chunks) > 0
        # Should have chunks from both files
        file_types = {c["metadata"]["file_type"] for c in chunks}
        assert "python" in file_types
        assert "typescript" in file_types


class TestRAGIntegration:
    """Tests for RAG integration with file types."""

    @pytest.mark.slow
    def test_index_file(self, rag_backend, temp_dir: Path, sample_python_code: str):
        """Test indexing a file."""
        from contextfs.filetypes.integration import RAGIntegration

        # Create test file
        test_file = temp_dir / "test.py"
        test_file.write_text(sample_python_code)

        integration = RAGIntegration(rag_backend)
        memory_ids = integration.index_file(test_file)

        assert len(memory_ids) > 0

        # Should be searchable
        results = rag_backend.search("MyClass greet", limit=5)
        assert len(results) > 0

    @pytest.mark.slow
    def test_index_directory(
        self, rag_backend, temp_dir: Path, sample_python_code: str, sample_go_code: str
    ):
        """Test indexing a directory."""
        from contextfs.filetypes.integration import RAGIntegration

        # Create test files
        (temp_dir / "main.py").write_text(sample_python_code)
        (temp_dir / "service.go").write_text(sample_go_code)

        integration = RAGIntegration(rag_backend)
        stats = integration.index_directory(temp_dir, extensions=[".py", ".go"])

        assert stats["files_processed"] == 2
        assert stats["memories_added"] > 0

    @pytest.mark.slow
    def test_search_with_context(self, rag_backend, temp_dir: Path, sample_python_code: str):
        """Test search with relationship context."""
        from contextfs.filetypes.integration import RAGIntegration

        # Create test file
        test_file = temp_dir / "test.py"
        test_file.write_text(sample_python_code)

        integration = RAGIntegration(rag_backend)
        integration.index_file(test_file)

        results = integration.search_with_context("class methods", limit=5)

        assert len(results) > 0
        assert "memory" in results[0]
        assert "score" in results[0]
