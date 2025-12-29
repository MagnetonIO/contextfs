"""
RAG Backend for ContextFS.

Provides semantic search using ChromaDB and configurable embedding backends.
Supports FastEmbed (ONNX) or sentence-transformers with optional GPU acceleration.
"""

import fcntl
import json
import os
from pathlib import Path

from contextfs.schemas import Memory, MemoryType, SearchResult


class ChromaLock:
    """File-based lock for ChromaDB multi-process safety.

    Uses blocking exclusive lock to prevent concurrent ChromaDB access.
    This is critical because ChromaDB's internal SQLite can corrupt
    when multiple processes access it simultaneously.
    """

    def __init__(self, lock_path: Path):
        self.lock_path = lock_path
        self._lock_file = None
        self._lock_count = 0  # Allow reentrant locking

    def acquire(self, timeout: float = 60.0, blocking: bool = True) -> bool:
        """Acquire exclusive lock. Returns True if acquired.

        Args:
            timeout: Maximum time to wait for lock (seconds)
            blocking: If True, wait for lock; if False, fail immediately
        """
        import logging
        import time

        logger = logging.getLogger(__name__)

        # Reentrant: if we already hold the lock, just increment count
        if self._lock_count > 0 and self._lock_file is not None:
            self._lock_count += 1
            return True

        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_file = open(self.lock_path, "w")  # noqa: SIM115 - intentionally kept open for lock

        start = time.time()
        while True:
            try:
                if blocking:
                    # Use blocking lock with timeout check
                    fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                else:
                    fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._lock_count = 1
                return True
            except BlockingIOError:
                elapsed = time.time() - start
                if elapsed > timeout:
                    logger.warning(
                        f"ChromaDB lock timeout after {elapsed:.1f}s. "
                        "Another contextfs process may be using ChromaDB."
                    )
                    self._lock_file.close()
                    self._lock_file = None
                    return False
                # Wait with exponential backoff (max 1s)
                wait_time = min(0.1 * (2 ** (elapsed / 10)), 1.0)
                time.sleep(wait_time)

    def release(self) -> None:
        """Release the lock."""
        if self._lock_count > 1:
            self._lock_count -= 1
            return

        if self._lock_file:
            try:
                fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
                self._lock_file.close()
            except Exception:
                pass
            self._lock_file = None
            self._lock_count = 0

    def __enter__(self):
        if not self.acquire():
            raise TimeoutError("Could not acquire ChromaDB lock - another process may be using it")
        return self

    def __exit__(self, *args):
        self.release()


class RAGBackend:
    """
    RAG backend using ChromaDB and configurable embedding backends.

    Provides:
    - Semantic embedding generation (FastEmbed or SentenceTransformers)
    - Vector similarity search
    - Hybrid search (semantic + keyword)
    - Optional GPU acceleration
    """

    def __init__(
        self,
        data_dir: Path,
        embedding_model: str = "all-MiniLM-L6-v2",
        collection_name: str = "contextfs_memories",
        embedding_backend: str = "auto",
        use_gpu: bool | None = None,
        parallel_workers: int | None = None,
        chroma_host: str | None = None,
        chroma_port: int = 8000,
        chroma_auto_server: bool = True,
    ):
        """
        Initialize RAG backend.

        Args:
            data_dir: Directory for ChromaDB storage (embedded mode)
            embedding_model: Embedding model name
            collection_name: ChromaDB collection name
            embedding_backend: "fastembed", "sentence_transformers", or "auto"
            use_gpu: Enable GPU acceleration (None = auto-detect)
            parallel_workers: Number of parallel workers for embedding (None = auto)
            chroma_host: ChromaDB server host (None = embedded mode, else HttpClient)
            chroma_port: ChromaDB server port (default 8000)
            chroma_auto_server: Auto-start ChromaDB server on corruption (default True)
        """
        self.data_dir = data_dir
        self.embedding_model_name = embedding_model
        self.collection_name = collection_name
        self._embedding_backend = embedding_backend
        self._use_gpu = use_gpu
        self._parallel_workers = parallel_workers
        self._chroma_host = chroma_host
        self._chroma_port = chroma_port
        self._chroma_auto_server = chroma_auto_server
        self._server_mode = chroma_host is not None
        self._auto_start_attempted = False  # Track if we tried to auto-start server

        self._chroma_dir = data_dir / "chroma_db"
        self._chroma_dir.mkdir(parents=True, exist_ok=True)

        # File lock for multi-process safety (only needed in embedded mode)
        self._lock = ChromaLock(data_dir / "chroma.lock")

        # Lazy initialization
        self._client = None
        self._collection = None
        self._embedder = None
        self._needs_rebuild = False  # Set to True after auto-recovery from corruption

    def _ensure_initialized(self) -> None:
        """Lazy initialize ChromaDB and embedding model."""
        if self._client is not None:
            # Validate collection is still valid (handles stale references after rebuild)
            if not self._validate_collection():
                self._reinitialize_client()
            return

        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize ChromaDB client and collection with auto-recovery."""
        import logging

        logger = logging.getLogger(__name__)

        try:
            import chromadb
            from chromadb.config import Settings

            if self._server_mode:
                # Use HttpClient for server mode (no file locking issues)
                logger.debug(
                    f"Connecting to ChromaDB server at {self._chroma_host}:{self._chroma_port}"
                )
                self._client = chromadb.HttpClient(
                    host=self._chroma_host,
                    port=self._chroma_port,
                    settings=Settings(anonymized_telemetry=False),
                )
                self._collection = self._client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
            else:
                # Use PersistentClient for embedded mode (with locking)
                # Lock during initialization to prevent concurrent init race conditions
                with self._lock:
                    self._client = chromadb.PersistentClient(
                        path=str(self._chroma_dir),
                        settings=Settings(anonymized_telemetry=False),
                    )
                    self._collection = self._client.get_or_create_collection(
                        name=self.collection_name,
                        metadata={"hnsw:space": "cosine"},
                    )

        except ImportError:
            raise ImportError("ChromaDB not installed. Install with: pip install chromadb")

        except Exception as e:
            # Check for Rust panic (corrupted ChromaDB) - only in embedded mode
            if not self._server_mode:
                error_str = str(e)
                is_rust_panic = (
                    "PanicException" in type(e).__name__
                    or "range start index" in error_str
                    or "pyo3_runtime" in error_str
                    or "rust" in error_str.lower()
                )

                if is_rust_panic:
                    logger.warning(
                        f"ChromaDB corruption detected (Rust panic): {e}. Auto-recovering..."
                    )
                    self._auto_recover_chromadb()
                    return

            raise

    def _auto_recover_chromadb(self) -> None:
        """Auto-recover from ChromaDB corruption by starting a server.

        If chroma_auto_server is enabled, starts a ChromaDB server using the
        existing data directory (preserving embeddings). Falls back to rebuild
        if server start fails.
        """
        import logging

        logger = logging.getLogger(__name__)

        # Clean up corrupted client reference
        self._client = None
        self._collection = None

        # Try to auto-start ChromaDB server (preserves existing data)
        if self._chroma_auto_server and not self._auto_start_attempted:
            self._auto_start_attempted = True
            if self._try_auto_start_server():
                return  # Server started, we're now in server mode

        # If server didn't work, try to rebuild embedded mode
        logger.info("Rebuilding ChromaDB in embedded mode...")
        self._rebuild_embedded_chromadb()

    def _try_auto_start_server(self) -> bool:
        """Try to auto-start ChromaDB server using existing data.

        Returns True if server started and connected successfully.
        """
        import logging
        import subprocess
        import sys
        import time

        logger = logging.getLogger(__name__)

        # Check if server is already running
        if self._try_connect_to_server():
            logger.info("ChromaDB server already running, switching to server mode")
            self._server_mode = True
            self._chroma_host = "localhost"
            return True

        logger.info(f"Starting ChromaDB server with existing data at {self._chroma_dir}")

        try:
            # Start ChromaDB server in background using existing data
            cmd = [
                sys.executable,
                "-m",
                "chromadb.cli",
                "run",
                "--path",
                str(self._chroma_dir),
                "--host",
                "127.0.0.1",
                "--port",
                str(self._chroma_port),
            ]

            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

            # Wait for server to be ready
            for _ in range(30):  # 3 seconds max
                time.sleep(0.1)
                if self._try_connect_to_server():
                    logger.info("ChromaDB server started successfully")
                    self._server_mode = True
                    self._chroma_host = "127.0.0.1"
                    return True

            logger.warning("ChromaDB server did not start in time")
            return False

        except Exception as e:
            logger.warning(f"Failed to start ChromaDB server: {e}")
            return False

    def _try_connect_to_server(self) -> bool:
        """Try to connect to ChromaDB server. Returns True if successful."""
        try:
            import chromadb
            from chromadb.config import Settings

            client = chromadb.HttpClient(
                host="127.0.0.1",
                port=self._chroma_port,
                settings=Settings(anonymized_telemetry=False),
            )
            # Test connection
            client.heartbeat()

            # Connection successful, use this client
            self._client = client
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            return True
        except Exception:
            return False

    def _rebuild_embedded_chromadb(self) -> None:
        """Rebuild ChromaDB in embedded mode (last resort)."""
        import logging
        import shutil

        logger = logging.getLogger(__name__)

        # Remove corrupted data only if we must rebuild
        if self._chroma_dir.exists():
            logger.info(f"Removing corrupted ChromaDB at {self._chroma_dir}")
            shutil.rmtree(self._chroma_dir)

        self._chroma_dir.mkdir(parents=True, exist_ok=True)

        try:
            import chromadb
            from chromadb.config import Settings

            self._client = chromadb.PersistentClient(
                path=str(self._chroma_dir),
                settings=Settings(anonymized_telemetry=False),
            )

            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            logger.info("ChromaDB initialized fresh. Background rebuild needed.")
            self._needs_rebuild = True

        except Exception as e:
            logger.error(f"Failed to recover ChromaDB: {e}")
            raise RuntimeError(f"ChromaDB unrecoverable: {e}") from e

    @property
    def needs_rebuild(self) -> bool:
        """Check if ChromaDB needs to be rebuilt from SQLite."""
        return self._needs_rebuild

    def mark_rebuilt(self) -> None:
        """Mark ChromaDB as rebuilt (called after successful rebuild)."""
        self._needs_rebuild = False

    def _validate_collection(self) -> bool:
        """Check if collection reference is still valid."""
        if self._collection is None:
            return False
        try:
            # Try a lightweight operation to validate collection exists
            self._collection.count()
            return True
        except Exception:
            # Collection reference is stale
            return False

    def _reinitialize_client(self) -> None:
        """Reinitialize ChromaDB client (used when collection becomes stale)."""
        import logging

        logging.getLogger(__name__).info(
            "Reinitializing ChromaDB client (stale collection detected)"
        )
        self._client = None
        self._collection = None
        self._initialize_client()

    def _ensure_embedder(self) -> None:
        """Initialize embedding backend if not already done."""
        if self._embedder is not None:
            return

        from contextfs.embedding import create_embedder

        # Auto-detect GPU if not specified
        use_gpu = self._use_gpu
        if use_gpu is None:
            # Check environment variable
            use_gpu = os.environ.get("CONTEXTFS_USE_GPU", "").lower() in ("1", "true", "yes")

        self._embedder = create_embedder(
            model_name=self.embedding_model_name,
            backend=self._embedding_backend,
            use_gpu=use_gpu,
            parallel=self._parallel_workers,
        )

    def _get_embedding(self, text: str) -> list[float]:
        """Generate embedding for text."""
        self._ensure_initialized()
        self._ensure_embedder()
        return self._embedder.encode_single(text)

    def _get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts in batch (much faster)."""
        self._ensure_initialized()
        self._ensure_embedder()
        return self._embedder.encode(texts)

    def add_memory(self, memory: Memory) -> None:
        """
        Add a memory to the vector store.

        Args:
            memory: Memory object to add
        """
        self._ensure_initialized()

        # Combine content and summary for embedding
        text = memory.content
        if memory.summary:
            text = f"{memory.summary}\n{text}"

        # Generate embedding (outside lock - slow operation)
        embedding = self._get_embedding(text)

        # Store in ChromaDB (with lock for multi-process safety)
        with self._lock:
            self._collection.add(
                ids=[memory.id],
                embeddings=[embedding],
                documents=[memory.content],
                metadatas=[
                    {
                        "type": memory.type.value,
                        "tags": json.dumps(memory.tags),
                        "namespace_id": memory.namespace_id,
                        "summary": memory.summary or "",
                        "created_at": memory.created_at.isoformat(),
                        "source_repo": memory.source_repo or "",
                        "project": memory.project or "",
                        "source_tool": memory.source_tool or "",
                        "source_file": memory.source_file or "",
                    }
                ],
            )

    def add_memories_batch(self, memories: list[Memory]) -> int:
        """
        Add multiple memories in batch (much faster than individual adds).

        Args:
            memories: List of Memory objects to add

        Returns:
            Number of memories successfully added
        """
        if not memories:
            return 0

        self._ensure_initialized()

        # Prepare texts for batch embedding
        texts = []
        for memory in memories:
            text = memory.content
            if memory.summary:
                text = f"{memory.summary}\n{text}"
            texts.append(text)

        # Batch encode all texts at once (outside lock - slow operation)
        embeddings = self._get_embeddings_batch(texts)

        # Prepare batch data for ChromaDB
        ids = [m.id for m in memories]
        documents = [m.content for m in memories]
        metadatas = [
            {
                "type": m.type.value,
                "tags": json.dumps(m.tags),
                "namespace_id": m.namespace_id,
                "summary": m.summary or "",
                "created_at": m.created_at.isoformat(),
                "source_repo": m.source_repo or "",
                "project": m.project or "",
                "source_tool": m.source_tool or "",
                "source_file": m.source_file or "",
            }
            for m in memories
        ]

        # Add all at once to ChromaDB (with lock for multi-process safety)
        with self._lock:
            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

        return len(memories)

    def remove_memory(self, memory_id: str) -> None:
        """Remove a memory from the vector store."""
        self._ensure_initialized()

        try:
            with self._lock:
                self._collection.delete(ids=[memory_id])
        except Exception:
            pass  # Ignore if not found

    def delete_by_namespace(self, namespace_id: str) -> int:
        """
        Delete all memories in a namespace.

        Args:
            namespace_id: Namespace to clear

        Returns:
            Number of memories deleted
        """
        self._ensure_initialized()

        try:
            with self._lock:
                # Get all memory IDs in this namespace
                results = self._collection.get(
                    where={"namespace_id": namespace_id},
                    include=[],  # Don't need documents/embeddings, just IDs
                )

                ids_to_delete = results.get("ids", [])
                if ids_to_delete:
                    self._collection.delete(ids=ids_to_delete)
                    return len(ids_to_delete)
            return 0
        except Exception:
            return 0

    def search(
        self,
        query: str,
        limit: int = 10,
        type: MemoryType | None = None,
        tags: list[str] | None = None,
        namespace_id: str | None = None,
        min_score: float = 0.3,
    ) -> list[SearchResult]:
        """
        Search for similar memories.

        Args:
            query: Search query
            limit: Maximum results
            type: Filter by memory type
            tags: Filter by tags
            namespace_id: Filter by namespace
            min_score: Minimum similarity score (0-1)

        Returns:
            List of SearchResult objects
        """
        self._ensure_initialized()

        # Generate query embedding
        query_embedding = self._get_embedding(query)

        # Build where filter
        where = {}
        if namespace_id:
            where["namespace_id"] = namespace_id
        if type:
            where["type"] = type.value

        # Query ChromaDB
        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=limit * 2,  # Get extra for filtering
                where=where if where else None,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            # Return empty on error
            return []

        # Process results
        search_results = []

        if results and results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            documents = results["documents"][0] if results["documents"] else []
            metadatas = results["metadatas"][0] if results["metadatas"] else []
            distances = results["distances"][0] if results["distances"] else []

            for i, memory_id in enumerate(ids):
                # Convert distance to similarity score (cosine distance)
                distance = distances[i] if i < len(distances) else 1.0
                score = 1.0 - (distance / 2.0)  # Cosine distance to similarity

                if score < min_score:
                    continue

                metadata = metadatas[i] if i < len(metadatas) else {}

                # Filter by tags if specified
                if tags:
                    memory_tags = json.loads(metadata.get("tags", "[]"))
                    if not any(t in memory_tags for t in tags):
                        continue

                # Build Memory object
                from datetime import datetime

                memory = Memory(
                    id=memory_id,
                    content=documents[i] if i < len(documents) else "",
                    type=MemoryType(metadata.get("type", "fact")),
                    tags=json.loads(metadata.get("tags", "[]")),
                    summary=metadata.get("summary") or None,
                    namespace_id=metadata.get("namespace_id", "global"),
                    created_at=datetime.fromisoformat(
                        metadata.get("created_at", datetime.now().isoformat())
                    ),
                    source_repo=metadata.get("source_repo") or None,
                    project=metadata.get("project") or None,
                    source_tool=metadata.get("source_tool") or None,
                    source_file=metadata.get("source_file") or None,
                )

                search_results.append(
                    SearchResult(
                        memory=memory,
                        score=score,
                    )
                )

                if len(search_results) >= limit:
                    break

        return search_results

    def update_memory(self, memory: Memory) -> None:
        """Update a memory in the vector store."""
        self.remove_memory(memory.id)
        self.add_memory(memory)

    def get_stats(self) -> dict:
        """Get statistics about the vector store."""
        self._ensure_initialized()

        return {
            "total_memories": self._collection.count(),
            "embedding_model": self.embedding_model_name,
            "collection_name": self.collection_name,
        }

    def close(self) -> None:
        """Close the backend."""
        # ChromaDB handles cleanup automatically
        self._client = None
        self._collection = None
        self._embedding_model = None

    def reset_database(self) -> bool:
        """
        Reset the ChromaDB database by deleting and recreating it.

        Use this when the database becomes corrupted (e.g., after version upgrades).

        Returns:
            True if reset successful, False otherwise
        """
        import shutil

        try:
            # Close any open connections
            self.close()

            # Delete the ChromaDB directory
            if self._chroma_dir.exists():
                shutil.rmtree(self._chroma_dir)

            # Recreate the directory
            self._chroma_dir.mkdir(parents=True, exist_ok=True)

            # Re-initialize
            self._ensure_initialized()

            return True
        except Exception as e:
            import logging

            logging.getLogger(__name__).error(f"Failed to reset ChromaDB: {e}")
            return False


class DocumentProcessor:
    """
    Process documents for ingestion.

    Handles chunking, tokenization, and metadata extraction.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize document processor.

        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._tokenizer = None

    def _ensure_tokenizer(self) -> None:
        """Lazy initialize tokenizer."""
        if self._tokenizer is not None:
            return

        try:
            import tiktoken

            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            # Fallback to simple word-based counting
            self._tokenizer = None

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        self._ensure_tokenizer()

        if self._tokenizer:
            return len(self._tokenizer.encode(text))
        else:
            # Fallback: approximate 1 token per 4 characters
            return len(text) // 4

    def chunk_text(self, text: str) -> list[str]:
        """
        Split text into chunks.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        self._ensure_tokenizer()

        # Split by paragraphs first
        paragraphs = text.split("\n\n")

        chunks = []
        current_chunk = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self.count_tokens(para)

            # If single paragraph exceeds chunk size, split by sentences
            if para_tokens > self.chunk_size:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0

                # Split large paragraph
                sentences = para.replace(". ", ".\n").split("\n")
                for sentence in sentences:
                    sent_tokens = self.count_tokens(sentence)
                    if current_tokens + sent_tokens > self.chunk_size:
                        if current_chunk:
                            chunks.append(" ".join(current_chunk))
                        current_chunk = [sentence]
                        current_tokens = sent_tokens
                    else:
                        current_chunk.append(sentence)
                        current_tokens += sent_tokens

            elif current_tokens + para_tokens > self.chunk_size:
                # Start new chunk
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_tokens = para_tokens

            else:
                current_chunk.append(para)
                current_tokens += para_tokens

        # Don't forget last chunk
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def process_file(self, file_path: Path) -> list[dict]:
        """
        Process a file into chunks with metadata.

        Args:
            file_path: Path to file

        Returns:
            List of dicts with 'content' and 'metadata'
        """
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return []

        chunks = self.chunk_text(content)

        results = []
        for i, chunk in enumerate(chunks):
            results.append(
                {
                    "content": chunk,
                    "metadata": {
                        "source_file": str(file_path),
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                    },
                }
            )

        return results
