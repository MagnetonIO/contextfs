"""
RAG Backend for ContextFS.

Provides semantic search using ChromaDB and sentence-transformers.
Integrated from local_rag_pipeline.
"""

import json
from pathlib import Path

from contextfs.schemas import Memory, MemoryType, SearchResult


class RAGBackend:
    """
    RAG backend using ChromaDB and sentence-transformers.

    Provides:
    - Semantic embedding generation
    - Vector similarity search
    - Hybrid search (semantic + keyword)
    """

    def __init__(
        self,
        data_dir: Path,
        embedding_model: str = "all-MiniLM-L6-v2",
        collection_name: str = "contextfs_memories",
    ):
        """
        Initialize RAG backend.

        Args:
            data_dir: Directory for ChromaDB storage
            embedding_model: Sentence transformer model name
            collection_name: ChromaDB collection name
        """
        self.data_dir = data_dir
        self.embedding_model_name = embedding_model
        self.collection_name = collection_name

        self._chroma_dir = data_dir / "chroma_db"
        self._chroma_dir.mkdir(parents=True, exist_ok=True)

        # Lazy initialization
        self._client = None
        self._collection = None
        self._embedding_model = None

    def _ensure_initialized(self) -> None:
        """Lazy initialize ChromaDB and embedding model."""
        if self._client is not None:
            return

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

        except ImportError:
            raise ImportError("ChromaDB not installed. Install with: pip install chromadb")

        try:
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer(self.embedding_model_name)

        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Install with: pip install sentence-transformers"
            )

    def _get_embedding(self, text: str) -> list[float]:
        """Generate embedding for text."""
        self._ensure_initialized()
        embedding = self._embedding_model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def _get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts in batch (much faster)."""
        self._ensure_initialized()
        embeddings = self._embedding_model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        )
        return embeddings.tolist()

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

        embedding = self._get_embedding(text)

        # Store in ChromaDB
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

        # Batch encode all texts at once (much faster)
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

        # Add all at once to ChromaDB
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
