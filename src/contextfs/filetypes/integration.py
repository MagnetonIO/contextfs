"""
Integration between file type system and RAG backend.

Provides smart document processing with:
- Type-aware parsing and chunking
- Relationship-enhanced embeddings
- Cross-reference indexing
"""

import logging
from pathlib import Path

from contextfs.filetypes.base import (
    ChunkStrategy,
    ParsedDocument,
)
from contextfs.filetypes.linker import CrossReferenceLinker
from contextfs.filetypes.registry import FileTypeRegistry, get_handler
from contextfs.schemas import Memory, MemoryType

logger = logging.getLogger(__name__)


class SmartDocumentProcessor:
    """
    Process documents using file type-specific handlers.

    Uses AST parsing, semantic chunking, and relationship
    extraction for optimal embedding preparation.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        linker: CrossReferenceLinker | None = None,
    ):
        """
        Initialize processor.

        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks
            linker: Optional cross-reference linker for relationship tracking
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.linker = linker or CrossReferenceLinker()
        self.registry = FileTypeRegistry()
        self._parsed_cache: dict[str, ParsedDocument] = {}

    def process_file(self, file_path: Path) -> list[dict]:
        """
        Process a file using type-specific handler.

        Args:
            file_path: Path to file

        Returns:
            List of chunks with content, metadata, and embedding text
        """
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")
            return []

        handler = get_handler(str(file_path))
        if not handler:
            # Fall back to generic processing
            return self._process_generic(file_path, content)

        try:
            # Parse with type-specific handler
            doc = handler.parse(content, str(file_path))
            self._parsed_cache[str(file_path)] = doc

            # Add to linker for cross-reference tracking
            self.linker.add_document(doc)

            # Convert chunks to result format
            results = []
            for chunk in doc.chunks:
                results.append(
                    {
                        "content": chunk.content,
                        "embedding_text": chunk.embedding_text or chunk.content,
                        "metadata": {
                            "source_file": str(file_path),
                            "file_type": doc.file_type,
                            "language": doc.language,
                            "chunk_index": chunk.chunk_index,
                            "total_chunks": chunk.total_chunks,
                            "strategy": chunk.strategy.value if chunk.strategy else "unknown",
                            "node_ids": chunk.node_ids,
                            "summary": chunk.summary,
                            "keywords": chunk.keywords,
                            "document_id": doc.id,
                            "breadcrumb": chunk.breadcrumb,
                        },
                    }
                )

            return results

        except Exception as e:
            logger.warning(f"Type-specific parsing failed for {file_path}: {e}")
            return self._process_generic(file_path, content)

    def _process_generic(self, file_path: Path, content: str) -> list[dict]:
        """Fallback generic processing."""
        from contextfs.filetypes.handlers.generic import GenericTextHandler

        handler = GenericTextHandler()
        doc = handler.parse(content, str(file_path))

        results = []
        for chunk in doc.chunks:
            results.append(
                {
                    "content": chunk.content,
                    "embedding_text": chunk.embedding_text or chunk.content,
                    "metadata": {
                        "source_file": str(file_path),
                        "file_type": "text",
                        "chunk_index": chunk.chunk_index,
                        "total_chunks": chunk.total_chunks,
                        "strategy": ChunkStrategy.FIXED_SIZE.value,
                        "document_id": doc.id,
                    },
                }
            )

        return results

    def process_directory(
        self,
        directory: Path,
        extensions: list[str] | None = None,
        recursive: bool = True,
    ) -> list[dict]:
        """
        Process all files in a directory.

        Args:
            directory: Directory path
            extensions: Filter by extensions (e.g., ['.py', '.md'])
            recursive: Process subdirectories

        Returns:
            List of all chunks from all files
        """
        all_chunks = []
        pattern = "**/*" if recursive else "*"

        for file_path in directory.glob(pattern):
            if not file_path.is_file():
                continue

            if extensions and file_path.suffix not in extensions:
                continue

            chunks = self.process_file(file_path)
            all_chunks.extend(chunks)

        # After processing all files, resolve cross-references
        self.linker.link_all()

        return all_chunks

    def get_document(self, file_path: str) -> ParsedDocument | None:
        """Get cached parsed document."""
        return self._parsed_cache.get(file_path)

    def get_relationships(self, file_path: str) -> list[dict]:
        """Get relationships for a file."""
        doc = self._parsed_cache.get(file_path)
        if not doc:
            return []

        rels = self.linker.extractor.get_outgoing(doc.id)
        return [rel.model_dump() for rel in rels]

    def get_cross_references(self, file_path: str) -> list[dict]:
        """Get cross-references for a file."""
        doc = self._parsed_cache.get(file_path)
        if not doc:
            return []

        xrefs = self.linker.get_references_from(doc.id)
        return [xref.model_dump() for xref in xrefs]


class RAGIntegration:
    """
    Integrates file type system with RAG backend.

    Provides enhanced semantic search with:
    - File type-aware embeddings
    - Relationship context
    - Cross-reference boosting
    """

    def __init__(self, rag_backend, processor: SmartDocumentProcessor | None = None):
        """
        Initialize integration.

        Args:
            rag_backend: RAGBackend instance
            processor: Optional SmartDocumentProcessor
        """
        self.rag = rag_backend
        self.processor = processor or SmartDocumentProcessor()
        self._chunk_to_memory: dict[str, str] = {}

    def index_file(self, file_path: Path, namespace_id: str = "global") -> list[str]:
        """
        Index a file into the RAG backend.

        Args:
            file_path: Path to file
            namespace_id: Namespace for memories

        Returns:
            List of created memory IDs
        """
        chunks = self.processor.process_file(file_path)
        memory_ids = []

        for chunk in chunks:
            # Create memory from chunk
            memory = Memory(
                content=chunk["content"],
                type=MemoryType.FACT,
                tags=self._extract_tags(chunk),
                summary=chunk["metadata"].get("summary"),
                namespace_id=namespace_id,
            )

            # Use embedding text if available
            embedding_text = chunk.get("embedding_text", chunk["content"])

            # Add to RAG with custom embedding text
            self._add_with_embedding_text(memory, embedding_text)
            memory_ids.append(memory.id)

            # Track chunk-to-memory mapping
            doc_id = chunk["metadata"].get("document_id", "")
            chunk_idx = chunk["metadata"].get("chunk_index", 0)
            self._chunk_to_memory[f"{doc_id}:{chunk_idx}"] = memory.id

        return memory_ids

    def _add_with_embedding_text(self, memory: Memory, embedding_text: str) -> None:
        """Add memory with custom embedding text."""
        self.rag._ensure_initialized()

        embedding = self.rag._get_embedding(embedding_text)

        import json

        self.rag._collection.add(
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
                }
            ],
        )

    def _extract_tags(self, chunk: dict) -> list[str]:
        """Extract tags from chunk metadata."""
        tags = []
        metadata = chunk.get("metadata", {})

        if metadata.get("file_type"):
            tags.append(f"type:{metadata['file_type']}")

        if metadata.get("language"):
            tags.append(f"lang:{metadata['language']}")

        if metadata.get("keywords"):
            tags.extend(metadata["keywords"][:5])

        if metadata.get("source_file"):
            ext = Path(metadata["source_file"]).suffix
            if ext:
                tags.append(f"ext:{ext}")

        return tags

    def index_directory(
        self,
        directory: Path,
        namespace_id: str = "global",
        extensions: list[str] | None = None,
    ) -> dict:
        """
        Index all files in a directory.

        Args:
            directory: Directory path
            namespace_id: Namespace for memories
            extensions: Filter by extensions

        Returns:
            Stats about indexing
        """
        chunks = self.processor.process_directory(directory, extensions)

        memory_ids = []
        for chunk in chunks:
            memory = Memory(
                content=chunk["content"],
                type=MemoryType.FACT,
                tags=self._extract_tags(chunk),
                summary=chunk["metadata"].get("summary"),
                namespace_id=namespace_id,
            )

            embedding_text = chunk.get("embedding_text", chunk["content"])
            self._add_with_embedding_text(memory, embedding_text)
            memory_ids.append(memory.id)

        # Get cross-reference stats
        xref_stats = self.processor.linker.to_dict()

        return {
            "files_processed": len({c["metadata"]["source_file"] for c in chunks}),
            "chunks_created": len(chunks),
            "memories_added": len(memory_ids),
            "relationships": xref_stats.get("total", 0),
            "cross_references": len(self.processor.linker.cross_references),
        }

    def search_with_context(
        self,
        query: str,
        limit: int = 10,
        include_related: bool = True,
        namespace_id: str | None = None,
    ) -> list[dict]:
        """
        Search with relationship context.

        Args:
            query: Search query
            limit: Maximum results
            include_related: Include related documents
            namespace_id: Filter by namespace

        Returns:
            Search results with context
        """

        # Basic semantic search
        results = self.rag.search(
            query=query,
            limit=limit,
            namespace_id=namespace_id,
        )

        enhanced_results = []
        for result in results:
            enhanced = {
                "memory": result.memory.model_dump(),
                "score": result.score,
                "related": [],
            }

            if include_related:
                # Find related documents via cross-references
                related = self._find_related(result.memory.id)
                enhanced["related"] = related

            enhanced_results.append(enhanced)

        return enhanced_results

    def _find_related(self, memory_id: str) -> list[dict]:
        """Find related documents via cross-references."""
        related = []

        # Reverse lookup: find document/chunk for memory
        for chunk_key, mem_id in self._chunk_to_memory.items():
            if mem_id == memory_id:
                doc_id, chunk_idx = chunk_key.rsplit(":", 1)

                # Get cross-references for this document
                for xref in self.processor.linker.cross_references:
                    if xref.source_document_id == doc_id:
                        # Find memory for target
                        target_doc = self.processor.linker.index.get_by_id(xref.target_document_id)
                        if target_doc:
                            related.append(
                                {
                                    "type": xref.reference_type,
                                    "file_path": target_doc.file_path,
                                    "document_id": target_doc.id,
                                }
                            )

                break

        return related

    def get_file_summary(self, file_path: str) -> dict | None:
        """Get summary and structure of a parsed file."""
        doc = self.processor.get_document(file_path)
        if not doc:
            return None

        return {
            "file_path": doc.file_path,
            "file_type": doc.file_type,
            "language": doc.language,
            "title": doc.title,
            "line_count": doc.line_count,
            "char_count": doc.char_count,
            "node_count": doc.node_count,
            "chunk_count": len(doc.chunks),
            "symbols": list(doc.symbols.keys()),
            "metadata": doc.metadata,
            "relationships": self.processor.get_relationships(file_path),
            "cross_references": self.processor.get_cross_references(file_path),
        }

    def get_dependency_graph(self) -> dict:
        """Get document dependency graph."""
        return self.processor.linker.build_graph()
