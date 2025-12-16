"""
Generic text file handler.

Fallback handler for unrecognized file types.
Uses simple line-based chunking.
"""

import logging
from pathlib import Path

from contextfs.filetypes.base import (
    ChunkStrategy,
    DocumentChunk,
    DocumentNode,
    FileTypeHandler,
    NodeType,
    ParsedDocument,
    Relationship,
    SourceLocation,
)

logger = logging.getLogger(__name__)


class GenericTextHandler(FileTypeHandler):
    """Generic handler for text files."""

    name: str = "generic"
    extensions: list[str] = [".txt", ".text", ".log", ".rst", ".org", ".adoc"]
    mime_types: list[str] = ["text/plain"]
    chunk_strategy: ChunkStrategy = ChunkStrategy.FIXED_SIZE
    default_chunk_size: int = 1000

    def parse(self, content: str, file_path: str) -> ParsedDocument:
        """Parse generic text file."""
        lines = content.split("\n")

        root = DocumentNode(
            type=NodeType.ROOT,
            name=Path(file_path).stem,
            content=content,
            location=SourceLocation(start_line=1, end_line=len(lines)),
        )

        doc = ParsedDocument(
            file_path=file_path,
            file_type="text",
            raw_content=content,
            root=root,
            line_count=len(lines),
            char_count=len(content),
            node_count=1,
        )

        doc.chunks = self.chunk(doc)
        return doc

    def chunk(
        self,
        document: ParsedDocument,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[DocumentChunk]:
        """Chunk text by paragraphs or fixed size."""
        chunk_size = chunk_size or self.default_chunk_size
        chunk_overlap = chunk_overlap or self.default_chunk_overlap

        content = document.raw_content
        chunks: list[DocumentChunk] = []

        # Try paragraph-based chunking first
        paragraphs = content.split("\n\n")

        if len(paragraphs) > 1:
            current_chunk = []
            current_tokens = 0

            for para in paragraphs:
                para_tokens = self._count_tokens(para)

                if current_tokens + para_tokens > chunk_size and current_chunk:
                    chunk_content = "\n\n".join(current_chunk)
                    chunk = DocumentChunk(
                        content=chunk_content,
                        document_id=document.id,
                        file_path=document.file_path,
                        chunk_index=len(chunks),
                        strategy=ChunkStrategy.SEMANTIC,
                    )
                    chunk.embedding_text = self.prepare_for_embedding(chunk, document)
                    chunks.append(chunk)

                    # Overlap: keep last paragraph
                    current_chunk = [current_chunk[-1]] if current_chunk else []
                    current_tokens = self._count_tokens(current_chunk[0]) if current_chunk else 0

                current_chunk.append(para)
                current_tokens += para_tokens

            if current_chunk:
                chunk_content = "\n\n".join(current_chunk)
                chunk = DocumentChunk(
                    content=chunk_content,
                    document_id=document.id,
                    file_path=document.file_path,
                    chunk_index=len(chunks),
                    strategy=ChunkStrategy.SEMANTIC,
                )
                chunk.embedding_text = self.prepare_for_embedding(chunk, document)
                chunks.append(chunk)

        else:
            # Fixed-size chunking
            chunks = self._fixed_size_chunk(content, chunk_size, chunk_overlap, document)

        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks

    def _fixed_size_chunk(
        self,
        content: str,
        chunk_size: int,
        overlap: int,
        document: ParsedDocument,
    ) -> list[DocumentChunk]:
        """Chunk by fixed character size."""
        chunks: list[DocumentChunk] = []
        char_chunk_size = chunk_size * 4  # Approximate chars per token
        char_overlap = overlap * 4

        start = 0
        while start < len(content):
            end = min(start + char_chunk_size, len(content))

            # Try to break at sentence or word boundary
            if end < len(content):
                # Look for sentence end
                for i in range(end, max(start, end - 100), -1):
                    if content[i] in ".!?\n":
                        end = i + 1
                        break
                else:
                    # Look for word boundary
                    for i in range(end, max(start, end - 50), -1):
                        if content[i] == " ":
                            end = i + 1
                            break

            chunk_content = content[start:end]
            chunk = DocumentChunk(
                content=chunk_content,
                document_id=document.id,
                file_path=document.file_path,
                chunk_index=len(chunks),
                strategy=ChunkStrategy.FIXED_SIZE,
            )
            chunk.embedding_text = self.prepare_for_embedding(chunk, document)
            chunks.append(chunk)

            # Move start forward, ensuring progress (avoid infinite loop for small files)
            new_start = end - char_overlap
            start = end if new_start <= start else new_start

        return chunks

    def extract_relationships(self, document: ParsedDocument) -> list[Relationship]:
        """Generic files typically have no relationships."""
        return []

    def prepare_for_embedding(
        self,
        chunk: DocumentChunk,
        document: ParsedDocument,
    ) -> str:
        """Prepare generic text for embedding."""
        parts = [
            f"File: {Path(document.file_path).name}",
            chunk.content,
        ]
        return "\n\n".join(parts)
