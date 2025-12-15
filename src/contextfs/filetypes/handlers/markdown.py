"""
Markdown file handler with structure extraction.

Extracts:
- Headers and document structure
- Code blocks with language
- Links and images
- Frontmatter (YAML)
- Lists and blockquotes
"""

import logging
import re
from pathlib import Path
from typing import ClassVar

from contextfs.filetypes.base import (
    ChunkStrategy,
    DocumentChunk,
    DocumentNode,
    FileTypeHandler,
    NodeType,
    ParsedDocument,
    Relationship,
    RelationType,
    SourceLocation,
)

logger = logging.getLogger(__name__)


class MarkdownHandler(FileTypeHandler):
    """Handler for Markdown files."""

    name: str = "markdown"
    extensions: list[str] = [".md", ".markdown", ".mdown", ".mkd", ".mdx"]
    mime_types: list[str] = ["text/markdown", "text/x-markdown"]
    chunk_strategy: ChunkStrategy = ChunkStrategy.AST_BOUNDARY

    # Patterns
    HEADER_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    CODE_BLOCK_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
    LINK_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    IMAGE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
    FRONTMATTER_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"^---\n(.*?)\n---", re.DOTALL)

    def parse(self, content: str, file_path: str) -> ParsedDocument:
        """Parse Markdown file."""
        lines = content.split("\n")

        root = DocumentNode(
            type=NodeType.DOCUMENT,
            name=Path(file_path).stem,
            content="",
            location=SourceLocation(start_line=1, end_line=len(lines)),
        )

        symbols: dict[str, DocumentNode] = {}
        references: list[DocumentNode] = []

        # Extract frontmatter
        frontmatter = self._extract_frontmatter(content)
        if frontmatter:
            fm_node = DocumentNode(
                type=NodeType.FRONTMATTER,
                content=frontmatter,
                location=SourceLocation(start_line=1, end_line=frontmatter.count("\n") + 3),
                parent_id=root.id,
            )
            root.children.append(fm_node)

        # Parse headers into hierarchy
        self._parse_headers(content, root, symbols)

        # Extract code blocks
        self._extract_code_blocks(content, root)

        # Extract links
        self._extract_links(content, references)

        # Create document
        doc = ParsedDocument(
            file_path=file_path,
            file_type="markdown",
            raw_content=content,
            root=root,
            symbols=symbols,
            references=references,
            language="markdown",
            line_count=len(lines),
            char_count=len(content),
            node_count=len(root.walk()),
            title=self._extract_title(content),
            metadata={
                "has_frontmatter": frontmatter is not None,
                "code_blocks": len([n for n in root.walk() if n.type == NodeType.CODE_BLOCK]),
                "links": len(references),
            },
        )

        doc.chunks = self.chunk(doc)
        return doc

    def _extract_frontmatter(self, content: str) -> str | None:
        """Extract YAML frontmatter."""
        match = self.FRONTMATTER_PATTERN.match(content)
        return match.group(1) if match else None

    def _extract_title(self, content: str) -> str | None:
        """Extract document title from first H1."""
        for match in self.HEADER_PATTERN.finditer(content):
            if len(match.group(1)) == 1:  # H1
                return match.group(2).strip()
        return None

    def _parse_headers(
        self,
        content: str,
        root: DocumentNode,
        symbols: dict[str, DocumentNode],
    ) -> None:
        """Parse headers into hierarchical structure."""
        lines = content.split("\n")
        header_stack: list[DocumentNode] = []
        current_content: list[tuple[int, str]] = []
        last_header: DocumentNode | None = None

        for line_num, line in enumerate(lines, 1):
            header_match = self.HEADER_PATTERN.match(line)

            if header_match:
                # Save content to previous header
                if last_header and current_content:
                    last_header.content = "\n".join(text for _, text in current_content)
                    last_header.location.end_line = current_content[-1][0]

                level = len(header_match.group(1))
                title = header_match.group(2).strip()

                header_node = DocumentNode(
                    type=NodeType.HEADING,
                    name=title,
                    content="",
                    location=SourceLocation(start_line=line_num, end_line=line_num),
                    attributes={"level": level},
                )

                # Manage hierarchy
                while header_stack and header_stack[-1].attributes.get("level", 0) >= level:
                    header_stack.pop()

                if header_stack:
                    header_node.parent_id = header_stack[-1].id
                    header_stack[-1].children.append(header_node)
                else:
                    header_node.parent_id = root.id
                    root.children.append(header_node)

                header_stack.append(header_node)
                last_header = header_node
                current_content = []

                # Create slug for symbol lookup
                slug = self._create_slug(title)
                symbols[slug] = header_node

            else:
                current_content.append((line_num, line))

        # Finalize last header
        if last_header and current_content:
            last_header.content = "\n".join(text for _, text in current_content)
            last_header.location.end_line = current_content[-1][0]

    def _create_slug(self, title: str) -> str:
        """Create URL-friendly slug from title."""
        slug = title.lower()
        slug = re.sub(r"[^\w\s-]", "", slug)
        slug = re.sub(r"\s+", "-", slug)
        return slug

    def _extract_code_blocks(self, content: str, root: DocumentNode) -> None:
        """Extract code blocks."""
        for match in self.CODE_BLOCK_PATTERN.finditer(content):
            language = match.group(1) or "text"
            code = match.group(2)
            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1

            code_node = DocumentNode(
                type=NodeType.CODE_BLOCK,
                content=code,
                location=SourceLocation(
                    start_line=line_num,
                    end_line=line_num + code.count("\n") + 2,
                ),
                parent_id=root.id,
                attributes={"language": language},
            )
            root.children.append(code_node)

    def _extract_links(self, content: str, references: list[DocumentNode]) -> None:
        """Extract links and images."""
        # Links
        for match in self.LINK_PATTERN.finditer(content):
            text = match.group(1)
            url = match.group(2)
            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1

            link_node = DocumentNode(
                type=NodeType.LINK,
                name=text,
                content=match.group(0),
                target=url,
                location=SourceLocation(start_line=line_num, end_line=line_num),
                attributes={"url": url, "text": text},
            )
            references.append(link_node)

        # Images
        for match in self.IMAGE_PATTERN.finditer(content):
            alt = match.group(1)
            url = match.group(2)
            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1

            img_node = DocumentNode(
                type=NodeType.IMAGE,
                name=alt,
                content=match.group(0),
                target=url,
                location=SourceLocation(start_line=line_num, end_line=line_num),
                attributes={"url": url, "alt": alt},
            )
            references.append(img_node)

    def chunk(
        self,
        document: ParsedDocument,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[DocumentChunk]:
        """Chunk Markdown by headers."""
        chunks: list[DocumentChunk] = []
        chunk_size = chunk_size or self.default_chunk_size

        # Get all headers
        headers = [n for n in document.root.walk() if n.type == NodeType.HEADING]

        for header in headers:
            if not header.content.strip():
                continue

            breadcrumb = self._create_breadcrumb(header, document)

            chunk = DocumentChunk(
                content=f"# {header.name}\n\n{header.content}",
                document_id=document.id,
                file_path=document.file_path,
                node_ids=[header.id],
                location=header.location,
                chunk_index=len(chunks),
                strategy=ChunkStrategy.AST_BOUNDARY,
                breadcrumb=breadcrumb,
                summary=f"Section: {header.name}",
            )
            chunk.embedding_text = self.prepare_for_embedding(chunk, document)
            chunks.append(chunk)

        # Update totals
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks

    def extract_relationships(self, document: ParsedDocument) -> list[Relationship]:
        """Extract relationships from Markdown links."""
        relationships: list[Relationship] = []

        for ref in document.references:
            if ref.type == NodeType.LINK:
                url = ref.attributes.get("url", "")

                # Determine relationship type
                if url.startswith("#"):
                    rel_type = RelationType.REFERENCES
                elif url.endswith(".md") or url.endswith(".markdown"):
                    rel_type = RelationType.LINKS_TO
                else:
                    rel_type = RelationType.LINKS_TO

                rel = Relationship(
                    type=rel_type,
                    source_document_id=document.id,
                    source_node_id=ref.id,
                    source_path=document.file_path,
                    target_name=url,
                    context=ref.content,
                    location=ref.location,
                    attributes={"link_text": ref.name},
                )
                relationships.append(rel)

        return relationships

    def prepare_for_embedding(
        self,
        chunk: DocumentChunk,
        document: ParsedDocument,
    ) -> str:
        """Prepare Markdown chunk for embedding."""
        parts = []

        if document.title:
            parts.append(f"Document: {document.title}")

        if chunk.breadcrumb:
            parts.append(f"Section: {' > '.join(chunk.breadcrumb)}")

        # Clean content
        content = self._clean_markdown(chunk.content)
        parts.append(content)

        return "\n\n".join(parts)

    def _clean_markdown(self, content: str) -> str:
        """Clean Markdown for embedding."""
        # Remove code blocks but keep description
        content = re.sub(r"```\w*\n.*?```", "[code block]", content, flags=re.DOTALL)
        # Remove inline code
        content = re.sub(r"`[^`]+`", "[code]", content)
        # Remove images
        content = re.sub(r"!\[[^\]]*\]\([^)]+\)", "[image]", content)
        # Simplify links
        content = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", content)
        return content.strip()
