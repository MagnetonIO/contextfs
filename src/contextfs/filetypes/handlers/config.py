"""
Configuration file handlers (JSON, YAML, TOML).

Extracts:
- Key-value structure
- Nested objects and arrays
- Schema information
"""

import json
import logging
from pathlib import Path
from typing import Any

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


class ConfigHandlerBase(FileTypeHandler):
    """Base class for configuration file handlers."""

    chunk_strategy: ChunkStrategy = ChunkStrategy.FIXED_SIZE
    default_chunk_size: int = 500

    def _build_tree(
        self,
        data: Any,
        parent_id: str,
        path: str = "",
    ) -> list[DocumentNode]:
        """Build AST from parsed data."""
        nodes = []

        if isinstance(data, dict):
            for key, value in data.items():
                key_path = f"{path}.{key}" if path else key

                if isinstance(value, dict):
                    node = DocumentNode(
                        type=NodeType.OBJECT,
                        name=key,
                        content=json.dumps(value, indent=2),
                        parent_id=parent_id,
                        attributes={"path": key_path, "keys": list(value.keys())},
                    )
                    node.children = self._build_tree(value, node.id, key_path)
                    nodes.append(node)

                elif isinstance(value, list):
                    node = DocumentNode(
                        type=NodeType.ARRAY,
                        name=key,
                        content=json.dumps(value, indent=2),
                        parent_id=parent_id,
                        attributes={"path": key_path, "length": len(value)},
                    )
                    nodes.append(node)

                else:
                    node = DocumentNode(
                        type=NodeType.KEY_VALUE,
                        name=key,
                        content=f"{key}: {json.dumps(value)}",
                        parent_id=parent_id,
                        attributes={"path": key_path, "value": value, "type": type(value).__name__},
                    )
                    nodes.append(node)

        return nodes

    def chunk(
        self,
        document: ParsedDocument,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[DocumentChunk]:
        """Chunk config by top-level keys."""
        chunks: list[DocumentChunk] = []

        # Each top-level object becomes a chunk
        for node in document.root.children:
            if node.type == NodeType.OBJECT:
                chunk = DocumentChunk(
                    content=node.content,
                    document_id=document.id,
                    file_path=document.file_path,
                    node_ids=[node.id],
                    chunk_index=len(chunks),
                    strategy=ChunkStrategy.AST_BOUNDARY,
                    keywords=[node.name] + node.attributes.get("keys", [])[:5],
                    summary=f"Config section: {node.name}",
                )
                chunk.embedding_text = self.prepare_for_embedding(chunk, document)
                chunks.append(chunk)

        # If no object chunks, chunk entire document
        if not chunks:
            chunk = DocumentChunk(
                content=document.raw_content,
                document_id=document.id,
                file_path=document.file_path,
                chunk_index=0,
                total_chunks=1,
            )
            chunk.embedding_text = self.prepare_for_embedding(chunk, document)
            chunks.append(chunk)

        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks

    def extract_relationships(self, document: ParsedDocument) -> list[Relationship]:
        """Extract relationships from config references."""
        return []  # Config files typically don't have external references

    def prepare_for_embedding(
        self,
        chunk: DocumentChunk,
        document: ParsedDocument,
    ) -> str:
        """Prepare config chunk for embedding."""
        parts = [
            f"Config file: {Path(document.file_path).name}",
            f"Format: {document.file_type}",
        ]

        if chunk.summary:
            parts.append(chunk.summary)

        # Add keys for object chunks
        node = document.get_node(chunk.node_ids[0]) if chunk.node_ids else None
        if node and node.attributes.get("keys"):
            parts.append(f"Keys: {', '.join(node.attributes['keys'])}")

        parts.append(f"Content:\n{chunk.content}")

        return "\n\n".join(parts)


class JSONHandler(ConfigHandlerBase):
    """Handler for JSON files."""

    name: str = "json"
    extensions: list[str] = [".json", ".jsonc", ".json5"]
    mime_types: list[str] = ["application/json"]

    def parse(self, content: str, file_path: str) -> ParsedDocument:
        """Parse JSON file."""
        lines = content.split("\n")

        root = DocumentNode(
            type=NodeType.OBJECT,
            name=Path(file_path).stem,
            content="",
            location=SourceLocation(start_line=1, end_line=len(lines)),
        )

        symbols: dict[str, DocumentNode] = {}

        try:
            # Handle JSON with comments (JSONC)
            clean_content = self._strip_comments(content)
            data = json.loads(clean_content)

            root.children = self._build_tree(data, root.id)

            # Register top-level keys as symbols
            if isinstance(data, dict):
                for key in data:
                    for child in root.children:
                        if child.name == key:
                            symbols[key] = child

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error in {file_path}: {e}")
            root.content = content

        doc = ParsedDocument(
            file_path=file_path,
            file_type="json",
            raw_content=content,
            root=root,
            symbols=symbols,
            language="json",
            line_count=len(lines),
            char_count=len(content),
            node_count=len(root.walk()),
            metadata={"is_package_json": Path(file_path).name == "package.json"},
        )

        doc.chunks = self.chunk(doc)
        return doc

    def _strip_comments(self, content: str) -> str:
        """Strip comments from JSONC."""
        import re

        # Remove single-line comments
        content = re.sub(r"//.*$", "", content, flags=re.MULTILINE)
        # Remove multi-line comments
        content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
        return content


class YAMLHandler(ConfigHandlerBase):
    """Handler for YAML files."""

    name: str = "yaml"
    extensions: list[str] = [".yaml", ".yml"]
    mime_types: list[str] = ["text/yaml", "application/x-yaml"]

    def parse(self, content: str, file_path: str) -> ParsedDocument:
        """Parse YAML file."""
        lines = content.split("\n")

        root = DocumentNode(
            type=NodeType.OBJECT,
            name=Path(file_path).stem,
            content="",
            location=SourceLocation(start_line=1, end_line=len(lines)),
        )

        symbols: dict[str, DocumentNode] = {}

        try:
            import yaml

            data = yaml.safe_load(content)

            if data:
                root.children = self._build_tree(data, root.id)

                if isinstance(data, dict):
                    for key in data:
                        for child in root.children:
                            if child.name == key:
                                symbols[key] = child

        except ImportError:
            logger.warning("PyYAML not installed, skipping YAML parsing")
            root.content = content
        except Exception as e:
            logger.warning(f"YAML parse error in {file_path}: {e}")
            root.content = content

        doc = ParsedDocument(
            file_path=file_path,
            file_type="yaml",
            raw_content=content,
            root=root,
            symbols=symbols,
            language="yaml",
            line_count=len(lines),
            char_count=len(content),
            node_count=len(root.walk()),
            metadata={
                "is_docker_compose": "docker-compose" in Path(file_path).name,
                "is_github_actions": ".github" in file_path,
            },
        )

        doc.chunks = self.chunk(doc)
        return doc


class TOMLHandler(ConfigHandlerBase):
    """Handler for TOML files."""

    name: str = "toml"
    extensions: list[str] = [".toml"]
    mime_types: list[str] = ["application/toml"]

    def parse(self, content: str, file_path: str) -> ParsedDocument:
        """Parse TOML file."""
        lines = content.split("\n")

        root = DocumentNode(
            type=NodeType.OBJECT,
            name=Path(file_path).stem,
            content="",
            location=SourceLocation(start_line=1, end_line=len(lines)),
        )

        symbols: dict[str, DocumentNode] = {}

        try:
            import tomllib

            data = tomllib.loads(content)

            root.children = self._build_tree(data, root.id)

            for key in data:
                for child in root.children:
                    if child.name == key:
                        symbols[key] = child

        except ImportError:
            try:
                import tomli as tomllib

                data = tomllib.loads(content)
                root.children = self._build_tree(data, root.id)
            except ImportError:
                logger.warning("tomllib/tomli not installed, skipping TOML parsing")
                root.content = content
        except Exception as e:
            logger.warning(f"TOML parse error in {file_path}: {e}")
            root.content = content

        doc = ParsedDocument(
            file_path=file_path,
            file_type="toml",
            raw_content=content,
            root=root,
            symbols=symbols,
            language="toml",
            line_count=len(lines),
            char_count=len(content),
            node_count=len(root.walk()),
            metadata={
                "is_pyproject": Path(file_path).name == "pyproject.toml",
                "is_cargo": Path(file_path).name == "Cargo.toml",
            },
        )

        doc.chunks = self.chunk(doc)
        return doc
