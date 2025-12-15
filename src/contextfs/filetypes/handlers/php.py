"""
PHP file handler.

Extracts:
- Classes, interfaces, traits
- Functions and methods
- Namespaces and use statements
- Constants and properties
- Attributes (PHP 8+)
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


class PHPHandler(FileTypeHandler):
    """Handler for PHP files."""

    name: str = "php"
    extensions: list[str] = [".php", ".phtml", ".php3", ".php4", ".php5", ".phps"]
    mime_types: list[str] = ["text/x-php", "application/x-php"]
    chunk_strategy: ChunkStrategy = ChunkStrategy.AST_BOUNDARY

    # Patterns
    NAMESPACE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"namespace\s+([\w\\]+)\s*;")
    USE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"use\s+([\w\\]+)(?:\s+as\s+(\w+))?\s*;", re.MULTILINE
    )
    CLASS_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:#\[([^\]]+)\]\s*)*(?:(abstract|final)\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([^{]+))?\s*\{",
        re.MULTILINE,
    )
    INTERFACE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"interface\s+(\w+)(?:\s+extends\s+([^{]+))?\s*\{", re.MULTILINE
    )
    TRAIT_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"trait\s+(\w+)\s*\{", re.MULTILINE)
    ENUM_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"enum\s+(\w+)(?:\s*:\s*(\w+))?(?:\s+implements\s+([^{]+))?\s*\{", re.MULTILINE
    )
    FUNCTION_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:#\[([^\]]+)\]\s*)*(?:(public|private|protected)\s+)?(?:(static)\s+)?function\s+(\w+)\s*\(([^)]*)\)(?:\s*:\s*(\??\w+(?:\|[\w?]+)*))?\s*\{",
        re.MULTILINE,
    )
    CONST_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:(public|private|protected)\s+)?const\s+(\w+)\s*=", re.MULTILINE
    )
    PROPERTY_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:(public|private|protected)\s+)?(?:(static|readonly)\s+)?(?:(\??\w+)\s+)?\$(\w+)",
        re.MULTILINE,
    )
    PHPDOC_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"/\*\*\s*(.*?)\s*\*/", re.DOTALL)
    ATTRIBUTE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"#\[(\w+)(?:\([^]]*\))?\]")

    def parse(self, content: str, file_path: str) -> ParsedDocument:
        """Parse PHP file."""
        lines = content.split("\n")

        root = DocumentNode(
            type=NodeType.MODULE,
            name=Path(file_path).stem,
            content="",
            location=SourceLocation(start_line=1, end_line=len(lines)),
        )

        symbols: dict[str, DocumentNode] = {}
        references: list[DocumentNode] = []

        # Extract namespace
        namespace = self._extract_namespace(content)

        # Extract use statements
        self._extract_uses(content, root, references)

        # Extract classes
        self._extract_classes(content, root, symbols)

        # Extract interfaces
        self._extract_interfaces(content, root, symbols)

        # Extract traits
        self._extract_traits(content, root, symbols)

        # Extract enums (PHP 8.1+)
        self._extract_enums(content, root, symbols)

        # Extract standalone functions
        self._extract_functions(content, root, symbols)

        doc = ParsedDocument(
            file_path=file_path,
            file_type="php",
            raw_content=content,
            root=root,
            symbols=symbols,
            references=references,
            language="php",
            line_count=len(lines),
            char_count=len(content),
            node_count=len(root.walk()),
            metadata={
                "namespace": namespace,
                "has_strict_types": "declare(strict_types=1)" in content,
            },
        )

        doc.chunks = self.chunk(doc)
        return doc

    def _extract_namespace(self, content: str) -> str | None:
        """Extract namespace declaration."""
        match = self.NAMESPACE_PATTERN.search(content)
        return match.group(1) if match else None

    def _extract_uses(
        self,
        content: str,
        root: DocumentNode,
        references: list[DocumentNode],
    ) -> None:
        """Extract use statements."""
        for match in self.USE_PATTERN.finditer(content):
            fqn = match.group(1)
            alias = match.group(2)

            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1

            use_node = DocumentNode(
                type=NodeType.IMPORT,
                name=fqn.split("\\")[-1],
                content=match.group(0),
                target=fqn,
                location=SourceLocation(start_line=line_num, end_line=line_num),
                parent_id=root.id,
                attributes={
                    "fqn": fqn,
                    "alias": alias,
                },
            )
            root.children.append(use_node)
            references.append(use_node)

    def _extract_classes(
        self,
        content: str,
        root: DocumentNode,
        symbols: dict[str, DocumentNode],
    ) -> None:
        """Extract class definitions."""
        for match in self.CLASS_PATTERN.finditer(content):
            attributes = match.group(1)
            modifier = match.group(2)
            name = match.group(3)
            extends = match.group(4)
            implements = match.group(5)

            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1
            end_line = self._find_block_end(content, match.end() - 1, line_num)

            phpdoc = self._find_phpdoc(content, start_pos)

            class_node = DocumentNode(
                type=NodeType.CLASS,
                name=name,
                content=content[match.start() : self._get_pos_at_line(content, end_line + 1)],
                signature=f"class {name}",
                docstring=phpdoc,
                location=SourceLocation(start_line=line_num, end_line=end_line),
                parent_id=root.id,
                attributes={
                    "is_abstract": modifier == "abstract",
                    "is_final": modifier == "final",
                    "extends": extends,
                    "implements": [i.strip() for i in implements.split(",")] if implements else [],
                    "attributes": self._parse_attributes(attributes) if attributes else [],
                },
            )

            # Extract methods
            class_content = content[match.end() : self._get_pos_at_line(content, end_line)]
            self._extract_methods(class_content, class_node, line_num)

            root.children.append(class_node)
            symbols[name] = class_node

    def _extract_interfaces(
        self,
        content: str,
        root: DocumentNode,
        symbols: dict[str, DocumentNode],
    ) -> None:
        """Extract interface definitions."""
        for match in self.INTERFACE_PATTERN.finditer(content):
            name = match.group(1)
            extends = match.group(2)

            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1
            end_line = self._find_block_end(content, match.end() - 1, line_num)

            phpdoc = self._find_phpdoc(content, start_pos)

            interface_node = DocumentNode(
                type=NodeType.CLASS,
                name=name,
                content=content[match.start() : self._get_pos_at_line(content, end_line + 1)],
                signature=f"interface {name}",
                docstring=phpdoc,
                location=SourceLocation(start_line=line_num, end_line=end_line),
                parent_id=root.id,
                attributes={
                    "is_interface": True,
                    "extends": [e.strip() for e in extends.split(",")] if extends else [],
                },
            )
            root.children.append(interface_node)
            symbols[name] = interface_node

    def _extract_traits(
        self,
        content: str,
        root: DocumentNode,
        symbols: dict[str, DocumentNode],
    ) -> None:
        """Extract trait definitions."""
        for match in self.TRAIT_PATTERN.finditer(content):
            name = match.group(1)

            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1
            end_line = self._find_block_end(content, match.end() - 1, line_num)

            phpdoc = self._find_phpdoc(content, start_pos)

            trait_node = DocumentNode(
                type=NodeType.CLASS,
                name=name,
                content=content[match.start() : self._get_pos_at_line(content, end_line + 1)],
                signature=f"trait {name}",
                docstring=phpdoc,
                location=SourceLocation(start_line=line_num, end_line=end_line),
                parent_id=root.id,
                attributes={"is_trait": True},
            )
            root.children.append(trait_node)
            symbols[name] = trait_node

    def _extract_enums(
        self,
        content: str,
        root: DocumentNode,
        symbols: dict[str, DocumentNode],
    ) -> None:
        """Extract enum definitions (PHP 8.1+)."""
        for match in self.ENUM_PATTERN.finditer(content):
            name = match.group(1)
            backing_type = match.group(2)
            implements = match.group(3)

            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1
            end_line = self._find_block_end(content, match.end() - 1, line_num)

            enum_node = DocumentNode(
                type=NodeType.VARIABLE,
                name=name,
                content=content[match.start() : self._get_pos_at_line(content, end_line + 1)],
                location=SourceLocation(start_line=line_num, end_line=end_line),
                parent_id=root.id,
                attributes={
                    "is_enum": True,
                    "backing_type": backing_type,
                    "implements": [i.strip() for i in implements.split(",")] if implements else [],
                },
            )
            root.children.append(enum_node)
            symbols[name] = enum_node

    def _extract_functions(
        self,
        content: str,
        root: DocumentNode,
        symbols: dict[str, DocumentNode],
    ) -> None:
        """Extract standalone function definitions."""
        for match in self.FUNCTION_PATTERN.finditer(content):
            # Skip if inside a class (check for class context)
            before = content[: match.start()]
            if before.count("{") != before.count("}"):
                continue

            attributes = match.group(1)
            match.group(2)
            bool(match.group(3))
            name = match.group(4)
            params = match.group(5)
            return_type = match.group(6)

            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1
            end_line = self._find_block_end(content, match.end() - 1, line_num)

            phpdoc = self._find_phpdoc(content, start_pos)

            func_node = DocumentNode(
                type=NodeType.FUNCTION,
                name=name,
                content=content[match.start() : self._get_pos_at_line(content, end_line + 1)],
                signature=f"function {name}({params}){f': {return_type}' if return_type else ''}",
                return_type=return_type,
                docstring=phpdoc,
                location=SourceLocation(start_line=line_num, end_line=end_line),
                parent_id=root.id,
                attributes={
                    "attributes": self._parse_attributes(attributes) if attributes else [],
                },
            )
            root.children.append(func_node)
            symbols[name] = func_node

    def _extract_methods(
        self,
        content: str,
        parent: DocumentNode,
        base_line: int,
    ) -> None:
        """Extract method definitions within a class."""
        for match in self.FUNCTION_PATTERN.finditer(content):
            attributes = match.group(1)
            visibility = match.group(2) or "public"
            is_static = bool(match.group(3))
            name = match.group(4)
            params = match.group(5)
            return_type = match.group(6)

            start_pos = match.start()
            line_num = base_line + content[:start_pos].count("\n")

            phpdoc = self._find_phpdoc(content, start_pos)

            method_node = DocumentNode(
                type=NodeType.FUNCTION,
                name=name,
                content=match.group(0),
                signature=f"{visibility} function {name}({params}){f': {return_type}' if return_type else ''}",
                return_type=return_type,
                docstring=phpdoc,
                location=SourceLocation(start_line=line_num, end_line=line_num),
                parent_id=parent.id,
                attributes={
                    "visibility": visibility,
                    "is_static": is_static,
                    "attributes": self._parse_attributes(attributes) if attributes else [],
                },
            )
            parent.children.append(method_node)

    def _find_phpdoc(self, content: str, pos: int) -> str | None:
        """Find PHPDoc comment preceding a position."""
        search_start = max(0, pos - 1000)
        segment = content[search_start:pos]

        match = re.search(r"/\*\*\s*(.*?)\s*\*/\s*$", segment, re.DOTALL)
        if match:
            doc = match.group(1)
            doc = re.sub(r"^\s*\*\s?", "", doc, flags=re.MULTILINE)
            return doc.strip()
        return None

    def _parse_attributes(self, text: str) -> list[str]:
        """Parse PHP 8 attribute names."""
        return [m.group(1) for m in self.ATTRIBUTE_PATTERN.finditer(text)]

    def _find_block_end(self, content: str, start: int, start_line: int) -> int:
        """Find the end line of a block starting at {."""
        depth = 0
        line = start_line

        for i, char in enumerate(content[start:], start):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return line
            elif char == "\n":
                line += 1

        return line

    def _get_pos_at_line(self, content: str, line: int) -> int:
        """Get character position at start of line."""
        for i, c in enumerate(content):
            if line <= 1:
                return i
            if c == "\n":
                line -= 1
                i + 1
        return len(content)

    def chunk(
        self,
        document: ParsedDocument,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[DocumentChunk]:
        """Chunk PHP by class/function definitions."""
        chunks: list[DocumentChunk] = []

        for node in document.root.children:
            if node.type in (NodeType.CLASS, NodeType.FUNCTION):
                keywords = self._extract_keywords(node)

                chunk = DocumentChunk(
                    content=node.content,
                    document_id=document.id,
                    file_path=document.file_path,
                    node_ids=[node.id],
                    location=node.location,
                    chunk_index=len(chunks),
                    strategy=ChunkStrategy.AST_BOUNDARY,
                    keywords=keywords,
                    summary=self._create_summary(node),
                )
                chunk.embedding_text = self.prepare_for_embedding(chunk, document)
                chunks.append(chunk)

        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks

    def extract_relationships(self, document: ParsedDocument) -> list[Relationship]:
        """Extract relationships from use statements and inheritance."""
        relationships: list[Relationship] = []

        for ref in document.references:
            if ref.type == NodeType.IMPORT:
                rel = Relationship(
                    type=RelationType.IMPORTS,
                    source_document_id=document.id,
                    source_node_id=ref.id,
                    source_path=document.file_path,
                    target_name=ref.target,
                    context=ref.content,
                    location=ref.location,
                    attributes={"alias": ref.attributes.get("alias")},
                )
                relationships.append(rel)

        for node in document.root.walk():
            if node.type == NodeType.CLASS:
                extends = node.attributes.get("extends")
                if (
                    extends
                    and not node.attributes.get("is_interface")
                    and not node.attributes.get("is_trait")
                ):
                    rel = Relationship(
                        type=RelationType.INHERITS,
                        source_document_id=document.id,
                        source_node_id=node.id,
                        source_name=node.name,
                        source_path=document.file_path,
                        target_name=extends,
                        location=node.location,
                    )
                    relationships.append(rel)

                for impl in node.attributes.get("implements", []):
                    rel = Relationship(
                        type=RelationType.IMPLEMENTS,
                        source_document_id=document.id,
                        source_node_id=node.id,
                        source_name=node.name,
                        source_path=document.file_path,
                        target_name=impl,
                        location=node.location,
                    )
                    relationships.append(rel)

        return relationships

    def prepare_for_embedding(
        self,
        chunk: DocumentChunk,
        document: ParsedDocument,
    ) -> str:
        """Prepare PHP chunk for embedding."""
        parts = []
        node = document.get_node(chunk.node_ids[0]) if chunk.node_ids else None

        if document.metadata.get("namespace"):
            parts.append(f"PHP Namespace: {document.metadata['namespace']}")

        if node:
            if node.type == NodeType.CLASS:
                if node.attributes.get("is_interface"):
                    parts.append(f"PHP Interface: {node.name}")
                elif node.attributes.get("is_trait"):
                    parts.append(f"PHP Trait: {node.name}")
                else:
                    parts.append(f"PHP Class: {node.name}")
            elif node.type == NodeType.FUNCTION:
                parts.append(f"PHP Function: {node.name}")

            if node.signature:
                parts.append(f"Signature: {node.signature}")
            if node.docstring:
                parts.append(f"Doc: {node.docstring[:200]}")

        parts.append(f"Code:\n{chunk.content}")

        if chunk.keywords:
            parts.append(f"Keywords: {', '.join(chunk.keywords)}")

        return "\n\n".join(parts)

    def _extract_keywords(self, node: DocumentNode) -> list[str]:
        """Extract keywords from node."""
        keywords = []
        if node.name:
            keywords.append(node.name)

        if node.type == NodeType.CLASS:
            if node.attributes.get("extends"):
                keywords.append(node.attributes["extends"])
            keywords.extend(node.attributes.get("implements", []))

        return keywords

    def _create_summary(self, node: DocumentNode) -> str:
        """Create summary for node."""
        if node.type == NodeType.CLASS:
            if node.attributes.get("is_interface"):
                return f"PHP interface {node.name}"
            if node.attributes.get("is_trait"):
                return f"PHP trait {node.name}"
            modifier = ""
            if node.attributes.get("is_abstract"):
                modifier = "abstract "
            elif node.attributes.get("is_final"):
                modifier = "final "
            return f"PHP {modifier}class {node.name}"
        elif node.type == NodeType.FUNCTION:
            return f"PHP function {node.name}"
        return ""
