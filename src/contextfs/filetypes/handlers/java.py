"""
Java file handler.

Extracts:
- Classes, interfaces, enums, records
- Methods and constructors
- Annotations
- Package and imports
- Generics
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


class JavaHandler(FileTypeHandler):
    """Handler for Java files."""

    name: str = "java"
    extensions: list[str] = [".java"]
    mime_types: list[str] = ["text/x-java-source", "text/java"]
    chunk_strategy: ChunkStrategy = ChunkStrategy.AST_BOUNDARY

    # Patterns
    PACKAGE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"package\s+([\w.]+)\s*;")
    IMPORT_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"import\s+(static\s+)?([\w.]+(?:\.\*)?)\s*;", re.MULTILINE
    )
    CLASS_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:(@\w+(?:\([^)]*\))?)\s+)*(?:(public|private|protected)\s+)?(?:(abstract|final)\s+)?(?:(static)\s+)?class\s+(\w+)(?:<([^>]+)>)?(?:\s+extends\s+(\w+)(?:<[^>]+>)?)?(?:\s+implements\s+([^{]+))?\s*\{",
        re.MULTILINE,
    )
    INTERFACE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:(public|private|protected)\s+)?interface\s+(\w+)(?:<([^>]+)>)?(?:\s+extends\s+([^{]+))?\s*\{",
        re.MULTILINE,
    )
    ENUM_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:(public|private|protected)\s+)?enum\s+(\w+)(?:\s+implements\s+([^{]+))?\s*\{",
        re.MULTILINE,
    )
    RECORD_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:(public|private|protected)\s+)?record\s+(\w+)\s*\(([^)]*)\)(?:\s+implements\s+([^{]+))?\s*\{",
        re.MULTILINE,
    )
    METHOD_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:(@\w+(?:\([^)]*\))?)\s+)*(?:(public|private|protected)\s+)?(?:(static|final|abstract|synchronized)\s+)*(?:<([^>]+)>\s+)?(\w+(?:<[^>]+>)?(?:\[\])*)\s+(\w+)\s*\(([^)]*)\)(?:\s+throws\s+([^{;]+))?\s*[{;]",
        re.MULTILINE,
    )
    CONSTRUCTOR_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:(public|private|protected)\s+)?(\w+)\s*\(([^)]*)\)(?:\s+throws\s+([^{]+))?\s*\{",
        re.MULTILINE,
    )
    ANNOTATION_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"@(\w+)(?:\([^)]*\))?")
    JAVADOC_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"/\*\*\s*(.*?)\s*\*/", re.DOTALL)

    def parse(self, content: str, file_path: str) -> ParsedDocument:
        """Parse Java file."""
        lines = content.split("\n")

        root = DocumentNode(
            type=NodeType.MODULE,
            name=Path(file_path).stem,
            content="",
            location=SourceLocation(start_line=1, end_line=len(lines)),
        )

        symbols: dict[str, DocumentNode] = {}
        references: list[DocumentNode] = []

        # Extract package
        package = self._extract_package(content)

        # Extract imports
        self._extract_imports(content, root, references)

        # Extract classes
        self._extract_classes(content, root, symbols)

        # Extract interfaces
        self._extract_interfaces(content, root, symbols)

        # Extract enums
        self._extract_enums(content, root, symbols)

        # Extract records (Java 14+)
        self._extract_records(content, root, symbols)

        doc = ParsedDocument(
            file_path=file_path,
            file_type="java",
            raw_content=content,
            root=root,
            symbols=symbols,
            references=references,
            language="java",
            line_count=len(lines),
            char_count=len(content),
            node_count=len(root.walk()),
            metadata={
                "package": package,
                "has_main": "public static void main" in content,
            },
        )

        doc.chunks = self.chunk(doc)
        return doc

    def _extract_package(self, content: str) -> str | None:
        """Extract package declaration."""
        match = self.PACKAGE_PATTERN.search(content)
        return match.group(1) if match else None

    def _extract_imports(
        self,
        content: str,
        root: DocumentNode,
        references: list[DocumentNode],
    ) -> None:
        """Extract import statements."""
        for match in self.IMPORT_PATTERN.finditer(content):
            is_static = bool(match.group(1))
            import_path = match.group(2)

            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1

            import_node = DocumentNode(
                type=NodeType.IMPORT,
                name=import_path,
                content=match.group(0),
                target=import_path,
                location=SourceLocation(start_line=line_num, end_line=line_num),
                parent_id=root.id,
                attributes={
                    "import_path": import_path,
                    "is_static": is_static,
                    "is_wildcard": import_path.endswith(".*"),
                },
            )
            root.children.append(import_node)
            references.append(import_node)

    def _extract_classes(
        self,
        content: str,
        root: DocumentNode,
        symbols: dict[str, DocumentNode],
    ) -> None:
        """Extract class definitions."""
        for match in self.CLASS_PATTERN.finditer(content):
            annotations = match.group(1)
            visibility = match.group(2) or "package"
            modifier = match.group(3)
            is_static = bool(match.group(4))
            name = match.group(5)
            generics = match.group(6)
            extends = match.group(7)
            implements = match.group(8)

            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1
            end_line = self._find_block_end(content, match.end() - 1, line_num)

            javadoc = self._find_preceding_javadoc(content, start_pos)

            class_node = DocumentNode(
                type=NodeType.CLASS,
                name=name,
                content=content[match.start() : self._get_pos_at_line(content, end_line + 1)],
                signature=f"class {name}{f'<{generics}>' if generics else ''}",
                docstring=javadoc,
                location=SourceLocation(start_line=line_num, end_line=end_line),
                parent_id=root.id,
                attributes={
                    "visibility": visibility,
                    "is_abstract": modifier == "abstract",
                    "is_final": modifier == "final",
                    "is_static": is_static,
                    "generics": generics,
                    "extends": extends,
                    "implements": [i.strip() for i in implements.split(",")] if implements else [],
                    "annotations": self._parse_annotations(annotations) if annotations else [],
                },
            )

            # Extract methods within class
            self._extract_methods(
                content[match.end() : self._get_pos_at_line(content, end_line)],
                class_node,
                line_num,
            )

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
            visibility = match.group(1) or "package"
            name = match.group(2)
            generics = match.group(3)
            extends = match.group(4)

            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1
            end_line = self._find_block_end(content, match.end() - 1, line_num)

            javadoc = self._find_preceding_javadoc(content, start_pos)

            interface_node = DocumentNode(
                type=NodeType.CLASS,
                name=name,
                content=content[match.start() : self._get_pos_at_line(content, end_line + 1)],
                signature=f"interface {name}{f'<{generics}>' if generics else ''}",
                docstring=javadoc,
                location=SourceLocation(start_line=line_num, end_line=end_line),
                parent_id=root.id,
                attributes={
                    "is_interface": True,
                    "visibility": visibility,
                    "generics": generics,
                    "extends": [e.strip() for e in extends.split(",")] if extends else [],
                },
            )
            root.children.append(interface_node)
            symbols[name] = interface_node

    def _extract_enums(
        self,
        content: str,
        root: DocumentNode,
        symbols: dict[str, DocumentNode],
    ) -> None:
        """Extract enum definitions."""
        for match in self.ENUM_PATTERN.finditer(content):
            visibility = match.group(1) or "package"
            name = match.group(2)
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
                    "visibility": visibility,
                    "implements": [i.strip() for i in implements.split(",")] if implements else [],
                },
            )
            root.children.append(enum_node)
            symbols[name] = enum_node

    def _extract_records(
        self,
        content: str,
        root: DocumentNode,
        symbols: dict[str, DocumentNode],
    ) -> None:
        """Extract record definitions (Java 14+)."""
        for match in self.RECORD_PATTERN.finditer(content):
            visibility = match.group(1) or "package"
            name = match.group(2)
            components = match.group(3)
            implements = match.group(4)

            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1
            end_line = self._find_block_end(content, match.end() - 1, line_num)

            record_node = DocumentNode(
                type=NodeType.CLASS,
                name=name,
                content=content[match.start() : self._get_pos_at_line(content, end_line + 1)],
                signature=f"record {name}({components})",
                location=SourceLocation(start_line=line_num, end_line=end_line),
                parent_id=root.id,
                attributes={
                    "is_record": True,
                    "visibility": visibility,
                    "components": components,
                    "implements": [i.strip() for i in implements.split(",")] if implements else [],
                },
            )
            root.children.append(record_node)
            symbols[name] = record_node

    def _extract_methods(
        self,
        content: str,
        parent: DocumentNode,
        base_line: int,
    ) -> None:
        """Extract method definitions within a class."""
        for match in self.METHOD_PATTERN.finditer(content):
            annotations = match.group(1)
            visibility = match.group(2) or "package"
            modifiers = match.group(3) or ""
            generics = match.group(4)
            return_type = match.group(5)
            name = match.group(6)
            params = match.group(7)
            throws = match.group(8)

            start_pos = match.start()
            line_num = base_line + content[:start_pos].count("\n")

            method_node = DocumentNode(
                type=NodeType.FUNCTION,
                name=name,
                content=match.group(0),
                signature=f"{return_type} {name}({params})",
                return_type=return_type,
                location=SourceLocation(start_line=line_num, end_line=line_num),
                parent_id=parent.id,
                attributes={
                    "visibility": visibility,
                    "is_static": "static" in modifiers,
                    "is_abstract": "abstract" in modifiers,
                    "is_final": "final" in modifiers,
                    "is_synchronized": "synchronized" in modifiers,
                    "generics": generics,
                    "throws": [t.strip() for t in throws.split(",")] if throws else [],
                    "annotations": self._parse_annotations(annotations) if annotations else [],
                },
            )
            parent.children.append(method_node)

    def _parse_annotations(self, text: str) -> list[str]:
        """Parse annotation names from text."""
        return [m.group(1) for m in self.ANNOTATION_PATTERN.finditer(text)]

    def _find_preceding_javadoc(self, content: str, pos: int) -> str | None:
        """Find Javadoc comment preceding a position."""
        search_start = max(0, pos - 1000)
        segment = content[search_start:pos]

        match = re.search(r"/\*\*\s*(.*?)\s*\*/\s*$", segment, re.DOTALL)
        if match:
            doc = match.group(1)
            # Clean up Javadoc formatting
            doc = re.sub(r"^\s*\*\s?", "", doc, flags=re.MULTILINE)
            return doc.strip()
        return None

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
        """Chunk Java by class/interface definitions."""
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
        """Extract relationships from imports and inheritance."""
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
                    attributes={
                        "is_static": ref.attributes.get("is_static", False),
                        "is_wildcard": ref.attributes.get("is_wildcard", False),
                    },
                )
                relationships.append(rel)

        for node in document.root.walk():
            if node.type == NodeType.CLASS:
                extends = node.attributes.get("extends")
                if extends and not node.attributes.get("is_interface"):
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
        """Prepare Java chunk for embedding."""
        parts = []
        node = document.get_node(chunk.node_ids[0]) if chunk.node_ids else None

        if document.metadata.get("package"):
            parts.append(f"Package: {document.metadata['package']}")

        if node:
            if node.type == NodeType.CLASS:
                if node.attributes.get("is_interface"):
                    parts.append(f"Java Interface: {node.name}")
                elif node.attributes.get("is_record"):
                    parts.append(f"Java Record: {node.name}")
                else:
                    parts.append(f"Java Class: {node.name}")
            elif node.type == NodeType.FUNCTION:
                parts.append(f"Java Method: {node.name}")

            if node.signature:
                parts.append(f"Signature: {node.signature}")
            if node.docstring:
                parts.append(f"Description: {node.docstring}")

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
            keywords.extend(node.attributes.get("annotations", []))

        return keywords

    def _create_summary(self, node: DocumentNode) -> str:
        """Create summary for node."""
        if node.type == NodeType.CLASS:
            if node.attributes.get("is_interface"):
                return f"Java interface {node.name}"
            if node.attributes.get("is_record"):
                return f"Java record {node.name}"
            if node.attributes.get("is_abstract"):
                return f"Abstract class {node.name}"
            return f"Java class {node.name}"
        elif node.type == NodeType.FUNCTION:
            return f"Method {node.name}"
        return ""
