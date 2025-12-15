"""
C# file handler.

Extracts:
- Classes, interfaces, structs, records
- Methods, properties, events
- Namespaces and using directives
- Attributes/decorators
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


class CSharpHandler(FileTypeHandler):
    """Handler for C# files."""

    name: str = "csharp"
    extensions: list[str] = [".cs", ".csx"]
    mime_types: list[str] = ["text/x-csharp"]
    chunk_strategy: ChunkStrategy = ChunkStrategy.AST_BOUNDARY

    # Patterns
    USING_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"using\s+(?:static\s+)?([\w.]+)(?:\s*=\s*([\w.]+))?\s*;", re.MULTILINE
    )
    NAMESPACE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"namespace\s+([\w.]+)\s*[{;]", re.MULTILINE
    )
    CLASS_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:\[([^\]]+)\]\s*)*(?:(public|private|protected|internal)\s+)?(?:(abstract|sealed|static|partial)\s+)*(?:(class|struct|record|interface))\s+(\w+)(?:<([^>]+)>)?(?:\s*:\s*([^{]+))?\s*(?:where\s+[^{]+)?\s*\{",
        re.MULTILINE,
    )
    METHOD_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:\[([^\]]+)\]\s*)*(?:(public|private|protected|internal)\s+)?(?:(static|virtual|override|abstract|async|partial)\s+)*(?:(\w+(?:<[^>]+>)?(?:\?)?(?:\[\])*)\s+)?(\w+)\s*(?:<([^>]+)>)?\s*\(([^)]*)\)\s*(?:where\s+[^{]+)?\s*[{;=>]",
        re.MULTILINE,
    )
    PROPERTY_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:(public|private|protected|internal)\s+)?(?:(static|virtual|override|abstract)\s+)?(\w+(?:<[^>]+>)?(?:\?)?)\s+(\w+)\s*\{",
        re.MULTILINE,
    )
    ENUM_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:(public|private|protected|internal)\s+)?enum\s+(\w+)(?:\s*:\s*(\w+))?\s*\{",
        re.MULTILINE,
    )
    DELEGATE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:(public|private|protected|internal)\s+)?delegate\s+(\w+(?:<[^>]+>)?)\s+(\w+)\s*\(([^)]*)\)\s*;",
        re.MULTILINE,
    )
    ATTRIBUTE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"\[(\w+)(?:\([^\]]*\))?\]")

    def parse(self, content: str, file_path: str) -> ParsedDocument:
        """Parse C# file."""
        lines = content.split("\n")

        root = DocumentNode(
            type=NodeType.MODULE,
            name=Path(file_path).stem,
            content="",
            location=SourceLocation(start_line=1, end_line=len(lines)),
        )

        symbols: dict[str, DocumentNode] = {}
        references: list[DocumentNode] = []

        # Extract using directives
        self._extract_usings(content, root, references)

        # Extract namespaces
        namespace = self._extract_namespace(content)

        # Extract classes/structs/interfaces
        self._extract_types(content, root, symbols)

        # Extract enums
        self._extract_enums(content, root, symbols)

        doc = ParsedDocument(
            file_path=file_path,
            file_type="csharp",
            raw_content=content,
            root=root,
            symbols=symbols,
            references=references,
            language="csharp",
            line_count=len(lines),
            char_count=len(content),
            node_count=len(root.walk()),
            metadata={
                "namespace": namespace,
                "has_linq": "from " in content and " select " in content,
                "has_async": "async " in content,
            },
        )

        doc.chunks = self.chunk(doc)
        return doc

    def _extract_usings(
        self,
        content: str,
        root: DocumentNode,
        references: list[DocumentNode],
    ) -> None:
        """Extract using directives."""
        for match in self.USING_PATTERN.finditer(content):
            namespace = match.group(1)
            alias = match.group(2)
            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1

            using_node = DocumentNode(
                type=NodeType.IMPORT,
                name=namespace,
                content=match.group(0),
                target=namespace,
                location=SourceLocation(start_line=line_num, end_line=line_num),
                parent_id=root.id,
                attributes={
                    "namespace": namespace,
                    "alias": alias,
                    "is_static": "static" in match.group(0),
                },
            )
            root.children.append(using_node)
            references.append(using_node)

    def _extract_namespace(self, content: str) -> str | None:
        """Extract namespace declaration."""
        match = self.NAMESPACE_PATTERN.search(content)
        return match.group(1) if match else None

    def _extract_types(
        self,
        content: str,
        root: DocumentNode,
        symbols: dict[str, DocumentNode],
    ) -> None:
        """Extract class/struct/interface/record definitions."""
        for match in self.CLASS_PATTERN.finditer(content):
            attributes = match.group(1)
            visibility = match.group(2) or "internal"
            modifiers = match.group(3) or ""
            kind = match.group(4)
            name = match.group(5)
            generics = match.group(6)
            bases = match.group(7)

            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1
            end_line = self._find_block_end(content, match.end() - 1, line_num)

            # Parse base types
            base_classes = []
            interfaces = []
            if bases:
                for base in bases.split(","):
                    base = base.strip()
                    if base.startswith("I") and base[1:2].isupper():
                        interfaces.append(base)
                    else:
                        base_classes.append(base)

            type_node = DocumentNode(
                type=NodeType.CLASS,
                name=name,
                content=content[match.start() : self._get_pos_at_line(content, end_line + 1)],
                signature=f"{kind} {name}{f'<{generics}>' if generics else ''}",
                location=SourceLocation(start_line=line_num, end_line=end_line),
                parent_id=root.id,
                attributes={
                    "kind": kind,
                    "visibility": visibility,
                    "is_abstract": "abstract" in modifiers,
                    "is_sealed": "sealed" in modifiers,
                    "is_static": "static" in modifiers,
                    "is_partial": "partial" in modifiers,
                    "generics": generics,
                    "base_classes": base_classes,
                    "interfaces": interfaces,
                    "attributes": self._parse_attributes(attributes) if attributes else [],
                },
            )

            # Extract methods within type
            type_content = content[match.end() : self._get_pos_at_line(content, end_line)]
            self._extract_methods(type_content, type_node, line_num)

            root.children.append(type_node)
            symbols[name] = type_node

    def _extract_methods(
        self,
        content: str,
        parent: DocumentNode,
        base_line: int,
    ) -> None:
        """Extract method definitions within a type."""
        for match in self.METHOD_PATTERN.finditer(content):
            attributes = match.group(1)
            visibility = match.group(2) or "private"
            modifiers = match.group(3) or ""
            return_type = match.group(4)
            name = match.group(5)
            generics = match.group(6)
            params = match.group(7)

            # Skip constructors (return_type is None and name matches class)
            if not return_type and name == parent.name:
                continue

            # Skip property accessors
            if name in ("get", "set", "add", "remove"):
                continue

            start_pos = match.start()
            line_num = base_line + content[:start_pos].count("\n")

            method_node = DocumentNode(
                type=NodeType.FUNCTION,
                name=name,
                content=match.group(0),
                signature=f"{return_type or 'void'} {name}{f'<{generics}>' if generics else ''}({params})",
                return_type=return_type,
                location=SourceLocation(start_line=line_num, end_line=line_num),
                parent_id=parent.id,
                attributes={
                    "visibility": visibility,
                    "is_static": "static" in modifiers,
                    "is_virtual": "virtual" in modifiers,
                    "is_override": "override" in modifiers,
                    "is_abstract": "abstract" in modifiers,
                    "is_async": "async" in modifiers,
                    "generics": generics,
                    "attributes": self._parse_attributes(attributes) if attributes else [],
                },
            )
            parent.children.append(method_node)

    def _extract_enums(
        self,
        content: str,
        root: DocumentNode,
        symbols: dict[str, DocumentNode],
    ) -> None:
        """Extract enum definitions."""
        for match in self.ENUM_PATTERN.finditer(content):
            visibility = match.group(1) or "internal"
            name = match.group(2)
            base_type = match.group(3)

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
                    "base_type": base_type,
                },
            )
            root.children.append(enum_node)
            symbols[name] = enum_node

    def _parse_attributes(self, text: str) -> list[str]:
        """Parse attribute names from text."""
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
        """Chunk C# by type definitions."""
        chunks: list[DocumentChunk] = []

        for node in document.root.children:
            if node.type == NodeType.CLASS:
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
        """Extract relationships from using directives and inheritance."""
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
                        "alias": ref.attributes.get("alias"),
                    },
                )
                relationships.append(rel)

        for node in document.root.walk():
            if node.type == NodeType.CLASS:
                for base in node.attributes.get("base_classes", []):
                    rel = Relationship(
                        type=RelationType.INHERITS,
                        source_document_id=document.id,
                        source_node_id=node.id,
                        source_name=node.name,
                        source_path=document.file_path,
                        target_name=base,
                        location=node.location,
                    )
                    relationships.append(rel)

                for interface in node.attributes.get("interfaces", []):
                    rel = Relationship(
                        type=RelationType.IMPLEMENTS,
                        source_document_id=document.id,
                        source_node_id=node.id,
                        source_name=node.name,
                        source_path=document.file_path,
                        target_name=interface,
                        location=node.location,
                    )
                    relationships.append(rel)

        return relationships

    def prepare_for_embedding(
        self,
        chunk: DocumentChunk,
        document: ParsedDocument,
    ) -> str:
        """Prepare C# chunk for embedding."""
        parts = []
        node = document.get_node(chunk.node_ids[0]) if chunk.node_ids else None

        if document.metadata.get("namespace"):
            parts.append(f"Namespace: {document.metadata['namespace']}")

        if node:
            kind = node.attributes.get("kind", "class")
            parts.append(f"C# {kind.title()}: {node.name}")

            if node.signature:
                parts.append(f"Signature: {node.signature}")

        parts.append(f"Code:\n{chunk.content}")

        if chunk.keywords:
            parts.append(f"Keywords: {', '.join(chunk.keywords)}")

        return "\n\n".join(parts)

    def _extract_keywords(self, node: DocumentNode) -> list[str]:
        """Extract keywords from node."""
        keywords = []
        if node.name:
            keywords.append(node.name)

        keywords.extend(node.attributes.get("base_classes", []))
        keywords.extend(node.attributes.get("interfaces", []))
        keywords.extend(node.attributes.get("attributes", []))

        return keywords

    def _create_summary(self, node: DocumentNode) -> str:
        """Create summary for node."""
        kind = node.attributes.get("kind", "class")
        modifiers = []
        if node.attributes.get("is_abstract"):
            modifiers.append("abstract")
        if node.attributes.get("is_static"):
            modifiers.append("static")
        if node.attributes.get("is_sealed"):
            modifiers.append("sealed")

        prefix = " ".join(modifiers + [kind])
        return f"C# {prefix} {node.name}"
