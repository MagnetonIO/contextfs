"""
Swift file handler.

Extracts:
- Classes, structs, enums, protocols
- Functions and methods
- Extensions
- Properties and computed properties
- Generics and associated types
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


class SwiftHandler(FileTypeHandler):
    """Handler for Swift files."""

    name: str = "swift"
    extensions: list[str] = [".swift"]
    mime_types: list[str] = ["text/x-swift"]
    chunk_strategy: ChunkStrategy = ChunkStrategy.AST_BOUNDARY

    # Patterns
    IMPORT_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"import\s+(?:(\w+)\s+)?(\w+(?:\.\w+)*)", re.MULTILINE
    )
    CLASS_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:@\w+(?:\([^)]*\))?\s*)*(?:(open|public|internal|fileprivate|private)\s+)?(?:(final)\s+)?class\s+(\w+)(?:<([^>]+)>)?(?:\s*:\s*([^{]+))?\s*(?:where\s+[^{]+)?\s*\{",
        re.MULTILINE,
    )
    STRUCT_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:@\w+(?:\([^)]*\))?\s*)*(?:(public|internal|fileprivate|private)\s+)?struct\s+(\w+)(?:<([^>]+)>)?(?:\s*:\s*([^{]+))?\s*(?:where\s+[^{]+)?\s*\{",
        re.MULTILINE,
    )
    ENUM_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:@\w+(?:\([^)]*\))?\s*)*(?:(public|internal|fileprivate|private)\s+)?(?:indirect\s+)?enum\s+(\w+)(?:<([^>]+)>)?(?:\s*:\s*([^{]+))?\s*\{",
        re.MULTILINE,
    )
    PROTOCOL_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:@\w+(?:\([^)]*\))?\s*)*(?:(public|internal|fileprivate|private)\s+)?protocol\s+(\w+)(?:\s*:\s*([^{]+))?\s*\{",
        re.MULTILINE,
    )
    EXTENSION_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:(public|internal|fileprivate|private)\s+)?extension\s+(\w+)(?:<[^>]+>)?(?:\s*:\s*([^{]+))?\s*(?:where\s+[^{]+)?\s*\{",
        re.MULTILINE,
    )
    FUNC_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:@\w+(?:\([^)]*\))?\s*)*(?:(open|public|internal|fileprivate|private)\s+)?(?:(static|class|mutating|override)\s+)?func\s+(\w+)(?:<([^>]+)>)?\s*\(([^)]*)\)(?:\s*(?:throws|rethrows))?\s*(?:->\s*([^{]+))?\s*\{",
        re.MULTILINE,
    )
    INIT_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:(public|internal|fileprivate|private)\s+)?(?:(convenience|required)\s+)?init(?:\?|!)?\s*\(([^)]*)\)(?:\s*throws)?\s*\{",
        re.MULTILINE,
    )
    PROPERTY_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:(public|internal|fileprivate|private)\s+)?(?:(static|class|lazy|weak)\s+)?(?:var|let)\s+(\w+)\s*:\s*([^={\n]+)",
        re.MULTILINE,
    )
    TYPEALIAS_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:(public|internal|fileprivate|private)\s+)?typealias\s+(\w+)(?:<([^>]+)>)?\s*=\s*([^\n]+)",
        re.MULTILINE,
    )

    def parse(self, content: str, file_path: str) -> ParsedDocument:
        """Parse Swift file."""
        lines = content.split("\n")

        root = DocumentNode(
            type=NodeType.MODULE,
            name=Path(file_path).stem,
            content="",
            location=SourceLocation(start_line=1, end_line=len(lines)),
        )

        symbols: dict[str, DocumentNode] = {}
        references: list[DocumentNode] = []

        # Extract imports
        self._extract_imports(content, root, references)

        # Extract classes
        self._extract_classes(content, root, symbols)

        # Extract structs
        self._extract_structs(content, root, symbols)

        # Extract enums
        self._extract_enums(content, root, symbols)

        # Extract protocols
        self._extract_protocols(content, root, symbols)

        # Extract extensions
        self._extract_extensions(content, root, symbols)

        # Extract standalone functions
        self._extract_functions(content, root, symbols)

        doc = ParsedDocument(
            file_path=file_path,
            file_type="swift",
            raw_content=content,
            root=root,
            symbols=symbols,
            references=references,
            language="swift",
            line_count=len(lines),
            char_count=len(content),
            node_count=len(root.walk()),
            metadata={
                "has_swiftui": "SwiftUI" in content,
                "has_combine": "Combine" in content,
                "has_async": "async " in content,
            },
        )

        doc.chunks = self.chunk(doc)
        return doc

    def _extract_imports(
        self,
        content: str,
        root: DocumentNode,
        references: list[DocumentNode],
    ) -> None:
        """Extract import statements."""
        for match in self.IMPORT_PATTERN.finditer(content):
            kind = match.group(1)  # class, struct, func, etc.
            module = match.group(2)

            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1

            import_node = DocumentNode(
                type=NodeType.IMPORT,
                name=module,
                content=match.group(0),
                target=module,
                location=SourceLocation(start_line=line_num, end_line=line_num),
                parent_id=root.id,
                attributes={
                    "module": module,
                    "kind": kind,
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
            access = match.group(1) or "internal"
            is_final = bool(match.group(2))
            name = match.group(3)
            generics = match.group(4)
            inheritance = match.group(5)

            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1
            end_line = self._find_block_end(content, match.end() - 1, line_num)

            doc_comment = self._find_doc_comment(content, start_pos)

            # Parse inheritance
            superclass = None
            protocols = []
            if inheritance:
                parts = [p.strip() for p in inheritance.split(",")]
                # First non-protocol is superclass (doesn't start with protocol naming convention)
                for i, part in enumerate(parts):
                    if i == 0 and not self._is_protocol_name(part):
                        superclass = part
                    else:
                        protocols.append(part)

            class_node = DocumentNode(
                type=NodeType.CLASS,
                name=name,
                content=content[match.start() : self._get_pos_at_line(content, end_line + 1)],
                signature=f"class {name}{f'<{generics}>' if generics else ''}",
                docstring=doc_comment,
                location=SourceLocation(start_line=line_num, end_line=end_line),
                parent_id=root.id,
                attributes={
                    "access": access,
                    "is_final": is_final,
                    "generics": generics,
                    "superclass": superclass,
                    "protocols": protocols,
                },
            )
            root.children.append(class_node)
            symbols[name] = class_node

    def _extract_structs(
        self,
        content: str,
        root: DocumentNode,
        symbols: dict[str, DocumentNode],
    ) -> None:
        """Extract struct definitions."""
        for match in self.STRUCT_PATTERN.finditer(content):
            access = match.group(1) or "internal"
            name = match.group(2)
            generics = match.group(3)
            protocols = match.group(4)

            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1
            end_line = self._find_block_end(content, match.end() - 1, line_num)

            doc_comment = self._find_doc_comment(content, start_pos)

            struct_node = DocumentNode(
                type=NodeType.CLASS,
                name=name,
                content=content[match.start() : self._get_pos_at_line(content, end_line + 1)],
                signature=f"struct {name}{f'<{generics}>' if generics else ''}",
                docstring=doc_comment,
                location=SourceLocation(start_line=line_num, end_line=end_line),
                parent_id=root.id,
                attributes={
                    "is_struct": True,
                    "access": access,
                    "generics": generics,
                    "protocols": [p.strip() for p in protocols.split(",")] if protocols else [],
                },
            )
            root.children.append(struct_node)
            symbols[name] = struct_node

    def _extract_enums(
        self,
        content: str,
        root: DocumentNode,
        symbols: dict[str, DocumentNode],
    ) -> None:
        """Extract enum definitions."""
        for match in self.ENUM_PATTERN.finditer(content):
            access = match.group(1) or "internal"
            name = match.group(2)
            generics = match.group(3)
            match.group(4)

            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1
            end_line = self._find_block_end(content, match.end() - 1, line_num)

            doc_comment = self._find_doc_comment(content, start_pos)

            enum_node = DocumentNode(
                type=NodeType.VARIABLE,
                name=name,
                content=content[match.start() : self._get_pos_at_line(content, end_line + 1)],
                signature=f"enum {name}{f'<{generics}>' if generics else ''}",
                docstring=doc_comment,
                location=SourceLocation(start_line=line_num, end_line=end_line),
                parent_id=root.id,
                attributes={
                    "is_enum": True,
                    "access": access,
                    "generics": generics,
                    "is_indirect": "indirect"
                    in content[max(0, match.start() - 20) : match.start()],
                },
            )
            root.children.append(enum_node)
            symbols[name] = enum_node

    def _extract_protocols(
        self,
        content: str,
        root: DocumentNode,
        symbols: dict[str, DocumentNode],
    ) -> None:
        """Extract protocol definitions."""
        for match in self.PROTOCOL_PATTERN.finditer(content):
            access = match.group(1) or "internal"
            name = match.group(2)
            inherits = match.group(3)

            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1
            end_line = self._find_block_end(content, match.end() - 1, line_num)

            doc_comment = self._find_doc_comment(content, start_pos)

            protocol_node = DocumentNode(
                type=NodeType.CLASS,
                name=name,
                content=content[match.start() : self._get_pos_at_line(content, end_line + 1)],
                signature=f"protocol {name}",
                docstring=doc_comment,
                location=SourceLocation(start_line=line_num, end_line=end_line),
                parent_id=root.id,
                attributes={
                    "is_protocol": True,
                    "access": access,
                    "inherits": [i.strip() for i in inherits.split(",")] if inherits else [],
                },
            )
            root.children.append(protocol_node)
            symbols[name] = protocol_node

    def _extract_extensions(
        self,
        content: str,
        root: DocumentNode,
        symbols: dict[str, DocumentNode],
    ) -> None:
        """Extract extension definitions."""
        for match in self.EXTENSION_PATTERN.finditer(content):
            match.group(1)
            extended_type = match.group(2)
            protocols = match.group(3)

            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1
            end_line = self._find_block_end(content, match.end() - 1, line_num)

            ext_name = f"extension {extended_type}"

            ext_node = DocumentNode(
                type=NodeType.CLASS,
                name=ext_name,
                content=content[match.start() : self._get_pos_at_line(content, end_line + 1)],
                signature=ext_name,
                location=SourceLocation(start_line=line_num, end_line=end_line),
                parent_id=root.id,
                attributes={
                    "is_extension": True,
                    "extended_type": extended_type,
                    "protocols": [p.strip() for p in protocols.split(",")] if protocols else [],
                },
            )
            root.children.append(ext_node)
            symbols[ext_name] = ext_node

    def _extract_functions(
        self,
        content: str,
        root: DocumentNode,
        symbols: dict[str, DocumentNode],
    ) -> None:
        """Extract standalone function definitions."""
        for match in self.FUNC_PATTERN.finditer(content):
            # Skip if inside a type
            before = content[: match.start()]
            if before.count("{") != before.count("}"):
                continue

            access = match.group(1) or "internal"
            modifier = match.group(2)
            name = match.group(3)
            generics = match.group(4)
            params = match.group(5)
            return_type = match.group(6)

            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1
            end_line = self._find_block_end(content, match.end() - 1, line_num)

            doc_comment = self._find_doc_comment(content, start_pos)

            func_node = DocumentNode(
                type=NodeType.FUNCTION,
                name=name,
                content=content[match.start() : self._get_pos_at_line(content, end_line + 1)],
                signature=f"func {name}({params}){f' -> {return_type.strip()}' if return_type else ''}",
                return_type=return_type.strip() if return_type else None,
                docstring=doc_comment,
                location=SourceLocation(start_line=line_num, end_line=end_line),
                parent_id=root.id,
                attributes={
                    "access": access,
                    "is_static": modifier == "static",
                    "is_class": modifier == "class",
                    "is_mutating": modifier == "mutating",
                    "generics": generics,
                },
            )
            root.children.append(func_node)
            symbols[name] = func_node

    def _is_protocol_name(self, name: str) -> bool:
        """Check if name follows Swift protocol naming convention."""
        # Protocols often end with 'able', 'ible', 'Protocol', or 'Delegate'
        return any(
            name.endswith(suffix) for suffix in ["able", "ible", "Protocol", "Delegate", "Type"]
        )

    def _find_doc_comment(self, content: str, pos: int) -> str | None:
        """Find Swift doc comment preceding a position."""
        search_start = max(0, pos - 500)
        segment = content[search_start:pos]

        # Check for /// comments
        doc_lines = []
        for line in reversed(segment.rstrip().split("\n")):
            line = line.strip()
            if line.startswith("///"):
                doc_lines.insert(0, line[3:].strip())
            elif line.startswith("@") or not line:
                continue
            else:
                break

        if doc_lines:
            return "\n".join(doc_lines)

        # Check for /** */ comments
        match = re.search(r"/\*\*\s*(.*?)\s*\*/\s*$", segment, re.DOTALL)
        if match:
            return match.group(1).replace("*", "").strip()

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
        """Chunk Swift by type/function definitions."""
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
                )
                relationships.append(rel)

        for node in document.root.walk():
            if node.type == NodeType.CLASS:
                superclass = node.attributes.get("superclass")
                if superclass:
                    rel = Relationship(
                        type=RelationType.INHERITS,
                        source_document_id=document.id,
                        source_node_id=node.id,
                        source_name=node.name,
                        source_path=document.file_path,
                        target_name=superclass,
                        location=node.location,
                    )
                    relationships.append(rel)

                for protocol in node.attributes.get("protocols", []):
                    rel = Relationship(
                        type=RelationType.IMPLEMENTS,
                        source_document_id=document.id,
                        source_node_id=node.id,
                        source_name=node.name,
                        source_path=document.file_path,
                        target_name=protocol,
                        location=node.location,
                    )
                    relationships.append(rel)

        return relationships

    def prepare_for_embedding(
        self,
        chunk: DocumentChunk,
        document: ParsedDocument,
    ) -> str:
        """Prepare Swift chunk for embedding."""
        parts = ["Language: Swift"]
        node = document.get_node(chunk.node_ids[0]) if chunk.node_ids else None

        if node:
            if node.type == NodeType.CLASS:
                if node.attributes.get("is_protocol"):
                    parts.append(f"Swift Protocol: {node.name}")
                elif node.attributes.get("is_struct"):
                    parts.append(f"Swift Struct: {node.name}")
                elif node.attributes.get("is_extension"):
                    parts.append(f"Swift Extension: {node.name}")
                else:
                    parts.append(f"Swift Class: {node.name}")
            elif node.type == NodeType.FUNCTION:
                parts.append(f"Swift Function: {node.name}")

            if node.signature:
                parts.append(f"Signature: {node.signature}")
            if node.docstring:
                parts.append(f"Doc: {node.docstring}")

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
            if node.attributes.get("superclass"):
                keywords.append(node.attributes["superclass"])
            keywords.extend(node.attributes.get("protocols", []))

        return keywords

    def _create_summary(self, node: DocumentNode) -> str:
        """Create summary for node."""
        if node.type == NodeType.CLASS:
            if node.attributes.get("is_protocol"):
                return f"Swift protocol {node.name}"
            if node.attributes.get("is_struct"):
                return f"Swift struct {node.name}"
            if node.attributes.get("is_extension"):
                return f"Swift {node.name}"
            return f"Swift class {node.name}"
        elif node.type == NodeType.FUNCTION:
            return f"Swift func {node.name}"
        return ""
