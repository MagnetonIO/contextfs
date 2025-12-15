"""
TypeScript file handler with enhanced type support.

Extracts:
- Interfaces and type aliases
- Generics and type parameters
- Decorators and metadata
- Module declarations
- Enums
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


class TypeScriptHandler(FileTypeHandler):
    """Handler for TypeScript files with enhanced type support."""

    name: str = "typescript"
    extensions: list[str] = [".ts", ".tsx", ".mts", ".cts"]
    mime_types: list[str] = ["application/typescript", "text/typescript"]
    chunk_strategy: ChunkStrategy = ChunkStrategy.AST_BOUNDARY

    # Patterns
    IMPORT_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"import\s+(?:type\s+)?(?:(?:\{([^}]+)\}|(\w+)|\*\s+as\s+(\w+))\s+from\s+)?['\"]([^'\"]+)['\"]",
        re.MULTILINE,
    )
    EXPORT_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"export\s+(?:default\s+)?(?:async\s+)?(?:function|class|const|let|var|interface|type|enum)\s+(\w+)",
        re.MULTILINE,
    )
    FUNCTION_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*(?:<([^>]+)>)?\s*\(([^)]*)\)(?:\s*:\s*([^{]+))?\s*\{",
        re.MULTILINE,
    )
    ARROW_FUNCTION_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:export\s+)?(?:const|let|var)\s+(\w+)\s*(?::\s*[^=]+)?\s*=\s*(?:async\s+)?(?:<([^>]+)>)?\s*\([^)]*\)\s*(?::\s*[^=]+)?\s*=>",
        re.MULTILINE,
    )
    CLASS_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:export\s+)?(?:abstract\s+)?class\s+(\w+)(?:<([^>]+)>)?(?:\s+extends\s+(\w+)(?:<[^>]+>)?)?(?:\s+implements\s+([^{]+))?\s*\{",
        re.MULTILINE,
    )
    INTERFACE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:export\s+)?interface\s+(\w+)(?:<([^>]+)>)?(?:\s+extends\s+([^{]+))?\s*\{", re.MULTILINE
    )
    TYPE_ALIAS_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:export\s+)?type\s+(\w+)(?:<([^>]+)>)?\s*=\s*([^;]+);", re.MULTILINE
    )
    ENUM_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:export\s+)?(?:const\s+)?enum\s+(\w+)\s*\{", re.MULTILINE
    )
    DECORATOR_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"@(\w+)(?:\([^)]*\))?", re.MULTILINE)
    NAMESPACE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:export\s+)?(?:declare\s+)?(?:namespace|module)\s+(\w+)\s*\{", re.MULTILINE
    )

    def parse(self, content: str, file_path: str) -> ParsedDocument:
        """Parse TypeScript file."""
        lines = content.split("\n")
        is_tsx = file_path.endswith(".tsx")

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

        # Extract interfaces
        self._extract_interfaces(content, root, symbols)

        # Extract type aliases
        self._extract_type_aliases(content, root, symbols)

        # Extract enums
        self._extract_enums(content, root, symbols)

        # Extract classes
        self._extract_classes(content, root, symbols)

        # Extract functions
        self._extract_functions(content, root, symbols)

        # Extract namespaces
        self._extract_namespaces(content, root, symbols)

        doc = ParsedDocument(
            file_path=file_path,
            file_type="typescript",
            raw_content=content,
            root=root,
            symbols=symbols,
            references=references,
            language="typescript",
            line_count=len(lines),
            char_count=len(content),
            node_count=len(root.walk()),
            metadata={
                "is_tsx": is_tsx,
                "has_jsx": is_tsx and ("<" in content and "/>" in content),
                "has_decorators": bool(self.DECORATOR_PATTERN.search(content)),
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
            named = match.group(1)
            default = match.group(2)
            namespace = match.group(3)
            module = match.group(4)

            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1
            is_type_import = "import type" in match.group(0)

            import_node = DocumentNode(
                type=NodeType.IMPORT,
                name=module,
                content=match.group(0),
                target=module,
                location=SourceLocation(start_line=line_num, end_line=line_num),
                parent_id=root.id,
                attributes={
                    "module": module,
                    "default": default,
                    "named": [n.strip() for n in named.split(",")] if named else [],
                    "namespace": namespace,
                    "is_relative": module.startswith("."),
                    "is_type_import": is_type_import,
                },
            )
            root.children.append(import_node)
            references.append(import_node)

    def _extract_interfaces(
        self,
        content: str,
        root: DocumentNode,
        symbols: dict[str, DocumentNode],
    ) -> None:
        """Extract interface definitions."""
        for match in self.INTERFACE_PATTERN.finditer(content):
            name = match.group(1)
            generics = match.group(2)
            extends = match.group(3)
            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1

            end_line = self._find_block_end(content, match.end() - 1, line_num)

            interface_node = DocumentNode(
                type=NodeType.CLASS,
                name=name,
                content=content[match.start() : self._get_pos_at_line(content, end_line + 1)],
                signature=f"interface {name}{f'<{generics}>' if generics else ''}",
                location=SourceLocation(start_line=line_num, end_line=end_line),
                parent_id=root.id,
                attributes={
                    "is_interface": True,
                    "generics": generics,
                    "extends": [e.strip() for e in extends.split(",")] if extends else [],
                    "is_exported": "export" in content[max(0, match.start() - 15) : match.start()],
                },
            )
            root.children.append(interface_node)
            symbols[name] = interface_node

    def _extract_type_aliases(
        self,
        content: str,
        root: DocumentNode,
        symbols: dict[str, DocumentNode],
    ) -> None:
        """Extract type alias definitions."""
        for match in self.TYPE_ALIAS_PATTERN.finditer(content):
            name = match.group(1)
            generics = match.group(2)
            type_def = match.group(3)
            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1

            type_node = DocumentNode(
                type=NodeType.VARIABLE,
                name=name,
                content=match.group(0),
                signature=f"type {name}{f'<{generics}>' if generics else ''} = ...",
                location=SourceLocation(
                    start_line=line_num, end_line=line_num + type_def.count("\n")
                ),
                parent_id=root.id,
                attributes={
                    "is_type_alias": True,
                    "generics": generics,
                    "type_definition": type_def.strip(),
                    "is_exported": "export" in content[max(0, match.start() - 15) : match.start()],
                },
            )
            root.children.append(type_node)
            symbols[name] = type_node

    def _extract_enums(
        self,
        content: str,
        root: DocumentNode,
        symbols: dict[str, DocumentNode],
    ) -> None:
        """Extract enum definitions."""
        for match in self.ENUM_PATTERN.finditer(content):
            name = match.group(1)
            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1

            end_line = self._find_block_end(content, match.end() - 1, line_num)
            is_const = "const enum" in match.group(0)

            enum_node = DocumentNode(
                type=NodeType.VARIABLE,
                name=name,
                content=content[match.start() : self._get_pos_at_line(content, end_line + 1)],
                location=SourceLocation(start_line=line_num, end_line=end_line),
                parent_id=root.id,
                attributes={
                    "is_enum": True,
                    "is_const_enum": is_const,
                    "is_exported": "export" in content[max(0, match.start() - 15) : match.start()],
                },
            )
            root.children.append(enum_node)
            symbols[name] = enum_node

    def _extract_classes(
        self,
        content: str,
        root: DocumentNode,
        symbols: dict[str, DocumentNode],
    ) -> None:
        """Extract class definitions."""
        for match in self.CLASS_PATTERN.finditer(content):
            name = match.group(1)
            generics = match.group(2)
            extends = match.group(3)
            implements = match.group(4)
            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1

            end_line = self._find_block_end(content, match.end() - 1, line_num)

            # Check for decorators
            decorators = self._find_decorators(content, start_pos)

            class_node = DocumentNode(
                type=NodeType.CLASS,
                name=name,
                content=content[match.start() : self._get_pos_at_line(content, end_line + 1)],
                signature=f"class {name}{f'<{generics}>' if generics else ''}",
                location=SourceLocation(start_line=line_num, end_line=end_line),
                parent_id=root.id,
                attributes={
                    "generics": generics,
                    "extends": extends,
                    "implements": [i.strip() for i in implements.split(",")] if implements else [],
                    "is_abstract": "abstract"
                    in content[max(0, match.start() - 15) : match.start()],
                    "is_exported": "export" in content[max(0, match.start() - 15) : match.start()],
                    "decorators": decorators,
                },
            )
            root.children.append(class_node)
            symbols[name] = class_node

    def _extract_functions(
        self,
        content: str,
        root: DocumentNode,
        symbols: dict[str, DocumentNode],
    ) -> None:
        """Extract function definitions."""
        # Regular functions
        for match in self.FUNCTION_PATTERN.finditer(content):
            name = match.group(1)
            generics = match.group(2)
            params = match.group(3)
            return_type = match.group(4)
            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1

            end_line = self._find_block_end(content, match.end() - 1, line_num)

            func_node = DocumentNode(
                type=NodeType.FUNCTION,
                name=name,
                content=content[match.start() : self._get_pos_at_line(content, end_line + 1)],
                signature=f"function {name}{f'<{generics}>' if generics else ''}({params}){f': {return_type.strip()}' if return_type else ''}",
                return_type=return_type.strip() if return_type else None,
                location=SourceLocation(start_line=line_num, end_line=end_line),
                parent_id=root.id,
                attributes={
                    "generics": generics,
                    "is_async": "async" in content[max(0, match.start() - 10) : match.start()],
                    "is_exported": "export" in content[max(0, match.start() - 15) : match.start()],
                },
            )
            root.children.append(func_node)
            symbols[name] = func_node

        # Arrow functions
        for match in self.ARROW_FUNCTION_PATTERN.finditer(content):
            name = match.group(1)
            if name in symbols:
                continue

            generics = match.group(2)
            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1

            func_node = DocumentNode(
                type=NodeType.FUNCTION,
                name=name,
                content=match.group(0),
                signature=f"const {name}{f'<{generics}>' if generics else ''} = ...",
                location=SourceLocation(start_line=line_num, end_line=line_num),
                parent_id=root.id,
                attributes={
                    "is_arrow": True,
                    "generics": generics,
                    "is_async": "async" in match.group(0),
                    "is_exported": "export" in content[max(0, match.start() - 15) : match.start()],
                },
            )
            root.children.append(func_node)
            symbols[name] = func_node

    def _extract_namespaces(
        self,
        content: str,
        root: DocumentNode,
        symbols: dict[str, DocumentNode],
    ) -> None:
        """Extract namespace/module declarations."""
        for match in self.NAMESPACE_PATTERN.finditer(content):
            name = match.group(1)
            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1

            end_line = self._find_block_end(content, match.end() - 1, line_num)

            ns_node = DocumentNode(
                type=NodeType.MODULE,
                name=name,
                content=content[match.start() : self._get_pos_at_line(content, end_line + 1)],
                location=SourceLocation(start_line=line_num, end_line=end_line),
                parent_id=root.id,
                attributes={
                    "is_namespace": True,
                    "is_declare": "declare" in match.group(0),
                    "is_exported": "export" in content[max(0, match.start() - 15) : match.start()],
                },
            )
            root.children.append(ns_node)
            symbols[name] = ns_node

    def _find_decorators(self, content: str, pos: int) -> list[str]:
        """Find decorators preceding a position."""
        search_start = max(0, pos - 500)
        segment = content[search_start:pos]

        decorators = []
        for match in self.DECORATOR_PATTERN.finditer(segment):
            decorators.append(match.group(1))

        return decorators

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
        """Chunk TypeScript by definitions."""
        chunks: list[DocumentChunk] = []

        for node in document.root.children:
            if node.type in (NodeType.FUNCTION, NodeType.CLASS, NodeType.MODULE):
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
        """Extract relationships from imports and type references."""
        relationships: list[Relationship] = []

        for ref in document.references:
            if ref.type == NodeType.IMPORT:
                module = ref.attributes.get("module", ref.target)

                rel = Relationship(
                    type=RelationType.IMPORTS,
                    source_document_id=document.id,
                    source_node_id=ref.id,
                    source_path=document.file_path,
                    target_name=module,
                    context=ref.content,
                    location=ref.location,
                    attributes={
                        "is_relative": ref.attributes.get("is_relative", False),
                        "is_type_import": ref.attributes.get("is_type_import", False),
                        "named_imports": ref.attributes.get("named", []),
                    },
                )
                relationships.append(rel)

        # Extract inheritance and implements
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
        """Prepare TypeScript chunk for embedding."""
        parts = []
        node = document.get_node(chunk.node_ids[0]) if chunk.node_ids else None

        if node:
            if node.type == NodeType.FUNCTION:
                parts.append(f"TypeScript Function: {node.name}")
                if node.signature:
                    parts.append(f"Signature: {node.signature}")
            elif node.type == NodeType.CLASS:
                if node.attributes.get("is_interface"):
                    parts.append(f"TypeScript Interface: {node.name}")
                else:
                    parts.append(f"TypeScript Class: {node.name}")
                if node.signature:
                    parts.append(f"Signature: {node.signature}")
            elif node.type == NodeType.MODULE:
                parts.append(f"TypeScript Namespace: {node.name}")

            if node.attributes.get("generics"):
                parts.append(f"Generic Parameters: {node.attributes['generics']}")

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
            keywords.extend(node.attributes.get("decorators", []))

        return keywords

    def _create_summary(self, node: DocumentNode) -> str:
        """Create summary for node."""
        if node.type == NodeType.FUNCTION:
            prefix = "Async function" if node.attributes.get("is_async") else "Function"
            return f"{prefix} {node.name}"
        elif node.type == NodeType.CLASS:
            if node.attributes.get("is_interface"):
                return f"Interface {node.name}"
            if node.attributes.get("is_abstract"):
                return f"Abstract class {node.name}"
            return f"Class {node.name}"
        elif node.type == NodeType.MODULE:
            return f"Namespace {node.name}"
        return ""
