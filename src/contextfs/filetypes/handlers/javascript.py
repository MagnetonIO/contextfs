"""
JavaScript/TypeScript file handler.

Extracts:
- Functions, classes, methods
- Imports and exports
- JSDoc comments
- React components
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


class JavaScriptHandler(FileTypeHandler):
    """Handler for JavaScript and TypeScript files."""

    name: str = "javascript"
    extensions: list[str] = [".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"]
    mime_types: list[str] = [
        "application/javascript",
        "text/javascript",
        "application/typescript",
    ]
    chunk_strategy: ChunkStrategy = ChunkStrategy.AST_BOUNDARY

    # Patterns
    IMPORT_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"import\s+(?:(?:\{([^}]+)\}|(\w+)|\*\s+as\s+(\w+))\s+from\s+)?['\"]([^'\"]+)['\"]",
        re.MULTILINE,
    )
    EXPORT_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"export\s+(?:default\s+)?(?:async\s+)?(?:function|class|const|let|var|interface|type)\s+(\w+)",
        re.MULTILINE,
    )
    FUNCTION_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)(?:\s*:\s*([^{]+))?\s*\{",
        re.MULTILINE,
    )
    ARROW_FUNCTION_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*(?::\s*[^=]+)?\s*=>",
        re.MULTILINE,
    )
    CLASS_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:export\s+)?(?:abstract\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([^{]+))?\s*\{",
        re.MULTILINE,
    )
    INTERFACE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:export\s+)?interface\s+(\w+)(?:\s+extends\s+([^{]+))?\s*\{", re.MULTILINE
    )
    TYPE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:export\s+)?type\s+(\w+)(?:<[^>]+>)?\s*=", re.MULTILINE
    )
    JSDOC_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"/\*\*\s*(.*?)\s*\*/", re.DOTALL)

    def parse(self, content: str, file_path: str) -> ParsedDocument:
        """Parse JavaScript/TypeScript file."""
        lines = content.split("\n")
        is_typescript = file_path.endswith((".ts", ".tsx"))
        is_react = file_path.endswith((".jsx", ".tsx"))

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

        # Extract functions
        self._extract_functions(content, root, symbols)

        # Extract classes
        self._extract_classes(content, root, symbols)

        # Extract interfaces (TypeScript)
        if is_typescript:
            self._extract_interfaces(content, root, symbols)
            self._extract_types(content, root, symbols)

        # Extract exports
        self._extract_exports(content, root, symbols)

        doc = ParsedDocument(
            file_path=file_path,
            file_type="typescript" if is_typescript else "javascript",
            raw_content=content,
            root=root,
            symbols=symbols,
            references=references,
            language="typescript" if is_typescript else "javascript",
            line_count=len(lines),
            char_count=len(content),
            node_count=len(root.walk()),
            metadata={
                "is_typescript": is_typescript,
                "is_react": is_react,
                "has_jsx": "<" in content and "/>" in content,
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
                },
            )
            root.children.append(import_node)
            references.append(import_node)

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
            params = match.group(2)
            return_type = match.group(3)
            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1

            # Find function end (simple brace matching)
            end_line = self._find_block_end(content, match.end() - 1, line_num)

            # Get JSDoc if present
            jsdoc = self._find_preceding_jsdoc(content, start_pos)

            func_node = DocumentNode(
                type=NodeType.FUNCTION,
                name=name,
                content=content[match.start() : self._get_pos_at_line(content, end_line + 1)],
                signature=f"function {name}({params}){': ' + return_type.strip() if return_type else ''}",
                docstring=jsdoc,
                return_type=return_type.strip() if return_type else None,
                location=SourceLocation(start_line=line_num, end_line=end_line),
                parent_id=root.id,
                attributes={
                    "is_async": "async" in content[max(0, match.start() - 10) : match.start()],
                    "is_exported": "export" in content[max(0, match.start() - 15) : match.start()],
                },
            )
            root.children.append(func_node)
            symbols[name] = func_node

        # Arrow functions
        for match in self.ARROW_FUNCTION_PATTERN.finditer(content):
            name = match.group(1)
            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1

            jsdoc = self._find_preceding_jsdoc(content, start_pos)

            func_node = DocumentNode(
                type=NodeType.FUNCTION,
                name=name,
                content=match.group(0),
                docstring=jsdoc,
                location=SourceLocation(start_line=line_num, end_line=line_num),
                parent_id=root.id,
                attributes={
                    "is_arrow": True,
                    "is_async": "async" in content[max(0, match.start() - 10) : match.start()],
                },
            )
            root.children.append(func_node)
            symbols[name] = func_node

    def _extract_classes(
        self,
        content: str,
        root: DocumentNode,
        symbols: dict[str, DocumentNode],
    ) -> None:
        """Extract class definitions."""
        for match in self.CLASS_PATTERN.finditer(content):
            name = match.group(1)
            extends = match.group(2)
            implements = match.group(3)
            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1

            end_line = self._find_block_end(content, match.end() - 1, line_num)
            jsdoc = self._find_preceding_jsdoc(content, start_pos)

            class_node = DocumentNode(
                type=NodeType.CLASS,
                name=name,
                content=content[match.start() : self._get_pos_at_line(content, end_line + 1)],
                docstring=jsdoc,
                location=SourceLocation(start_line=line_num, end_line=end_line),
                parent_id=root.id,
                attributes={
                    "extends": extends,
                    "implements": [i.strip() for i in implements.split(",")] if implements else [],
                    "is_exported": "export" in content[max(0, match.start() - 15) : match.start()],
                    "is_abstract": "abstract"
                    in content[max(0, match.start() - 15) : match.start()],
                },
            )
            root.children.append(class_node)
            symbols[name] = class_node

    def _extract_interfaces(
        self,
        content: str,
        root: DocumentNode,
        symbols: dict[str, DocumentNode],
    ) -> None:
        """Extract TypeScript interfaces."""
        for match in self.INTERFACE_PATTERN.finditer(content):
            name = match.group(1)
            extends = match.group(2)
            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1

            end_line = self._find_block_end(content, match.end() - 1, line_num)

            interface_node = DocumentNode(
                type=NodeType.CLASS,  # Using CLASS for interface
                name=name,
                content=content[match.start() : self._get_pos_at_line(content, end_line + 1)],
                location=SourceLocation(start_line=line_num, end_line=end_line),
                parent_id=root.id,
                attributes={
                    "is_interface": True,
                    "extends": [e.strip() for e in extends.split(",")] if extends else [],
                },
            )
            root.children.append(interface_node)
            symbols[name] = interface_node

    def _extract_types(
        self,
        content: str,
        root: DocumentNode,
        symbols: dict[str, DocumentNode],
    ) -> None:
        """Extract TypeScript type aliases."""
        for match in self.TYPE_PATTERN.finditer(content):
            name = match.group(1)
            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1

            type_node = DocumentNode(
                type=NodeType.VARIABLE,  # Using VARIABLE for type alias
                name=name,
                content=match.group(0),
                location=SourceLocation(start_line=line_num, end_line=line_num),
                parent_id=root.id,
                attributes={"is_type_alias": True},
            )
            root.children.append(type_node)
            symbols[name] = type_node

    def _extract_exports(
        self,
        content: str,
        root: DocumentNode,
        symbols: dict[str, DocumentNode],
    ) -> None:
        """Extract export statements."""
        for match in self.EXPORT_PATTERN.finditer(content):
            name = match.group(1)
            if name not in symbols:
                start_pos = match.start()
                line_num = content[:start_pos].count("\n") + 1

                export_node = DocumentNode(
                    type=NodeType.VARIABLE,
                    name=name,
                    content=match.group(0),
                    location=SourceLocation(start_line=line_num, end_line=line_num),
                    parent_id=root.id,
                    attributes={"is_exported": True},
                )
                root.children.append(export_node)
                symbols[name] = export_node

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

    def _find_preceding_jsdoc(self, content: str, pos: int) -> str | None:
        """Find JSDoc comment preceding a position."""
        # Look backwards for */
        search_start = max(0, pos - 500)
        segment = content[search_start:pos]

        match = re.search(r"/\*\*\s*(.*?)\s*\*/\s*$", segment, re.DOTALL)
        if match:
            return match.group(1).replace("* ", "").strip()
        return None

    def chunk(
        self,
        document: ParsedDocument,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[DocumentChunk]:
        """Chunk JavaScript/TypeScript by definitions."""
        chunks: list[DocumentChunk] = []

        for node in document.root.children:
            if node.type in (NodeType.FUNCTION, NodeType.CLASS):
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
        """Extract relationships from imports."""
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
                        "named_imports": ref.attributes.get("named", []),
                    },
                )
                relationships.append(rel)

        # Extract inheritance
        for node in document.root.walk():
            if node.type == NodeType.CLASS:
                extends = node.attributes.get("extends")
                if extends:
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

        return relationships

    def prepare_for_embedding(
        self,
        chunk: DocumentChunk,
        document: ParsedDocument,
    ) -> str:
        """Prepare JS/TS chunk for embedding."""
        parts = []
        node = document.get_node(chunk.node_ids[0]) if chunk.node_ids else None

        if node:
            if node.type == NodeType.FUNCTION:
                parts.append(f"Function: {node.name}")
                if node.signature:
                    parts.append(f"Signature: {node.signature}")
            elif node.type == NodeType.CLASS:
                if node.attributes.get("is_interface"):
                    parts.append(f"Interface: {node.name}")
                else:
                    parts.append(f"Class: {node.name}")

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

        return keywords

    def _create_summary(self, node: DocumentNode) -> str:
        """Create summary for node."""
        if node.type == NodeType.FUNCTION:
            prefix = "Async function" if node.attributes.get("is_async") else "Function"
            return f"{prefix} {node.name}"
        elif node.type == NodeType.CLASS:
            if node.attributes.get("is_interface"):
                return f"Interface {node.name}"
            return f"Class {node.name}"
        return ""
