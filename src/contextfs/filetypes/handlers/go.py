"""
Go file handler.

Extracts:
- Packages and imports
- Functions and methods
- Structs and interfaces
- Type definitions
- Constants and variables
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


class GoHandler(FileTypeHandler):
    """Handler for Go files."""

    name: str = "go"
    extensions: list[str] = [".go"]
    mime_types: list[str] = ["text/x-go"]
    chunk_strategy: ChunkStrategy = ChunkStrategy.AST_BOUNDARY

    # Patterns
    PACKAGE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"package\s+(\w+)")
    IMPORT_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r'import\s+(?:(\w+)\s+)?"([^"]+)"', re.MULTILINE
    )
    IMPORT_BLOCK_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"import\s*\(\s*((?:[^)]+))\)", re.DOTALL
    )
    FUNC_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"func\s+(?:\((\w+)\s+\*?(\w+)\)\s+)?(\w+)\s*(?:\[([^\]]+)\])?\s*\(([^)]*)\)\s*(?:\(([^)]*)\)|(\w+(?:\s*,\s*\w+)*)?)?\s*\{",
        re.MULTILINE,
    )
    STRUCT_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"type\s+(\w+)\s+struct\s*\{", re.MULTILINE
    )
    INTERFACE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"type\s+(\w+)\s+interface\s*\{", re.MULTILINE
    )
    TYPE_ALIAS_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"type\s+(\w+)\s+(?!=\s*struct|interface)(\w+(?:\[[^\]]*\])?)", re.MULTILINE
    )
    CONST_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"const\s+(\w+)(?:\s+\w+)?\s*=", re.MULTILINE
    )
    VAR_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"var\s+(\w+)\s+(\w+(?:\[[^\]]*\])?)", re.MULTILINE
    )

    def parse(self, content: str, file_path: str) -> ParsedDocument:
        """Parse Go file."""
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

        # Extract structs
        self._extract_structs(content, root, symbols)

        # Extract interfaces
        self._extract_interfaces(content, root, symbols)

        # Extract functions
        self._extract_functions(content, root, symbols)

        # Extract type aliases
        self._extract_type_aliases(content, root, symbols)

        doc = ParsedDocument(
            file_path=file_path,
            file_type="go",
            raw_content=content,
            root=root,
            symbols=symbols,
            references=references,
            language="go",
            line_count=len(lines),
            char_count=len(content),
            node_count=len(root.walk()),
            metadata={
                "package": package,
                "is_main": package == "main",
                "is_test": file_path.endswith("_test.go"),
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
        # Single imports
        for match in self.IMPORT_PATTERN.finditer(content):
            alias = match.group(1)
            import_path = match.group(2)
            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1

            import_node = DocumentNode(
                type=NodeType.IMPORT,
                name=import_path.split("/")[-1],
                content=match.group(0),
                target=import_path,
                location=SourceLocation(start_line=line_num, end_line=line_num),
                parent_id=root.id,
                attributes={
                    "import_path": import_path,
                    "alias": alias,
                    "is_stdlib": "." not in import_path.split("/")[0],
                },
            )
            root.children.append(import_node)
            references.append(import_node)

        # Import blocks
        for block_match in self.IMPORT_BLOCK_PATTERN.finditer(content):
            block_content = block_match.group(1)
            block_start = block_match.start()
            base_line = content[:block_start].count("\n") + 1

            for line in block_content.strip().split("\n"):
                line = line.strip()
                if not line or line.startswith("//"):
                    continue

                # Parse individual import line
                import_match = re.match(r'(?:(\w+)\s+)?"([^"]+)"', line)
                if import_match:
                    alias = import_match.group(1)
                    import_path = import_match.group(2)

                    import_node = DocumentNode(
                        type=NodeType.IMPORT,
                        name=import_path.split("/")[-1],
                        content=line,
                        target=import_path,
                        location=SourceLocation(start_line=base_line, end_line=base_line),
                        parent_id=root.id,
                        attributes={
                            "import_path": import_path,
                            "alias": alias,
                            "is_stdlib": "." not in import_path.split("/")[0],
                        },
                    )
                    references.append(import_node)

    def _extract_structs(
        self,
        content: str,
        root: DocumentNode,
        symbols: dict[str, DocumentNode],
    ) -> None:
        """Extract struct definitions."""
        for match in self.STRUCT_PATTERN.finditer(content):
            name = match.group(1)
            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1
            end_line = self._find_block_end(content, match.end() - 1, line_num)

            # Check for doc comment
            doc_comment = self._find_doc_comment(content, start_pos)

            struct_node = DocumentNode(
                type=NodeType.CLASS,
                name=name,
                content=content[match.start() : self._get_pos_at_line(content, end_line + 1)],
                signature=f"type {name} struct",
                docstring=doc_comment,
                location=SourceLocation(start_line=line_num, end_line=end_line),
                parent_id=root.id,
                attributes={
                    "is_struct": True,
                    "is_exported": name[0].isupper(),
                },
            )
            root.children.append(struct_node)
            symbols[name] = struct_node

    def _extract_interfaces(
        self,
        content: str,
        root: DocumentNode,
        symbols: dict[str, DocumentNode],
    ) -> None:
        """Extract interface definitions."""
        for match in self.INTERFACE_PATTERN.finditer(content):
            name = match.group(1)
            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1
            end_line = self._find_block_end(content, match.end() - 1, line_num)

            doc_comment = self._find_doc_comment(content, start_pos)

            interface_node = DocumentNode(
                type=NodeType.CLASS,
                name=name,
                content=content[match.start() : self._get_pos_at_line(content, end_line + 1)],
                signature=f"type {name} interface",
                docstring=doc_comment,
                location=SourceLocation(start_line=line_num, end_line=end_line),
                parent_id=root.id,
                attributes={
                    "is_interface": True,
                    "is_exported": name[0].isupper(),
                },
            )
            root.children.append(interface_node)
            symbols[name] = interface_node

    def _extract_functions(
        self,
        content: str,
        root: DocumentNode,
        symbols: dict[str, DocumentNode],
    ) -> None:
        """Extract function and method definitions."""
        for match in self.FUNC_PATTERN.finditer(content):
            receiver_name = match.group(1)
            receiver_type = match.group(2)
            func_name = match.group(3)
            type_params = match.group(4)
            params = match.group(5)
            return_tuple = match.group(6)
            return_single = match.group(7)

            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1
            end_line = self._find_block_end(content, match.end() - 1, line_num)

            doc_comment = self._find_doc_comment(content, start_pos)

            return_type = return_tuple or return_single or ""
            is_method = bool(receiver_type)

            signature = "func "
            if is_method:
                signature += f"({receiver_name} *{receiver_type}) "
            signature += f"{func_name}({params})"
            if return_type:
                signature += f" {return_type}"

            func_node = DocumentNode(
                type=NodeType.FUNCTION,
                name=func_name,
                content=content[match.start() : self._get_pos_at_line(content, end_line + 1)],
                signature=signature,
                return_type=return_type.strip() if return_type else None,
                docstring=doc_comment,
                location=SourceLocation(start_line=line_num, end_line=end_line),
                parent_id=root.id,
                attributes={
                    "is_method": is_method,
                    "receiver_type": receiver_type,
                    "receiver_name": receiver_name,
                    "is_exported": func_name[0].isupper(),
                    "type_params": type_params,
                },
            )
            root.children.append(func_node)

            # Use qualified name for methods
            key = f"{receiver_type}.{func_name}" if is_method else func_name
            symbols[key] = func_node

    def _extract_type_aliases(
        self,
        content: str,
        root: DocumentNode,
        symbols: dict[str, DocumentNode],
    ) -> None:
        """Extract type alias definitions."""
        for match in self.TYPE_ALIAS_PATTERN.finditer(content):
            name = match.group(1)
            underlying = match.group(2)

            # Skip if already captured as struct/interface
            if name in symbols:
                continue

            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1

            type_node = DocumentNode(
                type=NodeType.VARIABLE,
                name=name,
                content=match.group(0),
                location=SourceLocation(start_line=line_num, end_line=line_num),
                parent_id=root.id,
                attributes={
                    "is_type_alias": True,
                    "underlying_type": underlying,
                    "is_exported": name[0].isupper(),
                },
            )
            root.children.append(type_node)
            symbols[name] = type_node

    def _find_doc_comment(self, content: str, pos: int) -> str | None:
        """Find Go doc comment preceding a position."""
        search_start = max(0, pos - 500)
        segment = content[search_start:pos]

        # Look for // comments immediately before
        lines = segment.rstrip().split("\n")
        doc_lines = []

        for line in reversed(lines):
            line = line.strip()
            if line.startswith("//"):
                doc_lines.insert(0, line[2:].strip())
            elif not line:
                continue
            else:
                break

        return "\n".join(doc_lines) if doc_lines else None

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
        """Chunk Go by definitions."""
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
        """Extract relationships from imports."""
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
                        "alias": ref.attributes.get("alias"),
                        "is_stdlib": ref.attributes.get("is_stdlib", False),
                    },
                )
                relationships.append(rel)

        return relationships

    def prepare_for_embedding(
        self,
        chunk: DocumentChunk,
        document: ParsedDocument,
    ) -> str:
        """Prepare Go chunk for embedding."""
        parts = []
        node = document.get_node(chunk.node_ids[0]) if chunk.node_ids else None

        if document.metadata.get("package"):
            parts.append(f"Go Package: {document.metadata['package']}")

        if node:
            if node.type == NodeType.CLASS:
                if node.attributes.get("is_interface"):
                    parts.append(f"Go Interface: {node.name}")
                else:
                    parts.append(f"Go Struct: {node.name}")
            elif node.type == NodeType.FUNCTION:
                if node.attributes.get("is_method"):
                    receiver = node.attributes.get("receiver_type")
                    parts.append(f"Go Method: {receiver}.{node.name}")
                else:
                    parts.append(f"Go Function: {node.name}")

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

        if node.attributes.get("receiver_type"):
            keywords.append(node.attributes["receiver_type"])

        return keywords

    def _create_summary(self, node: DocumentNode) -> str:
        """Create summary for node."""
        if node.type == NodeType.CLASS:
            if node.attributes.get("is_interface"):
                return f"Go interface {node.name}"
            return f"Go struct {node.name}"
        elif node.type == NodeType.FUNCTION:
            if node.attributes.get("is_method"):
                receiver = node.attributes.get("receiver_type")
                return f"Method {receiver}.{node.name}"
            return f"Go function {node.name}"
        return ""
