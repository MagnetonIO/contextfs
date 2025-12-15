"""
Shell/Bash file handler.

Extracts:
- Functions
- Variables and exports
- Aliases
- Source/include statements
- Shebang
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


class ShellHandler(FileTypeHandler):
    """Handler for Shell/Bash files."""

    name: str = "shell"
    extensions: list[str] = [".sh", ".bash", ".zsh", ".fish", ".ksh", ".csh", ".tcsh"]
    mime_types: list[str] = ["text/x-shellscript", "application/x-sh"]
    chunk_strategy: ChunkStrategy = ChunkStrategy.AST_BOUNDARY

    # Patterns
    SHEBANG_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"^#!\s*(/\S+)")
    FUNCTION_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:function\s+)?(\w+)\s*\(\s*\)\s*\{", re.MULTILINE
    )
    VARIABLE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"^(\w+)=(.*)$", re.MULTILINE)
    EXPORT_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"^export\s+(\w+)(?:=(.*))?$", re.MULTILINE
    )
    ALIAS_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"^alias\s+(\w+)=['\"]?([^'\"#\n]+)['\"]?", re.MULTILINE
    )
    SOURCE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:source|\\.)\s+['\"]?([^'\"#\n]+)['\"]?", re.MULTILINE
    )

    def parse(self, content: str, file_path: str) -> ParsedDocument:
        """Parse Shell file."""
        lines = content.split("\n")

        root = DocumentNode(
            type=NodeType.MODULE,
            name=Path(file_path).stem,
            content="",
            location=SourceLocation(start_line=1, end_line=len(lines)),
        )

        symbols: dict[str, DocumentNode] = {}
        references: list[DocumentNode] = []

        # Extract shebang
        shebang = self._extract_shebang(content)

        # Extract source statements
        self._extract_sources(content, root, references)

        # Extract functions
        self._extract_functions(content, root, symbols)

        # Extract exports
        self._extract_exports(content, root, symbols)

        # Extract aliases
        self._extract_aliases(content, root, symbols)

        doc = ParsedDocument(
            file_path=file_path,
            file_type="shell",
            raw_content=content,
            root=root,
            symbols=symbols,
            references=references,
            language=self._detect_shell(shebang, file_path),
            line_count=len(lines),
            char_count=len(content),
            node_count=len(root.walk()),
            metadata={
                "shebang": shebang,
                "shell_type": self._detect_shell(shebang, file_path),
                "is_executable": shebang is not None,
            },
        )

        doc.chunks = self.chunk(doc)
        return doc

    def _extract_shebang(self, content: str) -> str | None:
        """Extract shebang line."""
        match = self.SHEBANG_PATTERN.match(content)
        return match.group(1) if match else None

    def _detect_shell(self, shebang: str | None, file_path: str) -> str:
        """Detect shell type from shebang or extension."""
        if shebang:
            if "bash" in shebang:
                return "bash"
            elif "zsh" in shebang:
                return "zsh"
            elif "fish" in shebang:
                return "fish"
            elif "ksh" in shebang:
                return "ksh"
            elif "csh" in shebang or "tcsh" in shebang:
                return "csh"

        ext = Path(file_path).suffix
        ext_map = {
            ".bash": "bash",
            ".zsh": "zsh",
            ".fish": "fish",
            ".ksh": "ksh",
            ".csh": "csh",
            ".tcsh": "tcsh",
        }
        return ext_map.get(ext, "bash")

    def _extract_sources(
        self,
        content: str,
        root: DocumentNode,
        references: list[DocumentNode],
    ) -> None:
        """Extract source/include statements."""
        for match in self.SOURCE_PATTERN.finditer(content):
            sourced_file = match.group(1).strip()
            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1

            source_node = DocumentNode(
                type=NodeType.IMPORT,
                name=sourced_file,
                content=match.group(0),
                target=sourced_file,
                location=SourceLocation(start_line=line_num, end_line=line_num),
                parent_id=root.id,
                attributes={
                    "file": sourced_file,
                    "is_relative": not sourced_file.startswith("/"),
                },
            )
            root.children.append(source_node)
            references.append(source_node)

    def _extract_functions(
        self,
        content: str,
        root: DocumentNode,
        symbols: dict[str, DocumentNode],
    ) -> None:
        """Extract function definitions."""
        for match in self.FUNCTION_PATTERN.finditer(content):
            name = match.group(1)
            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1
            end_line = self._find_block_end(content, match.end() - 1, line_num)

            doc_comment = self._find_doc_comment(content, start_pos)

            func_node = DocumentNode(
                type=NodeType.FUNCTION,
                name=name,
                content=content[match.start() : self._get_pos_at_line(content, end_line + 1)],
                signature=f"{name}()",
                docstring=doc_comment,
                location=SourceLocation(start_line=line_num, end_line=end_line),
                parent_id=root.id,
            )
            root.children.append(func_node)
            symbols[name] = func_node

    def _extract_exports(
        self,
        content: str,
        root: DocumentNode,
        symbols: dict[str, DocumentNode],
    ) -> None:
        """Extract export statements."""
        for match in self.EXPORT_PATTERN.finditer(content):
            name = match.group(1)
            value = match.group(2)
            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1

            export_node = DocumentNode(
                type=NodeType.VARIABLE,
                name=name,
                content=match.group(0),
                location=SourceLocation(start_line=line_num, end_line=line_num),
                parent_id=root.id,
                attributes={
                    "is_export": True,
                    "value": value.strip() if value else None,
                },
            )
            root.children.append(export_node)
            symbols[name] = export_node

    def _extract_aliases(
        self,
        content: str,
        root: DocumentNode,
        symbols: dict[str, DocumentNode],
    ) -> None:
        """Extract alias definitions."""
        for match in self.ALIAS_PATTERN.finditer(content):
            name = match.group(1)
            command = match.group(2)
            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1

            alias_node = DocumentNode(
                type=NodeType.VARIABLE,
                name=name,
                content=match.group(0),
                location=SourceLocation(start_line=line_num, end_line=line_num),
                parent_id=root.id,
                attributes={
                    "is_alias": True,
                    "command": command.strip(),
                },
            )
            root.children.append(alias_node)
            symbols[name] = alias_node

    def _find_doc_comment(self, content: str, pos: int) -> str | None:
        """Find shell doc comment preceding a position."""
        search_start = max(0, pos - 300)
        segment = content[search_start:pos]

        doc_lines = []
        for line in reversed(segment.rstrip().split("\n")):
            line = line.strip()
            if line.startswith("#") and not line.startswith("#!"):
                doc_lines.insert(0, line[1:].strip())
            elif not line:
                continue
            else:
                break

        return "\n".join(doc_lines) if doc_lines else None

    def _find_block_end(self, content: str, start: int, start_line: int) -> int:
        """Find the end line of a function block."""
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
        """Chunk Shell by function definitions."""
        chunks: list[DocumentChunk] = []

        for node in document.root.children:
            if node.type == NodeType.FUNCTION:
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
                    summary=f"Shell function {node.name}",
                )
                chunk.embedding_text = self.prepare_for_embedding(chunk, document)
                chunks.append(chunk)

        # If no functions, chunk the whole file
        if not chunks and document.raw_content.strip():
            chunk = DocumentChunk(
                content=document.raw_content,
                document_id=document.id,
                file_path=document.file_path,
                chunk_index=0,
                total_chunks=1,
                strategy=ChunkStrategy.FIXED_SIZE,
                summary=f"Shell script {document.root.name}",
            )
            chunk.embedding_text = self.prepare_for_embedding(chunk, document)
            chunks.append(chunk)

        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks

    def extract_relationships(self, document: ParsedDocument) -> list[Relationship]:
        """Extract relationships from source statements."""
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
                        "is_relative": ref.attributes.get("is_relative", False),
                    },
                )
                relationships.append(rel)

        return relationships

    def prepare_for_embedding(
        self,
        chunk: DocumentChunk,
        document: ParsedDocument,
    ) -> str:
        """Prepare Shell chunk for embedding."""
        parts = []
        shell_type = document.metadata.get("shell_type", "bash")
        parts.append(f"Language: {shell_type.title()} Shell Script")

        node = document.get_node(chunk.node_ids[0]) if chunk.node_ids else None

        if node:
            if node.type == NodeType.FUNCTION:
                parts.append(f"Shell Function: {node.name}")
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
        return keywords
