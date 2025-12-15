"""
C/C++ file handler.

Extracts:
- Classes, structs, unions
- Functions and methods
- Namespaces
- Templates
- Preprocessor directives
- Header includes
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


class CppHandler(FileTypeHandler):
    """Handler for C/C++ files."""

    name: str = "cpp"
    extensions: list[str] = [
        ".cpp",
        ".cc",
        ".cxx",
        ".c++",
        ".c",
        ".h",
        ".hpp",
        ".hxx",
        ".h++",
        ".hh",
    ]
    mime_types: list[str] = ["text/x-c", "text/x-c++", "text/x-c-header"]
    chunk_strategy: ChunkStrategy = ChunkStrategy.AST_BOUNDARY

    # Patterns
    INCLUDE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r'#include\s+[<"]([^>"]+)[>"]', re.MULTILINE
    )
    DEFINE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"#define\s+(\w+)(?:\([^)]*\))?\s*(.*?)(?=\n(?!\\)|\Z)", re.MULTILINE | re.DOTALL
    )
    NAMESPACE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"namespace\s+(\w+)\s*\{", re.MULTILINE
    )
    CLASS_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:template\s*<([^>]+)>\s*)?(class|struct)\s+(?:__declspec\([^)]+\)\s+)?(\w+)(?:\s*:\s*(public|private|protected)\s+(\w+)(?:<[^>]+>)?(?:\s*,\s*(?:public|private|protected)\s+\w+(?:<[^>]+>)?)*)?\s*\{",
        re.MULTILINE,
    )
    FUNCTION_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:template\s*<([^>]+)>\s*)?(?:(virtual|static|inline|explicit|constexpr)\s+)*(?:(\w+(?:<[^>]+>)?(?:\s*[*&]+)?)\s+)?(\w+)\s*\(([^)]*)\)\s*(?:const)?\s*(?:override)?\s*(?:final)?\s*(?:noexcept(?:\([^)]*\))?)?\s*(?:->([^{;]+))?\s*[{;]",
        re.MULTILINE,
    )
    TYPEDEF_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"typedef\s+(.+?)\s+(\w+)\s*;", re.MULTILINE
    )
    USING_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"using\s+(\w+)\s*=\s*([^;]+);", re.MULTILINE
    )
    ENUM_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"enum\s+(?:class\s+)?(\w+)(?:\s*:\s*\w+)?\s*\{", re.MULTILINE
    )

    def parse(self, content: str, file_path: str) -> ParsedDocument:
        """Parse C/C++ file."""
        lines = content.split("\n")
        is_header = file_path.endswith((".h", ".hpp", ".hxx", ".h++", ".hh"))
        is_cpp = file_path.endswith((".cpp", ".cc", ".cxx", ".c++"))

        root = DocumentNode(
            type=NodeType.MODULE,
            name=Path(file_path).stem,
            content="",
            location=SourceLocation(start_line=1, end_line=len(lines)),
        )

        symbols: dict[str, DocumentNode] = {}
        references: list[DocumentNode] = []

        # Extract includes
        self._extract_includes(content, root, references)

        # Extract macros
        self._extract_macros(content, root, symbols)

        # Extract namespaces
        self._extract_namespaces(content, root, symbols)

        # Extract classes/structs
        self._extract_classes(content, root, symbols)

        # Extract functions
        self._extract_functions(content, root, symbols)

        # Extract enums
        self._extract_enums(content, root, symbols)

        # Extract typedefs and using
        self._extract_typedefs(content, root, symbols)

        doc = ParsedDocument(
            file_path=file_path,
            file_type="cpp" if is_cpp else ("c" if file_path.endswith(".c") else "c_header"),
            raw_content=content,
            root=root,
            symbols=symbols,
            references=references,
            language="cpp",
            line_count=len(lines),
            char_count=len(content),
            node_count=len(root.walk()),
            metadata={
                "is_header": is_header,
                "has_templates": "template" in content,
                "has_pragma_once": "#pragma once" in content,
            },
        )

        doc.chunks = self.chunk(doc)
        return doc

    def _extract_includes(
        self,
        content: str,
        root: DocumentNode,
        references: list[DocumentNode],
    ) -> None:
        """Extract include directives."""
        for match in self.INCLUDE_PATTERN.finditer(content):
            header = match.group(1)
            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1
            is_system = match.group(0).startswith("#include <")

            include_node = DocumentNode(
                type=NodeType.IMPORT,
                name=header,
                content=match.group(0),
                target=header,
                location=SourceLocation(start_line=line_num, end_line=line_num),
                parent_id=root.id,
                attributes={
                    "header": header,
                    "is_system": is_system,
                },
            )
            root.children.append(include_node)
            references.append(include_node)

    def _extract_macros(
        self,
        content: str,
        root: DocumentNode,
        symbols: dict[str, DocumentNode],
    ) -> None:
        """Extract macro definitions."""
        for match in self.DEFINE_PATTERN.finditer(content):
            name = match.group(1)
            value = match.group(2).strip()
            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1

            macro_node = DocumentNode(
                type=NodeType.VARIABLE,
                name=name,
                content=match.group(0),
                location=SourceLocation(start_line=line_num, end_line=line_num + value.count("\\")),
                parent_id=root.id,
                attributes={
                    "is_macro": True,
                    "is_function_macro": "(" in match.group(0).split()[1]
                    if len(match.group(0).split()) > 1
                    else False,
                },
            )
            root.children.append(macro_node)
            symbols[name] = macro_node

    def _extract_namespaces(
        self,
        content: str,
        root: DocumentNode,
        symbols: dict[str, DocumentNode],
    ) -> None:
        """Extract namespace definitions."""
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
                attributes={"is_namespace": True},
            )
            root.children.append(ns_node)
            symbols[name] = ns_node

    def _extract_classes(
        self,
        content: str,
        root: DocumentNode,
        symbols: dict[str, DocumentNode],
    ) -> None:
        """Extract class/struct definitions."""
        for match in self.CLASS_PATTERN.finditer(content):
            template = match.group(1)
            kind = match.group(2)
            name = match.group(3)
            base_access = match.group(4)
            base_class = match.group(5)

            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1
            end_line = self._find_block_end(content, match.end() - 1, line_num)

            class_node = DocumentNode(
                type=NodeType.CLASS,
                name=name,
                content=content[match.start() : self._get_pos_at_line(content, end_line + 1)],
                signature=f"{'template<' + template + '> ' if template else ''}{kind} {name}",
                location=SourceLocation(start_line=line_num, end_line=end_line),
                parent_id=root.id,
                attributes={
                    "is_struct": kind == "struct",
                    "is_template": bool(template),
                    "template_params": template,
                    "base_class": base_class,
                    "base_access": base_access,
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
        for match in self.FUNCTION_PATTERN.finditer(content):
            template = match.group(1)
            modifier = match.group(2)
            return_type = match.group(3)
            name = match.group(4)
            params = match.group(5)
            trailing_return = match.group(6)

            # Skip if it's a control statement
            if name in ("if", "for", "while", "switch", "catch"):
                continue

            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1

            # Determine if declaration or definition
            is_definition = match.group(0).rstrip().endswith("{")
            end_line = line_num
            if is_definition:
                end_line = self._find_block_end(content, match.end() - 1, line_num)

            func_node = DocumentNode(
                type=NodeType.FUNCTION,
                name=name,
                content=content[match.start() : self._get_pos_at_line(content, end_line + 1)]
                if is_definition
                else match.group(0),
                signature=f"{return_type or ''} {name}({params})",
                return_type=trailing_return.strip() if trailing_return else return_type,
                location=SourceLocation(start_line=line_num, end_line=end_line),
                parent_id=root.id,
                attributes={
                    "is_template": bool(template),
                    "template_params": template,
                    "is_virtual": modifier == "virtual",
                    "is_static": modifier == "static",
                    "is_inline": modifier == "inline",
                    "is_definition": is_definition,
                },
            )

            if name not in symbols:
                root.children.append(func_node)
                symbols[name] = func_node

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

            is_class = "enum class" in match.group(0)

            enum_node = DocumentNode(
                type=NodeType.VARIABLE,
                name=name,
                content=content[match.start() : self._get_pos_at_line(content, end_line + 1)],
                location=SourceLocation(start_line=line_num, end_line=end_line),
                parent_id=root.id,
                attributes={
                    "is_enum": True,
                    "is_enum_class": is_class,
                },
            )
            root.children.append(enum_node)
            symbols[name] = enum_node

    def _extract_typedefs(
        self,
        content: str,
        root: DocumentNode,
        symbols: dict[str, DocumentNode],
    ) -> None:
        """Extract typedef and using declarations."""
        for match in self.TYPEDEF_PATTERN.finditer(content):
            original = match.group(1)
            name = match.group(2)
            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1

            typedef_node = DocumentNode(
                type=NodeType.VARIABLE,
                name=name,
                content=match.group(0),
                location=SourceLocation(start_line=line_num, end_line=line_num),
                parent_id=root.id,
                attributes={
                    "is_typedef": True,
                    "original_type": original.strip(),
                },
            )
            root.children.append(typedef_node)
            symbols[name] = typedef_node

        for match in self.USING_PATTERN.finditer(content):
            name = match.group(1)
            alias_type = match.group(2)
            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1

            using_node = DocumentNode(
                type=NodeType.VARIABLE,
                name=name,
                content=match.group(0),
                location=SourceLocation(start_line=line_num, end_line=line_num),
                parent_id=root.id,
                attributes={
                    "is_type_alias": True,
                    "alias_type": alias_type.strip(),
                },
            )
            if name not in symbols:
                root.children.append(using_node)
                symbols[name] = using_node

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
        """Chunk C/C++ by definitions."""
        chunks: list[DocumentChunk] = []

        for node in document.root.children:
            if node.type in (NodeType.CLASS, NodeType.FUNCTION, NodeType.MODULE):
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
        """Extract relationships from includes and inheritance."""
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
                        "is_system": ref.attributes.get("is_system", False),
                    },
                )
                relationships.append(rel)

        for node in document.root.walk():
            if node.type == NodeType.CLASS:
                base = node.attributes.get("base_class")
                if base:
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

        return relationships

    def prepare_for_embedding(
        self,
        chunk: DocumentChunk,
        document: ParsedDocument,
    ) -> str:
        """Prepare C/C++ chunk for embedding."""
        parts = []
        node = document.get_node(chunk.node_ids[0]) if chunk.node_ids else None

        lang = "C++" if document.file_type == "cpp" else "C"
        parts.append(f"Language: {lang}")

        if node:
            if node.type == NodeType.CLASS:
                kind = "Struct" if node.attributes.get("is_struct") else "Class"
                parts.append(f"{lang} {kind}: {node.name}")
            elif node.type == NodeType.FUNCTION:
                parts.append(f"{lang} Function: {node.name}")
            elif node.type == NodeType.MODULE:
                parts.append(f"{lang} Namespace: {node.name}")

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

        if node.type == NodeType.CLASS and node.attributes.get("base_class"):
            keywords.append(node.attributes["base_class"])

        return keywords

    def _create_summary(self, node: DocumentNode) -> str:
        """Create summary for node."""
        if node.type == NodeType.CLASS:
            kind = "Struct" if node.attributes.get("is_struct") else "Class"
            template = "Template " if node.attributes.get("is_template") else ""
            return f"{template}{kind} {node.name}"
        elif node.type == NodeType.FUNCTION:
            template = "Template " if node.attributes.get("is_template") else ""
            return f"{template}Function {node.name}"
        elif node.type == NodeType.MODULE:
            return f"Namespace {node.name}"
        return ""
