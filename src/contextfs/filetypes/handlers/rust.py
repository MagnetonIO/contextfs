"""
Rust file handler.

Extracts:
- Modules and use statements
- Functions and methods
- Structs, enums, traits
- Impl blocks
- Macros
- Lifetimes and generics
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


class RustHandler(FileTypeHandler):
    """Handler for Rust files."""

    name: str = "rust"
    extensions: list[str] = [".rs"]
    mime_types: list[str] = ["text/x-rust"]
    chunk_strategy: ChunkStrategy = ChunkStrategy.AST_BOUNDARY

    # Patterns
    USE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"use\s+([\w:]+)(?:::\{([^}]+)\}|::(\w+))?;", re.MULTILINE
    )
    MOD_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:pub\s+)?mod\s+(\w+)\s*[{;]", re.MULTILINE
    )
    STRUCT_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:#\[([^\]]+)\]\s*)*(?:pub(?:\([^)]+\))?\s+)?struct\s+(\w+)(?:<([^>]+)>)?(?:\s*\([^)]+\))?(?:\s*where\s+[^{]+)?\s*[{;]",
        re.MULTILINE,
    )
    ENUM_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:#\[([^\]]+)\]\s*)*(?:pub(?:\([^)]+\))?\s+)?enum\s+(\w+)(?:<([^>]+)>)?\s*\{",
        re.MULTILINE,
    )
    TRAIT_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:pub(?:\([^)]+\))?\s+)?(?:unsafe\s+)?trait\s+(\w+)(?:<([^>]+)>)?(?:\s*:\s*([^{]+))?\s*\{",
        re.MULTILINE,
    )
    IMPL_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:unsafe\s+)?impl(?:<([^>]+)>)?\s+(?:(\w+)(?:<[^>]+>)?\s+for\s+)?(\w+)(?:<[^>]+>)?(?:\s*where\s+[^{]+)?\s*\{",
        re.MULTILINE,
    )
    FN_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:#\[([^\]]+)\]\s*)*(?:pub(?:\([^)]+\))?\s+)?(?:async\s+)?(?:unsafe\s+)?(?:const\s+)?(?:extern\s+\"[^\"]+\"\s+)?fn\s+(\w+)(?:<([^>]+)>)?\s*\(([^)]*)\)(?:\s*->\s*([^{;]+))?\s*(?:where\s+[^{]+)?\s*[{;]",
        re.MULTILINE,
    )
    CONST_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:pub(?:\([^)]+\))?\s+)?const\s+(\w+)\s*:\s*([^=]+)\s*=", re.MULTILINE
    )
    STATIC_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:pub(?:\([^)]+\))?\s+)?static\s+(?:mut\s+)?(\w+)\s*:\s*([^=]+)\s*=", re.MULTILINE
    )
    TYPE_ALIAS_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:pub(?:\([^)]+\))?\s+)?type\s+(\w+)(?:<([^>]+)>)?\s*=\s*([^;]+);", re.MULTILINE
    )
    MACRO_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"macro_rules!\s+(\w+)\s*\{", re.MULTILINE
    )
    DERIVE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"#\[derive\(([^)]+)\)\]")
    DOC_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"///\s*(.*)")

    def parse(self, content: str, file_path: str) -> ParsedDocument:
        """Parse Rust file."""
        lines = content.split("\n")

        root = DocumentNode(
            type=NodeType.MODULE,
            name=Path(file_path).stem,
            content="",
            location=SourceLocation(start_line=1, end_line=len(lines)),
        )

        symbols: dict[str, DocumentNode] = {}
        references: list[DocumentNode] = []

        # Extract use statements
        self._extract_uses(content, root, references)

        # Extract modules
        self._extract_modules(content, root, symbols)

        # Extract structs
        self._extract_structs(content, root, symbols)

        # Extract enums
        self._extract_enums(content, root, symbols)

        # Extract traits
        self._extract_traits(content, root, symbols)

        # Extract impl blocks
        self._extract_impls(content, root, symbols)

        # Extract functions
        self._extract_functions(content, root, symbols)

        # Extract macros
        self._extract_macros(content, root, symbols)

        doc = ParsedDocument(
            file_path=file_path,
            file_type="rust",
            raw_content=content,
            root=root,
            symbols=symbols,
            references=references,
            language="rust",
            line_count=len(lines),
            char_count=len(content),
            node_count=len(root.walk()),
            metadata={
                "is_lib": file_path.endswith("lib.rs"),
                "is_main": file_path.endswith("main.rs"),
                "has_unsafe": "unsafe " in content,
                "has_async": "async " in content,
            },
        )

        doc.chunks = self.chunk(doc)
        return doc

    def _extract_uses(
        self,
        content: str,
        root: DocumentNode,
        references: list[DocumentNode],
    ) -> None:
        """Extract use statements."""
        for match in self.USE_PATTERN.finditer(content):
            path = match.group(1)
            items = match.group(2)
            single = match.group(3)

            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1

            import_node = DocumentNode(
                type=NodeType.IMPORT,
                name=path.split("::")[-1] if not items else path,
                content=match.group(0),
                target=path,
                location=SourceLocation(start_line=line_num, end_line=line_num),
                parent_id=root.id,
                attributes={
                    "path": path,
                    "items": [i.strip() for i in items.split(",")]
                    if items
                    else [single]
                    if single
                    else [],
                    "is_std": path.startswith("std::"),
                    "is_crate": path.startswith("crate::"),
                },
            )
            root.children.append(import_node)
            references.append(import_node)

    def _extract_modules(
        self,
        content: str,
        root: DocumentNode,
        symbols: dict[str, DocumentNode],
    ) -> None:
        """Extract module declarations."""
        for match in self.MOD_PATTERN.finditer(content):
            name = match.group(1)
            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1

            is_inline = match.group(0).endswith("{")
            end_line = line_num
            if is_inline:
                end_line = self._find_block_end(content, match.end() - 1, line_num)

            mod_node = DocumentNode(
                type=NodeType.MODULE,
                name=name,
                content=content[match.start() : self._get_pos_at_line(content, end_line + 1)]
                if is_inline
                else match.group(0),
                location=SourceLocation(start_line=line_num, end_line=end_line),
                parent_id=root.id,
                attributes={
                    "is_inline": is_inline,
                    "is_pub": "pub " in match.group(0),
                },
            )
            root.children.append(mod_node)
            symbols[name] = mod_node

    def _extract_structs(
        self,
        content: str,
        root: DocumentNode,
        symbols: dict[str, DocumentNode],
    ) -> None:
        """Extract struct definitions."""
        for match in self.STRUCT_PATTERN.finditer(content):
            match.group(1)
            name = match.group(2)
            generics = match.group(3)

            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1

            is_tuple = "(" in match.group(0) and "{" not in match.group(0)
            end_line = line_num
            if match.group(0).rstrip().endswith("{"):
                end_line = self._find_block_end(content, match.end() - 1, line_num)

            doc = self._find_doc_comment(content, start_pos)
            derives = self._extract_derives(content, start_pos)

            struct_node = DocumentNode(
                type=NodeType.CLASS,
                name=name,
                content=content[match.start() : self._get_pos_at_line(content, end_line + 1)],
                signature=f"struct {name}{f'<{generics}>' if generics else ''}",
                docstring=doc,
                location=SourceLocation(start_line=line_num, end_line=end_line),
                parent_id=root.id,
                attributes={
                    "is_struct": True,
                    "is_tuple_struct": is_tuple,
                    "is_pub": "pub " in match.group(0),
                    "generics": generics,
                    "derives": derives,
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
            match.group(1)
            name = match.group(2)
            generics = match.group(3)

            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1
            end_line = self._find_block_end(content, match.end() - 1, line_num)

            doc = self._find_doc_comment(content, start_pos)
            derives = self._extract_derives(content, start_pos)

            enum_node = DocumentNode(
                type=NodeType.VARIABLE,
                name=name,
                content=content[match.start() : self._get_pos_at_line(content, end_line + 1)],
                signature=f"enum {name}{f'<{generics}>' if generics else ''}",
                docstring=doc,
                location=SourceLocation(start_line=line_num, end_line=end_line),
                parent_id=root.id,
                attributes={
                    "is_enum": True,
                    "is_pub": "pub " in match.group(0),
                    "generics": generics,
                    "derives": derives,
                },
            )
            root.children.append(enum_node)
            symbols[name] = enum_node

    def _extract_traits(
        self,
        content: str,
        root: DocumentNode,
        symbols: dict[str, DocumentNode],
    ) -> None:
        """Extract trait definitions."""
        for match in self.TRAIT_PATTERN.finditer(content):
            name = match.group(1)
            generics = match.group(2)
            bounds = match.group(3)

            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1
            end_line = self._find_block_end(content, match.end() - 1, line_num)

            doc = self._find_doc_comment(content, start_pos)

            trait_node = DocumentNode(
                type=NodeType.CLASS,
                name=name,
                content=content[match.start() : self._get_pos_at_line(content, end_line + 1)],
                signature=f"trait {name}{f'<{generics}>' if generics else ''}",
                docstring=doc,
                location=SourceLocation(start_line=line_num, end_line=end_line),
                parent_id=root.id,
                attributes={
                    "is_trait": True,
                    "is_pub": "pub " in match.group(0),
                    "is_unsafe": "unsafe " in match.group(0),
                    "generics": generics,
                    "bounds": bounds.strip() if bounds else None,
                },
            )
            root.children.append(trait_node)
            symbols[name] = trait_node

    def _extract_impls(
        self,
        content: str,
        root: DocumentNode,
        symbols: dict[str, DocumentNode],
    ) -> None:
        """Extract impl blocks."""
        for match in self.IMPL_PATTERN.finditer(content):
            generics = match.group(1)
            trait_name = match.group(2)
            type_name = match.group(3)

            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1
            end_line = self._find_block_end(content, match.end() - 1, line_num)

            impl_name = f"impl {trait_name} for {type_name}" if trait_name else f"impl {type_name}"

            impl_node = DocumentNode(
                type=NodeType.CLASS,
                name=impl_name,
                content=content[match.start() : self._get_pos_at_line(content, end_line + 1)],
                signature=impl_name,
                location=SourceLocation(start_line=line_num, end_line=end_line),
                parent_id=root.id,
                attributes={
                    "is_impl": True,
                    "trait": trait_name,
                    "type": type_name,
                    "is_unsafe": "unsafe " in match.group(0),
                    "generics": generics,
                },
            )
            root.children.append(impl_node)
            symbols[impl_name] = impl_node

    def _extract_functions(
        self,
        content: str,
        root: DocumentNode,
        symbols: dict[str, DocumentNode],
    ) -> None:
        """Extract function definitions."""
        for match in self.FN_PATTERN.finditer(content):
            match.group(1)
            name = match.group(2)
            generics = match.group(3)
            params = match.group(4)
            return_type = match.group(5)

            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1

            is_definition = match.group(0).rstrip().endswith("{")
            end_line = line_num
            if is_definition:
                end_line = self._find_block_end(content, match.end() - 1, line_num)

            doc = self._find_doc_comment(content, start_pos)

            func_node = DocumentNode(
                type=NodeType.FUNCTION,
                name=name,
                content=content[match.start() : self._get_pos_at_line(content, end_line + 1)]
                if is_definition
                else match.group(0),
                signature=f"fn {name}({params}){f' -> {return_type.strip()}' if return_type else ''}",
                return_type=return_type.strip() if return_type else None,
                docstring=doc,
                location=SourceLocation(start_line=line_num, end_line=end_line),
                parent_id=root.id,
                attributes={
                    "is_pub": "pub " in match.group(0),
                    "is_async": "async " in match.group(0),
                    "is_unsafe": "unsafe " in match.group(0),
                    "is_const": "const " in match.group(0),
                    "generics": generics,
                },
            )

            if name not in symbols:
                root.children.append(func_node)
                symbols[name] = func_node

    def _extract_macros(
        self,
        content: str,
        root: DocumentNode,
        symbols: dict[str, DocumentNode],
    ) -> None:
        """Extract macro definitions."""
        for match in self.MACRO_PATTERN.finditer(content):
            name = match.group(1)
            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1
            end_line = self._find_block_end(content, match.end() - 1, line_num)

            macro_node = DocumentNode(
                type=NodeType.FUNCTION,
                name=name,
                content=content[match.start() : self._get_pos_at_line(content, end_line + 1)],
                signature=f"macro_rules! {name}",
                location=SourceLocation(start_line=line_num, end_line=end_line),
                parent_id=root.id,
                attributes={"is_macro": True},
            )
            root.children.append(macro_node)
            symbols[f"{name}!"] = macro_node

    def _find_doc_comment(self, content: str, pos: int) -> str | None:
        """Find doc comment preceding a position."""
        search_start = max(0, pos - 500)
        segment = content[search_start:pos]

        doc_lines = []
        for line in reversed(segment.rstrip().split("\n")):
            line = line.strip()
            if line.startswith("///"):
                doc_lines.insert(0, line[3:].strip())
            elif line.startswith("#[") or not line:
                continue
            else:
                break

        return "\n".join(doc_lines) if doc_lines else None

    def _extract_derives(self, content: str, pos: int) -> list[str]:
        """Extract derive macros before a position."""
        search_start = max(0, pos - 200)
        segment = content[search_start:pos]

        derives = []
        for match in self.DERIVE_PATTERN.finditer(segment):
            derives.extend(d.strip() for d in match.group(1).split(","))

        return derives

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
        """Chunk Rust by definitions."""
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
        """Extract relationships from use statements and impls."""
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
                        "is_std": ref.attributes.get("is_std", False),
                        "items": ref.attributes.get("items", []),
                    },
                )
                relationships.append(rel)

        for node in document.root.walk():
            if node.type == NodeType.CLASS and node.attributes.get("is_impl"):
                trait = node.attributes.get("trait")
                impl_type = node.attributes.get("type")

                if trait:
                    rel = Relationship(
                        type=RelationType.IMPLEMENTS,
                        source_document_id=document.id,
                        source_node_id=node.id,
                        source_name=impl_type,
                        source_path=document.file_path,
                        target_name=trait,
                        location=node.location,
                    )
                    relationships.append(rel)

        return relationships

    def prepare_for_embedding(
        self,
        chunk: DocumentChunk,
        document: ParsedDocument,
    ) -> str:
        """Prepare Rust chunk for embedding."""
        parts = ["Language: Rust"]
        node = document.get_node(chunk.node_ids[0]) if chunk.node_ids else None

        if node:
            if node.type == NodeType.CLASS:
                if node.attributes.get("is_trait"):
                    parts.append(f"Rust Trait: {node.name}")
                elif node.attributes.get("is_impl"):
                    parts.append(f"Rust Impl: {node.name}")
                elif node.attributes.get("is_struct"):
                    parts.append(f"Rust Struct: {node.name}")
            elif node.type == NodeType.FUNCTION:
                if node.attributes.get("is_macro"):
                    parts.append(f"Rust Macro: {node.name}")
                else:
                    parts.append(f"Rust Function: {node.name}")

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

        if node.attributes.get("trait"):
            keywords.append(node.attributes["trait"])
        if node.attributes.get("type"):
            keywords.append(node.attributes["type"])
        keywords.extend(node.attributes.get("derives", []))

        return keywords

    def _create_summary(self, node: DocumentNode) -> str:
        """Create summary for node."""
        if node.type == NodeType.CLASS:
            if node.attributes.get("is_trait"):
                return f"Rust trait {node.name}"
            if node.attributes.get("is_impl"):
                return f"Rust {node.name}"
            if node.attributes.get("is_struct"):
                return f"Rust struct {node.name}"
        elif node.type == NodeType.FUNCTION:
            if node.attributes.get("is_macro"):
                return f"Rust macro {node.name}"
            return f"Rust fn {node.name}"
        return ""
