"""
Ruby file handler.

Extracts:
- Classes and modules
- Methods (instance and class)
- Blocks and procs
- Require/require_relative statements
- Attr accessors
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


class RubyHandler(FileTypeHandler):
    """Handler for Ruby files."""

    name: str = "ruby"
    extensions: list[str] = [".rb", ".rake", ".gemspec", ".ru", ".erb"]
    mime_types: list[str] = ["text/x-ruby", "application/x-ruby"]
    chunk_strategy: ChunkStrategy = ChunkStrategy.AST_BOUNDARY

    # Patterns
    REQUIRE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"require(?:_relative)?\s+['\"]([^'\"]+)['\"]", re.MULTILINE
    )
    MODULE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"module\s+(\w+(?:::\w+)*)", re.MULTILINE
    )
    CLASS_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"class\s+(\w+(?:::\w+)*)(?:\s*<\s*(\w+(?:::\w+)*))?\s*$", re.MULTILINE
    )
    METHOD_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"def\s+(self\.)?(\w+[?!=]?)\s*(?:\(([^)]*)\))?", re.MULTILINE
    )
    ATTR_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"attr_(reader|writer|accessor)\s+(.+?)(?=\n|#)", re.MULTILINE
    )
    CONSTANT_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"([A-Z][A-Z0-9_]*)\s*=", re.MULTILINE)
    INCLUDE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:include|extend|prepend)\s+(\w+(?:::\w+)*)", re.MULTILINE
    )

    def parse(self, content: str, file_path: str) -> ParsedDocument:
        """Parse Ruby file."""
        lines = content.split("\n")

        root = DocumentNode(
            type=NodeType.MODULE,
            name=Path(file_path).stem,
            content="",
            location=SourceLocation(start_line=1, end_line=len(lines)),
        )

        symbols: dict[str, DocumentNode] = {}
        references: list[DocumentNode] = []

        # Extract requires
        self._extract_requires(content, root, references)

        # Extract modules
        self._extract_modules(content, root, symbols)

        # Extract classes
        self._extract_classes(content, root, symbols)

        # Extract standalone methods
        self._extract_methods(content, root, symbols)

        doc = ParsedDocument(
            file_path=file_path,
            file_type="ruby",
            raw_content=content,
            root=root,
            symbols=symbols,
            references=references,
            language="ruby",
            line_count=len(lines),
            char_count=len(content),
            node_count=len(root.walk()),
            metadata={
                "is_rails": "Rails" in content or "ActiveRecord" in content,
                "is_spec": "_spec.rb" in file_path or "spec/" in file_path,
                "is_rake": file_path.endswith(".rake"),
            },
        )

        doc.chunks = self.chunk(doc)
        return doc

    def _extract_requires(
        self,
        content: str,
        root: DocumentNode,
        references: list[DocumentNode],
    ) -> None:
        """Extract require statements."""
        for match in self.REQUIRE_PATTERN.finditer(content):
            lib = match.group(1)
            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1

            is_relative = "require_relative" in match.group(0)

            require_node = DocumentNode(
                type=NodeType.IMPORT,
                name=lib,
                content=match.group(0),
                target=lib,
                location=SourceLocation(start_line=line_num, end_line=line_num),
                parent_id=root.id,
                attributes={
                    "lib": lib,
                    "is_relative": is_relative,
                },
            )
            root.children.append(require_node)
            references.append(require_node)

    def _extract_modules(
        self,
        content: str,
        root: DocumentNode,
        symbols: dict[str, DocumentNode],
    ) -> None:
        """Extract module definitions."""
        for match in self.MODULE_PATTERN.finditer(content):
            name = match.group(1)
            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1
            end_line = self._find_end(content, match.end(), line_num)

            doc_comment = self._find_doc_comment(content, start_pos)

            module_node = DocumentNode(
                type=NodeType.MODULE,
                name=name,
                content=content[match.start() : self._get_pos_at_line(content, end_line + 1)],
                docstring=doc_comment,
                location=SourceLocation(start_line=line_num, end_line=end_line),
                parent_id=root.id,
                attributes={"is_module": True},
            )
            root.children.append(module_node)
            symbols[name] = module_node

    def _extract_classes(
        self,
        content: str,
        root: DocumentNode,
        symbols: dict[str, DocumentNode],
    ) -> None:
        """Extract class definitions."""
        for match in self.CLASS_PATTERN.finditer(content):
            name = match.group(1)
            superclass = match.group(2)
            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1
            end_line = self._find_end(content, match.end(), line_num)

            doc_comment = self._find_doc_comment(content, start_pos)

            # Extract includes/extends within class
            class_content = content[match.end() : self._get_pos_at_line(content, end_line)]
            includes = [m.group(1) for m in self.INCLUDE_PATTERN.finditer(class_content)]

            class_node = DocumentNode(
                type=NodeType.CLASS,
                name=name,
                content=content[match.start() : self._get_pos_at_line(content, end_line + 1)],
                signature=f"class {name}{f' < {superclass}' if superclass else ''}",
                docstring=doc_comment,
                location=SourceLocation(start_line=line_num, end_line=end_line),
                parent_id=root.id,
                attributes={
                    "superclass": superclass,
                    "includes": includes,
                },
            )

            # Extract methods
            self._extract_class_methods(class_content, class_node, line_num)

            root.children.append(class_node)
            symbols[name] = class_node

    def _extract_methods(
        self,
        content: str,
        root: DocumentNode,
        symbols: dict[str, DocumentNode],
    ) -> None:
        """Extract standalone method definitions."""
        lines = content.split("\n")
        in_class = False
        class_depth = 0

        for i, line in enumerate(lines):
            if re.match(r"\s*(class|module)\s+", line):
                in_class = True
                class_depth += 1
            elif re.match(r"\s*end\s*$", line) and in_class:
                class_depth -= 1
                if class_depth == 0:
                    in_class = False

            if not in_class:
                match = self.METHOD_PATTERN.match(line.strip())
                if match:
                    is_class_method = bool(match.group(1))
                    name = match.group(2)
                    params = match.group(3) or ""

                    line_num = i + 1
                    end_line = self._find_method_end(lines, i)

                    doc_comment = self._find_line_doc_comment(lines, i)

                    method_node = DocumentNode(
                        type=NodeType.FUNCTION,
                        name=name,
                        content="\n".join(lines[i : end_line + 1]),
                        signature=f"def {name}({params})",
                        docstring=doc_comment,
                        location=SourceLocation(start_line=line_num, end_line=end_line + 1),
                        parent_id=root.id,
                        attributes={
                            "is_class_method": is_class_method,
                        },
                    )
                    root.children.append(method_node)
                    symbols[name] = method_node

    def _extract_class_methods(
        self,
        content: str,
        parent: DocumentNode,
        base_line: int,
    ) -> None:
        """Extract method definitions within a class."""
        for match in self.METHOD_PATTERN.finditer(content):
            is_class_method = bool(match.group(1))
            name = match.group(2)
            params = match.group(3) or ""

            start_pos = match.start()
            line_num = base_line + content[:start_pos].count("\n")

            method_node = DocumentNode(
                type=NodeType.FUNCTION,
                name=name,
                content=match.group(0),
                signature=f"def {'self.' if is_class_method else ''}{name}({params})",
                location=SourceLocation(start_line=line_num, end_line=line_num),
                parent_id=parent.id,
                attributes={
                    "is_class_method": is_class_method,
                    "is_predicate": name.endswith("?"),
                    "is_bang": name.endswith("!"),
                    "is_setter": name.endswith("="),
                },
            )
            parent.children.append(method_node)

    def _find_end(self, content: str, start: int, start_line: int) -> int:
        """Find matching 'end' for Ruby block."""
        depth = 1
        line = start_line
        keywords = re.compile(
            r"\b(class|module|def|do|if|unless|case|while|until|for|begin)\b(?!\s*:)"
        )
        end_pattern = re.compile(r"\bend\b")

        for i, char in enumerate(content[start:], start):
            if char == "\n":
                line += 1
                # Check current line for keywords
                line_end = content.find("\n", i + 1)
                if line_end == -1:
                    line_end = len(content)
                current_line = content[i + 1 : line_end]

                # Count block openers
                depth += len(keywords.findall(current_line))
                # Count ends
                depth -= len(end_pattern.findall(current_line))

                if depth <= 0:
                    return line

        return line

    def _find_method_end(self, lines: list[str], start: int) -> int:
        """Find end of method definition."""
        depth = 1
        for i in range(start + 1, len(lines)):
            line = lines[i].strip()
            if re.match(r"(def|class|module|do|if|unless|case|while|until|for|begin)\b", line):
                depth += 1
            elif line == "end":
                depth -= 1
                if depth == 0:
                    return i
        return start

    def _find_doc_comment(self, content: str, pos: int) -> str | None:
        """Find Ruby doc comment preceding a position."""
        search_start = max(0, pos - 500)
        segment = content[search_start:pos]

        doc_lines = []
        for line in reversed(segment.rstrip().split("\n")):
            line = line.strip()
            if line.startswith("#"):
                doc_lines.insert(0, line[1:].strip())
            elif not line:
                continue
            else:
                break

        return "\n".join(doc_lines) if doc_lines else None

    def _find_line_doc_comment(self, lines: list[str], line_idx: int) -> str | None:
        """Find doc comment on preceding lines."""
        doc_lines = []
        for i in range(line_idx - 1, -1, -1):
            line = lines[i].strip()
            if line.startswith("#"):
                doc_lines.insert(0, line[1:].strip())
            elif not line:
                continue
            else:
                break
        return "\n".join(doc_lines) if doc_lines else None

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
        """Chunk Ruby by class/module/method definitions."""
        chunks: list[DocumentChunk] = []

        for node in document.root.children:
            if node.type in (NodeType.CLASS, NodeType.MODULE, NodeType.FUNCTION):
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
        """Extract relationships from requires and inheritance."""
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
                    attributes={"is_relative": ref.attributes.get("is_relative", False)},
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

                for include in node.attributes.get("includes", []):
                    rel = Relationship(
                        type=RelationType.IMPLEMENTS,
                        source_document_id=document.id,
                        source_node_id=node.id,
                        source_name=node.name,
                        source_path=document.file_path,
                        target_name=include,
                        location=node.location,
                    )
                    relationships.append(rel)

        return relationships

    def prepare_for_embedding(
        self,
        chunk: DocumentChunk,
        document: ParsedDocument,
    ) -> str:
        """Prepare Ruby chunk for embedding."""
        parts = ["Language: Ruby"]
        node = document.get_node(chunk.node_ids[0]) if chunk.node_ids else None

        if node:
            if node.type == NodeType.CLASS:
                parts.append(f"Ruby Class: {node.name}")
            elif node.type == NodeType.MODULE:
                parts.append(f"Ruby Module: {node.name}")
            elif node.type == NodeType.FUNCTION:
                parts.append(f"Ruby Method: {node.name}")

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
            keywords.extend(node.attributes.get("includes", []))

        return keywords

    def _create_summary(self, node: DocumentNode) -> str:
        """Create summary for node."""
        if node.type == NodeType.CLASS:
            return f"Ruby class {node.name}"
        elif node.type == NodeType.MODULE:
            return f"Ruby module {node.name}"
        elif node.type == NodeType.FUNCTION:
            if node.attributes.get("is_class_method"):
                return f"Ruby class method {node.name}"
            return f"Ruby method {node.name}"
        return ""
