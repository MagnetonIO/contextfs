"""
LaTeX file handler with AST parsing.

Extracts:
- Document structure (sections, chapters)
- Equations and math environments
- Figures, tables, algorithms
- Citations and references
- Labels and cross-references
- Custom macros and environments
"""

import logging
import re
from dataclasses import dataclass
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


@dataclass
class LaTeXToken:
    """A parsed LaTeX token."""

    type: str
    content: str
    line: int
    col: int
    args: list[str] = None
    options: str = None

    def __post_init__(self):
        if self.args is None:
            self.args = []


class LaTeXHandler(FileTypeHandler):
    """Handler for LaTeX files with structural parsing."""

    name: str = "latex"
    extensions: list[str] = [".tex", ".latex", ".ltx", ".sty", ".cls", ".bib"]
    mime_types: list[str] = ["text/x-tex", "application/x-tex", "application/x-latex"]
    chunk_strategy: ChunkStrategy = ChunkStrategy.AST_BOUNDARY

    # LaTeX patterns
    COMMAND_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"\\([a-zA-Z@]+)\*?(?:\[([^\]]*)\])?(?:\{([^}]*)\})*"
    )
    ENVIRONMENT_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"\\begin\{(\w+)\}(.*?)\\end\{\1\}", re.DOTALL
    )
    SECTION_COMMANDS: ClassVar[list[str]] = [
        "part",
        "chapter",
        "section",
        "subsection",
        "subsubsection",
        "paragraph",
    ]
    MATH_ENVIRONMENTS: ClassVar[list[str]] = [
        "equation",
        "align",
        "gather",
        "multline",
        "eqnarray",
        "math",
        "displaymath",
    ]
    THEOREM_ENVIRONMENTS: ClassVar[list[str]] = [
        "theorem",
        "lemma",
        "corollary",
        "proposition",
        "definition",
        "proof",
        "example",
        "remark",
    ]
    FLOAT_ENVIRONMENTS: ClassVar[list[str]] = ["figure", "table", "algorithm", "listing"]

    def parse(self, content: str, file_path: str) -> ParsedDocument:
        """Parse LaTeX file into structured document."""
        lines = content.split("\n")

        # Create root node
        root = DocumentNode(
            type=NodeType.DOCUMENT,
            name=Path(file_path).stem,
            content="",
            location=SourceLocation(start_line=1, end_line=len(lines)),
        )

        symbols: dict[str, DocumentNode] = {}
        references: list[DocumentNode] = []

        # Extract document class and packages
        doc_class = self._extract_document_class(content)
        packages = self._extract_packages(content)

        # Parse structure
        self._parse_structure(content, root, symbols, references)

        # Extract additional elements
        self._extract_equations(content, root, symbols)
        self._extract_citations(content, references)
        self._extract_labels(content, symbols)
        self._extract_refs(content, references)

        # Create document
        doc = ParsedDocument(
            file_path=file_path,
            file_type="latex",
            raw_content=content,
            root=root,
            symbols=symbols,
            references=references,
            language="latex",
            line_count=len(lines),
            char_count=len(content),
            node_count=len(root.walk()),
            title=self._extract_title(content),
            metadata={
                "document_class": doc_class,
                "packages": packages,
                "has_bibliography": "\\bibliography" in content or "\\printbibliography" in content,
                "has_abstract": "\\begin{abstract}" in content,
            },
        )

        # Generate chunks
        doc.chunks = self.chunk(doc)

        return doc

    def _extract_document_class(self, content: str) -> str | None:
        """Extract document class."""
        match = re.search(r"\\documentclass(?:\[.*?\])?\{(\w+)\}", content)
        return match.group(1) if match else None

    def _extract_packages(self, content: str) -> list[str]:
        """Extract used packages."""
        packages = []
        for match in re.finditer(r"\\usepackage(?:\[.*?\])?\{([^}]+)\}", content):
            # Handle comma-separated packages
            for pkg in match.group(1).split(","):
                packages.append(pkg.strip())
        return packages

    def _extract_title(self, content: str) -> str | None:
        """Extract document title."""
        match = re.search(r"\\title\{([^}]+)\}", content)
        return match.group(1) if match else None

    def _parse_structure(
        self,
        content: str,
        root: DocumentNode,
        symbols: dict[str, DocumentNode],
        references: list[DocumentNode],
    ) -> None:
        """Parse document structure (sections, etc.)."""
        lines = content.split("\n")
        current_section: DocumentNode | None = None
        section_stack: list[DocumentNode] = []
        current_content_lines: list[tuple[int, str]] = []

        for line_num, line in enumerate(lines, 1):
            # Check for section commands
            for i, cmd in enumerate(self.SECTION_COMMANDS):
                pattern = rf"\\{cmd}\*?\{{([^}}]+)\}}"
                match = re.search(pattern, line)
                if match:
                    # Save current content to previous section
                    if current_section and current_content_lines:
                        current_section.content = "\n".join(
                            text for _, text in current_content_lines
                        )
                        current_section.location.end_line = (
                            current_content_lines[-1][0] if current_content_lines else line_num - 1
                        )

                    # Create section node
                    node_type = self._section_to_node_type(cmd)
                    section_name = match.group(1)

                    section_node = DocumentNode(
                        type=node_type,
                        name=section_name,
                        content="",
                        location=SourceLocation(start_line=line_num, end_line=line_num),
                        attributes={
                            "level": i,
                            "command": cmd,
                            "starred": "*" in line,
                        },
                    )

                    # Manage section hierarchy
                    while section_stack and section_stack[-1].attributes.get("level", -1) >= i:
                        section_stack.pop()

                    if section_stack:
                        section_node.parent_id = section_stack[-1].id
                        section_stack[-1].children.append(section_node)
                    else:
                        section_node.parent_id = root.id
                        root.children.append(section_node)

                    section_stack.append(section_node)
                    current_section = section_node
                    current_content_lines = []
                    symbols[section_name] = section_node
                    break
            else:
                # Regular content line
                if current_section:
                    current_content_lines.append((line_num, line))

        # Finalize last section
        if current_section and current_content_lines:
            current_section.content = "\n".join(text for _, text in current_content_lines)
            current_section.location.end_line = current_content_lines[-1][0]

    def _section_to_node_type(self, cmd: str) -> NodeType:
        """Convert section command to node type."""
        mapping = {
            "part": NodeType.SECTION,
            "chapter": NodeType.CHAPTER,
            "section": NodeType.SECTION,
            "subsection": NodeType.SUBSECTION,
            "subsubsection": NodeType.SUBSECTION,
            "paragraph": NodeType.PARAGRAPH,
        }
        return mapping.get(cmd, NodeType.SECTION)

    def _extract_equations(
        self,
        content: str,
        root: DocumentNode,
        symbols: dict[str, DocumentNode],
    ) -> None:
        """Extract equations and math environments."""
        # Numbered equations with labels
        for env in self.MATH_ENVIRONMENTS:
            pattern = rf"\\begin\{{{env}\*?\}}(.*?)\\end\{{{env}\*?\}}"
            for match in re.finditer(pattern, content, re.DOTALL):
                eq_content = match.group(1)
                start_pos = match.start()
                line_num = content[:start_pos].count("\n") + 1

                # Extract label if present
                label_match = re.search(r"\\label\{([^}]+)\}", eq_content)
                label = label_match.group(1) if label_match else None

                eq_node = DocumentNode(
                    type=NodeType.EQUATION,
                    name=label,
                    content=match.group(0),
                    location=SourceLocation(
                        start_line=line_num,
                        end_line=line_num + eq_content.count("\n"),
                    ),
                    attributes={
                        "environment": env,
                        "label": label,
                        "starred": "*" in match.group(0)[:20],
                    },
                    parent_id=root.id,
                )

                root.children.append(eq_node)
                if label:
                    symbols[label] = eq_node

        # Inline equations (display math)
        for match in re.finditer(r"\$\$(.+?)\$\$", content, re.DOTALL):
            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1

            eq_node = DocumentNode(
                type=NodeType.EQUATION,
                content=match.group(0),
                location=SourceLocation(start_line=line_num, end_line=line_num),
                attributes={"environment": "displaymath", "inline": False},
                parent_id=root.id,
            )
            root.children.append(eq_node)

    def _extract_citations(
        self,
        content: str,
        references: list[DocumentNode],
    ) -> None:
        """Extract citations."""
        cite_commands = [
            "cite",
            "citep",
            "citet",
            "citeauthor",
            "citeyear",
            "parencite",
            "textcite",
        ]
        pattern = rf"\\({'|'.join(cite_commands)})\{{([^}}]+)\}}"

        for match in re.finditer(pattern, content):
            cmd = match.group(1)
            keys = match.group(2).split(",")
            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1

            for key in keys:
                key = key.strip()
                cite_node = DocumentNode(
                    type=NodeType.CITATION,
                    name=key,
                    content=match.group(0),
                    target=key,  # Citation key
                    location=SourceLocation(start_line=line_num, end_line=line_num),
                    attributes={"command": cmd},
                )
                references.append(cite_node)

    def _extract_labels(
        self,
        content: str,
        symbols: dict[str, DocumentNode],
    ) -> None:
        """Extract labels for cross-referencing."""
        for match in re.finditer(r"\\label\{([^}]+)\}", content):
            label = match.group(1)
            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1

            if label not in symbols:
                label_node = DocumentNode(
                    type=NodeType.LABEL,
                    name=label,
                    content=match.group(0),
                    location=SourceLocation(start_line=line_num, end_line=line_num),
                )
                symbols[label] = label_node

    def _extract_refs(
        self,
        content: str,
        references: list[DocumentNode],
    ) -> None:
        """Extract references to labels."""
        ref_commands = ["ref", "eqref", "pageref", "autoref", "cref", "Cref"]
        pattern = rf"\\({'|'.join(ref_commands)})\{{([^}}]+)\}}"

        for match in re.finditer(pattern, content):
            cmd = match.group(1)
            label = match.group(2)
            start_pos = match.start()
            line_num = content[:start_pos].count("\n") + 1

            ref_node = DocumentNode(
                type=NodeType.REFERENCE,
                name=label,
                content=match.group(0),
                target=label,
                location=SourceLocation(start_line=line_num, end_line=line_num),
                attributes={"command": cmd},
            )
            references.append(ref_node)

    def chunk(
        self,
        document: ParsedDocument,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[DocumentChunk]:
        """Chunk LaTeX document by sections."""
        chunks: list[DocumentChunk] = []
        chunk_size = chunk_size or self.default_chunk_size

        # Get sections and equations
        sections = (
            document.get_nodes_by_type(NodeType.SECTION)
            + document.get_nodes_by_type(NodeType.CHAPTER)
            + document.get_nodes_by_type(NodeType.SUBSECTION)
        )
        equations = document.get_nodes_by_type(NodeType.EQUATION)

        # Chunk sections
        for i, section in enumerate(sections):
            if not section.content.strip():
                continue

            breadcrumb = self._create_breadcrumb(section, document)
            keywords = self._extract_keywords(section)

            # Check if section content is too large
            if self._count_tokens(section.content) > chunk_size:
                # Split section into smaller chunks
                sub_chunks = self._split_section(section, chunk_size, document)
                chunks.extend(sub_chunks)
            else:
                chunk = DocumentChunk(
                    content=section.content,
                    document_id=document.id,
                    file_path=document.file_path,
                    node_ids=[section.id],
                    location=section.location,
                    chunk_index=len(chunks),
                    total_chunks=0,  # Updated later
                    strategy=ChunkStrategy.AST_BOUNDARY,
                    breadcrumb=breadcrumb,
                    keywords=keywords,
                    summary=f"Section: {section.name}",
                )
                chunk.embedding_text = self.prepare_for_embedding(chunk, document)
                chunks.append(chunk)

        # Chunk standalone equations
        for eq in equations:
            if eq.attributes.get("label"):
                chunk = DocumentChunk(
                    content=eq.content,
                    document_id=document.id,
                    file_path=document.file_path,
                    node_ids=[eq.id],
                    location=eq.location,
                    chunk_index=len(chunks),
                    strategy=ChunkStrategy.AST_BOUNDARY,
                    keywords=["equation", eq.name] if eq.name else ["equation"],
                    summary=f"Equation: {eq.name}" if eq.name else "Equation",
                )
                chunk.embedding_text = self.prepare_for_embedding(chunk, document)
                chunks.append(chunk)

        # Update total chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks

    def _split_section(
        self,
        section: DocumentNode,
        chunk_size: int,
        document: ParsedDocument,
    ) -> list[DocumentChunk]:
        """Split a large section into smaller chunks."""
        chunks = []
        paragraphs = section.content.split("\n\n")
        current_content = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self._count_tokens(para)

            if current_tokens + para_tokens > chunk_size and current_content:
                chunk = DocumentChunk(
                    content="\n\n".join(current_content),
                    document_id=document.id,
                    file_path=document.file_path,
                    node_ids=[section.id],
                    chunk_index=len(chunks),
                    strategy=ChunkStrategy.SEMANTIC,
                    breadcrumb=self._create_breadcrumb(section, document),
                    summary=f"Section: {section.name} (part {len(chunks) + 1})",
                )
                chunk.embedding_text = self.prepare_for_embedding(chunk, document)
                chunks.append(chunk)
                current_content = [para]
                current_tokens = para_tokens
            else:
                current_content.append(para)
                current_tokens += para_tokens

        # Last chunk
        if current_content:
            chunk = DocumentChunk(
                content="\n\n".join(current_content),
                document_id=document.id,
                file_path=document.file_path,
                node_ids=[section.id],
                chunk_index=len(chunks),
                strategy=ChunkStrategy.SEMANTIC,
                breadcrumb=self._create_breadcrumb(section, document),
                summary=f"Section: {section.name} (part {len(chunks) + 1})",
            )
            chunk.embedding_text = self.prepare_for_embedding(chunk, document)
            chunks.append(chunk)

        return chunks

    def extract_relationships(self, document: ParsedDocument) -> list[Relationship]:
        """Extract relationships from LaTeX references."""
        relationships: list[Relationship] = []

        for ref in document.references:
            if ref.type == NodeType.CITATION:
                rel = Relationship(
                    type=RelationType.CITES,
                    source_document_id=document.id,
                    source_node_id=ref.id,
                    source_path=document.file_path,
                    target_name=ref.target,
                    context=ref.content,
                    location=ref.location,
                    attributes={"citation_key": ref.target},
                )
                relationships.append(rel)

            elif ref.type == NodeType.REFERENCE:
                rel = Relationship(
                    type=RelationType.REFERENCES,
                    source_document_id=document.id,
                    source_node_id=ref.id,
                    source_path=document.file_path,
                    target_name=ref.target,
                    context=ref.content,
                    location=ref.location,
                    attributes={"label": ref.target},
                )
                relationships.append(rel)

        # Check for \input and \include
        for match in re.finditer(r"\\(input|include)\{([^}]+)\}", document.raw_content):
            cmd = match.group(1)
            file_ref = match.group(2)
            start_pos = match.start()
            line_num = document.raw_content[:start_pos].count("\n") + 1

            rel = Relationship(
                type=RelationType.INCLUDES,
                source_document_id=document.id,
                source_path=document.file_path,
                target_name=file_ref,
                location=SourceLocation(start_line=line_num, end_line=line_num),
                attributes={"command": cmd},
            )
            relationships.append(rel)

        return relationships

    def prepare_for_embedding(
        self,
        chunk: DocumentChunk,
        document: ParsedDocument,
    ) -> str:
        """Prepare LaTeX chunk for embedding."""
        parts = []

        # Add document context
        if document.title:
            parts.append(f"Document: {document.title}")

        # Add breadcrumb
        if chunk.breadcrumb:
            parts.append(f"Section: {' > '.join(chunk.breadcrumb)}")

        # Add summary
        if chunk.summary:
            parts.append(chunk.summary)

        # Process content - expand common macros, normalize math
        processed_content = self._normalize_latex(chunk.content)
        parts.append(processed_content)

        # Add keywords
        if chunk.keywords:
            parts.append(f"Keywords: {', '.join(chunk.keywords)}")

        return "\n\n".join(parts)

    def _normalize_latex(self, content: str) -> str:
        """Normalize LaTeX content for embedding."""
        # Remove comments
        content = re.sub(r"(?<!\\)%.*$", "", content, flags=re.MULTILINE)

        # Expand common macros
        content = content.replace("\\textbf{", "**").replace("}", "**", 1)
        content = content.replace("\\textit{", "*").replace("}", "*", 1)
        content = content.replace("\\emph{", "*").replace("}", "*", 1)

        # Normalize whitespace
        content = re.sub(r"\s+", " ", content)

        # Keep math environments but mark them
        content = re.sub(r"\$([^$]+)\$", r"[math: \1]", content)

        return content.strip()

    def _extract_keywords(self, node: DocumentNode) -> list[str]:
        """Extract keywords from LaTeX node."""
        keywords = []

        if node.name:
            keywords.append(node.name)

        # Extract emphasized terms
        for match in re.finditer(r"\\(?:textbf|textit|emph)\{([^}]+)\}", node.content):
            keywords.append(match.group(1))

        # Extract defined terms
        for match in re.finditer(r"\\(?:newcommand|def)\\(\w+)", node.content):
            keywords.append(match.group(1))

        return keywords[:20]  # Limit
