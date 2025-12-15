"""
Python file handler with AST parsing.

Extracts:
- Classes, functions, methods
- Imports and dependencies
- Docstrings and type hints
- Decorators and attributes
"""

import ast
import logging
from pathlib import Path

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


class PythonHandler(FileTypeHandler):
    """Handler for Python files with full AST parsing."""

    name: str = "python"
    extensions: list[str] = [".py", ".pyi", ".pyw"]
    mime_types: list[str] = ["text/x-python", "application/x-python"]
    chunk_strategy: ChunkStrategy = ChunkStrategy.AST_BOUNDARY

    def parse(self, content: str, file_path: str) -> ParsedDocument:
        """Parse Python file into AST."""
        # Parse AST
        try:
            tree = ast.parse(content, filename=file_path)
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
            # Return minimal document on parse error
            return self._create_minimal_document(content, file_path)

        # Create root node
        root = DocumentNode(
            type=NodeType.MODULE,
            name=Path(file_path).stem,
            content="",
            location=SourceLocation(start_line=1, end_line=content.count("\n") + 1),
        )

        # Extract symbols
        symbols: dict[str, DocumentNode] = {}
        references: list[DocumentNode] = []

        # Process AST
        for node in ast.iter_child_nodes(tree):
            child = self._process_node(node, content, root.id)
            if child:
                root.children.append(child)
                if child.name:
                    symbols[child.name] = child

                # Collect nested symbols
                for nested in child.walk():
                    if nested.name and nested.type in (
                        NodeType.CLASS,
                        NodeType.FUNCTION,
                        NodeType.METHOD,
                    ):
                        qualified = f"{child.name}.{nested.name}" if child.name else nested.name
                        symbols[qualified] = nested

                # Collect import references
                if child.type == NodeType.IMPORT:
                    references.append(child)

        # Create document
        doc = ParsedDocument(
            file_path=file_path,
            file_type="python",
            raw_content=content,
            root=root,
            symbols=symbols,
            references=references,
            language="python",
            line_count=content.count("\n") + 1,
            char_count=len(content),
            node_count=len(root.walk()),
            metadata={
                "has_main": "__main__" in content,
                "is_package": Path(file_path).name == "__init__.py",
            },
        )

        # Generate chunks
        doc.chunks = self.chunk(doc)

        return doc

    def _process_node(
        self,
        node: ast.AST,
        content: str,
        parent_id: str,
    ) -> DocumentNode | None:
        """Process an AST node."""
        if isinstance(node, ast.ClassDef):
            return self._process_class(node, content, parent_id)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            return self._process_function(node, content, parent_id)
        elif isinstance(node, ast.Import):
            return self._process_import(node, content, parent_id)
        elif isinstance(node, ast.ImportFrom):
            return self._process_import_from(node, content, parent_id)
        elif isinstance(node, ast.Assign):
            return self._process_assignment(node, content, parent_id)
        return None

    def _process_class(
        self,
        node: ast.ClassDef,
        content: str,
        parent_id: str,
    ) -> DocumentNode:
        """Process a class definition."""
        # Get source segment
        source = ast.get_source_segment(content, node) or ""

        # Extract docstring
        docstring = ast.get_docstring(node)

        # Extract bases
        bases = [ast.unparse(b) for b in node.bases]

        # Extract decorators
        decorators = [ast.unparse(d) for d in node.decorator_list]

        doc_node = DocumentNode(
            type=NodeType.CLASS,
            name=node.name,
            content=source,
            docstring=docstring,
            location=SourceLocation(
                start_line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                start_col=node.col_offset,
                end_col=node.end_col_offset,
            ),
            parent_id=parent_id,
            attributes={
                "bases": bases,
                "decorators": decorators,
            },
            annotations=decorators,
        )

        # Process methods
        for child in node.body:
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                method = self._process_function(child, content, doc_node.id, is_method=True)
                doc_node.children.append(method)
            elif isinstance(child, ast.Assign):
                attr = self._process_assignment(child, content, doc_node.id)
                if attr:
                    doc_node.children.append(attr)

        return doc_node

    def _process_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        content: str,
        parent_id: str,
        is_method: bool = False,
    ) -> DocumentNode:
        """Process a function/method definition."""
        source = ast.get_source_segment(content, node) or ""
        docstring = ast.get_docstring(node)

        # Build signature
        params = []
        for arg in node.args.args:
            param = {"name": arg.arg}
            if arg.annotation:
                param["type"] = ast.unparse(arg.annotation)
            params.append(param)

        # Return type
        return_type = ast.unparse(node.returns) if node.returns else None

        # Decorators
        decorators = [ast.unparse(d) for d in node.decorator_list]

        # Build signature string
        sig_parts = [p["name"] + (f": {p['type']}" if "type" in p else "") for p in params]
        signature = f"def {node.name}({', '.join(sig_parts)})"
        if return_type:
            signature += f" -> {return_type}"

        node_type = NodeType.METHOD if is_method else NodeType.FUNCTION
        is_async = isinstance(node, ast.AsyncFunctionDef)

        return DocumentNode(
            type=node_type,
            name=node.name,
            content=source,
            signature=signature,
            docstring=docstring,
            return_type=return_type,
            parameters=params,
            location=SourceLocation(
                start_line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                start_col=node.col_offset,
                end_col=node.end_col_offset,
            ),
            parent_id=parent_id,
            attributes={
                "is_async": is_async,
                "is_private": node.name.startswith("_"),
                "is_dunder": node.name.startswith("__") and node.name.endswith("__"),
            },
            annotations=decorators,
        )

    def _process_import(
        self,
        node: ast.Import,
        content: str,
        parent_id: str,
    ) -> DocumentNode:
        """Process an import statement."""
        names = [(alias.name, alias.asname) for alias in node.names]
        source = ast.get_source_segment(content, node) or ""

        return DocumentNode(
            type=NodeType.IMPORT,
            name=names[0][0] if len(names) == 1 else None,
            content=source,
            target=names[0][0],  # For relationship extraction
            location=SourceLocation(
                start_line=node.lineno,
                end_line=node.end_lineno or node.lineno,
            ),
            parent_id=parent_id,
            attributes={
                "imports": [{"module": n, "alias": a} for n, a in names],
                "is_from": False,
            },
        )

    def _process_import_from(
        self,
        node: ast.ImportFrom,
        content: str,
        parent_id: str,
    ) -> DocumentNode:
        """Process a from...import statement."""
        module = node.module or ""
        names = [(alias.name, alias.asname) for alias in node.names]
        source = ast.get_source_segment(content, node) or ""

        return DocumentNode(
            type=NodeType.IMPORT,
            name=module,
            content=source,
            target=module,
            location=SourceLocation(
                start_line=node.lineno,
                end_line=node.end_lineno or node.lineno,
            ),
            parent_id=parent_id,
            attributes={
                "module": module,
                "imports": [{"name": n, "alias": a} for n, a in names],
                "is_from": True,
                "level": node.level,  # Relative import level
            },
        )

    def _process_assignment(
        self,
        node: ast.Assign,
        content: str,
        parent_id: str,
    ) -> DocumentNode | None:
        """Process a variable assignment."""
        if not node.targets:
            return None

        target = node.targets[0]
        if not isinstance(target, ast.Name):
            return None

        source = ast.get_source_segment(content, node) or ""

        return DocumentNode(
            type=NodeType.VARIABLE,
            name=target.id,
            content=source,
            location=SourceLocation(
                start_line=node.lineno,
                end_line=node.end_lineno or node.lineno,
            ),
            parent_id=parent_id,
            attributes={
                "is_constant": target.id.isupper(),
            },
        )

    def _create_minimal_document(self, content: str, file_path: str) -> ParsedDocument:
        """Create minimal document for unparseable files."""
        root = DocumentNode(
            type=NodeType.MODULE,
            name=Path(file_path).stem,
            content=content,
        )

        return ParsedDocument(
            file_path=file_path,
            file_type="python",
            raw_content=content,
            root=root,
            line_count=content.count("\n") + 1,
            char_count=len(content),
            metadata={"parse_error": True},
        )

    def chunk(
        self,
        document: ParsedDocument,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[DocumentChunk]:
        """Chunk Python document by AST boundaries."""
        chunk_size = chunk_size or self.default_chunk_size
        chunks: list[DocumentChunk] = []

        # Get all top-level and nested definitions
        nodes_to_chunk = []
        for node in document.root.children:
            if node.type in (NodeType.CLASS, NodeType.FUNCTION):
                nodes_to_chunk.append(node)
                # Add methods as separate chunks if class is large
                if node.type == NodeType.CLASS:
                    for child in node.children:
                        if child.type == NodeType.METHOD:
                            nodes_to_chunk.append(child)

        # Create chunks
        for i, node in enumerate(nodes_to_chunk):
            breadcrumb = self._create_breadcrumb(node, document)
            keywords = self._extract_keywords(node)

            chunk = DocumentChunk(
                content=node.content,
                document_id=document.id,
                file_path=document.file_path,
                node_ids=[node.id],
                location=node.location,
                chunk_index=i,
                total_chunks=len(nodes_to_chunk),
                strategy=ChunkStrategy.AST_BOUNDARY,
                breadcrumb=breadcrumb,
                keywords=keywords,
                summary=self._create_summary(node),
            )

            # Prepare embedding text
            chunk.embedding_text = self.prepare_for_embedding(chunk, document)
            chunk.token_count = self._count_tokens(chunk.embedding_text)

            chunks.append(chunk)

        # If no chunks created, chunk entire file
        if not chunks:
            chunk = DocumentChunk(
                content=document.raw_content,
                document_id=document.id,
                file_path=document.file_path,
                node_ids=[document.root.id],
                chunk_index=0,
                total_chunks=1,
            )
            chunk.embedding_text = self.prepare_for_embedding(chunk, document)
            chunks.append(chunk)

        return chunks

    def extract_relationships(self, document: ParsedDocument) -> list[Relationship]:
        """Extract relationships from Python imports."""
        relationships: list[Relationship] = []

        for ref in document.references:
            if ref.type != NodeType.IMPORT:
                continue

            attrs = ref.attributes
            module = attrs.get("module") or ref.target

            if not module:
                continue

            rel = Relationship(
                type=RelationType.IMPORTS,
                source_document_id=document.id,
                source_node_id=ref.id,
                source_path=document.file_path,
                target_name=module,
                context=ref.content,
                location=ref.location,
                attributes={
                    "is_from": attrs.get("is_from", False),
                    "imports": attrs.get("imports", []),
                    "level": attrs.get("level", 0),
                },
            )
            relationships.append(rel)

        # Extract inheritance relationships
        for node in document.root.walk():
            if node.type == NodeType.CLASS:
                bases = node.attributes.get("bases", [])
                for base in bases:
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
        """Prepare Python chunk for embedding."""
        parts = []

        # Add file context
        parts.append(f"File: {Path(document.file_path).name}")

        # Add breadcrumb
        if chunk.breadcrumb:
            parts.append(f"Path: {' > '.join(chunk.breadcrumb)}")

        # Add summary if available
        if chunk.summary:
            parts.append(f"Summary: {chunk.summary}")

        # Get the primary node
        node = None
        if chunk.node_ids:
            node = document.get_node(chunk.node_ids[0])

        if node:
            # Add signature for functions/methods
            if node.signature:
                parts.append(f"Signature: {node.signature}")

            # Add docstring
            if node.docstring:
                parts.append(f"Docstring: {node.docstring}")

        # Add the code
        parts.append(f"Code:\n{chunk.content}")

        # Add keywords
        if chunk.keywords:
            parts.append(f"Keywords: {', '.join(chunk.keywords)}")

        return "\n\n".join(parts)

    def _extract_keywords(self, node: DocumentNode) -> list[str]:
        """Extract keywords from a node."""
        keywords = []

        if node.name:
            keywords.append(node.name)

        if node.type.value not in keywords:
            keywords.append(node.type.value)

        # Add decorator names
        for decorator in node.annotations:
            if "." in decorator:
                keywords.append(decorator.split(".")[-1])
            else:
                keywords.append(decorator.split("(")[0])

        # Add parameter names
        for param in node.parameters:
            keywords.append(param.get("name", ""))

        return [k for k in keywords if k]

    def _create_summary(self, node: DocumentNode) -> str:
        """Create a summary for a node."""
        parts = []

        if node.type == NodeType.CLASS:
            parts.append(f"Class {node.name}")
            if node.attributes.get("bases"):
                parts.append(f"inherits from {', '.join(node.attributes['bases'])}")
        elif node.type in (NodeType.FUNCTION, NodeType.METHOD):
            prefix = "Method" if node.type == NodeType.METHOD else "Function"
            is_async = node.attributes.get("is_async", False)
            if is_async:
                prefix = f"Async {prefix}"
            parts.append(f"{prefix} {node.name}")
            if node.return_type:
                parts.append(f"returns {node.return_type}")

        if node.docstring:
            # First line of docstring
            first_line = node.docstring.split("\n")[0].strip()
            if first_line:
                parts.append(f"- {first_line}")

        return " ".join(parts)
