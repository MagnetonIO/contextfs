"""
Base classes for file type handling.

Provides Pydantic models for:
- Document structure (AST nodes)
- Chunks for embedding
- Relationships between documents
- Cross-references
"""

import hashlib
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, computed_field

# ==================== Enums ====================


class NodeType(str, Enum):
    """Types of AST nodes across all file types."""

    # Generic
    ROOT = "root"
    SECTION = "section"
    PARAGRAPH = "paragraph"
    TEXT = "text"
    COMMENT = "comment"

    # Code
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"
    IMPORT = "import"
    DECORATOR = "decorator"
    DOCSTRING = "docstring"

    # LaTeX
    DOCUMENT = "document"
    CHAPTER = "chapter"
    SUBSECTION = "subsection"
    EQUATION = "equation"
    FIGURE = "figure"
    TABLE = "table"
    CITATION = "citation"
    LABEL = "label"
    REFERENCE = "reference"
    BIBLIOGRAPHY = "bibliography"
    ABSTRACT = "abstract"
    THEOREM = "theorem"
    PROOF = "proof"
    DEFINITION = "definition"
    ENVIRONMENT = "environment"
    MACRO = "macro"

    # SQL
    SCHEMA = "schema"
    CREATE_TABLE = "create_table"
    CREATE_VIEW = "create_view"
    CREATE_INDEX = "create_index"
    CREATE_FUNCTION = "create_function"
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    ALTER = "alter"
    DROP = "drop"
    COLUMN = "column"
    CONSTRAINT = "constraint"
    FOREIGN_KEY = "foreign_key"
    PRIMARY_KEY = "primary_key"

    # Markdown
    HEADING = "heading"
    CODE_BLOCK = "code_block"
    INLINE_CODE = "inline_code"
    LINK = "link"
    IMAGE = "image"
    LIST = "list"
    LIST_ITEM = "list_item"
    BLOCKQUOTE = "blockquote"
    FRONTMATTER = "frontmatter"

    # Config (JSON/YAML/TOML)
    OBJECT = "object"
    ARRAY = "array"
    KEY_VALUE = "key_value"

    # Other
    UNKNOWN = "unknown"


class RelationType(str, Enum):
    """Types of relationships between documents/nodes."""

    # Code relationships
    IMPORTS = "imports"
    IMPORTED_BY = "imported_by"
    CALLS = "calls"
    CALLED_BY = "called_by"
    INHERITS = "inherits"
    INHERITED_BY = "inherited_by"
    IMPLEMENTS = "implements"
    USES = "uses"
    USED_BY = "used_by"
    DEFINES = "defines"
    DEFINED_BY = "defined_by"

    # Document relationships
    REFERENCES = "references"
    REFERENCED_BY = "referenced_by"
    CITES = "cites"
    CITED_BY = "cited_by"
    INCLUDES = "includes"
    INCLUDED_BY = "included_by"
    LINKS_TO = "links_to"
    LINKED_FROM = "linked_from"

    # Schema relationships
    FOREIGN_KEY_TO = "foreign_key_to"
    FOREIGN_KEY_FROM = "foreign_key_from"
    DEPENDS_ON = "depends_on"
    DEPENDENCY_OF = "dependency_of"

    # Semantic relationships
    SIMILAR_TO = "similar_to"
    RELATED_TO = "related_to"
    SUPERSEDES = "supersedes"
    SUPERSEDED_BY = "superseded_by"


class ChunkStrategy(str, Enum):
    """Strategies for chunking documents."""

    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    AST_BOUNDARY = "ast_boundary"
    SLIDING_WINDOW = "sliding_window"
    RECURSIVE = "recursive"


# ==================== Document Models ====================


class SourceLocation(BaseModel):
    """Location in source file."""

    start_line: int
    end_line: int
    start_col: int | None = None
    end_col: int | None = None

    @computed_field
    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1


class DocumentNode(BaseModel):
    """
    A node in the document AST.

    Represents a structural element like a function, section, or table.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    type: NodeType
    name: str | None = None
    content: str = ""
    location: SourceLocation | None = None

    # AST structure
    children: list["DocumentNode"] = Field(default_factory=list)
    parent_id: str | None = None

    # Metadata
    attributes: dict[str, Any] = Field(default_factory=dict)
    annotations: list[str] = Field(default_factory=list)

    # For code nodes
    signature: str | None = None
    docstring: str | None = None
    return_type: str | None = None
    parameters: list[dict[str, Any]] = Field(default_factory=list)

    # For references
    target: str | None = None  # What this node references
    resolved_target_id: str | None = None  # Resolved node ID

    @computed_field
    @property
    def qualified_name(self) -> str:
        """Full qualified name including parent context."""
        if self.name:
            return self.name
        return f"{self.type.value}_{self.id[:8]}"

    def find_children(self, node_type: NodeType) -> list["DocumentNode"]:
        """Find all children of a specific type."""
        return [c for c in self.children if c.type == node_type]

    def walk(self) -> list["DocumentNode"]:
        """Walk all nodes in the tree."""
        nodes = [self]
        for child in self.children:
            nodes.extend(child.walk())
        return nodes

    model_config = {"arbitrary_types_allowed": True}


class DocumentChunk(BaseModel):
    """
    A chunk of document content for embedding.

    Includes context from the AST and metadata for retrieval.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    content: str

    # Source tracking
    document_id: str
    file_path: str
    node_ids: list[str] = Field(default_factory=list)  # AST nodes in this chunk
    location: SourceLocation | None = None

    # Chunking metadata
    chunk_index: int = 0
    total_chunks: int = 1
    strategy: ChunkStrategy = ChunkStrategy.AST_BOUNDARY

    # Context for better retrieval
    context_before: str = ""  # Text before this chunk
    context_after: str = ""  # Text after this chunk
    breadcrumb: list[str] = Field(default_factory=list)  # Path in AST

    # Semantic metadata
    summary: str | None = None
    keywords: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)

    # Embedding preparation
    embedding_text: str | None = None  # Preprocessed text for embedding
    token_count: int | None = None

    @computed_field
    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode()).hexdigest()[:16]

    def prepare_for_embedding(self) -> str:
        """Get text optimized for embedding."""
        if self.embedding_text:
            return self.embedding_text

        parts = []
        if self.breadcrumb:
            parts.append(" > ".join(self.breadcrumb))
        if self.summary:
            parts.append(self.summary)
        parts.append(self.content)
        if self.keywords:
            parts.append(f"Keywords: {', '.join(self.keywords)}")

        return "\n".join(parts)


class ParsedDocument(BaseModel):
    """
    A fully parsed document with AST and metadata.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    file_path: str
    file_type: str

    # Content
    raw_content: str
    encoding: str = "utf-8"

    # AST
    root: DocumentNode

    # Extracted elements
    symbols: dict[str, DocumentNode] = Field(default_factory=dict)  # name -> node
    references: list[DocumentNode] = Field(default_factory=list)  # Unresolved refs

    # Chunks
    chunks: list[DocumentChunk] = Field(default_factory=list)

    # Metadata
    title: str | None = None
    language: str | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    parsed_at: datetime = Field(default_factory=datetime.now)

    # Statistics
    line_count: int = 0
    char_count: int = 0
    node_count: int = 0

    # File-type specific metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    @computed_field
    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.raw_content.encode()).hexdigest()[:16]

    def get_node(self, node_id: str) -> DocumentNode | None:
        """Find a node by ID."""
        for node in self.root.walk():
            if node.id == node_id:
                return node
        return None

    def get_symbol(self, name: str) -> DocumentNode | None:
        """Find a symbol by name."""
        return self.symbols.get(name)

    def get_nodes_by_type(self, node_type: NodeType) -> list[DocumentNode]:
        """Get all nodes of a specific type."""
        return [n for n in self.root.walk() if n.type == node_type]


# ==================== Relationship Models ====================


class Relationship(BaseModel):
    """
    A relationship between two documents or nodes.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    type: RelationType

    # Source
    source_document_id: str
    source_node_id: str | None = None
    source_name: str | None = None
    source_path: str | None = None

    # Target
    target_document_id: str | None = None  # None if external
    target_node_id: str | None = None
    target_name: str
    target_path: str | None = None

    # Metadata
    confidence: float = 1.0
    context: str | None = None  # Surrounding code/text
    location: SourceLocation | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=datetime.now)

    @computed_field
    @property
    def is_resolved(self) -> bool:
        return self.target_document_id is not None


class CrossReference(BaseModel):
    """
    A cross-reference between documents.

    Used for linking related content across the codebase.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])

    # Source
    source_document_id: str
    source_chunk_id: str | None = None
    source_text: str

    # Target
    target_document_id: str
    target_chunk_id: str | None = None
    target_text: str

    # Reference type
    ref_type: str  # e.g., "citation", "import", "link", "see_also"

    # Metadata
    bidirectional: bool = False
    weight: float = 1.0
    context: str | None = None

    created_at: datetime = Field(default_factory=datetime.now)


# ==================== File Handler Base ====================


class FileTypeHandler(BaseModel, ABC):
    """
    Abstract base class for file type handlers.

    Each file type implements:
    - parse(): Extract AST and structure
    - chunk(): Split into meaningful chunks
    - extract_relationships(): Find relationships to other files
    - prepare_for_embedding(): Optimize text for vector search
    """

    # File type identification
    name: str
    extensions: list[str]
    mime_types: list[str] = Field(default_factory=list)

    # Chunking configuration
    default_chunk_size: int = 1000
    default_chunk_overlap: int = 200
    chunk_strategy: ChunkStrategy = ChunkStrategy.AST_BOUNDARY

    # Parsing options
    parse_comments: bool = True
    extract_docstrings: bool = True
    resolve_imports: bool = True

    model_config = {"arbitrary_types_allowed": True}

    @abstractmethod
    def parse(self, content: str, file_path: str) -> ParsedDocument:
        """
        Parse file content into a structured document.

        Args:
            content: Raw file content
            file_path: Path to the file

        Returns:
            ParsedDocument with AST and metadata
        """
        ...

    @abstractmethod
    def chunk(
        self,
        document: ParsedDocument,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[DocumentChunk]:
        """
        Split document into chunks for embedding.

        Args:
            document: Parsed document
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks

        Returns:
            List of DocumentChunk objects
        """
        ...

    @abstractmethod
    def extract_relationships(
        self,
        document: ParsedDocument,
    ) -> list[Relationship]:
        """
        Extract relationships to other documents.

        Args:
            document: Parsed document

        Returns:
            List of unresolved Relationship objects
        """
        ...

    @abstractmethod
    def prepare_for_embedding(
        self,
        chunk: DocumentChunk,
        document: ParsedDocument,
    ) -> str:
        """
        Prepare chunk text for embedding.

        Apply file-type specific preprocessing:
        - Normalize syntax
        - Expand macros
        - Add context

        Args:
            chunk: Document chunk
            document: Parent document

        Returns:
            Preprocessed text for embedding
        """
        ...

    def can_handle(self, file_path: str) -> bool:
        """Check if this handler can process the file."""
        path = Path(file_path)
        return path.suffix.lower() in self.extensions

    def get_language(self, file_path: str) -> str | None:
        """Get the language/format of the file."""
        return self.name

    def _count_tokens(self, text: str) -> int:
        """Estimate token count."""
        # Rough estimate: 1 token per 4 characters
        return len(text) // 4

    def _create_breadcrumb(
        self,
        node: DocumentNode,
        document: ParsedDocument,
    ) -> list[str]:
        """Create breadcrumb path for a node."""
        breadcrumb = []
        current = node

        while current:
            if current.name:
                breadcrumb.insert(0, current.name)
            elif current.type not in (NodeType.ROOT, NodeType.TEXT):
                breadcrumb.insert(0, current.type.value)

            # Find parent
            if current.parent_id:
                current = document.get_node(current.parent_id)
            else:
                break

        return breadcrumb
