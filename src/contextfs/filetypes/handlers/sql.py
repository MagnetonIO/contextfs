"""
SQL file handler with schema extraction.

Extracts:
- Table definitions (CREATE TABLE)
- Views, indexes, functions
- Foreign key relationships
- Queries (SELECT, INSERT, UPDATE, DELETE)
- Comments and documentation
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


class SQLHandler(FileTypeHandler):
    """Handler for SQL files with schema extraction."""

    name: str = "sql"
    extensions: list[str] = [".sql", ".ddl", ".dml"]
    mime_types: list[str] = ["application/sql", "text/x-sql"]
    chunk_strategy: ChunkStrategy = ChunkStrategy.AST_BOUNDARY

    # SQL patterns
    CREATE_TABLE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"CREATE\s+(?:TEMP(?:ORARY)?\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([`\"\[]?\w+[`\"\]]?(?:\.[`\"\[]?\w+[`\"\]]?)?)\s*\((.*?)\)\s*;?",
        re.IGNORECASE | re.DOTALL,
    )
    CREATE_VIEW_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"CREATE\s+(?:OR\s+REPLACE\s+)?(?:TEMP(?:ORARY)?\s+)?VIEW\s+([`\"\[]?\w+[`\"\]]?(?:\.[`\"\[]?\w+[`\"\]]?)?)\s+AS\s+(.*?);",
        re.IGNORECASE | re.DOTALL,
    )
    CREATE_INDEX_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"CREATE\s+(?:UNIQUE\s+)?INDEX\s+(?:IF\s+NOT\s+EXISTS\s+)?([`\"\[]?\w+[`\"\]]?)\s+ON\s+([`\"\[]?\w+[`\"\]]?)\s*\((.*?)\)\s*;?",
        re.IGNORECASE | re.DOTALL,
    )
    CREATE_FUNCTION_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"CREATE\s+(?:OR\s+REPLACE\s+)?FUNCTION\s+([`\"\[]?\w+[`\"\]]?)\s*\((.*?)\).*?(?:RETURNS|LANGUAGE)",
        re.IGNORECASE | re.DOTALL,
    )
    FOREIGN_KEY_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"FOREIGN\s+KEY\s*\(([^)]+)\)\s*REFERENCES\s+([`\"\[]?\w+[`\"\]]?)\s*\(([^)]+)\)",
        re.IGNORECASE,
    )
    COLUMN_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"([`\"\[]?\w+[`\"\]]?)\s+([\w\(\),\s]+?)(?:(?:NOT\s+NULL|NULL|PRIMARY\s+KEY|UNIQUE|DEFAULT|REFERENCES|CHECK|CONSTRAINT)\b.*?)?(?:,|$)",
        re.IGNORECASE,
    )

    def parse(self, content: str, file_path: str) -> ParsedDocument:
        """Parse SQL file into structured document."""
        lines = content.split("\n")

        root = DocumentNode(
            type=NodeType.SCHEMA,
            name=Path(file_path).stem,
            content="",
            location=SourceLocation(start_line=1, end_line=len(lines)),
        )

        symbols: dict[str, DocumentNode] = {}
        references: list[DocumentNode] = []

        # Split into statements
        statements = self._split_statements(content)

        for stmt_content, start_line in statements:
            node = self._parse_statement(stmt_content, start_line, root.id)
            if node:
                root.children.append(node)
                if node.name:
                    symbols[node.name] = node

                # Extract foreign key references
                self._extract_foreign_keys(node, references)

        # Create document
        doc = ParsedDocument(
            file_path=file_path,
            file_type="sql",
            raw_content=content,
            root=root,
            symbols=symbols,
            references=references,
            language=self._detect_dialect(content),
            line_count=len(lines),
            char_count=len(content),
            node_count=len(root.walk()),
            metadata={
                "dialect": self._detect_dialect(content),
                "table_count": len([n for n in root.children if n.type == NodeType.CREATE_TABLE]),
                "has_views": any(n.type == NodeType.CREATE_VIEW for n in root.children),
                "has_functions": any(n.type == NodeType.CREATE_FUNCTION for n in root.children),
            },
        )

        doc.chunks = self.chunk(doc)
        return doc

    def _split_statements(self, content: str) -> list[tuple[str, int]]:
        """Split SQL into individual statements."""
        statements = []
        current_stmt = []
        current_start = 1
        in_string = False
        string_char = None

        lines = content.split("\n")

        for i, line in enumerate(lines, 1):
            # Track string literals
            for j, char in enumerate(line):
                if not in_string and char in ("'", '"'):
                    in_string = True
                    string_char = char
                elif in_string and char == string_char:
                    # Check for escaped quote
                    if j > 0 and line[j - 1] != "\\":
                        in_string = False

            current_stmt.append(line)

            # Check for statement end (semicolon outside string)
            if not in_string and ";" in line:
                stmt_text = "\n".join(current_stmt).strip()
                if stmt_text:
                    statements.append((stmt_text, current_start))
                current_stmt = []
                current_start = i + 1

        # Last statement (might not end with semicolon)
        if current_stmt:
            stmt_text = "\n".join(current_stmt).strip()
            if stmt_text:
                statements.append((stmt_text, current_start))

        return statements

    def _parse_statement(
        self,
        content: str,
        start_line: int,
        parent_id: str,
    ) -> DocumentNode | None:
        """Parse a single SQL statement."""
        content_upper = content.upper().strip()
        end_line = start_line + content.count("\n")

        # CREATE TABLE
        match = self.CREATE_TABLE_PATTERN.search(content)
        if match:
            table_name = self._clean_identifier(match.group(1))
            columns_str = match.group(2)
            columns = self._parse_columns(columns_str)

            return DocumentNode(
                type=NodeType.CREATE_TABLE,
                name=table_name,
                content=content,
                location=SourceLocation(start_line=start_line, end_line=end_line),
                parent_id=parent_id,
                attributes={
                    "columns": columns,
                    "primary_keys": self._extract_primary_keys(columns_str),
                    "has_foreign_keys": "FOREIGN KEY" in content_upper,
                },
                children=[
                    DocumentNode(
                        type=NodeType.COLUMN,
                        name=col["name"],
                        content=f"{col['name']} {col['type']}",
                        attributes=col,
                    )
                    for col in columns
                ],
            )

        # CREATE VIEW
        match = self.CREATE_VIEW_PATTERN.search(content)
        if match:
            view_name = self._clean_identifier(match.group(1))
            query = match.group(2)

            return DocumentNode(
                type=NodeType.CREATE_VIEW,
                name=view_name,
                content=content,
                location=SourceLocation(start_line=start_line, end_line=end_line),
                parent_id=parent_id,
                attributes={
                    "query": query,
                    "referenced_tables": self._extract_table_references(query),
                },
            )

        # CREATE INDEX
        match = self.CREATE_INDEX_PATTERN.search(content)
        if match:
            index_name = self._clean_identifier(match.group(1))
            table_name = self._clean_identifier(match.group(2))
            columns = [c.strip() for c in match.group(3).split(",")]

            return DocumentNode(
                type=NodeType.CREATE_INDEX,
                name=index_name,
                content=content,
                location=SourceLocation(start_line=start_line, end_line=end_line),
                parent_id=parent_id,
                attributes={
                    "table": table_name,
                    "columns": columns,
                    "unique": "UNIQUE" in content_upper,
                },
            )

        # CREATE FUNCTION
        match = self.CREATE_FUNCTION_PATTERN.search(content)
        if match:
            func_name = self._clean_identifier(match.group(1))
            params = match.group(2)

            return DocumentNode(
                type=NodeType.CREATE_FUNCTION,
                name=func_name,
                content=content,
                location=SourceLocation(start_line=start_line, end_line=end_line),
                parent_id=parent_id,
                attributes={
                    "parameters": params,
                    "language": self._extract_function_language(content),
                },
            )

        # SELECT statement
        if content_upper.startswith("SELECT"):
            return DocumentNode(
                type=NodeType.SELECT,
                content=content,
                location=SourceLocation(start_line=start_line, end_line=end_line),
                parent_id=parent_id,
                attributes={
                    "tables": self._extract_table_references(content),
                },
            )

        # INSERT
        if content_upper.startswith("INSERT"):
            table_match = re.search(
                r"INSERT\s+INTO\s+([`\"\[]?\w+[`\"\]]?)", content, re.IGNORECASE
            )
            table_name = self._clean_identifier(table_match.group(1)) if table_match else None

            return DocumentNode(
                type=NodeType.INSERT,
                name=table_name,
                content=content,
                location=SourceLocation(start_line=start_line, end_line=end_line),
                parent_id=parent_id,
                attributes={"table": table_name},
            )

        # UPDATE
        if content_upper.startswith("UPDATE"):
            table_match = re.search(r"UPDATE\s+([`\"\[]?\w+[`\"\]]?)", content, re.IGNORECASE)
            table_name = self._clean_identifier(table_match.group(1)) if table_match else None

            return DocumentNode(
                type=NodeType.UPDATE,
                name=table_name,
                content=content,
                location=SourceLocation(start_line=start_line, end_line=end_line),
                parent_id=parent_id,
                attributes={"table": table_name},
            )

        # DELETE
        if content_upper.startswith("DELETE"):
            table_match = re.search(
                r"DELETE\s+FROM\s+([`\"\[]?\w+[`\"\]]?)", content, re.IGNORECASE
            )
            table_name = self._clean_identifier(table_match.group(1)) if table_match else None

            return DocumentNode(
                type=NodeType.DELETE,
                name=table_name,
                content=content,
                location=SourceLocation(start_line=start_line, end_line=end_line),
                parent_id=parent_id,
                attributes={"table": table_name},
            )

        # ALTER
        if content_upper.startswith("ALTER"):
            return DocumentNode(
                type=NodeType.ALTER,
                content=content,
                location=SourceLocation(start_line=start_line, end_line=end_line),
                parent_id=parent_id,
            )

        # DROP
        if content_upper.startswith("DROP"):
            return DocumentNode(
                type=NodeType.DROP,
                content=content,
                location=SourceLocation(start_line=start_line, end_line=end_line),
                parent_id=parent_id,
            )

        return None

    def _clean_identifier(self, name: str) -> str:
        """Remove quotes/brackets from identifier."""
        return name.strip('`"[]')

    def _parse_columns(self, columns_str: str) -> list[dict]:
        """Parse column definitions."""
        columns = []
        # Split by comma, but not within parentheses
        depth = 0
        current = ""

        for char in columns_str:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif char == "," and depth == 0:
                if current.strip():
                    col = self._parse_single_column(current.strip())
                    if col:
                        columns.append(col)
                current = ""
                continue
            current += char

        if current.strip():
            col = self._parse_single_column(current.strip())
            if col:
                columns.append(col)

        return columns

    def _parse_single_column(self, col_str: str) -> dict | None:
        """Parse a single column definition."""
        col_str = col_str.strip()

        # Skip constraints
        if col_str.upper().startswith(
            ("PRIMARY KEY", "FOREIGN KEY", "CONSTRAINT", "UNIQUE", "CHECK", "INDEX")
        ):
            return None

        # Extract column name and type
        parts = col_str.split(None, 2)
        if len(parts) < 2:
            return None

        name = self._clean_identifier(parts[0])
        data_type = parts[1].upper()

        # Check for constraints
        rest = " ".join(parts[2:]) if len(parts) > 2 else ""

        return {
            "name": name,
            "type": data_type,
            "nullable": "NOT NULL" not in rest.upper(),
            "primary_key": "PRIMARY KEY" in rest.upper(),
            "unique": "UNIQUE" in rest.upper(),
            "default": self._extract_default(rest),
        }

    def _extract_default(self, rest: str) -> str | None:
        """Extract default value."""
        match = re.search(r"DEFAULT\s+([^\s,]+)", rest, re.IGNORECASE)
        return match.group(1) if match else None

    def _extract_primary_keys(self, columns_str: str) -> list[str]:
        """Extract primary key columns."""
        pks = []

        # Inline PRIMARY KEY
        for match in re.finditer(
            r"([`\"\[]?\w+[`\"\]]?)\s+\w+.*?PRIMARY\s+KEY", columns_str, re.IGNORECASE
        ):
            pks.append(self._clean_identifier(match.group(1)))

        # Constraint PRIMARY KEY
        match = re.search(r"PRIMARY\s+KEY\s*\(([^)]+)\)", columns_str, re.IGNORECASE)
        if match:
            for col in match.group(1).split(","):
                pks.append(self._clean_identifier(col.strip()))

        return pks

    def _extract_foreign_keys(
        self,
        node: DocumentNode,
        references: list[DocumentNode],
    ) -> None:
        """Extract foreign key relationships."""
        if node.type != NodeType.CREATE_TABLE:
            return

        for match in self.FOREIGN_KEY_PATTERN.finditer(node.content):
            from_cols = [c.strip() for c in match.group(1).split(",")]
            ref_table = self._clean_identifier(match.group(2))
            ref_cols = [c.strip() for c in match.group(3).split(",")]

            fk_node = DocumentNode(
                type=NodeType.FOREIGN_KEY,
                name=f"{node.name}_fk_{ref_table}",
                content=match.group(0),
                target=ref_table,
                attributes={
                    "from_table": node.name,
                    "from_columns": from_cols,
                    "to_table": ref_table,
                    "to_columns": ref_cols,
                },
            )
            references.append(fk_node)

    def _extract_table_references(self, query: str) -> list[str]:
        """Extract tables referenced in a query."""
        tables = set()

        # FROM clause
        for match in re.finditer(r"\bFROM\s+([`\"\[]?\w+[`\"\]]?)", query, re.IGNORECASE):
            tables.add(self._clean_identifier(match.group(1)))

        # JOIN clauses
        for match in re.finditer(r"\bJOIN\s+([`\"\[]?\w+[`\"\]]?)", query, re.IGNORECASE):
            tables.add(self._clean_identifier(match.group(1)))

        return list(tables)

    def _extract_function_language(self, content: str) -> str | None:
        """Extract function language (PL/pgSQL, etc.)."""
        match = re.search(r"LANGUAGE\s+(\w+)", content, re.IGNORECASE)
        return match.group(1) if match else None

    def _detect_dialect(self, content: str) -> str:
        """Detect SQL dialect."""
        content_upper = content.upper()

        if "SERIAL" in content_upper or "RETURNING" in content_upper:
            return "postgresql"
        elif "AUTO_INCREMENT" in content_upper or "ENGINE=" in content_upper:
            return "mysql"
        elif "AUTOINCREMENT" in content_upper and "WITHOUT ROWID" not in content_upper:
            return "sqlite"
        elif "GO\n" in content or "SET NOCOUNT" in content_upper:
            return "tsql"
        else:
            return "ansi"

    def chunk(
        self,
        document: ParsedDocument,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[DocumentChunk]:
        """Chunk SQL document by statements."""
        chunks: list[DocumentChunk] = []

        # Chunk each table/view as separate
        for node in document.root.children:
            if node.type in (NodeType.CREATE_TABLE, NodeType.CREATE_VIEW, NodeType.CREATE_FUNCTION):
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

        # Update totals
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks

    def extract_relationships(self, document: ParsedDocument) -> list[Relationship]:
        """Extract relationships from SQL schema."""
        relationships: list[Relationship] = []

        for ref in document.references:
            if ref.type == NodeType.FOREIGN_KEY:
                attrs = ref.attributes

                rel = Relationship(
                    type=RelationType.FOREIGN_KEY_TO,
                    source_document_id=document.id,
                    source_node_id=ref.id,
                    source_name=attrs.get("from_table"),
                    source_path=document.file_path,
                    target_name=attrs.get("to_table"),
                    context=ref.content,
                    attributes={
                        "from_columns": attrs.get("from_columns"),
                        "to_columns": attrs.get("to_columns"),
                    },
                )
                relationships.append(rel)

        # View dependencies
        for node in document.root.children:
            if node.type == NodeType.CREATE_VIEW:
                for table in node.attributes.get("referenced_tables", []):
                    rel = Relationship(
                        type=RelationType.DEPENDS_ON,
                        source_document_id=document.id,
                        source_node_id=node.id,
                        source_name=node.name,
                        source_path=document.file_path,
                        target_name=table,
                        attributes={"dependency_type": "view_source"},
                    )
                    relationships.append(rel)

        return relationships

    def prepare_for_embedding(
        self,
        chunk: DocumentChunk,
        document: ParsedDocument,
    ) -> str:
        """Prepare SQL chunk for embedding."""
        parts = []
        node = document.get_node(chunk.node_ids[0]) if chunk.node_ids else None

        if node:
            # Add context
            if node.type == NodeType.CREATE_TABLE:
                parts.append(f"SQL Table: {node.name}")
                cols = node.attributes.get("columns", [])
                if cols:
                    col_names = [c["name"] for c in cols]
                    parts.append(f"Columns: {', '.join(col_names)}")
                pks = node.attributes.get("primary_keys", [])
                if pks:
                    parts.append(f"Primary Key: {', '.join(pks)}")

            elif node.type == NodeType.CREATE_VIEW:
                parts.append(f"SQL View: {node.name}")
                tables = node.attributes.get("referenced_tables", [])
                if tables:
                    parts.append(f"References: {', '.join(tables)}")

            elif node.type == NodeType.CREATE_FUNCTION:
                parts.append(f"SQL Function: {node.name}")

        # Add normalized SQL
        parts.append(f"SQL:\n{self._normalize_sql(chunk.content)}")

        if chunk.keywords:
            parts.append(f"Keywords: {', '.join(chunk.keywords)}")

        return "\n\n".join(parts)

    def _normalize_sql(self, content: str) -> str:
        """Normalize SQL for embedding."""
        # Remove comments
        content = re.sub(r"--.*$", "", content, flags=re.MULTILINE)
        content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)

        # Normalize whitespace
        content = re.sub(r"\s+", " ", content)

        return content.strip()

    def _extract_keywords(self, node: DocumentNode) -> list[str]:
        """Extract keywords from SQL node."""
        keywords = []

        if node.name:
            keywords.append(node.name)

        if node.type == NodeType.CREATE_TABLE:
            keywords.append("table")
            for col in node.attributes.get("columns", []):
                keywords.append(col["name"])

        elif node.type == NodeType.CREATE_VIEW:
            keywords.append("view")
            keywords.extend(node.attributes.get("referenced_tables", []))

        elif node.type == NodeType.CREATE_INDEX:
            keywords.append("index")
            keywords.append(node.attributes.get("table", ""))

        return [k for k in keywords if k]

    def _create_summary(self, node: DocumentNode) -> str:
        """Create summary for SQL node."""
        if node.type == NodeType.CREATE_TABLE:
            cols = node.attributes.get("columns", [])
            return f"Table {node.name} with {len(cols)} columns"
        elif node.type == NodeType.CREATE_VIEW:
            return f"View {node.name}"
        elif node.type == NodeType.CREATE_FUNCTION:
            return f"Function {node.name}"
        elif node.type == NodeType.CREATE_INDEX:
            return f"Index {node.name} on {node.attributes.get('table', 'unknown')}"
        return ""
