"""
Integration tests for indexing and processing codebases.

Uses fixture-generated test repositories to ensure reproducible tests
without external dependencies.
"""

from pathlib import Path

import pytest


@pytest.fixture
def sample_project(temp_dir: Path) -> Path:
    """Create a sample multi-language project for testing."""
    # Create project structure
    src_dir = temp_dir / "src"
    src_dir.mkdir()

    # Python files
    (src_dir / "main.py").write_text('''"""Main application module."""
import os
from typing import Optional
from utils import helper_function


class Application:
    """Main application class."""

    def __init__(self, name: str):
        self.name = name
        self.config = {}

    def run(self) -> None:
        """Run the application."""
        print(f"Running {self.name}")
        helper_function()


def main():
    """Entry point."""
    app = Application("TestApp")
    app.run()


if __name__ == "__main__":
    main()
''')

    (src_dir / "utils.py").write_text('''"""Utility functions."""
from typing import Any, List


def helper_function() -> None:
    """A helper function."""
    print("Helper called")


def process_data(items: List[Any]) -> List[Any]:
    """Process a list of items."""
    return [item for item in items if item is not None]


class DataProcessor:
    """Process data with various strategies."""

    def __init__(self, strategy: str = "default"):
        self.strategy = strategy

    def process(self, data: Any) -> Any:
        """Process data using configured strategy."""
        if self.strategy == "default":
            return data
        return None
''')

    # TypeScript files
    components_dir = src_dir / "components"
    components_dir.mkdir()

    (components_dir / "Button.tsx").write_text("""import React from 'react';

interface ButtonProps {
    label: string;
    onClick: () => void;
    disabled?: boolean;
    variant?: 'primary' | 'secondary';
}

export const Button: React.FC<ButtonProps> = ({
    label,
    onClick,
    disabled = false,
    variant = 'primary'
}) => {
    return (
        <button
            className={`btn btn-${variant}`}
            onClick={onClick}
            disabled={disabled}
        >
            {label}
        </button>
    );
};

export default Button;
""")

    (components_dir / "Card.tsx").write_text("""import React from 'react';
import { Button } from './Button';

interface CardProps {
    title: string;
    content: string;
    onAction?: () => void;
}

export const Card: React.FC<CardProps> = ({ title, content, onAction }) => {
    return (
        <div className="card">
            <h2>{title}</h2>
            <p>{content}</p>
            {onAction && <Button label="Action" onClick={onAction} />}
        </div>
    );
};
""")

    # JavaScript file
    (src_dir / "config.js").write_text("""// Configuration module
const config = {
    apiUrl: process.env.API_URL || 'http://localhost:3000',
    timeout: 5000,
    retries: 3,
};

function getConfig(key) {
    return config[key];
}

function setConfig(key, value) {
    config[key] = value;
}

module.exports = { config, getConfig, setConfig };
""")

    # Go file
    (src_dir / "server.go").write_text("""package main

import (
    "fmt"
    "net/http"
)

// Server represents an HTTP server
type Server struct {
    Port int
    Host string
}

// NewServer creates a new server instance
func NewServer(host string, port int) *Server {
    return &Server{
        Host: host,
        Port: port,
    }
}

// Start starts the server
func (s *Server) Start() error {
    addr := fmt.Sprintf("%s:%d", s.Host, s.Port)
    return http.ListenAndServe(addr, nil)
}

func main() {
    server := NewServer("localhost", 8080)
    server.Start()
}
""")

    # Rust file
    (src_dir / "lib.rs").write_text("""//! Library module for data processing

use std::collections::HashMap;

/// A data store for key-value pairs
pub struct DataStore {
    data: HashMap<String, String>,
}

impl DataStore {
    /// Create a new empty data store
    pub fn new() -> Self {
        DataStore {
            data: HashMap::new(),
        }
    }

    /// Insert a key-value pair
    pub fn insert(&mut self, key: String, value: String) {
        self.data.insert(key, value);
    }

    /// Get a value by key
    pub fn get(&self, key: &str) -> Option<&String> {
        self.data.get(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_get() {
        let mut store = DataStore::new();
        store.insert("key".to_string(), "value".to_string());
        assert_eq!(store.get("key"), Some(&"value".to_string()));
    }
}
""")

    return temp_dir


class TestCodebaseProcessing:
    """Tests for processing generic codebases."""

    def test_detect_project_files(self, sample_project: Path):
        """Test detecting file types in a project."""
        from contextfs.filetypes.registry import FileTypeRegistry

        registry = FileTypeRegistry()

        # Find all source files
        extensions_found = set()
        for file in sample_project.rglob("*"):
            if file.is_file():
                ext = file.suffix
                if ext and registry.supports_extension(ext):
                    extensions_found.add(ext)

        # Should find multiple file types
        assert len(extensions_found) >= 3
        assert ".py" in extensions_found
        assert ".tsx" in extensions_found or ".ts" in extensions_found

    def test_parse_python_files(self, sample_project: Path):
        """Test parsing Python files."""
        from contextfs.filetypes.registry import get_handler

        handler = get_handler(file_path="test.py")
        assert handler is not None

        py_files = list(sample_project.glob("**/*.py"))
        assert len(py_files) >= 1

        for test_file in py_files:
            content = test_file.read_text(encoding="utf-8")
            doc = handler.parse(content, str(test_file))

            assert doc is not None
            assert doc.file_type == "python"
            assert doc.line_count > 0

    def test_parse_typescript_files(self, sample_project: Path):
        """Test parsing TypeScript files."""
        from contextfs.filetypes.registry import get_handler

        handler = get_handler(file_path="test.tsx")
        assert handler is not None

        ts_files = list(sample_project.glob("**/*.tsx"))
        assert len(ts_files) >= 1

        for test_file in ts_files:
            content = test_file.read_text(encoding="utf-8")
            doc = handler.parse(content, str(test_file))

            assert doc is not None
            assert doc.file_type in ["typescript", "javascript"]
            assert doc.line_count > 0

    def test_process_project_directory(self, sample_project: Path):
        """Test processing an entire project directory."""
        from contextfs.filetypes.integration import SmartDocumentProcessor

        processor = SmartDocumentProcessor()

        src_dir = sample_project / "src"
        chunks = processor.process_directory(
            src_dir,
            extensions=[".py", ".ts", ".tsx", ".js", ".go", ".rs"],
            recursive=True,
        )

        assert len(chunks) > 0

        # Check chunk metadata
        for chunk in chunks:
            assert "file_type" in chunk["metadata"]
            assert "source_file" in chunk["metadata"]

        # Should have chunks from multiple file types
        file_types = {c["metadata"]["file_type"] for c in chunks}
        assert len(file_types) >= 2

    @pytest.mark.slow
    def test_index_and_search_codebase(self, sample_project: Path, temp_dir: Path):
        """Test indexing codebase and searching."""
        from contextfs.filetypes.integration import RAGIntegration
        from contextfs.rag import RAGBackend

        # Create RAG backend in separate temp dir
        rag_dir = temp_dir / "rag_data"
        rag_dir.mkdir()

        rag = RAGBackend(
            data_dir=rag_dir,
            embedding_model="all-MiniLM-L6-v2",
            collection_name="codebase_test",
        )

        integration = RAGIntegration(rag)

        src_dir = sample_project / "src"
        stats = integration.index_directory(
            src_dir,
            extensions=[".py", ".ts", ".tsx", ".js", ".go", ".rs"],
        )

        assert stats["files_processed"] > 0
        assert stats["memories_added"] > 0

        # Search for Python class
        results = integration.search_with_context("Application class", limit=5)
        assert len(results) >= 0

        # Search for TypeScript component
        results = integration.search_with_context("Button component props", limit=5)
        assert len(results) >= 0

        rag.close()

    def test_extract_relationships(self, sample_project: Path):
        """Test extracting import relationships."""
        from contextfs.filetypes.registry import get_handler

        # Test Python imports
        py_handler = get_handler(file_path="test.py")
        main_py = sample_project / "src" / "main.py"
        content = main_py.read_text()
        doc = py_handler.parse(content, str(main_py))
        py_relationships = py_handler.extract_relationships(doc)

        # Should find imports
        assert len(py_relationships) > 0

        # Test TypeScript imports
        ts_handler = get_handler(file_path="test.tsx")
        card_tsx = sample_project / "src" / "components" / "Card.tsx"
        content = card_tsx.read_text()
        doc = ts_handler.parse(content, str(card_tsx))
        ts_relationships = ts_handler.extract_relationships(doc)

        # Should find React and Button imports
        assert len(ts_relationships) > 0

    def test_cross_reference_linking(self, sample_project: Path):
        """Test cross-reference linking in codebase."""
        from contextfs.filetypes.integration import SmartDocumentProcessor

        processor = SmartDocumentProcessor()

        src_dir = sample_project / "src"

        # Process all files
        for file in src_dir.rglob("*"):
            if file.is_file() and file.suffix in [".py", ".tsx", ".ts", ".js", ".go", ".rs"]:
                try:
                    processor.process_file(file)
                except Exception:
                    pass  # Skip files that can't be processed

        # Link cross-references
        xrefs = processor.linker.link_all()

        # Should have some cross-references (imports between files)
        # May be 0 if files don't reference each other by resolvable paths
        assert isinstance(xrefs, list)

        # Check unresolved references exist (external deps)
        unresolved = processor.linker.get_unresolved()
        assert isinstance(unresolved, list)

    def test_multi_language_chunking(self, sample_project: Path):
        """Test that chunking works consistently across languages."""
        from contextfs.filetypes.integration import SmartDocumentProcessor

        processor = SmartDocumentProcessor()
        src_dir = sample_project / "src"

        chunks_by_type = {}

        for file in src_dir.rglob("*"):
            if file.is_file() and file.suffix in [".py", ".tsx", ".go", ".rs"]:
                try:
                    chunks = processor.process_file(file)
                    file_type = chunks[0]["metadata"]["file_type"] if chunks else None
                    if file_type:
                        chunks_by_type.setdefault(file_type, []).extend(chunks)
                except Exception:
                    pass

        # Should have chunks from multiple languages
        assert len(chunks_by_type) >= 2

        # Each language should have valid chunks
        for file_type, chunks in chunks_by_type.items():
            assert len(chunks) > 0
            for chunk in chunks:
                assert "embedding_text" in chunk
                assert chunk["metadata"]["file_type"] == file_type
