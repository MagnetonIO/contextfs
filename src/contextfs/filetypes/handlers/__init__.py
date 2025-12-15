"""
File type handlers for various file formats.

Supported languages:
- Python (.py)
- JavaScript (.js, .jsx, .mjs, .cjs)
- TypeScript (.ts, .tsx, .mts, .cts)
- Java (.java)
- C/C++ (.c, .cpp, .cc, .h, .hpp)
- C# (.cs)
- Go (.go)
- Rust (.rs)
- PHP (.php)
- Ruby (.rb, .rake)
- Swift (.swift)
- Shell/Bash (.sh, .bash, .zsh)
- SQL (.sql)
- LaTeX (.tex)
- Markdown (.md)
- Config (JSON, YAML, TOML)

To add a new handler:
1. Create a new file in this directory (e.g., `kotlin.py`)
2. Inherit from `FileTypeHandler` in `contextfs.filetypes.base`
3. Implement: `parse()`, `chunk()`, `extract_relationships()`, `prepare_for_embedding()`
4. Add to this `__init__.py` and `DEFAULT_HANDLERS` in `registry.py`
"""

from contextfs.filetypes.handlers.config import JSONHandler, TOMLHandler, YAMLHandler
from contextfs.filetypes.handlers.cpp import CppHandler
from contextfs.filetypes.handlers.csharp import CSharpHandler
from contextfs.filetypes.handlers.generic import GenericTextHandler
from contextfs.filetypes.handlers.go import GoHandler
from contextfs.filetypes.handlers.java import JavaHandler
from contextfs.filetypes.handlers.javascript import JavaScriptHandler
from contextfs.filetypes.handlers.latex import LaTeXHandler
from contextfs.filetypes.handlers.markdown import MarkdownHandler
from contextfs.filetypes.handlers.php import PHPHandler
from contextfs.filetypes.handlers.python import PythonHandler
from contextfs.filetypes.handlers.ruby import RubyHandler
from contextfs.filetypes.handlers.rust import RustHandler
from contextfs.filetypes.handlers.shell import ShellHandler
from contextfs.filetypes.handlers.sql import SQLHandler
from contextfs.filetypes.handlers.swift import SwiftHandler
from contextfs.filetypes.handlers.typescript import TypeScriptHandler

__all__ = [
    # Top 10 languages
    "PythonHandler",
    "JavaScriptHandler",
    "TypeScriptHandler",
    "JavaHandler",
    "CppHandler",
    "CSharpHandler",
    "GoHandler",
    "RustHandler",
    "PHPHandler",
    "RubyHandler",
    "SwiftHandler",
    "ShellHandler",
    # Specialized
    "SQLHandler",
    "LaTeXHandler",
    "MarkdownHandler",
    # Config
    "JSONHandler",
    "YAMLHandler",
    "TOMLHandler",
    # Fallback
    "GenericTextHandler",
]
