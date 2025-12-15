"""
File Type Registry

Provides:
- Handler registration and lookup
- File type detection
- Factory for creating handlers
"""

import logging
import mimetypes
from pathlib import Path
from typing import Optional

from contextfs.filetypes.base import FileTypeHandler, ParsedDocument

logger = logging.getLogger(__name__)


class FileTypeRegistry:
    """
    Registry for file type handlers.

    Supports:
    - Extension-based lookup
    - MIME type lookup
    - Custom handler registration
    - Handler priority/override
    """

    _instance: Optional["FileTypeRegistry"] = None
    _handlers: dict[str, type[FileTypeHandler]] = {}
    _extension_map: dict[str, str] = {}  # extension -> handler name
    _mime_map: dict[str, str] = {}  # mime type -> handler name

    def __new__(cls) -> "FileTypeRegistry":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._register_builtin_handlers()

    def _register_builtin_handlers(self) -> None:
        """Register all built-in handlers."""
        # Import handlers here to avoid circular imports
        from contextfs.filetypes.handlers.config import (
            JSONHandler,
            TOMLHandler,
            YAMLHandler,
        )
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

        # Handler registration order matters for extension conflicts
        # TypeScript before JavaScript (more specific)
        handlers = [
            # Top 10 programming languages
            PythonHandler,
            TypeScriptHandler,  # Before JavaScript (overrides .ts, .tsx)
            JavaScriptHandler,
            JavaHandler,
            CppHandler,
            CSharpHandler,
            GoHandler,
            RustHandler,
            PHPHandler,
            RubyHandler,
            SwiftHandler,
            ShellHandler,
            # Specialized formats
            SQLHandler,
            LaTeXHandler,
            MarkdownHandler,
            # Configuration
            JSONHandler,
            YAMLHandler,
            TOMLHandler,
            # Fallback (must be last)
            GenericTextHandler,
        ]

        for handler_cls in handlers:
            self.register(handler_cls)

    def register(
        self,
        handler_cls: type[FileTypeHandler],
        override: bool = False,
    ) -> None:
        """
        Register a file type handler.

        Args:
            handler_cls: Handler class to register
            override: Whether to override existing handlers
        """
        # Create instance to get metadata
        handler = handler_cls()
        name = handler.name

        if name in self._handlers and not override:
            logger.debug(f"Handler {name} already registered, skipping")
            return

        self._handlers[name] = handler_cls

        # Map extensions
        for ext in handler.extensions:
            ext_lower = ext.lower()
            if ext_lower not in self._extension_map or override:
                self._extension_map[ext_lower] = name

        # Map MIME types
        for mime in handler.mime_types:
            if mime not in self._mime_map or override:
                self._mime_map[mime] = name

        logger.debug(f"Registered handler: {name} for {handler.extensions}")

    def get_handler(
        self,
        file_path: str | None = None,
        extension: str | None = None,
        mime_type: str | None = None,
        handler_name: str | None = None,
    ) -> FileTypeHandler | None:
        """
        Get a handler for a file.

        Priority:
        1. Explicit handler name
        2. Extension
        3. MIME type
        4. Fallback to generic

        Args:
            file_path: Path to file
            extension: File extension
            mime_type: MIME type
            handler_name: Explicit handler name

        Returns:
            FileTypeHandler instance or None
        """
        name = None

        # Explicit handler
        if handler_name and handler_name in self._handlers:
            name = handler_name

        # Extension lookup
        elif file_path:
            ext = Path(file_path).suffix.lower()
            name = self._extension_map.get(ext)

        elif extension:
            ext = extension.lower()
            if not ext.startswith("."):
                ext = f".{ext}"
            name = self._extension_map.get(ext)

        # MIME type lookup
        elif mime_type:
            name = self._mime_map.get(mime_type)

        # Try to detect MIME from file path
        if not name and file_path:
            detected_mime, _ = mimetypes.guess_type(file_path)
            if detected_mime:
                name = self._mime_map.get(detected_mime)

        # Fallback to generic
        if not name:
            name = "generic"

        if name and name in self._handlers:
            return self._handlers[name]()

        return None

    def parse_file(
        self,
        file_path: str,
        content: str | None = None,
    ) -> ParsedDocument | None:
        """
        Parse a file using the appropriate handler.

        Args:
            file_path: Path to file
            content: File content (read if not provided)

        Returns:
            ParsedDocument or None
        """
        handler = self.get_handler(file_path=file_path)
        if not handler:
            logger.warning(f"No handler found for {file_path}")
            return None

        if content is None:
            try:
                content = Path(file_path).read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                logger.error(f"Failed to read {file_path}: {e}")
                return None

        try:
            return handler.parse(content, file_path)
        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            return None

    def list_handlers(self) -> list[str]:
        """List all registered handler names."""
        return list(self._handlers.keys())

    def list_extensions(self) -> dict[str, str]:
        """List all extension -> handler mappings."""
        return dict(self._extension_map)

    def supports_extension(self, extension: str) -> bool:
        """Check if an extension is supported."""
        ext = extension.lower()
        if not ext.startswith("."):
            ext = f".{ext}"
        return ext in self._extension_map


# Module-level functions
_registry: FileTypeRegistry | None = None


def get_registry() -> FileTypeRegistry:
    """Get the global registry instance."""
    global _registry
    if _registry is None:
        _registry = FileTypeRegistry()
    return _registry


def get_handler(
    file_path: str | None = None,
    extension: str | None = None,
    mime_type: str | None = None,
    handler_name: str | None = None,
) -> FileTypeHandler | None:
    """Get a handler for a file."""
    return get_registry().get_handler(
        file_path=file_path,
        extension=extension,
        mime_type=mime_type,
        handler_name=handler_name,
    )


def parse_file(file_path: str, content: str | None = None) -> ParsedDocument | None:
    """Parse a file using the appropriate handler."""
    return get_registry().parse_file(file_path, content)


def register_handler(handler_cls: type[FileTypeHandler], override: bool = False) -> None:
    """Register a custom handler."""
    get_registry().register(handler_cls, override)
