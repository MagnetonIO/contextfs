"""
Schema registry for the ContextFS formal type system.

Provides a central registry for schema types that:
- Maps schema names to their Pydantic classes
- Enables runtime type resolution from strings/dicts
- Integrates with existing STRUCTURED_DATA_CLASSES

The registry bridges the gap between serialized type names
and their runtime Python class representations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from contextfs.types.base import BaseSchema

if TYPE_CHECKING:
    pass


class SchemaRegistry:
    """
    Central registry for schema types.

    Provides schema lookup and resolution services that integrate
    with the existing ContextFS type infrastructure.

    The registry uses a two-tier lookup:
    1. First checks the explicit registry (_schemas)
    2. Falls back to STRUCTURED_DATA_CLASSES from schemas.py

    Class Attributes:
        _schemas: Dictionary mapping schema names to classes.

    Example:
        >>> from contextfs.types import SchemaRegistry, BaseSchema
        >>>
        >>> class CustomSchema(BaseSchema):
        ...     _schema_name = "custom"
        ...     data: str
        ...
        >>> SchemaRegistry.register("custom", CustomSchema)
        >>> schema_cls = SchemaRegistry.get("custom")
        >>> schema_cls(data="hello")
        CustomSchema(data='hello')
    """

    _schemas: dict[str, type[BaseSchema]] = {}
    _initialized: bool = False

    @classmethod
    def _ensure_initialized(cls) -> None:
        """Ensure built-in schemas are registered."""
        if cls._initialized:
            return

        # Import here to avoid circular imports
        try:
            from contextfs.schemas import STRUCTURED_DATA_CLASSES

            # STRUCTURED_DATA_CLASSES maps type names to Pydantic classes
            # These are compatible with BaseSchema
            for name, schema_cls in STRUCTURED_DATA_CLASSES.items():
                if name not in cls._schemas:
                    cls._schemas[name] = schema_cls  # type: ignore
        except ImportError:
            pass

        cls._initialized = True

    @classmethod
    def register(cls, name: str, schema_type: type[BaseSchema]) -> None:
        """
        Register a schema type.

        Args:
            name: The schema name (used for lookup).
            schema_type: The schema class.

        Example:
            >>> SchemaRegistry.register("decision", DecisionSchema)
        """
        cls._schemas[name] = schema_type

    @classmethod
    def unregister(cls, name: str) -> bool:
        """
        Unregister a schema type.

        Args:
            name: The schema name to remove.

        Returns:
            True if schema was removed, False if not found.
        """
        if name in cls._schemas:
            del cls._schemas[name]
            return True
        return False

    @classmethod
    def get(cls, name: str) -> type[BaseSchema] | None:
        """
        Get a schema type by name.

        Performs two-tier lookup:
        1. Explicit registry
        2. STRUCTURED_DATA_CLASSES fallback

        Args:
            name: The schema name to look up.

        Returns:
            The schema class if found, None otherwise.

        Example:
            >>> DecisionSchema = SchemaRegistry.get("decision")
            >>> DecisionSchema is not None
            True
        """
        cls._ensure_initialized()

        # Check explicit registry first
        if name in cls._schemas:
            return cls._schemas[name]

        # Fallback to STRUCTURED_DATA_CLASSES
        try:
            from contextfs.schemas import STRUCTURED_DATA_CLASSES

            return STRUCTURED_DATA_CLASSES.get(name)  # type: ignore
        except ImportError:
            return None

    @classmethod
    def get_or_raise(cls, name: str) -> type[BaseSchema]:
        """
        Get a schema type by name, raising if not found.

        Args:
            name: The schema name to look up.

        Returns:
            The schema class.

        Raises:
            KeyError: If schema is not registered.
        """
        schema = cls.get(name)
        if schema is None:
            raise KeyError(f"Unknown schema type: {name}")
        return schema

    @classmethod
    def resolve(
        cls,
        data: dict[str, Any],
        type_hint: str | None = None,
    ) -> BaseSchema:
        """
        Resolve a dictionary to a typed schema instance.

        Uses the 'type' field as discriminator if type_hint not provided.

        Args:
            data: Dictionary with schema fields.
            type_hint: Optional explicit type name.

        Returns:
            Validated schema instance.

        Raises:
            ValueError: If schema type cannot be determined or is unknown.

        Example:
            >>> data = {"type": "decision", "decision": "Use PostgreSQL"}
            >>> schema = SchemaRegistry.resolve(data)
            >>> schema.decision
            'Use PostgreSQL'
        """
        # Determine schema name
        schema_name = type_hint or data.get("type")
        if not schema_name:
            raise ValueError(
                "Cannot resolve schema: no type_hint provided and no 'type' field in data"
            )

        # Get schema class
        schema_cls = cls.get(schema_name)
        if schema_cls is None:
            raise ValueError(f"Unknown schema type: {schema_name}")

        # Validate and return
        return schema_cls.model_validate(data)

    @classmethod
    def try_resolve(
        cls,
        data: dict[str, Any],
        type_hint: str | None = None,
    ) -> BaseSchema | dict[str, Any]:
        """
        Try to resolve data to schema, returning dict on failure.

        Useful for backward compatibility where you want typed
        access when possible but fallback to raw dict.

        Args:
            data: Dictionary with schema fields.
            type_hint: Optional explicit type name.

        Returns:
            Schema instance if resolvable, original dict otherwise.
        """
        try:
            return cls.resolve(data, type_hint)
        except (ValueError, KeyError, Exception):
            return data

    @classmethod
    def list_schemas(cls) -> list[str]:
        """
        List all registered schema names.

        Returns:
            List of registered schema names.
        """
        cls._ensure_initialized()
        return list(cls._schemas.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if a schema name is registered.

        Args:
            name: The schema name to check.

        Returns:
            True if registered, False otherwise.
        """
        return cls.get(name) is not None

    @classmethod
    def clear(cls) -> None:
        """Clear all registered schemas (for testing)."""
        cls._schemas.clear()
        cls._initialized = False


# =============================================================================
# Schema Discovery Utilities
# =============================================================================


def auto_register_schema(schema_cls: type[BaseSchema]) -> type[BaseSchema]:
    """
    Decorator to auto-register a schema class.

    Uses the _schema_name class variable as the registration key.

    Example:
        >>> @auto_register_schema
        ... class MySchema(BaseSchema):
        ...     _schema_name = "my_schema"
        ...     field: str
    """
    schema_name = getattr(schema_cls, "_schema_name", None)
    if schema_name:
        SchemaRegistry.register(schema_name, schema_cls)
    return schema_cls


def get_schema_for_memory_type(memory_type: str) -> type[BaseSchema] | None:
    """
    Get the schema class for a memory type.

    Maps MemoryType enum values to their corresponding schema classes.

    Args:
        memory_type: Memory type string (e.g., "decision", "error").

    Returns:
        Schema class if found, None otherwise.
    """
    return SchemaRegistry.get(memory_type)
