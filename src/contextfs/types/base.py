"""
Base types for the ContextFS formal type system.

Implements Definition 5.1 Type Grammar from the typed-memory paper:
    BaseType ::= String | Int | Float | Bool | DateTime | UUID

This module provides:
- Type aliases for base types (String, Int, Float, Bool, DateTime, UUIDType)
- TypeVars for generic schema-indexed types (S, E, T, K, V)
- BaseSchema class - foundation for all typed schemas

The type system enables both runtime (Pydantic) and static (mypy/pyright) enforcement.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar, TypeAlias, TypeVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict

# =============================================================================
# BaseType Aliases (Definition 5.1)
# =============================================================================
# These aliases provide the formal type grammar vocabulary while mapping
# directly to Python's native types.

#: String type alias
String: TypeAlias = str

#: Integer type alias
Int: TypeAlias = int

#: Float type alias
Float: TypeAlias = float

#: Boolean type alias
Bool: TypeAlias = bool

#: DateTime type alias
DateTime: TypeAlias = datetime

#: UUID type alias (named UUIDType to avoid conflict with uuid.UUID import)
UUIDType: TypeAlias = UUID


# =============================================================================
# Type Variables for Generic Types
# =============================================================================
# These TypeVars enable schema-indexed types like Mem[S] and VersionedMem[S]

#: Schema type variable - bound to BaseSchema for schema-indexed types
S = TypeVar("S", bound="BaseSchema")

#: Schema type variable (secondary) for operations involving two schemas
S2 = TypeVar("S2", bound="BaseSchema")

#: Entity type variable - bound to Entity for typed references
E = TypeVar("E", bound="BaseModel")

#: Generic type variable for collection types
T = TypeVar("T")

#: Key type variable for Map types
K = TypeVar("K")

#: Value type variable for Map types
V = TypeVar("V")


# =============================================================================
# BaseSchema Class
# =============================================================================


class BaseSchema(BaseModel):
    """
    Base class for all typed schemas in the formal type system.

    Corresponds to the Schema component of: MemoryType ::= Mem Schema

    Schemas define the structure of memory content with:
    - Type-safe fields with Pydantic validation
    - Optional invariants via validators
    - Support for schema evolution and versioning

    Every schema has a `_schema_name` class variable that identifies
    the schema type for registry lookup and serialization.

    Attributes:
        _schema_name: Class variable identifying the schema type.

    Example:
        >>> class DecisionSchema(BaseSchema):
        ...     _schema_name: ClassVar[str] = "decision"
        ...     decision: str
        ...     rationale: str | None = None
        ...
        >>> schema = DecisionSchema(decision="Use PostgreSQL")
        >>> schema._schema_name
        'decision'
    """

    model_config = ConfigDict(
        extra="allow",  # Allow additional fields for extensibility
        frozen=False,  # Allow mutation (versioning creates new instances)
        validate_assignment=True,  # Validate on field assignment
        use_enum_values=True,  # Serialize enums as values
    )

    #: Schema name for registry lookup and type discrimination
    _schema_name: ClassVar[str] = "base"

    @classmethod
    def schema_name(cls) -> str:
        """Get the schema name for this type."""
        return cls._schema_name

    def to_dict(self) -> dict[str, Any]:
        """Convert schema to dictionary for storage.

        Uses Pydantic's model_dump with settings appropriate for
        JSON serialization and storage.

        Returns:
            Dictionary representation of the schema.
        """
        return self.model_dump(mode="json", exclude_none=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaseSchema:
        """Create schema instance from dictionary.

        Args:
            data: Dictionary with schema fields.

        Returns:
            Schema instance validated against the model.
        """
        return cls.model_validate(data)


# =============================================================================
# Option Type Helper
# =============================================================================


def Option(type_: type[T]) -> type[T | None]:
    """
    Create an Option type (nullable type).

    Corresponds to: OptionType ::= Option Type

    This is primarily for documentation/clarity since Python's
    `T | None` syntax already provides this functionality.

    Args:
        type_: The base type to make optional.

    Returns:
        The union type T | None.

    Example:
        >>> from contextfs.types.base import Option, String
        >>> # These are equivalent:
        >>> field1: Option(String)  # Formal grammar style
        >>> field2: str | None      # Python native style
    """
    return type_ | None  # type: ignore[return-value]


# =============================================================================
# Type Checking Utilities
# =============================================================================

if TYPE_CHECKING:
    # Type aliases for use in type hints

    # Re-export for type checking
    SchemaT = TypeVar("SchemaT", bound=BaseSchema)
