"""
Entity and Reference types for the ContextFS formal type system.

Implements from Definition 5.1 Type Grammar:
    EntityType ::= Entity Name Schema
    RefType    ::= Ref EntityType

This module provides:
- Entity[S]: A named entity with a typed schema
- Ref[E]: A typed reference to an entity with lazy loading

Entities are the foundation for structured memory content,
while Refs provide type-safe pointers between memories.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Generic, get_args, get_origin
from uuid import uuid4

from pydantic import BaseModel, Field

from contextfs.types.base import BaseSchema, E, S

if TYPE_CHECKING:
    pass


# =============================================================================
# Entity Type (Definition 5.1: EntityType ::= Entity Name Schema)
# =============================================================================


class Entity(BaseModel, Generic[S]):
    """
    A named entity with a typed schema.

    Corresponds to: EntityType ::= Entity Name Schema

    Entities represent structured data with:
    - A unique identifier
    - A human-readable name
    - A typed schema containing the actual data

    The schema type S is preserved through generics, enabling
    type-safe access to entity data.

    Attributes:
        id: Unique identifier for the entity.
        name: Human-readable name.
        schema_data: The typed schema data.

    Type Parameters:
        S: The schema type, must be a subclass of BaseSchema.

    Example:
        >>> from contextfs.types import Entity, BaseSchema
        >>>
        >>> class UserSchema(BaseSchema):
        ...     _schema_name = "user"
        ...     username: str
        ...     email: str
        ...
        >>> user = Entity[UserSchema](
        ...     name="John Doe",
        ...     schema_data=UserSchema(username="johnd", email="john@example.com")
        ... )
        >>> user.schema_data.username
        'johnd'
    """

    id: str = Field(default_factory=lambda: str(uuid4())[:12])
    name: str
    schema_data: Any  # Actually S, but Pydantic needs Any for runtime

    model_config = {"extra": "forbid"}

    @classmethod
    def get_schema_type(cls) -> type[BaseSchema] | None:
        """
        Get the schema type parameter for this Entity class.

        Returns:
            The schema type if specified, None otherwise.

        Example:
            >>> class UserEntity(Entity[UserSchema]): pass
            >>> UserEntity.get_schema_type()
            <class 'UserSchema'>
        """
        # Walk through __orig_bases__ to find Generic parameter
        for base in getattr(cls, "__orig_bases__", []):
            origin = get_origin(base)
            if origin is Entity or (origin is not None and issubclass(origin, Entity)):
                args = get_args(base)
                if args and isinstance(args[0], type):
                    return args[0]
        return None

    @property
    def schema_name(self) -> str:
        """Get the schema name from the schema data."""
        if hasattr(self.schema_data, "_schema_name"):
            return self.schema_data._schema_name
        if hasattr(self.schema_data, "schema_name"):
            return self.schema_data.schema_name()
        return "unknown"

    def to_dict(self) -> dict[str, Any]:
        """Convert entity to dictionary for storage."""
        schema_dict = (
            self.schema_data.model_dump()
            if hasattr(self.schema_data, "model_dump")
            else self.schema_data
        )
        return {
            "id": self.id,
            "name": self.name,
            "schema_name": self.schema_name,
            "schema_data": schema_dict,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        schema_type: type[S] | None = None,
    ) -> Entity[S]:
        """
        Create entity from dictionary.

        Args:
            data: Dictionary containing entity fields.
            schema_type: Optional schema type for validation.

        Returns:
            Entity instance with validated schema data.
        """
        schema_data = data.get("schema_data", {})
        if schema_type is not None:
            schema_data = schema_type.model_validate(schema_data)
        return cls(
            id=data.get("id", str(uuid4())[:12]),
            name=data["name"],
            schema_data=schema_data,
        )


# =============================================================================
# Ref Type (Definition 5.1: RefType ::= Ref EntityType)
# =============================================================================


class Ref(Generic[E]):
    """
    A typed reference to an entity with lazy loading.

    Corresponds to: RefType ::= Ref EntityType

    Refs provide type-safe pointers between memories without
    requiring the target to be loaded. Features:
    - Lazy loading via resolver callback
    - Cached resolution (load once)
    - Type-safe access to referenced entity
    - Equality based on ID (not resolved value)

    Attributes:
        id: The ID of the referenced entity.

    Type Parameters:
        E: The entity type being referenced.

    Example:
        >>> from contextfs.types import Ref, Entity
        >>>
        >>> # Create a reference
        >>> user_ref: Ref[Entity[UserSchema]] = Ref("user123")
        >>>
        >>> # Resolve when needed (with resolver)
        >>> def load_user(id: str) -> Entity[UserSchema]:
        ...     return storage.load_entity(id)
        ...
        >>> user_ref = Ref("user123", resolver=load_user)
        >>> user = user_ref.resolve()  # Loads and caches
        >>> user.name
        'John Doe'
    """

    __slots__ = ("_id", "_resolver", "_cache", "_resolved")

    def __init__(
        self,
        entity_id: str,
        resolver: Callable[[str], E | None] | None = None,
    ) -> None:
        """
        Create a reference to an entity.

        Args:
            entity_id: The ID of the referenced entity.
            resolver: Optional callback to load the entity when needed.
        """
        self._id = entity_id
        self._resolver = resolver
        self._cache: E | None = None
        self._resolved = False

    @property
    def id(self) -> str:
        """Get the ID of the referenced entity."""
        return self._id

    @property
    def is_resolved(self) -> bool:
        """Check if the reference has been resolved."""
        return self._resolved

    def resolve(self) -> E | None:
        """
        Resolve the reference to the actual entity.

        Loads the entity using the resolver callback if not already cached.
        Returns None if no resolver is set or if resolution fails.

        Returns:
            The resolved entity or None.
        """
        if not self._resolved and self._resolver is not None:
            try:
                self._cache = self._resolver(self._id)
            except Exception:
                self._cache = None
            self._resolved = True
        return self._cache

    def with_resolver(self, resolver: Callable[[str], E | None]) -> Ref[E]:
        """
        Create a new Ref with a resolver attached.

        This is useful when you have a Ref from storage and need
        to attach a resolver for lazy loading.

        Args:
            resolver: Callback to load the entity.

        Returns:
            New Ref instance with the resolver attached.
        """
        return Ref(self._id, resolver=resolver)

    def invalidate(self) -> None:
        """Invalidate the cache, forcing re-resolution on next access."""
        self._cache = None
        self._resolved = False

    def __eq__(self, other: object) -> bool:
        """
        Compare references by ID.

        Two Refs are equal if they reference the same entity ID,
        regardless of resolution state or cached values.
        """
        if isinstance(other, Ref):
            return self._id == other._id
        if isinstance(other, str):
            return self._id == other
        return False

    def __hash__(self) -> int:
        """Hash based on entity ID."""
        return hash(self._id)

    def __repr__(self) -> str:
        status = "resolved" if self._resolved else "unresolved"
        return f"Ref({self._id!r}, {status})"

    def __str__(self) -> str:
        return f"Ref({self._id})"

    # Serialization support

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {"ref_id": self._id, "ref_type": "entity"}

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        resolver: Callable[[str], E | None] | None = None,
    ) -> Ref[E]:
        """Create Ref from dictionary."""
        return cls(data["ref_id"], resolver=resolver)


# =============================================================================
# RefList - A list of typed references
# =============================================================================


class RefList(Generic[E]):
    """
    A list of typed references to entities.

    Provides batch operations for working with multiple references.
    """

    __slots__ = ("_refs",)

    def __init__(self, refs: list[Ref[E]] | None = None) -> None:
        self._refs: list[Ref[E]] = refs or []

    def add(self, ref: Ref[E]) -> None:
        """Add a reference to the list."""
        self._refs.append(ref)

    def add_id(
        self,
        entity_id: str,
        resolver: Callable[[str], E | None] | None = None,
    ) -> Ref[E]:
        """Create and add a reference by ID."""
        ref = Ref[E](entity_id, resolver=resolver)
        self._refs.append(ref)
        return ref

    @property
    def ids(self) -> list[str]:
        """Get all referenced entity IDs."""
        return [ref.id for ref in self._refs]

    def resolve_all(self) -> list[E | None]:
        """Resolve all references."""
        return [ref.resolve() for ref in self._refs]

    def with_resolver(
        self,
        resolver: Callable[[str], E | None],
    ) -> RefList[E]:
        """Create new RefList with resolver attached to all refs."""
        return RefList([ref.with_resolver(resolver) for ref in self._refs])

    def __len__(self) -> int:
        return len(self._refs)

    def __iter__(self):
        return iter(self._refs)

    def __getitem__(self, index: int) -> Ref[E]:
        return self._refs[index]

    def to_dict(self) -> list[dict[str, Any]]:
        """Convert to list of dicts for storage."""
        return [ref.to_dict() for ref in self._refs]

    @classmethod
    def from_dict(
        cls,
        data: list[dict[str, Any]],
        resolver: Callable[[str], E | None] | None = None,
    ) -> RefList[E]:
        """Create RefList from list of dicts."""
        refs = [Ref.from_dict(d, resolver=resolver) for d in data]
        return cls(refs)
