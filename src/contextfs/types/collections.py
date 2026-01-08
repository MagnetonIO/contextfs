"""
Typed collection types for the ContextFS formal type system.

Implements from Definition 5.1 Type Grammar:
    ListType ::= List Type
    SetType  ::= Set Type where Ord Type
    MapType  ::= Map KeyType ValueType

This module provides typed wrappers for Python collections that:
- Maintain element type information at runtime
- Support Pydantic serialization
- Enable type-safe operations with Generic type parameters
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Generic

from pydantic import BaseModel, Field

from contextfs.types.base import K, T, V

# =============================================================================
# TypedList (Definition 5.1: ListType ::= List Type)
# =============================================================================


class TypedList(BaseModel, Generic[T]):
    """
    A typed list with element type tracking.

    Corresponds to: ListType ::= List Type

    Provides a Pydantic-serializable list that maintains
    element type information for runtime type checking.

    Type Parameters:
        T: The element type.

    Attributes:
        items: The underlying list of items.
        element_type_name: Name of element type for serialization.

    Example:
        >>> tags = TypedList[str](items=["python", "memory"])
        >>> tags.append("ai")
        >>> tags.items
        ['python', 'memory', 'ai']
    """

    items: list[Any] = Field(default_factory=list)
    element_type_name: str | None = None

    model_config = {"extra": "allow"}

    @classmethod
    def of(cls, *items: T, element_type: type[T] | None = None) -> TypedList[T]:
        """
        Create a TypedList from items.

        Args:
            *items: Items to include in the list.
            element_type: Optional element type for metadata.

        Returns:
            New TypedList containing the items.
        """
        type_name = element_type.__name__ if element_type else None
        return cls(items=list(items), element_type_name=type_name)

    def append(self, item: T) -> None:
        """Append an item to the list."""
        self.items.append(item)

    def extend(self, items: list[T]) -> None:
        """Extend the list with multiple items."""
        self.items.extend(items)

    def remove(self, item: T) -> None:
        """Remove an item from the list."""
        self.items.remove(item)

    def pop(self, index: int = -1) -> T:
        """Remove and return item at index."""
        return self.items.pop(index)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[T]:
        return iter(self.items)

    def __getitem__(self, index: int) -> T:
        return self.items[index]

    def __setitem__(self, index: int, value: T) -> None:
        self.items[index] = value

    def __contains__(self, item: T) -> bool:
        return item in self.items

    def __repr__(self) -> str:
        type_str = f"[{self.element_type_name}]" if self.element_type_name else ""
        return f"TypedList{type_str}({self.items!r})"


# =============================================================================
# TypedSet (Definition 5.1: SetType ::= Set Type where Ord Type)
# =============================================================================


class TypedSet(BaseModel, Generic[T]):
    """
    A typed set with element type tracking.

    Corresponds to: SetType ::= Set Type where Ord Type

    Uses frozenset internally for immutability and hashability,
    but provides mutable operations that return new instances.

    Type Parameters:
        T: The element type (must be hashable).

    Attributes:
        items: The underlying set of items.
        element_type_name: Name of element type for serialization.

    Example:
        >>> tags = TypedSet[str].of("python", "memory", "python")
        >>> len(tags)  # Duplicates removed
        2
    """

    items: frozenset[Any] = Field(default_factory=frozenset)
    element_type_name: str | None = None

    model_config = {"extra": "allow", "frozen": True}

    @classmethod
    def of(cls, *items: T, element_type: type[T] | None = None) -> TypedSet[T]:
        """
        Create a TypedSet from items.

        Args:
            *items: Items to include in the set.
            element_type: Optional element type for metadata.

        Returns:
            New TypedSet containing the unique items.
        """
        type_name = element_type.__name__ if element_type else None
        return cls(items=frozenset(items), element_type_name=type_name)

    def add(self, item: T) -> TypedSet[T]:
        """Return new set with item added."""
        return TypedSet(
            items=self.items | {item},
            element_type_name=self.element_type_name,
        )

    def remove(self, item: T) -> TypedSet[T]:
        """Return new set with item removed."""
        return TypedSet(
            items=self.items - {item},
            element_type_name=self.element_type_name,
        )

    def union(self, other: TypedSet[T]) -> TypedSet[T]:
        """Return union of two sets."""
        return TypedSet(
            items=self.items | other.items,
            element_type_name=self.element_type_name,
        )

    def intersection(self, other: TypedSet[T]) -> TypedSet[T]:
        """Return intersection of two sets."""
        return TypedSet(
            items=self.items & other.items,
            element_type_name=self.element_type_name,
        )

    def difference(self, other: TypedSet[T]) -> TypedSet[T]:
        """Return difference of two sets."""
        return TypedSet(
            items=self.items - other.items,
            element_type_name=self.element_type_name,
        )

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[T]:
        return iter(self.items)

    def __contains__(self, item: T) -> bool:
        return item in self.items

    def __repr__(self) -> str:
        type_str = f"[{self.element_type_name}]" if self.element_type_name else ""
        return f"TypedSet{type_str}({set(self.items)!r})"

    def to_list(self) -> list[T]:
        """Convert to sorted list."""
        try:
            return sorted(self.items)  # type: ignore
        except TypeError:
            return list(self.items)


# =============================================================================
# TypedMap (Definition 5.1: MapType ::= Map KeyType ValueType)
# =============================================================================


class TypedMap(BaseModel, Generic[K, V]):
    """
    A typed map/dictionary with key-value type tracking.

    Corresponds to: MapType ::= Map KeyType ValueType

    Provides a Pydantic-serializable dictionary that maintains
    key and value type information.

    Type Parameters:
        K: The key type.
        V: The value type.

    Attributes:
        items: The underlying dictionary.
        key_type_name: Name of key type for serialization.
        value_type_name: Name of value type for serialization.

    Example:
        >>> scores = TypedMap[str, int](items={"alice": 100, "bob": 85})
        >>> scores["charlie"] = 90
        >>> scores.items
        {'alice': 100, 'bob': 85, 'charlie': 90}
    """

    items: dict[Any, Any] = Field(default_factory=dict)
    key_type_name: str | None = None
    value_type_name: str | None = None

    model_config = {"extra": "allow"}

    @classmethod
    def of(
        cls,
        items: dict[K, V] | None = None,
        key_type: type[K] | None = None,
        value_type: type[V] | None = None,
    ) -> TypedMap[K, V]:
        """
        Create a TypedMap from a dictionary.

        Args:
            items: Dictionary of items.
            key_type: Optional key type for metadata.
            value_type: Optional value type for metadata.

        Returns:
            New TypedMap containing the items.
        """
        return cls(
            items=dict(items) if items else {},
            key_type_name=key_type.__name__ if key_type else None,
            value_type_name=value_type.__name__ if value_type else None,
        )

    def get(self, key: K, default: V | None = None) -> V | None:
        """Get value for key with optional default."""
        return self.items.get(key, default)

    def set(self, key: K, value: V) -> None:
        """Set value for key."""
        self.items[key] = value

    def delete(self, key: K) -> V | None:
        """Delete key and return its value."""
        return self.items.pop(key, None)

    def keys(self) -> list[K]:
        """Get all keys."""
        return list(self.items.keys())

    def values(self) -> list[V]:
        """Get all values."""
        return list(self.items.values())

    def entries(self) -> list[tuple[K, V]]:
        """Get all key-value pairs."""
        return list(self.items.items())

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[K]:
        return iter(self.items)

    def __getitem__(self, key: K) -> V:
        return self.items[key]

    def __setitem__(self, key: K, value: V) -> None:
        self.items[key] = value

    def __delitem__(self, key: K) -> None:
        del self.items[key]

    def __contains__(self, key: K) -> bool:
        return key in self.items

    def __repr__(self) -> str:
        key_str = self.key_type_name or "Any"
        val_str = self.value_type_name or "Any"
        return f"TypedMap[{key_str}, {val_str}]({self.items!r})"
