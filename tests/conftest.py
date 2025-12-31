"""
Pytest configuration and shared fixtures.
"""

import sys
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_python_code() -> str:
    """Sample Python code for testing."""
    return '''
"""Sample module for testing."""

from typing import Optional
import os

class MyClass:
    """A sample class."""

    def __init__(self, name: str):
        """Initialize with name."""
        self.name = name

    def greet(self) -> str:
        """Return greeting."""
        return f"Hello, {self.name}!"


def helper_function(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y


async def async_function() -> None:
    """An async function."""
    pass
'''


@pytest.fixture
def sample_typescript_code() -> str:
    """Sample TypeScript code for testing."""
    return """
import { useState, useEffect } from 'react';
import type { User } from './types';

interface Props {
    name: string;
    age?: number;
}

type Status = 'active' | 'inactive';

export class UserService {
    private users: User[] = [];

    async getUser(id: string): Promise<User | null> {
        return this.users.find(u => u.id === id) ?? null;
    }
}

export function useUser(id: string) {
    const [user, setUser] = useState<User | null>(null);

    useEffect(() => {
        // fetch user
    }, [id]);

    return user;
}
"""


@pytest.fixture
def sample_java_code() -> str:
    """Sample Java code for testing."""
    return """
package com.example.app;

import java.util.List;
import java.util.Optional;

/**
 * A sample Java class.
 */
public class UserService {
    private final UserRepository repository;

    public UserService(UserRepository repository) {
        this.repository = repository;
    }

    /**
     * Find user by ID.
     * @param id User ID
     * @return Optional user
     */
    public Optional<User> findById(String id) {
        return repository.findById(id);
    }
}

interface UserRepository {
    Optional<User> findById(String id);
    List<User> findAll();
}
"""


@pytest.fixture
def sample_go_code() -> str:
    """Sample Go code for testing."""
    return """
package main

import (
    "fmt"
    "net/http"
)

// User represents a user in the system.
type User struct {
    ID   string
    Name string
}

// UserService handles user operations.
type UserService struct {
    users map[string]*User
}

// NewUserService creates a new UserService.
func NewUserService() *UserService {
    return &UserService{
        users: make(map[string]*User),
    }
}

// GetUser retrieves a user by ID.
func (s *UserService) GetUser(id string) (*User, error) {
    user, ok := s.users[id]
    if !ok {
        return nil, fmt.Errorf("user not found: %s", id)
    }
    return user, nil
}
"""


@pytest.fixture
def sample_rust_code() -> str:
    """Sample Rust code for testing."""
    return """
use std::collections::HashMap;

/// A user in the system.
#[derive(Debug, Clone)]
pub struct User {
    pub id: String,
    pub name: String,
}

/// Service for managing users.
pub struct UserService {
    users: HashMap<String, User>,
}

impl UserService {
    /// Create a new UserService.
    pub fn new() -> Self {
        Self {
            users: HashMap::new(),
        }
    }

    /// Get a user by ID.
    pub fn get_user(&self, id: &str) -> Option<&User> {
        self.users.get(id)
    }
}

impl Default for UserService {
    fn default() -> Self {
        Self::new()
    }
}
"""


@pytest.fixture
def sample_markdown() -> str:
    """Sample Markdown for testing."""
    return """
# Sample Document

This is a sample markdown document.

## Introduction

Some introductory text.

### Subsection

More detailed content here.

```python
def hello():
    print("Hello, World!")
```

## Conclusion

Final thoughts.

[Link to docs](./docs/README.md)
"""


@pytest.fixture
def sample_sql() -> str:
    """Sample SQL for testing."""
    return """
-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) NOT NULL UNIQUE,
    name VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Posts table
CREATE TABLE posts (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    content TEXT,
    published BOOLEAN DEFAULT false
);

-- Create index
CREATE INDEX idx_posts_user_id ON posts(user_id);

-- Sample query
SELECT u.name, COUNT(p.id) as post_count
FROM users u
LEFT JOIN posts p ON u.id = p.user_id
GROUP BY u.id, u.name;
"""


@pytest.fixture
def sample_json() -> str:
    """Sample JSON for testing."""
    return """
{
    "name": "my-project",
    "version": "1.0.0",
    "dependencies": {
        "react": "^18.0.0",
        "typescript": "^5.0.0"
    },
    "scripts": {
        "build": "tsc",
        "test": "jest"
    }
}
"""


@pytest.fixture
def haven_project_path() -> Path:
    """Path to haven project for integration tests."""
    path = Path("/Users/mlong/Documents/Development/haven/design-gem-studio")
    if path.exists():
        return path
    pytest.skip("Haven project not available")


@pytest.fixture
def rag_backend(temp_dir: Path):
    """Create a RAG backend for testing."""
    from contextfs.rag import RAGBackend

    backend = RAGBackend(
        data_dir=temp_dir,
        embedding_model="all-MiniLM-L6-v2",
        collection_name="test_collection",
    )
    yield backend
    backend.close()


@pytest.fixture
def fts_backend(temp_dir: Path):
    """Create an FTS backend for testing."""
    import sqlite3

    from contextfs.fts import FTSBackend

    # FTS triggers require the memories table to exist
    # Schema must match the full memories table including source_tool, project
    db_path = temp_dir / "test.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY NOT NULL,
            content TEXT NOT NULL,
            type TEXT NOT NULL,
            tags TEXT,
            summary TEXT,
            namespace_id TEXT NOT NULL,
            source_file TEXT,
            source_repo TEXT,
            source_tool TEXT,
            project TEXT,
            session_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            metadata TEXT,
            structured_data TEXT
        )
    """)
    conn.commit()
    conn.close()

    backend = FTSBackend(db_path=db_path)
    yield backend
    backend.close()
