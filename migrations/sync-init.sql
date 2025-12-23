-- ContextFS Sync Service - PostgreSQL Initialization
-- This script runs on first database creation

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "vector";

-- =============================================================================
-- Devices Table
-- Tracks registered sync devices
-- =============================================================================
CREATE TABLE IF NOT EXISTS devices (
    device_id TEXT PRIMARY KEY,
    device_name TEXT NOT NULL,
    platform TEXT NOT NULL,
    client_version TEXT NOT NULL,
    registered_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_sync_at TIMESTAMPTZ,
    sync_cursor TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}'
);

-- =============================================================================
-- Sync State Table
-- Tracks sync progress per device
-- =============================================================================
CREATE TABLE IF NOT EXISTS sync_state (
    device_id TEXT PRIMARY KEY,
    last_push_at TIMESTAMPTZ,
    last_pull_at TIMESTAMPTZ,
    push_cursor TIMESTAMPTZ,
    pull_cursor TIMESTAMPTZ,
    total_pushed INTEGER DEFAULT 0,
    total_pulled INTEGER DEFAULT 0,
    total_conflicts INTEGER DEFAULT 0
);

-- =============================================================================
-- Memories Table
-- Core memory storage with sync fields
-- =============================================================================
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    type TEXT NOT NULL DEFAULT 'fact',
    tags TEXT[] DEFAULT '{}',
    summary TEXT,
    namespace_id TEXT NOT NULL DEFAULT 'global',

    -- Portable source reference (for cross-machine sync)
    repo_url TEXT,
    repo_name TEXT,
    relative_path TEXT,

    -- Legacy source fields
    source_file TEXT,
    source_repo TEXT,
    source_tool TEXT,

    -- Context
    project TEXT,
    session_id TEXT,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Sync fields
    vector_clock JSONB DEFAULT '{}',
    content_hash TEXT,
    deleted_at TIMESTAMPTZ,
    last_modified_by TEXT,

    -- Metadata and embedding
    metadata JSONB DEFAULT '{}',
    embedding vector(384)
);

-- Memory indexes
CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace_id);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
CREATE INDEX IF NOT EXISTS idx_memories_updated ON memories(updated_at);
CREATE INDEX IF NOT EXISTS idx_memories_deleted ON memories(deleted_at);
CREATE INDEX IF NOT EXISTS idx_memories_content_hash ON memories(content_hash);
CREATE INDEX IF NOT EXISTS idx_memories_repo_url ON memories(repo_url);
CREATE INDEX IF NOT EXISTS idx_memories_tags ON memories USING GIN(tags);

-- Vector similarity index (IVFFlat for efficient search)
-- Note: Run this after inserting some data for better list count estimation
-- CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories
--     USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Full-text search
CREATE INDEX IF NOT EXISTS idx_memories_content_fts ON memories
    USING GIN(to_tsvector('english', content));

-- =============================================================================
-- Sessions Table
-- Session tracking with sync support
-- =============================================================================
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    label TEXT,
    namespace_id TEXT NOT NULL DEFAULT 'global',
    tool TEXT NOT NULL DEFAULT 'contextfs',

    -- Portable repo reference
    repo_url TEXT,
    repo_name TEXT,

    -- Legacy field
    repo_path TEXT,

    branch TEXT,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    summary TEXT,
    metadata JSONB DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Sync fields
    vector_clock JSONB DEFAULT '{}',
    content_hash TEXT,
    deleted_at TIMESTAMPTZ,
    last_modified_by TEXT
);

-- Session indexes
CREATE INDEX IF NOT EXISTS idx_sessions_namespace ON sessions(namespace_id);
CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at);
CREATE INDEX IF NOT EXISTS idx_sessions_deleted ON sessions(deleted_at);

-- =============================================================================
-- Memory Edges Table
-- Relationship tracking between memories
-- =============================================================================
CREATE TABLE IF NOT EXISTS memory_edges (
    from_id TEXT NOT NULL,
    to_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    weight FLOAT DEFAULT 1.0,
    created_by TEXT,
    metadata JSONB DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Sync fields
    vector_clock JSONB DEFAULT '{}',
    deleted_at TIMESTAMPTZ,
    last_modified_by TEXT,

    PRIMARY KEY (from_id, to_id, relation)
);

-- Edge indexes
CREATE INDEX IF NOT EXISTS idx_edges_from ON memory_edges(from_id);
CREATE INDEX IF NOT EXISTS idx_edges_to ON memory_edges(to_id);
CREATE INDEX IF NOT EXISTS idx_edges_updated ON memory_edges(updated_at);

-- =============================================================================
-- Messages Table (optional, for session message history)
-- =============================================================================
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY DEFAULT uuid_generate_v4()::TEXT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',

    -- Sync fields
    vector_clock JSONB DEFAULT '{}',
    deleted_at TIMESTAMPTZ,
    last_modified_by TEXT
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);

-- =============================================================================
-- Helper Functions
-- =============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply to all tables
CREATE TRIGGER memories_updated_at
    BEFORE UPDATE ON memories
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER sessions_updated_at
    BEFORE UPDATE ON sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER memory_edges_updated_at
    BEFORE UPDATE ON memory_edges
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- =============================================================================
-- Grants
-- =============================================================================
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO contextfs;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO contextfs;
