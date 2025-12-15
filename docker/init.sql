-- ContextFS PostgreSQL Schema
-- This script initializes the database for global memory sync

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Memories table
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    type TEXT NOT NULL,
    tags JSONB DEFAULT '[]',
    summary TEXT,
    namespace_id TEXT NOT NULL,
    source_file TEXT,
    source_repo TEXT,
    session_id TEXT,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    metadata JSONB DEFAULT '{}',
    machine_id TEXT,
    sync_version INTEGER DEFAULT 1
);

-- Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    label TEXT,
    namespace_id TEXT NOT NULL,
    tool TEXT NOT NULL,
    repo_path TEXT,
    branch TEXT,
    started_at TIMESTAMPTZ NOT NULL,
    ended_at TIMESTAMPTZ,
    summary TEXT,
    metadata JSONB DEFAULT '{}',
    machine_id TEXT,
    sync_version INTEGER DEFAULT 1
);

-- Messages table
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    metadata JSONB DEFAULT '{}'
);

-- Memory references for cross-linking
CREATE TABLE IF NOT EXISTS memory_references (
    id TEXT PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    source_memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    target_memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    reference_type TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(source_memory_id, target_memory_id, reference_type)
);

-- Sync state tracking
CREATE TABLE IF NOT EXISTS sync_state (
    machine_id TEXT PRIMARY KEY,
    last_sync_at TIMESTAMPTZ,
    sync_version INTEGER DEFAULT 0
);

-- Namespaces table (optional - for namespace metadata)
CREATE TABLE IF NOT EXISTS namespaces (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    parent_id TEXT REFERENCES namespaces(id),
    repo_path TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace_id);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
CREATE INDEX IF NOT EXISTS idx_memories_updated ON memories(updated_at);
CREATE INDEX IF NOT EXISTS idx_memories_machine ON memories(machine_id);
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);
CREATE INDEX IF NOT EXISTS idx_memories_tags ON memories USING gin(tags);

CREATE INDEX IF NOT EXISTS idx_sessions_namespace ON sessions(namespace_id);
CREATE INDEX IF NOT EXISTS idx_sessions_tool ON sessions(tool);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_label ON sessions(label);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);

CREATE INDEX IF NOT EXISTS idx_refs_source ON memory_references(source_memory_id);
CREATE INDEX IF NOT EXISTS idx_refs_target ON memory_references(target_memory_id);
CREATE INDEX IF NOT EXISTS idx_refs_type ON memory_references(reference_type);

-- Full-text search index
CREATE INDEX IF NOT EXISTS idx_memories_fts ON memories
    USING gin(to_tsvector('english', content || ' ' || COALESCE(summary, '')));

-- Trigram index for fuzzy search
CREATE INDEX IF NOT EXISTS idx_memories_trgm ON memories
    USING gin(content gin_trgm_ops);

-- Views for common queries

-- Recent memories by namespace
CREATE OR REPLACE VIEW recent_memories AS
SELECT m.*, s.tool as session_tool
FROM memories m
LEFT JOIN sessions s ON m.session_id = s.id
ORDER BY m.created_at DESC;

-- Memory statistics by namespace
CREATE OR REPLACE VIEW namespace_stats AS
SELECT
    namespace_id,
    COUNT(*) as memory_count,
    COUNT(DISTINCT type) as type_count,
    MIN(created_at) as first_memory,
    MAX(created_at) as last_memory
FROM memories
GROUP BY namespace_id;

-- Memory type distribution
CREATE OR REPLACE VIEW type_distribution AS
SELECT
    type,
    namespace_id,
    COUNT(*) as count
FROM memories
GROUP BY type, namespace_id
ORDER BY namespace_id, type;

-- Functions for search

-- Full-text search with ranking
CREATE OR REPLACE FUNCTION search_memories(
    query_text TEXT,
    limit_count INTEGER DEFAULT 20,
    namespace_filter TEXT DEFAULT NULL,
    type_filter TEXT DEFAULT NULL
)
RETURNS TABLE (
    id TEXT,
    content TEXT,
    type TEXT,
    tags JSONB,
    summary TEXT,
    namespace_id TEXT,
    created_at TIMESTAMPTZ,
    rank REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        m.id,
        m.content,
        m.type,
        m.tags,
        m.summary,
        m.namespace_id,
        m.created_at,
        ts_rank(
            to_tsvector('english', m.content || ' ' || COALESCE(m.summary, '')),
            plainto_tsquery('english', query_text)
        ) as rank
    FROM memories m
    WHERE
        to_tsvector('english', m.content || ' ' || COALESCE(m.summary, ''))
        @@ plainto_tsquery('english', query_text)
        AND (namespace_filter IS NULL OR m.namespace_id = namespace_filter)
        AND (type_filter IS NULL OR m.type = type_filter)
    ORDER BY rank DESC
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- Fuzzy search using trigrams
CREATE OR REPLACE FUNCTION fuzzy_search_memories(
    query_text TEXT,
    limit_count INTEGER DEFAULT 20,
    similarity_threshold REAL DEFAULT 0.3
)
RETURNS TABLE (
    id TEXT,
    content TEXT,
    type TEXT,
    namespace_id TEXT,
    similarity REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        m.id,
        m.content,
        m.type,
        m.namespace_id,
        similarity(m.content, query_text) as sim
    FROM memories m
    WHERE similarity(m.content, query_text) > similarity_threshold
    ORDER BY sim DESC
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO contextfs;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO contextfs;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO contextfs;
