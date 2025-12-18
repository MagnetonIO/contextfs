-- ContextFS PostgreSQL Initialization Script
-- Creates tables for unified storage with pgvector support

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- =============================================================================
-- Namespaces Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS namespaces (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    parent_id TEXT REFERENCES namespaces(id) ON DELETE SET NULL,
    repo_path TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_namespaces_repo ON namespaces(repo_path);

-- =============================================================================
-- Memories Table (with vector embedding)
-- =============================================================================
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    type TEXT NOT NULL DEFAULT 'fact',
    tags TEXT[] DEFAULT '{}',
    summary TEXT,
    namespace_id TEXT NOT NULL DEFAULT 'global',
    source_file TEXT,
    source_repo TEXT,
    source_tool TEXT,
    project TEXT,
    session_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',
    -- pgvector embedding (384 dimensions for all-MiniLM-L6-v2)
    embedding vector(384)
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace_id);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project);
CREATE INDEX IF NOT EXISTS idx_memories_source_repo ON memories(source_repo);
CREATE INDEX IF NOT EXISTS idx_memories_source_tool ON memories(source_tool);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_tags ON memories USING GIN(tags);

-- Vector similarity index (IVFFlat for faster approximate search)
CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Full-text search index
CREATE INDEX IF NOT EXISTS idx_memories_fts ON memories
    USING GIN(to_tsvector('english', content || ' ' || COALESCE(summary, '')));

-- =============================================================================
-- Memory Edges Table (Graph Relationships)
-- =============================================================================
CREATE TABLE IF NOT EXISTS memory_edges (
    from_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    to_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    relation TEXT NOT NULL,
    weight FLOAT NOT NULL DEFAULT 1.0 CHECK (weight >= 0 AND weight <= 1),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by TEXT,
    metadata JSONB DEFAULT '{}',
    PRIMARY KEY (from_id, to_id, relation)
);

CREATE INDEX IF NOT EXISTS idx_edges_from ON memory_edges(from_id);
CREATE INDEX IF NOT EXISTS idx_edges_to ON memory_edges(to_id);
CREATE INDEX IF NOT EXISTS idx_edges_relation ON memory_edges(relation);

-- Partial index for lineage queries
CREATE INDEX IF NOT EXISTS idx_edges_lineage ON memory_edges(from_id, to_id)
    WHERE relation IN ('evolved_from', 'merged_from', 'split_from',
                       'evolved_into', 'merged_into', 'split_into');

-- =============================================================================
-- Sessions Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    label TEXT,
    namespace_id TEXT NOT NULL DEFAULT 'global',
    tool TEXT NOT NULL DEFAULT 'contextfs',
    repo_path TEXT,
    branch TEXT,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    summary TEXT,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_sessions_namespace ON sessions(namespace_id);
CREATE INDEX IF NOT EXISTS idx_sessions_label ON sessions(label);
CREATE INDEX IF NOT EXISTS idx_sessions_tool ON sessions(tool);

-- =============================================================================
-- Messages Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);

-- =============================================================================
-- Index Status Table (for auto-indexing)
-- =============================================================================
CREATE TABLE IF NOT EXISTS index_status (
    repo_path TEXT PRIMARY KEY,
    last_indexed TIMESTAMPTZ,
    file_count INTEGER DEFAULT 0,
    memory_count INTEGER DEFAULT 0,
    status TEXT DEFAULT 'pending',
    error_message TEXT,
    metadata JSONB DEFAULT '{}'
);

-- =============================================================================
-- Indexed Files Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS indexed_files (
    file_path TEXT NOT NULL,
    repo_path TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    memory_ids TEXT[] DEFAULT '{}',
    PRIMARY KEY (file_path, repo_path)
);

CREATE INDEX IF NOT EXISTS idx_indexed_files_repo ON indexed_files(repo_path);

-- =============================================================================
-- Indexed Commits Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS indexed_commits (
    commit_hash TEXT NOT NULL,
    repo_path TEXT NOT NULL,
    indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (commit_hash, repo_path)
);

-- =============================================================================
-- Helper Functions
-- =============================================================================

-- Function to search memories by semantic similarity
CREATE OR REPLACE FUNCTION search_memories_semantic(
    query_embedding vector(384),
    match_threshold FLOAT DEFAULT 0.3,
    match_count INT DEFAULT 10,
    filter_namespace TEXT DEFAULT NULL,
    filter_type TEXT DEFAULT NULL,
    filter_project TEXT DEFAULT NULL
)
RETURNS TABLE (
    id TEXT,
    content TEXT,
    type TEXT,
    tags TEXT[],
    summary TEXT,
    namespace_id TEXT,
    similarity FLOAT
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
        1 - (m.embedding <=> query_embedding) AS similarity
    FROM memories m
    WHERE
        m.embedding IS NOT NULL
        AND (filter_namespace IS NULL OR m.namespace_id = filter_namespace)
        AND (filter_type IS NULL OR m.type = filter_type)
        AND (filter_project IS NULL OR m.project = filter_project)
        AND 1 - (m.embedding <=> query_embedding) >= match_threshold
    ORDER BY m.embedding <=> query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

-- Function to get memory lineage (ancestors)
CREATE OR REPLACE FUNCTION get_memory_ancestors(
    memory_id TEXT,
    max_depth INT DEFAULT 10
)
RETURNS TABLE (
    ancestor_id TEXT,
    relation TEXT,
    depth INT
) AS $$
BEGIN
    RETURN QUERY
    WITH RECURSIVE lineage AS (
        -- Base case: direct ancestors
        SELECT
            e.to_id AS ancestor_id,
            e.relation,
            1 AS depth
        FROM memory_edges e
        WHERE e.from_id = memory_id
        AND e.relation IN ('evolved_from', 'merged_from', 'split_from')

        UNION ALL

        -- Recursive case: ancestors of ancestors
        SELECT
            e.to_id,
            e.relation,
            l.depth + 1
        FROM memory_edges e
        JOIN lineage l ON e.from_id = l.ancestor_id
        WHERE e.relation IN ('evolved_from', 'merged_from', 'split_from')
        AND l.depth < max_depth
    )
    SELECT * FROM lineage ORDER BY depth;
END;
$$ LANGUAGE plpgsql;

-- Function to get memory lineage (descendants)
CREATE OR REPLACE FUNCTION get_memory_descendants(
    memory_id TEXT,
    max_depth INT DEFAULT 10
)
RETURNS TABLE (
    descendant_id TEXT,
    relation TEXT,
    depth INT
) AS $$
BEGIN
    RETURN QUERY
    WITH RECURSIVE lineage AS (
        -- Base case: direct descendants
        SELECT
            e.from_id AS descendant_id,
            e.relation,
            1 AS depth
        FROM memory_edges e
        WHERE e.to_id = memory_id
        AND e.relation IN ('evolved_from', 'merged_from', 'split_from')

        UNION ALL

        -- Recursive case: descendants of descendants
        SELECT
            e.from_id,
            e.relation,
            l.depth + 1
        FROM memory_edges e
        JOIN lineage l ON e.to_id = l.descendant_id
        WHERE e.relation IN ('evolved_from', 'merged_from', 'split_from')
        AND l.depth < max_depth
    )
    SELECT * FROM lineage ORDER BY depth;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Grants
-- =============================================================================
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO contextfs;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO contextfs;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO contextfs;
