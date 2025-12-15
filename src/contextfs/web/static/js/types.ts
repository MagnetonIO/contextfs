/**
 * ContextFS Web UI Type Definitions
 */

export type MemoryType =
    | 'fact'
    | 'decision'
    | 'procedural'
    | 'episodic'
    | 'user'
    | 'code'
    | 'error';

export interface Memory {
    id: string;
    content: string;
    type: MemoryType;
    tags: string[];
    summary: string | null;
    namespace_id: string;
    source_file: string | null;
    source_repo: string | null;
    session_id: string | null;
    created_at: string;
    updated_at: string;
    metadata: Record<string, unknown>;
}

export interface SearchResult {
    memory: Memory;
    score: number;
    highlights: string[];
    source?: 'fts' | 'rag' | 'hybrid' | null;
}

export type SearchMode = 'hybrid' | 'smart' | 'fts' | 'dual';

export interface DualSearchResult {
    fts: SearchResult[];
    rag: SearchResult[];
}

export interface Session {
    id: string;
    label: string | null;
    namespace_id: string;
    tool: string;
    repo_path: string | null;
    branch: string | null;
    started_at: string;
    ended_at: string | null;
    summary: string | null;
    metadata: Record<string, unknown>;
    message_count?: number;
}

export interface SessionMessage {
    id: string;
    role: string;
    content: string;
    timestamp: string;
    metadata: Record<string, unknown>;
}

export interface Stats {
    total_memories: number;
    memories_by_type: Record<MemoryType, number>;
    total_sessions: number;
    namespaces: string[];
    fts_indexed: number;
    rag_indexed: number;
}

export interface APIResponse<T> {
    success: boolean;
    data?: T;
    error?: string;
}

// sql.js types
declare global {
    interface Window {
        initSqlJs: (config?: SqlJsConfig) => Promise<SqlJsStatic>;
    }
}

export interface SqlJsConfig {
    locateFile?: (filename: string) => string;
}

export interface SqlJsStatic {
    Database: new (data?: ArrayLike<number>) => SqlJsDatabase;
}

export interface SqlJsDatabase {
    run(sql: string, params?: unknown[]): void;
    exec(sql: string, params?: unknown[]): QueryExecResult[];
    getRowsModified(): number;
    export(): Uint8Array;
    close(): void;
}

export interface QueryExecResult {
    columns: string[];
    values: unknown[][];
}
