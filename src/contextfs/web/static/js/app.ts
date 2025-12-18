/**
 * ContextFS Memory Browser - Main Application
 *
 * A lightweight web UI for browsing and searching memories
 * using sql.js for client-side SQLite queries.
 */

import type {
    Memory,
    MemoryType,
    SearchResult,
    SearchMode,
    DualSearchResult,
    Session,
    Stats,
    APIResponse,
    SqlJsStatic,
    SqlJsDatabase,
    QueryExecResult,
} from './types.js';

class ContextFSBrowser {
    private db: SqlJsDatabase | null = null;
    private apiBase: string = '/api';
    private useSqlJs: boolean = false;

    // Pagination state - sessions
    private sessionsOffset: number = 0;
    private sessionsPageSize: number = 20;
    private sessionsHasMore: boolean = true;

    // Pagination state - search
    private searchOffset: number = 0;
    private searchPageSize: number = 20;
    private searchHasMore: boolean = true;
    private currentSearchParams: {
        query: string;
        type?: MemoryType;
        namespace?: string;
        searchMode: SearchMode;
    } | null = null;
    private currentSearchResults: SearchResult[] = [];

    // DOM elements
    private searchInput!: HTMLInputElement;
    private searchBtn!: HTMLButtonElement;
    private typeFilter!: HTMLSelectElement;
    private namespaceFilter!: HTMLSelectElement;
    private resultsContainer!: HTMLElement;

    // Namespace display names cache
    private namespaceDisplayNames: Map<string, string> = new Map();
    private dualResultsContainer!: HTMLElement;
    private ftsResultsContainer!: HTMLElement;
    private ragResultsContainer!: HTMLElement;
    private recentContainer!: HTMLElement;
    private sessionsContainer!: HTMLElement;
    private statsContainer!: HTMLElement;
    private modal!: HTMLElement;
    private modalBody!: HTMLElement;
    private dbStatus!: HTMLElement;
    private memoryCount!: HTMLElement;

    constructor() {
        this.initElements();
        this.initEventListeners();
        this.init();
    }

    private initElements(): void {
        this.searchInput = document.getElementById('search-input') as HTMLInputElement;
        this.searchBtn = document.getElementById('search-btn') as HTMLButtonElement;
        this.typeFilter = document.getElementById('type-filter') as HTMLSelectElement;
        this.namespaceFilter = document.getElementById('namespace-filter') as HTMLSelectElement;
        this.resultsContainer = document.getElementById('results') as HTMLElement;
        this.dualResultsContainer = document.getElementById('dual-results') as HTMLElement;
        this.ftsResultsContainer = document.getElementById('fts-results') as HTMLElement;
        this.ragResultsContainer = document.getElementById('rag-results') as HTMLElement;
        this.recentContainer = document.getElementById('recent-memories') as HTMLElement;
        this.sessionsContainer = document.getElementById('sessions-list') as HTMLElement;
        this.statsContainer = document.getElementById('stats-content') as HTMLElement;
        this.modal = document.getElementById('memory-modal') as HTMLElement;
        this.modalBody = document.getElementById('modal-body') as HTMLElement;
        this.dbStatus = document.getElementById('db-status') as HTMLElement;
        this.memoryCount = document.getElementById('memory-count') as HTMLElement;
    }

    private initEventListeners(): void {
        // Search
        this.searchBtn.addEventListener('click', () => this.search());
        this.searchInput.addEventListener('keypress', (e: KeyboardEvent) => {
            if (e.key === 'Enter') this.search();
        });

        // Filters
        this.typeFilter.addEventListener('change', () => this.search());
        this.namespaceFilter.addEventListener('change', () => this.search());

        // Tabs
        document.querySelectorAll('.tab').forEach((tab) => {
            tab.addEventListener('click', (e: Event) => this.switchTab(e));
        });

        // Modal
        const closeBtn = this.modal.querySelector('.close');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => this.closeModal());
        }
        this.modal.addEventListener('click', (e: MouseEvent) => {
            if (e.target === this.modal) this.closeModal();
        });

        // Footer actions
        document.getElementById('export-btn')?.addEventListener('click', () => this.exportMemories());
        document.getElementById('sync-btn')?.addEventListener('click', () => this.syncToPostgres());
    }

    private async init(): Promise<void> {
        try {
            // Try to initialize sql.js for offline support
            await this.initSqlJs();
            this.dbStatus.textContent = 'sql.js ready';
            this.useSqlJs = true;
        } catch (error) {
            console.log('sql.js not available, using API mode');
            this.dbStatus.textContent = 'API mode';
            this.useSqlJs = false;
        }

        // Load initial data
        await this.loadNamespaces();
        await this.loadRecent();
        await this.loadSessions();
        await this.loadStats();

        // Show recent memories on the search tab initially
        await this.showRecentInSearch();
    }

    private async initSqlJs(): Promise<void> {
        if (typeof window.initSqlJs === 'undefined') {
            throw new Error('sql.js not loaded');
        }

        const SQL: SqlJsStatic = await window.initSqlJs({
            locateFile: (file: string) => `https://cdnjs.cloudflare.com/ajax/libs/sql.js/1.10.2/${file}`,
        });

        // Try to load database from API
        try {
            const response = await fetch(`${this.apiBase}/database`);
            if (response.ok) {
                const buffer = await response.arrayBuffer();
                this.db = new SQL.Database(new Uint8Array(buffer));
            } else {
                this.db = new SQL.Database();
            }
        } catch {
            this.db = new SQL.Database();
        }
    }

    // ==================== Search ====================

    private async search(loadMore: boolean = false): Promise<void> {
        const query = this.searchInput.value.trim();
        if (!query) {
            // Show recent memories when no query
            await this.showRecentInSearch();
            return;
        }

        const type = this.typeFilter.value as MemoryType | '';
        const namespace = this.namespaceFilter.value;
        const searchMode: SearchMode = 'hybrid'; // Default to hybrid search

        // Reset pagination for new search
        if (!loadMore) {
            this.searchOffset = 0;
            this.searchHasMore = true;
            this.currentSearchResults = [];
            this.currentSearchParams = {
                query,
                type: type || undefined,
                namespace: namespace || undefined,
                searchMode,
            };
        }

        try {
            // Show single view
            this.showSingleView();
            if (!loadMore) {
                this.resultsContainer.innerHTML = '<div class="loading"></div>';
            }

            const results = await this.searchAPI(query, type || undefined, namespace || undefined, searchMode, this.searchOffset);

            // Check if there are more results
            this.searchHasMore = results.length === this.searchPageSize;
            this.searchOffset += results.length;
            this.currentSearchResults = loadMore ? [...this.currentSearchResults, ...results] : results;

            this.renderResults(results, loadMore);
        } catch (error) {
            this.resultsContainer.innerHTML = `<p class="placeholder">Error: ${(error as Error).message}</p>`;
        }
    }

    private async loadMoreSearch(): Promise<void> {
        await this.search(true);
    }

    private showSingleView(): void {
        this.resultsContainer.style.display = 'block';
        this.dualResultsContainer.style.display = 'none';
    }

    private showDualView(): void {
        this.resultsContainer.style.display = 'none';
        this.dualResultsContainer.style.display = 'grid';
    }

    private async showRecentInSearch(): Promise<void> {
        this.showSingleView();
        this.resultsContainer.innerHTML = '<div class="loading"></div>';

        // Get current filter values
        const type = this.typeFilter.value as MemoryType | '';
        const namespace = this.namespaceFilter.value;

        try {
            const memories = await this.getRecentAPI(20, type || undefined, namespace || undefined);
            if (memories.length === 0) {
                const filterMsg = type ? ` of type "${type}"` : '';
                this.resultsContainer.innerHTML = `<p class="placeholder">No memories${filterMsg} found.</p>`;
                return;
            }

            const title = type ? `Recent ${type.charAt(0).toUpperCase() + type.slice(1)} Memories` : 'Recent Memories';
            this.resultsContainer.innerHTML = `<h3 class="section-title">${title}</h3>` +
                memories.map((m) => this.renderMemoryCard({
                    memory: m,
                    score: 1.0,
                    highlights: [],
                })).join('');

            this.resultsContainer.querySelectorAll('.memory-card').forEach((card) => {
                card.addEventListener('click', () => {
                    const id = card.getAttribute('data-id');
                    if (id) {
                        const memory = memories.find((m) => m.id === id);
                        if (memory) {
                            this.showMemoryDetail(id, [{ memory, score: 1.0, highlights: [] }]);
                        }
                    }
                });
            });
        } catch (error) {
            this.resultsContainer.innerHTML = `<p class="placeholder">Error loading memories: ${(error as Error).message}</p>`;
        }
    }

    private searchLocal(
        query: string,
        type?: MemoryType,
        namespace?: string,
        limit: number = 20,
        offset: number = 0
    ): SearchResult[] {
        if (!this.db) return [];

        // FTS5 search
        let sql = `
            SELECT m.*, bm25(memories_fts, 0, 10.0, 5.0, 2.0, 0, 0) as rank
            FROM memories m
            JOIN memories_fts fts ON m.id = fts.id
            WHERE memories_fts MATCH ?
        `;
        const params: unknown[] = [this.prepareFtsQuery(query)];

        if (namespace) {
            sql += ' AND m.namespace_id = ?';
            params.push(namespace);
        }

        if (type) {
            sql += ' AND m.type = ?';
            params.push(type);
        }

        sql += ` ORDER BY rank LIMIT ${limit} OFFSET ${offset}`;

        try {
            const results = this.db.exec(sql, params);
            if (!results.length || !results[0].values.length) return [];

            return results[0].values.map((row) => this.rowToSearchResult(results[0].columns, row));
        } catch {
            // Fallback to LIKE search
            return this.searchLocalFallback(query, type, namespace, limit, offset);
        }
    }

    private searchLocalFallback(
        query: string,
        type?: MemoryType,
        namespace?: string,
        limit: number = 20,
        offset: number = 0
    ): SearchResult[] {
        if (!this.db) return [];

        let sql = `SELECT * FROM memories WHERE (content LIKE ? OR summary LIKE ?)`;
        const params: unknown[] = [`%${query}%`, `%${query}%`];

        if (namespace) {
            sql += ' AND namespace_id = ?';
            params.push(namespace);
        }

        if (type) {
            sql += ' AND type = ?';
            params.push(type);
        }

        sql += ` ORDER BY created_at DESC LIMIT ${limit} OFFSET ${offset}`;

        const results = this.db.exec(sql, params);
        if (!results.length || !results[0].values.length) return [];

        return results[0].values.map((row) => ({
            memory: this.rowToMemory(results[0].columns, row),
            score: 0.5,
            highlights: [],
        }));
    }

    private prepareFtsQuery(query: string): string {
        // Add prefix matching for better recall
        if (!query.includes('"') && !query.includes('*')) {
            const words = query.split(/\s+/);
            if (words.length === 1) {
                return `${words[0]}*`;
            }
            return words.map((w) => `${w}*`).join(' OR ');
        }
        return query;
    }

    private async searchAPI(
        query: string,
        type?: MemoryType,
        namespace?: string,
        searchMode: SearchMode = 'hybrid',
        offset: number = 0
    ): Promise<SearchResult[]> {
        const params = new URLSearchParams({
            q: query,
            limit: String(this.searchPageSize),
            offset: String(offset),
        });

        // Map search mode to API parameters
        if (searchMode === 'smart') {
            params.set('smart', 'true');
        } else if (searchMode === 'fts') {
            params.set('semantic', 'false');
        } else {
            params.set('semantic', 'true');
        }

        if (type) params.set('type', type);
        if (namespace) params.set('namespace', namespace);

        const response = await fetch(`${this.apiBase}/search?${params}`);
        const data: APIResponse<SearchResult[]> = await response.json();

        if (!data.success || !data.data) {
            throw new Error(data.error || 'Search failed');
        }

        return data.data;
    }

    private async searchDualAPI(
        query: string,
        type?: MemoryType,
        namespace?: string,
    ): Promise<DualSearchResult> {
        const params = new URLSearchParams({
            q: query,
            limit: '20',
        });

        if (type) params.set('type', type);
        if (namespace) params.set('namespace', namespace);

        const response = await fetch(`${this.apiBase}/search/dual?${params}`);
        const data: APIResponse<DualSearchResult> = await response.json();

        if (!data.success || !data.data) {
            throw new Error(data.error || 'Dual search failed');
        }

        return data.data;
    }

    private renderDualResults(results: DualSearchResult): void {
        // Render FTS results
        if (!results.fts.length) {
            this.ftsResultsContainer.innerHTML = '<p class="placeholder">No FTS results</p>';
        } else {
            this.ftsResultsContainer.innerHTML = results.fts.map((r) => this.renderMemoryCard(r)).join('');
            this.addCardClickHandlers(this.ftsResultsContainer, results.fts);
        }

        // Render RAG results
        if (!results.rag.length) {
            this.ragResultsContainer.innerHTML = '<p class="placeholder">No RAG results</p>';
        } else {
            this.ragResultsContainer.innerHTML = results.rag.map((r) => this.renderMemoryCard(r)).join('');
            this.addCardClickHandlers(this.ragResultsContainer, results.rag);
        }
    }

    private addCardClickHandlers(container: HTMLElement, results: SearchResult[]): void {
        container.querySelectorAll('.memory-card').forEach((card) => {
            card.addEventListener('click', () => {
                const id = card.getAttribute('data-id');
                if (id) this.showMemoryDetail(id, results);
            });
        });
    }

    private renderResults(results: SearchResult[], append: boolean = false): void {
        if (!results.length && !append) {
            this.resultsContainer.innerHTML = '<p class="placeholder">No memories found</p>';
            return;
        }

        const resultsHtml = results.map((r) => this.renderMemoryCard(r)).join('');

        const loadMoreHtml = this.searchHasMore ? `
            <button class="load-more-btn" id="load-more-search">Load More Results</button>
        ` : '';

        if (append) {
            // Remove old load more button if exists
            const oldBtn = this.resultsContainer.querySelector('#load-more-search');
            if (oldBtn) oldBtn.remove();
            // Append new results
            this.resultsContainer.insertAdjacentHTML('beforeend', resultsHtml + loadMoreHtml);
        } else {
            this.resultsContainer.innerHTML = resultsHtml + loadMoreHtml;
        }

        // Add click handlers for memory cards
        this.resultsContainer.querySelectorAll('.memory-card').forEach((card) => {
            if (!card.hasAttribute('data-listener')) {
                card.setAttribute('data-listener', 'true');
                card.addEventListener('click', () => {
                    const id = card.getAttribute('data-id');
                    if (id) this.showMemoryDetail(id, this.currentSearchResults);
                });
            }
        });

        // Add click handler for load more button
        const loadMoreBtn = this.resultsContainer.querySelector('#load-more-search');
        if (loadMoreBtn) {
            loadMoreBtn.addEventListener('click', () => this.loadMoreSearch());
        }
    }

    private renderMemoryCard(result: SearchResult): string {
        const { memory, score, highlights, source } = result;
        let displayContent = memory.content;

        // Apply highlights
        if (highlights.length > 0) {
            displayContent = highlights[0];
        }

        const tags = memory.tags.length > 0
            ? `<div class="memory-tags">${memory.tags.map((t) => `<span class="tag">${this.escapeHtml(t)}</span>`).join('')}</div>`
            : '';

        // Source badge
        const sourceBadge = source
            ? `<span class="source-badge source-${source}">${source.toUpperCase()}</span>`
            : '';

        return `
            <div class="memory-card" data-id="${memory.id}">
                <div class="memory-header">
                    <span class="memory-type ${memory.type}">${memory.type}</span>
                    ${sourceBadge}
                    <span class="memory-score">${(score * 100).toFixed(0)}% match</span>
                </div>
                <div class="memory-content">${this.escapeHtml(displayContent)}</div>
                <div class="memory-meta">
                    <span>${this.formatDate(memory.created_at)}</span>
                    ${tags}
                </div>
            </div>
        `;
    }

    // ==================== Recent & Sessions ====================

    private async loadRecent(): Promise<void> {
        try {
            let memories: Memory[];

            if (this.useSqlJs && this.db) {
                memories = this.getRecentLocal(20);
            } else {
                memories = await this.getRecentAPI(20);
            }

            this.renderRecentMemories(memories);
            this.memoryCount.textContent = `${memories.length}+ memories`;
        } catch (error) {
            this.recentContainer.innerHTML = `<p class="placeholder">Error loading memories</p>`;
        }
    }

    private getRecentLocal(limit: number = 20): Memory[] {
        if (!this.db) return [];

        const results = this.db.exec(
            `SELECT * FROM memories ORDER BY created_at DESC LIMIT ${limit}`
        );

        if (!results.length || !results[0].values.length) return [];

        return results[0].values.map((row) => this.rowToMemory(results[0].columns, row));
    }

    private async getRecentAPI(limit: number = 20, type?: MemoryType, namespace?: string): Promise<Memory[]> {
        const params = new URLSearchParams({ limit: String(limit) });
        if (type) params.set('type', type);
        if (namespace) params.set('namespace', namespace);

        const response = await fetch(`${this.apiBase}/memories?${params}`);
        const data: APIResponse<Memory[]> = await response.json();

        if (!data.success || !data.data) {
            throw new Error(data.error || 'Failed to load memories');
        }

        return data.data;
    }

    private renderRecentMemories(memories: Memory[]): void {
        if (!memories.length) {
            this.recentContainer.innerHTML = '<p class="placeholder">No memories yet</p>';
            return;
        }

        this.recentContainer.innerHTML = memories.map((m) => this.renderMemoryCard({
            memory: m,
            score: 1.0,
            highlights: [],
        })).join('');

        this.recentContainer.querySelectorAll('.memory-card').forEach((card) => {
            card.addEventListener('click', () => {
                const id = card.getAttribute('data-id');
                if (id) {
                    const result = memories.find((m) => m.id === id);
                    if (result) {
                        this.showMemoryDetail(id, [{ memory: result, score: 1.0, highlights: [] }]);
                    }
                }
            });
        });
    }

    private async loadSessions(reset: boolean = true): Promise<void> {
        try {
            if (reset) {
                this.sessionsOffset = 0;
                this.sessionsHasMore = true;
                this.sessionsContainer.innerHTML = '<div class="loading"></div>';
            }

            let sessions: Session[];

            if (this.useSqlJs && this.db) {
                sessions = this.getSessionsLocal(this.sessionsPageSize, this.sessionsOffset);
            } else {
                sessions = await this.getSessionsAPI(this.sessionsPageSize, this.sessionsOffset);
            }

            // Check if there are more sessions
            this.sessionsHasMore = sessions.length === this.sessionsPageSize;
            this.sessionsOffset += sessions.length;

            this.renderSessions(sessions, !reset);
        } catch (error) {
            this.sessionsContainer.innerHTML = `<p class="placeholder">Error loading sessions</p>`;
        }
    }

    private async loadMoreSessions(): Promise<void> {
        await this.loadSessions(false);
    }

    private getSessionsLocal(limit: number = 20, offset: number = 0): Session[] {
        if (!this.db) return [];

        const results = this.db.exec(
            `SELECT s.*, COUNT(m.id) as message_count
             FROM sessions s
             LEFT JOIN messages m ON s.id = m.session_id
             GROUP BY s.id
             ORDER BY s.started_at DESC
             LIMIT ${limit} OFFSET ${offset}`
        );

        if (!results.length || !results[0].values.length) return [];

        return results[0].values.map((row) => this.rowToSession(results[0].columns, row));
    }

    private async getSessionsAPI(limit: number = 20, offset: number = 0): Promise<Session[]> {
        const response = await fetch(`${this.apiBase}/sessions?limit=${limit}&offset=${offset}`);
        const data: APIResponse<Session[]> = await response.json();

        if (!data.success || !data.data) {
            throw new Error(data.error || 'Failed to load sessions');
        }

        return data.data;
    }

    private renderSessions(sessions: Session[], append: boolean = false): void {
        if (!sessions.length && !append) {
            this.sessionsContainer.innerHTML = '<p class="placeholder">No sessions yet</p>';
            return;
        }

        const sessionsHtml = sessions.map((s) => `
            <div class="session-card" data-id="${s.id}">
                <div class="session-header">
                    <span class="session-tool">${this.escapeHtml(s.tool)}</span>
                    <span class="session-date">${this.formatDate(s.started_at)}</span>
                </div>
                <div class="session-summary">
                    ${s.label ? this.escapeHtml(s.label) : `Session ${s.id.substring(0, 8)}`}
                    ${s.message_count ? ` • ${s.message_count} messages` : ''}
                </div>
            </div>
        `).join('');

        const loadMoreHtml = this.sessionsHasMore ? `
            <button class="load-more-btn" id="load-more-sessions">Load More Sessions</button>
        ` : '';

        if (append) {
            // Remove old load more button if exists
            const oldBtn = this.sessionsContainer.querySelector('#load-more-sessions');
            if (oldBtn) oldBtn.remove();
            // Append new sessions
            this.sessionsContainer.insertAdjacentHTML('beforeend', sessionsHtml + loadMoreHtml);
        } else {
            this.sessionsContainer.innerHTML = sessionsHtml + loadMoreHtml;
        }

        // Add click handlers for session cards
        this.sessionsContainer.querySelectorAll('.session-card').forEach((card) => {
            if (!card.hasAttribute('data-listener')) {
                card.setAttribute('data-listener', 'true');
                card.addEventListener('click', () => {
                    const id = card.getAttribute('data-id');
                    if (id) this.showSessionDetail(id);
                });
            }
        });

        // Add click handler for load more button
        const loadMoreBtn = this.sessionsContainer.querySelector('#load-more-sessions');
        if (loadMoreBtn) {
            loadMoreBtn.addEventListener('click', () => this.loadMoreSessions());
        }
    }

    // ==================== Stats ====================

    private async loadStats(): Promise<void> {
        try {
            let stats: Stats;

            if (this.useSqlJs && this.db) {
                stats = this.getStatsLocal();
            } else {
                stats = await this.getStatsAPI();
            }

            this.renderStats(stats);
        } catch (error) {
            this.statsContainer.innerHTML = `<p class="placeholder">Error loading statistics</p>`;
        }
    }

    private getStatsLocal(): Stats {
        if (!this.db) {
            return {
                total_memories: 0,
                memories_by_type: {} as Record<MemoryType, number>,
                total_sessions: 0,
                namespaces: [],
                fts_indexed: 0,
                rag_indexed: 0,
            };
        }

        const countResult = this.db.exec('SELECT COUNT(*) FROM memories');
        const totalMemories = countResult[0]?.values[0]?.[0] as number || 0;

        const typeResult = this.db.exec('SELECT type, COUNT(*) FROM memories GROUP BY type');
        const memoriesByType: Record<string, number> = {};
        if (typeResult[0]) {
            typeResult[0].values.forEach((row) => {
                memoriesByType[row[0] as string] = row[1] as number;
            });
        }

        const sessionResult = this.db.exec('SELECT COUNT(*) FROM sessions');
        const totalSessions = sessionResult[0]?.values[0]?.[0] as number || 0;

        const namespaceResult = this.db.exec('SELECT DISTINCT namespace_id FROM memories');
        const namespaces = namespaceResult[0]?.values.map((row) => row[0] as string) || [];

        return {
            total_memories: totalMemories,
            memories_by_type: memoriesByType as Record<MemoryType, number>,
            total_sessions: totalSessions,
            namespaces,
            fts_indexed: totalMemories,
            rag_indexed: 0,
        };
    }

    private async getStatsAPI(): Promise<Stats> {
        const response = await fetch(`${this.apiBase}/stats`);
        const data: APIResponse<Stats> = await response.json();

        if (!data.success || !data.data) {
            throw new Error(data.error || 'Failed to load stats');
        }

        return data.data;
    }

    private renderStats(stats: Stats): void {
        const typeBreakdown = Object.entries(stats.memories_by_type)
            .map(([type, count]) => `<div class="stat-card"><div class="stat-value">${count}</div><div class="stat-label">${type}</div></div>`)
            .join('');

        this.statsContainer.innerHTML = `
            <div class="stat-card">
                <div class="stat-value">${stats.total_memories}</div>
                <div class="stat-label">Total Memories</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${stats.total_sessions}</div>
                <div class="stat-label">Sessions</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${stats.namespaces.length}</div>
                <div class="stat-label">Namespaces</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${stats.fts_indexed}</div>
                <div class="stat-label">FTS Indexed</div>
            </div>
            ${typeBreakdown}
        `;
    }

    private async loadNamespaces(): Promise<void> {
        try {
            interface NamespaceInfo {
                id: string;
                display_name: string;
                source_repo: string | null;
            }

            let namespaces: NamespaceInfo[];

            if (this.useSqlJs && this.db) {
                // Get namespace IDs and source repos from local DB
                const result = this.db.exec(
                    'SELECT DISTINCT namespace_id, source_repo FROM memories'
                );
                namespaces = (result[0]?.values || []).map((row) => {
                    const nsId = row[0] as string;
                    const sourceRepo = row[1] as string | null;
                    let displayName = nsId;

                    if (sourceRepo) {
                        const repoName = sourceRepo.replace(/\/$/, '').split('/').pop();
                        if (repoName) displayName = repoName;
                    } else if (nsId === 'global') {
                        displayName = 'global';
                    } else if (nsId.startsWith('repo-')) {
                        displayName = `repo-${nsId.slice(5, 13)}...`;
                    }

                    return { id: nsId, display_name: displayName, source_repo: sourceRepo };
                });
            } else {
                const response = await fetch(`${this.apiBase}/namespaces`);
                const data: APIResponse<NamespaceInfo[]> = await response.json();
                namespaces = data.data || [];
            }

            // Cache display names and populate dropdown
            namespaces.forEach((ns) => {
                this.namespaceDisplayNames.set(ns.id, ns.display_name);
                const option = document.createElement('option');
                option.value = ns.id;
                option.textContent = ns.display_name;
                if (ns.source_repo) {
                    option.title = ns.source_repo; // Show full path on hover
                }
                this.namespaceFilter.appendChild(option);
            });
        } catch {
            // Ignore namespace loading errors
        }
    }

    // ==================== Modal ====================

    private showMemoryDetail(id: string, results: SearchResult[]): void {
        const result = results.find((r) => r.memory.id === id);
        if (!result) return;

        const { memory } = result;

        this.modalBody.innerHTML = `
            <div class="modal-header">
                <span class="memory-type ${memory.type}">${memory.type}</span>
                <h3>${memory.summary || `Memory ${memory.id.substring(0, 8)}`}</h3>
            </div>
            <div class="modal-body">
                <pre>${this.escapeHtml(memory.content)}</pre>
                <div class="memory-meta" style="margin-top: 16px;">
                    <p><strong>ID:</strong> ${memory.id}</p>
                    <p><strong>Namespace:</strong> ${memory.namespace_id}</p>
                    <p><strong>Created:</strong> ${this.formatDate(memory.created_at)}</p>
                    ${memory.tags.length ? `<p><strong>Tags:</strong> ${memory.tags.join(', ')}</p>` : ''}
                    ${memory.source_file ? `<p><strong>Source:</strong> ${memory.source_file}</p>` : ''}
                </div>
            </div>
        `;

        this.modal.classList.add('active');
    }

    private async showSessionDetail(id: string): Promise<void> {
        // Load session messages from API or local DB
        this.modalBody.innerHTML = '<div class="loading"></div>';
        this.modal.classList.add('active');

        try {
            const response = await fetch(`${this.apiBase}/sessions/${id}`);
            const data: APIResponse<Session & { messages: { role: string; content: string; timestamp: string }[] }> = await response.json();

            if (!data.success || !data.data) {
                throw new Error(data.error || 'Failed to load session');
            }

            const session = data.data;
            const messages = session.messages || [];

            this.modalBody.innerHTML = `
                <div class="modal-header">
                    <h3>${session.label || `Session ${session.id.substring(0, 8)}`}</h3>
                    <p class="text-muted">${session.tool} • ${this.formatDate(session.started_at)}</p>
                </div>
                <div class="modal-body">
                    ${messages.map((m) => `
                        <div class="message ${m.role}">
                            <strong>${m.role}:</strong>
                            <p>${this.escapeHtml(m.content)}</p>
                        </div>
                    `).join('')}
                </div>
            `;
        } catch (error) {
            this.modalBody.innerHTML = `<p class="placeholder">Error: ${(error as Error).message}</p>`;
        }
    }

    private closeModal(): void {
        this.modal.classList.remove('active');
    }

    // ==================== Actions ====================

    private async exportMemories(): Promise<void> {
        try {
            const response = await fetch(`${this.apiBase}/export`);
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);

            const a = document.createElement('a');
            a.href = url;
            a.download = `contextfs-export-${new Date().toISOString().split('T')[0]}.json`;
            a.click();

            URL.revokeObjectURL(url);
        } catch (error) {
            alert(`Export failed: ${(error as Error).message}`);
        }
    }

    private async syncToPostgres(): Promise<void> {
        if (!confirm('Sync all memories to PostgreSQL?')) return;

        try {
            const response = await fetch(`${this.apiBase}/sync`, {
                method: 'POST',
            });
            const data: APIResponse<{ synced: number }> = await response.json();

            if (data.success) {
                alert(`Synced ${data.data?.synced || 0} memories to PostgreSQL`);
            } else {
                throw new Error(data.error || 'Sync failed');
            }
        } catch (error) {
            alert(`Sync failed: ${(error as Error).message}`);
        }
    }

    // ==================== Tabs ====================

    private switchTab(e: Event): void {
        const tab = e.target as HTMLElement;
        const tabName = tab.getAttribute('data-tab');
        if (!tabName) return;

        // Update tab buttons
        document.querySelectorAll('.tab').forEach((t) => t.classList.remove('active'));
        tab.classList.add('active');

        // Update content
        document.querySelectorAll('.tab-content').forEach((c) => c.classList.remove('active'));
        document.getElementById(`${tabName}-tab`)?.classList.add('active');
    }

    // ==================== Helpers ====================

    private rowToMemory(columns: string[], row: unknown[]): Memory {
        const obj: Record<string, unknown> = {};
        columns.forEach((col, i) => {
            obj[col] = row[i];
        });

        return {
            id: obj.id as string,
            content: obj.content as string,
            type: obj.type as MemoryType,
            tags: typeof obj.tags === 'string' ? JSON.parse(obj.tags) : [],
            summary: obj.summary as string | null,
            namespace_id: obj.namespace_id as string,
            source_file: obj.source_file as string | null,
            source_repo: obj.source_repo as string | null,
            session_id: obj.session_id as string | null,
            created_at: obj.created_at as string,
            updated_at: obj.updated_at as string,
            metadata: typeof obj.metadata === 'string' ? JSON.parse(obj.metadata) : {},
        };
    }

    private rowToSearchResult(columns: string[], row: unknown[]): SearchResult {
        const rankIndex = columns.indexOf('rank');
        const rank = rankIndex >= 0 ? row[rankIndex] as number : 0;

        // Exclude rank column for memory parsing
        const memoryColumns = columns.filter((_, i) => i !== rankIndex);
        const memoryRow = row.filter((_, i) => i !== rankIndex);

        return {
            memory: this.rowToMemory(memoryColumns, memoryRow),
            score: 1.0 / (1.0 + Math.abs(rank)),
            highlights: [],
        };
    }

    private rowToSession(columns: string[], row: unknown[]): Session {
        const obj: Record<string, unknown> = {};
        columns.forEach((col, i) => {
            obj[col] = row[i];
        });

        return {
            id: obj.id as string,
            label: obj.label as string | null,
            namespace_id: obj.namespace_id as string,
            tool: obj.tool as string,
            repo_path: obj.repo_path as string | null,
            branch: obj.branch as string | null,
            started_at: obj.started_at as string,
            ended_at: obj.ended_at as string | null,
            summary: obj.summary as string | null,
            metadata: typeof obj.metadata === 'string' ? JSON.parse(obj.metadata) : {},
            message_count: obj.message_count as number | undefined,
        };
    }

    private escapeHtml(text: string): string {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    private formatDate(dateStr: string): string {
        try {
            const date = new Date(dateStr);
            return date.toLocaleDateString('en-US', {
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit',
            });
        } catch {
            return dateStr;
        }
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new ContextFSBrowser();
});
