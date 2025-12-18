/**
 * ContextFS Memory Browser - Main Application
 *
 * A lightweight web UI for browsing and searching memories
 * using sql.js for client-side SQLite queries.
 */
class ContextFSBrowser {
    constructor() {
        this.db = null;
        this.apiBase = '/api';
        this.useSqlJs = false;
        // Pagination state - sessions
        this.sessionsOffset = 0;
        this.sessionsPageSize = 20;
        this.sessionsHasMore = true;
        // Pagination state - search
        this.searchOffset = 0;
        this.searchPageSize = 20;
        this.searchHasMore = true;
        this.currentSearchParams = null;
        this.currentSearchResults = [];
        // Namespace display names cache
        this.namespaceDisplayNames = new Map();
        this.initElements();
        this.initEventListeners();
        this.init();
    }
    initElements() {
        this.searchInput = document.getElementById('search-input');
        this.searchBtn = document.getElementById('search-btn');
        this.typeFilter = document.getElementById('type-filter');
        this.namespaceFilter = document.getElementById('namespace-filter');
        this.resultsContainer = document.getElementById('results');
        this.dualResultsContainer = document.getElementById('dual-results');
        this.ftsResultsContainer = document.getElementById('fts-results');
        this.ragResultsContainer = document.getElementById('rag-results');
        this.recentContainer = document.getElementById('recent-memories');
        this.sessionsContainer = document.getElementById('sessions-list');
        this.statsContainer = document.getElementById('stats-content');
        this.modal = document.getElementById('memory-modal');
        this.modalBody = document.getElementById('modal-body');
        this.dbStatus = document.getElementById('db-status');
        this.memoryCount = document.getElementById('memory-count');
    }
    initEventListeners() {
        // Search
        this.searchBtn.addEventListener('click', () => this.search());
        this.searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter')
                this.search();
        });
        // Filters
        this.typeFilter.addEventListener('change', () => this.search());
        this.namespaceFilter.addEventListener('change', () => this.search());
        // Tabs
        document.querySelectorAll('.tab').forEach((tab) => {
            tab.addEventListener('click', (e) => this.switchTab(e));
        });
        // Modal
        const closeBtn = this.modal.querySelector('.close');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => this.closeModal());
        }
        this.modal.addEventListener('click', (e) => {
            if (e.target === this.modal)
                this.closeModal();
        });
        // Footer actions
        document.getElementById('export-btn')?.addEventListener('click', () => this.exportMemories());
        document.getElementById('sync-btn')?.addEventListener('click', () => this.syncToPostgres());
    }
    async init() {
        try {
            // Try to initialize sql.js for offline support
            await this.initSqlJs();
            this.dbStatus.textContent = 'sql.js ready';
            this.useSqlJs = true;
        }
        catch (error) {
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
    async initSqlJs() {
        if (typeof window.initSqlJs === 'undefined') {
            throw new Error('sql.js not loaded');
        }
        const SQL = await window.initSqlJs({
            locateFile: (file) => `https://cdnjs.cloudflare.com/ajax/libs/sql.js/1.10.2/${file}`,
        });
        // Try to load database from API
        try {
            const response = await fetch(`${this.apiBase}/database`);
            if (response.ok) {
                const buffer = await response.arrayBuffer();
                this.db = new SQL.Database(new Uint8Array(buffer));
            }
            else {
                this.db = new SQL.Database();
            }
        }
        catch {
            this.db = new SQL.Database();
        }
    }
    // ==================== Search ====================
    async search(loadMore = false) {
        const query = this.searchInput.value.trim();
        if (!query) {
            // Show recent memories when no query
            await this.showRecentInSearch();
            return;
        }
        const type = this.typeFilter.value;
        const namespace = this.namespaceFilter.value;
        const searchMode = 'hybrid'; // Default to hybrid search
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
        }
        catch (error) {
            this.resultsContainer.innerHTML = `<p class="placeholder">Error: ${error.message}</p>`;
        }
    }
    async loadMoreSearch() {
        await this.search(true);
    }
    showSingleView() {
        this.resultsContainer.style.display = 'block';
        this.dualResultsContainer.style.display = 'none';
    }
    showDualView() {
        this.resultsContainer.style.display = 'none';
        this.dualResultsContainer.style.display = 'grid';
    }
    async showRecentInSearch() {
        this.showSingleView();
        this.resultsContainer.innerHTML = '<div class="loading"></div>';
        // Get current filter values
        const type = this.typeFilter.value;
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
        }
        catch (error) {
            this.resultsContainer.innerHTML = `<p class="placeholder">Error loading memories: ${error.message}</p>`;
        }
    }
    searchLocal(query, type, namespace, limit = 20, offset = 0) {
        if (!this.db)
            return [];
        // FTS5 search
        let sql = `
            SELECT m.*, bm25(memories_fts, 0, 10.0, 5.0, 2.0, 0, 0) as rank
            FROM memories m
            JOIN memories_fts fts ON m.id = fts.id
            WHERE memories_fts MATCH ?
        `;
        const params = [this.prepareFtsQuery(query)];
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
            if (!results.length || !results[0].values.length)
                return [];
            return results[0].values.map((row) => this.rowToSearchResult(results[0].columns, row));
        }
        catch {
            // Fallback to LIKE search
            return this.searchLocalFallback(query, type, namespace, limit, offset);
        }
    }
    searchLocalFallback(query, type, namespace, limit = 20, offset = 0) {
        if (!this.db)
            return [];
        let sql = `SELECT * FROM memories WHERE (content LIKE ? OR summary LIKE ?)`;
        const params = [`%${query}%`, `%${query}%`];
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
        if (!results.length || !results[0].values.length)
            return [];
        return results[0].values.map((row) => ({
            memory: this.rowToMemory(results[0].columns, row),
            score: 0.5,
            highlights: [],
        }));
    }
    prepareFtsQuery(query) {
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
    async searchAPI(query, type, namespace, searchMode = 'hybrid', offset = 0) {
        const params = new URLSearchParams({
            q: query,
            limit: String(this.searchPageSize),
            offset: String(offset),
        });
        // Map search mode to API parameters
        if (searchMode === 'smart') {
            params.set('smart', 'true');
        }
        else if (searchMode === 'fts') {
            params.set('semantic', 'false');
        }
        else {
            params.set('semantic', 'true');
        }
        if (type)
            params.set('type', type);
        if (namespace)
            params.set('namespace', namespace);
        const response = await fetch(`${this.apiBase}/search?${params}`);
        const data = await response.json();
        if (!data.success || !data.data) {
            throw new Error(data.error || 'Search failed');
        }
        return data.data;
    }
    async searchDualAPI(query, type, namespace) {
        const params = new URLSearchParams({
            q: query,
            limit: '20',
        });
        if (type)
            params.set('type', type);
        if (namespace)
            params.set('namespace', namespace);
        const response = await fetch(`${this.apiBase}/search/dual?${params}`);
        const data = await response.json();
        if (!data.success || !data.data) {
            throw new Error(data.error || 'Dual search failed');
        }
        return data.data;
    }
    renderDualResults(results) {
        // Render FTS results
        if (!results.fts.length) {
            this.ftsResultsContainer.innerHTML = '<p class="placeholder">No FTS results</p>';
        }
        else {
            this.ftsResultsContainer.innerHTML = results.fts.map((r) => this.renderMemoryCard(r)).join('');
            this.addCardClickHandlers(this.ftsResultsContainer, results.fts);
        }
        // Render RAG results
        if (!results.rag.length) {
            this.ragResultsContainer.innerHTML = '<p class="placeholder">No RAG results</p>';
        }
        else {
            this.ragResultsContainer.innerHTML = results.rag.map((r) => this.renderMemoryCard(r)).join('');
            this.addCardClickHandlers(this.ragResultsContainer, results.rag);
        }
    }
    addCardClickHandlers(container, results) {
        container.querySelectorAll('.memory-card').forEach((card) => {
            card.addEventListener('click', () => {
                const id = card.getAttribute('data-id');
                if (id)
                    this.showMemoryDetail(id, results);
            });
        });
    }
    renderResults(results, append = false) {
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
            if (oldBtn)
                oldBtn.remove();
            // Append new results
            this.resultsContainer.insertAdjacentHTML('beforeend', resultsHtml + loadMoreHtml);
        }
        else {
            this.resultsContainer.innerHTML = resultsHtml + loadMoreHtml;
        }
        // Add click handlers for memory cards
        this.resultsContainer.querySelectorAll('.memory-card').forEach((card) => {
            if (!card.hasAttribute('data-listener')) {
                card.setAttribute('data-listener', 'true');
                card.addEventListener('click', () => {
                    const id = card.getAttribute('data-id');
                    if (id)
                        this.showMemoryDetail(id, this.currentSearchResults);
                });
            }
        });
        // Add click handler for load more button
        const loadMoreBtn = this.resultsContainer.querySelector('#load-more-search');
        if (loadMoreBtn) {
            loadMoreBtn.addEventListener('click', () => this.loadMoreSearch());
        }
    }
    renderMemoryCard(result) {
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
    async loadRecent() {
        try {
            let memories;
            if (this.useSqlJs && this.db) {
                memories = this.getRecentLocal(20);
            }
            else {
                memories = await this.getRecentAPI(20);
            }
            this.renderRecentMemories(memories);
            this.memoryCount.textContent = `${memories.length}+ memories`;
        }
        catch (error) {
            this.recentContainer.innerHTML = `<p class="placeholder">Error loading memories</p>`;
        }
    }
    getRecentLocal(limit = 20) {
        if (!this.db)
            return [];
        const results = this.db.exec(`SELECT * FROM memories ORDER BY created_at DESC LIMIT ${limit}`);
        if (!results.length || !results[0].values.length)
            return [];
        return results[0].values.map((row) => this.rowToMemory(results[0].columns, row));
    }
    async getRecentAPI(limit = 20, type, namespace) {
        const params = new URLSearchParams({ limit: String(limit) });
        if (type)
            params.set('type', type);
        if (namespace)
            params.set('namespace', namespace);
        const response = await fetch(`${this.apiBase}/memories?${params}`);
        const data = await response.json();
        if (!data.success || !data.data) {
            throw new Error(data.error || 'Failed to load memories');
        }
        return data.data;
    }
    renderRecentMemories(memories) {
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
    async loadSessions(reset = true) {
        try {
            if (reset) {
                this.sessionsOffset = 0;
                this.sessionsHasMore = true;
                this.sessionsContainer.innerHTML = '<div class="loading"></div>';
            }
            let sessions;
            if (this.useSqlJs && this.db) {
                sessions = this.getSessionsLocal(this.sessionsPageSize, this.sessionsOffset);
            }
            else {
                sessions = await this.getSessionsAPI(this.sessionsPageSize, this.sessionsOffset);
            }
            // Check if there are more sessions
            this.sessionsHasMore = sessions.length === this.sessionsPageSize;
            this.sessionsOffset += sessions.length;
            this.renderSessions(sessions, !reset);
        }
        catch (error) {
            this.sessionsContainer.innerHTML = `<p class="placeholder">Error loading sessions</p>`;
        }
    }
    async loadMoreSessions() {
        await this.loadSessions(false);
    }
    getSessionsLocal(limit = 20, offset = 0) {
        if (!this.db)
            return [];
        const results = this.db.exec(`SELECT s.*, COUNT(m.id) as message_count
             FROM sessions s
             LEFT JOIN messages m ON s.id = m.session_id
             GROUP BY s.id
             ORDER BY s.started_at DESC
             LIMIT ${limit} OFFSET ${offset}`);
        if (!results.length || !results[0].values.length)
            return [];
        return results[0].values.map((row) => this.rowToSession(results[0].columns, row));
    }
    async getSessionsAPI(limit = 20, offset = 0) {
        const response = await fetch(`${this.apiBase}/sessions?limit=${limit}&offset=${offset}`);
        const data = await response.json();
        if (!data.success || !data.data) {
            throw new Error(data.error || 'Failed to load sessions');
        }
        return data.data;
    }
    renderSessions(sessions, append = false) {
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
            if (oldBtn)
                oldBtn.remove();
            // Append new sessions
            this.sessionsContainer.insertAdjacentHTML('beforeend', sessionsHtml + loadMoreHtml);
        }
        else {
            this.sessionsContainer.innerHTML = sessionsHtml + loadMoreHtml;
        }
        // Add click handlers for session cards
        this.sessionsContainer.querySelectorAll('.session-card').forEach((card) => {
            if (!card.hasAttribute('data-listener')) {
                card.setAttribute('data-listener', 'true');
                card.addEventListener('click', () => {
                    const id = card.getAttribute('data-id');
                    if (id)
                        this.showSessionDetail(id);
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
    async loadStats() {
        try {
            let stats;
            if (this.useSqlJs && this.db) {
                stats = this.getStatsLocal();
            }
            else {
                stats = await this.getStatsAPI();
            }
            this.renderStats(stats);
        }
        catch (error) {
            this.statsContainer.innerHTML = `<p class="placeholder">Error loading statistics</p>`;
        }
    }
    getStatsLocal() {
        if (!this.db) {
            return {
                total_memories: 0,
                memories_by_type: {},
                total_sessions: 0,
                namespaces: [],
                fts_indexed: 0,
                rag_indexed: 0,
            };
        }
        const countResult = this.db.exec('SELECT COUNT(*) FROM memories');
        const totalMemories = countResult[0]?.values[0]?.[0] || 0;
        const typeResult = this.db.exec('SELECT type, COUNT(*) FROM memories GROUP BY type');
        const memoriesByType = {};
        if (typeResult[0]) {
            typeResult[0].values.forEach((row) => {
                memoriesByType[row[0]] = row[1];
            });
        }
        const sessionResult = this.db.exec('SELECT COUNT(*) FROM sessions');
        const totalSessions = sessionResult[0]?.values[0]?.[0] || 0;
        const namespaceResult = this.db.exec('SELECT DISTINCT namespace_id FROM memories');
        const namespaces = namespaceResult[0]?.values.map((row) => row[0]) || [];
        return {
            total_memories: totalMemories,
            memories_by_type: memoriesByType,
            total_sessions: totalSessions,
            namespaces,
            fts_indexed: totalMemories,
            rag_indexed: 0,
        };
    }
    async getStatsAPI() {
        const response = await fetch(`${this.apiBase}/stats`);
        const data = await response.json();
        if (!data.success || !data.data) {
            throw new Error(data.error || 'Failed to load stats');
        }
        return data.data;
    }
    renderStats(stats) {
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
    async loadNamespaces() {
        try {
            let namespaces;
            if (this.useSqlJs && this.db) {
                // Get namespace IDs and source repos from local DB
                const result = this.db.exec('SELECT DISTINCT namespace_id, source_repo FROM memories');
                namespaces = (result[0]?.values || []).map((row) => {
                    const nsId = row[0];
                    const sourceRepo = row[1];
                    let displayName = nsId;
                    if (sourceRepo) {
                        const repoName = sourceRepo.replace(/\/$/, '').split('/').pop();
                        if (repoName)
                            displayName = repoName;
                    }
                    else if (nsId === 'global') {
                        displayName = 'global';
                    }
                    else if (nsId.startsWith('repo-')) {
                        displayName = `repo-${nsId.slice(5, 13)}...`;
                    }
                    return { id: nsId, display_name: displayName, source_repo: sourceRepo };
                });
            }
            else {
                const response = await fetch(`${this.apiBase}/namespaces`);
                const data = await response.json();
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
        }
        catch {
            // Ignore namespace loading errors
        }
    }
    // ==================== Modal ====================
    showMemoryDetail(id, results) {
        const result = results.find((r) => r.memory.id === id);
        if (!result)
            return;
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
    async showSessionDetail(id) {
        // Load session messages from API or local DB
        this.modalBody.innerHTML = '<div class="loading"></div>';
        this.modal.classList.add('active');
        try {
            const response = await fetch(`${this.apiBase}/sessions/${id}`);
            const data = await response.json();
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
        }
        catch (error) {
            this.modalBody.innerHTML = `<p class="placeholder">Error: ${error.message}</p>`;
        }
    }
    closeModal() {
        this.modal.classList.remove('active');
    }
    // ==================== Actions ====================
    async exportMemories() {
        try {
            const response = await fetch(`${this.apiBase}/export`);
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `contextfs-export-${new Date().toISOString().split('T')[0]}.json`;
            a.click();
            URL.revokeObjectURL(url);
        }
        catch (error) {
            alert(`Export failed: ${error.message}`);
        }
    }
    async syncToPostgres() {
        if (!confirm('Sync all memories to PostgreSQL?'))
            return;
        try {
            const response = await fetch(`${this.apiBase}/sync`, {
                method: 'POST',
            });
            const data = await response.json();
            if (data.success) {
                alert(`Synced ${data.data?.synced || 0} memories to PostgreSQL`);
            }
            else {
                throw new Error(data.error || 'Sync failed');
            }
        }
        catch (error) {
            alert(`Sync failed: ${error.message}`);
        }
    }
    // ==================== Tabs ====================
    switchTab(e) {
        const tab = e.target;
        const tabName = tab.getAttribute('data-tab');
        if (!tabName)
            return;
        // Update tab buttons
        document.querySelectorAll('.tab').forEach((t) => t.classList.remove('active'));
        tab.classList.add('active');
        // Update content
        document.querySelectorAll('.tab-content').forEach((c) => c.classList.remove('active'));
        document.getElementById(`${tabName}-tab`)?.classList.add('active');
    }
    // ==================== Helpers ====================
    rowToMemory(columns, row) {
        const obj = {};
        columns.forEach((col, i) => {
            obj[col] = row[i];
        });
        return {
            id: obj.id,
            content: obj.content,
            type: obj.type,
            tags: typeof obj.tags === 'string' ? JSON.parse(obj.tags) : [],
            summary: obj.summary,
            namespace_id: obj.namespace_id,
            source_file: obj.source_file,
            source_repo: obj.source_repo,
            session_id: obj.session_id,
            created_at: obj.created_at,
            updated_at: obj.updated_at,
            metadata: typeof obj.metadata === 'string' ? JSON.parse(obj.metadata) : {},
        };
    }
    rowToSearchResult(columns, row) {
        const rankIndex = columns.indexOf('rank');
        const rank = rankIndex >= 0 ? row[rankIndex] : 0;
        // Exclude rank column for memory parsing
        const memoryColumns = columns.filter((_, i) => i !== rankIndex);
        const memoryRow = row.filter((_, i) => i !== rankIndex);
        return {
            memory: this.rowToMemory(memoryColumns, memoryRow),
            score: 1.0 / (1.0 + Math.abs(rank)),
            highlights: [],
        };
    }
    rowToSession(columns, row) {
        const obj = {};
        columns.forEach((col, i) => {
            obj[col] = row[i];
        });
        return {
            id: obj.id,
            label: obj.label,
            namespace_id: obj.namespace_id,
            tool: obj.tool,
            repo_path: obj.repo_path,
            branch: obj.branch,
            started_at: obj.started_at,
            ended_at: obj.ended_at,
            summary: obj.summary,
            metadata: typeof obj.metadata === 'string' ? JSON.parse(obj.metadata) : {},
            message_count: obj.message_count,
        };
    }
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    formatDate(dateStr) {
        try {
            const date = new Date(dateStr);
            return date.toLocaleDateString('en-US', {
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit',
            });
        }
        catch {
            return dateStr;
        }
    }
}
// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new ContextFSBrowser();
});
export {};
//# sourceMappingURL=app.js.map