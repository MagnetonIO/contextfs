# ContextFS Sync Service Architecture

## Overview

The ContextFS sync service enables multi-device memory synchronization using vector clocks for conflict detection. This document describes the current implementation architecture.

## System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                         DEVICES                                  │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   Laptop        │   Desktop       │   Linux Server              │
│   (macOS)       │   (Windows)     │   (Ubuntu)                  │
├─────────────────┼─────────────────┼─────────────────────────────┤
│ ┌─────────────┐ │ ┌─────────────┐ │ ┌─────────────┐             │
│ │  ContextFS  │ │ │  ContextFS  │ │ │  ContextFS  │             │
│ │    Core     │ │ │    Core     │ │ │    Core     │             │
│ └──────┬──────┘ │ └──────┬──────┘ │ └──────┬──────┘             │
│        │        │        │        │        │                     │
│ ┌──────┴──────┐ │ ┌──────┴──────┐ │ ┌──────┴──────┐             │
│ │   SQLite    │ │ │   SQLite    │ │ │   SQLite    │             │
│ │  + ChromaDB │ │ │  + ChromaDB │ │ │  + ChromaDB │             │
│ └─────────────┘ │ └─────────────┘ │ └─────────────┘             │
└────────┬────────┴────────┬────────┴────────┬────────────────────┘
         │                 │                 │
         │    Push/Pull    │    Push/Pull    │    Push/Pull
         │    (HTTP)       │    (HTTP)       │    (HTTP)
         ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      SYNC SERVER                                 │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    FastAPI Application                   │    │
│  │                      (Port 8766)                         │    │
│  ├─────────────────────────────────────────────────────────┤    │
│  │  /api/sync/register  │  Device registration             │    │
│  │  /api/sync/push      │  Push local changes              │    │
│  │  /api/sync/pull      │  Pull server changes             │    │
│  │  /api/sync/status    │  Get sync status                 │    │
│  └──────────────────────┬──────────────────────────────────┘    │
│                         │                                        │
│  ┌──────────────────────┴──────────────────────────────────┐    │
│  │              PostgreSQL + pgvector                       │    │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐    │    │
│  │  │  memories   │ │  sessions   │ │  memory_edges   │    │    │
│  │  │  (synced)   │ │  (synced)   │ │    (synced)     │    │    │
│  │  └─────────────┘ └─────────────┘ └─────────────────┘    │    │
│  │  ┌─────────────┐ ┌─────────────┐                        │    │
│  │  │   devices   │ │ sync_state  │                        │    │
│  │  └─────────────┘ └─────────────┘                        │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Key Files

| File | Purpose |
|------|---------|
| `src/contextfs/sync/vector_clock.py` | VectorClock and DeviceTracker classes |
| `src/contextfs/sync/protocol.py` | Pydantic models for sync protocol |
| `src/contextfs/sync/client.py` | SyncClient for device-side operations |
| `src/contextfs/sync/cli.py` | CLI commands (`contextfs sync ...`) |
| `src/contextfs/sync/path_resolver.py` | Cross-machine path normalization |
| `service/api/sync_routes.py` | Server-side FastAPI endpoints |
| `service/db/models.py` | SQLAlchemy models for PostgreSQL |

## Vector Clock Implementation

### Data Structure

```python
class VectorClock(BaseModel):
    clock: dict[str, int]  # {device_id: counter}
```

Example:
```json
{"laptop-abc123": 5, "desktop-def456": 3, "server-xyz789": 1}
```

### Operations

| Operation | Description |
|-----------|-------------|
| `increment(device_id)` | Increment counter for device |
| `merge(other)` | Take max of each component |
| `happens_before(other)` | Check if self causally precedes other |
| `concurrent_with(other)` | Check if clocks are concurrent (conflict) |
| `dominates(other)` | Check if self causally follows other |
| `prune(active_devices, max_devices)` | Limit clock size |

### Happens-Before Relation

```
VC1 < VC2  iff  ∀d: VC1[d] ≤ VC2[d]  AND  ∃d: VC1[d] < VC2[d]
```

### Conflict Detection

| Client Clock | Server Clock | Result |
|--------------|--------------|--------|
| `{A:2, B:1}` | `{A:1, B:1}` | **Accept** (server behind) |
| `{A:1, B:1}` | `{A:2, B:1}` | **Reject** (client behind/stale) |
| `{A:2, B:1}` | `{A:1, B:2}` | **Conflict** (concurrent) |
| `{A:2, B:2}` | `{A:2, B:2}` | **Accept** (equal) |

## Sync Protocol

### Push Flow

```
Client                                    Server
  │                                         │
  │  1. Query local memories (since last sync)
  │                                         │
  │  2. Increment vector clock for each     │
  │                                         │
  │  3. Extract embeddings from ChromaDB    │
  │                                         │
  │  POST /api/sync/push ─────────────────► │
  │  {device_id, memories[], sessions[]}    │
  │                                         │
  │                        4. For each memory:
  │                           - Compare clocks
  │                           - Accept/Reject/Conflict
  │                           - Store embedding
  │                                         │
  │ ◄───────────────────── Response ────────│
  │  {accepted, rejected, conflicts[]}      │
  │                                         │
  │  5. Update local vector clocks          │
  │                                         │
```

### Pull Flow

```
Client                                    Server
  │                                         │
  │  POST /api/sync/pull ─────────────────► │
  │  {device_id, since_timestamp}           │
  │                                         │
  │                        1. Query memories
  │                           WHERE updated_at > since
  │                                         │
  │                        2. Include embeddings
  │                                         │
  │ ◄───────────────────── Response ────────│
  │  {memories[], sessions[], has_more}     │
  │                                         │
  │  3. Batch insert to SQLite (skip_rag)   │
  │                                         │
  │  4. Insert embeddings to ChromaDB       │
  │                                         │
  │  5. Update last_sync timestamp          │
  │                                         │
```

## Data Models

### SyncedMemory (Protocol)

```python
class SyncedMemory(SyncableEntity):
    # Core content
    content: str
    type: str = "fact"
    tags: list[str]
    summary: str | None

    # Namespace
    namespace_id: str = "global"

    # Portable source (cross-machine)
    repo_url: str | None        # git@github.com:user/repo.git
    repo_name: str | None       # Human-readable name
    relative_path: str | None   # Path from repo root

    # Legacy fields
    source_file: str | None
    source_repo: str | None
    source_tool: str | None

    # Sync metadata (inherited)
    vector_clock: dict[str, int]
    content_hash: str | None
    deleted_at: datetime | None
    last_modified_by: str | None

    # Embedding (synced!)
    embedding: list[float] | None  # 384-dim vector
```

### SyncedMemoryModel (PostgreSQL)

```python
class SyncedMemoryModel(Base):
    __tablename__ = "memories"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    content: Mapped[str]
    type: Mapped[str]
    tags: Mapped[list[str]] = mapped_column(ARRAY(Text))

    # Sync fields
    vector_clock: Mapped[dict] = mapped_column(JSONB)
    content_hash: Mapped[str | None]
    deleted_at: Mapped[datetime | None]
    last_modified_by: Mapped[str | None]

    # Embedding (pgvector)
    embedding = mapped_column(Vector(384), nullable=True)
```

## Embedding Synchronization

### Problem

Traditional approaches require either:
1. **Centralized queries** - Network dependency for every search
2. **Local recomputation** - Expensive (10-50ms/memory on CPU)

### Solution

Sync embeddings alongside content:

```
Push: ChromaDB → Extract → HTTP → PostgreSQL (pgvector)
Pull: PostgreSQL → HTTP → Insert → ChromaDB (no recompute!)
```

### Implementation

**Push (client.py:_get_embeddings_from_chroma)**
```python
def _get_embeddings_from_chroma(self, memory_ids: list[str]) -> dict[str, list[float]]:
    collection = self.ctx.rag._collection
    result = collection.get(ids=memory_ids, include=["embeddings"])
    return {id: list(emb) for id, emb in zip(result["ids"], result["embeddings"])}
```

**Pull (client.py:_add_embeddings_to_chroma)**
```python
def _add_embeddings_to_chroma(self, embeddings: list[tuple]):
    collection = self.ctx.rag._collection
    collection.upsert(ids=ids, embeddings=vectors, documents=documents)
```

## Device Management

### Device Registration

```python
class DeviceRegistration(BaseModel):
    device_id: str       # laptop-abc123 (auto-generated)
    device_name: str     # "My Laptop"
    platform: str        # darwin, linux, windows
    client_version: str  # "0.1.0"
```

### Device ID Generation

```python
def _get_or_create_device_id(self) -> str:
    hostname = socket.gethostname()
    mac = uuid.getnode()
    return f"{hostname}-{mac:012x}"[:32]
```

Stored in `~/.contextfs/device_id`

### Sync State Persistence

Stored in SQLite `sync_state` table:

| Column | Type | Description |
|--------|------|-------------|
| device_id | TEXT | Primary key |
| server_url | TEXT | Sync server URL |
| last_sync_at | TIMESTAMP | Last successful sync |
| last_push_at | TIMESTAMP | Last push |
| last_pull_at | TIMESTAMP | Last pull |
| device_tracker | TEXT (JSON) | DeviceTracker data |

## Path Normalization

### Problem

Absolute paths differ across machines:
- macOS: `/Users/mlong/code/myrepo/src/file.py`
- Linux: `/home/mlong/code/myrepo/src/file.py`
- Windows: `C:\Users\mlong\code\myrepo\src\file.py`

### Solution: Portable Paths

Store repository URL + relative path:

```python
class PortablePath:
    repo_url: str        # git@github.com:user/repo.git
    repo_name: str       # myrepo
    relative_path: str   # src/file.py
```

### PathResolver

```python
def normalize(self, absolute_path: str) -> PortablePath:
    """Convert absolute path to portable format."""
    repo_root = find_git_root(absolute_path)
    repo_url = get_git_remote(repo_root)
    relative = os.path.relpath(absolute_path, repo_root)
    return PortablePath(repo_url=repo_url, relative_path=relative)

def resolve(self, portable: PortablePath) -> Path | None:
    """Convert portable path to local absolute path."""
    # Look up local clone of repo_url
    local_root = find_local_clone(portable.repo_url)
    return local_root / portable.relative_path
```

## CLI Commands

```bash
# Register device with sync server
contextfs sync register --server http://localhost:8766 --name "My Device"

# Push local changes
contextfs sync push --server http://localhost:8766
contextfs sync push --server http://localhost:8766 --all  # Push ALL memories

# Pull from server
contextfs sync pull --server http://localhost:8766
contextfs sync pull --server http://localhost:8766 --all  # Initial sync

# Full bidirectional sync
contextfs sync all --server http://localhost:8766

# Check status
contextfs sync status --server http://localhost:8766

# Run sync daemon
contextfs sync daemon --server http://localhost:8766 --interval 300
```

## Deployment

### Docker Compose

```yaml
# docker-compose.sync.yml
services:
  sync-postgres:
    image: pgvector/pgvector:pg15
    environment:
      POSTGRES_DB: contextfs_sync
      POSTGRES_USER: contextfs
      POSTGRES_PASSWORD: contextfs
    ports:
      - "5432:5432"

  sync-server:
    build: .
    environment:
      DATABASE_URL: postgresql+asyncpg://contextfs:contextfs@sync-postgres/contextfs_sync
    ports:
      - "8766:8766"
    depends_on:
      - sync-postgres
```

### Start Services

```bash
# Start infrastructure
docker-compose -f docker-compose.sync.yml up -d sync-postgres

# Run server locally (development)
python -m service.api.main

# Or run everything in Docker
docker-compose -f docker-compose.sync.yml up -d
```

## Conflict Resolution

### Current Strategy

Conflicts are returned to the client for manual resolution:

```python
class ConflictInfo(BaseModel):
    entity_id: str
    entity_type: str  # "memory", "session", "edge"
    client_clock: dict[str, int]
    server_clock: dict[str, int]
    client_content: str | None
    server_content: str | None
    client_updated_at: datetime
    server_updated_at: datetime
```

### Future Options

1. **Last-Write-Wins (LWW)** - Use timestamp to auto-resolve
2. **LLM-Powered Merge** - Use AI to intelligently merge versions
3. **CRDT-Style** - Automatic merge for compatible operations (tag unions)

## Performance Characteristics

| Operation | 1K memories | 10K memories |
|-----------|-------------|--------------|
| Push (with embeddings) | ~350ms | ~3.7s |
| Pull (with embeddings) | ~390ms | ~3.8s |
| Incremental sync | ~50ms | ~55ms |

### Optimizations

1. **Batch Save** - Single transaction for bulk inserts
2. **Skip RAG on Pull** - Embeddings inserted directly, no recompute
3. **Pagination** - Large syncs paginated (1000 items/page)
4. **Content Hashing** - Skip identical content (deduplication)

## Security Considerations

### Current State

- Device ID-based identification (no authentication)
- Plain HTTP (use HTTPS in production)

### Recommended for Production

1. API key or OAuth authentication
2. TLS for transport security
3. Server-side encryption for sensitive memories
4. Rate limiting per device

## Future Enhancements

1. **CRDT Integration** - Automatic conflict resolution for compatible ops
2. **Selective Sync** - Namespace-based sync policies
3. **Federated Architecture** - Multiple sync servers
4. **End-to-End Encryption** - Client-side encryption with key sharing
5. **WebSocket Push** - Real-time sync notifications
