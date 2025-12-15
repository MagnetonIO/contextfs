# AIGNE Framework Integration Recommendations

This document outlines how [AIGNE Framework](https://github.com/AIGNE-io/aigne-framework) can integrate with ContextFS to provide a comprehensive AI memory solution.

> **Note**: This document was migrated from the original TypeScript implementation (contextfs-node).
> Code examples have been updated to Python to match the current implementation.

## Overview

AIGNE Framework provides:
- **Agent orchestration** — AIAgent, skills, and pipelines
- **AFS (Agentic File System)** — Virtual filesystem for agents with modules (history, local-fs, user-profile-memory)
- **Model abstraction** — OpenAI, Anthropic, and other providers

ContextFS provides:
- **MCP server** — Native Claude CLI/Desktop integration
- **CLI tool** — Direct memory management
- **Semantic search** — ChromaDB + sentence-transformers for RAG
- **Session import** — Claude CLI history integration

## Integration Points

### 1. ContextFS as AFS Module

Create a `contextfs-afs` package that exposes ContextFS storage through the AFS interface:

```python
from contextfs import ContextFS, MemoryType
from aigne.afs import AFSModule, AFSEntry

class ContextFSModule(AFSModule):
    name = "contextfs"
    description = "AI memory with semantic search"

    def __init__(self, project_root: str = None):
        self.ctx = ContextFS(data_dir=project_root)

    def list(self, path: str) -> dict:
        # Map memory types to paths: /fact, /decision, /procedural, etc.
        memory_type = path.split('/')[1] if '/' in path else None
        memories = self.ctx.list(type=memory_type, limit=100)
        return {
            "list": [
                {
                    "id": m.id,
                    "path": f"/{m.type}/{m.id}",
                    "content": m.content,
                    "summary": m.summary,
                    "metadata": {"tags": m.tags},
                    "created_at": m.created_at
                }
                for m in memories
            ]
        }

    def read(self, path: str) -> dict:
        memory_id = path.split('/')[-1]
        memory = self.ctx.recall(memory_id)
        return {"result": self._to_afs_entry(memory) if memory else None}

    def write(self, path: str, content: dict) -> dict:
        memory_type = path.split('/')[1]
        memory = self.ctx.save(
            content["content"],
            type=MemoryType(memory_type),
            tags=content.get("metadata", {}).get("tags"),
            summary=content.get("summary")
        )
        return {"result": self._to_afs_entry(memory)}

    def search(self, path: str, query: str) -> dict:
        results = self.ctx.search(query)
        return {"list": [self._to_afs_entry(r.memory) for r in results]}
```

**Usage with AIGNE:**
```python
from aigne import AIGNE, AIAgent
from aigne.afs import AFS
from contextfs_afs import ContextFSModule

afs = AFS()
afs.mount(ContextFSModule())

agent = AIAgent.create(
    name="assistant",
    instructions="You have access to persistent memory via /modules/contextfs",
    afs=afs
)
```

### 2. Shared Session History

Use ContextFS to persist AIGNE agent sessions for cross-tool memory:

```python
from aigne import AIGNE
from contextfs import ContextFS
import atexit

ctx = ContextFS()
aigne = AIGNE(model=model)

# Listen to agent events
@aigne.on('agent_succeed')
def on_success(input_data, output_data):
    ctx.add_message('user', str(input_data))
    ctx.add_message('assistant', str(output_data))

# Save session on completion
atexit.register(ctx.end_session)
```

### 3. Decision Memory Persistence

Store architectural decisions from AIGNE agents for retrieval by Claude CLI:

```python
# In AIGNE agent
decision_agent = AIAgent.create(
    name="architect",
    instructions="When making decisions, save them using the contextfs_save tool",
    tools=[contextfs_save_tool]
)

# Later in Claude CLI
# User: "What decisions has the architect agent made?"
# Claude searches contextfs for type=decision
```

### 4. User Profile Synchronization

Share user preferences between AIGNE's `UserProfileMemory` and ContextFS:

```python
from aigne.afs import AFS
from aigne.afs.user_profile_memory import UserProfileMemory
from contextfs import ContextFS, MemoryType
import json

# Sync AIGNE user profile to ContextFS
ctx = ContextFS()
afs = AFS()
user_profile = UserProfileMemory(context=context)
afs.mount(user_profile)

# On profile update, save to ContextFS
@user_profile.on('profile_updated')
def sync_profile(profile):
    ctx.save(
        json.dumps(profile),
        type=MemoryType.USER,
        tags=['profile', 'preferences']
    )
```

### 5. Procedural Knowledge Transfer

AIGNE agents can discover workflows that become reusable procedural memories:

```python
# Agent discovers a workflow
workflow = await workflow_agent.invoke(
    message="Figure out how to deploy this app"
)

# Save as procedural memory
ctx.save(
    '\n'.join(workflow.steps),
    type=MemoryType.PROCEDURAL,
    tags=['deployment', 'workflow'],
    summary='Deployment workflow for this project'
)

# Later, any tool (AIGNE or Claude CLI) can retrieve:
deploy_steps = ctx.search('deployment workflow')
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      User's Project                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐              ┌──────────────────┐        │
│  │   AIGNE Agents   │              │   Claude CLI     │        │
│  │                  │              │                  │        │
│  │  ┌────────────┐  │              │  MCP Protocol    │        │
│  │  │ AIAgent    │  │              │        │         │        │
│  │  └─────┬──────┘  │              └────────┼─────────┘        │
│  │        │         │                       │                   │
│  │  ┌─────▼──────┐  │                       │                   │
│  │  │    AFS     │  │                       │                   │
│  │  └─────┬──────┘  │                       │                   │
│  └────────┼─────────┘                       │                   │
│           │                                 │                   │
│  ┌────────▼─────────────────────────────────▼────────┐         │
│  │                    ContextFS                       │         │
│  │  ┌─────────────┐  ┌─────────────┐  ┌───────────┐ │         │
│  │  │ AFS Module  │  │ MCP Server  │  │    CLI    │ │         │
│  │  └──────┬──────┘  └──────┬──────┘  └─────┬─────┘ │         │
│  │         └────────────────┼───────────────┘       │         │
│  │                    ┌─────▼─────┐                  │         │
│  │                    │  Storage  │                  │         │
│  │                    │ ChromaDB  │                  │         │
│  │                    └───────────┘                  │         │
│  └───────────────────────────────────────────────────┘         │
│                                                                  │
│  ~/.contextfs/  ← Shared memory store                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Recommended Package Structure

```
contextfs/
├── src/contextfs/
│   ├── core.py           # Main ContextFS class
│   ├── rag.py            # RAG backend (ChromaDB + embeddings)
│   ├── schemas.py        # Pydantic models
│   ├── mcp_server.py     # MCP protocol server
│   └── plugins/
│       ├── claude_code.py
│       ├── gemini.py
│       └── codex.py
└── ...

aigne-framework/
├── afs/
│   └── contextfs/      # Optional: thin wrapper pointing to contextfs
└── ...
```

## Migration Path

1. **Phase 1 (Current):** ContextFS standalone, MCP-only integration
2. **Phase 2:** Create `contextfs-afs` package for AIGNE integration
3. **Phase 3:** AIGNE can optionally depend on contextfs for enhanced memory
4. **Phase 4:** Shared cloud sync for team collaboration (enterprise feature)

## Benefits

| Feature | AIGNE Only | ContextFS Only | AIGNE + ContextFS |
|---------|------------|----------------|-------------------|
| Agent orchestration | ✓ | | ✓ |
| Claude CLI memory | | ✓ | ✓ |
| Semantic search | | ✓ | ✓ |
| Session import | | ✓ | ✓ |
| Cross-tool memory | | ✓ | ✓ |
| Agent → CLI knowledge transfer | | | ✓ |

## Next Steps

1. Publish contextfs to PyPI as standalone package
2. Create `contextfs-afs` package with AIGNE integration
3. Add example in `aigne-framework/examples/afs-contextfs`
4. Document in both repos
