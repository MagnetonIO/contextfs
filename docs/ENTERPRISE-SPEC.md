# ContextFS: Enterprise AI Memory Layer as a Service

## Product Vision

**ContextFS** is the **Supabase for AI Context** — an open-source, enterprise-grade memory layer that gives every AI conversation persistent, searchable, team-shareable context.

> "Git for your AI conversations — project-local context that follows your code"

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ContextFS Platform                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Context   │  │   Memory    │  │    Sync     │  │   Search    │        │
│  │   Capture   │  │   Store     │  │   Engine    │  │   & RAG     │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                │                │
│  ┌──────┴────────────────┴────────────────┴────────────────┴──────┐        │
│  │                     Core Engine (Rust/TypeScript)               │        │
│  │  • Session Management    • Token Optimization                   │        │
│  │  • Memory Consolidation  • Embedding Pipeline                   │        │
│  │  • Context Construction  • Event Processing                     │        │
│  └─────────────────────────────┬───────────────────────────────────┘        │
│                                │                                            │
│  ┌─────────────────────────────┴───────────────────────────────────┐        │
│  │                      Storage Layer                               │        │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │        │
│  │  │ Postgres │  │  Vector  │  │  Object  │  │  Graph   │        │        │
│  │  │  + JSONB │  │   Store  │  │  Storage │  │   Store  │        │        │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  CLI │ SDK │ REST API │ WebSocket │ MCP Server │ IDE Extensions            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Platform Components (Supabase-Style)

### 1. Context Database (≈ Supabase Database)

**Purpose**: Persistent storage for all AI interactions, memories, and metadata.

| Feature | Description |
|---------|-------------|
| **PostgreSQL Core** | Full Postgres with JSONB for flexible context schemas |
| **pgvector Extension** | Native vector embeddings for semantic search |
| **Temporal Tables** | Automatic versioning of all context changes |
| **Row-Level Security** | Fine-grained access control per user/project/team |
| **Direct SQL Access** | Query your context data with full SQL power |

**Schema Design**:
```sql
-- Core tables
sessions          -- Conversation sessions with metadata
memories          -- Extracted facts, decisions, learnings
embeddings        -- Vector representations for search
projects          -- Project-level context boundaries
users             -- User profiles and preferences
audit_logs        -- Compliance and debugging trail

-- Memory taxonomy tables
memory_fact       -- Atomic factual statements (key-value)
memory_episodic   -- Session summaries
memory_procedural -- How-to knowledge, workflows
memory_user       -- User preferences, personalization
```

### 2. Context Capture (≈ Supabase Realtime)

**Purpose**: Real-time ingestion of AI conversations from any tool.

| Feature | Description |
|---------|-------------|
| **WebSocket Streaming** | Live capture of ongoing conversations |
| **CLI Wrappers** | `ctx wrap <tool>` pipes any CLI tool through capture |
| **IDE Hooks** | VS Code, JetBrains extensions for inline capture |
| **Webhook Ingestion** | HTTP endpoints for custom integrations |
| **MCP Server** | Model Context Protocol for native LLM integration |

**Supported Integrations**:
- Claude CLI / Claude Code
- Aider
- Cursor
- Continue
- GitHub Copilot Chat
- OpenAI API (direct)
- Any stdin/stdout tool

### 3. Context Search (≈ Supabase Vector + Full-Text)

**Purpose**: Find relevant context across all historical interactions.

| Feature | Description |
|---------|-------------|
| **Semantic Search** | Vector similarity using embeddings |
| **Hybrid Search** | Combined vector + keyword (BM25) ranking |
| **Faceted Search** | Filter by project, date, user, tags |
| **GraphRAG** | Knowledge graph traversal for related context |
| **Temporal Decay** | Recent context weighted higher |

**Query Interface**:
```typescript
// Semantic search
ctx.search("authentication decisions", {
  project: "my-app",
  limit: 10,
  threshold: 0.7
});

// Structured query
ctx.query({
  type: "memory_fact",
  tags: ["architecture", "auth"],
  since: "2025-01-01"
});
```

### 4. Context Sync (≈ Supabase Realtime Sync)

**Purpose**: Synchronize context across devices, team members, and environments.

| Feature | Description |
|---------|-------------|
| **Project Sync** | `.contextfs/` syncs like git (conflict resolution) |
| **Team Sharing** | Share context within organization boundaries |
| **Selective Sync** | Choose what context to share vs. keep private |
| **Offline Support** | Local-first with background sync |
| **Conflict Resolution** | CRDT-based merge for concurrent edits |

### 5. Context Functions (≈ Supabase Edge Functions)

**Purpose**: Serverless compute for context processing pipelines.

| Feature | Description |
|---------|-------------|
| **Deno Runtime** | TypeScript/JavaScript edge functions |
| **Triggers** | Fire on context events (new session, memory update) |
| **Scheduled Jobs** | Cron-based consolidation, cleanup, reports |
| **Custom Extractors** | Define domain-specific memory extraction |
| **Webhooks** | Outbound notifications to external systems |

**Example Function**:
```typescript
// Auto-extract architecture decisions
export async function onSessionComplete(session: Session) {
  const decisions = await extractDecisions(session.content);
  for (const decision of decisions) {
    await ctx.memories.create({
      type: "architecture_decision",
      content: decision,
      tags: ["adr", session.project],
    });
  }
}
```

### 6. Context Auth (≈ Supabase Auth)

**Purpose**: Identity and access management for context data.

| Feature | Description |
|---------|-------------|
| **API Keys** | Project-scoped keys for CLI/SDK access |
| **OAuth/OIDC** | SSO via Google, GitHub, Okta, Azure AD |
| **SAML 2.0** | Enterprise SSO integration |
| **RBAC** | Roles: Owner, Admin, Member, Viewer |
| **Project Isolation** | Strict boundaries between project contexts |

### 7. Context Studio (≈ Supabase Studio)

**Purpose**: Web dashboard for managing context, teams, and settings.

| Feature | Description |
|---------|-------------|
| **Session Browser** | View, search, annotate past sessions |
| **Memory Explorer** | Browse extracted memories by type |
| **Analytics Dashboard** | Usage metrics, token costs, trends |
| **Team Management** | Invite members, assign roles |
| **Settings & Config** | API keys, integrations, preferences |

---

## Pricing Tiers (Supabase Model)

### Free Tier — $0/month
*For individual developers and experimentation*

| Resource | Limit |
|----------|-------|
| Projects | 2 |
| Sessions/month | 500 |
| Memory storage | 100MB |
| Vector storage | 50MB |
| Search queries/month | 1,000 |
| Embedding operations | 10,000/month |
| Team members | 1 (solo) |
| Retention | 30 days |
| Support | Community |

**Includes**:
- CLI + SDK access
- Basic semantic search
- Local-first mode
- Community Discord

**Limitations**:
- Projects pause after 7 days inactivity
- No SSO
- No audit logs
- Rate limited API

---

### Pro Tier — $25/month
*For serious individual developers and small teams*

| Resource | Limit |
|----------|-------|
| Projects | 10 |
| Sessions/month | 5,000 |
| Memory storage | 2GB |
| Vector storage | 500MB |
| Search queries/month | 50,000 |
| Embedding operations | 100,000/month |
| Team members | 5 |
| Retention | 1 year |
| Support | Email (48h) |

**Includes everything in Free, plus**:
- No project pausing
- Daily backups
- Spend caps (prevent overages)
- $10 compute credit included
- Custom embedding models
- Priority search indexing
- Basic analytics

**Usage-Based Add-ons**:
| Resource | Price |
|----------|-------|
| Additional storage | $0.10/GB/month |
| Additional embeddings | $0.0001/operation |
| Additional search queries | $0.001/query |
| Additional team seats | $10/seat/month |

---

### Team Tier — $299/month
*For teams requiring collaboration and compliance*

| Resource | Limit |
|----------|-------|
| Projects | 50 |
| Sessions/month | 50,000 |
| Memory storage | 20GB |
| Vector storage | 5GB |
| Search queries/month | 500,000 |
| Embedding operations | 1M/month |
| Team members | 20 |
| Retention | 3 years |
| Support | Email (24h) + Chat |

**Includes everything in Pro, plus**:
- **SSO/SAML** — Okta, Azure AD, Google Workspace
- **Audit Logs** — 90-day retention, exportable
- **SOC 2 Type II** report access
- **RBAC** — Fine-grained roles and permissions
- **Team Workspaces** — Isolated team boundaries
- **Context Sharing** — Controlled cross-project sharing
- **Advanced Analytics** — Usage by user, project, tool
- **Webhooks** — Real-time event notifications
- **SLA** — 99.9% uptime guarantee
- **Dedicated Slack channel**

---

### Enterprise Tier — Custom Pricing
*For organizations with advanced security and scale requirements*

**Typical starting point**: $2,000-5,000/month (varies by scale)

**Includes everything in Team, plus**:

| Category | Features |
|----------|----------|
| **Security** | HIPAA BAA, custom encryption keys (BYOK), VPC peering, IP allowlisting |
| **Compliance** | SOC 2 Type II, GDPR DPA, custom security reviews, penetration test reports |
| **Scale** | Unlimited projects, custom quotas, dedicated compute, read replicas |
| **Deployment** | Single-tenant option, BYO cloud (your AWS/GCP account), air-gapped |
| **Support** | Dedicated CSM, 4-hour SLA, custom training, architecture review |
| **Integration** | Custom SSO/IdP, SCIM provisioning, custom webhooks, API priority |
| **Data** | Unlimited retention, custom backup schedules, disaster recovery |
| **Governance** | Admin dashboard, org-wide policies, usage quotas per team |

**Enterprise Add-ons**:
| Add-on | Description |
|--------|-------------|
| **Dedicated Instance** | Isolated compute, no noisy neighbors |
| **Private Link** | AWS PrivateLink / GCP Private Service Connect |
| **Custom Embedding Models** | Fine-tuned models on your domain data |
| **On-Premise Connector** | Sync with air-gapped systems |
| **Professional Services** | Implementation, migration, training |

---

## Cloud Infrastructure (GCP/AWS)

### Primary: Google Cloud Platform (GCP)

**Rationale**: Cost-effective, strong AI/ML ecosystem, Supabase-comparable.

```
┌─────────────────────────────────────────────────────────────────┐
│                    GCP Infrastructure                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐    ┌─────────────────┐                     │
│  │  Cloud Run      │    │  Cloud Functions│                     │
│  │  (API + Workers)│    │  (Event Triggers)│                    │
│  └────────┬────────┘    └────────┬────────┘                     │
│           │                      │                               │
│  ┌────────┴──────────────────────┴────────┐                     │
│  │           Cloud Load Balancer           │                     │
│  └────────────────────┬────────────────────┘                     │
│                       │                                          │
│  ┌────────────────────┴────────────────────┐                     │
│  │              Cloud Armor (WAF)           │                     │
│  └────────────────────┬────────────────────┘                     │
│                       │                                          │
│  ┌──────────┬─────────┴──────────┬──────────┐                   │
│  │          │                    │          │                    │
│  ▼          ▼                    ▼          ▼                    │
│ ┌────┐   ┌─────────┐   ┌──────────┐   ┌─────────┐              │
│ │Cloud│   │Cloud SQL│   │ AlloyDB  │   │ Cloud   │              │
│ │ CDN │   │(Postgres)│  │ (Vector) │   │ Storage │              │
│ └────┘   └─────────┘   └──────────┘   └─────────┘              │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  Pub/Sub    │  │  Memorystore│  │  Secret     │             │
│  │  (Events)   │  │  (Redis)    │  │  Manager    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                  │
│  ┌─────────────────────────────────────────────────┐            │
│  │          Vertex AI (Embeddings + LLM)           │            │
│  └─────────────────────────────────────────────────┘            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Component Mapping**:

| Component | GCP Service | Purpose |
|-----------|-------------|---------|
| API Server | Cloud Run | Stateless API, auto-scaling |
| Workers | Cloud Run Jobs | Background processing |
| Event Functions | Cloud Functions | Triggers, webhooks |
| Primary DB | Cloud SQL (Postgres 15) | Context metadata, memories |
| Vector DB | AlloyDB + pgvector | Semantic search |
| Cache | Memorystore (Redis) | Session cache, rate limiting |
| Object Storage | Cloud Storage | Large context exports, backups |
| CDN | Cloud CDN | Static assets, SDK distribution |
| Events | Pub/Sub | Async event processing |
| Secrets | Secret Manager | API keys, credentials |
| Embeddings | Vertex AI | Text embeddings API |
| Auth | Firebase Auth / Identity Platform | User authentication |
| Monitoring | Cloud Monitoring + Logging | Observability |

### Alternative: Amazon Web Services (AWS)

```
┌─────────────────────────────────────────────────────────────────┐
│                    AWS Infrastructure                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  API: ECS Fargate / Lambda          Events: EventBridge          │
│  DB: Aurora Postgres + pgvector     Cache: ElastiCache (Redis)   │
│  Storage: S3                        CDN: CloudFront              │
│  Auth: Cognito                      Secrets: Secrets Manager     │
│  Embeddings: Bedrock                Monitoring: CloudWatch       │
│  WAF: AWS WAF                       VPC: Private subnets         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Multi-Region Strategy

| Tier | Regions | Strategy |
|------|---------|----------|
| Free/Pro | Single (us-central1) | Cost optimization |
| Team | 3 regions | Active-passive failover |
| Enterprise | 6+ regions | Active-active, data residency |

**Initial Regions**:
1. `us-central1` (Iowa) — Primary
2. `europe-west1` (Belgium) — EU data residency
3. `asia-east1` (Taiwan) — APAC coverage

---

## API & SDK Specification

### REST API

```
Base URL: https://api.contextfs.dev/v1

Authentication:
  Header: Authorization: Bearer <api_key>

Endpoints:

# Sessions
POST   /sessions                    Create new session
GET    /sessions                    List sessions
GET    /sessions/:id                Get session details
PUT    /sessions/:id                Update session
DELETE /sessions/:id                Delete session
POST   /sessions/:id/append         Append to session

# Memories
POST   /memories                    Create memory
GET    /memories                    List memories
GET    /memories/:id                Get memory
PUT    /memories/:id                Update memory
DELETE /memories/:id                Delete memory

# Search
POST   /search                      Semantic search
POST   /search/hybrid               Hybrid search
POST   /search/graph                Graph traversal

# Projects
POST   /projects                    Create project
GET    /projects                    List projects
GET    /projects/:id                Get project
PUT    /projects/:id                Update project
DELETE /projects/:id                Delete project

# Sync
POST   /sync/push                   Push local changes
POST   /sync/pull                   Pull remote changes
GET    /sync/status                 Get sync status

# Admin (Team/Enterprise)
GET    /admin/audit-logs            Get audit logs
GET    /admin/usage                 Get usage metrics
POST   /admin/users                 Invite user
DELETE /admin/users/:id             Remove user
```

### TypeScript SDK

```typescript
import { ContextFS } from '@contextfs/sdk';

// Initialize
const ctx = new ContextFS({
  apiKey: process.env.CONTEXTFS_API_KEY,
  project: 'my-project',
});

// Capture session
const session = await ctx.sessions.create({
  tool: 'claude-cli',
  metadata: { branch: 'feature/auth' },
});

await session.append({
  role: 'user',
  content: 'How should I implement JWT auth?',
});

await session.append({
  role: 'assistant',
  content: 'For JWT auth, I recommend...',
});

await session.complete();

// Search context
const results = await ctx.search('JWT authentication', {
  type: 'memory_fact',
  limit: 5,
});

// Create memory
await ctx.memories.create({
  type: 'fact',
  content: 'We use RS256 for JWT signing',
  tags: ['auth', 'security'],
});

// Sync
await ctx.sync.push();
```

### CLI Interface

```bash
# Installation
npm install -g @contextfs/cli

# Authentication
ctx auth login
ctx auth logout
ctx auth status

# Project setup
ctx init                           # Initialize .contextfs/
ctx link <project-id>              # Link to cloud project
ctx status                         # Show sync status

# Capture
ctx wrap <command>                 # Wrap any CLI tool
ctx capture --tool claude          # Start capture session
ctx append "user message"          # Append to current session
ctx complete                       # End current session

# Search
ctx search "query"                 # Semantic search
ctx search --type fact "query"     # Filter by memory type
ctx search --since 7d "query"      # Filter by time

# Memory
ctx memory add "fact content"      # Add memory
ctx memory list                    # List memories
ctx memory show <id>               # Show memory details

# Sync
ctx push                           # Push to cloud
ctx pull                           # Pull from cloud
ctx sync                           # Bidirectional sync

# Team (Team/Enterprise)
ctx team invite user@example.com
ctx team list
ctx team remove <user-id>

# Admin (Enterprise)
ctx admin audit --since 30d
ctx admin usage --format json
ctx admin export --project <id>
```

### MCP Server

```json
{
  "mcpServers": {
    "contextfs": {
      "command": "npx",
      "args": ["@contextfs/mcp-server"],
      "env": {
        "CONTEXTFS_API_KEY": "<your-api-key>"
      }
    }
  }
}
```

**Available Tools**:
- `contextfs_search` — Search historical context
- `contextfs_remember` — Store a memory
- `contextfs_recall` — Recall specific memory by ID
- `contextfs_list_sessions` — List recent sessions
- `contextfs_get_session` — Get session details

---

## Enterprise Features Deep Dive

### 1. Single Sign-On (SSO)

**Supported Providers**:
- SAML 2.0 (Okta, OneLogin, PingFederate)
- OIDC (Azure AD, Google Workspace, Auth0)
- Custom IdP via SAML/OIDC

**Configuration**:
```yaml
# Organization SSO settings
sso:
  enabled: true
  provider: saml
  metadata_url: https://okta.example.com/metadata.xml
  attribute_mapping:
    email: user.email
    name: user.displayName
    groups: user.groups
  auto_provision: true
  default_role: member
```

### 2. Audit Logging

**Captured Events**:
| Event | Data Captured |
|-------|---------------|
| `session.created` | User, project, tool, timestamp |
| `session.completed` | Session ID, duration, token count |
| `memory.created` | Memory type, content hash, user |
| `memory.deleted` | Memory ID, user, reason |
| `search.executed` | Query (hashed), user, results count |
| `user.invited` | Inviter, invitee, role |
| `user.removed` | Admin, user, reason |
| `settings.changed` | Setting, old value, new value, user |
| `api_key.created` | Key prefix, user, scope |
| `export.requested` | User, scope, format |

**Query Interface**:
```bash
# CLI
ctx admin audit --since 30d --user alice@company.com --format json

# API
GET /admin/audit-logs?since=2025-01-01&actor=alice@company.com
```

### 3. Role-Based Access Control (RBAC)

| Role | Permissions |
|------|-------------|
| **Owner** | Full access, billing, delete org |
| **Admin** | Manage users, projects, settings |
| **Member** | Create/edit sessions, memories, search |
| **Viewer** | Read-only access to shared context |

**Custom Roles (Enterprise)**:
```yaml
roles:
  - name: architect
    permissions:
      - sessions:read
      - sessions:create
      - memories:*
      - search:*
      - projects:read
    projects: ["core-platform", "api-gateway"]
```

### 4. Data Governance

**Retention Policies**:
```yaml
retention:
  sessions:
    default: 365d
    pii_detected: 90d
    marked_sensitive: 30d
  memories:
    default: unlimited
    temporary: 7d
  audit_logs:
    default: 90d
    compliance: 7y
```

**Data Classification**:
- Automatic PII detection (email, phone, SSN patterns)
- Manual sensitivity tagging
- Encryption at rest with customer-managed keys (BYOK)

### 5. Compliance Certifications

| Certification | Tier | Status |
|---------------|------|--------|
| SOC 2 Type II | Team+ | Available |
| GDPR | All | Compliant |
| HIPAA | Enterprise | BAA available |
| ISO 27001 | Enterprise | Roadmap |
| FedRAMP | Enterprise | Roadmap |

---

## Usage-Based Billing Model

### Billable Dimensions

| Dimension | Unit | Free | Pro | Team | Enterprise |
|-----------|------|------|-----|------|------------|
| Sessions | per session | 500/mo | 5K/mo | 50K/mo | Custom |
| Storage | GB-month | 100MB | 2GB | 20GB | Custom |
| Embeddings | per operation | 10K/mo | 100K/mo | 1M/mo | Custom |
| Search | per query | 1K/mo | 50K/mo | 500K/mo | Custom |
| Sync | per operation | 1K/mo | 10K/mo | 100K/mo | Custom |
| Egress | GB transferred | 1GB | 10GB | 100GB | Custom |

### Overage Pricing (Pro/Team)

| Dimension | Overage Rate |
|-----------|--------------|
| Sessions | $0.005/session |
| Storage | $0.10/GB/month |
| Embeddings | $0.0001/operation |
| Search | $0.001/query |
| Sync | $0.0005/operation |
| Egress | $0.05/GB |

### Spend Controls

```yaml
# Project-level spend cap
spend_cap:
  enabled: true
  monthly_limit: 100.00  # USD
  alerts:
    - threshold: 50%
      notify: [email]
    - threshold: 80%
      notify: [email, slack]
    - threshold: 100%
      action: pause_overages
```

---

## Deployment Options

### 1. ContextFS Cloud (Default)
- Fully managed, multi-tenant
- Auto-scaling, zero maintenance
- Available: Free, Pro, Team, Enterprise

### 2. Dedicated Cloud (Enterprise)
- Single-tenant in ContextFS infrastructure
- Dedicated compute, isolated network
- Custom regions, compliance requirements

### 3. BYO Cloud (Enterprise)
- Deployed in customer's AWS/GCP account
- Customer controls data residency
- ContextFS manages software updates
- Customer manages infrastructure costs

### 4. Self-Hosted (Open Source)
- Deploy anywhere (Docker, K8s)
- Community support only
- No SLA, no managed updates
- Ideal for air-gapped environments

---

## Competitive Positioning

| Feature | ContextFS | Mem0 | Zep | Letta |
|---------|-----------|------|-----|-------|
| **Focus** | Developer context | AI memory | Agent memory | Agent memory |
| **Local-first** | ✅ | ❌ | ❌ | ❌ |
| **Git integration** | ✅ | ❌ | ❌ | ❌ |
| **CLI wrappers** | ✅ | ❌ | ❌ | ❌ |
| **IDE extensions** | ✅ | ❌ | ❌ | ❌ |
| **Team sharing** | ✅ | ✅ | ✅ | ❌ |
| **Self-hosted** | ✅ | ✅ | ✅ | ✅ |
| **Enterprise SSO** | ✅ | ✅ | ❌ | ❌ |
| **Audit logs** | ✅ | ✅ | ❌ | ❌ |
| **Open source** | ✅ | ✅ | ✅ | ✅ |

**Differentiator**: ContextFS is the only solution designed specifically for **developer workflows** with native CLI tool wrapping, git-trackable local storage, and IDE integration.

---

## Go-to-Market: Enterprise Sales Motion

### Target Segments

1. **Consulting Firms (10-100 engineers)**
   - Pain: Knowledge lost between client engagements
   - Value: Capture institutional knowledge, faster onboarding

2. **Platform Engineering Teams**
   - Pain: AI tool sprawl, no governance
   - Value: Unified context layer, audit trail

3. **Regulated Industries (Finance, Healthcare)**
   - Pain: Compliance requirements for AI usage
   - Value: Audit logs, data governance, HIPAA/SOC 2

### Sales Playbook

| Stage | Action |
|-------|--------|
| **Awareness** | Content marketing, Show HN, dev conferences |
| **Interest** | Free tier adoption, community engagement |
| **Evaluation** | Pro tier trial, technical POC |
| **Decision** | Team tier pilot (2-4 weeks) |
| **Purchase** | Enterprise contract negotiation |
| **Expansion** | Land-and-expand within organization |

### Pricing Psychology

- **Free → Pro**: Remove friction, prove value ($25 is "no-brainer")
- **Pro → Team**: SSO/compliance are hard requirements, not nice-to-haves
- **Team → Enterprise**: Custom needs justify custom pricing

---

## Implementation Roadmap

### Phase 1: Core Platform (Months 1-2)
- [ ] Cloud infrastructure (GCP)
- [ ] Core API (sessions, memories, search)
- [ ] CLI v1.0
- [ ] TypeScript SDK
- [ ] Basic web dashboard
- [ ] Free + Pro tiers

### Phase 2: Team Features (Months 3-4)
- [ ] Team workspaces
- [ ] SSO (SAML/OIDC)
- [ ] Audit logging
- [ ] RBAC
- [ ] Team tier launch

### Phase 3: Enterprise (Months 5-6)
- [ ] Dedicated instances
- [ ] BYO cloud deployment
- [ ] Advanced compliance (SOC 2, HIPAA)
- [ ] Custom integrations
- [ ] Enterprise tier launch

### Phase 4: Scale (Months 7-12)
- [ ] Multi-region expansion
- [ ] Advanced analytics
- [ ] Custom embedding models
- [ ] Marketplace (integrations)
- [ ] Mobile apps

---

## Success Metrics

| Metric | Month 3 | Month 6 | Month 12 |
|--------|---------|---------|----------|
| Registered users | 5,000 | 20,000 | 100,000 |
| Paid customers | 100 | 500 | 2,000 |
| Team tier customers | 10 | 50 | 200 |
| Enterprise customers | 0 | 5 | 20 |
| MRR | $5K | $50K | $300K |
| ARR | - | $600K | $3.6M |

---

## Sources & References

- [Supabase Pricing](https://supabase.com/pricing)
- [Supabase Architecture](https://supabase.com/docs/guides/getting-started/architecture)
- [Mem0 Platform](https://mem0.ai/)
- [Zep Context Engineering](https://www.getzep.com/)
- [Everything is Context Paper](https://arxiv.org/abs/2512.05470v1)
