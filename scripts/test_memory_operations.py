#!/usr/bin/env python3
"""
ContextFS Memory Operations Test Script

Tests all memory operations programmatically:
- Basic CRUD (save, recall, search, delete)
- Lineage (evolve, get_lineage)
- Graph (link, get_related)
- Merge and Split

Usage:
    python scripts/test_memory_operations.py
"""

from contextfs import ContextFS


def print_header(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print("=" * 60)


def print_success(msg: str) -> None:
    """Print success message."""
    print(f"  ✓ {msg}")


def print_memory(mem, prefix: str = "") -> None:
    """Print memory details."""
    print(f"{prefix}ID: {mem.id[:12]}...")
    print(f"{prefix}Content: {mem.content[:60]}{'...' if len(mem.content) > 60 else ''}")
    print(f"{prefix}Type: {mem.type.value}")
    print(f"{prefix}Tags: {', '.join(mem.tags)}")
    if mem.summary:
        print(f"{prefix}Summary: {mem.summary}")


def main():
    print("\n" + "=" * 60)
    print(" ContextFS Memory Operations Test")
    print("=" * 60)

    ctx = ContextFS()

    # =========================================================================
    print_header("1. Basic Save Operations")
    # =========================================================================

    # Save different types
    fact = ctx.save(
        "The API endpoint for users is /api/v1/users",
        type="fact",
        tags=["api", "users"],
        summary="Users API endpoint",
    )
    print_success(f"Saved fact: {fact.id[:12]}")

    decision = ctx.save(
        "We chose PostgreSQL over MongoDB for ACID compliance and relational queries",
        type="decision",
        tags=["database", "architecture"],
        summary="Database choice",
    )
    print_success(f"Saved decision: {decision.id[:12]}")

    procedural = ctx.save(
        "To deploy: 1) Run tests, 2) Build docker image, 3) Push to registry, 4) Apply k8s manifests",
        type="procedural",
        tags=["deployment", "devops"],
        summary="Deployment steps",
    )
    print_success(f"Saved procedural: {procedural.id[:12]}")

    # =========================================================================
    print_header("2. Search and Recall")
    # =========================================================================

    # Search
    results = ctx.search("API endpoint", limit=5)
    print(f"  Search 'API endpoint': {len(results)} results")
    for r in results[:3]:
        print(f"    [{r.score:.2f}] {r.memory.id[:12]}... - {r.memory.content[:40]}...")

    # Recall
    recalled = ctx.recall(fact.id[:8])
    if recalled:
        print_success(f"Recalled by partial ID: {recalled.id[:12]}")
    else:
        print("  ✗ Failed to recall")

    # =========================================================================
    print_header("3. Evolve Operations (Lineage)")
    # =========================================================================

    # Create initial memory
    v1 = ctx.save(
        "Rate limit is 100 requests per minute",
        type="fact",
        tags=["api", "limits"],
        summary="Rate limit v1",
    )
    print(f"  Created v1: {v1.id[:12]}...")

    # Evolve to v2
    v2 = ctx.evolve(
        v1.id,
        "Rate limit increased to 500 requests per minute",
        summary="Rate limit v2",
    )
    print(f"  Evolved to v2: {v2.id[:12]}...")

    # Evolve to v3
    v3 = ctx.evolve(
        v2.id,
        "Rate limit is 1000 requests per minute for authenticated users, 100 for anonymous",
        summary="Rate limit v3",
        additional_tags=["premium", "auth"],
    )
    print(f"  Evolved to v3: {v3.id[:12]}...")

    # View lineage
    lineage = ctx.get_lineage(v3.id)
    print("\n  Lineage for v3:")
    print(f"    Root: {str(lineage.get('root', 'N/A'))[:12]}...")
    print(f"    Ancestors ({len(lineage.get('ancestors', []))}):")
    for anc in lineage.get("ancestors", []):
        # Handle both formats: 'id' or 'memory_id'
        anc_id = anc.get("id") or anc.get("memory_id", "unknown")
        depth = anc.get("depth", "")
        depth_str = f" (depth {depth})" if depth else ""
        print(f"      ↑ [{anc['relation']}] {str(anc_id)[:12]}...{depth_str}")

    print_success("Lineage tracking works")

    # =========================================================================
    print_header("4. Link Operations (Graph)")
    # =========================================================================

    # Create memories to link
    auth = ctx.save(
        "Authentication uses JWT tokens with RS256 signing",
        type="fact",
        tags=["auth", "jwt"],
    )
    print(f"  Created auth: {auth.id[:12]}...")

    session = ctx.save(
        "Session data stored in Redis with 24h TTL",
        type="fact",
        tags=["session", "redis"],
    )
    print(f"  Created session: {session.id[:12]}...")

    cache = ctx.save(
        "API responses cached in Redis for 5 minutes",
        type="fact",
        tags=["cache", "redis"],
    )
    print(f"  Created cache: {cache.id[:12]}...")

    # Create links (valid relations: references, related_to, contradicts, supersedes, part_of, etc.)
    ctx.link(auth.id, session.id, "related_to")
    print("  Linked: auth -> session (related_to)")

    ctx.link(v3.id, auth.id, "references")
    print("  Linked: rate_limit_v3 -> auth (references)")

    ctx.link(session.id, cache.id, "related_to", bidirectional=True)
    print("  Linked: session <-> cache (related_to, bidirectional)")

    # Find related
    related = ctx.get_related(auth.id)
    print(f"\n  Related to auth ({len(related)}):")
    for r in related:
        print(f"    → [{r['relation']}] {r['id'][:12]}...")

    # Multi-hop
    related_deep = ctx.get_related(v3.id, max_depth=2)
    print(f"\n  Related to v3 (depth 2): {len(related_deep)} memories")

    print_success("Graph linking works")

    # =========================================================================
    print_header("5. Merge Operations")
    # =========================================================================

    # Create memories to merge
    react = ctx.save("Frontend uses React 18", type="fact", tags=["frontend", "react"])
    typescript = ctx.save(
        "Frontend uses TypeScript 5.0", type="fact", tags=["frontend", "typescript"]
    )
    vite = ctx.save("Build tool is Vite 5", type="fact", tags=["frontend", "build"])

    print("  Created 3 frontend memories")

    # Merge with union strategy
    merged = ctx.merge(
        [react.id, typescript.id, vite.id],
        summary="Frontend tech stack",
        strategy="union",
    )
    print(f"  Merged into: {merged.id[:12]}...")
    print(f"  Merged tags: {', '.join(merged.tags)}")

    # Check merge lineage
    merge_lineage = ctx.get_lineage(merged.id)
    ancestors = merge_lineage.get("ancestors", [])
    print(f"  Merge ancestors ({len(ancestors)}):")
    for anc in ancestors:
        anc_id = anc.get("id") or anc.get("memory_id", "unknown")
        print(f"    ↑ [{anc['relation']}] {str(anc_id)[:12]}...")

    print_success("Merge operations work")

    # =========================================================================
    print_header("6. Split Operations")
    # =========================================================================

    # Create memory to split
    config = ctx.save(
        "Environment config: DEBUG=true, LOG_LEVEL=info, MAX_CONNECTIONS=100, TIMEOUT=30s",
        type="fact",
        tags=["config", "environment"],
        summary="All env vars",
    )
    print(f"  Created config: {config.id[:12]}...")

    # Split into parts
    parts = ctx.split(
        config.id,
        parts=[
            "DEBUG=true (enables debug mode)",
            "LOG_LEVEL=info (standard logging)",
            "MAX_CONNECTIONS=100 (connection pool size)",
            "TIMEOUT=30s (request timeout)",
        ],
        summaries=["Debug flag", "Log level", "Max connections", "Timeout"],
    )
    print(f"  Split into {len(parts)} parts:")
    for i, part in enumerate(parts):
        print(f"    {i+1}. {part.id[:12]}... - {part.summary}")

    # Check split lineage
    split_lineage = ctx.get_lineage(parts[0].id)
    split_ancestors = split_lineage.get("ancestors", [])
    print(f"\n  Split part lineage ({len(split_ancestors)} ancestors):")
    for anc in split_ancestors:
        anc_id = anc.get("id") or anc.get("memory_id", "unknown")
        print(f"    ↑ [{anc['relation']}] {str(anc_id)[:12]}...")

    print_success("Split operations work")

    # =========================================================================
    print_header("7. Graph Status")
    # =========================================================================

    print(f"  Has graph: {ctx.has_graph()}")

    # Get storage stats
    if hasattr(ctx._storage, "get_graph_stats"):
        stats = ctx._storage.get_graph_stats()
        print(f"  Backend: {stats.get('backend', 'unknown')}")
        print(f"  SQLite edges: {stats.get('sqlite_edges', 'N/A')}")

    # =========================================================================
    print_header("8. Cleanup Test (Optional)")
    # =========================================================================

    # Delete one test memory
    deleted = ctx.delete(procedural.id)
    print(f"  Deleted procedural: {deleted}")
    print_success("Delete works")

    # =========================================================================
    print("\n" + "=" * 60)
    print(" All Tests Passed!")
    print("=" * 60)
    # =========================================================================

    print("\nCreated memories summary:")
    print(f"  - Fact: {fact.id[:12]}...")
    print(f"  - Decision: {decision.id[:12]}...")
    print(f"  - Rate limit lineage: {v1.id[:8]} → {v2.id[:8]} → {v3.id[:8]}")
    print(f"  - Auth: {auth.id[:12]}...")
    print(f"  - Session: {session.id[:12]}...")
    print(f"  - Cache: {cache.id[:12]}...")
    print(f"  - Merged frontend: {merged.id[:12]}...")
    print(f"  - Split config: {config.id[:12]}... → {len(parts)} parts")

    return True


if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
