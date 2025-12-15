"""
MCP Server for ContextFS.

Provides memory operations via Model Context Protocol.
Works with Claude Desktop, Claude Code, and any MCP client.
"""

import asyncio

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from contextfs.core import ContextFS
from contextfs.schemas import MemoryType

# Global ContextFS instance
_ctx: ContextFS | None = None


def get_ctx() -> ContextFS:
    """Get or create ContextFS instance."""
    global _ctx
    if _ctx is None:
        _ctx = ContextFS(auto_load=True)
    return _ctx


# Create MCP server
server = Server("contextfs")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools."""
    return [
        Tool(
            name="contextfs_save",
            description="Save a memory to ContextFS. Use for facts, decisions, procedures, or session summaries.",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Content to save",
                    },
                    "type": {
                        "type": "string",
                        "enum": [
                            "fact",
                            "decision",
                            "procedural",
                            "episodic",
                            "user",
                            "code",
                            "error",
                        ],
                        "description": "Memory type",
                        "default": "fact",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for categorization",
                    },
                    "summary": {
                        "type": "string",
                        "description": "Brief summary",
                    },
                    "save_session": {
                        "type": "string",
                        "enum": ["current", "previous"],
                        "description": "Save session instead of memory",
                    },
                    "label": {
                        "type": "string",
                        "description": "Label for session",
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="contextfs_search",
            description="Search memories using semantic similarity",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum results",
                        "default": 5,
                    },
                    "type": {
                        "type": "string",
                        "enum": [
                            "fact",
                            "decision",
                            "procedural",
                            "episodic",
                            "user",
                            "code",
                            "error",
                        ],
                        "description": "Filter by type",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="contextfs_recall",
            description="Recall a specific memory by ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Memory ID (can be partial, at least 8 chars)",
                    },
                },
                "required": ["id"],
            },
        ),
        Tool(
            name="contextfs_list",
            description="List recent memories",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "number",
                        "description": "Maximum results",
                        "default": 10,
                    },
                    "type": {
                        "type": "string",
                        "enum": [
                            "fact",
                            "decision",
                            "procedural",
                            "episodic",
                            "user",
                            "code",
                            "error",
                        ],
                        "description": "Filter by type",
                    },
                },
            },
        ),
        Tool(
            name="contextfs_sessions",
            description="List recent sessions",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "number",
                        "description": "Maximum results",
                        "default": 10,
                    },
                    "label": {
                        "type": "string",
                        "description": "Filter by label",
                    },
                    "tool": {
                        "type": "string",
                        "description": "Filter by tool (claude-code, gemini, etc.)",
                    },
                },
            },
        ),
        Tool(
            name="contextfs_load_session",
            description="Load a session's messages into context",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID (can be partial)",
                    },
                    "label": {
                        "type": "string",
                        "description": "Session label",
                    },
                    "max_messages": {
                        "type": "number",
                        "description": "Maximum messages to return",
                        "default": 20,
                    },
                },
            },
        ),
        Tool(
            name="contextfs_message",
            description="Add a message to the current session",
            inputSchema={
                "type": "object",
                "properties": {
                    "role": {
                        "type": "string",
                        "enum": ["user", "assistant", "system"],
                        "description": "Message role",
                    },
                    "content": {
                        "type": "string",
                        "description": "Message content",
                    },
                },
                "required": ["role", "content"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    ctx = get_ctx()

    try:
        if name == "contextfs_save":
            # Check if saving session
            if arguments.get("save_session"):
                session = ctx.get_current_session()
                if session:
                    if arguments.get("label"):
                        session.label = arguments["label"]
                    ctx.end_session(generate_summary=True)
                    return [
                        TextContent(
                            type="text",
                            text=f"Session saved.\nSession ID: {session.id}\nLabel: {session.label or 'none'}",
                        )
                    ]
                else:
                    return [TextContent(type="text", text="No active session to save.")]

            # Save memory
            content = arguments.get("content", "")
            if not content:
                return [TextContent(type="text", text="Error: content is required")]

            memory_type = MemoryType(arguments.get("type", "fact"))
            tags = arguments.get("tags", [])
            summary = arguments.get("summary")

            memory = ctx.save(
                content=content,
                type=memory_type,
                tags=tags,
                summary=summary,
            )

            return [
                TextContent(
                    type="text",
                    text=f"Memory saved successfully.\nID: {memory.id}\nType: {memory.type.value}",
                )
            ]

        elif name == "contextfs_search":
            query = arguments.get("query", "")
            limit = arguments.get("limit", 5)
            type_filter = MemoryType(arguments["type"]) if arguments.get("type") else None

            results = ctx.search(query, limit=limit, type=type_filter)

            if not results:
                return [TextContent(type="text", text="No memories found.")]

            output = []
            for r in results:
                output.append(f"[{r.memory.id}] ({r.score:.2f}) [{r.memory.type.value}]")
                if r.memory.summary:
                    output.append(f"  Summary: {r.memory.summary}")
                output.append(f"  {r.memory.content[:200]}...")
                if r.memory.tags:
                    output.append(f"  Tags: {', '.join(r.memory.tags)}")
                output.append("")

            return [TextContent(type="text", text="\n".join(output))]

        elif name == "contextfs_recall":
            memory_id = arguments.get("id", "")
            memory = ctx.recall(memory_id)

            if not memory:
                return [TextContent(type="text", text=f"Memory not found: {memory_id}")]

            output = [
                f"ID: {memory.id}",
                f"Type: {memory.type.value}",
                f"Created: {memory.created_at.isoformat()}",
            ]
            if memory.summary:
                output.append(f"Summary: {memory.summary}")
            if memory.tags:
                output.append(f"Tags: {', '.join(memory.tags)}")
            output.append(f"\nContent:\n{memory.content}")

            return [TextContent(type="text", text="\n".join(output))]

        elif name == "contextfs_list":
            limit = arguments.get("limit", 10)
            type_filter = MemoryType(arguments["type"]) if arguments.get("type") else None

            memories = ctx.list_recent(limit=limit, type=type_filter)

            if not memories:
                return [TextContent(type="text", text="No memories found.")]

            output = []
            for m in memories:
                line = f"[{m.id}] [{m.type.value}]"
                if m.summary:
                    line += f" {m.summary}"
                else:
                    line += f" {m.content[:50]}..."
                output.append(line)

            return [TextContent(type="text", text="\n".join(output))]

        elif name == "contextfs_sessions":
            limit = arguments.get("limit", 10)
            label = arguments.get("label")
            tool = arguments.get("tool")

            sessions = ctx.list_sessions(limit=limit, label=label, tool=tool)

            if not sessions:
                return [TextContent(type="text", text="No sessions found.")]

            output = []
            for s in sessions:
                line = f"[{s.id[:12]}] {s.tool}"
                if s.label:
                    line += f" ({s.label})"
                line += f" - {s.started_at.strftime('%Y-%m-%d %H:%M')}"
                output.append(line)

            return [TextContent(type="text", text="\n".join(output))]

        elif name == "contextfs_load_session":
            session_id = arguments.get("session_id")
            label = arguments.get("label")
            max_messages = arguments.get("max_messages", 20)

            session = ctx.load_session(session_id=session_id, label=label)

            if not session:
                return [TextContent(type="text", text="Session not found.")]

            output = [
                f"Session: {session.id}",
                f"Tool: {session.tool}",
                f"Started: {session.started_at.isoformat()}",
            ]
            if session.label:
                output.append(f"Label: {session.label}")
            if session.summary:
                output.append(f"Summary: {session.summary}")

            output.append(f"\nMessages ({len(session.messages)}):\n")

            for msg in session.messages[-max_messages:]:
                output.append(f"[{msg.role}] {msg.content[:500]}")
                output.append("")

            return [TextContent(type="text", text="\n".join(output))]

        elif name == "contextfs_message":
            role = arguments.get("role", "user")
            content = arguments.get("content", "")

            msg = ctx.add_message(role, content)

            return [
                TextContent(
                    type="text",
                    text=f"Message added to session.\nMessage ID: {msg.id}",
                )
            ]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def run_server():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    """Entry point for MCP server."""
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
