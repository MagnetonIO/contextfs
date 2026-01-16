"""Memory extraction from conversation transcripts."""

import json
import re
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from contextfs.schemas import MemoryType

from .utils import get_ctx

extract_app = typer.Typer(
    name="extract",
    help="Extract memories from conversations",
    no_args_is_help=True,
)

console = Console()


# Patterns that indicate learnings/decisions
LEARNING_PATTERNS = [
    # Decision patterns
    (r"(?i)decided to|chose to|going with|we('ll)? use|let's use|better to", "decision"),
    (r"(?i)the reason|because|rationale|this approach|tradeoff", "decision"),
    # Error patterns
    (r"(?i)error:|exception:|failed:|traceback|bug|issue:", "error"),
    (r"(?i)fixed by|resolved by|solution:|the fix", "error"),
    # Fact patterns
    (r"(?i)i learned|discovered|found out|turns out|important:", "fact"),
    (r"(?i)note:|remember:|key point|takeaway", "fact"),
    # Procedure patterns
    (r"(?i)steps?:|workflow:|process:|to do this", "procedural"),
    (r"(?i)first,.*then,|step \d", "procedural"),
]


def extract_from_text(text: str) -> list[dict[str, Any]]:
    """Extract potential memories from text using pattern matching."""
    extractions = []

    # Split into sentences/paragraphs
    chunks = re.split(r"\n\n+|(?<=[.!?])\s+", text)

    for chunk in chunks:
        chunk = chunk.strip()
        if len(chunk) < 20:  # Skip very short chunks
            continue

        for pattern, mem_type in LEARNING_PATTERNS:
            if re.search(pattern, chunk):
                extractions.append(
                    {
                        "type": mem_type,
                        "content": chunk[:500],  # Limit content length
                        "pattern": pattern,
                    }
                )
                break  # Only match first pattern per chunk

    return extractions


def parse_transcript(transcript_path: Path) -> list[dict]:
    """Parse a Claude Code transcript file."""
    messages = []

    with open(transcript_path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                msg_type = entry.get("type")

                if msg_type == "assistant":
                    content = entry.get("message", {}).get("content", "")
                    if isinstance(content, list):
                        # Handle structured content
                        text_parts = []
                        for part in content:
                            if part.get("type") == "text":
                                text_parts.append(part.get("text", ""))
                        content = "\n".join(text_parts)
                    messages.append({"role": "assistant", "content": content})

                elif msg_type == "human":
                    content = entry.get("message", {}).get("content", "")
                    messages.append({"role": "user", "content": content})

            except json.JSONDecodeError:
                continue

    return messages


@extract_app.command("transcript")
def extract_transcript(
    transcript: Path = typer.Argument(
        None, help="Path to transcript JSONL file (or reads from stdin)"
    ),
    save: bool = typer.Option(False, "--save", "-s", help="Save extracted memories"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output"),
    agent_role: str = typer.Option(None, "--agent-role", "-a", help="Agent role tag"),
    project: str = typer.Option(None, "--project", "-p", help="Project name for grouping"),
):
    """Extract memories from a Claude Code transcript.

    Analyzes the conversation and extracts:
    - Decisions made (with rationale)
    - Errors encountered and solutions
    - Facts learned
    - Procedures discovered

    Example (Stop hook):
        contextfs extract transcript --save --quiet

    Example (manual):
        contextfs extract transcript /path/to/transcript.jsonl
    """
    transcript_path = transcript

    # Try to get transcript from stdin (hook input)
    if not transcript_path and not sys.stdin.isatty():
        import select

        if select.select([sys.stdin], [], [], 0.1)[0]:
            try:
                hook_input = json.load(sys.stdin)
                if "transcript_path" in hook_input:
                    transcript_path = Path(hook_input["transcript_path"]).expanduser()
            except Exception:
                pass

    if not transcript_path:
        if not quiet:
            console.print("[yellow]No transcript provided[/yellow]")
        return

    if not transcript_path.exists():
        if not quiet:
            console.print(f"[red]Transcript not found: {transcript_path}[/red]")
        raise typer.Exit(1)

    # Parse transcript
    messages = parse_transcript(transcript_path)
    if not messages:
        if not quiet:
            console.print("[yellow]No messages found in transcript[/yellow]")
        return

    # Extract from assistant messages only
    assistant_text = "\n\n".join(m["content"] for m in messages if m["role"] == "assistant")

    extractions = extract_from_text(assistant_text)

    if not extractions:
        if not quiet:
            console.print("[dim]No extractable memories found[/dim]")
        return

    # Deduplicate similar extractions
    seen_content = set()
    unique_extractions = []
    for ext in extractions:
        content_hash = ext["content"][:100]
        if content_hash not in seen_content:
            seen_content.add(content_hash)
            unique_extractions.append(ext)

    if not quiet:
        console.print(f"[cyan]Found {len(unique_extractions)} potential memories[/cyan]")

    # Save if requested
    if save:
        ctx = get_ctx()
        saved_count = 0

        tags = ["auto-extracted"]
        if agent_role:
            tags.append(f"{agent_role}-agent")

        for ext in unique_extractions[:10]:  # Limit to 10 per session
            try:
                memory_type = MemoryType(ext["type"])

                # Build structured data for types that require it
                structured_data = None
                if memory_type == MemoryType.DECISION:
                    structured_data = {
                        "decision": ext["content"][:200],
                        "rationale": "Auto-extracted from conversation",
                        "alternatives": [],
                    }
                elif memory_type == MemoryType.ERROR:
                    structured_data = {
                        "error_type": "auto-extracted",
                        "message": ext["content"][:200],
                        "resolution": "See full content",
                    }
                elif memory_type == MemoryType.PROCEDURAL:
                    structured_data = {
                        "title": "Auto-extracted procedure",
                        "steps": [ext["content"][:500]],
                    }

                ctx.save(
                    content=ext["content"],
                    type=memory_type,
                    tags=tags,
                    summary=f"Auto-extracted {ext['type']}: {ext['content'][:50]}...",
                    source_tool=f"contextfs-extract{f'-{agent_role}' if agent_role else ''}",
                    structured_data=structured_data,
                    project=project,
                )
                saved_count += 1

            except Exception as e:
                if not quiet:
                    console.print(f"[yellow]Failed to save: {e}[/yellow]")

        if not quiet:
            console.print(f"[green]Saved {saved_count} memories[/green]")
        elif saved_count > 0:
            print(f"Auto-extracted {saved_count} memories")

    else:
        # Just display extractions
        for ext in unique_extractions:
            console.print(f"\n[cyan]{ext['type'].upper()}[/cyan]")
            console.print(
                ext["content"][:200] + "..." if len(ext["content"]) > 200 else ext["content"]
            )


@extract_app.command("patterns")
def show_patterns():
    """Show the patterns used for memory extraction."""
    console.print("[bold]Memory Extraction Patterns[/bold]\n")

    for pattern, mem_type in LEARNING_PATTERNS:
        console.print(f"[cyan]{mem_type}[/cyan]: {pattern}")


@extract_app.command("test")
def test_extraction(
    text: str = typer.Argument(..., help="Text to test extraction on"),
):
    """Test memory extraction on sample text."""
    extractions = extract_from_text(text)

    if not extractions:
        console.print("[yellow]No patterns matched[/yellow]")
        return

    for ext in extractions:
        console.print(f"\n[cyan]{ext['type']}[/cyan] (pattern: {ext['pattern']})")
        console.print(ext["content"])
