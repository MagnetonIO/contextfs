"""
Agent base classes for memory-native LLM agents.

This module provides:
- AgentProvider: Abstract interface for LLM providers (Anthropic, OpenAI, etc.)
- Agent: Memory-native agent that persists all state to ContextFS
- Tool: Wrapper for callable tools that agents can use
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from contextfs.schemas import Memory
from contextfs.types.versioned import ChangeReason

if TYPE_CHECKING:
    from contextfs import ContextFS


@dataclass
class ToolCall:
    """Represents a tool call from the LLM."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolResult:
    """Result of executing a tool."""

    tool_call_id: str
    output: Any
    error: str | None = None
    duration_ms: int | None = None


@dataclass
class Tool:
    """Wrapper for a callable tool that an agent can use."""

    name: str
    description: str
    func: Callable[..., Any]
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_function(cls, func: Callable[..., Any], name: str | None = None) -> Tool:
        """Create a Tool from a function, extracting schema from type hints."""
        func_name = name or func.__name__
        doc = func.__doc__ or f"Execute {func_name}"

        # Extract parameters from type hints
        sig = inspect.signature(func)
        params: dict[str, Any] = {
            "type": "object",
            "properties": {},
            "required": [],
        }

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            param_type = "string"  # Default
            if param.annotation is not inspect.Parameter.empty:
                if param.annotation is int:
                    param_type = "integer"
                elif param.annotation is float:
                    param_type = "number"
                elif param.annotation is bool:
                    param_type = "boolean"
                elif param.annotation is list:
                    param_type = "array"
                elif param.annotation is dict:
                    param_type = "object"

            params["properties"][param_name] = {"type": param_type}

            if param.default is inspect.Parameter.empty:
                params["required"].append(param_name)

        return cls(
            name=func_name,
            description=doc.strip().split("\n")[0],  # First line of docstring
            func=func,
            parameters=params,
        )

    def to_schema(self) -> dict[str, Any]:
        """Convert to LLM tool schema format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    async def execute(self, **kwargs: Any) -> Any:
        """Execute the tool with given arguments."""
        if asyncio.iscoroutinefunction(self.func):
            return await self.func(**kwargs)
        return self.func(**kwargs)


class AgentProvider(ABC):
    """Abstract interface for LLM providers."""

    model_name: str = "unknown"

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Send a completion request to the LLM.

        Args:
            messages: Conversation messages in standard format.
            tools: Tool schemas in provider-specific format.
            max_tokens: Maximum tokens to generate.
            **kwargs: Provider-specific options.

        Returns:
            Response dict with:
            - content: str | None - Text response
            - tool_calls: list[ToolCall] | None - Tool calls to execute
            - stop_reason: str - Why generation stopped
            - usage: dict - Token usage stats
        """
        pass

    def format_tool_result(self, tool_call_id: str, result: Any) -> dict[str, Any]:
        """Format tool result for the provider's expected format."""
        return {
            "type": "tool_result",
            "tool_use_id": tool_call_id,
            "content": json.dumps(result) if not isinstance(result, str) else result,
        }


class Agent:
    """
    Memory-native LLM agent with tool use.

    All agent execution state is persisted to ContextFS memories:
    - agent_run: Overall execution record
    - step: Individual tool calls and responses

    Example:
        ctx = ContextFS()
        provider = AnthropicProvider()

        agent = Agent(
            name="analyst",
            provider=provider,
            ctx=ctx,
            tools=[Tool.from_function(fetch_data)],
            system_prompt="You analyze data."
        )

        result = await agent.run("Analyze the sales data")
    """

    def __init__(
        self,
        name: str,
        provider: AgentProvider,
        ctx: ContextFS,
        tools: list[Tool] | None = None,
        system_prompt: str | None = None,
        max_iterations: int = 10,
    ):
        """
        Initialize an agent.

        Args:
            name: Agent identifier.
            provider: LLM provider for completions.
            ctx: ContextFS instance for memory persistence.
            tools: List of tools the agent can use.
            system_prompt: System prompt for the agent.
            max_iterations: Maximum tool-use iterations to prevent infinite loops.
        """
        self.name = name
        self.provider = provider
        self.ctx = ctx
        self.tools = {t.name: t for t in (tools or [])}
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations

        self._run_memory: Memory | None = None
        self._step_count: int = 0
        self._total_tokens: int = 0

    def add_tool(self, tool: Tool) -> None:
        """Add a tool to the agent."""
        self.tools[tool.name] = tool

    def add_function(self, func: Callable[..., Any], name: str | None = None) -> None:
        """Add a function as a tool."""
        tool = Tool.from_function(func, name)
        self.add_tool(tool)

    async def run(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Execute the agent with a prompt.

        Args:
            prompt: User prompt to process.
            context: Optional context data to include.
            **kwargs: Additional options passed to provider.

        Returns:
            Final text response from the agent.

        Raises:
            Exception: If execution fails after all retries.
        """
        # Create agent_run memory
        self._run_memory = self.ctx.save(
            Memory.agent_run(
                content=f"Agent run: {prompt[:100]}{'...' if len(prompt) > 100 else ''}",
                agent_name=self.name,
                model=self.provider.model_name,
                status="running",
                summary=f"Agent {self.name} processing: {prompt[:50]}",
            )
        )
        self._step_count = 0
        self._total_tokens = 0

        start_time = datetime.now(timezone.utc)

        try:
            result = await self._execute(prompt, context, **kwargs)

            # Update run memory to completed
            self._update_run_status(
                status="completed",
                started_at=start_time.isoformat(),
                completed_at=datetime.now(timezone.utc).isoformat(),
            )

            return result

        except Exception as e:
            # Update run memory to failed
            self._update_run_status(
                status="failed",
                error=str(e),
                started_at=start_time.isoformat(),
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
            raise

    async def _execute(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """Internal execution loop with tool use."""
        messages: list[dict[str, Any]] = []

        # Add system prompt
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # Add context if provided
        if context:
            context_str = json.dumps(context, indent=2)
            messages.append(
                {
                    "role": "user",
                    "content": f"Context:\n```json\n{context_str}\n```\n\n{prompt}",
                }
            )
        else:
            messages.append({"role": "user", "content": prompt})

        # Get tool schemas
        tool_schemas = [t.to_schema() for t in self.tools.values()] if self.tools else None

        # Execution loop
        for iteration in range(self.max_iterations):
            response = await self.provider.complete(
                messages=messages,
                tools=tool_schemas,
                **kwargs,
            )

            # Track token usage
            usage = response.get("usage", {})
            self._total_tokens += usage.get("input_tokens", 0) + usage.get("output_tokens", 0)

            # Check for tool calls
            tool_calls = response.get("tool_calls", [])

            if tool_calls:
                # Add assistant message with tool calls
                messages.append(
                    {
                        "role": "assistant",
                        "content": response.get("content"),
                        "tool_calls": tool_calls,
                    }
                )

                # Execute each tool call
                tool_results = []
                for tc in tool_calls:
                    result = await self._execute_tool(tc)
                    tool_results.append(self.provider.format_tool_result(tc["id"], result.output))

                # Add tool results
                messages.append(
                    {
                        "role": "user",
                        "content": tool_results,
                    }
                )

            else:
                # No tool calls - return final response
                return response.get("content", "")

        # Max iterations reached
        return response.get("content", "Max iterations reached without final response.")

    async def _execute_tool(self, tool_call: dict[str, Any]) -> ToolResult:
        """Execute a single tool call and save as step memory."""
        tool_name = tool_call.get("name", "")
        tool_input = tool_call.get("input", {})
        tool_id = tool_call.get("id", "")

        self._step_count += 1
        start_time = time.perf_counter()

        # Save step memory (before execution)
        step_memory = self.ctx.save(
            Memory.step(
                content=f"Tool call: {tool_name}",
                task_id=self._run_memory.id if self._run_memory else "",
                step_number=self._step_count,
                action="tool_call",
                summary=f"Calling {tool_name}",
            )
        )

        try:
            tool = self.tools.get(tool_name)
            if not tool:
                error = f"Unknown tool: {tool_name}"
                return ToolResult(tool_call_id=tool_id, output=None, error=error)

            output = await tool.execute(**tool_input)
            duration_ms = int((time.perf_counter() - start_time) * 1000)

            # Update step memory with result
            self.ctx._lineage.evolve(
                step_memory.id,
                new_content=f"Tool call: {tool_name} -> {str(output)[:100]}",
                reason=ChangeReason.OBSERVATION,
            )

            return ToolResult(
                tool_call_id=tool_id,
                output=output,
                duration_ms=duration_ms,
            )

        except Exception as e:
            duration_ms = int((time.perf_counter() - start_time) * 1000)

            # Update step memory with error
            self.ctx._lineage.evolve(
                step_memory.id,
                new_content=f"Tool call failed: {tool_name} -> {str(e)}",
                reason=ChangeReason.OBSERVATION,
            )

            return ToolResult(
                tool_call_id=tool_id,
                output=None,
                error=str(e),
                duration_ms=duration_ms,
            )

    def _update_run_status(self, **updates: Any) -> None:
        """Update the agent run memory with new status."""
        if not self._run_memory:
            return

        # Build new content
        status = updates.get("status", "unknown")
        content = f"Agent run ({status}): {self.name}"
        if updates.get("error"):
            content += f" - Error: {updates['error']}"

        # Add token info to structured data
        updates["total_tokens"] = self._total_tokens

        self.ctx._lineage.evolve(
            self._run_memory.id,
            new_content=content,
            reason=ChangeReason.OBSERVATION,
        )
