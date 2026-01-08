"""
ContextFS Agent Framework.

Provides memory-native LLM agents with tool use capabilities.
All agent state is persisted to ContextFS memories for full observability.

Example:
    from contextfs import ContextFS
    from contextfs.agents import Agent, AnthropicProvider

    ctx = ContextFS()
    provider = AnthropicProvider()

    agent = Agent(
        name="analyst",
        provider=provider,
        ctx=ctx,
        system_prompt="You analyze data and provide insights."
    )

    result = await agent.run("Analyze the sales data")
"""

from contextfs.agents.base import Agent, AgentProvider, Tool
from contextfs.agents.providers import AnthropicProvider, OpenAIProvider

__all__ = [
    "Agent",
    "AgentProvider",
    "Tool",
    "AnthropicProvider",
    "OpenAIProvider",
]
