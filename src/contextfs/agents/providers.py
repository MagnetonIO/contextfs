"""
LLM Provider implementations for the Agent framework.

Provides concrete implementations of AgentProvider for:
- Anthropic (Claude models)
- OpenAI (GPT models)

Both providers normalize responses to a common format for the Agent class.
"""

from __future__ import annotations

import json
import os
from typing import Any

from contextfs.agents.base import AgentProvider


class AnthropicProvider(AgentProvider):
    """
    Anthropic Claude provider.

    Requires the `anthropic` package and ANTHROPIC_API_KEY environment variable.

    Example:
        provider = AnthropicProvider(model="claude-sonnet-4-20250514")
        response = await provider.complete(messages=[...])
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
    ):
        """
        Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key. Defaults to ANTHROPIC_API_KEY env var.
            model: Model to use. Defaults to claude-sonnet-4-20250514.
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model_name = model
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy-load the Anthropic client."""
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError("anthropic package required. Install with: pip install anthropic")
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send completion request to Claude."""
        client = self._get_client()

        # Extract system message
        system = None
        api_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system = msg.get("content", "")
            else:
                api_messages.append(self._format_message(msg))

        # Build request
        request: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "messages": api_messages,
        }

        if system:
            request["system"] = system

        if tools:
            request["tools"] = tools

        # Add any extra kwargs
        request.update(kwargs)

        # Make request (sync, but we're in async context)
        response = client.messages.create(**request)

        return self._parse_response(response)

    def _format_message(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Format message for Anthropic API."""
        role = msg.get("role", "user")
        content = msg.get("content")

        # Handle tool results
        if isinstance(content, list) and all(
            isinstance(c, dict) and c.get("type") == "tool_result" for c in content
        ):
            return {"role": "user", "content": content}

        # Handle assistant messages with tool calls
        if role == "assistant" and msg.get("tool_calls"):
            blocks = []
            if content:
                blocks.append({"type": "text", "text": content})
            for tc in msg["tool_calls"]:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc["input"],
                    }
                )
            return {"role": "assistant", "content": blocks}

        return {"role": role, "content": content}

    def _parse_response(self, response: Any) -> dict[str, Any]:
        """Parse Anthropic response to common format."""
        content = None
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content = block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )

        return {
            "content": content,
            "tool_calls": tool_calls if tool_calls else None,
            "stop_reason": response.stop_reason,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        }

    def format_tool_result(self, tool_call_id: str, result: Any) -> dict[str, Any]:
        """Format tool result for Anthropic."""
        content = json.dumps(result) if not isinstance(result, str) else result
        return {
            "type": "tool_result",
            "tool_use_id": tool_call_id,
            "content": content,
        }


class OpenAIProvider(AgentProvider):
    """
    OpenAI GPT provider.

    Requires the `openai` package and OPENAI_API_KEY environment variable.

    Example:
        provider = OpenAIProvider(model="gpt-4")
        response = await provider.complete(messages=[...])
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4",
    ):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key. Defaults to OPENAI_API_KEY env var.
            model: Model to use. Defaults to gpt-4.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model_name = model
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy-load the OpenAI client."""
        if self._client is None:
            try:
                import openai
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")
            self._client = openai.OpenAI(api_key=self.api_key)
        return self._client

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send completion request to GPT."""
        client = self._get_client()

        # Format messages for OpenAI
        api_messages = [self._format_message(msg) for msg in messages]

        # Build request
        request: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "messages": api_messages,
        }

        if tools:
            # Convert to OpenAI tool format
            request["tools"] = [self._format_tool(t) for t in tools]

        # Add any extra kwargs
        request.update(kwargs)

        # Make request
        response = client.chat.completions.create(**request)

        return self._parse_response(response)

    def _format_message(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Format message for OpenAI API."""
        role = msg.get("role", "user")
        content = msg.get("content")

        # Handle tool results - OpenAI expects tool results as separate messages
        if isinstance(content, list) and all(
            isinstance(c, dict) and c.get("type") == "tool_result" for c in content
        ):
            # Return first result (OpenAI handles differently)
            return {
                "role": "tool",
                "tool_call_id": content[0].get("tool_use_id", ""),
                "content": content[0].get("content", ""),
            }

        # Handle assistant messages with tool calls
        if role == "assistant" and msg.get("tool_calls"):
            result: dict[str, Any] = {"role": "assistant"}
            if content:
                result["content"] = content
            result["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["input"]),
                    },
                }
                for tc in msg["tool_calls"]
            ]
            return result

        return {"role": role, "content": content}

    def _format_tool(self, tool: dict[str, Any]) -> dict[str, Any]:
        """Convert tool schema to OpenAI format."""
        return {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
            },
        }

    def _parse_response(self, response: Any) -> dict[str, Any]:
        """Parse OpenAI response to common format."""
        choice = response.choices[0]
        message = choice.message

        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": json.loads(tc.function.arguments),
                }
                for tc in message.tool_calls
            ]

        return {
            "content": message.content,
            "tool_calls": tool_calls,
            "stop_reason": choice.finish_reason,
            "usage": {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            },
        }

    def format_tool_result(self, tool_call_id: str, result: Any) -> dict[str, Any]:
        """Format tool result for OpenAI."""
        content = json.dumps(result) if not isinstance(result, str) else result
        return {
            "type": "tool_result",
            "tool_use_id": tool_call_id,
            "content": content,
        }
