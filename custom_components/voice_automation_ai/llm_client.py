"""Abstraction layer for LLM providers (Anthropic Claude, Ollama)."""
from __future__ import annotations

import json
import logging
from typing import Any

import anthropic
import requests

_LOGGER = logging.getLogger(__name__)

# Unified tool format used internally - converted per provider
TOOL_DEFINITIONS = [
    {
        "name": "list_automations",
        "description": "List all automations defined in automations.yaml.",
        "parameters": {},
    },
    {
        "name": "create_automation",
        "description": "Create a new Home Assistant automation.",
        "parameters": {
            "yaml_content": {
                "type": "string",
                "description": "The automation as a YAML string with alias, description, mode, trigger, condition, and action.",
                "required": True,
            },
        },
    },
    {
        "name": "edit_automation",
        "description": "Edit an existing automation by ID.",
        "parameters": {
            "automation_id": {"type": "string", "description": "The automation ID.", "required": True},
            "yaml_content": {"type": "string", "description": "The complete updated automation as YAML.", "required": True},
        },
    },
    {
        "name": "delete_automation",
        "description": "Delete an automation by ID.",
        "parameters": {
            "automation_id": {"type": "string", "description": "The automation ID.", "required": True},
        },
    },
    {
        "name": "list_scripts",
        "description": "List all scripts defined in scripts.yaml.",
        "parameters": {},
    },
    {
        "name": "create_script",
        "description": "Create a new Home Assistant script.",
        "parameters": {
            "script_name": {"type": "string", "description": "Script identifier (lowercase, underscores).", "required": True},
            "yaml_content": {"type": "string", "description": "The script body as YAML.", "required": True},
        },
    },
    {
        "name": "edit_script",
        "description": "Edit an existing script by name.",
        "parameters": {
            "script_name": {"type": "string", "description": "The script name.", "required": True},
            "yaml_content": {"type": "string", "description": "The complete updated script body as YAML.", "required": True},
        },
    },
    {
        "name": "delete_script",
        "description": "Delete a script by name.",
        "parameters": {
            "script_name": {"type": "string", "description": "The script name.", "required": True},
        },
    },
    {
        "name": "list_scenes",
        "description": "List all scenes defined in scenes.yaml.",
        "parameters": {},
    },
    {
        "name": "create_scene",
        "description": "Create a new Home Assistant scene.",
        "parameters": {
            "yaml_content": {"type": "string", "description": "The scene as YAML with name and entities.", "required": True},
        },
    },
    {
        "name": "edit_scene",
        "description": "Edit an existing scene by ID.",
        "parameters": {
            "scene_id": {"type": "string", "description": "The scene ID.", "required": True},
            "yaml_content": {"type": "string", "description": "The complete updated scene as YAML.", "required": True},
        },
    },
    {
        "name": "delete_scene",
        "description": "Delete a scene by ID.",
        "parameters": {
            "scene_id": {"type": "string", "description": "The scene ID.", "required": True},
        },
    },
    {
        "name": "call_service",
        "description": "Call a Home Assistant service to control devices. Examples: turn on/off lights, lock/unlock doors, set climate temperature, play/pause media, open/close covers.",
        "parameters": {
            "domain": {"type": "string", "description": "The service domain (e.g. light, switch, climate, lock, cover, media_player, fan).", "required": True},
            "service": {"type": "string", "description": "The service to call (e.g. turn_on, turn_off, toggle, lock, unlock, set_temperature, open_cover, close_cover).", "required": True},
            "entity_id": {"type": "string", "description": "The entity ID to target (e.g. light.living_room).", "required": True},
            "service_data": {"type": "string", "description": "Optional JSON string of additional service data (e.g. {\"brightness\": 255, \"color_name\": \"red\"}).", "required": False},
        },
    },
    {
        "name": "get_entity_state",
        "description": "Get the current state and attributes of a Home Assistant entity.",
        "parameters": {
            "entity_id": {"type": "string", "description": "The entity ID to query (e.g. light.living_room, sensor.temperature).", "required": True},
        },
    },
]


class LLMResponse:
    """Unified response from any LLM provider."""

    def __init__(
        self,
        text: str | None = None,
        tool_calls: list[dict] | None = None,
        raw_assistant_message: Any = None,
    ) -> None:
        """Initialize."""
        self.text = text
        self.tool_calls = tool_calls or []
        self.raw_assistant_message = raw_assistant_message

    @property
    def has_tool_calls(self) -> bool:
        """Return True if the response contains tool calls."""
        return len(self.tool_calls) > 0


class BaseLLMClient:
    """Base class for LLM clients."""

    def create_message(
        self,
        model: str,
        system_prompt: str,
        messages: list[dict],
        max_tokens: int,
        tools: bool = True,
    ) -> LLMResponse:
        """Send a message and return a unified response."""
        raise NotImplementedError

    def create_simple_message(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
    ) -> str:
        """Send a simple prompt and return text (no tools). Used for YAML generation."""
        raise NotImplementedError

    def add_tool_results(
        self,
        messages: list[dict],
        response: LLMResponse,
        tool_results: list[dict],
    ) -> None:
        """Add assistant message and tool results to the conversation in provider format."""
        raise NotImplementedError

    def validate_connection(self, model: str) -> None:
        """Validate the connection works. Raises on failure."""
        raise NotImplementedError


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude API client."""

    def __init__(self, api_key: str, timeout: int = 30) -> None:
        """Initialize."""
        self._client = anthropic.Anthropic(api_key=api_key)
        self._timeout = timeout

    def _to_anthropic_tools(self) -> list[dict]:
        """Convert unified tool defs to Anthropic format."""
        tools = []
        for tool in TOOL_DEFINITIONS:
            properties = {}
            required = []
            for param_name, param_def in tool["parameters"].items():
                properties[param_name] = {
                    "type": param_def["type"],
                    "description": param_def["description"],
                }
                if param_def.get("required"):
                    required.append(param_name)

            tools.append({
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            })
        return tools

    def create_message(
        self,
        model: str,
        system_prompt: str,
        messages: list[dict],
        max_tokens: int,
        tools: bool = True,
    ) -> LLMResponse:
        """Send a message to Claude."""
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = self._to_anthropic_tools()

        response = self._client.messages.create(**kwargs, timeout=self._timeout)

        text_parts = []
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input,
                })

        return LLMResponse(
            text=" ".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            raw_assistant_message=[
                {"type": "text", "text": block.text} if block.type == "text"
                else {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
                for block in response.content
            ],
        )

    def create_simple_message(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
    ) -> str:
        """Send a simple prompt to Claude."""
        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            timeout=self._timeout,
        )
        return response.content[0].text.strip()

    def add_tool_results(
        self,
        messages: list[dict],
        response: LLMResponse,
        tool_results: list[dict],
    ) -> None:
        """Add tool results in Anthropic format."""
        messages.append({"role": "assistant", "content": response.raw_assistant_message})
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": result["tool_call_id"],
                    "content": result["content"],
                }
                for result in tool_results
            ],
        })

    def validate_connection(self, model: str) -> None:
        """Validate API key by making a minimal request."""
        try:
            self._client.messages.create(
                model=model,
                max_tokens=10,
                messages=[{"role": "user", "content": "test"}],
            )
        except anthropic.AuthenticationError:
            raise
        except Exception as err:
            raise ConnectionError(f"Failed to connect to Anthropic API: {err}") from err


class OllamaClient(BaseLLMClient):
    """Ollama local LLM client (OpenAI-compatible API)."""

    def __init__(self, host: str = "http://localhost:11434", timeout: int = 120) -> None:
        """Initialize."""
        self._host = host.rstrip("/")
        self._timeout = timeout

    def _to_ollama_tools(self) -> list[dict]:
        """Convert unified tool defs to Ollama/OpenAI function-calling format."""
        tools = []
        for tool in TOOL_DEFINITIONS:
            properties = {}
            required = []
            for param_name, param_def in tool["parameters"].items():
                properties[param_name] = {
                    "type": param_def["type"],
                    "description": param_def["description"],
                }
                if param_def.get("required"):
                    required.append(param_name)

            tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            })
        return tools

    def create_message(
        self,
        model: str,
        system_prompt: str,
        messages: list[dict],
        max_tokens: int,
        tools: bool = True,
    ) -> LLMResponse:
        """Send a message to Ollama."""
        ollama_messages = [{"role": "system", "content": system_prompt}] + messages

        payload: dict[str, Any] = {
            "model": model,
            "messages": ollama_messages,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        if tools:
            payload["tools"] = self._to_ollama_tools()

        resp = requests.post(
            f"{self._host}/api/chat",
            json=payload,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        message = data.get("message", {})
        text = message.get("content")
        tool_calls = []

        for tc in message.get("tool_calls", []):
            func = tc.get("function", {})
            tool_calls.append({
                "id": func.get("name", ""),  # Ollama doesn't always provide IDs
                "name": func.get("name", ""),
                "arguments": func.get("arguments", {}),
            })

        return LLMResponse(
            text=text if text else None,
            tool_calls=tool_calls,
            raw_assistant_message=message,
        )

    def create_simple_message(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
    ) -> str:
        """Send a simple prompt to Ollama."""
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"num_predict": max_tokens},
        }

        resp = requests.post(
            f"{self._host}/api/chat",
            json=payload,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "").strip()

    def add_tool_results(
        self,
        messages: list[dict],
        response: LLMResponse,
        tool_results: list[dict],
    ) -> None:
        """Add tool results in Ollama/OpenAI format."""
        # Add assistant message with tool calls
        assistant_msg = response.raw_assistant_message
        messages.append(assistant_msg)

        # Add each tool result as a separate message
        for result in tool_results:
            messages.append({
                "role": "tool",
                "content": result["content"],
            })

    def validate_connection(self, model: str) -> None:
        """Validate Ollama is reachable and model is available."""
        try:
            resp = requests.get(f"{self._host}/api/tags", timeout=10)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]

            # Strip :latest tag for comparison
            model_base = model.split(":")[0]
            available = [m.split(":")[0] for m in models]

            if model_base not in available:
                raise ConnectionError(
                    f"Model '{model}' not found in Ollama. "
                    f"Available: {', '.join(models)}. "
                    f"Pull it with: ollama pull {model}"
                )
        except requests.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self._host}. "
                "Make sure Ollama is running."
            )
        except requests.RequestException as err:
            raise ConnectionError(f"Ollama connection error: {err}") from err


def create_llm_client(provider: str, **kwargs: Any) -> BaseLLMClient:
    """Factory function to create the appropriate LLM client."""
    if provider == "anthropic":
        return AnthropicClient(
            api_key=kwargs["api_key"],
            timeout=kwargs.get("timeout", 30),
        )
    elif provider == "ollama":
        return OllamaClient(
            host=kwargs.get("host", "http://localhost:11434"),
            timeout=kwargs.get("timeout", 120),
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
