"""Abstraction layer for LLM providers (Anthropic Claude, Ollama)."""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import aiohttp
import anthropic

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
    # ── Blueprint tools ──
    {
        "name": "list_blueprints",
        "description": "List all blueprints in the blueprints directory. Returns name, description, and domain for each.",
        "parameters": {
            "domain": {"type": "string", "description": "Blueprint domain: 'automation' or 'script'. Defaults to 'automation'.", "required": False},
        },
    },
    {
        "name": "read_blueprint",
        "description": "Read the full YAML content of a blueprint file. Use this before editing to see the current content.",
        "parameters": {
            "blueprint_name": {"type": "string", "description": "The blueprint filename without .yaml extension.", "required": True},
            "domain": {"type": "string", "description": "Blueprint domain: 'automation' or 'script'. Defaults to 'automation'.", "required": False},
        },
    },
    {
        "name": "create_blueprint",
        "description": "Create a new blueprint file. The YAML must include a 'blueprint:' key with name, description, domain, and input definitions. Use !input tags for configurable parameters.",
        "parameters": {
            "blueprint_name": {"type": "string", "description": "Filename for the blueprint (lowercase, underscores, no .yaml extension).", "required": True},
            "yaml_content": {"type": "string", "description": "The complete blueprint YAML content including blueprint metadata, triggers, conditions, and actions.", "required": True},
            "domain": {"type": "string", "description": "Blueprint domain: 'automation' or 'script'. Defaults to 'automation'.", "required": False},
        },
    },
    {
        "name": "edit_blueprint",
        "description": "Edit an existing blueprint by replacing its content. Changes propagate to ALL automations/scripts using this blueprint after reload.",
        "parameters": {
            "blueprint_name": {"type": "string", "description": "The blueprint filename without .yaml extension.", "required": True},
            "yaml_content": {"type": "string", "description": "The complete updated blueprint YAML content.", "required": True},
            "domain": {"type": "string", "description": "Blueprint domain: 'automation' or 'script'. Defaults to 'automation'.", "required": False},
        },
    },
    {
        "name": "delete_blueprint",
        "description": "Delete a blueprint file. Warning: automations/scripts using this blueprint will break.",
        "parameters": {
            "blueprint_name": {"type": "string", "description": "The blueprint filename without .yaml extension.", "required": True},
            "domain": {"type": "string", "description": "Blueprint domain: 'automation' or 'script'. Defaults to 'automation'.", "required": False},
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

    # ── Async variants (default: delegate to sync for backwards compatibility) ──

    async def async_create_message(
        self,
        model: str,
        system_prompt: str,
        messages: list[dict],
        max_tokens: int,
        tools: bool = True,
    ) -> LLMResponse:
        """Async variant. Default implementation calls sync version."""
        return self.create_message(model, system_prompt, messages, max_tokens, tools)

    async def async_create_simple_message(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
    ) -> str:
        """Async variant. Default implementation calls sync version."""
        return self.create_simple_message(model, prompt, max_tokens)

    async def async_validate_connection(self, model: str) -> None:
        """Async variant. Default implementation calls sync version."""
        return self.validate_connection(model)

    @property
    def is_async(self) -> bool:
        """Whether this client supports native async. Override in subclasses."""
        return False


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
    """Ollama local LLM client using native async aiohttp."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        timeout: int = 120,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> None:
        """Initialize."""
        self._host = host.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._temperature = temperature
        self._top_p = top_p

    @property
    def is_async(self) -> bool:
        """Ollama client is natively async."""
        return True

    def _next_tool_call_id(self) -> str:
        """Generate a unique tool call ID."""
        return f"call_{uuid.uuid4().hex[:12]}"

    def _build_options(self, max_tokens: int) -> dict:
        """Build the options dict for Ollama API."""
        options: dict[str, Any] = {"num_predict": max_tokens}
        if self._temperature is not None:
            options["temperature"] = self._temperature
        if self._top_p is not None:
            options["top_p"] = self._top_p
        return options

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

    # ── Sync methods raise (this client is async-only) ──

    def create_message(self, *args: Any, **kwargs: Any) -> LLMResponse:
        """Not supported. Use async_create_message."""
        raise NotImplementedError("OllamaClient is async-only. Use async_create_message.")

    def create_simple_message(self, *args: Any, **kwargs: Any) -> str:
        """Not supported. Use async_create_simple_message."""
        raise NotImplementedError("OllamaClient is async-only. Use async_create_simple_message.")

    def validate_connection(self, *args: Any, **kwargs: Any) -> None:
        """Not supported. Use async_validate_connection."""
        raise NotImplementedError("OllamaClient is async-only. Use async_validate_connection.")

    # ── Async implementations ──

    async def async_create_message(
        self,
        model: str,
        system_prompt: str,
        messages: list[dict],
        max_tokens: int,
        tools: bool = True,
    ) -> LLMResponse:
        """Send a message to Ollama with streaming for better timeout handling."""
        ollama_messages = [{"role": "system", "content": system_prompt}] + messages

        payload: dict[str, Any] = {
            "model": model,
            "messages": ollama_messages,
            "options": self._build_options(max_tokens),
        }
        if tools:
            payload["tools"] = self._to_ollama_tools()

        data = await self._post_stream("/api/chat", payload)

        message = data.get("message", {})
        text = message.get("content")
        tool_calls = []

        for tc in message.get("tool_calls", []):
            func = tc.get("function", {})
            tool_calls.append({
                "id": self._next_tool_call_id(),
                "name": func.get("name", ""),
                "arguments": func.get("arguments", {}),
            })

        return LLMResponse(
            text=text if text else None,
            tool_calls=tool_calls,
            raw_assistant_message=message,
        )

    async def async_create_simple_message(
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
            "options": self._build_options(max_tokens),
        }
        data = await self._post("/api/chat", payload)
        return data.get("message", {}).get("content", "").strip()

    def add_tool_results(
        self,
        messages: list[dict],
        response: LLMResponse,
        tool_results: list[dict],
    ) -> None:
        """Add tool results in Ollama/OpenAI format."""
        assistant_msg = response.raw_assistant_message
        messages.append(assistant_msg)

        for result in tool_results:
            messages.append({
                "role": "tool",
                "content": result["content"],
            })

    async def async_validate_connection(self, model: str) -> None:
        """Validate Ollama is reachable and model is available.

        Raises ConnectionError if the host is unreachable.
        Raises ValueError if the model is not found.
        """
        try:
            data = await self._get("/api/tags")
        except aiohttp.ClientConnectorError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self._host}. "
                "Make sure Ollama is running."
            )
        except aiohttp.ClientError as err:
            raise ConnectionError(f"Ollama connection error: {err}") from err

        models = [m["name"] for m in data.get("models", [])]
        model_base = model.split(":")[0]
        available = [m.split(":")[0] for m in models]

        if model_base not in available:
            raise ValueError(
                f"Model '{model}' not found in Ollama. "
                f"Available: {', '.join(models)}. "
                f"Pull it with: ollama pull {model}"
            )

    async def async_fetch_models(self) -> dict[str, str]:
        """Fetch installed models from Ollama. Returns {model_name: display_label}."""
        try:
            data = await self._get("/api/tags")
            models = {}
            for m in data.get("models", []):
                name = m["name"]
                detail = m.get("details", {})
                param_size = detail.get("parameter_size", "")
                if param_size:
                    label = f"{name} ({param_size})"
                else:
                    size_gb = m.get("size", 0) / (1024**3)
                    label = f"{name} ({size_gb:.1f} GB)" if size_gb > 0 else name
                models[name] = label
            return models
        except Exception:
            return {}

    # ── Private helpers ──

    def _describe_error(self, err: Exception | None) -> str:
        """Build a human-readable error message from an aiohttp exception."""
        if isinstance(err, aiohttp.ClientConnectorError):
            return (
                f"Cannot connect to Ollama at {self._host}. "
                "Make sure Ollama is running and the host URL is correct."
            )
        if isinstance(err, aiohttp.ClientResponseError):
            if err.status == 404:
                return (
                    f"Ollama at {self._host} returned 404 (not found). "
                    "The model may not be pulled or the API path is wrong."
                )
            return f"Ollama at {self._host} returned HTTP {err.status}: {err.message}"
        return f"Ollama request to {self._host} failed: {err}"


    async def _post(self, path: str, payload: dict, retries: int = 2) -> dict:
        """POST to Ollama with retry on timeout."""
        url = f"{self._host}{path}"
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                async with aiohttp.ClientSession(timeout=self._timeout) as session:
                    async with session.post(url, json=payload) as resp:
                        resp.raise_for_status()
                        return await resp.json()
            except TimeoutError as err:
                last_error = err
                if attempt < retries:
                    _LOGGER.warning(
                        "Ollama request timed out (attempt %d/%d), retrying...",
                        attempt + 1, retries + 1,
                    )
                    continue
            except aiohttp.ClientConnectorError as err:
                last_error = err
                break
            except aiohttp.ClientResponseError as err:
                last_error = err
                break
            except aiohttp.ClientError as err:
                last_error = err
                break
        raise ConnectionError(
            self._describe_error(last_error)
        ) from last_error

    async def _post_stream(self, path: str, payload: dict, retries: int = 1) -> dict:
        """POST to Ollama with streaming, accumulating the full response.

        Streaming gives better timeout behavior -- tokens arriving prove the
        model is working, preventing premature timeout on slow generations.
        """
        url = f"{self._host}{path}"
        payload = {**payload, "stream": True}
        last_error: Exception | None = None

        for attempt in range(retries + 1):
            try:
                accumulated_content = ""
                accumulated_tool_calls: list[dict] = []
                final_message: dict = {}

                async with aiohttp.ClientSession(timeout=self._timeout) as session:
                    async with session.post(url, json=payload) as resp:
                        resp.raise_for_status()
                        async for line in resp.content:
                            line_str = line.strip()
                            if not line_str:
                                continue
                            try:
                                chunk = json.loads(line_str)
                            except json.JSONDecodeError:
                                continue

                            message = chunk.get("message", {})
                            accumulated_content += message.get("content", "")

                            if message.get("tool_calls"):
                                accumulated_tool_calls.extend(message["tool_calls"])

                            if chunk.get("done"):
                                final_message = message
                                break

                final_message["content"] = accumulated_content
                if accumulated_tool_calls:
                    final_message["tool_calls"] = accumulated_tool_calls

                return {"message": final_message}

            except TimeoutError as err:
                last_error = err
                if attempt < retries:
                    _LOGGER.warning(
                        "Ollama streaming timed out (attempt %d/%d), retrying...",
                        attempt + 1, retries + 1,
                    )
                    continue
            except aiohttp.ClientConnectorError as err:
                last_error = err
                break
            except aiohttp.ClientResponseError as err:
                last_error = err
                break
            except aiohttp.ClientError as err:
                last_error = err
                break

        raise ConnectionError(
            self._describe_error(last_error)
        ) from last_error

    async def _get(self, path: str) -> dict:
        """GET from Ollama."""
        url = f"{self._host}{path}"
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        ) as session:
            async with session.get(url) as resp:
                resp.raise_for_status()
                return await resp.json()


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
            temperature=kwargs.get("temperature"),
            top_p=kwargs.get("top_p"),
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
