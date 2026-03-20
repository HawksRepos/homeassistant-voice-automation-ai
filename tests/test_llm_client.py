"""Tests for LLM client abstraction layer."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from custom_components.voice_automation_ai.llm_client import (
    TOOL_DEFINITIONS,
    AnthropicClient,
    BaseLLMClient,
    LLMResponse,
    OllamaClient,
    create_llm_client,
)


# ── Async helper for mocking aiohttp ──


class _AsyncContextManager:
    """Helper to mock async context managers (aiohttp sessions/responses)."""

    def __init__(self, return_value):
        self._return_value = return_value

    async def __aenter__(self):
        return self._return_value

    async def __aexit__(self, *args):
        pass


class _AsyncLineIterator:
    """Mock async iterator for streaming response lines."""

    def __init__(self, lines: list[bytes]):
        self._items = iter(lines)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration


def _mock_aiohttp_session(response_json=None, stream_lines=None, get_json=None, raise_error=None):
    """Create a mock aiohttp.ClientSession for testing.

    Args:
        response_json: JSON dict for non-streaming POST response
        stream_lines: list of bytes for streaming response
        get_json: JSON dict for GET response
        raise_error: exception to raise on request
    """
    mock_session = MagicMock()

    if raise_error:
        mock_session.post = MagicMock(side_effect=raise_error)
        mock_session.get = MagicMock(side_effect=raise_error)
    else:
        if response_json is not None:
            mock_resp = MagicMock()
            mock_resp.json = AsyncMock(return_value=response_json)
            mock_resp.raise_for_status = MagicMock()
            mock_resp.status = 200
            mock_session.post = MagicMock(return_value=_AsyncContextManager(mock_resp))

        if stream_lines is not None:
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.status = 200
            mock_resp.content = _AsyncLineIterator(stream_lines)
            mock_session.post = MagicMock(return_value=_AsyncContextManager(mock_resp))

        if get_json is not None:
            mock_get_resp = MagicMock()
            mock_get_resp.json = AsyncMock(return_value=get_json)
            mock_get_resp.raise_for_status = MagicMock()
            mock_get_resp.status = 200
            mock_session.get = MagicMock(return_value=_AsyncContextManager(mock_get_resp))

    return _AsyncContextManager(mock_session)


# ── LLMResponse ──


class TestLLMResponse:
    """Tests for the unified LLMResponse class."""

    def test_empty_response(self):
        r = LLMResponse()
        assert r.text is None
        assert r.tool_calls == []
        assert r.has_tool_calls is False
        assert r.raw_assistant_message is None

    def test_text_only_response(self):
        r = LLMResponse(text="Hello world")
        assert r.text == "Hello world"
        assert r.has_tool_calls is False

    def test_tool_calls_response(self):
        calls = [{"id": "tc1", "name": "list_automations", "arguments": {}}]
        r = LLMResponse(tool_calls=calls)
        assert r.has_tool_calls is True
        assert len(r.tool_calls) == 1
        assert r.tool_calls[0]["name"] == "list_automations"

    def test_mixed_response(self):
        calls = [{"id": "tc1", "name": "call_service", "arguments": {"domain": "light"}}]
        r = LLMResponse(text="Turning on the light", tool_calls=calls)
        assert r.text == "Turning on the light"
        assert r.has_tool_calls is True

    def test_raw_assistant_message_preserved(self):
        raw = {"role": "assistant", "content": "test"}
        r = LLMResponse(raw_assistant_message=raw)
        assert r.raw_assistant_message is raw


# ── TOOL_DEFINITIONS ──


class TestToolDefinitions:
    """Tests for the unified TOOL_DEFINITIONS structure."""

    def test_all_19_tools_defined(self):
        assert len(TOOL_DEFINITIONS) == 19

    def test_every_tool_has_required_keys(self):
        for tool in TOOL_DEFINITIONS:
            assert "name" in tool, f"Tool missing 'name': {tool}"
            assert "description" in tool, f"Tool {tool['name']} missing 'description'"
            assert "parameters" in tool, f"Tool {tool['name']} missing 'parameters'"

    def test_call_service_has_all_params(self):
        cs = next(t for t in TOOL_DEFINITIONS if t["name"] == "call_service")
        params = cs["parameters"]
        assert "domain" in params
        assert "service" in params
        assert "entity_id" in params
        assert "service_data" in params
        assert params["service_data"].get("required") is False

    def test_get_entity_state_has_entity_id(self):
        ges = next(t for t in TOOL_DEFINITIONS if t["name"] == "get_entity_state")
        assert "entity_id" in ges["parameters"]
        assert ges["parameters"]["entity_id"]["required"] is True

    def test_tool_names_are_unique(self):
        names = [t["name"] for t in TOOL_DEFINITIONS]
        assert len(names) == len(set(names))

    def test_blueprint_tools_exist(self):
        bp_tools = [t["name"] for t in TOOL_DEFINITIONS if "blueprint" in t["name"]]
        assert set(bp_tools) == {
            "list_blueprints", "read_blueprint",
            "create_blueprint", "edit_blueprint", "delete_blueprint",
        }


# ── BaseLLMClient async defaults ──


class TestBaseLLMClientAsync:
    """Test async method defaults on BaseLLMClient."""

    def test_is_async_default_false(self):
        client = BaseLLMClient()
        assert client.is_async is False


# ── Anthropic tool conversion ──


class TestAnthropicToolConversion:
    """Test Anthropic tool format conversion."""

    def test_to_anthropic_tools_format(self):
        client = AnthropicClient.__new__(AnthropicClient)
        tools = client._to_anthropic_tools()

        assert len(tools) == 19
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            schema = tool["input_schema"]
            assert schema["type"] == "object"
            assert "properties" in schema
            assert "required" in schema

    def test_anthropic_call_service_schema(self):
        client = AnthropicClient.__new__(AnthropicClient)
        tools = client._to_anthropic_tools()
        cs = next(t for t in tools if t["name"] == "call_service")
        schema = cs["input_schema"]
        assert "domain" in schema["properties"]
        assert "service" in schema["properties"]
        assert "entity_id" in schema["properties"]
        assert "domain" in schema["required"]
        assert "service" in schema["required"]
        assert "entity_id" in schema["required"]
        # service_data is optional
        assert "service_data" not in schema["required"]

    def test_anthropic_empty_params_tool(self):
        client = AnthropicClient.__new__(AnthropicClient)
        tools = client._to_anthropic_tools()
        la = next(t for t in tools if t["name"] == "list_automations")
        assert la["input_schema"]["properties"] == {}
        assert la["input_schema"]["required"] == []


# ── Anthropic add_tool_results ──


class TestAnthropicAddToolResults:
    """Test Anthropic tool result message formatting."""

    def test_appends_two_messages(self):
        client = AnthropicClient.__new__(AnthropicClient)
        messages = []
        response = LLMResponse(
            raw_assistant_message=[
                {"type": "tool_use", "id": "tc1", "name": "list_automations", "input": {}}
            ]
        )
        tool_results = [
            {"tool_call_id": "tc1", "content": '{"success": true}'}
        ]

        client.add_tool_results(messages, response, tool_results)

        assert len(messages) == 2
        assert messages[0]["role"] == "assistant"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"][0]["type"] == "tool_result"
        assert messages[1]["content"][0]["tool_use_id"] == "tc1"


# ── Anthropic create_message ──


class TestAnthropicCreateMessage:
    """Test Anthropic create_message with mocked API."""

    @patch("custom_components.voice_automation_ai.llm_client.anthropic")
    def test_text_response(self, mock_anthropic):
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Hello!"

        mock_response = MagicMock()
        mock_response.content = [text_block]
        mock_client.messages.create.return_value = mock_response

        client = AnthropicClient(api_key="sk-test")
        result = client.create_message(
            model="claude-sonnet-4-5-20250929",
            system_prompt="test",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
            tools=False,
        )

        assert result.text == "Hello!"
        assert result.has_tool_calls is False

    @patch("custom_components.voice_automation_ai.llm_client.anthropic")
    def test_tool_use_response(self, mock_anthropic):
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "tc_123"
        tool_block.name = "list_automations"
        tool_block.input = {}

        mock_response = MagicMock()
        mock_response.content = [tool_block]
        mock_client.messages.create.return_value = mock_response

        client = AnthropicClient(api_key="sk-test")
        result = client.create_message(
            model="claude-sonnet-4-5-20250929",
            system_prompt="test",
            messages=[{"role": "user", "content": "list automations"}],
            max_tokens=100,
        )

        assert result.has_tool_calls is True
        assert result.tool_calls[0]["name"] == "list_automations"
        assert result.tool_calls[0]["id"] == "tc_123"


# ── Ollama tool conversion ──


class TestOllamaToolConversion:
    """Test Ollama/OpenAI tool format conversion."""

    def test_to_ollama_tools_format(self):
        client = OllamaClient.__new__(OllamaClient)
        tools = client._to_ollama_tools()

        assert len(tools) == 19
        for tool in tools:
            assert tool["type"] == "function"
            assert "function" in tool
            func = tool["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func
            assert func["parameters"]["type"] == "object"

    def test_ollama_call_service_schema(self):
        client = OllamaClient.__new__(OllamaClient)
        tools = client._to_ollama_tools()
        cs = next(
            t for t in tools
            if t["function"]["name"] == "call_service"
        )
        params = cs["function"]["parameters"]
        assert "domain" in params["properties"]
        assert "service_data" not in params["required"]


# ── Ollama add_tool_results ──


class TestOllamaAddToolResults:
    """Test Ollama tool result message formatting."""

    def test_appends_messages(self):
        client = OllamaClient.__new__(OllamaClient)
        messages = []
        raw_msg = {"role": "assistant", "content": "", "tool_calls": []}
        response = LLMResponse(raw_assistant_message=raw_msg)
        tool_results = [
            {"tool_call_id": "tc1", "content": '{"success": true}'},
            {"tool_call_id": "tc2", "content": '{"success": true}'},
        ]

        client.add_tool_results(messages, response, tool_results)

        # 1 assistant message + 2 tool result messages
        assert len(messages) == 3
        assert messages[0]["role"] == "assistant"
        assert messages[1]["role"] == "tool"
        assert messages[2]["role"] == "tool"


# ── Ollama is_async ──


class TestOllamaIsAsync:
    """Test OllamaClient is_async property."""

    def test_is_async_true(self):
        client = OllamaClient()
        assert client.is_async is True

    def test_sync_methods_raise(self):
        client = OllamaClient()
        with pytest.raises(NotImplementedError):
            client.create_message()
        with pytest.raises(NotImplementedError):
            client.create_simple_message()
        with pytest.raises(NotImplementedError):
            client.validate_connection()


# ── Ollama _build_options ──


class TestOllamaBuildOptions:
    """Test OllamaClient._build_options."""

    def test_basic_options(self):
        client = OllamaClient()
        options = client._build_options(4096)
        assert options["num_predict"] == 4096
        assert "temperature" not in options
        assert "top_p" not in options

    def test_with_temperature(self):
        client = OllamaClient(temperature=0.7)
        options = client._build_options(4096)
        assert options["temperature"] == 0.7

    def test_with_top_p(self):
        client = OllamaClient(top_p=0.9)
        options = client._build_options(4096)
        assert options["top_p"] == 0.9

    def test_with_both(self):
        client = OllamaClient(temperature=0.5, top_p=0.8)
        options = client._build_options(1024)
        assert options["num_predict"] == 1024
        assert options["temperature"] == 0.5
        assert options["top_p"] == 0.8


# ── Ollama tool call IDs ──


class TestOllamaToolCallIds:
    """Test OllamaClient generates unique tool call IDs."""

    def test_ids_are_unique(self):
        client = OllamaClient()
        ids = {client._next_tool_call_id() for _ in range(100)}
        assert len(ids) == 100

    def test_id_format(self):
        client = OllamaClient()
        tc_id = client._next_tool_call_id()
        assert tc_id.startswith("call_")
        assert len(tc_id) == 17  # "call_" + 12 hex chars


# ── Ollama async_create_message ──


class TestOllamaAsyncCreateMessage:
    """Test Ollama async_create_message with mocked aiohttp."""

    async def test_text_response(self):
        # Build streaming response (NDJSON lines)
        stream_lines = [
            json.dumps({"message": {"role": "assistant", "content": "Hello "}, "done": False}).encode() + b"\n",
            json.dumps({"message": {"role": "assistant", "content": "world!"}, "done": True}).encode() + b"\n",
        ]
        mock_session = _mock_aiohttp_session(stream_lines=stream_lines)

        with patch("custom_components.voice_automation_ai.llm_client.aiohttp.ClientSession", return_value=mock_session):
            client = OllamaClient(host="http://localhost:11434")
            result = await client.async_create_message(
                model="llama3.1",
                system_prompt="test",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=100,
                tools=False,
            )

        assert result.text == "Hello world!"
        assert result.has_tool_calls is False

    async def test_tool_call_response(self):
        # Tool calls come in the final message
        final_msg = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "call_service",
                            "arguments": {
                                "domain": "light",
                                "service": "turn_on",
                                "entity_id": "light.living_room",
                            },
                        }
                    }
                ],
            },
            "done": True,
        }
        stream_lines = [json.dumps(final_msg).encode() + b"\n"]
        mock_session = _mock_aiohttp_session(stream_lines=stream_lines)

        with patch("custom_components.voice_automation_ai.llm_client.aiohttp.ClientSession", return_value=mock_session):
            client = OllamaClient()
            result = await client.async_create_message(
                model="llama3.1",
                system_prompt="test",
                messages=[{"role": "user", "content": "turn on light"}],
                max_tokens=100,
            )

        assert result.has_tool_calls is True
        assert result.tool_calls[0]["name"] == "call_service"
        assert result.tool_calls[0]["arguments"]["domain"] == "light"
        # ID should be a generated unique ID
        assert result.tool_calls[0]["id"].startswith("call_")

    async def test_empty_content_returns_none_text(self):
        stream_lines = [
            json.dumps({"message": {"role": "assistant", "content": ""}, "done": True}).encode() + b"\n",
        ]
        mock_session = _mock_aiohttp_session(stream_lines=stream_lines)

        with patch("custom_components.voice_automation_ai.llm_client.aiohttp.ClientSession", return_value=mock_session):
            client = OllamaClient()
            result = await client.async_create_message(
                model="llama3.1",
                system_prompt="test",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=100,
                tools=False,
            )

        assert result.text is None


# ── Ollama async_create_simple_message ──


class TestOllamaAsyncCreateSimpleMessage:
    """Test Ollama async_create_simple_message."""

    async def test_returns_stripped_text(self):
        response_json = {
            "message": {"role": "assistant", "content": "  Hello from Ollama!  "}
        }
        mock_session = _mock_aiohttp_session(response_json=response_json)

        with patch("custom_components.voice_automation_ai.llm_client.aiohttp.ClientSession", return_value=mock_session):
            client = OllamaClient()
            result = await client.async_create_simple_message(
                model="llama3.1",
                prompt="hi",
                max_tokens=100,
            )

        assert result == "Hello from Ollama!"


# ── Ollama async_validate_connection ──


class TestOllamaAsyncValidateConnection:
    """Test Ollama async connection validation."""

    async def test_model_available(self):
        get_json = {
            "models": [{"name": "llama3.1:latest"}, {"name": "mistral:latest"}]
        }
        mock_session = _mock_aiohttp_session(get_json=get_json)

        with patch("custom_components.voice_automation_ai.llm_client.aiohttp.ClientSession", return_value=mock_session):
            client = OllamaClient()
            await client.async_validate_connection("llama3.1")  # should not raise

    async def test_model_missing(self):
        get_json = {
            "models": [{"name": "mistral:latest"}]
        }
        mock_session = _mock_aiohttp_session(get_json=get_json)

        with patch("custom_components.voice_automation_ai.llm_client.aiohttp.ClientSession", return_value=mock_session):
            client = OllamaClient()
            with pytest.raises(ValueError, match="not found"):
                await client.async_validate_connection("llama3.1")

    async def test_connection_refused(self):
        mock_session = _mock_aiohttp_session(
            raise_error=aiohttp.ClientConnectorError(
                MagicMock(), OSError("refused")
            )
        )

        with patch("custom_components.voice_automation_ai.llm_client.aiohttp.ClientSession", return_value=mock_session):
            client = OllamaClient()
            with pytest.raises(ConnectionError, match="Cannot connect"):
                await client.async_validate_connection("llama3.1")


# ── Ollama async_fetch_models ──


class TestOllamaAsyncFetchModels:
    """Test Ollama model discovery."""

    async def test_fetches_models(self):
        get_json = {
            "models": [
                {"name": "llama3.1:latest", "size": 4_000_000_000, "details": {"parameter_size": "8B"}},
                {"name": "mistral:latest", "size": 3_500_000_000, "details": {}},
            ]
        }
        mock_session = _mock_aiohttp_session(get_json=get_json)

        with patch("custom_components.voice_automation_ai.llm_client.aiohttp.ClientSession", return_value=mock_session):
            client = OllamaClient()
            models = await client.async_fetch_models()

        assert "llama3.1:latest" in models
        assert "mistral:latest" in models
        assert "8B" in models["llama3.1:latest"]

    async def test_returns_empty_on_error(self):
        mock_session = _mock_aiohttp_session(
            raise_error=aiohttp.ClientConnectorError(
                MagicMock(), OSError("refused")
            )
        )

        with patch("custom_components.voice_automation_ai.llm_client.aiohttp.ClientSession", return_value=mock_session):
            client = OllamaClient()
            models = await client.async_fetch_models()

        assert models == {}


# ── Factory ──


class TestCreateLLMClientFactory:
    """Test the factory function."""

    @patch("custom_components.voice_automation_ai.llm_client.anthropic")
    def test_create_anthropic_client(self, mock_anthropic):
        client = create_llm_client("anthropic", api_key="sk-test")
        assert isinstance(client, AnthropicClient)

    def test_create_ollama_client(self):
        client = create_llm_client("ollama", host="http://localhost:11434")
        assert isinstance(client, OllamaClient)

    def test_create_ollama_default_host(self):
        client = create_llm_client("ollama")
        assert isinstance(client, OllamaClient)

    def test_create_ollama_with_gen_params(self):
        client = create_llm_client("ollama", temperature=0.7, top_p=0.9)
        assert isinstance(client, OllamaClient)
        assert client._temperature == 0.7
        assert client._top_p == 0.9

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_llm_client("openai", api_key="test")
