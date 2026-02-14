"""Tests for LLM client abstraction layer."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from custom_components.voice_automation_ai.llm_client import (
    TOOL_DEFINITIONS,
    AnthropicClient,
    BaseLLMClient,
    LLMResponse,
    OllamaClient,
    create_llm_client,
)


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

    def test_all_14_tools_defined(self):
        assert len(TOOL_DEFINITIONS) == 14

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


# ── Anthropic tool conversion ──


class TestAnthropicToolConversion:
    """Test Anthropic tool format conversion."""

    def test_to_anthropic_tools_format(self):
        client = AnthropicClient.__new__(AnthropicClient)
        tools = client._to_anthropic_tools()

        assert len(tools) == 14
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

        assert len(tools) == 14
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


# ── Ollama create_message ──


class TestOllamaCreateMessage:
    """Test Ollama create_message with mocked HTTP."""

    @patch("custom_components.voice_automation_ai.llm_client.requests")
    def test_text_response(self, mock_requests):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"role": "assistant", "content": "Hello from Ollama!"}
        }
        mock_resp.raise_for_status = MagicMock()
        mock_requests.post.return_value = mock_resp

        client = OllamaClient(host="http://localhost:11434")
        result = client.create_message(
            model="llama3.1",
            system_prompt="test",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
            tools=False,
        )

        assert result.text == "Hello from Ollama!"
        assert result.has_tool_calls is False

    @patch("custom_components.voice_automation_ai.llm_client.requests")
    def test_tool_call_response(self, mock_requests):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
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
            }
        }
        mock_resp.raise_for_status = MagicMock()
        mock_requests.post.return_value = mock_resp

        client = OllamaClient()
        result = client.create_message(
            model="llama3.1",
            system_prompt="test",
            messages=[{"role": "user", "content": "turn on living room light"}],
            max_tokens=100,
        )

        assert result.has_tool_calls is True
        assert result.tool_calls[0]["name"] == "call_service"
        assert result.tool_calls[0]["arguments"]["domain"] == "light"

    @patch("custom_components.voice_automation_ai.llm_client.requests")
    def test_empty_content_returns_none_text(self, mock_requests):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"role": "assistant", "content": ""}
        }
        mock_resp.raise_for_status = MagicMock()
        mock_requests.post.return_value = mock_resp

        client = OllamaClient()
        result = client.create_message(
            model="llama3.1",
            system_prompt="test",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
            tools=False,
        )
        assert result.text is None


# ── Ollama validate_connection ──


class TestOllamaValidateConnection:
    """Test Ollama connection validation."""

    @patch("custom_components.voice_automation_ai.llm_client.requests")
    def test_model_available(self, mock_requests):
        import requests as real_requests

        mock_requests.ConnectionError = real_requests.ConnectionError
        mock_requests.RequestException = real_requests.RequestException

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "models": [{"name": "llama3.1:latest"}, {"name": "mistral:latest"}]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_requests.get.return_value = mock_resp

        client = OllamaClient()
        client.validate_connection("llama3.1")  # should not raise

    @patch("custom_components.voice_automation_ai.llm_client.requests")
    def test_model_missing(self, mock_requests):
        import requests as real_requests

        # Must set real exception classes so except clauses work
        mock_requests.ConnectionError = real_requests.ConnectionError
        mock_requests.RequestException = real_requests.RequestException

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "models": [{"name": "mistral:latest"}]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_requests.get.return_value = mock_resp

        client = OllamaClient()
        with pytest.raises(ConnectionError, match="not found"):
            client.validate_connection("llama3.1")

    @patch("custom_components.voice_automation_ai.llm_client.requests")
    def test_connection_refused(self, mock_requests):
        import requests as real_requests

        mock_requests.ConnectionError = real_requests.ConnectionError
        mock_requests.RequestException = real_requests.RequestException
        mock_requests.get.side_effect = real_requests.ConnectionError("refused")

        client = OllamaClient()
        with pytest.raises(ConnectionError, match="Cannot connect"):
            client.validate_connection("llama3.1")


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

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_llm_client("openai", api_key="test")
