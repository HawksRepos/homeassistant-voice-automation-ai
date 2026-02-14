"""Tests for config_flow.py."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.voice_automation_ai.const import (
    CONF_API_KEY,
    CONF_LANGUAGE,
    CONF_MAX_HISTORY_TURNS,
    CONF_MAX_TOKENS,
    CONF_MODEL,
    CONF_OLLAMA_HOST,
    CONF_PROVIDER,
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_HISTORY_TURNS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_MODEL,
    DOMAIN,
    LANGUAGES,
    PROVIDER_ANTHROPIC,
    PROVIDER_OLLAMA,
)
from custom_components.voice_automation_ai.config_flow import (
    CannotConnect,
    VoiceAutomationAIConfigFlow,
    VoiceAutomationAIOptionsFlow,
    validate_connection,
)


# ── validate_connection ──


class TestValidateConnection:
    """Test the validate_connection helper."""

    async def test_successful_validation(self, hass):
        with patch(
            "custom_components.voice_automation_ai.config_flow.create_llm_client"
        ) as mock_factory:
            mock_client = MagicMock()
            mock_client.validate_connection = MagicMock()
            mock_factory.return_value = mock_client

            result = await validate_connection(
                hass, PROVIDER_ANTHROPIC, api_key="sk-test", model=DEFAULT_MODEL
            )
            assert result["title"] == "Voice Automation AI"

    async def test_failed_validation_raises(self, hass):
        with patch(
            "custom_components.voice_automation_ai.config_flow.create_llm_client"
        ) as mock_factory:
            mock_client = MagicMock()
            mock_client.validate_connection.side_effect = ConnectionError("refused")
            mock_factory.return_value = mock_client

            with pytest.raises(CannotConnect):
                await validate_connection(
                    hass, PROVIDER_ANTHROPIC, api_key="sk-bad", model=DEFAULT_MODEL
                )


# ── Config flow: user step ──


class TestConfigFlowUserStep:
    """Test the initial provider selection step."""

    async def test_shows_form_when_no_input(self, hass):
        flow = VoiceAutomationAIConfigFlow()
        flow.hass = hass

        result = await flow.async_step_user(user_input=None)
        assert result["step_id"] == "user"

    async def test_anthropic_routes_correctly(self, hass):
        flow = VoiceAutomationAIConfigFlow()
        flow.hass = hass

        with patch.object(flow, "async_step_anthropic", new_callable=AsyncMock) as mock_step:
            mock_step.return_value = {"step_id": "anthropic"}
            result = await flow.async_step_user({CONF_PROVIDER: PROVIDER_ANTHROPIC})
            mock_step.assert_awaited_once()

    async def test_ollama_routes_correctly(self, hass):
        flow = VoiceAutomationAIConfigFlow()
        flow.hass = hass

        with patch.object(flow, "async_step_ollama", new_callable=AsyncMock) as mock_step:
            mock_step.return_value = {"step_id": "ollama"}
            result = await flow.async_step_user({CONF_PROVIDER: PROVIDER_OLLAMA})
            mock_step.assert_awaited_once()


# ── Config flow: anthropic step ──


class TestConfigFlowAnthropicStep:
    """Test the Anthropic configuration step."""

    async def test_shows_form_when_no_input(self, hass):
        flow = VoiceAutomationAIConfigFlow()
        flow.hass = hass

        result = await flow.async_step_anthropic(user_input=None)
        assert result["step_id"] == "anthropic"

    async def test_creates_entry_on_success(self, hass):
        flow = VoiceAutomationAIConfigFlow()
        flow.hass = hass

        with patch(
            "custom_components.voice_automation_ai.config_flow.validate_connection",
            return_value={"title": "Voice Automation AI"},
        ):
            result = await flow.async_step_anthropic(
                {CONF_API_KEY: "sk-ant-valid-key", CONF_MODEL: DEFAULT_MODEL}
            )

            assert result["type"] == "create_entry"
            assert result["data"][CONF_PROVIDER] == PROVIDER_ANTHROPIC
            assert result["data"][CONF_API_KEY] == "sk-ant-valid-key"
            assert result["options"][CONF_MAX_TOKENS] == DEFAULT_MAX_TOKENS
            assert result["options"][CONF_MAX_HISTORY_TURNS] == DEFAULT_MAX_HISTORY_TURNS

    async def test_shows_error_on_cannot_connect(self, hass):
        flow = VoiceAutomationAIConfigFlow()
        flow.hass = hass

        with patch(
            "custom_components.voice_automation_ai.config_flow.validate_connection",
            side_effect=CannotConnect("Connection failed"),
        ):
            result = await flow.async_step_anthropic(
                {CONF_API_KEY: "sk-ant-bad-key"}
            )

            assert result["errors"]["base"] == "cannot_connect"

    async def test_language_auto_detection(self, hass):
        """When HA language is in LANGUAGES, it should be auto-detected."""
        flow = VoiceAutomationAIConfigFlow()
        flow.hass = hass
        flow.hass.config.language = "es"

        with patch(
            "custom_components.voice_automation_ai.config_flow.validate_connection",
            return_value={"title": "Voice Automation AI"},
        ):
            result = await flow.async_step_anthropic(
                {CONF_API_KEY: "sk-ant-valid", CONF_MODEL: DEFAULT_MODEL}
            )

            assert result["data"][CONF_LANGUAGE] == "es"

    async def test_unknown_language_defaults_to_english(self, hass):
        flow = VoiceAutomationAIConfigFlow()
        flow.hass = hass
        flow.hass.config.language = "zh"

        with patch(
            "custom_components.voice_automation_ai.config_flow.validate_connection",
            return_value={"title": "Voice Automation AI"},
        ):
            result = await flow.async_step_anthropic(
                {CONF_API_KEY: "sk-ant-valid", CONF_MODEL: DEFAULT_MODEL}
            )

            assert result["data"][CONF_LANGUAGE] == DEFAULT_LANGUAGE


# ── Config flow: ollama step ──


class TestConfigFlowOllamaStep:
    """Test the Ollama configuration step."""

    async def test_creates_entry(self, hass):
        flow = VoiceAutomationAIConfigFlow()
        flow.hass = hass

        with patch(
            "custom_components.voice_automation_ai.config_flow.validate_connection",
            return_value={"title": "Voice Automation AI"},
        ):
            result = await flow.async_step_ollama(
                {CONF_OLLAMA_HOST: "http://192.168.1.100:11434", CONF_MODEL: "llama3.1"}
            )

            assert result["data"][CONF_PROVIDER] == PROVIDER_OLLAMA
            assert result["data"][CONF_OLLAMA_HOST] == "http://192.168.1.100:11434"

    async def test_default_host(self, hass):
        flow = VoiceAutomationAIConfigFlow()
        flow.hass = hass

        with patch(
            "custom_components.voice_automation_ai.config_flow.validate_connection",
            return_value={"title": "Voice Automation AI"},
        ):
            result = await flow.async_step_ollama({CONF_MODEL: "llama3.1"})

            assert result["data"][CONF_OLLAMA_HOST] == DEFAULT_OLLAMA_HOST


# ── Options flow ──


class TestOptionsFlow:
    """Test the options flow."""

    async def test_shows_form_when_no_input(self):
        entry = MagicMock()
        entry.data = {CONF_PROVIDER: PROVIDER_ANTHROPIC}
        entry.options = {
            CONF_MODEL: DEFAULT_MODEL,
            CONF_LANGUAGE: DEFAULT_LANGUAGE,
            CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
            CONF_MAX_HISTORY_TURNS: DEFAULT_MAX_HISTORY_TURNS,
        }

        flow = VoiceAutomationAIOptionsFlow(entry)
        result = await flow.async_step_init(user_input=None)
        assert result["step_id"] == "init"

    async def test_saves_input(self):
        entry = MagicMock()
        entry.data = {CONF_PROVIDER: PROVIDER_ANTHROPIC}
        entry.options = {}

        flow = VoiceAutomationAIOptionsFlow(entry)

        user_input = {
            CONF_MODEL: "claude-opus-4-1-20250805",
            CONF_LANGUAGE: "fr",
            CONF_MAX_TOKENS: 8192,
            CONF_MAX_HISTORY_TURNS: 20,
        }

        result = await flow.async_step_init(user_input=user_input)
        assert result["type"] == "create_entry"
        assert result["data"] == user_input

    async def test_ollama_provider_accepted(self):
        """Options flow for Ollama provider should work."""
        entry = MagicMock()
        entry.data = {CONF_PROVIDER: PROVIDER_OLLAMA}
        entry.options = {
            CONF_MODEL: DEFAULT_OLLAMA_MODEL,
            CONF_LANGUAGE: DEFAULT_LANGUAGE,
            CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
            CONF_MAX_HISTORY_TURNS: DEFAULT_MAX_HISTORY_TURNS,
        }

        flow = VoiceAutomationAIOptionsFlow(entry)
        result = await flow.async_step_init(user_input=None)
        assert result["step_id"] == "init"
