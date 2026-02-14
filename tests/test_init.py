"""Tests for __init__.py: migration, service handlers, YAML security."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from custom_components.voice_automation_ai.const import (
    BLOCKED_SERVICE_DOMAINS,
    CONF_LANGUAGE,
    CONF_MAX_HISTORY_TURNS,
    CONF_MAX_TOKENS,
    CONF_MODEL,
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_HISTORY_TURNS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DOMAIN,
    PROVIDER_ANTHROPIC,
)


# ── Config migration ──


class TestConfigMigration:
    """Test async_migrate_entry version migrations."""

    async def test_v1_to_v2_adds_provider(self, hass):
        from custom_components.voice_automation_ai import async_migrate_entry

        entry = MagicMock()
        entry.version = 1
        entry.data = {"api_key": "sk-test", "model": "claude-3-opus-20240229"}
        entry.options = {}
        hass.config_entries.async_update_entry = MagicMock()

        # Make version update so the v2->v3 step can also run
        def fake_update(config_entry, **kwargs):
            if "version" in kwargs:
                config_entry.version = kwargs["version"]
            if "data" in kwargs:
                config_entry.data = kwargs["data"]
            if "options" in kwargs:
                config_entry.options = kwargs["options"]

        hass.config_entries.async_update_entry = MagicMock(side_effect=fake_update)

        result = await async_migrate_entry(hass, entry)
        assert result is True
        assert entry.version == 3

    async def test_v2_to_v3_seeds_options(self, hass):
        from custom_components.voice_automation_ai import async_migrate_entry

        entry = MagicMock()
        entry.version = 2
        entry.data = {
            "provider": "anthropic",
            "api_key": "sk-test",
            "model": "claude-3-opus-20240229",
            "language": "es",
        }
        entry.options = {}

        def fake_update(config_entry, **kwargs):
            if "version" in kwargs:
                config_entry.version = kwargs["version"]
            if "options" in kwargs:
                config_entry.options = kwargs["options"]

        hass.config_entries.async_update_entry = MagicMock(side_effect=fake_update)

        result = await async_migrate_entry(hass, entry)
        assert result is True

        assert entry.options[CONF_MODEL] == "claude-3-opus-20240229"
        assert entry.options[CONF_LANGUAGE] == "es"
        assert entry.options[CONF_MAX_TOKENS] == DEFAULT_MAX_TOKENS
        assert entry.options[CONF_MAX_HISTORY_TURNS] == DEFAULT_MAX_HISTORY_TURNS

    async def test_v3_is_noop(self, hass):
        from custom_components.voice_automation_ai import async_migrate_entry

        entry = MagicMock()
        entry.version = 3
        hass.config_entries.async_update_entry = MagicMock()

        result = await async_migrate_entry(hass, entry)
        assert result is True
        hass.config_entries.async_update_entry.assert_not_called()

    async def test_future_version_returns_false(self, hass):
        from custom_components.voice_automation_ai import async_migrate_entry

        entry = MagicMock()
        entry.version = 4

        result = await async_migrate_entry(hass, entry)
        assert result is False

    async def test_v1_to_v3_full_chain(self, hass):
        """v1 entry should migrate through v2 to v3."""
        from custom_components.voice_automation_ai import async_migrate_entry

        entry = MagicMock()
        entry.version = 1
        entry.data = {"api_key": "sk-test"}
        entry.options = {}

        def fake_update(config_entry, **kwargs):
            if "version" in kwargs:
                config_entry.version = kwargs["version"]
            if "data" in kwargs:
                config_entry.data = kwargs["data"]
            if "options" in kwargs:
                config_entry.options = kwargs["options"]

        hass.config_entries.async_update_entry = MagicMock(side_effect=fake_update)

        result = await async_migrate_entry(hass, entry)
        assert result is True
        assert entry.version == 3
        assert "provider" in entry.data
        assert CONF_MAX_TOKENS in entry.options


# ── YAML blocked service check ──


class TestCheckYAMLForBlockedServices:
    """Test the module-level _check_yaml_for_blocked_services."""

    def test_blocks_shell_command(self):
        from custom_components.voice_automation_ai import _check_yaml_for_blocked_services

        data = {"action": [{"service": "shell_command.run"}]}
        assert _check_yaml_for_blocked_services(data) is not None

    def test_blocks_all_blocked_domains(self):
        from custom_components.voice_automation_ai import _check_yaml_for_blocked_services

        for domain in BLOCKED_SERVICE_DOMAINS:
            data = {"action": [{"service": f"{domain}.test"}]}
            result = _check_yaml_for_blocked_services(data)
            assert result is not None, f"Domain '{domain}' was not blocked"

    def test_allows_safe_yaml(self):
        from custom_components.voice_automation_ai import _check_yaml_for_blocked_services

        data = {
            "alias": "Safe Automation",
            "action": [{"service": "light.turn_on", "entity_id": "light.kitchen"}],
        }
        assert _check_yaml_for_blocked_services(data) is None


# ── _generate_yaml ──


class TestGenerateYAML:
    """Test _generate_yaml function."""

    def test_strips_markdown_fences(self):
        from custom_components.voice_automation_ai import _generate_yaml

        config = {
            "provider": "anthropic",
            "api_key": "sk-test",
            "model": "claude-sonnet-4-5-20250929",
            "language": "en",
            "max_tokens": 4096,
        }

        with patch(
            "custom_components.voice_automation_ai.create_llm_client"
        ) as mock_factory:
            mock_client = MagicMock()
            mock_client.create_simple_message.return_value = (
                "```yaml\nalias: Test\ntrigger: []\n```"
            )
            mock_factory.return_value = mock_client

            result = _generate_yaml(
                config, "test description", "automation", "light: light.test"
            )
            assert "```" not in result
            assert "alias: Test" in result

    def test_raises_on_llm_error(self):
        from custom_components.voice_automation_ai import _generate_yaml

        config = {
            "provider": "anthropic",
            "api_key": "sk-test",
            "model": "claude-sonnet-4-5-20250929",
            "language": "en",
            "max_tokens": 4096,
        }

        with patch(
            "custom_components.voice_automation_ai.create_llm_client"
        ) as mock_factory:
            mock_client = MagicMock()
            mock_client.create_simple_message.side_effect = Exception("API down")
            mock_factory.return_value = mock_client

            with pytest.raises(Exception, match="Failed to generate YAML"):
                _generate_yaml(config, "test", "automation", "")


# ── _build_llm_client_kwargs ──


class TestBuildLLMClientKwargs:
    """Test _build_llm_client_kwargs helper."""

    def test_anthropic_kwargs(self):
        from custom_components.voice_automation_ai import _build_llm_client_kwargs

        config = {"provider": "anthropic", "api_key": "sk-test"}
        kwargs = _build_llm_client_kwargs(config)
        assert kwargs["api_key"] == "sk-test"
        assert "timeout" in kwargs

    def test_ollama_kwargs(self):
        from custom_components.voice_automation_ai import _build_llm_client_kwargs

        config = {"provider": "ollama", "ollama_host": "http://myhost:11434"}
        kwargs = _build_llm_client_kwargs(config)
        assert kwargs["host"] == "http://myhost:11434"

    def test_ollama_default_host(self):
        from custom_components.voice_automation_ai import _build_llm_client_kwargs

        config = {"provider": "ollama"}
        kwargs = _build_llm_client_kwargs(config)
        assert kwargs["host"] == "http://localhost:11434"
