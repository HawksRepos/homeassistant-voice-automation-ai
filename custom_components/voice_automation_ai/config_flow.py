"""Config flow for Voice Automation AI integration."""
from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResult
from homeassistant.exceptions import HomeAssistantError

from .const import (
    ANTHROPIC_MODELS,
    CONF_API_KEY,
    CONF_LANGUAGE,
    CONF_MODEL,
    CONF_OLLAMA_HOST,
    CONF_PROVIDER,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_PROVIDER,
    DOMAIN,
    LANGUAGES,
    OLLAMA_MODELS,
    PROVIDERS,
    PROVIDER_ANTHROPIC,
    PROVIDER_OLLAMA,
)
from .llm_client import create_llm_client

_LOGGER = logging.getLogger(__name__)


async def validate_connection(
    hass: HomeAssistant, provider: str, **kwargs: Any
) -> dict[str, str]:
    """Validate the LLM connection."""
    try:
        client = create_llm_client(provider, **kwargs)
        model = kwargs.get("model", DEFAULT_MODEL)
        await hass.async_add_executor_job(client.validate_connection, model)
        return {"title": "Voice Automation AI"}
    except Exception as err:
        _LOGGER.error("Connection validation failed: %s", err)
        raise CannotConnect(str(err))


class VoiceAutomationAIConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Voice Automation AI."""

    VERSION = 2

    def __init__(self) -> None:
        """Initialize."""
        self._provider: str | None = None

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Step 1: Choose provider."""
        if user_input is not None:
            self._provider = user_input[CONF_PROVIDER]
            if self._provider == PROVIDER_ANTHROPIC:
                return await self.async_step_anthropic()
            return await self.async_step_ollama()

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema({
                vol.Required(CONF_PROVIDER, default=DEFAULT_PROVIDER): vol.In(PROVIDERS),
            }),
        )

    async def async_step_anthropic(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Step 2a: Anthropic configuration."""
        errors: dict[str, str] = {}

        if user_input is not None:
            try:
                info = await validate_connection(
                    self.hass,
                    PROVIDER_ANTHROPIC,
                    api_key=user_input[CONF_API_KEY],
                    model=user_input.get(CONF_MODEL, DEFAULT_MODEL),
                )

                ha_language = self.hass.config.language
                detected_language = ha_language if ha_language in LANGUAGES else DEFAULT_LANGUAGE

                return self.async_create_entry(
                    title=info["title"],
                    data={
                        CONF_PROVIDER: PROVIDER_ANTHROPIC,
                        CONF_API_KEY: user_input[CONF_API_KEY],
                        CONF_MODEL: user_input.get(CONF_MODEL, DEFAULT_MODEL),
                        CONF_LANGUAGE: detected_language,
                    },
                )

            except CannotConnect:
                errors["base"] = "cannot_connect"
            except Exception:
                _LOGGER.exception("Unexpected exception")
                errors["base"] = "unknown"

        return self.async_show_form(
            step_id="anthropic",
            data_schema=vol.Schema({
                vol.Required(CONF_API_KEY): str,
                vol.Optional(CONF_MODEL, default=DEFAULT_MODEL): vol.In(ANTHROPIC_MODELS),
            }),
            errors=errors,
        )

    async def async_step_ollama(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Step 2b: Ollama configuration."""
        errors: dict[str, str] = {}

        if user_input is not None:
            try:
                info = await validate_connection(
                    self.hass,
                    PROVIDER_OLLAMA,
                    host=user_input.get(CONF_OLLAMA_HOST, DEFAULT_OLLAMA_HOST),
                    model=user_input.get(CONF_MODEL, DEFAULT_OLLAMA_MODEL),
                )

                ha_language = self.hass.config.language
                detected_language = ha_language if ha_language in LANGUAGES else DEFAULT_LANGUAGE

                return self.async_create_entry(
                    title=info["title"],
                    data={
                        CONF_PROVIDER: PROVIDER_OLLAMA,
                        CONF_OLLAMA_HOST: user_input.get(CONF_OLLAMA_HOST, DEFAULT_OLLAMA_HOST),
                        CONF_MODEL: user_input.get(CONF_MODEL, DEFAULT_OLLAMA_MODEL),
                        CONF_LANGUAGE: detected_language,
                    },
                )

            except CannotConnect:
                errors["base"] = "cannot_connect"
            except Exception:
                _LOGGER.exception("Unexpected exception")
                errors["base"] = "unknown"

        return self.async_show_form(
            step_id="ollama",
            data_schema=vol.Schema({
                vol.Optional(CONF_OLLAMA_HOST, default=DEFAULT_OLLAMA_HOST): str,
                vol.Optional(CONF_MODEL, default=DEFAULT_OLLAMA_MODEL): vol.In(OLLAMA_MODELS),
            }),
            errors=errors,
        )

    async def async_step_import(self, import_data: dict[str, Any]) -> FlowResult:
        """Import a config entry from configuration.yaml."""
        return await self.async_step_user(import_data)


class CannotConnect(HomeAssistantError):
    """Error to indicate we cannot connect."""
