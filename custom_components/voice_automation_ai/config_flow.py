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
    CONF_MAX_HISTORY_TURNS,
    CONF_MAX_TOKENS,
    CONF_MODEL,
    CONF_OLLAMA_HOST,
    CONF_PROVIDER,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_HISTORY_TURNS,
    DEFAULT_MAX_TOKENS,
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
from .llm_client import OllamaClient, create_llm_client

_LOGGER = logging.getLogger(__name__)


async def validate_connection(
    hass: HomeAssistant, provider: str, **kwargs: Any
) -> dict[str, str]:
    """Validate the LLM connection."""
    try:
        client = create_llm_client(provider, **kwargs)
        model = kwargs.get("model", DEFAULT_MODEL)
        if client.is_async:
            await client.async_validate_connection(model)
        else:
            await hass.async_add_executor_job(client.validate_connection, model)
        return {"title": "Voice Automation AI"}
    except Exception as err:
        _LOGGER.error("Connection validation failed: %s", err)
        raise CannotConnect(str(err))


class VoiceAutomationAIConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Voice Automation AI."""

    VERSION = 3

    def __init__(self) -> None:
        """Initialize."""
        self._provider: str | None = None
        self._ollama_host: str | None = None
        self._discovered_models: dict[str, str] = {}

    @staticmethod
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> VoiceAutomationAIOptionsFlow:
        """Get the options flow handler."""
        return VoiceAutomationAIOptionsFlow(config_entry)

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
                model = user_input.get(CONF_MODEL, DEFAULT_MODEL)
                info = await validate_connection(
                    self.hass,
                    PROVIDER_ANTHROPIC,
                    api_key=user_input[CONF_API_KEY],
                    model=model,
                )

                ha_language = self.hass.config.language
                detected_language = ha_language if ha_language in LANGUAGES else DEFAULT_LANGUAGE

                return self.async_create_entry(
                    title=info["title"],
                    data={
                        CONF_PROVIDER: PROVIDER_ANTHROPIC,
                        CONF_API_KEY: user_input[CONF_API_KEY],
                        CONF_MODEL: model,
                        CONF_LANGUAGE: detected_language,
                    },
                    options={
                        CONF_MODEL: model,
                        CONF_LANGUAGE: detected_language,
                        CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
                        CONF_MAX_HISTORY_TURNS: DEFAULT_MAX_HISTORY_TURNS,
                    },
                )

            except CannotConnect:
                errors["base"] = "cannot_connect"
            except Exception as err:
                _LOGGER.error("Unexpected error during Anthropic setup: %s", type(err).__name__)
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
        """Step 2b: Ollama host configuration."""
        errors: dict[str, str] = {}

        if user_input is not None:
            host = user_input.get(CONF_OLLAMA_HOST, DEFAULT_OLLAMA_HOST)
            self._ollama_host = host

            # Warn if using unencrypted HTTP over a network
            if host.startswith("http://") and not any(
                host.startswith(f"http://{local}")
                for local in ("localhost", "127.0.0.1", "[::1]")
            ):
                _LOGGER.warning(
                    "Ollama host '%s' uses unencrypted HTTP. "
                    "Prompts and responses will be transmitted in plaintext.",
                    host,
                )

            # Try to discover models from this host
            try:
                client = OllamaClient(host=host)
                self._discovered_models = await client.async_fetch_models()
            except Exception as err:
                _LOGGER.warning("Ollama model discovery failed for %s: %s", host, err)
                self._discovered_models = {}

            if self._discovered_models:
                return await self.async_step_ollama_model()
            else:
                errors["base"] = "cannot_connect"

        return self.async_show_form(
            step_id="ollama",
            data_schema=vol.Schema({
                vol.Optional(CONF_OLLAMA_HOST, default=DEFAULT_OLLAMA_HOST): str,
            }),
            errors=errors,
        )

    async def async_step_ollama_model(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Step 2c: Select Ollama model from discovered models."""
        errors: dict[str, str] = {}

        if user_input is not None:
            try:
                model = user_input.get(CONF_MODEL, DEFAULT_OLLAMA_MODEL)
                info = await validate_connection(
                    self.hass,
                    PROVIDER_OLLAMA,
                    host=self._ollama_host,
                    model=model,
                )

                ha_language = self.hass.config.language
                detected_language = ha_language if ha_language in LANGUAGES else DEFAULT_LANGUAGE

                return self.async_create_entry(
                    title=info["title"],
                    data={
                        CONF_PROVIDER: PROVIDER_OLLAMA,
                        CONF_OLLAMA_HOST: self._ollama_host,
                        CONF_MODEL: model,
                        CONF_LANGUAGE: detected_language,
                    },
                    options={
                        CONF_MODEL: model,
                        CONF_LANGUAGE: detected_language,
                        CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
                        CONF_MAX_HISTORY_TURNS: DEFAULT_MAX_HISTORY_TURNS,
                    },
                )

            except CannotConnect:
                errors["base"] = "cannot_connect"
            except Exception as err:
                _LOGGER.error("Unexpected error during Ollama model setup: %s", type(err).__name__)
                errors["base"] = "unknown"

        # Build model dropdown from discovered models
        model_options = dict(self._discovered_models)
        default = DEFAULT_OLLAMA_MODEL
        if default not in model_options and model_options:
            default = next(iter(model_options))

        return self.async_show_form(
            step_id="ollama_model",
            data_schema=vol.Schema({
                vol.Required(CONF_MODEL, default=default): vol.In(model_options),
            }),
            errors=errors,
        )

    async def async_step_import(self, import_data: dict[str, Any]) -> FlowResult:
        """Import a config entry from configuration.yaml."""
        return await self.async_step_user(import_data)


class VoiceAutomationAIOptionsFlow(config_entries.OptionsFlow):
    """Handle options flow for Voice Automation AI."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize."""
        self._config_entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage runtime options."""
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        # Determine which model list to show based on provider
        provider = self._config_entry.data.get(CONF_PROVIDER, DEFAULT_PROVIDER)
        if provider == PROVIDER_ANTHROPIC:
            model_options = dict(ANTHROPIC_MODELS)
            default_model = DEFAULT_MODEL
        else:
            # Try dynamic model discovery for Ollama
            host = self._config_entry.data.get(CONF_OLLAMA_HOST, DEFAULT_OLLAMA_HOST)
            try:
                client = OllamaClient(host=host)
                model_options = await client.async_fetch_models()
            except Exception as err:
                _LOGGER.warning("Ollama model discovery failed for options flow: %s", err)
                model_options = {}
            if not model_options:
                model_options = dict(OLLAMA_MODELS)
            default_model = DEFAULT_OLLAMA_MODEL

        # Current values (options -> data -> defaults)
        current_model = self._config_entry.options.get(
            CONF_MODEL, self._config_entry.data.get(CONF_MODEL, default_model)
        )
        current_language = self._config_entry.options.get(
            CONF_LANGUAGE, self._config_entry.data.get(CONF_LANGUAGE, DEFAULT_LANGUAGE)
        )
        current_max_tokens = self._config_entry.options.get(
            CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS
        )
        current_max_history = self._config_entry.options.get(
            CONF_MAX_HISTORY_TURNS, DEFAULT_MAX_HISTORY_TURNS
        )

        # If current model is custom (not in list), add it so dropdown works
        if current_model not in model_options:
            model_options[current_model] = current_model

        data_schema_fields: dict = {
            vol.Required(CONF_MODEL, default=current_model): vol.In(model_options),
            vol.Required(CONF_LANGUAGE, default=current_language): vol.In(LANGUAGES),
            vol.Required(CONF_MAX_TOKENS, default=current_max_tokens): vol.All(
                vol.Coerce(int), vol.Range(min=256, max=32768)
            ),
            vol.Required(CONF_MAX_HISTORY_TURNS, default=current_max_history): vol.All(
                vol.Coerce(int), vol.Range(min=1, max=50)
            ),
        }

        # Add generation parameters for Ollama only
        if provider == PROVIDER_OLLAMA:
            current_temperature = self._config_entry.options.get(CONF_TEMPERATURE)
            current_top_p = self._config_entry.options.get(CONF_TOP_P)
            data_schema_fields[vol.Optional(CONF_TEMPERATURE, default=current_temperature)] = vol.Any(
                None, vol.All(vol.Coerce(float), vol.Range(min=0.0, max=2.0))
            )
            data_schema_fields[vol.Optional(CONF_TOP_P, default=current_top_p)] = vol.Any(
                None, vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1.0))
            )

        data_schema = vol.Schema(data_schema_fields)

        return self.async_show_form(step_id="init", data_schema=data_schema)


class CannotConnect(HomeAssistantError):
    """Error to indicate we cannot connect."""
