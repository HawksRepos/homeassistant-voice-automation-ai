"""Config flow for Voice Automation AI integration."""
from __future__ import annotations

import logging
from typing import Any

import anthropic
import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResult
from homeassistant.exceptions import HomeAssistantError

from .const import (
    CONF_API_KEY,
    CONF_LANGUAGE,
    CONF_MODEL,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DOMAIN,
    LANGUAGES,
    MODELS,
)

_LOGGER = logging.getLogger(__name__)


async def validate_api_key(hass: HomeAssistant, api_key: str) -> dict[str, str]:
    """Validate the Anthropic API key."""
    try:
        client = anthropic.Anthropic(api_key=api_key)

        # Test API with a simple request
        await hass.async_add_executor_job(
            lambda: client.messages.create(
                model=DEFAULT_MODEL,
                max_tokens=10,
                messages=[{"role": "user", "content": "test"}],
            )
        )

        return {"title": "Voice Automation AI"}

    except anthropic.AuthenticationError:
        raise InvalidAuth
    except Exception as err:
        _LOGGER.error("Error validating API key: %s", err)
        raise CannotConnect


class VoiceAutomationAIConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Voice Automation AI."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            try:
                # Validate API key
                info = await validate_api_key(self.hass, user_input[CONF_API_KEY])

                # Auto-detect language from Home Assistant configuration
                ha_language = self.hass.config.language
                detected_language = ha_language if ha_language in LANGUAGES else DEFAULT_LANGUAGE

                # Add language to user input
                user_input[CONF_LANGUAGE] = detected_language

                # Create entry
                return self.async_create_entry(
                    title=info["title"],
                    data=user_input,
                )

            except CannotConnect:
                errors["base"] = "cannot_connect"
            except InvalidAuth:
                errors["base"] = "invalid_auth"
            except Exception:
                _LOGGER.exception("Unexpected exception")
                errors["base"] = "unknown"

        # Show form - Language is auto-detected from Home Assistant
        data_schema = vol.Schema({
            vol.Required(CONF_API_KEY): str,
            vol.Optional(CONF_MODEL, default=DEFAULT_MODEL): vol.In(MODELS),
        })

        return self.async_show_form(
            step_id="user",
            data_schema=data_schema,
            errors=errors,
        )

    async def async_step_import(self, import_data: dict[str, Any]) -> FlowResult:
        """Import a config entry from configuration.yaml."""
        return await self.async_step_user(import_data)


class CannotConnect(HomeAssistantError):
    """Error to indicate we cannot connect."""


class InvalidAuth(HomeAssistantError):
    """Error to indicate there is invalid auth."""
