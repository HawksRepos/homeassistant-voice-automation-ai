"""Shared fixtures and Home Assistant mocks for Voice Automation AI tests."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


# ─── Fake Home Assistant module hierarchy ───
# Must run BEFORE any custom_components import.


def _install_fake_ha_modules() -> dict:
    """Install stub modules into sys.modules so HA imports resolve."""

    # --- homeassistant.core ---
    core = MagicMock()

    class FakeHomeAssistant:
        """Minimal HA core mock."""

        def __init__(self):
            self.config = MagicMock()
            self.config.config_dir = "/fake/config"
            self.config.language = "en"
            self.data = {}
            self.services = MagicMock()
            self.services.async_call = AsyncMock()
            self.services.async_register = MagicMock()
            self.services.async_remove = MagicMock()
            self.states = MagicMock()
            self.config_entries = MagicMock()
            self.config_entries.async_reload = AsyncMock()

        async def async_add_executor_job(self, func, *args):
            """Run a sync function -- in tests just call it directly."""
            return func(*args)

    core.HomeAssistant = FakeHomeAssistant
    core.ServiceCall = MagicMock

    # --- homeassistant.config_entries ---
    config_entries_mod = MagicMock()

    class FakeConfigEntry:
        def __init__(
            self,
            entry_id="test_entry_id",
            data=None,
            options=None,
            version=3,
        ):
            self.entry_id = entry_id
            self.data = data or {}
            self.options = options or {}
            self.version = version

        def add_update_listener(self, listener):
            return lambda: None

        def async_on_unload(self, unsub):
            pass

    config_entries_mod.ConfigEntry = FakeConfigEntry

    # ConfigFlow and OptionsFlow need to be real-ish base classes
    class FakeConfigFlow:
        """Base config flow mock."""
        hass = None

        def __init_subclass__(cls, **kwargs):
            # Accept and ignore domain= and other keyword args from HA metaclass pattern
            super().__init_subclass__()

        def async_show_form(self, **kwargs):
            return kwargs

        def async_create_entry(self, **kwargs):
            return {"type": "create_entry", **kwargs}

        def async_abort(self, **kwargs):
            return {"type": "abort", **kwargs}

    class FakeOptionsFlow:
        """Base options flow mock."""

        def async_show_form(self, **kwargs):
            return kwargs

        def async_create_entry(self, **kwargs):
            return {"type": "create_entry", **kwargs}

    config_entries_mod.ConfigFlow = FakeConfigFlow
    config_entries_mod.OptionsFlow = FakeOptionsFlow

    # --- homeassistant.components.conversation ---
    conversation_mod = MagicMock()

    class FakeConversationEntity:
        _attr_has_entity_name = True
        _attr_name = None
        _attr_unique_id = None
        hass = None

    class FakeConversationInput:
        def __init__(self, text="", conversation_id=None, language="en"):
            self.text = text
            self.conversation_id = conversation_id
            self.language = language

    class FakeConversationResult:
        def __init__(self, conversation_id=None, response=None):
            self.conversation_id = conversation_id
            self.response = response

    conversation_mod.ConversationEntity = FakeConversationEntity
    conversation_mod.ConversationInput = FakeConversationInput
    conversation_mod.ConversationResult = FakeConversationResult

    # --- homeassistant.helpers ---
    helpers = MagicMock()
    intent_mod = MagicMock()

    class FakeIntentResponse:
        def __init__(self, language="en"):
            self.language = language
            self.speech = None

        def async_set_speech(self, text):
            self.speech = text

    intent_mod.IntentResponse = FakeIntentResponse

    entity_platform = MagicMock()
    entity_platform.AddEntitiesCallback = MagicMock

    config_validation = MagicMock()
    config_validation.string = str
    config_validation.boolean = bool

    # --- homeassistant.exceptions ---
    exceptions = MagicMock()

    class FakeHAError(Exception):
        pass

    exceptions.HomeAssistantError = FakeHAError

    # --- homeassistant.data_entry_flow ---
    data_entry_flow = MagicMock()
    data_entry_flow.FlowResult = dict

    # Build top-level homeassistant module with sub-module attributes
    ha_root = MagicMock()
    ha_root.core = core
    ha_root.config_entries = config_entries_mod
    ha_root.exceptions = exceptions
    ha_root.data_entry_flow = data_entry_flow

    components_mod = MagicMock()
    components_mod.conversation = conversation_mod

    helpers.intent = intent_mod
    helpers.entity_platform = entity_platform
    helpers.config_validation = config_validation

    # Wire everything into sys.modules
    modules = {
        "homeassistant": ha_root,
        "homeassistant.core": core,
        "homeassistant.config_entries": config_entries_mod,
        "homeassistant.components": components_mod,
        "homeassistant.components.conversation": conversation_mod,
        "homeassistant.helpers": helpers,
        "homeassistant.helpers.intent": intent_mod,
        "homeassistant.helpers.entity_platform": entity_platform,
        "homeassistant.helpers.config_validation": config_validation,
        "homeassistant.exceptions": exceptions,
        "homeassistant.data_entry_flow": data_entry_flow,
    }
    sys.modules.update(modules)

    return {
        "HomeAssistant": FakeHomeAssistant,
        "ConfigEntry": FakeConfigEntry,
        "ConversationInput": FakeConversationInput,
        "ConversationResult": FakeConversationResult,
        "IntentResponse": FakeIntentResponse,
        "HomeAssistantError": FakeHAError,
        "FakeConfigFlow": FakeConfigFlow,
        "FakeOptionsFlow": FakeOptionsFlow,
    }


# Install fakes BEFORE any imports from custom_components
_fakes = _install_fake_ha_modules()


# ─── Fixtures ───


@pytest.fixture
def hass():
    """Create a mock HomeAssistant instance."""
    return _fakes["HomeAssistant"]()


@pytest.fixture
def config_entry():
    """Create a mock ConfigEntry with Anthropic defaults."""
    return _fakes["ConfigEntry"](
        entry_id="test_entry_id",
        data={
            "provider": "anthropic",
            "api_key": "sk-ant-test-key",
            "model": "claude-sonnet-4-5-20250929",
            "language": "en",
        },
        options={
            "model": "claude-sonnet-4-5-20250929",
            "language": "en",
            "max_tokens": 4096,
            "max_history_turns": 10,
        },
        version=3,
    )


@pytest.fixture
def ollama_config_entry():
    """Create a mock ConfigEntry with Ollama defaults."""
    return _fakes["ConfigEntry"](
        entry_id="test_ollama_entry",
        data={
            "provider": "ollama",
            "ollama_host": "http://localhost:11434",
            "model": "llama3.1",
            "language": "en",
        },
        options={
            "model": "llama3.1",
            "language": "en",
            "max_tokens": 4096,
            "max_history_turns": 10,
        },
        version=3,
    )


@pytest.fixture
def file_manager(hass, tmp_path):
    """Create a HAConfigFileManager backed by a temp directory."""
    hass.config.config_dir = str(tmp_path)
    from custom_components.voice_automation_ai.file_manager import HAConfigFileManager

    return HAConfigFileManager(hass)


class FakeState:
    """Fake HA entity state for testing."""

    def __init__(self, entity_id, state="on", attributes=None):
        self.entity_id = entity_id
        self.state = state
        self.attributes = attributes or {}


@pytest.fixture
def mock_states():
    """Create a set of fake HA entity states for testing."""
    return [
        FakeState(
            "light.living_room",
            "on",
            {"brightness": 255, "friendly_name": "Living Room"},
        ),
        FakeState(
            "light.bedroom",
            "off",
            {"brightness": 0, "friendly_name": "Bedroom"},
        ),
        FakeState(
            "switch.garage_door",
            "off",
            {"friendly_name": "Garage Door"},
        ),
        FakeState(
            "climate.thermostat",
            "heat",
            {"temperature": 22, "current_temperature": 20.5},
        ),
        FakeState(
            "lock.front_door",
            "locked",
            {"friendly_name": "Front Door"},
        ),
        FakeState(
            "sensor.outdoor_temp",
            "18.5",
            {"unit_of_measurement": "C"},
        ),
        FakeState("binary_sensor.motion", "off", {}),
        FakeState(
            "person.john",
            "home",
            {
                "latitude": 51.5074,
                "longitude": -0.1278,
                "gps_accuracy": 10,
                "friendly_name": "John",
            },
        ),
    ]
