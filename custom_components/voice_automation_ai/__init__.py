"""Voice Automation AI integration for Home Assistant."""
from __future__ import annotations

import logging
import time
from typing import Any

import anthropic
import voluptuous as vol
import yaml

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv

from .const import (
    API_TIMEOUT,
    ATTR_AUTOMATION_ID,
    ATTR_DESCRIPTION,
    ATTR_PREVIEW,
    ATTR_SCENE_ID,
    ATTR_SCRIPT_NAME,
    ATTR_VALIDATE_ONLY,
    ATTR_YAML_CONTENT,
    CONF_API_KEY,
    CONF_LANGUAGE,
    CONF_MODEL,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DOMAIN,
    MAX_TOKENS,
    PLATFORMS,
    SERVICE_CREATE_AUTOMATION,
    SERVICE_CREATE_SCENE,
    SERVICE_CREATE_SCRIPT,
    SERVICE_DELETE_AUTOMATION,
    SERVICE_DELETE_SCENE,
    SERVICE_DELETE_SCRIPT,
    SERVICE_EDIT_AUTOMATION,
    SERVICE_EDIT_SCENE,
    SERVICE_EDIT_SCRIPT,
    SERVICE_LIST_AUTOMATIONS,
    SERVICE_LIST_SCENES,
    SERVICE_LIST_SCRIPTS,
    SERVICE_VALIDATE_AUTOMATION,
)
from .file_manager import HAConfigFileManager

_LOGGER = logging.getLogger(__name__)

# ── Service schemas ──

CREATE_AUTOMATION_SCHEMA = vol.Schema({
    vol.Required(ATTR_DESCRIPTION): cv.string,
    vol.Optional(ATTR_VALIDATE_ONLY, default=False): cv.boolean,
    vol.Optional(ATTR_PREVIEW, default=False): cv.boolean,
})

VALIDATE_AUTOMATION_SCHEMA = vol.Schema({
    vol.Required(ATTR_YAML_CONTENT): cv.string,
})

CREATE_SCRIPT_SCHEMA = vol.Schema({
    vol.Required(ATTR_DESCRIPTION): cv.string,
    vol.Optional(ATTR_SCRIPT_NAME): cv.string,
})

CREATE_SCENE_SCHEMA = vol.Schema({
    vol.Required(ATTR_DESCRIPTION): cv.string,
})

EDIT_AUTOMATION_SCHEMA = vol.Schema({
    vol.Required(ATTR_AUTOMATION_ID): cv.string,
    vol.Required(ATTR_DESCRIPTION): cv.string,
})

EDIT_SCRIPT_SCHEMA = vol.Schema({
    vol.Required(ATTR_SCRIPT_NAME): cv.string,
    vol.Required(ATTR_DESCRIPTION): cv.string,
})

EDIT_SCENE_SCHEMA = vol.Schema({
    vol.Required(ATTR_SCENE_ID): cv.string,
    vol.Required(ATTR_DESCRIPTION): cv.string,
})

DELETE_AUTOMATION_SCHEMA = vol.Schema({
    vol.Required(ATTR_AUTOMATION_ID): cv.string,
})

DELETE_SCRIPT_SCHEMA = vol.Schema({
    vol.Required(ATTR_SCRIPT_NAME): cv.string,
})

DELETE_SCENE_SCHEMA = vol.Schema({
    vol.Required(ATTR_SCENE_ID): cv.string,
})


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Voice Automation AI from a config entry."""
    hass.data.setdefault(DOMAIN, {})

    # Store configuration
    hass.data[DOMAIN][entry.entry_id] = {
        CONF_API_KEY: entry.data[CONF_API_KEY],
        CONF_MODEL: entry.data.get(CONF_MODEL, DEFAULT_MODEL),
        CONF_LANGUAGE: entry.data.get(CONF_LANGUAGE, DEFAULT_LANGUAGE),
    }

    # Create shared file manager
    if "file_manager" not in hass.data[DOMAIN]:
        hass.data[DOMAIN]["file_manager"] = HAConfigFileManager(hass)

    fm = hass.data[DOMAIN]["file_manager"]

    # ── Service handlers ──

    async def handle_create_automation(call: ServiceCall) -> None:
        """Handle create_automation service call."""
        description = call.data[ATTR_DESCRIPTION]
        validate_only = call.data.get(ATTR_VALIDATE_ONLY, False)
        preview = call.data.get(ATTR_PREVIEW, False)

        try:
            config = hass.data[DOMAIN][entry.entry_id]
            automation_yaml = await hass.async_add_executor_job(
                _generate_yaml,
                config[CONF_API_KEY],
                config[CONF_MODEL],
                config[CONF_LANGUAGE],
                description,
                "automation",
                fm.get_entities_context(),
            )

            automation_data = yaml.safe_load(automation_yaml)
            if isinstance(automation_data, list):
                automation_data = automation_data[0]

            if preview or validate_only:
                _LOGGER.info("Automation preview/validate: %s", automation_data.get("alias", "Unknown"))
                return

            await fm.add_automation(automation_data)
            _LOGGER.info("Automation created: %s", automation_data.get("alias", "Unknown"))

        except yaml.YAMLError as err:
            raise HomeAssistantError(f"Invalid YAML generated: {err}") from err
        except Exception as err:
            _LOGGER.error("Error creating automation: %s", err)
            raise HomeAssistantError(f"Failed to create automation: {err}") from err

    async def handle_validate_automation(call: ServiceCall) -> None:
        """Handle validate_automation service call."""
        yaml_content = call.data[ATTR_YAML_CONTENT]
        try:
            yaml.safe_load(yaml_content)
            _LOGGER.info("YAML validation passed")
        except yaml.YAMLError as err:
            raise HomeAssistantError(f"Invalid YAML: {err}") from err

    async def handle_create_script(call: ServiceCall) -> None:
        """Handle create_script service call."""
        description = call.data[ATTR_DESCRIPTION]
        script_name = call.data.get(ATTR_SCRIPT_NAME)

        try:
            config = hass.data[DOMAIN][entry.entry_id]
            script_yaml = await hass.async_add_executor_job(
                _generate_yaml,
                config[CONF_API_KEY],
                config[CONF_MODEL],
                config[CONF_LANGUAGE],
                description,
                "script",
                fm.get_entities_context(),
            )

            script_data = yaml.safe_load(script_yaml)
            if not script_name:
                script_name = script_data.get("alias", f"script_{int(time.time())}")
                script_name = script_name.lower().replace(" ", "_").replace("-", "_")

            await fm.add_script(script_name, script_data)
            _LOGGER.info("Script created: %s", script_name)

        except yaml.YAMLError as err:
            raise HomeAssistantError(f"Invalid YAML generated: {err}") from err
        except Exception as err:
            _LOGGER.error("Error creating script: %s", err)
            raise HomeAssistantError(f"Failed to create script: {err}") from err

    async def handle_create_scene(call: ServiceCall) -> None:
        """Handle create_scene service call."""
        description = call.data[ATTR_DESCRIPTION]

        try:
            config = hass.data[DOMAIN][entry.entry_id]
            scene_yaml = await hass.async_add_executor_job(
                _generate_yaml,
                config[CONF_API_KEY],
                config[CONF_MODEL],
                config[CONF_LANGUAGE],
                description,
                "scene",
                fm.get_entities_context(),
            )

            scene_data = yaml.safe_load(scene_yaml)
            if isinstance(scene_data, list):
                scene_data = scene_data[0]

            await fm.add_scene(scene_data)
            _LOGGER.info("Scene created: %s", scene_data.get("name", "Unknown"))

        except yaml.YAMLError as err:
            raise HomeAssistantError(f"Invalid YAML generated: {err}") from err
        except Exception as err:
            _LOGGER.error("Error creating scene: %s", err)
            raise HomeAssistantError(f"Failed to create scene: {err}") from err

    async def handle_edit_automation(call: ServiceCall) -> None:
        """Handle edit_automation service call."""
        automation_id = call.data[ATTR_AUTOMATION_ID]
        description = call.data[ATTR_DESCRIPTION]

        try:
            config = hass.data[DOMAIN][entry.entry_id]
            automations = await fm.read_automations()
            existing = next(
                (a for a in automations if str(a.get("id")) == str(automation_id)),
                None,
            )
            if not existing:
                raise HomeAssistantError(f"Automation '{automation_id}' not found")

            edit_prompt = (
                f"Here is the existing automation:\n{yaml.dump(existing, default_flow_style=False)}\n\n"
                f"Apply these changes: {description}\n\n"
                f"Return the complete updated automation as YAML."
            )
            updated_yaml = await hass.async_add_executor_job(
                _generate_yaml,
                config[CONF_API_KEY],
                config[CONF_MODEL],
                config[CONF_LANGUAGE],
                edit_prompt,
                "automation",
                fm.get_entities_context(),
            )
            updated_data = yaml.safe_load(updated_yaml)
            if isinstance(updated_data, list):
                updated_data = updated_data[0]

            await fm.update_automation(automation_id, updated_data)
            _LOGGER.info("Automation updated: %s", automation_id)

        except Exception as err:
            _LOGGER.error("Error editing automation: %s", err)
            raise HomeAssistantError(f"Failed to edit automation: {err}") from err

    async def handle_edit_script(call: ServiceCall) -> None:
        """Handle edit_script service call."""
        script_name = call.data[ATTR_SCRIPT_NAME]
        description = call.data[ATTR_DESCRIPTION]

        try:
            config = hass.data[DOMAIN][entry.entry_id]
            scripts = await fm.read_scripts()
            existing = scripts.get(script_name)
            if not existing:
                raise HomeAssistantError(f"Script '{script_name}' not found")

            edit_prompt = (
                f"Here is the existing script:\n{yaml.dump(existing, default_flow_style=False)}\n\n"
                f"Apply these changes: {description}\n\n"
                f"Return the complete updated script body as YAML."
            )
            updated_yaml = await hass.async_add_executor_job(
                _generate_yaml,
                config[CONF_API_KEY],
                config[CONF_MODEL],
                config[CONF_LANGUAGE],
                edit_prompt,
                "script",
                fm.get_entities_context(),
            )
            updated_data = yaml.safe_load(updated_yaml)
            await fm.update_script(script_name, updated_data)
            _LOGGER.info("Script updated: %s", script_name)

        except Exception as err:
            _LOGGER.error("Error editing script: %s", err)
            raise HomeAssistantError(f"Failed to edit script: {err}") from err

    async def handle_edit_scene(call: ServiceCall) -> None:
        """Handle edit_scene service call."""
        scene_id = call.data[ATTR_SCENE_ID]
        description = call.data[ATTR_DESCRIPTION]

        try:
            config = hass.data[DOMAIN][entry.entry_id]
            scenes = await fm.read_scenes()
            existing = next(
                (s for s in scenes if str(s.get("id")) == str(scene_id)),
                None,
            )
            if not existing:
                raise HomeAssistantError(f"Scene '{scene_id}' not found")

            edit_prompt = (
                f"Here is the existing scene:\n{yaml.dump(existing, default_flow_style=False)}\n\n"
                f"Apply these changes: {description}\n\n"
                f"Return the complete updated scene as YAML."
            )
            updated_yaml = await hass.async_add_executor_job(
                _generate_yaml,
                config[CONF_API_KEY],
                config[CONF_MODEL],
                config[CONF_LANGUAGE],
                edit_prompt,
                "scene",
                fm.get_entities_context(),
            )
            updated_data = yaml.safe_load(updated_yaml)
            if isinstance(updated_data, list):
                updated_data = updated_data[0]

            await fm.update_scene(scene_id, updated_data)
            _LOGGER.info("Scene updated: %s", scene_id)

        except Exception as err:
            _LOGGER.error("Error editing scene: %s", err)
            raise HomeAssistantError(f"Failed to edit scene: {err}") from err

    async def handle_delete_automation(call: ServiceCall) -> None:
        """Handle delete_automation service call."""
        try:
            await fm.delete_automation(call.data[ATTR_AUTOMATION_ID])
        except Exception as err:
            raise HomeAssistantError(f"Failed to delete automation: {err}") from err

    async def handle_delete_script(call: ServiceCall) -> None:
        """Handle delete_script service call."""
        try:
            await fm.delete_script(call.data[ATTR_SCRIPT_NAME])
        except Exception as err:
            raise HomeAssistantError(f"Failed to delete script: {err}") from err

    async def handle_delete_scene(call: ServiceCall) -> None:
        """Handle delete_scene service call."""
        try:
            await fm.delete_scene(call.data[ATTR_SCENE_ID])
        except Exception as err:
            raise HomeAssistantError(f"Failed to delete scene: {err}") from err

    async def handle_list_automations(call: ServiceCall) -> None:
        """Handle list_automations service call."""
        automations = await fm.read_automations()
        for a in automations:
            _LOGGER.info("Automation: id=%s, alias=%s", a.get("id", "N/A"), a.get("alias", "Unnamed"))

    async def handle_list_scripts(call: ServiceCall) -> None:
        """Handle list_scripts service call."""
        scripts = await fm.read_scripts()
        for name, body in scripts.items():
            _LOGGER.info("Script: name=%s, alias=%s", name, body.get("alias", name))

    async def handle_list_scenes(call: ServiceCall) -> None:
        """Handle list_scenes service call."""
        scenes = await fm.read_scenes()
        for s in scenes:
            _LOGGER.info("Scene: id=%s, name=%s", s.get("id", "N/A"), s.get("name", "Unnamed"))

    # ── Register all services ──

    services = [
        (SERVICE_CREATE_AUTOMATION, handle_create_automation, CREATE_AUTOMATION_SCHEMA),
        (SERVICE_VALIDATE_AUTOMATION, handle_validate_automation, VALIDATE_AUTOMATION_SCHEMA),
        (SERVICE_CREATE_SCRIPT, handle_create_script, CREATE_SCRIPT_SCHEMA),
        (SERVICE_CREATE_SCENE, handle_create_scene, CREATE_SCENE_SCHEMA),
        (SERVICE_EDIT_AUTOMATION, handle_edit_automation, EDIT_AUTOMATION_SCHEMA),
        (SERVICE_EDIT_SCRIPT, handle_edit_script, EDIT_SCRIPT_SCHEMA),
        (SERVICE_EDIT_SCENE, handle_edit_scene, EDIT_SCENE_SCHEMA),
        (SERVICE_DELETE_AUTOMATION, handle_delete_automation, DELETE_AUTOMATION_SCHEMA),
        (SERVICE_DELETE_SCRIPT, handle_delete_script, DELETE_SCRIPT_SCHEMA),
        (SERVICE_DELETE_SCENE, handle_delete_scene, DELETE_SCENE_SCHEMA),
        (SERVICE_LIST_AUTOMATIONS, handle_list_automations, None),
        (SERVICE_LIST_SCRIPTS, handle_list_scripts, None),
        (SERVICE_LIST_SCENES, handle_list_scenes, None),
    ]

    for service_name, handler, schema in services:
        hass.services.async_register(DOMAIN, service_name, handler, schema=schema)

    # Forward setup to conversation platform
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    # Unload conversation platform
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

    # Remove services
    all_services = [
        SERVICE_CREATE_AUTOMATION, SERVICE_VALIDATE_AUTOMATION,
        SERVICE_CREATE_SCRIPT, SERVICE_CREATE_SCENE,
        SERVICE_EDIT_AUTOMATION, SERVICE_EDIT_SCRIPT, SERVICE_EDIT_SCENE,
        SERVICE_DELETE_AUTOMATION, SERVICE_DELETE_SCRIPT, SERVICE_DELETE_SCENE,
        SERVICE_LIST_AUTOMATIONS, SERVICE_LIST_SCRIPTS, SERVICE_LIST_SCENES,
    ]
    for service_name in all_services:
        hass.services.async_remove(DOMAIN, service_name)

    # Clean up entry data
    hass.data[DOMAIN].pop(entry.entry_id, None)

    # Remove file manager if no more entries
    remaining_entries = [
        key for key in hass.data[DOMAIN] if key != "file_manager"
    ]
    if not remaining_entries:
        hass.data[DOMAIN].pop("file_manager", None)

    return unload_ok


def _generate_yaml(
    api_key: str,
    model: str,
    language: str,
    description: str,
    config_type: str,
    entities_context: str,
) -> str:
    """Generate YAML using Claude API (blocking - called from executor)."""
    prompts = {
        "automation": (
            "Generate a Home Assistant automation in YAML format for this request:\n\n"
            f"{description}\n\n"
            "Requirements:\n"
            "- Valid, complete YAML\n"
            "- Include alias, description, mode (default: single), trigger, condition, and action\n"
            "- Use only entity IDs that exist (listed below)\n"
            "- Return ONLY the YAML block as a single dict (not a list), no markdown, no explanations\n"
            "- Quote 'on'/'off' values as strings\n\n"
        ),
        "script": (
            "Generate a Home Assistant script body in YAML format for this request:\n\n"
            f"{description}\n\n"
            "Requirements:\n"
            "- Valid, complete YAML\n"
            "- Include alias, description, mode (default: single), and sequence\n"
            "- Use only entity IDs that exist (listed below)\n"
            "- Return ONLY the script body as a YAML dict (not including the script name key), "
            "no markdown, no explanations\n"
            "- Quote 'on'/'off' values as strings\n\n"
        ),
        "scene": (
            "Generate a Home Assistant scene in YAML format for this request:\n\n"
            f"{description}\n\n"
            "Requirements:\n"
            "- Valid, complete YAML\n"
            "- Include name and entities with their desired states\n"
            "- Use only entity IDs that exist (listed below)\n"
            "- Return ONLY the scene as a YAML dict, no markdown, no explanations\n"
            "- Quote 'on'/'off' values as strings\n\n"
        ),
    }

    prompt = prompts.get(config_type, prompts["automation"])
    prompt += f"Available entities:\n{entities_context}\n\n"
    prompt += f"Respond in {language}."

    client = anthropic.Anthropic(api_key=api_key)

    try:
        message = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
            timeout=API_TIMEOUT,
        )

        yaml_content = message.content[0].text.strip()

        # Clean up markdown code blocks if present
        if yaml_content.startswith("```"):
            lines = yaml_content.split("\n")
            yaml_content = "\n".join(lines[1:-1]) if len(lines) > 2 else yaml_content

        yaml_content = yaml_content.replace("```yaml", "").replace("```", "").strip()
        return yaml_content

    except Exception as err:
        _LOGGER.error("Error calling Claude API: %s", err)
        raise HomeAssistantError(f"Failed to generate YAML: {err}") from err
