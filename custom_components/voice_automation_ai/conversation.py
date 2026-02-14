"""Conversation agent for Voice Automation AI."""
from __future__ import annotations

import json
import logging
from typing import Any

import anthropic
import yaml

from homeassistant.components.conversation import (
    ConversationEntity,
    ConversationInput,
    ConversationResult,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    CONF_API_KEY,
    CONF_LANGUAGE,
    CONF_MODEL,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DOMAIN,
    MAX_TOKENS,
)
from .file_manager import HAConfigFileManager

_LOGGER = logging.getLogger(__name__)

# Maximum number of tool-use round-trips per conversation turn
MAX_TOOL_ROUNDS = 5

SYSTEM_PROMPT = """You are a Home Assistant voice automation assistant. You help users \
manage their smart home by creating, editing, deleting, and listing automations, \
scripts, and scenes.

When a user asks you to create, modify, or manage automations, scripts, or scenes, \
use the provided tools. Always use the tools - never just describe what YAML would \
look like without actually creating it.

Guidelines:
- When creating automations/scripts/scenes, generate valid Home Assistant YAML.
- For automations: use list format starting with '- alias:'. Include alias, \
description, mode (default: single), trigger, condition (can be []), and action.
- For scripts: use dict format with just the script body (alias, description, \
mode, sequence). The script name/key is provided separately.
- For scenes: use dict format with name, id, and entities.
- Always use entity IDs that exist in the user's Home Assistant instance (listed below).
- When editing, preserve fields the user didn't ask to change.
- Before deleting, confirm with the user what will be removed.
- When listing, provide a concise summary (alias/name and ID).
- Respond in the user's language.
- Be concise in your responses - this is a voice interface.

Available Home Assistant entities:
{entities}
"""

# Claude tool definitions for managing HA config
TOOLS = [
    {
        "name": "list_automations",
        "description": "List all automations defined in automations.yaml. Returns a list of automations with their IDs and aliases.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "create_automation",
        "description": "Create a new Home Assistant automation. The yaml_content should be a valid automation dict (NOT a list) with keys: alias, description, mode, trigger, condition, action.",
        "input_schema": {
            "type": "object",
            "properties": {
                "yaml_content": {
                    "type": "string",
                    "description": "The automation as a YAML string. Must be a single automation dict with alias, description, mode, trigger, condition, and action.",
                },
            },
            "required": ["yaml_content"],
        },
    },
    {
        "name": "edit_automation",
        "description": "Edit an existing automation by its ID. Provide the complete updated automation as YAML.",
        "input_schema": {
            "type": "object",
            "properties": {
                "automation_id": {
                    "type": "string",
                    "description": "The ID of the automation to edit.",
                },
                "yaml_content": {
                    "type": "string",
                    "description": "The complete updated automation as a YAML string.",
                },
            },
            "required": ["automation_id", "yaml_content"],
        },
    },
    {
        "name": "delete_automation",
        "description": "Delete an automation by its ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "automation_id": {
                    "type": "string",
                    "description": "The ID of the automation to delete.",
                },
            },
            "required": ["automation_id"],
        },
    },
    {
        "name": "list_scripts",
        "description": "List all scripts defined in scripts.yaml. Returns script names and their aliases.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "create_script",
        "description": "Create a new Home Assistant script. Provide a name (lowercase, underscores) and the script body as YAML with keys: alias, description, mode, sequence.",
        "input_schema": {
            "type": "object",
            "properties": {
                "script_name": {
                    "type": "string",
                    "description": "The script identifier (lowercase, underscores only, no hyphens).",
                },
                "yaml_content": {
                    "type": "string",
                    "description": "The script body as a YAML string with alias, description, mode, and sequence.",
                },
            },
            "required": ["script_name", "yaml_content"],
        },
    },
    {
        "name": "edit_script",
        "description": "Edit an existing script by name. Provide the complete updated script body as YAML.",
        "input_schema": {
            "type": "object",
            "properties": {
                "script_name": {
                    "type": "string",
                    "description": "The name/key of the script to edit.",
                },
                "yaml_content": {
                    "type": "string",
                    "description": "The complete updated script body as a YAML string.",
                },
            },
            "required": ["script_name", "yaml_content"],
        },
    },
    {
        "name": "delete_script",
        "description": "Delete a script by name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "script_name": {
                    "type": "string",
                    "description": "The name/key of the script to delete.",
                },
            },
            "required": ["script_name"],
        },
    },
    {
        "name": "list_scenes",
        "description": "List all scenes defined in scenes.yaml. Returns scene names and IDs.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "create_scene",
        "description": "Create a new Home Assistant scene. The yaml_content should be a scene dict with name and entities.",
        "input_schema": {
            "type": "object",
            "properties": {
                "yaml_content": {
                    "type": "string",
                    "description": "The scene as a YAML string with name and entities.",
                },
            },
            "required": ["yaml_content"],
        },
    },
    {
        "name": "edit_scene",
        "description": "Edit an existing scene by its ID. Provide the complete updated scene as YAML.",
        "input_schema": {
            "type": "object",
            "properties": {
                "scene_id": {
                    "type": "string",
                    "description": "The ID of the scene to edit.",
                },
                "yaml_content": {
                    "type": "string",
                    "description": "The complete updated scene as a YAML string.",
                },
            },
            "required": ["scene_id", "yaml_content"],
        },
    },
    {
        "name": "delete_scene",
        "description": "Delete a scene by its ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "scene_id": {
                    "type": "string",
                    "description": "The ID of the scene to delete.",
                },
            },
            "required": ["scene_id"],
        },
    },
]


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up conversation entity from config entry."""
    agent = VoiceAutomationAIConversationAgent(hass, config_entry)
    async_add_entities([agent])


class VoiceAutomationAIConversationAgent(ConversationEntity):
    """Voice Automation AI conversation agent."""

    _attr_has_entity_name = True
    _attr_name = "Voice Automation AI"

    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry) -> None:
        """Initialize the conversation agent."""
        self.hass = hass
        self._config_entry = config_entry
        self._attr_unique_id = f"{config_entry.entry_id}_conversation"

    @property
    def supported_languages(self) -> str:
        """Return supported languages (all)."""
        return "*"

    @property
    def _config(self) -> dict[str, Any]:
        """Get the current config data."""
        return self.hass.data[DOMAIN][self._config_entry.entry_id]

    @property
    def _file_manager(self) -> HAConfigFileManager:
        """Get the file manager instance."""
        return self.hass.data[DOMAIN]["file_manager"]

    async def async_process(
        self,
        user_input: ConversationInput,
    ) -> ConversationResult:
        """Process a conversation turn."""
        config = self._config
        api_key = config[CONF_API_KEY]
        model = config.get(CONF_MODEL, DEFAULT_MODEL)
        language = config.get(CONF_LANGUAGE, DEFAULT_LANGUAGE)

        entities_context = await self.hass.async_add_executor_job(
            self._file_manager.get_entities_context
        )
        system_prompt = SYSTEM_PROMPT.format(entities=entities_context)

        # Build messages from the user input
        messages = [{"role": "user", "content": user_input.text}]

        try:
            response_text = await self._call_claude_with_tools(
                api_key, model, system_prompt, messages
            )
        except Exception as err:
            _LOGGER.error("Error in conversation: %s", err)
            response_text = f"Sorry, I encountered an error: {err}"

        # Build the response
        intent_response = intent.IntentResponse(language=language)
        intent_response.async_set_speech(response_text)

        return ConversationResult(
            conversation_id=user_input.conversation_id,
            response=intent_response,
        )

    async def _call_claude_with_tools(
        self,
        api_key: str,
        model: str,
        system_prompt: str,
        messages: list[dict],
    ) -> str:
        """Call Claude API with tool-use support, handling tool calls in a loop."""
        client = anthropic.Anthropic(api_key=api_key)

        def _create_message(msgs: list[dict]) -> Any:
            """Make a blocking Claude API call."""
            return client.messages.create(
                model=model,
                max_tokens=MAX_TOKENS,
                system=system_prompt,
                messages=msgs,
                tools=TOOLS,
            )

        for _round in range(MAX_TOOL_ROUNDS):
            response = await self.hass.async_add_executor_job(
                _create_message, messages
            )

            # Check if Claude wants to use a tool
            if response.stop_reason == "tool_use":
                # Process all tool calls in this response
                tool_results = []
                assistant_content = []

                for block in response.content:
                    if block.type == "text":
                        assistant_content.append({
                            "type": "text",
                            "text": block.text,
                        })
                    elif block.type == "tool_use":
                        assistant_content.append({
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        })

                        # Execute the tool
                        result = await self._execute_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result, default=str),
                        })

                # Add assistant message and tool results to conversation
                messages.append({"role": "assistant", "content": assistant_content})
                messages.append({"role": "user", "content": tool_results})

            else:
                # Claude returned a final text response
                text_parts = []
                for block in response.content:
                    if block.type == "text":
                        text_parts.append(block.text)
                return " ".join(text_parts) if text_parts else "Done."

        return "I completed the requested operations."

    async def _execute_tool(self, tool_name: str, tool_input: dict) -> dict:
        """Execute a tool call and return the result."""
        fm = self._file_manager

        try:
            if tool_name == "list_automations":
                automations = await fm.read_automations()
                summary = [
                    {"id": a.get("id", "N/A"), "alias": a.get("alias", "Unnamed")}
                    for a in automations
                ]
                return {"success": True, "count": len(summary), "automations": summary}

            elif tool_name == "create_automation":
                data = yaml.safe_load(tool_input["yaml_content"])
                if isinstance(data, list):
                    data = data[0]
                automation_id = await fm.add_automation(data)
                return {
                    "success": True,
                    "automation_id": automation_id,
                    "alias": data.get("alias", "Unknown"),
                }

            elif tool_name == "edit_automation":
                data = yaml.safe_load(tool_input["yaml_content"])
                if isinstance(data, list):
                    data = data[0]
                await fm.update_automation(tool_input["automation_id"], data)
                return {"success": True, "automation_id": tool_input["automation_id"]}

            elif tool_name == "delete_automation":
                await fm.delete_automation(tool_input["automation_id"])
                return {"success": True, "automation_id": tool_input["automation_id"]}

            elif tool_name == "list_scripts":
                scripts = await fm.read_scripts()
                summary = [
                    {"name": name, "alias": body.get("alias", name)}
                    for name, body in scripts.items()
                ]
                return {"success": True, "count": len(summary), "scripts": summary}

            elif tool_name == "create_script":
                data = yaml.safe_load(tool_input["yaml_content"])
                name = await fm.add_script(tool_input["script_name"], data)
                return {"success": True, "script_name": name}

            elif tool_name == "edit_script":
                data = yaml.safe_load(tool_input["yaml_content"])
                await fm.update_script(tool_input["script_name"], data)
                return {"success": True, "script_name": tool_input["script_name"]}

            elif tool_name == "delete_script":
                await fm.delete_script(tool_input["script_name"])
                return {"success": True, "script_name": tool_input["script_name"]}

            elif tool_name == "list_scenes":
                scenes = await fm.read_scenes()
                summary = [
                    {"id": s.get("id", "N/A"), "name": s.get("name", "Unnamed")}
                    for s in scenes
                ]
                return {"success": True, "count": len(summary), "scenes": summary}

            elif tool_name == "create_scene":
                data = yaml.safe_load(tool_input["yaml_content"])
                if isinstance(data, list):
                    data = data[0]
                scene_id = await fm.add_scene(data)
                return {
                    "success": True,
                    "scene_id": scene_id,
                    "name": data.get("name", "Unknown"),
                }

            elif tool_name == "edit_scene":
                data = yaml.safe_load(tool_input["yaml_content"])
                if isinstance(data, list):
                    data = data[0]
                await fm.update_scene(tool_input["scene_id"], data)
                return {"success": True, "scene_id": tool_input["scene_id"]}

            elif tool_name == "delete_scene":
                await fm.delete_scene(tool_input["scene_id"])
                return {"success": True, "scene_id": tool_input["scene_id"]}

            else:
                return {"success": False, "error": f"Unknown tool: {tool_name}"}

        except yaml.YAMLError as err:
            return {"success": False, "error": f"Invalid YAML: {err}"}
        except ValueError as err:
            return {"success": False, "error": str(err)}
        except Exception as err:
            _LOGGER.error("Tool execution error (%s): %s", tool_name, err)
            return {"success": False, "error": str(err)}
