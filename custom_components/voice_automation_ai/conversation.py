"""Conversation agent for Voice Automation AI."""
from __future__ import annotations

import json
import logging
import uuid
from collections import OrderedDict
from typing import Any

import yaml


class _InputLoader(yaml.SafeLoader):
    """YAML loader that handles !input tags as plain strings."""

    pass


def _input_constructor(loader, node):
    return f"!input {loader.construct_scalar(node)}"


_InputLoader.add_constructor("!input", _input_constructor)

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
    ALLOWED_SERVICE_DOMAINS,
    API_TIMEOUT,
    BLOCKED_SERVICE_DOMAINS,
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
    DEFAULT_PROVIDER,
    DOMAIN,
    OLLAMA_TIMEOUT,
    PROVIDER_ANTHROPIC,
    SENSITIVE_ATTRIBUTE_KEYS,
)
from .file_manager import HAConfigFileManager
from .llm_client import BaseLLMClient, create_llm_client

_LOGGER = logging.getLogger(__name__)

# Maximum number of tool-use round-trips per conversation turn
MAX_TOOL_ROUNDS = 5

# Maximum number of concurrent conversations to track
MAX_CONVERSATIONS = 50

SYSTEM_PROMPT = """You are a Home Assistant smart home assistant. You can control \
devices and manage automations, scripts, scenes, and blueprints.

You can:
1. Control devices - turn on/off lights, lock doors, set temperatures, etc. \
Use the call_service tool.
2. Check device states - use get_entity_state to see current status.
3. Manage automations - create, edit, delete, and list automations.
4. Manage scripts - create, edit, delete, and list scripts.
5. Manage scenes - create, edit, delete, and list scenes.
6. Manage blueprints - create, read, edit, delete, and list blueprints. \
Blueprints are reusable templates. Changes to a blueprint affect ALL \
automations/scripts using it. Blueprint YAML uses !input tags for \
configurable parameters.

Guidelines:
- Always use the tools to perform actions - never just describe what to do.
- For device control, use call_service with the correct domain and service.
- For automations: YAML dict with alias, description, mode, trigger, condition, action.
- For scripts: YAML dict with alias, description, mode, sequence.
- For scenes: YAML dict with name, id, entities.
- For blueprints: YAML with blueprint key (name, description, domain, input), \
triggers, conditions, actions. Use !input to reference inputs.
- Only use entity IDs that exist (listed below).
- When editing, preserve fields the user didn't ask to change.
- Before deleting, confirm what will be removed.
- Respond in the user's language.
- Be concise - this is a voice interface.

Available Home Assistant entities:
{entities}
"""


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
        # Conversation history keyed by conversation_id
        self._conversations: OrderedDict[str, list[dict]] = OrderedDict()

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

    def _create_llm_client(self) -> BaseLLMClient:
        """Create an LLM client from the current config."""
        config = self._config
        provider = config.get(CONF_PROVIDER, DEFAULT_PROVIDER)

        if provider == PROVIDER_ANTHROPIC:
            return create_llm_client(
                provider,
                api_key=config[CONF_API_KEY],
                timeout=API_TIMEOUT,
            )
        else:
            return create_llm_client(
                provider,
                host=config.get(CONF_OLLAMA_HOST, DEFAULT_OLLAMA_HOST),
                timeout=OLLAMA_TIMEOUT,
                temperature=config.get(CONF_TEMPERATURE),
                top_p=config.get(CONF_TOP_P),
            )

    async def async_process(
        self,
        user_input: ConversationInput,
    ) -> ConversationResult:
        """Process a conversation turn."""
        config = self._config
        model = config.get(CONF_MODEL, DEFAULT_MODEL)
        language = config.get(CONF_LANGUAGE, DEFAULT_LANGUAGE)
        max_tokens = config.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        max_history_turns = config.get(CONF_MAX_HISTORY_TURNS, DEFAULT_MAX_HISTORY_TURNS)

        # Resolve or create conversation ID
        conversation_id = user_input.conversation_id or str(uuid.uuid4())

        entities_context = await self.hass.async_add_executor_job(
            self._file_manager.get_entities_context
        )
        system_prompt = SYSTEM_PROMPT.format(entities=entities_context)

        # Retrieve clean history (only user/assistant text pairs).
        # Tool-use exchanges are kept out of stored history to prevent
        # the Anthropic API from rejecting orphaned tool_result blocks
        # after history trimming.
        history = list(self._conversations.get(conversation_id, []))

        # Build working messages for this turn (history + new user message).
        # _call_with_tools may mutate this list by adding tool exchanges.
        messages = list(history)
        messages.append({"role": "user", "content": user_input.text})

        try:
            client = self._create_llm_client()
            response_text = await self._call_with_tools(
                client, model, system_prompt, messages, max_tokens
            )
        except ConnectionError as err:
            _LOGGER.error("LLM connection error: %s", err)
            response_text = (
                "Sorry, I could not reach the language model. "
                "Please check that the service is running and accessible."
            )
        except TimeoutError as err:
            _LOGGER.error("LLM request timed out: %s", err)
            response_text = (
                "Sorry, the language model took too long to respond. "
                "Try a shorter request or check if the model is overloaded."
            )
        except Exception as err:
            _LOGGER.error("Error in conversation: %s", err, exc_info=True)
            response_text = (
                "Sorry, I encountered an unexpected error. "
                "Check the Home Assistant logs for details."
            )

        # Only store clean user/assistant text in history (no tool exchanges)
        history.append({"role": "user", "content": user_input.text})
        history.append({"role": "assistant", "content": response_text})

        # Trim history to keep token usage bounded
        if len(history) > max_history_turns * 2:
            history = history[-(max_history_turns * 2):]

        # Store updated history, evicting oldest conversation if needed
        self._conversations[conversation_id] = history
        self._conversations.move_to_end(conversation_id)
        while len(self._conversations) > MAX_CONVERSATIONS:
            self._conversations.popitem(last=False)

        intent_response = intent.IntentResponse(language=language)
        intent_response.async_set_speech(response_text)

        return ConversationResult(
            conversation_id=conversation_id,
            response=intent_response,
        )

    async def _call_with_tools(
        self,
        client: BaseLLMClient,
        model: str,
        system_prompt: str,
        messages: list[dict],
        max_tokens: int,
    ) -> str:
        """Call LLM with tool-use support, handling tool calls in a loop."""

        async def _do_create_message(msgs: list[dict]) -> Any:
            if client.is_async:
                return await client.async_create_message(
                    model=model,
                    system_prompt=system_prompt,
                    messages=msgs,
                    max_tokens=max_tokens,
                    tools=True,
                )
            else:
                return await self.hass.async_add_executor_job(
                    client.create_message,
                    model, system_prompt, msgs, max_tokens, True,
                )

        for _round in range(MAX_TOOL_ROUNDS):
            response = await _do_create_message(messages)

            if response.has_tool_calls:
                tool_results = []
                for tc in response.tool_calls:
                    result = await self._execute_tool(tc["name"], tc["arguments"])
                    tool_results.append({
                        "tool_call_id": tc["id"],
                        "content": json.dumps(result, default=str),
                    })

                client.add_tool_results(messages, response, tool_results)
            else:
                return response.text or "Done."

        return "I completed the requested operations."

    @staticmethod
    def _check_yaml_for_blocked_services(data: dict | list) -> str | None:
        """Scan parsed YAML for references to blocked service domains.

        Recursively walks the data structure checking 'service' and 'action' keys,
        including nested choose/parallel/if-then/repeat/sequence blocks.
        Returns an error message if a blocked service is found, else None.
        """

        def _check_value(value) -> str | None:
            if isinstance(value, dict):
                for key in ("service", "action"):
                    svc = value.get(key)
                    if isinstance(svc, str) and "." in svc:
                        svc_domain = svc.split(".")[0]
                        if svc_domain in BLOCKED_SERVICE_DOMAINS:
                            return (
                                f"Blocked: generated YAML references restricted "
                                f"service domain '{svc_domain}'."
                            )
                for key in (
                    "action", "sequence", "then", "else", "default",
                    "choose", "parallel", "repeat",
                ):
                    nested = value.get(key)
                    if nested is not None:
                        result = _check_value(nested)
                        if result:
                            return result
                for v in value.values():
                    if isinstance(v, (dict, list)):
                        result = _check_value(v)
                        if result:
                            return result
            elif isinstance(value, list):
                for item in value:
                    result = _check_value(item)
                    if result:
                        return result
            return None

        return _check_value(data)

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
                blocked = self._check_yaml_for_blocked_services(data)
                if blocked:
                    return {"success": False, "error": blocked}
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
                blocked = self._check_yaml_for_blocked_services(data)
                if blocked:
                    return {"success": False, "error": blocked}
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
                blocked = self._check_yaml_for_blocked_services(data)
                if blocked:
                    return {"success": False, "error": blocked}
                name = await fm.add_script(tool_input["script_name"], data)
                return {"success": True, "script_name": name}

            elif tool_name == "edit_script":
                data = yaml.safe_load(tool_input["yaml_content"])
                blocked = self._check_yaml_for_blocked_services(data)
                if blocked:
                    return {"success": False, "error": blocked}
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

            elif tool_name == "call_service":
                domain = tool_input["domain"]
                service = tool_input["service"]
                entity_id = tool_input["entity_id"]

                # Security: block dangerous service domains
                if domain in BLOCKED_SERVICE_DOMAINS:
                    _LOGGER.warning(
                        "Blocked call to restricted domain: %s.%s", domain, service
                    )
                    return {
                        "success": False,
                        "error": f"Service domain '{domain}' is restricted for security reasons.",
                    }
                if domain not in ALLOWED_SERVICE_DOMAINS:
                    _LOGGER.warning(
                        "Blocked call to non-allowlisted domain: %s.%s", domain, service
                    )
                    return {
                        "success": False,
                        "error": (
                            f"Service domain '{domain}' is not in the allowed list. "
                            f"Allowed: {', '.join(sorted(ALLOWED_SERVICE_DOMAINS))}"
                        ),
                    }

                service_data = {}
                if "service_data" in tool_input and tool_input["service_data"]:
                    service_data = json.loads(tool_input["service_data"])
                service_data["entity_id"] = entity_id

                await self.hass.services.async_call(
                    domain, service, service_data, blocking=True
                )
                return {
                    "success": True,
                    "called": f"{domain}.{service}",
                    "entity_id": entity_id,
                }

            elif tool_name == "get_entity_state":
                entity_id = tool_input["entity_id"]
                state = self.hass.states.get(entity_id)
                if state is None:
                    return {"success": False, "error": f"Entity '{entity_id}' not found"}
                # Security: strip sensitive attributes before sending to LLM
                safe_attrs = {
                    k: v
                    for k, v in state.attributes.items()
                    if k.lower() not in SENSITIVE_ATTRIBUTE_KEYS
                }
                return {
                    "success": True,
                    "entity_id": entity_id,
                    "state": state.state,
                    "attributes": safe_attrs,
                }

            # ── Blueprint tools ──

            elif tool_name == "list_blueprints":
                bp_domain = tool_input.get("domain", "automation")
                blueprints = await fm.read_blueprints(bp_domain)
                return {"success": True, "count": len(blueprints), "blueprints": blueprints}

            elif tool_name == "read_blueprint":
                bp_domain = tool_input.get("domain", "automation")
                content = await fm.read_blueprint(bp_domain, tool_input["blueprint_name"])
                return {"success": True, "blueprint_name": tool_input["blueprint_name"], "content": content}

            elif tool_name == "create_blueprint":
                bp_domain = tool_input.get("domain", "automation")
                yaml_content = tool_input["yaml_content"]
                # Security: check for blocked services using !input-aware loader
                try:
                    data = yaml.load(yaml_content, Loader=_InputLoader)
                    if data:
                        blocked = self._check_yaml_for_blocked_services(data)
                        if blocked:
                            return {"success": False, "error": blocked}
                except yaml.YAMLError as err:
                    _LOGGER.warning("Blueprint YAML parse failed during security check: %s", err)
                    return {"success": False, "error": "Blueprint YAML could not be parsed for security validation."}
                await fm.add_blueprint(bp_domain, tool_input["blueprint_name"], yaml_content)
                return {"success": True, "blueprint_name": tool_input["blueprint_name"]}

            elif tool_name == "edit_blueprint":
                bp_domain = tool_input.get("domain", "automation")
                yaml_content = tool_input["yaml_content"]
                try:
                    data = yaml.load(yaml_content, Loader=_InputLoader)
                    if data:
                        blocked = self._check_yaml_for_blocked_services(data)
                        if blocked:
                            return {"success": False, "error": blocked}
                except yaml.YAMLError as err:
                    _LOGGER.warning("Blueprint YAML parse failed during security check: %s", err)
                    return {"success": False, "error": "Blueprint YAML could not be parsed for security validation."}
                await fm.update_blueprint(bp_domain, tool_input["blueprint_name"], yaml_content)
                return {"success": True, "blueprint_name": tool_input["blueprint_name"]}

            elif tool_name == "delete_blueprint":
                bp_domain = tool_input.get("domain", "automation")
                await fm.delete_blueprint(bp_domain, tool_input["blueprint_name"])
                return {"success": True, "blueprint_name": tool_input["blueprint_name"]}

            else:
                return {"success": False, "error": f"Unknown tool: {tool_name}"}

        except json.JSONDecodeError as err:
            return {"success": False, "error": f"Invalid JSON in service_data: {err}"}
        except yaml.YAMLError as err:
            return {"success": False, "error": f"Invalid YAML: {err}"}
        except ValueError as err:
            return {"success": False, "error": str(err)}
        except Exception as err:
            _LOGGER.error("Tool execution error (%s): %s", tool_name, err, exc_info=True)
            return {"success": False, "error": "An unexpected error occurred. Check the logs for details."}
