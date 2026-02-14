"""Tests for VoiceAutomationAIConversationAgent."""
from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from custom_components.voice_automation_ai.const import (
    ALLOWED_SERVICE_DOMAINS,
    BLOCKED_SERVICE_DOMAINS,
    DOMAIN,
    SENSITIVE_ATTRIBUTE_KEYS,
)
from custom_components.voice_automation_ai.conversation import (
    MAX_CONVERSATIONS,
    MAX_TOOL_ROUNDS,
    VoiceAutomationAIConversationAgent,
)


@pytest.fixture
def agent(hass, config_entry):
    """Create a conversation agent with mocked dependencies."""
    hass.data[DOMAIN] = {
        config_entry.entry_id: {
            "provider": "anthropic",
            "api_key": "sk-ant-test",
            "model": "claude-sonnet-4-5-20250929",
            "language": "en",
            "max_tokens": 4096,
            "max_history_turns": 10,
        },
        "file_manager": MagicMock(),
    }
    return VoiceAutomationAIConversationAgent(hass, config_entry)


@pytest.fixture
def file_manager_mock(agent):
    """Get the mocked file manager from the agent."""
    return agent.hass.data[DOMAIN]["file_manager"]


# ── Security: call_service domain enforcement ──


class TestCallServiceSecurity:
    """Test that call_service enforces domain allow/block lists."""

    async def test_blocked_domain_rejected(self, agent):
        """Every blocked domain must be rejected."""
        for domain in BLOCKED_SERVICE_DOMAINS:
            result = await agent._execute_tool(
                "call_service",
                {"domain": domain, "service": "anything", "entity_id": f"{domain}.test"},
            )
            assert result["success"] is False, f"Blocked domain {domain} was allowed"
            assert "restricted" in result["error"].lower()

    async def test_non_allowlisted_domain_rejected(self, agent):
        """Domains not in the allow list (and not blocked) must be rejected."""
        result = await agent._execute_tool(
            "call_service",
            {"domain": "notify", "service": "send_message", "entity_id": "notify.mobile"},
        )
        assert result["success"] is False
        assert "not in the allowed list" in result["error"]

    async def test_allowed_domain_accepted(self, agent):
        """Allowed domains must pass security and call the service."""
        agent.hass.services.async_call = AsyncMock()

        result = await agent._execute_tool(
            "call_service",
            {"domain": "light", "service": "turn_on", "entity_id": "light.living_room"},
        )
        assert result["success"] is True
        assert result["called"] == "light.turn_on"
        agent.hass.services.async_call.assert_awaited_once()

    async def test_all_allowed_domains_pass(self, agent):
        """Smoke test: every allowed domain should succeed."""
        agent.hass.services.async_call = AsyncMock()
        for domain in ALLOWED_SERVICE_DOMAINS:
            result = await agent._execute_tool(
                "call_service",
                {"domain": domain, "service": "turn_on", "entity_id": f"{domain}.test_entity"},
            )
            assert result["success"] is True, f"Allowed domain {domain} was rejected"

    async def test_service_data_json_parsed(self, agent):
        agent.hass.services.async_call = AsyncMock()
        result = await agent._execute_tool(
            "call_service",
            {
                "domain": "light",
                "service": "turn_on",
                "entity_id": "light.living_room",
                "service_data": '{"brightness": 128, "color_name": "red"}',
            },
        )
        assert result["success"] is True
        call_args = agent.hass.services.async_call.call_args
        service_data = call_args[0][2]
        assert service_data["brightness"] == 128
        assert service_data["entity_id"] == "light.living_room"

    async def test_invalid_json_service_data(self, agent):
        result = await agent._execute_tool(
            "call_service",
            {
                "domain": "light",
                "service": "turn_on",
                "entity_id": "light.test",
                "service_data": "not valid json{{{",
            },
        )
        assert result["success"] is False
        assert "Invalid JSON" in result["error"]


# ── Security: get_entity_state sensitive attribute stripping ──


class TestGetEntityStateSecurity:
    """Test that get_entity_state strips sensitive attributes."""

    async def test_strips_all_sensitive_attributes(self, agent):
        state = MagicMock()
        state.state = "home"
        state.attributes = {
            "friendly_name": "John",
            "latitude": 51.5074,
            "longitude": -0.1278,
            "gps_accuracy": 10,
            "token": "secret123",
            "access_token": "bearer_xyz",
            "api_key": "sk-secret",
            "api_token": "tok_123",
            "password": "p@ssw0rd",
            "secret": "mysecret",
            "ip_address": "192.168.1.1",
            "mac_address": "AA:BB:CC:DD:EE:FF",
            "stream_source": "rtsp://camera/stream",
            "entity_picture": "/local/pic.jpg",
            "icon": "mdi:account",  # NOT sensitive
        }
        agent.hass.states.get = MagicMock(return_value=state)

        result = await agent._execute_tool(
            "get_entity_state", {"entity_id": "person.john"}
        )

        assert result["success"] is True
        attrs = result["attributes"]
        assert "friendly_name" in attrs
        assert "icon" in attrs
        for key in SENSITIVE_ATTRIBUTE_KEYS:
            assert key not in attrs, f"Sensitive key '{key}' was not stripped"

    async def test_strips_case_insensitive(self, agent):
        """k.lower() comparison should catch mixed-case keys."""
        state = MagicMock()
        state.state = "on"
        state.attributes = {
            "Latitude": 51.0,
            "API_KEY": "secret",
            "Password": "p@ss",
            "brightness": 255,
        }
        agent.hass.states.get = MagicMock(return_value=state)

        result = await agent._execute_tool(
            "get_entity_state", {"entity_id": "light.test"}
        )
        attrs = result["attributes"]
        assert "brightness" in attrs
        assert "Latitude" not in attrs
        assert "API_KEY" not in attrs
        assert "Password" not in attrs

    async def test_entity_not_found(self, agent):
        agent.hass.states.get = MagicMock(return_value=None)
        result = await agent._execute_tool(
            "get_entity_state", {"entity_id": "light.nonexistent"}
        )
        assert result["success"] is False
        assert "not found" in result["error"]


# ── Security: YAML blocked service scanning ──


class TestYAMLBlockedServiceCheck:
    """Test _check_yaml_for_blocked_services."""

    def test_detects_shell_command(self):
        data = {
            "alias": "Evil Automation",
            "action": [{"service": "shell_command.run_script"}],
        }
        result = VoiceAutomationAIConversationAgent._check_yaml_for_blocked_services(data)
        assert result is not None
        assert "shell_command" in result

    def test_detects_python_script(self):
        data = {"action": [{"service": "python_script.my_script"}]}
        result = VoiceAutomationAIConversationAgent._check_yaml_for_blocked_services(data)
        assert result is not None
        assert "python_script" in result

    def test_detects_homeassistant_domain(self):
        data = {"action": [{"service": "homeassistant.restart"}]}
        result = VoiceAutomationAIConversationAgent._check_yaml_for_blocked_services(data)
        assert result is not None

    def test_detects_hassio_domain(self):
        data = {"action": [{"service": "hassio.addon_start"}]}
        result = VoiceAutomationAIConversationAgent._check_yaml_for_blocked_services(data)
        assert result is not None

    def test_allows_safe_automation(self):
        data = {
            "alias": "Turn on lights",
            "action": [{"service": "light.turn_on", "entity_id": "light.living_room"}],
        }
        result = VoiceAutomationAIConversationAgent._check_yaml_for_blocked_services(data)
        assert result is None

    def test_all_blocked_domains_detected(self):
        """Every domain in BLOCKED_SERVICE_DOMAINS must be caught."""
        for domain in BLOCKED_SERVICE_DOMAINS:
            data = {"action": [{"service": f"{domain}.some_action"}]}
            result = VoiceAutomationAIConversationAgent._check_yaml_for_blocked_services(data)
            assert result is not None, f"Blocked domain '{domain}' was not detected"

    def test_detects_nested_blocked_domain(self):
        data = {
            "action": [
                {
                    "choose": [
                        {
                            "conditions": [],
                            "sequence": [{"service": "shell_command.malicious"}],
                        }
                    ]
                }
            ],
        }
        result = VoiceAutomationAIConversationAgent._check_yaml_for_blocked_services(data)
        assert result is not None


# ── Tool execution: CRUD operations ──


class TestToolExecutionCRUD:
    """Test tool dispatch for automation/script/scene CRUD."""

    async def test_list_automations(self, agent, file_manager_mock):
        file_manager_mock.read_automations = AsyncMock(
            return_value=[
                {"id": "1", "alias": "Morning Lights"},
                {"id": "2", "alias": "Night Mode"},
            ]
        )
        result = await agent._execute_tool("list_automations", {})
        assert result["success"] is True
        assert result["count"] == 2

    async def test_create_automation_safe(self, agent, file_manager_mock):
        file_manager_mock.add_automation = AsyncMock(return_value="12345")
        yaml_content = yaml.dump(
            {
                "alias": "Test",
                "trigger": [{"platform": "state", "entity_id": "light.test"}],
                "action": [{"service": "light.turn_on"}],
            }
        )
        result = await agent._execute_tool(
            "create_automation", {"yaml_content": yaml_content}
        )
        assert result["success"] is True
        assert "automation_id" in result

    async def test_create_automation_blocked(self, agent, file_manager_mock):
        file_manager_mock.add_automation = AsyncMock()
        yaml_content = yaml.dump(
            {"alias": "Evil", "action": [{"service": "shell_command.evil"}]}
        )
        result = await agent._execute_tool(
            "create_automation", {"yaml_content": yaml_content}
        )
        assert result["success"] is False
        assert "Blocked" in result["error"]
        file_manager_mock.add_automation.assert_not_called()

    async def test_create_script_blocked(self, agent, file_manager_mock):
        file_manager_mock.add_script = AsyncMock()
        yaml_content = yaml.dump(
            {"alias": "Evil Script", "sequence": [{"service": "python_script.hack"}]}
        )
        result = await agent._execute_tool(
            "create_script",
            {"script_name": "evil_script", "yaml_content": yaml_content},
        )
        assert result["success"] is False
        file_manager_mock.add_script.assert_not_called()

    async def test_edit_automation_blocked(self, agent, file_manager_mock):
        file_manager_mock.update_automation = AsyncMock()
        yaml_content = yaml.dump(
            {"alias": "Evil", "action": [{"service": "homeassistant.restart"}]}
        )
        result = await agent._execute_tool(
            "edit_automation",
            {"automation_id": "123", "yaml_content": yaml_content},
        )
        assert result["success"] is False
        file_manager_mock.update_automation.assert_not_called()

    async def test_edit_script_blocked(self, agent, file_manager_mock):
        file_manager_mock.update_script = AsyncMock()
        yaml_content = yaml.dump(
            {"sequence": [{"service": "hassio.addon_stop"}]}
        )
        result = await agent._execute_tool(
            "edit_script",
            {"script_name": "test", "yaml_content": yaml_content},
        )
        assert result["success"] is False
        file_manager_mock.update_script.assert_not_called()

    async def test_delete_automation(self, agent, file_manager_mock):
        file_manager_mock.delete_automation = AsyncMock()
        result = await agent._execute_tool(
            "delete_automation", {"automation_id": "123"}
        )
        assert result["success"] is True
        file_manager_mock.delete_automation.assert_awaited_once_with("123")

    async def test_list_scripts(self, agent, file_manager_mock):
        file_manager_mock.read_scripts = AsyncMock(
            return_value={"flash_lights": {"alias": "Flash Lights"}}
        )
        result = await agent._execute_tool("list_scripts", {})
        assert result["success"] is True
        assert result["count"] == 1

    async def test_list_scenes(self, agent, file_manager_mock):
        file_manager_mock.read_scenes = AsyncMock(
            return_value=[{"id": "1", "name": "Movie Night"}]
        )
        result = await agent._execute_tool("list_scenes", {})
        assert result["success"] is True
        assert result["count"] == 1

    async def test_unknown_tool(self, agent):
        result = await agent._execute_tool("nonexistent_tool", {})
        assert result["success"] is False
        assert "Unknown tool" in result["error"]

    async def test_invalid_yaml(self, agent):
        result = await agent._execute_tool(
            "create_automation", {"yaml_content": "{{invalid yaml: [["}
        )
        assert result["success"] is False
        assert "Invalid YAML" in result["error"]


# ── Conversation history management ──


class TestConversationHistory:
    """Test history tracking, trimming, and LRU eviction."""

    async def test_new_conversation_gets_uuid(self, agent):
        """When no conversation_id is provided, a new UUID is generated."""
        with patch.object(agent, "_call_with_tools", return_value="Hello!"):
            user_input = MagicMock()
            user_input.text = "Hello"
            user_input.conversation_id = None

            result = await agent.async_process(user_input)
            assert result.conversation_id is not None
            # Should be a valid UUID
            uuid.UUID(result.conversation_id)

    async def test_history_trimmed_to_max_turns(self, agent):
        """History should be trimmed when it exceeds max_history_turns * 2."""
        agent.hass.data[DOMAIN][agent._config_entry.entry_id]["max_history_turns"] = 2

        with patch.object(agent, "_call_with_tools", return_value="Response"):
            conv_id = "test-conv"
            for i in range(5):
                user_input = MagicMock()
                user_input.text = f"Message {i}"
                user_input.conversation_id = conv_id
                await agent.async_process(user_input)

            history = agent._conversations[conv_id]
            # max_history_turns=2, so max messages = 4
            assert len(history) <= 4

    async def test_lru_eviction(self, agent):
        """Oldest conversations should be evicted when MAX_CONVERSATIONS is exceeded."""
        with patch.object(agent, "_call_with_tools", return_value="Ok"):
            for i in range(MAX_CONVERSATIONS + 5):
                user_input = MagicMock()
                user_input.text = f"Hi {i}"
                user_input.conversation_id = f"conv-{i}"
                await agent.async_process(user_input)

            assert len(agent._conversations) == MAX_CONVERSATIONS
            assert "conv-0" not in agent._conversations
            assert "conv-4" not in agent._conversations
            assert f"conv-{MAX_CONVERSATIONS + 4}" in agent._conversations


# ── Tool-use loop ──


class TestCallWithTools:
    """Test the tool-use loop logic."""

    async def test_no_tool_calls_returns_text(self, agent):
        from custom_components.voice_automation_ai.llm_client import LLMResponse

        mock_client = MagicMock()
        mock_client.create_message.return_value = LLMResponse(text="Just text")

        result = await agent._call_with_tools(
            mock_client, "model", "system", [{"role": "user", "content": "hi"}], 100
        )
        assert result == "Just text"

    async def test_tool_call_then_text(self, agent, file_manager_mock):
        from custom_components.voice_automation_ai.llm_client import LLMResponse

        file_manager_mock.read_automations = AsyncMock(return_value=[])

        responses = [
            LLMResponse(
                tool_calls=[{"id": "tc1", "name": "list_automations", "arguments": {}}],
                raw_assistant_message=[
                    {"type": "tool_use", "id": "tc1", "name": "list_automations", "input": {}}
                ],
            ),
            LLMResponse(text="You have no automations."),
        ]

        mock_client = MagicMock()
        mock_client.create_message.side_effect = responses
        mock_client.add_tool_results = MagicMock()

        result = await agent._call_with_tools(
            mock_client, "model", "system", [{"role": "user", "content": "list"}], 100
        )
        assert result == "You have no automations."
        assert mock_client.create_message.call_count == 2

    async def test_max_rounds_enforced(self, agent, file_manager_mock):
        from custom_components.voice_automation_ai.llm_client import LLMResponse

        file_manager_mock.read_automations = AsyncMock(return_value=[])

        tool_response = LLMResponse(
            tool_calls=[{"id": "tc1", "name": "list_automations", "arguments": {}}],
            raw_assistant_message=[
                {"type": "tool_use", "id": "tc1", "name": "list_automations", "input": {}}
            ],
        )

        mock_client = MagicMock()
        mock_client.create_message.return_value = tool_response
        mock_client.add_tool_results = MagicMock()

        result = await agent._call_with_tools(
            mock_client, "model", "system", [{"role": "user", "content": "list"}], 100
        )
        assert result == "I completed the requested operations."
        assert mock_client.create_message.call_count == MAX_TOOL_ROUNDS

    async def test_none_text_returns_default(self, agent):
        from custom_components.voice_automation_ai.llm_client import LLMResponse

        mock_client = MagicMock()
        mock_client.create_message.return_value = LLMResponse(text=None)

        result = await agent._call_with_tools(
            mock_client, "model", "system", [{"role": "user", "content": "hi"}], 100
        )
        assert result == "Done."


# ── Regression: tool exchanges must not pollute stored history ──


class TestHistoryExcludesToolExchanges:
    """Regression test for Anthropic API error 400.

    When tool_use/tool_result messages leak into stored conversation
    history, trimming can orphan tool_result blocks and cause:
        'unexpected tool_use_id found in tool_result blocks'
    """

    async def test_stored_history_contains_only_user_assistant_text(self, agent, file_manager_mock):
        """After a tool-using turn, stored history must only contain
        simple user/assistant text pairs - no tool exchanges."""
        from custom_components.voice_automation_ai.llm_client import LLMResponse

        file_manager_mock.read_automations = AsyncMock(return_value=[
            {"id": "1", "alias": "Morning Lights"},
        ])
        file_manager_mock.get_entities_context = MagicMock(return_value="light.test: on")

        # First response uses a tool, second response is the final text
        responses = [
            LLMResponse(
                tool_calls=[{"id": "tc1", "name": "list_automations", "arguments": {}}],
                raw_assistant_message=[
                    {"type": "tool_use", "id": "tc1", "name": "list_automations", "input": {}}
                ],
            ),
            LLMResponse(text="You have 1 automation: Morning Lights."),
        ]

        mock_client = MagicMock()
        mock_client.create_message.side_effect = responses
        mock_client.add_tool_results = MagicMock()

        with patch.object(agent, "_create_llm_client", return_value=mock_client):
            user_input = MagicMock()
            user_input.text = "list my automations"
            user_input.conversation_id = "test-conv-123"

            await agent.async_process(user_input)

        history = agent._conversations["test-conv-123"]

        # Should contain exactly 2 messages: user text + assistant text
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "list my automations"}
        assert history[1] == {"role": "assistant", "content": "You have 1 automation: Morning Lights."}

        # No tool_use, tool_result, or tool messages should be present
        for msg in history:
            assert msg["role"] in ("user", "assistant"), f"Unexpected role: {msg['role']}"
            assert isinstance(msg["content"], str), f"Content should be text, not: {type(msg['content'])}"

    async def test_multi_turn_with_tools_keeps_clean_history(self, agent, file_manager_mock):
        """Multiple conversation turns with tool usage should all store clean history."""
        from custom_components.voice_automation_ai.llm_client import LLMResponse

        file_manager_mock.read_automations = AsyncMock(return_value=[])
        file_manager_mock.read_scenes = AsyncMock(return_value=[
            {"id": "1", "name": "Movie Night"},
        ])
        file_manager_mock.get_entities_context = MagicMock(return_value="light.test: on")

        conv_id = "multi-turn-test"

        # Turn 1: tool call + text
        responses_turn1 = [
            LLMResponse(
                tool_calls=[{"id": "tc1", "name": "list_automations", "arguments": {}}],
                raw_assistant_message=[
                    {"type": "tool_use", "id": "tc1", "name": "list_automations", "input": {}}
                ],
            ),
            LLMResponse(text="No automations found."),
        ]

        mock_client1 = MagicMock()
        mock_client1.create_message.side_effect = responses_turn1
        mock_client1.add_tool_results = MagicMock()

        with patch.object(agent, "_create_llm_client", return_value=mock_client1):
            user_input = MagicMock()
            user_input.text = "list automations"
            user_input.conversation_id = conv_id
            await agent.async_process(user_input)

        # Turn 2: another tool call + text
        responses_turn2 = [
            LLMResponse(
                tool_calls=[{"id": "tc2", "name": "list_scenes", "arguments": {}}],
                raw_assistant_message=[
                    {"type": "tool_use", "id": "tc2", "name": "list_scenes", "input": {}}
                ],
            ),
            LLMResponse(text="You have 1 scene: Movie Night."),
        ]

        mock_client2 = MagicMock()
        mock_client2.create_message.side_effect = responses_turn2
        mock_client2.add_tool_results = MagicMock()

        with patch.object(agent, "_create_llm_client", return_value=mock_client2):
            user_input = MagicMock()
            user_input.text = "list scenes"
            user_input.conversation_id = conv_id
            await agent.async_process(user_input)

        history = agent._conversations[conv_id]

        # 2 turns * 2 messages each = 4 messages
        assert len(history) == 4
        assert all(msg["role"] in ("user", "assistant") for msg in history)
        assert all(isinstance(msg["content"], str) for msg in history)
        assert history[0]["content"] == "list automations"
        assert history[1]["content"] == "No automations found."
        assert history[2]["content"] == "list scenes"
        assert history[3]["content"] == "You have 1 scene: Movie Night."
