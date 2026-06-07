"""Tests for the shared security helpers."""
import yaml

from custom_components.voice_automation_ai.const import BLOCKED_SERVICE_DOMAINS
from custom_components.voice_automation_ai.security import (
    InputLoader,
    check_yaml_for_blocked_services,
)


class TestCheckYAMLForBlockedServices:
    """Test the shared blocked-service scanner."""

    def test_detects_shell_command(self):
        data = {"action": [{"service": "shell_command.run"}]}
        result = check_yaml_for_blocked_services(data)
        assert result is not None
        assert "shell_command" in result

    def test_action_key_form(self):
        """HA accepts 'action' as a synonym for 'service'."""
        data = {"action": [{"action": "python_script.go"}]}
        assert check_yaml_for_blocked_services(data) is not None

    def test_allows_safe(self):
        data = {"action": [{"service": "light.turn_on", "entity_id": "light.x"}]}
        assert check_yaml_for_blocked_services(data) is None

    def test_every_blocked_domain_detected(self):
        for domain in BLOCKED_SERVICE_DOMAINS:
            data = {"action": [{"service": f"{domain}.x"}]}
            assert check_yaml_for_blocked_services(data) is not None, domain

    def test_deeply_nested(self):
        data = {
            "action": [
                {"choose": [
                    {"conditions": [],
                     "sequence": [
                         {"parallel": [
                             {"if": [], "then": [{"service": "hassio.host_reboot"}]}
                         ]}
                     ]}
                ]}
            ]
        }
        assert check_yaml_for_blocked_services(data) is not None

    def test_domain_substring_in_description_not_flagged(self):
        """A blocked domain name appearing in free text must not false-positive."""
        data = {
            "alias": "Talk about shell_command and python_script",
            "description": "homeassistant.restart is dangerous",
            "action": [{"service": "light.turn_on"}],
        }
        assert check_yaml_for_blocked_services(data) is None

    def test_non_dict_input(self):
        assert check_yaml_for_blocked_services(None) is None
        assert check_yaml_for_blocked_services("just a string") is None
        assert check_yaml_for_blocked_services([]) is None


class TestInputLoader:
    """Test the !input-aware YAML loader."""

    def test_parses_input_tags(self):
        content = (
            "blueprint:\n"
            "  name: Test\n"
            "trigger:\n"
            "  platform: state\n"
            "  entity_id: !input motion_sensor\n"
        )
        data = yaml.load(content, Loader=InputLoader)
        assert data["trigger"]["entity_id"] == "!input motion_sensor"

    def test_blocked_service_in_blueprint_with_input(self):
        """A blueprint using !input must still be scannable for blocked services."""
        content = (
            "blueprint:\n"
            "  name: Evil\n"
            "  domain: automation\n"
            "trigger:\n"
            "  platform: state\n"
            "  entity_id: !input sensor\n"
            "action:\n"
            "  - service: shell_command.hack\n"
        )
        data = yaml.load(content, Loader=InputLoader)
        assert check_yaml_for_blocked_services(data) is not None
