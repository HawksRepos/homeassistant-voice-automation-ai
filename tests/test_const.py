"""Tests for security constants and configuration."""
from custom_components.voice_automation_ai.const import (
    ALLOWED_SERVICE_DOMAINS,
    BLOCKED_SERVICE_DOMAINS,
    DOMAIN,
    PROVIDERS,
    PROVIDER_ANTHROPIC,
    PROVIDER_OLLAMA,
    SENSITIVE_ATTRIBUTE_KEYS,
)


class TestSecuritySets:
    """Verify security set invariants."""

    def test_allowed_and_blocked_domains_do_not_overlap(self):
        overlap = ALLOWED_SERVICE_DOMAINS & BLOCKED_SERVICE_DOMAINS
        assert overlap == set(), (
            f"ALLOWED and BLOCKED domain sets overlap: {overlap}"
        )

    def test_blocked_domains_contains_dangerous_services(self):
        """Guard against accidental removal of critical blocks."""
        critical = {"shell_command", "python_script", "homeassistant", "hassio"}
        missing = critical - BLOCKED_SERVICE_DOMAINS
        assert missing == set(), f"Critical blocked domains removed: {missing}"

    def test_sensitive_attributes_contains_location_keys(self):
        location_keys = {"latitude", "longitude", "gps_accuracy"}
        missing = location_keys - SENSITIVE_ATTRIBUTE_KEYS
        assert missing == set(), f"Location-sensitive keys removed: {missing}"

    def test_sensitive_attributes_contains_credential_keys(self):
        credential_keys = {"token", "access_token", "api_key", "password", "secret"}
        missing = credential_keys - SENSITIVE_ATTRIBUTE_KEYS
        assert missing == set(), f"Credential-sensitive keys removed: {missing}"

    def test_blocked_domains_minimum_size(self):
        """Regression guard: ensure blocked set hasn't been hollowed out."""
        assert len(BLOCKED_SERVICE_DOMAINS) >= 9

    def test_allowed_domains_is_nonempty(self):
        assert len(ALLOWED_SERVICE_DOMAINS) > 0

    def test_providers_dict_matches_constants(self):
        assert PROVIDER_ANTHROPIC in PROVIDERS
        assert PROVIDER_OLLAMA in PROVIDERS

    def test_domain_is_set(self):
        assert DOMAIN == "voice_automation_ai"
