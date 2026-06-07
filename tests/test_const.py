"""Tests for security constants and configuration."""
from custom_components.voice_automation_ai.const import (
    ALLOWED_SERVICE_DOMAINS,
    ANTHROPIC_MODELS,
    BLOCKED_SERVICE_DOMAINS,
    DEFAULT_MODEL,
    DOMAIN,
    PROVIDERS,
    PROVIDER_ANTHROPIC,
    PROVIDER_OLLAMA,
    SENSITIVE_ATTRIBUTE_KEYS,
    SENSITIVE_SERVICE_DOMAINS,
    TARGET_BROADENING_KEYS,
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

    def test_sensitive_domains_are_subset_of_allowed(self):
        """Gated sensitive domains must still be allowlisted (so they work
        when the option is enabled)."""
        assert SENSITIVE_SERVICE_DOMAINS <= ALLOWED_SERVICE_DOMAINS

    def test_sensitive_domains_not_blocked(self):
        """Sensitive domains are gated, not hard-blocked."""
        assert SENSITIVE_SERVICE_DOMAINS.isdisjoint(BLOCKED_SERVICE_DOMAINS)

    def test_target_broadening_keys_present(self):
        for key in ("area_id", "device_id", "label_id", "floor_id", "target"):
            assert key in TARGET_BROADENING_KEYS


class TestAnthropicModelCatalog:
    """Guard the Anthropic model dropdown against retired/invalid IDs."""

    # IDs the Anthropic API now returns 404 for - must never reappear.
    RETIRED_MODEL_IDS = {
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
    }

    def test_no_retired_models_offered(self):
        offered = set(ANTHROPIC_MODELS)
        leaked = offered & self.RETIRED_MODEL_IDS
        assert leaked == set(), f"Retired model IDs still offered: {leaked}"

    def test_current_generation_present(self):
        for model_id in ("claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5"):
            assert model_id in ANTHROPIC_MODELS

    def test_default_model_is_offered(self):
        assert DEFAULT_MODEL in ANTHROPIC_MODELS

    def test_catalog_is_nonempty(self):
        assert len(ANTHROPIC_MODELS) > 0
