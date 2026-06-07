"""Constants for Voice Automation AI integration."""

# Integration domain
DOMAIN = "voice_automation_ai"

# Platforms
PLATFORMS = ["conversation"]

# Configuration
CONF_PROVIDER = "provider"
CONF_API_KEY = "api_key"
CONF_MODEL = "model"
CONF_LANGUAGE = "language"
CONF_OLLAMA_HOST = "ollama_host"

# Provider options
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_OLLAMA = "ollama"

PROVIDERS = {
    PROVIDER_ANTHROPIC: "Anthropic Claude (Cloud API)",
    PROVIDER_OLLAMA: "Ollama (Local LLM)",
}

# Defaults
DEFAULT_PROVIDER = PROVIDER_ANTHROPIC
DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_OLLAMA_MODEL = "llama3.1"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_LANGUAGE = "en"

# Options keys (runtime-tunable via Options Flow)
CONF_MAX_TOKENS = "max_tokens"
CONF_MAX_HISTORY_TURNS = "max_history_turns"
CONF_TEMPERATURE = "temperature"
CONF_TOP_P = "top_p"
CONF_ALLOW_SENSITIVE_ACTIONS = "allow_sensitive_actions"
CONF_ENABLE_MEMORY = "enable_memory"
CONF_MEMORY_RETENTION_DAYS = "memory_retention_days"

# Defaults for options
DEFAULT_MAX_TOKENS = 4096
DEFAULT_MAX_HISTORY_TURNS = 10
# Default True preserves prior behaviour on upgrade (locks/alarms were already
# controllable). Security-conscious users can disable it in the options flow.
DEFAULT_ALLOW_SENSITIVE_ACTIONS = True

# ── Long-term memory ──
DEFAULT_ENABLE_MEMORY = True
# Unpinned memories not reinforced within this many days are pruned on use.
DEFAULT_MEMORY_RETENTION_DAYS = 90
# Human-editable JSON file in the HA config directory.
MEMORY_FILENAME = "voice_automation_ai_memory.json"
# Hard caps to keep the injected context small and the file tidy.
MAX_MEMORY_ITEMS = 50
MAX_MEMORY_TEXT_LEN = 300
# Hard ceiling on the characters injected into the prompt per turn (keeps token
# use bounded and limits the blast radius of any single poisoned entry).
MEMORY_PROMPT_CHAR_BUDGET = 4000
# Suggested categories (free-form is allowed; these guide the model).
MEMORY_CATEGORIES = ("preference", "system", "improvement", "general")
# Substrings that cause a memory write to be refused (never persist secrets).
MEMORY_SECRET_KEYWORDS = (
    "password", "passwd", "secret", "token", "api key", "api_key",
    "apikey", "bearer ", "private key", "credential",
)

# Available Anthropic models.
# Keep current-generation models only; retired model IDs (e.g. claude-3-opus,
# claude-3-5-sonnet) are removed because the API returns 404 for them. A user's
# previously-saved model is re-added to the dropdown by the options flow, so
# dropping an older entry here never strands an existing install.
ANTHROPIC_MODELS = {
    "claude-opus-4-8": "Claude Opus 4.8 (Most capable)",
    "claude-sonnet-4-6": "Claude Sonnet 4.6 (Recommended)",
    "claude-haiku-4-5": "Claude Haiku 4.5 (Fastest & cheapest)",
    "claude-sonnet-4-5-20250929": "Claude Sonnet 4.5 (Legacy)",
}

# Common Ollama models (user can also type a custom name)
OLLAMA_MODELS = {
    "llama3.1": "Llama 3.1 (Recommended)",
    "llama3.1:70b": "Llama 3.1 70B (More powerful)",
    "qwen2.5": "Qwen 2.5 (Good tool use)",
    "mistral": "Mistral (Fast)",
    "deepseek-r1": "DeepSeek R1 (Reasoning)",
    "command-r": "Command R (Good for tools)",
}

# Back-compat alias
MODELS = ANTHROPIC_MODELS

# Supported languages
LANGUAGES = {
    "ca": "Català",
    "es": "Español",
    "en": "English",
    "fr": "Français",
    "de": "Deutsch",
}

# Services
SERVICE_CREATE_AUTOMATION = "create_automation"
SERVICE_VALIDATE_AUTOMATION = "validate_automation"
SERVICE_CREATE_SCRIPT = "create_script"
SERVICE_CREATE_SCENE = "create_scene"
SERVICE_EDIT_AUTOMATION = "edit_automation"
SERVICE_EDIT_SCRIPT = "edit_script"
SERVICE_EDIT_SCENE = "edit_scene"
SERVICE_DELETE_AUTOMATION = "delete_automation"
SERVICE_DELETE_SCRIPT = "delete_script"
SERVICE_DELETE_SCENE = "delete_scene"
SERVICE_LIST_AUTOMATIONS = "list_automations"
SERVICE_LIST_SCRIPTS = "list_scripts"
SERVICE_LIST_SCENES = "list_scenes"
SERVICE_CREATE_BLUEPRINT = "create_blueprint"
SERVICE_EDIT_BLUEPRINT = "edit_blueprint"
SERVICE_DELETE_BLUEPRINT = "delete_blueprint"
SERVICE_LIST_BLUEPRINTS = "list_blueprints"
SERVICE_ADD_MEMORY = "add_memory"
SERVICE_REMOVE_MEMORY = "remove_memory"
SERVICE_LIST_MEMORIES = "list_memories"
SERVICE_CLEAR_MEMORIES = "clear_memories"

# Attributes
ATTR_DESCRIPTION = "description"
ATTR_VALIDATE_ONLY = "validate_only"
ATTR_PREVIEW = "preview"
ATTR_AUTOMATION_ID = "automation_id"
ATTR_YAML_CONTENT = "yaml_content"
ATTR_SCRIPT_NAME = "script_name"
ATTR_SCENE_ID = "scene_id"
ATTR_BLUEPRINT_NAME = "blueprint_name"
ATTR_BLUEPRINT_DOMAIN = "blueprint_domain"
ATTR_TEXT = "text"
ATTR_CATEGORY = "category"
ATTR_PINNED = "pinned"
ATTR_QUERY = "query"
ATTR_CONFIRM = "confirm"

# API
API_TIMEOUT = 30
OLLAMA_TIMEOUT = 120
MAX_TOKENS = DEFAULT_MAX_TOKENS

# ── Security: service call restrictions ──

# Domains the LLM is allowed to call via call_service
ALLOWED_SERVICE_DOMAINS = {
    "light",
    "switch",
    "climate",
    "cover",
    "lock",
    "fan",
    "media_player",
    "vacuum",
    "scene",
    "script",
    "input_boolean",
    "input_number",
    "input_select",
    "input_text",
    "input_button",
    "input_datetime",
    "number",
    "select",
    "button",
    "siren",
    "water_heater",
    "humidifier",
    "remote",
    "alarm_control_panel",
}

# High-impact domains gated behind the "allow_sensitive_actions" option.
# These remain in ALLOWED_SERVICE_DOMAINS (so they work when enabled) but can be
# turned off so a voice command (or prompt-injected request) cannot unlock a door
# or disarm an alarm. Must be a subset of ALLOWED_SERVICE_DOMAINS.
SENSITIVE_SERVICE_DOMAINS = {
    "lock",
    "alarm_control_panel",
}

# Service-call data keys that broaden targeting beyond a single entity_id.
# Stripped from call_service payloads so the LLM cannot turn a single-entity
# request into an area/device/label-wide action.
TARGET_BROADENING_KEYS = {
    "area_id",
    "device_id",
    "label_id",
    "floor_id",
    "target",
}

# Domains that must NEVER be called (even if added to allowlist by mistake)
BLOCKED_SERVICE_DOMAINS = {
    "shell_command",
    "rest_command",
    "python_script",
    "homeassistant",
    "system_log",
    "hassio",
    "addon",
    "recorder",
    "logger",
}

# Sensitive entity attribute keys to strip from get_entity_state responses
SENSITIVE_ATTRIBUTE_KEYS = {
    "latitude",
    "longitude",
    "gps_accuracy",
    "token",
    "access_token",
    "api_key",
    "api_token",
    "password",
    "secret",
    "stream_source",
    "entity_picture",
    "ip_address",
    "mac_address",
}
