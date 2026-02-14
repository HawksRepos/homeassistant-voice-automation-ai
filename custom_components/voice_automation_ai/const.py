"""Constants for Voice Automation AI integration."""

# Integration domain
DOMAIN = "voice_automation_ai"

# Platforms
PLATFORMS = ["conversation"]

# Configuration
CONF_API_KEY = "api_key"
CONF_MODEL = "model"
CONF_LANGUAGE = "language"

# Defaults
DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_LANGUAGE = "en"

# Available models
MODELS = {
    "claude-sonnet-4-5-20250929": "Claude Sonnet 4.5 (Recommended)",
    "claude-opus-4-1-20250805": "Claude Opus 4.1 (Most powerful)",
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5 (Fastest & cheapest)",
    "claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet (Legacy)",
    "claude-3-opus-20240229": "Claude 3 Opus (Legacy)",
}

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

# Attributes
ATTR_DESCRIPTION = "description"
ATTR_VALIDATE_ONLY = "validate_only"
ATTR_PREVIEW = "preview"
ATTR_AUTOMATION_ID = "automation_id"
ATTR_YAML_CONTENT = "yaml_content"
ATTR_SCRIPT_NAME = "script_name"
ATTR_SCENE_ID = "scene_id"

# API
API_TIMEOUT = 30
MAX_TOKENS = 4096
