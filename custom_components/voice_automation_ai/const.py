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
DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_OLLAMA_MODEL = "llama3.1"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_LANGUAGE = "en"

# Available Anthropic models
ANTHROPIC_MODELS = {
    "claude-sonnet-4-5-20250929": "Claude Sonnet 4.5 (Recommended)",
    "claude-opus-4-1-20250805": "Claude Opus 4.1 (Most powerful)",
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5 (Fastest & cheapest)",
    "claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet (Legacy)",
    "claude-3-opus-20240229": "Claude 3 Opus (Legacy)",
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
OLLAMA_TIMEOUT = 120
MAX_TOKENS = 4096
