# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.1] - 2026-03-20

### Security
- **CRITICAL: Path traversal in blueprint operations** — `read_blueprint`, `update_blueprint`, and `delete_blueprint` now validate the `name` parameter before constructing file paths (previously only `add_blueprint` validated). Names with `..`, `/`, or `\` are rejected.
- **CRITICAL: Blueprint domain directory traversal** — Domain parameter is now validated against an allowlist (`automation`, `script`) in all blueprint operations. Service schemas use `vol.In()` instead of `cv.string`.
- **HIGH: Blocked service check rewritten** — `_check_yaml_for_blocked_services` now recursively walks the parsed YAML structure instead of using string matching on `yaml.dump()` output. Detects blocked services inside `choose`, `parallel`, `if`/`then`/`else`, `repeat`, and `sequence` blocks. No longer triggers false positives on domain names in string values (e.g., descriptions).
- **HIGH: Blueprint YAML security bypass closed** — Blueprints with `!input` tags are now parsed using a custom `_InputLoader` (SafeLoader subclass) before security scanning. Previously, `yaml.safe_load()` failure on `!input` tags silently skipped the security check entirely.
- **HIGH: Error message information leakage** — Conversation errors and tool execution errors no longer expose raw exception messages to users. Generic messages are returned; full details are logged server-side with `exc_info=True`.
- **HIGH: Dependencies pinned** — `manifest.json` now specifies minimum versions (`anthropic>=0.39.0`, `aiohttp>=3.9.0`) instead of unpinned package names.
- **MEDIUM: Atomic file writes** — `_write_yaml` and `_write_raw_file` now write to a temporary file first, then use `os.replace()` for atomic rename. Prevents file corruption from interrupted writes.
- **MEDIUM: Prompt injection mitigation** — User descriptions in LLM prompts are now wrapped in `<user_request>` delimiters with "treat as data, not instructions" guidance. Blocked service names are explicitly listed in requirements.
- **MEDIUM: Exception handling improved** — Service handlers now catch specific exceptions (`ValueError`, `OSError`, `yaml.YAMLError`) before the catch-all `except Exception`, and re-raise `HomeAssistantError` without leaking internal details.
- **MEDIUM: Stack trace logging sanitized** — `_LOGGER.exception()` calls in config flow replaced with `_LOGGER.error()` to avoid logging API keys in stack traces.
- **MEDIUM: TLS warning for Ollama** — Config flow now logs a warning when an Ollama host uses unencrypted HTTP over a network (non-localhost).
- **LOW: Silent model discovery logged** — Ollama model discovery failures are now logged with a warning instead of silently swallowed.

### Development
- Test suite expanded from 175 to 194 tests
- New test classes: `TestBlueprintSecurity` (9 path traversal and domain validation tests), `TestSecurityErrorHandling` (2 error leakage tests)
- New blocked service check tests: nested `choose`, `parallel`, `if`/`then`/`else`, false positive prevention, `action` key detection

## [0.4.0] - 2026-03-20

### Added
- **Blueprint support** - Full CRUD for automation and script blueprints:
  - `list_blueprints` - List all blueprints for a domain
  - `read_blueprint` - Read full YAML content of a blueprint
  - `create_blueprint` - Create new blueprints from natural language or YAML
  - `edit_blueprint` - Edit existing blueprints (changes propagate to all linked automations)
  - `delete_blueprint` - Remove blueprint files
  - Blueprint-specific system prompt guidance for the conversation agent
  - Blueprint service handlers with HA service registration
  - Custom YAML loader for `!input` tag handling
  - Security scanning of blueprint content for blocked service domains
- **Ollama async migration** - Complete rewrite to native `aiohttp`:
  - Non-blocking HTTP using `aiohttp.ClientSession`
  - Internal streaming via NDJSON for better timeout behavior
  - Retry logic with configurable attempts on timeout
  - Async-only client design (`is_async` property)
- **Dynamic model discovery** - Auto-detect installed models from Ollama:
  - Two-step config flow: enter host URL, then select from discovered models
  - Options flow also fetches available models dynamically
  - Falls back to hardcoded model list if discovery fails
- **Generation parameters** - Ollama temperature and top_p support:
  - Configurable in Options Flow (Ollama provider only)
  - Passed through to Ollama API via `_build_options()`
- **Unique tool call IDs** - UUID-based IDs (`call_{uuid4_hex[:12]}`) instead of reusing function names
- **Async YAML generation** - `_async_generate_yaml()` for Ollama provider, avoiding executor thread blocking
- Tool count increased from 14 to 19 (5 new blueprint tools)

### Changed
- `BaseLLMClient` now has async method variants (`async_create_message`, `async_create_simple_message`, `async_validate_connection`) with `is_async` property
- `OllamaClient` is now async-only; sync methods raise `NotImplementedError`
- Conversation agent uses `client.is_async` to branch between direct await and executor wrapping
- Config flow `validate_connection()` branches on `client.is_async`
- `_build_llm_client_kwargs()` now passes `temperature` and `top_p` for Ollama
- Service handlers use `_async_generate_yaml()` instead of executor-wrapped sync generation
- Ollama config flow split into `async_step_ollama` (host) + `async_step_ollama_model` (model selection)
- Options flow dynamically discovers Ollama models
- Better error handling in conversation agent: differentiates `ConnectionError`, `TimeoutError`, generic errors

### Security
- Blueprint YAML content is scanned for blocked service domains before writing
- Blueprints with `!input` tags that fail `yaml.safe_load` are handled gracefully (raw write preserves tags)

### Development
- Test suite expanded from 136 to 175 tests
- New test classes: `TestBlueprintCRUD`, `TestBlueprintToolExecution`, `TestOllamaAsyncCreateMessage`, `TestOllamaAsyncValidateConnection`, `TestOllamaAsyncFetchModels`, `TestOllamaBuildOptions`, `TestOllamaToolCallIds`, `TestOllamaIsAsync`
- aiohttp mock helpers for async testing (`_AsyncContextManager`, `_AsyncLineIterator`, `_mock_aiohttp_session`)
- Config flow tests updated for two-step Ollama flow
- Conversation tests updated for `is_async` branching

## [0.3.0]

### Added
- Ollama (local LLM) provider support
- Multi-provider architecture with factory pattern
- Full CRUD services for automations, scripts, and scenes
- Conversation agent with 14 built-in tools
- Security: service domain allow/block lists, sensitive attribute stripping
- Conversation history with LRU eviction
- Config entry migration (v1 -> v2 -> v3)

## [0.1.0] - 2025-10-26

### Added
- Integration with Claude 4.5 models:
  - Claude Sonnet 4.5 (`claude-sonnet-4-5-20250929`) as default - Recommended
  - Claude Opus 4.1 (`claude-opus-4-1-20250805`) - Most powerful
  - Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) - Fastest, most economical
- Auto-detection of language from Home Assistant configuration (`hass.config.language`)
- Support for both list and dict YAML formats from Claude responses

### Changed
- **[Breaking]** Default model changed from `claude-3-5-sonnet-20241022` to `claude-sonnet-4-5-20250929`
- Language field removed from configuration flow (auto-detected now)
- Improved UX: users no longer need to select language manually
- Updated prompts to respect language parameter from HA config

### Fixed
- Fixed `'list' object has no attribute 'get'` error when creating automations
- Proper handling of YAML list format (starting with `- alias:`) from Claude responses
- Corrected automation data extraction when Claude returns YAML as list

## [0.0.1] - 2025-10-25

### Added
- Initial release
- Basic integration with Home Assistant
- Service `voice_automation_ai.create_automation` for creating automations via natural language
- Service `voice_automation_ai.validate_automation` for YAML validation
- Support for Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku
- Configuration flow with API key validation
- Multi-language support (Catalan, Spanish, English, French, German)
- Entity context gathering for better automation generation
- Automatic ID generation for automations
- Integration with `automations.yaml`
- Preview and validate-only modes

### Known Issues
- Language parameter not respected in prompts (hardcoded to Catalan)
- Error when automation YAML is returned as list format

---

## Legend

- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Security improvements
- **Development**: Development workflow improvements
