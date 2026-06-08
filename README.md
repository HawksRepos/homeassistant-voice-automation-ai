# Voice Automation AI for Home Assistant

[![hacs_badge](https://img.shields.io/badge/HACS-Custom-41BDF5.svg)](https://github.com/hacs/integration)
[![GitHub release](https://img.shields.io/github/release/HawksRepos/homeassistant-voice-automation-ai.svg)](https://GitHub.com/HawksRepos/homeassistant-voice-automation-ai/releases/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Development Status](https://img.shields.io/badge/status-beta-orange.svg)

> **BETA VERSION - NOT FULLY TESTED YET**
>
> This integration is in **early development** and has not been extensively tested in production environments.
>
> - Code is complete and functional
> - Limited real-world testing
> - Use at your own risk
> - Please report any issues you encounter
>
> **Testers welcome!** Your feedback will help make this integration stable and production-ready.

Control your smart home with natural language using AI. Supports **Anthropic Claude** (cloud), **Google Gemini** (cloud), and **Ollama** (local) as LLM providers. Create and manage automations, scripts, scenes, and blueprints through voice commands or service calls.

## Features

- **Voice-controlled home management** - Describe what you want in natural language
- **Multiple LLM providers** - Anthropic Claude (cloud), Google Gemini (cloud), or Ollama (local/self-hosted)
- **24 built-in tools** - The AI can control devices, query states, read existing configs, and manage configurations
- **Long-term memory** - A global, persistent memory of facts and preferences shared across every conversation, with automatic upkeep
- **Blueprint support** - Create, read, edit, and delete reusable blueprint templates
- **Full CRUD for automations, scripts, and scenes** - Create, read, edit, delete, and list
- **Device control** - Turn on/off lights, lock doors, set temperatures, and more
- **YAML validation** - Ensures generated configurations are valid before applying
- **Multilingual** - Supports English, Catalan, Spanish, French, and German
- **Context-aware** - Uses your existing Home Assistant entities for accurate responses
- **Security built-in** - Blocked service domains, sensitive attribute stripping, entity ID validation, optional gating of locks/alarms, and single-entity service targeting

## Installation

### HACS (Recommended)

1. Open HACS in your Home Assistant instance
2. Click on "Integrations"
3. Click the three dots in the top right corner
4. Select "Custom repositories"
5. Add `https://github.com/HawksRepos/homeassistant-voice-automation-ai` as an Integration
6. Click "Install" on the Voice Automation AI card
7. Restart Home Assistant

### Manual Installation

1. Copy the `custom_components/voice_automation_ai` folder to your `config/custom_components/` directory
2. Restart Home Assistant

## Configuration

### Anthropic Claude (Cloud)

1. Go to **Settings** > **Devices & Services**
2. Click **+ ADD INTEGRATION**
3. Search for **Voice Automation AI**
4. Select **Anthropic Claude** as the provider
5. Enter your **Anthropic API Key** (get one at https://console.anthropic.com/)
6. Select your preferred **Claude model**
7. Click **Submit**

### Google Gemini (Cloud)

1. Go to **Settings** > **Devices & Services**
2. Click **+ ADD INTEGRATION**
3. Search for **Voice Automation AI**
4. Select **Google Gemini** as the provider
5. Enter your **Gemini API Key** (get one at https://aistudio.google.com/)
6. Select your preferred **Gemini model**
7. Click **Submit**

> Gemini is a cloud provider (your prompts are sent to Google). For a fully-local, private setup, use Ollama.

### Ollama (Local)

1. Make sure [Ollama](https://ollama.ai/) is running on your network
2. Pull a model: `ollama pull llama3.1`
3. Go to **Settings** > **Devices & Services**
4. Click **+ ADD INTEGRATION**
5. Search for **Voice Automation AI**
6. Select **Ollama** as the provider
7. Enter your Ollama host URL (e.g., `http://192.168.1.100:11434`)
8. Select a model from the auto-discovered list
9. Click **Submit**

#### Ollama Features

- **Auto-discovery** - Models installed on your Ollama instance are automatically detected
- **Async HTTP** - Uses `aiohttp` for non-blocking communication
- **Streaming** - Internal streaming for better timeout handling on slow hardware
- **Generation parameters** - Configure temperature and top_p in the options flow
- **Retry logic** - Automatic retry on timeout with configurable attempts

## Usage

### Conversation Agent

1. Go to **Settings** > **Voice Assistants**
2. Select your assistant
3. Choose **Voice Automation AI** as the conversation agent
4. Talk to your assistant naturally:
   - "Turn on the living room lights"
   - "What's the temperature in the bedroom?"
   - "Create an automation that locks the door at 10 PM"
   - "List my blueprints"
   - "Create a blueprint for motion-activated lights"
   - "Hoover the kitchen" (robot vacuums; room cleaning needs Home Assistant 2026.3+)
   - "Remember that I prefer warm lighting at night"

### Via Service Call

You can also use the services directly in automations or scripts:

```yaml
service: voice_automation_ai.create_automation
data:
  description: "Turn on the kitchen lights when motion is detected"
```

```yaml
service: voice_automation_ai.create_blueprint
data:
  description: "A blueprint for motion-activated lights with configurable entity, delay, and brightness"
  blueprint_name: "motion_lights"
  blueprint_domain: "automation"
```

## Available Tools (24)

The AI conversation agent has access to 24 tools:

### Device Control
| Tool | Description |
|------|-------------|
| `call_service` | Call any HA service (light, switch, climate, etc.) |
| `get_entity_state` | Get current state and attributes of an entity |

### Long-Term Memory
| Tool | Description |
|------|-------------|
| `remember` | Save a durable fact or preference shared across all conversations |
| `forget` | Remove matching entries from long-term memory |

### Automations
| Tool | Description |
|------|-------------|
| `list_automations` | List all automations |
| `read_automation` | Read the full configuration of an automation by ID |
| `create_automation` | Create a new automation from YAML |
| `edit_automation` | Edit an existing automation by ID |
| `delete_automation` | Delete an automation by ID |

### Scripts
| Tool | Description |
|------|-------------|
| `list_scripts` | List all scripts |
| `read_script` | Read the full configuration of a script by name |
| `create_script` | Create a new script |
| `edit_script` | Edit an existing script by name |
| `delete_script` | Delete a script by name |

### Scenes
| Tool | Description |
|------|-------------|
| `list_scenes` | List all scenes |
| `read_scene` | Read the full configuration of a scene by ID |
| `create_scene` | Create a new scene |
| `edit_scene` | Edit an existing scene by ID |
| `delete_scene` | Delete a scene by ID |

### Blueprints
| Tool | Description |
|------|-------------|
| `list_blueprints` | List all blueprints for a domain (including source subfolders) |
| `read_blueprint` | Read the full YAML of a blueprint |
| `create_blueprint` | Create a new blueprint file |
| `edit_blueprint` | Edit a blueprint (propagates to all linked automations) |
| `delete_blueprint` | Delete a blueprint file |

## Services

### Automation Services
- `voice_automation_ai.create_automation` - Generate from natural language description
- `voice_automation_ai.edit_automation` - Modify an existing automation
- `voice_automation_ai.delete_automation` - Remove an automation
- `voice_automation_ai.list_automations` - List all automations
- `voice_automation_ai.validate_automation` - Validate YAML syntax

### Script Services
- `voice_automation_ai.create_script` - Generate from natural language description
- `voice_automation_ai.edit_script` - Modify an existing script
- `voice_automation_ai.delete_script` - Remove a script
- `voice_automation_ai.list_scripts` - List all scripts

### Scene Services
- `voice_automation_ai.create_scene` - Generate from natural language description
- `voice_automation_ai.edit_scene` - Modify an existing scene
- `voice_automation_ai.delete_scene` - Remove a scene
- `voice_automation_ai.list_scenes` - List all scenes

### Blueprint Services
- `voice_automation_ai.create_blueprint` - Generate a blueprint from natural language
- `voice_automation_ai.edit_blueprint` - Modify a blueprint (changes propagate to all linked automations/scripts)
- `voice_automation_ai.delete_blueprint` - Remove a blueprint
- `voice_automation_ai.list_blueprints` - List all blueprints for a domain

### Memory Services
- `voice_automation_ai.add_memory` - Save a durable fact/preference (optionally pinned)
- `voice_automation_ai.remove_memory` - Remove memories matching a text query
- `voice_automation_ai.list_memories` - Log all stored memories
- `voice_automation_ai.clear_memories` - Remove all stored memories

## Long-Term Memory

The assistant keeps a single, global memory shared across every conversation, so it learns your home and preferences over time (e.g. "the main bedroom light is `light.main_bedroom_light`", "prefers warm light at night", improvement requests you make).

- **Stored** in `config/voice_automation_ai_memory.json` — human-readable and editable; hand-edits take effect on the next message.
- **Token-light** — a small, capped block is added to the prompt and sits in the cached prefix, so it costs almost nothing per turn.
- **Self-maintaining** — unpinned facts not reinforced within the retention window (default 90 days) are pruned automatically; the total is capped (50 items / a per-turn character budget); pinned facts never expire.
- **Safe** — the assistant refuses to store secrets (passwords, tokens, keys), and the injected block is treated as reference data, never as commands — a stored note can't authorize an unsafe action (the service allowlist and locks/alarms gating still apply regardless).
- **Manage it** via the assistant ("remember that…", "forget…") or the memory services above. Disable it entirely with the *Enable Long-Term Memory* option.
- **Resetting** — wiping everything requires confirmation: call `voice_automation_ai.clear_memories` with `confirm: true` (the assistant only has a *targeted* forget and will confirm before bulk changes). A clean reset is a handy way to clear out odd, stale context.

> **New devices show up immediately:** the list of controllable entities is rebuilt from live state on every message, so adding a device makes it available on your next request — caching never serves a stale device list.

## Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| Provider | Anthropic Claude, Google Gemini, or Ollama | Anthropic |
| API Key | Anthropic or Gemini API key (cloud providers) | Required for cloud |
| Ollama Host | URL of Ollama instance | `http://localhost:11434` |
| Model | LLM model to use | Claude Sonnet 4.6 / Gemini 2.5 Flash / llama3.1 |
| Language | Language for AI responses | Auto-detected from HA |
| Max Tokens | Maximum response tokens (256-32768) | 4096 |
| History Turns | Conversation turns to remember (1-50) | 10 |
| Allow Sensitive Actions | Permit voice control of locks/alarm panels | On |
| Enable Long-Term Memory | Persist facts/preferences across conversations | On |
| Memory Retention (days) | Auto-remove unpinned facts after N days (1-3650) | 90 |
| Temperature | Randomness control (Ollama only, 0.0-2.0) | Model default |
| Top P | Nucleus sampling (Ollama only, 0.0-1.0) | Model default |

### Available Models

**Anthropic Claude:**
- **Claude Opus 4.8** - Most capable
- **Claude Sonnet 4.6** (Recommended) - Best balance of speed and intelligence
- **Claude Haiku 4.5** - Fastest and most economical
- **Claude Sonnet 4.5** (Legacy)

**Google Gemini:**
- **Gemini 2.5 Flash** (Recommended) - Fast, capable, cost-effective
- **Gemini 2.5 Pro** - Most capable
- **Gemini 2.5 Flash-Lite** - Fastest and cheapest
- **Gemini 3.5 Flash** - Latest

**Ollama:**
- Any model installed on your Ollama instance is auto-discovered
- Recommended: llama3.1, qwen2.5, command-r (good tool use support)

## Security

This integration includes several security measures:

- **Service domain allow/block lists** - Only safe service domains can be called
- **Blocked domains** - `shell_command`, `python_script`, `homeassistant`, `hassio`, etc. are always blocked
- **Sensitive attribute stripping** - Location, credentials, and tokens are stripped from entity state responses
- **Entity ID validation** - Regex validation prevents prompt injection via entity names
- **Blueprint YAML scanning** - Generated blueprints are checked for blocked services before writing

## Troubleshooting

### "Cannot connect to API"
- Check your internet connection (for Anthropic)
- Verify your API key is correct
- For Ollama: ensure the service is running and accessible at the configured URL

### "Model not found" (Ollama)
- Pull the model first: `ollama pull llama3.1`
- Check that the Ollama host URL is correct

### Slow responses (Ollama)
- Try a smaller model (7B/8B parameters)
- Reduce the max tokens setting
- Ensure your hardware has sufficient resources

### Automation doesn't work as expected
- Check the generated automation in **Settings** > **Automations**
- Provide more specific descriptions
- Use `list_automations` to verify it was created

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Powered by [Anthropic Claude AI](https://www.anthropic.com/) and [Ollama](https://ollama.ai/)
- Inspired by the Home Assistant community

## Support

- [Report bugs](https://github.com/HawksRepos/homeassistant-voice-automation-ai/issues)
- [Request features](https://github.com/HawksRepos/homeassistant-voice-automation-ai/issues)
- [Community discussion](https://community.home-assistant.io/)

---

**Note:** This is a custom integration and is not affiliated with or endorsed by Home Assistant, Anthropic, or Ollama.
