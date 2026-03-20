# Voice Automation AI for Home Assistant

[![hacs_badge](https://img.shields.io/badge/HACS-Custom-41BDF5.svg)](https://github.com/hacs/integration)
[![GitHub release](https://img.shields.io/github/release/jjtortosa/homeassistant-voice-automation-ai.svg)](https://GitHub.com/jjtortosa/homeassistant-voice-automation-ai/releases/)
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

Control your smart home with natural language using AI. Supports **Anthropic Claude** (cloud) and **Ollama** (local) as LLM providers. Create and manage automations, scripts, scenes, and blueprints through voice commands or service calls.

## Features

- **Voice-controlled home management** - Describe what you want in natural language
- **Dual LLM provider support** - Anthropic Claude (cloud) or Ollama (local/self-hosted)
- **19 built-in tools** - The AI can control devices, query states, and manage configurations
- **Blueprint support** - Create, read, edit, and delete reusable blueprint templates
- **Full CRUD for automations, scripts, and scenes** - Create, edit, delete, and list
- **Device control** - Turn on/off lights, lock doors, set temperatures, and more
- **YAML validation** - Ensures generated configurations are valid before applying
- **Multilingual** - Supports English, Catalan, Spanish, French, and German
- **Context-aware** - Uses your existing Home Assistant entities for accurate responses
- **Security built-in** - Blocked service domains, sensitive attribute stripping, entity ID validation

## Installation

### HACS (Recommended)

1. Open HACS in your Home Assistant instance
2. Click on "Integrations"
3. Click the three dots in the top right corner
4. Select "Custom repositories"
5. Add `https://github.com/jjtortosa/homeassistant-voice-automation-ai` as an Integration
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

## Available Tools (19)

The AI conversation agent has access to 19 tools:

### Device Control
| Tool | Description |
|------|-------------|
| `call_service` | Call any HA service (light, switch, climate, etc.) |
| `get_entity_state` | Get current state and attributes of an entity |

### Automations
| Tool | Description |
|------|-------------|
| `list_automations` | List all automations |
| `create_automation` | Create a new automation from YAML |
| `edit_automation` | Edit an existing automation by ID |
| `delete_automation` | Delete an automation by ID |

### Scripts
| Tool | Description |
|------|-------------|
| `list_scripts` | List all scripts |
| `create_script` | Create a new script |
| `edit_script` | Edit an existing script by name |
| `delete_script` | Delete a script by name |

### Scenes
| Tool | Description |
|------|-------------|
| `list_scenes` | List all scenes |
| `create_scene` | Create a new scene |
| `edit_scene` | Edit an existing scene by ID |
| `delete_scene` | Delete a scene by ID |

### Blueprints
| Tool | Description |
|------|-------------|
| `list_blueprints` | List all blueprints for a domain |
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

## Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| Provider | Anthropic Claude or Ollama | Anthropic |
| API Key | Anthropic API key (Claude only) | Required for Claude |
| Ollama Host | URL of Ollama instance | `http://localhost:11434` |
| Model | LLM model to use | Claude Sonnet 4.5 / llama3.1 |
| Language | Language for AI responses | Auto-detected from HA |
| Max Tokens | Maximum response tokens (256-32768) | 4096 |
| History Turns | Conversation turns to remember (1-50) | 10 |
| Temperature | Randomness control (Ollama only, 0.0-2.0) | Model default |
| Top P | Nucleus sampling (Ollama only, 0.0-1.0) | Model default |

### Available Models

**Anthropic Claude:**
- **Claude Sonnet 4.5** (Recommended) - Best balance of speed and intelligence
- **Claude Opus 4.1** - Most powerful
- **Claude Haiku 4.5** - Fastest and most economical

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

- [Report bugs](https://github.com/jjtortosa/homeassistant-voice-automation-ai/issues)
- [Request features](https://github.com/jjtortosa/homeassistant-voice-automation-ai/issues)
- [Community discussion](https://community.home-assistant.io/)

---

**Note:** This is a custom integration and is not affiliated with or endorsed by Home Assistant, Anthropic, or Ollama.
