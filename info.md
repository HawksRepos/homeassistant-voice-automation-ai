# Voice Automation AI

> **⚠️ BETA - Early Development Version**
>
> This integration is functional but **not fully tested** yet. Testers welcome!

Create Home Assistant automations using **voice commands** powered by Claude AI! 🎤🤖

## What is this?

This integration lets you create complex automations just by describing what you want in plain language. No more YAML editing!

## Features

✅ **Voice-controlled** - Just speak or type what you want
✅ **Smart** - Powered by Claude 3.5 Sonnet AI
✅ **Multilingual** - Supports 5 languages
✅ **Preview mode** - See the YAML before creating
✅ **Context-aware** - Uses your existing entities

## Quick Example

**You say:** "Create an automation that turns on the lights when I arrive home"

**It creates:**
```yaml
- alias: "Lights on when arriving home"
  trigger:
    - platform: state
      entity_id: person.you
      to: home
  action:
    - service: light.turn_on
      target:
        area_id: all
```

## Installation

1. Click **Install** above
2. Restart Home Assistant
3. Go to **Settings** → **Integrations** → **Add Integration**
4. Search for **Voice Automation AI**
5. Enter your Anthropic API key (get one free at console.anthropic.com)

## Cost

Very affordable! About **$0.01 per automation** created.

## More Info

[Full Documentation](https://github.com/HawksRepos/homeassistant-voice-automation-ai)
