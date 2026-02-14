"""Centralized async-safe YAML file operations for HA config files."""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

import yaml

from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


class HAConfigFileManager:
    """Manages read/write operations for automations.yaml, scripts.yaml, and scenes.yaml."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the file manager."""
        self._hass = hass
        self._config_dir = Path(hass.config.config_dir)
        self._locks: dict[str, asyncio.Lock] = {
            "automations": asyncio.Lock(),
            "scripts": asyncio.Lock(),
            "scenes": asyncio.Lock(),
        }

    def _get_path(self, filename: str) -> Path:
        """Get full path to a config file."""
        return self._config_dir / filename

    def _read_yaml(self, path: Path) -> Any:
        """Read and parse a YAML file (blocking - run in executor)."""
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _write_yaml(self, path: Path, data: Any) -> None:
        """Write data to a YAML file (blocking - run in executor)."""
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(
                data,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

    # ── Automations (list format with 'id' field) ──

    async def read_automations(self) -> list[dict]:
        """Read all automations from automations.yaml."""
        async with self._locks["automations"]:
            path = self._get_path("automations.yaml")
            data = await self._hass.async_add_executor_job(self._read_yaml, path)
            if data is None:
                return []
            if isinstance(data, list):
                return data
            return [data]

    async def add_automation(self, automation: dict) -> str:
        """Add a new automation. Returns the assigned ID."""
        async with self._locks["automations"]:
            path = self._get_path("automations.yaml")
            data = await self._hass.async_add_executor_job(self._read_yaml, path)
            automations = data if isinstance(data, list) else ([] if data is None else [data])

            if "id" not in automation:
                automation["id"] = str(int(time.time() * 1000))

            automations.append(automation)
            await self._hass.async_add_executor_job(self._write_yaml, path, automations)
            await self._reload("automation")

            _LOGGER.info("Automation added: %s", automation.get("id"))
            return automation["id"]

    async def update_automation(self, automation_id: str, automation: dict) -> None:
        """Update an existing automation by ID."""
        async with self._locks["automations"]:
            path = self._get_path("automations.yaml")
            data = await self._hass.async_add_executor_job(self._read_yaml, path)
            automations = data if isinstance(data, list) else ([] if data is None else [data])

            for i, existing in enumerate(automations):
                if str(existing.get("id")) == str(automation_id):
                    automation["id"] = str(automation_id)
                    automations[i] = automation
                    await self._hass.async_add_executor_job(self._write_yaml, path, automations)
                    await self._reload("automation")
                    _LOGGER.info("Automation updated: %s", automation_id)
                    return

            raise ValueError(f"Automation with id '{automation_id}' not found")

    async def delete_automation(self, automation_id: str) -> None:
        """Delete an automation by ID."""
        async with self._locks["automations"]:
            path = self._get_path("automations.yaml")
            data = await self._hass.async_add_executor_job(self._read_yaml, path)
            automations = data if isinstance(data, list) else ([] if data is None else [data])

            original_len = len(automations)
            automations = [a for a in automations if str(a.get("id")) != str(automation_id)]

            if len(automations) == original_len:
                raise ValueError(f"Automation with id '{automation_id}' not found")

            await self._hass.async_add_executor_job(self._write_yaml, path, automations)
            await self._reload("automation")
            _LOGGER.info("Automation deleted: %s", automation_id)

    # ── Scripts (dict format keyed by script name) ──

    async def read_scripts(self) -> dict[str, dict]:
        """Read all scripts from scripts.yaml."""
        async with self._locks["scripts"]:
            path = self._get_path("scripts.yaml")
            data = await self._hass.async_add_executor_job(self._read_yaml, path)
            if data is None:
                return {}
            if isinstance(data, dict):
                return data
            return {}

    async def add_script(self, name: str, script: dict) -> str:
        """Add a new script. Returns the script name."""
        async with self._locks["scripts"]:
            path = self._get_path("scripts.yaml")
            data = await self._hass.async_add_executor_job(self._read_yaml, path)
            scripts = data if isinstance(data, dict) else {}

            # Sanitize name: lowercase, underscores only
            clean_name = name.lower().replace("-", "_").replace(" ", "_")

            if clean_name in scripts:
                raise ValueError(f"Script '{clean_name}' already exists")

            scripts[clean_name] = script
            await self._hass.async_add_executor_job(self._write_yaml, path, scripts)
            await self._reload("script")
            _LOGGER.info("Script added: %s", clean_name)
            return clean_name

    async def update_script(self, name: str, script: dict) -> None:
        """Update an existing script by name."""
        async with self._locks["scripts"]:
            path = self._get_path("scripts.yaml")
            data = await self._hass.async_add_executor_job(self._read_yaml, path)
            scripts = data if isinstance(data, dict) else {}

            if name not in scripts:
                raise ValueError(f"Script '{name}' not found")

            scripts[name] = script
            await self._hass.async_add_executor_job(self._write_yaml, path, scripts)
            await self._reload("script")
            _LOGGER.info("Script updated: %s", name)

    async def delete_script(self, name: str) -> None:
        """Delete a script by name."""
        async with self._locks["scripts"]:
            path = self._get_path("scripts.yaml")
            data = await self._hass.async_add_executor_job(self._read_yaml, path)
            scripts = data if isinstance(data, dict) else {}

            if name not in scripts:
                raise ValueError(f"Script '{name}' not found")

            del scripts[name]
            await self._hass.async_add_executor_job(self._write_yaml, path, scripts)
            await self._reload("script")
            _LOGGER.info("Script deleted: %s", name)

    # ── Scenes (list format with 'id' field) ──

    async def read_scenes(self) -> list[dict]:
        """Read all scenes from scenes.yaml."""
        async with self._locks["scenes"]:
            path = self._get_path("scenes.yaml")
            data = await self._hass.async_add_executor_job(self._read_yaml, path)
            if data is None:
                return []
            if isinstance(data, list):
                return data
            return [data]

    async def add_scene(self, scene: dict) -> str:
        """Add a new scene. Returns the assigned ID."""
        async with self._locks["scenes"]:
            path = self._get_path("scenes.yaml")
            data = await self._hass.async_add_executor_job(self._read_yaml, path)
            scenes = data if isinstance(data, list) else ([] if data is None else [data])

            if "id" not in scene:
                scene["id"] = str(int(time.time() * 1000))

            scenes.append(scene)
            await self._hass.async_add_executor_job(self._write_yaml, path, scenes)
            await self._reload("scene")
            _LOGGER.info("Scene added: %s", scene.get("id"))
            return scene["id"]

    async def update_scene(self, scene_id: str, scene: dict) -> None:
        """Update an existing scene by ID."""
        async with self._locks["scenes"]:
            path = self._get_path("scenes.yaml")
            data = await self._hass.async_add_executor_job(self._read_yaml, path)
            scenes = data if isinstance(data, list) else ([] if data is None else [data])

            for i, existing in enumerate(scenes):
                if str(existing.get("id")) == str(scene_id):
                    scene["id"] = str(scene_id)
                    scenes[i] = scene
                    await self._hass.async_add_executor_job(self._write_yaml, path, scenes)
                    await self._reload("scene")
                    _LOGGER.info("Scene updated: %s", scene_id)
                    return

            raise ValueError(f"Scene with id '{scene_id}' not found")

    async def delete_scene(self, scene_id: str) -> None:
        """Delete a scene by ID."""
        async with self._locks["scenes"]:
            path = self._get_path("scenes.yaml")
            data = await self._hass.async_add_executor_job(self._read_yaml, path)
            scenes = data if isinstance(data, list) else ([] if data is None else [data])

            original_len = len(scenes)
            scenes = [s for s in scenes if str(s.get("id")) != str(scene_id)]

            if len(scenes) == original_len:
                raise ValueError(f"Scene with id '{scene_id}' not found")

            await self._hass.async_add_executor_job(self._write_yaml, path, scenes)
            await self._reload("scene")
            _LOGGER.info("Scene deleted: %s", scene_id)

    # ── Helpers ──

    async def _reload(self, domain: str) -> None:
        """Reload a HA domain after file changes."""
        try:
            await self._hass.services.async_call(domain, "reload")
        except Exception:
            _LOGGER.warning("Failed to reload %s - a manual reload may be needed", domain)

    def get_entities_context(self) -> str:
        """Get a summary of all available entities for Claude context."""
        states = self._hass.states.async_all()

        entities_by_domain: dict[str, list[str]] = {}
        for state in states:
            domain = state.entity_id.split(".")[0]
            entities_by_domain.setdefault(domain, []).append(state.entity_id)

        lines = []
        for domain in sorted(entities_by_domain):
            entities = entities_by_domain[domain]
            lines.append(f"{domain}: {', '.join(entities)}")

        return "\n".join(lines) if lines else "No entities available"
