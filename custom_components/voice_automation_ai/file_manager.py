"""Centralized async-safe YAML file operations for HA config files."""
from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

# Pattern for valid HA entity IDs: domain.object_id (alphanumeric + underscores)
_ENTITY_ID_PATTERN = re.compile(r"^[a-z_]+\.[a-z0-9_]+$")

# Pattern for valid blueprint names: alphanumeric + underscores + hyphens
_BLUEPRINT_NAME_PATTERN = re.compile(r"^[a-z0-9_-]+$")

# Valid blueprint domains (prevents directory traversal via domain parameter)
_VALID_BLUEPRINT_DOMAINS = {"automation", "script"}


class HAConfigFileManager:
    """Manages read/write operations for automations.yaml, scripts.yaml, scenes.yaml, and blueprints."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the file manager."""
        self._hass = hass
        self._config_dir = Path(hass.config.config_dir)
        self._locks: dict[str, asyncio.Lock] = {
            "automations": asyncio.Lock(),
            "scripts": asyncio.Lock(),
            "scenes": asyncio.Lock(),
            "blueprints": asyncio.Lock(),
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
        """Write data to a YAML file atomically (blocking - run in executor).

        Writes to a temp file first, then renames to target path.
        """
        fd, tmp_path = tempfile.mkstemp(suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                yaml.dump(
                    data,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                )
            os.replace(tmp_path, path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

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

    # ── Blueprints (individual files in config/blueprints/<domain>/) ──

    @staticmethod
    def _validate_blueprint_name(name: str) -> str:
        """Validate and sanitize a blueprint name. Returns the clean name.

        Raises ValueError if the name is invalid or contains path traversal.
        """
        clean_name = name.lower().replace(" ", "_").replace("-", "_")
        if not _BLUEPRINT_NAME_PATTERN.match(clean_name):
            raise ValueError(
                f"Invalid blueprint name: '{name}'. "
                "Use lowercase letters, numbers, underscores, and hyphens."
            )
        if ".." in name or "/" in name or "\\" in name:
            raise ValueError(f"Invalid blueprint name: '{name}'. Path components not allowed.")
        return clean_name

    @staticmethod
    def _validate_blueprint_domain(domain: str) -> str:
        """Validate the blueprint domain against the allowlist.

        Raises ValueError if the domain is not allowed.
        """
        if domain not in _VALID_BLUEPRINT_DOMAINS:
            raise ValueError(
                f"Invalid blueprint domain: '{domain}'. "
                f"Allowed: {', '.join(sorted(_VALID_BLUEPRINT_DOMAINS))}"
            )
        return domain

    def _blueprint_dir(self, domain: str) -> Path:
        """Get the blueprint directory for a given domain, creating it if needed."""
        self._validate_blueprint_domain(domain)
        bp_dir = self._config_dir / "blueprints" / domain
        bp_dir.mkdir(parents=True, exist_ok=True)
        return bp_dir

    def _read_blueprint_metadata(self, path: Path) -> dict:
        """Read only the blueprint metadata from a file (blocking).

        Uses a custom YAML loader that handles !input tags by treating
        them as plain strings, so we can extract the blueprint: section
        without errors.
        """
        if not path.exists():
            return {}

        class _InputLoader(yaml.SafeLoader):
            pass

        def _input_constructor(loader, node):
            return f"!input {loader.construct_scalar(node)}"

        _InputLoader.add_constructor("!input", _input_constructor)

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.load(f, Loader=_InputLoader)
            if isinstance(data, dict) and "blueprint" in data:
                return data["blueprint"]
        except yaml.YAMLError:
            pass
        return {}

    def _read_raw_file(self, path: Path) -> str:
        """Read a file as raw text (blocking)."""
        if not path.exists():
            raise ValueError(f"File not found: {path.name}")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _write_raw_file(self, path: Path, content: str) -> None:
        """Write raw text to a file atomically (blocking).

        Writes to a temp file first, then renames to target path.
        This prevents file corruption from interrupted writes.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(tmp_path, path)
        except BaseException:
            # Clean up temp file on any failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _delete_file(self, path: Path) -> None:
        """Delete a file (blocking)."""
        if not path.exists():
            raise ValueError(f"File not found: {path.name}")
        path.unlink()

    def _list_blueprint_files(self, domain: str) -> list[Path]:
        """List all .yaml files in a blueprint directory (blocking)."""
        bp_dir = self._blueprint_dir(domain)
        return sorted(bp_dir.glob("*.yaml"))

    async def read_blueprints(self, domain: str = "automation") -> list[dict]:
        """List all blueprints for a domain. Returns summary with name and description."""
        self._validate_blueprint_domain(domain)
        async with self._locks["blueprints"]:
            files = await self._hass.async_add_executor_job(
                self._list_blueprint_files, domain
            )
            blueprints = []
            for path in files:
                metadata = await self._hass.async_add_executor_job(
                    self._read_blueprint_metadata, path
                )
                blueprints.append({
                    "name": path.stem,
                    "blueprint_name": metadata.get("name", path.stem),
                    "description": metadata.get("description", ""),
                    "domain": metadata.get("domain", domain),
                })
            return blueprints

    async def read_blueprint(self, domain: str, name: str) -> str:
        """Read a single blueprint file as raw YAML text."""
        clean_name = self._validate_blueprint_name(name)
        async with self._locks["blueprints"]:
            bp_dir = self._blueprint_dir(domain)
            path = bp_dir / f"{clean_name}.yaml"
            # Defense-in-depth: verify path is within expected directory
            path.resolve().relative_to(bp_dir.resolve())
            return await self._hass.async_add_executor_job(self._read_raw_file, path)

    async def add_blueprint(self, domain: str, name: str, yaml_content: str) -> str:
        """Create a new blueprint file. Returns the blueprint name."""
        clean_name = self._validate_blueprint_name(name)

        async with self._locks["blueprints"]:
            bp_dir = self._blueprint_dir(domain)
            path = bp_dir / f"{clean_name}.yaml"

            if path.exists():
                raise ValueError(f"Blueprint '{clean_name}' already exists in {domain}")

            await self._hass.async_add_executor_job(self._write_raw_file, path, yaml_content)
            await self._reload(domain)
            _LOGGER.info("Blueprint added: %s/%s", domain, clean_name)
            return clean_name

    async def update_blueprint(self, domain: str, name: str, yaml_content: str) -> None:
        """Update an existing blueprint file."""
        clean_name = self._validate_blueprint_name(name)
        async with self._locks["blueprints"]:
            bp_dir = self._blueprint_dir(domain)
            path = bp_dir / f"{clean_name}.yaml"
            path.resolve().relative_to(bp_dir.resolve())

            if not path.exists():
                raise ValueError(f"Blueprint '{clean_name}' not found in {domain}")

            await self._hass.async_add_executor_job(self._write_raw_file, path, yaml_content)
            await self._reload(domain)
            _LOGGER.info("Blueprint updated: %s/%s", domain, clean_name)

    async def delete_blueprint(self, domain: str, name: str) -> None:
        """Delete a blueprint file."""
        clean_name = self._validate_blueprint_name(name)
        async with self._locks["blueprints"]:
            bp_dir = self._blueprint_dir(domain)
            path = bp_dir / f"{clean_name}.yaml"
            path.resolve().relative_to(bp_dir.resolve())

            await self._hass.async_add_executor_job(self._delete_file, path)
            await self._reload(domain)
            _LOGGER.info("Blueprint deleted: %s/%s", domain, clean_name)

    # ── Helpers ──

    async def _reload(self, domain: str) -> None:
        """Reload a HA domain after file changes."""
        try:
            await self._hass.services.async_call(domain, "reload")
        except Exception:
            _LOGGER.warning("Failed to reload %s - a manual reload may be needed", domain)

    def get_entities_context(self) -> str:
        """Get a compact summary of available entities for LLM context.

        Prioritizes controllable domains and limits output to keep token
        usage reasonable (~2000 tokens instead of 10,000+).
        """
        states = self._hass.states.async_all()

        entities_by_domain: dict[str, list[str]] = {}
        for state in states:
            entity_id = state.entity_id
            # Sanitize: skip entity IDs that don't match the expected pattern
            # This prevents prompt injection via crafted entity names
            if not _ENTITY_ID_PATTERN.match(entity_id):
                _LOGGER.debug("Skipping malformed entity ID: %r", entity_id)
                continue
            domain = entity_id.split(".")[0]
            entities_by_domain.setdefault(domain, []).append(entity_id)

        # Controllable domains first (most useful for voice control)
        priority_domains = [
            "light", "switch", "climate", "cover", "lock", "fan",
            "media_player", "vacuum", "camera", "scene", "script",
        ]
        # Info domains (useful for status queries)
        info_domains = ["sensor", "binary_sensor", "person", "weather"]

        lines = []
        for domain in priority_domains:
            if domain in entities_by_domain:
                entities = entities_by_domain[domain]
                lines.append(f"{domain}: {', '.join(entities)}")

        for domain in info_domains:
            if domain in entities_by_domain:
                entities = entities_by_domain[domain]
                # Limit info domains to 20 entities to save tokens
                if len(entities) > 20:
                    lines.append(f"{domain}: {', '.join(entities[:20])} (and {len(entities) - 20} more)")
                else:
                    lines.append(f"{domain}: {', '.join(entities)}")

        return "\n".join(lines) if lines else "No entities available"
