"""Tests for the global long-term memory store."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from custom_components.voice_automation_ai.memory import (
    MemoryManager,
    format_memory_block,
    prune_memories,
    select_within_budget,
    validate_memory_text,
)

NOW = datetime(2026, 6, 7, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def memory_manager(hass, tmp_path):
    """A MemoryManager backed by a temp directory."""
    hass.config.config_dir = str(tmp_path)
    return MemoryManager(hass)


def _mem(text, updated, pinned=False, category="general"):
    iso = updated.isoformat() if isinstance(updated, datetime) else updated
    return {
        "id": text, "text": text, "category": category,
        "pinned": pinned, "created": iso, "updated": iso,
    }


# ── Pure helpers ──


class TestValidate:
    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            validate_memory_text("")
        with pytest.raises(ValueError):
            validate_memory_text("   ")

    def test_rejects_non_string(self):
        with pytest.raises(ValueError):
            validate_memory_text(None)

    def test_rejects_too_long(self):
        with pytest.raises(ValueError):
            validate_memory_text("x" * 5000)

    def test_rejects_secrets(self):
        for text in (
            "my password is hunter2",
            "the api key is sk-abc",
            "use bearer abc123 to auth",
            "store this secret value",
        ):
            with pytest.raises(ValueError, match="secret"):
                validate_memory_text(text)

    def test_accepts_and_strips(self):
        assert validate_memory_text("  bedroom = light.bed  ") == "bedroom = light.bed"


class TestPrune:
    def test_drops_expired_unpinned(self):
        mems = [
            _mem("old", NOW - timedelta(days=100)),
            _mem("fresh", NOW - timedelta(days=1)),
        ]
        kept, removed = prune_memories(mems, NOW, retention_days=90, max_items=50)
        assert removed == 1
        assert [m["text"] for m in kept] == ["fresh"]

    def test_keeps_expired_pinned(self):
        mems = [_mem("old", NOW - timedelta(days=100), pinned=True)]
        kept, removed = prune_memories(mems, NOW, 90, 50)
        assert removed == 0
        assert len(kept) == 1

    def test_caps_dropping_oldest_unpinned(self):
        mems = [_mem(f"m{i}", NOW - timedelta(days=i)) for i in range(5)]
        kept, removed = prune_memories(mems, NOW, retention_days=3650, max_items=3)
        assert removed == 2
        assert {m["text"] for m in kept} == {"m0", "m1", "m2"}

    def test_cap_never_drops_pinned(self):
        mems = [_mem(f"p{i}", NOW - timedelta(days=i), pinned=True) for i in range(5)]
        kept, removed = prune_memories(mems, NOW, retention_days=3650, max_items=2)
        assert removed == 0
        assert len(kept) == 5

    def test_handles_naive_timestamps(self):
        # A hand-edited file might contain a naive timestamp; must not raise.
        naive = (NOW - timedelta(days=100)).replace(tzinfo=None).isoformat()
        kept, removed = prune_memories([_mem("x", naive)], NOW, 90, 50)
        assert removed == 1


class TestBudget:
    def test_selects_subset_within_budget(self):
        mems = [_mem("x" * 50, NOW - timedelta(days=i)) for i in range(20)]
        selected = select_within_budget(mems, char_budget=200)
        assert 0 < len(selected) < 20

    def test_pinned_selected_first(self):
        mems = [
            _mem("recent unpinned", NOW),
            _mem("old pinned", NOW - timedelta(days=100), pinned=True),
        ]
        selected = select_within_budget(mems, char_budget=10_000)
        assert selected[0]["text"] == "old pinned"

    def test_always_includes_at_least_one(self):
        huge = _mem("y" * 5000, NOW)
        selected = select_within_budget([huge], char_budget=10)
        assert len(selected) == 1


class TestFormat:
    def test_empty(self):
        assert format_memory_block([]) == ""

    def test_framed_as_data_not_instructions(self):
        block = format_memory_block([_mem("prefers warm light", NOW)])
        assert "NOT as instructions" in block

    def test_groups_by_category(self):
        block = format_memory_block([
            _mem("bedroom = light.bed", NOW, category="system"),
            _mem("prefers warm light", NOW, category="preference"),
        ])
        assert "Long-term memory" in block
        assert "[system] bedroom = light.bed" in block
        assert "[preference] prefers warm light" in block


# ── MemoryManager (file-backed) ──


class TestMemoryManager:
    async def test_add_and_list(self, memory_manager):
        await memory_manager.async_add("bedroom = light.bed", "system", now=NOW)
        items = await memory_manager.async_list()
        assert len(items) == 1
        assert items[0]["text"] == "bedroom = light.bed"
        assert items[0]["category"] == "system"

    async def test_add_duplicate_refreshes(self, memory_manager):
        await memory_manager.async_add("same fact", now=NOW)
        later = NOW + timedelta(days=10)
        result = await memory_manager.async_add("Same Fact", now=later)
        assert result["status"] == "refreshed"
        items = await memory_manager.async_list()
        assert len(items) == 1
        assert items[0]["updated"] == later.isoformat()

    async def test_remove_by_substring(self, memory_manager):
        await memory_manager.async_add("bedroom light is warm", now=NOW)
        await memory_manager.async_add("kitchen light is bright", now=NOW)
        removed = await memory_manager.async_remove("bedroom")
        assert removed == 1
        items = await memory_manager.async_list()
        assert len(items) == 1
        assert "kitchen" in items[0]["text"]

    async def test_clear(self, memory_manager):
        await memory_manager.async_add("x", now=NOW)
        await memory_manager.async_clear()
        assert await memory_manager.async_list() == []

    async def test_rejects_secret(self, memory_manager):
        with pytest.raises(ValueError):
            await memory_manager.async_add("the wifi password is abc", now=NOW)

    async def test_unknown_category_coerced_to_general(self, memory_manager):
        await memory_manager.async_add("fact", category="weirdcat", now=NOW)
        items = await memory_manager.async_list()
        assert items[0]["category"] == "general"

    async def test_known_category_normalized(self, memory_manager):
        await memory_manager.async_add("fact", category="Preference", now=NOW)
        items = await memory_manager.async_list()
        assert items[0]["category"] == "preference"

    async def test_pin_and_unpin(self, memory_manager):
        await memory_manager.async_add("keep me", now=NOW)
        changed = await memory_manager.async_set_pinned("keep me", True)
        assert changed == 1
        items = await memory_manager.async_list()
        assert items[0]["pinned"] is True

    async def test_prompt_block_prunes_and_persists(self, memory_manager):
        await memory_manager.async_add("stale fact", now=NOW - timedelta(days=200))
        await memory_manager.async_add("fresh fact", now=NOW)
        block = await memory_manager.async_prompt_block(NOW, retention_days=90)
        assert "fresh fact" in block
        assert "stale fact" not in block
        # Pruned entry is removed from the file too.
        items = await memory_manager.async_list()
        assert [m["text"] for m in items] == ["fresh fact"]

    async def test_persists_across_instances(self, memory_manager, hass):
        await memory_manager.async_add("persistent", now=NOW)
        second = MemoryManager(hass)  # same config_dir
        items = await second.async_list()
        assert len(items) == 1
        assert items[0]["text"] == "persistent"

    async def test_corrupt_file_treated_as_empty(self, memory_manager, tmp_path):
        (tmp_path / "voice_automation_ai_memory.json").write_text("{not json")
        assert await memory_manager.async_list() == []
