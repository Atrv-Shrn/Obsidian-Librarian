"""Tests for :mod:`obsidian_librarian.agent.prompts` — HITL branch + skill loading.

Validates the propose-then-confirm contract wiring without a live LLM: the system prompt
must (a) include the pending-write marker exactly when WRITE_CONFIRM is on (and the
machine-readable ``{marker} target=… op=…`` line), (b) drop confirmation entirely when off,
(c) carry the bundled Obsidian skill, and (d) respect a custom PENDING_WRITE_MARKER.
"""

from __future__ import annotations

from obsidian_librarian.agent import prompts
from obsidian_librarian.agent.prompts import build_system_prompt, load_obsidian_skill


def _set_env(monkeypatch, **kw):
    for k, v in kw.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, str(v))


def test_skill_loads_nonempty_markdown():
    skill = load_obsidian_skill()
    assert isinstance(skill, str)
    assert skill.strip()  # the bundled obsidian.md is not empty
    # It's markdown — at least one heading line is expected in a skill file.
    assert "#" in skill


def test_prompt_includes_skill_and_role():
    p = build_system_prompt()
    # Role line + skill section header are both present.
    assert "Obsidian Librarian" in p
    assert "# Obsidian Specialist Skill" in p
    # A fragment of the skill text survives into the prompt.
    assert load_obsidian_skill().splitlines()[0] in p


def test_hitl_on_contains_marker_and_stop_instruction(monkeypatch):
    _set_env(monkeypatch, WRITE_CONFIRM="true", PENDING_WRITE_MARKER="[PENDING_WRITE]")
    prompts.build_system_prompt.cache_clear()
    p = build_system_prompt()
    assert "[PENDING_WRITE]" in p
    assert "target=" in p and "op=" in p
    assert "STOP" in p
    assert "Confirming writes" in p


def test_hitl_off_omits_marker(monkeypatch):
    _set_env(monkeypatch, WRITE_CONFIRM="false")
    prompts.build_system_prompt.cache_clear()
    p = build_system_prompt()
    # No pending-write marker / propose-then-confirm machinery when confirmation is off.
    assert "[PENDING_WRITE]" not in p
    assert "target=" not in p
    assert "STOP" not in p
    # But the writes section is still there (direct-write guidance).
    assert "write tools" in p.lower()


def test_custom_marker_is_substituted(monkeypatch):
    custom = "<<AWAIT_CONFIRM>>"
    _set_env(monkeypatch, WRITE_CONFIRM="true", PENDING_WRITE_MARKER=custom)
    prompts.build_system_prompt.cache_clear()
    p = build_system_prompt()
    assert custom in p
    # The default marker must not leak through when a custom one is set.
    assert "[PENDING_WRITE]" not in p