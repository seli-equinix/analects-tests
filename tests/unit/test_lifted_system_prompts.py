"""Invariant 2 guard: no system-prompt strings live in confucius/ source
code; everything must be DB-backed via `prompt_loader.get_template` /
`get_template_raw` against a slug.

These tests pin the lifted slugs (`extension.note_extraction` and
`router.tool_selector`) so a regression — someone re-introducing a
hardcoded fallback constant in note_observer or expert_router — fails
fast in CI.
"""

from __future__ import annotations

import pathlib

import pytest


_CONFUCIUS = pathlib.Path(__file__).resolve().parents[2] / "confucius"


def test_extraction_system_template_is_not_in_source():
    """EXTRACTION_SYSTEM_TEMPLATE constant must not exist in confucius/.
    Seed body lives in `confucius/server/seeds/prompts.py:EXTRACTION_SYSTEM`;
    note_observer reads it via slug `extension.note_extraction`."""
    matches = []
    for p in _CONFUCIUS.rglob("*.py"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        # Match an assignment / declaration of the OLD constant, not the
        # comment that documents its removal.
        for lineno, line in enumerate(text.splitlines(), start=1):
            if "EXTRACTION_SYSTEM_TEMPLATE" in line and "=" in line and not line.lstrip().startswith("#"):
                matches.append(f"{p.relative_to(_CONFUCIUS.parent)}:{lineno}: {line.strip()}")
    assert not matches, (
        "EXTRACTION_SYSTEM_TEMPLATE assignment found in source — invariant 2 "
        "violated. Offending lines:\n" + "\n".join(matches)
    )


def test_tool_selector_system_prompt_is_not_in_source():
    """TOOL_SELECTOR_SYSTEM_PROMPT constant must not exist in confucius/.
    Seed body lives in `confucius/server/seeds/prompts.py:TOOL_SELECTOR_SYSTEM`;
    expert_router reads it via slug `router.tool_selector`."""
    matches = []
    for p in _CONFUCIUS.rglob("*.py"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if "TOOL_SELECTOR_SYSTEM_PROMPT" in line and "=" in line and not line.lstrip().startswith("#"):
                matches.append(f"{p.relative_to(_CONFUCIUS.parent)}:{lineno}: {line.strip()}")
    assert not matches, (
        "TOOL_SELECTOR_SYSTEM_PROMPT assignment found in source — invariant 2 "
        "violated. Offending lines:\n" + "\n".join(matches)
    )


def test_lifted_slugs_have_seed_bodies():
    """The two lifted slugs must have body entries in seeds/prompts.py
    so the entrypoint seeder populates the DB on first boot. Imports
    seeds directly (pure-Python module, no Django needed)."""
    import importlib.util
    seeds_path = _CONFUCIUS / "server" / "seeds" / "prompts.py"
    spec = importlib.util.spec_from_file_location("_seeds_prompts", seeds_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # SLUG_MAP wires the extraction key → canonical slug
    assert mod.SLUG_MAP.get("extraction_system") == "extension.note_extraction"
    assert mod.SLUG_MAP.get("tool_selector_system") == "router.tool_selector"

    # _BODY_MAP wires the extraction key → body constant; both must
    # exist and the body must contain the .format() placeholders the
    # consumer code expects.
    defaults = mod.get_new_defaults()
    note_body = defaults.get("extension.note_extraction") or ""
    selector_body = defaults.get("router.tool_selector") or ""

    assert note_body, "extension.note_extraction seed body is empty"
    assert "{project_names}" in note_body, (
        "note-extraction seed body missing {project_names} placeholder "
        "that note_observer.py calls .format(project_names=...) on"
    )

    assert selector_body, "router.tool_selector seed body is empty"
    assert "{current_route}" in selector_body, (
        "tool-selector seed body missing {current_route} placeholder"
    )
    assert "{current_tools}" in selector_body, (
        "tool-selector seed body missing {current_tools} placeholder"
    )


def test_seed_version_was_bumped():
    """Adding a new slug requires bumping SEED_VERSION so the entrypoint
    seed runner re-applies on the next container boot."""
    import importlib.util
    seeds_path = _CONFUCIUS / "server" / "seeds" / "prompts.py"
    spec = importlib.util.spec_from_file_location("_seeds_prompts2", seeds_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.SEED_VERSION >= 12, (
        f"SEED_VERSION = {mod.SEED_VERSION}; should be >= 12 after the "
        "tool-selector lift (was 11, bumped to 12 in this change)"
    )
