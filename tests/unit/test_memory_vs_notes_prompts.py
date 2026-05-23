"""Memory vs Notes prompt-text invariants.

Regression guard for P21965_5-retrieval-modes Mode 5: the LLM picked
`search_memory` instead of `search_notes` because `_MEMORY_GROUP_BODY`
claimed `/notes/...` as a hierarchical-memory path convention AND
"design notes" as a write_memory use case, while NOTES had no
group-level prompt at all (it was in `_SKIP_GROUPS`).

These invariants prevent the same drift from sneaking back in via
future prompt edits.

Pure-Python text inspection — no Memgraph, no langchain, no bs4 needed.
Runs on node5 venv AND inside the cca-tests container.
"""
from __future__ import annotations

from confucius.server.seeds.prompts import get_new_defaults


def _get(slug: str) -> str:
    """Fetch a seed-default prompt body by slug."""
    bodies = get_new_defaults()
    body = bodies.get(slug, "")
    assert body, f"seed default missing for slug={slug!r}"
    return body


# ── Invariant 1: _MEMORY_GROUP_BODY must not claim "notes" ────────────


class TestMemoryGroupNoLongerClaimsNotes:
    """`tool.memory_group` must not list `/notes/` as a memory path
    convention, nor describe write_memory as the tool for "design notes".
    The word "notes" is fine in context that REFERS to search_notes (the
    disambiguation block we added). What's NOT fine is "/notes/" or
    "design notes"."""

    def test_no_notes_path_convention(self):
        body = _get("tool.memory_group")
        # /notes/ was the smoking gun in P21965 — the path convention
        # actively mapped the user-phrase "notes" to hierarchical memory.
        assert "`/notes/" not in body, (
            "tool.memory_group must NOT list `/notes/` as a path "
            "convention — it falsely claims the word 'notes' for "
            "hierarchical memory and steals queries meant for "
            "search_notes. See P21965_5-retrieval-modes."
        )

    def test_no_design_notes_use_case(self):
        body = _get("tool.memory_group")
        assert "design notes" not in body.lower(), (
            "tool.memory_group must NOT describe write_memory as the "
            "tool for 'design notes' — the phrase 'notes' should map to "
            "search_notes."
        )

    def test_mentions_search_notes_distinction(self):
        """The memory group should explicitly point to search_notes for
        cross-session work. After Fix 1 we added a 'DISTINCT from
        search_notes' block."""
        body = _get("tool.memory_group")
        assert "search_notes" in body, (
            "tool.memory_group should reference search_notes so the LLM "
            "learns the distinction; got body without that reference."
        )


# ── Invariant 2: _NOTES_GROUP_BODY exists with the right content ──────


class TestNotesGroupExists:
    """`tool.notes_group` must exist as a seed default with a body that
    claims ownership of past-session "notes" queries."""

    def test_notes_group_seed_exists(self):
        bodies = get_new_defaults()
        assert "tool.notes_group" in bodies, (
            "tool.notes_group must be in seed defaults — P21965 root "
            "cause was that NOTES had no group-level prompt at all."
        )
        body = bodies["tool.notes_group"]
        assert body.strip(), "tool.notes_group body must not be empty"

    def test_notes_group_mentions_search_notes(self):
        body = _get("tool.notes_group")
        assert "search_notes" in body, (
            "tool.notes_group must mention search_notes (its primary tool)"
        )

    def test_notes_group_distinguishes_from_search_memory(self):
        body = _get("tool.notes_group")
        assert "search_memory" in body, (
            "tool.notes_group must explicitly mention search_memory to "
            "teach the LLM the distinction; without that the LLM "
            "reverts to its prior bias."
        )

    def test_notes_group_has_user_phrase_triggers(self):
        body = _get("tool.notes_group").lower()
        # At least one of the trigger phrases the test prompt uses must
        # appear in the notes group, so the LLM has a direct mapping.
        triggers = ["search your notes", "recall", "previous session"]
        hits = [t for t in triggers if t in body]
        assert hits, (
            f"tool.notes_group must contain at least one user-phrase "
            f"trigger from {triggers!r}; got body without any of them."
        )


# ── Invariant 3: tool.search_notes per-tool description has triggers ──


class TestSearchNotesDescriptionHasTriggers:
    """Even if the group prompt is somehow stripped, the per-tool
    description should still pull the LLM toward search_notes when the
    user message contains the trigger words. Defense in depth."""

    def test_search_notes_description_has_trigger_words(self):
        body = _get("tool.search_notes").lower()
        triggers = ["notes", "recall", "previous session", "what have we learned"]
        hits = [t for t in triggers if t in body]
        assert len(hits) >= 2, (
            f"tool.search_notes must contain at least 2 of {triggers!r} "
            f"so the LLM picks this tool when the user phrasing matches; "
            f"got hits={hits!r}"
        )

    def test_search_notes_distinguishes_from_search_memory(self):
        body = _get("tool.search_notes")
        assert "search_memory" in body, (
            "tool.search_notes must reference search_memory to make the "
            "distinction explicit at the per-tool level too."
        )


# ── Invariant 4: NOTES removed from _SKIP_GROUPS ──────────────────────
# These checks need to import tool_group_prompts which transitively pulls
# in the orchestrator runtime (langchain, bs4) — only present inside the
# cca container. Skip on node5 where those deps aren't installed; CI runs
# the unit-tests bucket inside cca-tests/test-runner which has them.
import pytest  # noqa: E402

_SKIP_GROUPS_DEPS_AVAILABLE = True
try:
    import bs4  # noqa: F401
    import langchain_core  # noqa: F401
except ImportError:
    _SKIP_GROUPS_DEPS_AVAILABLE = False


@pytest.mark.skipif(
    not _SKIP_GROUPS_DEPS_AVAILABLE,
    reason="tool_group_prompts pulls in orchestrator deps (bs4, "
           "langchain_core) only installed in the cca container.",
)
class TestNotesGroupNotSkipped:
    """tool_group_prompts._SKIP_GROUPS must NOT contain ToolGroup.NOTES;
    otherwise the new _NOTES_GROUP_BODY would never reach the LLM
    regardless of what we put in the seed."""

    def test_notes_not_in_skip_groups(self):
        from confucius.server.tool_group_prompts import _SKIP_GROUPS
        from confucius.server.tool_groups import ToolGroup
        assert ToolGroup.NOTES not in _SKIP_GROUPS, (
            "ToolGroup.NOTES must NOT be in _SKIP_GROUPS after the fix. "
            "If it is, _NOTES_GROUP_BODY is silently dropped and the "
            "P21965 regression returns."
        )

    def test_memory_still_not_skipped(self):
        """Sanity check: MEMORY group still active (regression guard
        against accidental skip)."""
        from confucius.server.tool_group_prompts import _SKIP_GROUPS
        from confucius.server.tool_groups import ToolGroup
        assert ToolGroup.MEMORY not in _SKIP_GROUPS
