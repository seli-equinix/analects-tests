"""Pre-validation directive errors for memory tools.

Regression guard for P22165_5-retrieval-modes Turn 3 — the LLM emitted
`edit_memory({"path": "/workspace/EVA/research/foo.md", "old_str": "..."})`
missing `new_str`. Pre-fix, Pydantic raised a verbose ValidationError stack
trace that gave the LLM no actionable next step. Post-fix,
`_pre_validate_memory_tool_input` emits a clean directive:

    edit_memory requires `new_str` (replacement text). If you wanted to
    REPLACE the whole node, use `write_memory(path, content)` instead …

These invariants prevent the same drift from sneaking back via future
refactors to the memory extension's input handling.

Pure-Python function tests — only the helper itself, no Memgraph or
extension wiring needed. The helper does NOT import langchain or bs4,
so this runs on node5 AND in the cca-tests CI image.
"""
from __future__ import annotations

import pytest


# Use importorskip in case future deps creep into the module path. The
# helper itself is import-light, but the file lives next to extension.py
# which transitively imports the orchestrator runtime.
pytest.importorskip(
    "langchain_core",
    reason="memory/hierarchical/extension transitively imports "
           "orchestrator runtime; runs in cca-tests CI image.",
)

from confucius.orchestrator.extensions.memory.hierarchical.extension import (  # noqa: E402
    _pre_validate_memory_tool_input,
)


# ── edit_memory directives ───────────────────────────────────────────


class TestEditMemoryDirectives:
    def test_missing_new_str_directs_to_write_memory(self):
        with pytest.raises(ValueError) as excinfo:
            _pre_validate_memory_tool_input("edit_memory", {
                "path": "research/foo.md",
                "old_str": "old content",
                # new_str missing
            })
        msg = str(excinfo.value)
        assert "new_str" in msg
        assert "write_memory" in msg, (
            "missing-new_str directive must point at write_memory as "
            "the right alternative; got: " + msg
        )

    def test_missing_path_redirects_to_str_replace_editor(self):
        with pytest.raises(ValueError) as excinfo:
            _pre_validate_memory_tool_input("edit_memory", {
                "old_str": "x",
                "new_str": "y",
                # path missing
            })
        msg = str(excinfo.value)
        assert "path" in msg
        assert "str_replace_editor" in msg, (
            "missing-path directive must mention str_replace_editor as "
            "the workspace-files alternative; got: " + msg
        )

    def test_missing_old_str_directs_to_write_memory(self):
        with pytest.raises(ValueError) as excinfo:
            _pre_validate_memory_tool_input("edit_memory", {
                "path": "research/foo.md",
                "new_str": "y",
                # old_str missing
            })
        msg = str(excinfo.value)
        assert "old_str" in msg
        assert "write_memory" in msg

    def test_workspace_path_normalized_not_rejected(self):
        # edit_memory used to REJECT a /workspace path (redirect to
        # str_replace_editor); that raised a hard tool_errors failure the
        # model couldn't cleanly recover from (sweep regression
        # api-sdk-docs/powershell-docs 2026-06-02). It now NORMALIZES the
        # absolute path to a relative memory-node path IN PLACE and
        # proceeds — mirroring write_memory's tolerance.
        inp = {
            "path": "/workspace/EVA/research/add_vm_from_template_callers.md",
            "old_str": "x",
            "new_str": "y",
        }
        _pre_validate_memory_tool_input("edit_memory", inp)  # must NOT raise
        assert inp["path"] == "EVA/research/add_vm_from_template_callers.md", (
            "edit_memory must strip the /workspace/ prefix to a relative "
            f"memory path; got {inp['path']!r}"
        )

    def test_bare_absolute_path_normalized(self):
        # A bare leading "/" (e.g. "/plan/foo.md") is also normalized to a
        # relative memory node path rather than addressing a missing node.
        inp = {"path": "/plan/powershell-module.md", "old_str": "x", "new_str": "y"}
        _pre_validate_memory_tool_input("edit_memory", inp)
        assert inp["path"] == "plan/powershell-module.md"

    def test_valid_input_returns_silently(self):
        # All three required fields present, no workspace prefix → pass.
        _pre_validate_memory_tool_input("edit_memory", {
            "path": "research/foo.md",
            "old_str": "old content",
            "new_str": "new content",
        })
        # No raise = test passes.

    def test_empty_new_str_is_valid(self):
        # `new_str=""` is the deletion semantic — should pass.
        _pre_validate_memory_tool_input("edit_memory", {
            "path": "research/foo.md",
            "old_str": "to delete",
            "new_str": "",
        })


# ── write_memory directives ──────────────────────────────────────────


class TestWriteMemoryDirectives:
    def test_missing_path(self):
        with pytest.raises(ValueError) as excinfo:
            _pre_validate_memory_tool_input("write_memory", {
                "content": "some content",
            })
        assert "path" in str(excinfo.value)

    def test_missing_content(self):
        with pytest.raises(ValueError) as excinfo:
            _pre_validate_memory_tool_input("write_memory", {
                "path": "research/foo.md",
            })
        msg = str(excinfo.value)
        assert "content" in msg
        # Empty-string hint for clarity:
        assert "empty file" in msg.lower() or 'content=""' in msg

    def test_empty_content_is_valid(self):
        # `content=""` for an empty file should pass.
        _pre_validate_memory_tool_input("write_memory", {
            "path": "research/foo.md",
            "content": "",
        })

    def test_workspace_path_normalized_not_rejected(self):
        # 2026-05-30 (P22486 fix): write_memory NORMALIZES an absolute
        # /workspace path to a relative memory path in place, rather than
        # rejecting it (rejecting derailed the agent into a retry loop).
        inp = {"path": "/workspace/EVA/foo.md", "content": "x"}
        _pre_validate_memory_tool_input("write_memory", inp)  # no raise
        assert inp["path"] == "EVA/foo.md", inp

    def test_bare_absolute_path_normalized(self):
        # A bare absolute path (e.g. "/plan/...") also lands inside the
        # memory namespace by stripping the leading slash.
        inp = {"path": "/plan/progress.md", "content": "x"}
        _pre_validate_memory_tool_input("write_memory", inp)
        assert inp["path"] == "plan/progress.md", inp

    def test_valid_input_returns_silently(self):
        _pre_validate_memory_tool_input("write_memory", {
            "path": "research/foo.md",
            "content": "some content",
            "tags": ["research"],
        })


# ── delete_memory directives ─────────────────────────────────────────


class TestDeleteMemoryDirectives:
    def test_missing_paths(self):
        with pytest.raises(ValueError) as excinfo:
            _pre_validate_memory_tool_input("delete_memory", {})
        msg = str(excinfo.value)
        assert "paths" in msg
        # Should mention list shape:
        assert "LIST" in msg or "list" in msg
        # Example provided:
        assert "research/foo.md" in msg

    def test_valid_paths_list_passes(self):
        _pre_validate_memory_tool_input("delete_memory", {
            "paths": ["research/foo.md", "old/notes.md"],
        })


# ── read_memory directives ───────────────────────────────────────────


class TestReadMemoryDirectives:
    def test_missing_path(self):
        with pytest.raises(ValueError) as excinfo:
            _pre_validate_memory_tool_input("read_memory", {})
        msg = str(excinfo.value)
        assert "path" in msg
        # Workspace-files redirect:
        assert "str_replace_editor" in msg or "view" in msg.lower()


# ── search_memory: NO required fields (P22496 regression fix) ─────────


class TestSearchMemoryHasNoRequiredFields:
    """SearchMemoryInput has only Optional fields (path_pattern,
    content_pattern, tags) + max_results — and NO `query` field. The
    earlier "search_memory requires query" check was a bug that rejected
    every valid pattern search (P22496/P22497/P22503/P22507/P22493).
    Pre-validation must now accept all of these without raising."""

    def test_path_pattern_only_does_not_raise(self):
        _pre_validate_memory_tool_input("search_memory", {
            "path_pattern": "research/*", "max_results": 20,
        })

    def test_content_pattern_only_does_not_raise(self):
        _pre_validate_memory_tool_input("search_memory", {
            "content_pattern": "anomaly|isolation|forest",
        })

    def test_tags_only_does_not_raise(self):
        _pre_validate_memory_tool_input("search_memory", {"tags": ["x"]})

    def test_empty_search_does_not_raise(self):
        # An empty search legitimately means "list all" — valid.
        _pre_validate_memory_tool_input("search_memory", {})


# ── Non-dict inputs pass through (let Pydantic handle) ───────────────


class TestNonDictInputs:
    def test_none_passes_through(self):
        # No raise — Pydantic will reject downstream.
        _pre_validate_memory_tool_input("edit_memory", None)

    def test_string_input_passes_through(self):
        _pre_validate_memory_tool_input("edit_memory", "not a dict")

    def test_list_input_passes_through(self):
        _pre_validate_memory_tool_input("edit_memory", ["x", "y"])


# ── Unknown tool names — silent no-op ────────────────────────────────


class TestUnknownToolName:
    def test_snooze_reminder_silent(self):
        # snooze_reminder is delegated to parent class — helper shouldn't
        # raise for tools it doesn't know about.
        _pre_validate_memory_tool_input("snooze_reminder", {
            "minutes": 10,
        })

    def test_random_name_silent(self):
        _pre_validate_memory_tool_input("not_a_real_tool", {
            "foo": "bar",
        })
