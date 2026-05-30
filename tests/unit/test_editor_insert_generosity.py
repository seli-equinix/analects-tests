"""str_replace_editor: extend create-path generosity to insert/str_replace.

Regression guard for P22371_eva-full-trace Turn 2. The agent built a
helper file via `command=insert` and racked up 4 unrecovered tool errors
because the "smart resolution" generosity in `_normalize_editor_input`
was wired ONLY for `command=create`:
  Gap 1 — `file_text` (the create content field) wasn't moved to `new_str`
          for insert/str_replace, so the content was dropped.
  Gap 2 — an empty-but-present `path:""` slipped past the last-edited
          fallback (which only checked absence), resolving to "." →
          "outside /workspace".
  Gap 3 — `insert` on a non-existent file wasn't redirected to `create`
          (you can't insert into a file that doesn't exist).

These tests pin all three fixes + the no-op/no-clobber guards.

Pure-Python for Fixes 1/2 (module function); Fix 3 mocks the create
handler. importorskip-guarded for node5 (edit.py pulls in the
orchestrator runtime / bs4); runs in the cca-tests CI image.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip(
    "bs4",
    reason="file/edit.py pulls in the orchestrator runtime (bs4); "
           "runs in the cca-tests CI image.",
)

from confucius.orchestrator.extensions.file.edit import (  # noqa: E402
    _normalize_editor_input,
    FileEditExtension,
)
from confucius.core.chat_models.bedrock.api.invoke_model import (  # noqa: E402
    anthropic as ant,
)


# ── Fix 1: content rescue (file_text → new_str) for insert/str_replace ──


class TestContentRescue:
    def test_insert_file_text_moved_to_new_str(self):
        out = _normalize_editor_input({
            "command": "insert", "path": "/workspace/f.py",
            "insert_line": 26, "file_text": "the content",
        })
        assert out.get("new_str") == "the content", out
        assert "file_text" not in out, "file_text should be consumed"

    def test_str_replace_file_text_moved_to_new_str(self):
        out = _normalize_editor_input({
            "command": "str_replace", "path": "/workspace/f.py",
            "old_str": "a", "file_text": "b",
        })
        assert out.get("new_str") == "b"
        assert "file_text" not in out

    def test_content_alias_rescued_for_insert(self):
        # `content` is alias-folded to file_text, then rescued to new_str.
        out = _normalize_editor_input({
            "command": "insert", "path": "/workspace/f.py",
            "insert_line": 1, "content": "via content alias",
        })
        assert out.get("new_str") == "via content alias"

    def test_no_clobber_when_new_str_present(self):
        # If the model correctly supplied new_str, never overwrite it.
        out = _normalize_editor_input({
            "command": "insert", "path": "/workspace/f.py",
            "insert_line": 1, "new_str": "correct", "file_text": "wrong",
        })
        assert out.get("new_str") == "correct"

    def test_create_unaffected_by_rescue(self):
        # create still keeps its file_text (the rescue is insert/str_replace only).
        out = _normalize_editor_input({
            "command": "create", "path": "/workspace/f.py",
            "file_text": "hello",
        })
        assert out.get("file_text") == "hello"


# ── Fix 2: empty/missing path → last-edited file ──


class TestEmptyPathFallback:
    def test_empty_path_uses_last_edited(self):
        out = _normalize_editor_input(
            {"command": "insert", "path": "", "insert_line": 1, "new_str": "X"},
            last_edited_path="/workspace/prev.py",
        )
        assert out.get("path") == "/workspace/prev.py"

    def test_absent_path_uses_last_edited(self):
        out = _normalize_editor_input(
            {"command": "str_replace", "old_str": "a", "new_str": "b"},
            last_edited_path="/workspace/prev.py",
        )
        assert out.get("path") == "/workspace/prev.py"

    def test_present_path_not_overridden(self):
        out = _normalize_editor_input(
            {"command": "insert", "path": "/workspace/explicit.py",
             "insert_line": 1, "new_str": "X"},
            last_edited_path="/workspace/prev.py",
        )
        assert out.get("path") == "/workspace/explicit.py"

    def test_empty_path_no_last_edited_stays_empty(self):
        # No continuity to fall back to — path stays empty (handler/policy
        # will surface a clear error; we don't invent a path here).
        out = _normalize_editor_input(
            {"command": "insert", "path": "", "insert_line": 1, "new_str": "X"},
            last_edited_path="",
        )
        assert not out.get("path")


# ── Fix 3: insert on non-existent file → redirect to create ──


def _ext() -> FileEditExtension:
    ext = FileEditExtension(
        max_output_lines=500,
        enable_tool_use=True,
        editor_tool=ant.TextEditor(name="str_replace_editor"),
    )
    return ext


class TestInsertMissingFileRedirectsToCreate:
    # These call the async handler via asyncio.run() rather than
    # @pytest.mark.asyncio so they're invocation-independent (pytest-asyncio
    # auto-mode only engages under a whole-dir run, not a single-file one).

    def test_insert_missing_file_with_content_creates(self, tmp_path):
        ext = _ext()
        missing = tmp_path / "does_not_exist.py"
        edit_input = ant.TextEditorInput(
            command=ant.TextEditorCommand.INSERT,
            path=str(missing), new_str="full content", insert_line=None,
        )
        with patch.object(
            ext, "_on_create_command", new=AsyncMock(return_value="created"),
        ) as mock_create:
            result = asyncio.run(
                ext._on_insert_command(missing, edit_input, MagicMock())
            )
        assert result == "created"
        mock_create.assert_awaited_once()
        # The redirected edit_input must carry CREATE + file_text=content.
        passed = mock_create.call_args.kwargs.get("edit_input") or mock_create.call_args.args[1]
        assert passed.command == ant.TextEditorCommand.CREATE
        assert passed.file_text == "full content"

    def test_insert_missing_file_no_content_still_errors(self, tmp_path):
        # No content → don't silently create an empty file; the directive fires.
        ext = _ext()
        missing = tmp_path / "does_not_exist.py"
        edit_input = ant.TextEditorInput(
            command=ant.TextEditorCommand.INSERT,
            path=str(missing), new_str=None, insert_line=None,
        )
        with patch.object(ext, "_on_create_command", new=AsyncMock()) as mock_create:
            with pytest.raises(ValueError) as ei:
                asyncio.run(
                    ext._on_insert_command(missing, edit_input, MagicMock())
                )
        assert "insert_line" in str(ei.value)
        mock_create.assert_not_awaited()

    def test_insert_existing_file_no_redirect(self, tmp_path):
        # Existing file + valid insert → normal insert path, no create redirect.
        ext = _ext()
        existing = tmp_path / "real.py"
        existing.write_text("line1\nline2\n")
        edit_input = ant.TextEditorInput(
            command=ant.TextEditorCommand.INSERT,
            path=str(existing), new_str="inserted", insert_line=1,
        )
        with patch.object(ext, "_on_create_command", new=AsyncMock()) as mock_create, \
             patch.object(ext, "_on_insert", new=AsyncMock(return_value="inserted-ok")) as mock_insert:
            result = asyncio.run(
                ext._on_insert_command(existing, edit_input, MagicMock())
            )
        assert result == "inserted-ok"
        mock_create.assert_not_awaited()
        mock_insert.assert_awaited_once()
