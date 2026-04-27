"""Unit tests for str_replace_editor input normalization and error recovery.

Tests the defense-in-depth layers that make file operations reliable
regardless of model quality (path aliases, fence stripping, path inference,
dynamic corrections, one-shot examples, FileExistsError overwrite).
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from confucius.orchestrator.extensions.file.edit import (
    _normalize_editor_input,
    _strip_markdown_fences,
    _infer_path_from_content,
    _build_dynamic_correction,
    _FIELD_ALIASES,
)


# ── Layer 1a: Field Alias Normalization ──


class TestFieldAliases:
    """Verify all field aliases map to canonical names."""

    @pytest.mark.parametrize(
        "alias,canonical",
        [
            # file_text aliases
            ("content", "file_text"),
            ("text", "file_text"),
            ("file_content", "file_text"),
            ("code", "file_text"),
            ("body", "file_text"),
            # old_str aliases
            ("old_text", "old_str"),
            ("find", "old_str"),
            ("search", "old_str"),
            ("original", "old_str"),
            # new_str aliases
            ("new_text", "new_str"),
            ("replace", "new_str"),
            ("replacement", "new_str"),
            # insert_line aliases
            ("line", "insert_line"),
            ("line_number", "insert_line"),
            ("after_line", "insert_line"),
            # view_range aliases
            ("range", "view_range"),
            ("lines", "view_range"),
            # path aliases
            ("file_path", "path"),
            ("filename", "path"),
            ("file_name", "path"),
            ("filepath", "path"),
            ("file", "path"),
        ],
    )
    def test_alias_maps_to_canonical(self, alias: str, canonical: str):
        assert _FIELD_ALIASES[alias] == canonical

    def test_normalize_path_alias(self):
        raw = {"command": "create", "file_path": "/workspace/test.py", "file_text": "x = 1"}
        result = _normalize_editor_input(raw)
        assert result["path"] == "/workspace/test.py"
        assert "file_path" not in result

    def test_normalize_content_alias(self):
        raw = {"command": "create", "path": "/workspace/test.py", "content": "x = 1"}
        result = _normalize_editor_input(raw)
        assert result["file_text"] == "x = 1"
        assert "content" not in result

    def test_canonical_names_pass_through(self):
        raw = {"command": "create", "path": "/workspace/test.py", "file_text": "x = 1"}
        result = _normalize_editor_input(raw)
        assert result == raw

    def test_canonical_takes_priority_over_alias(self):
        """If both canonical and alias are present, canonical wins."""
        raw = {"command": "create", "path": "/a.py", "file_text": "correct", "content": "wrong"}
        result = _normalize_editor_input(raw)
        assert result["file_text"] == "correct"


# ── Layer 1b: Markdown Fence Stripping ──


class TestMarkdownFenceStripping:
    def test_strip_python_fence(self):
        text = '```python\ndef main():\n    pass\n```'
        result = _strip_markdown_fences(text)
        assert result == "def main():\n    pass"

    def test_strip_bare_fence(self):
        text = '```\nsome code\n```'
        result = _strip_markdown_fences(text)
        assert result == "some code"

    def test_no_strip_markdown_file(self):
        """Don't strip fences if the content is a markdown document."""
        text = '```markdown\n# Title\n\nBody text\n```'
        result = _strip_markdown_fences(text)
        assert result == text  # Unchanged

    def test_no_strip_md_file(self):
        text = '```md\n# Title\n```'
        result = _strip_markdown_fences(text)
        assert result == text  # Unchanged

    def test_no_strip_plain_text(self):
        text = 'def main():\n    pass'
        result = _strip_markdown_fences(text)
        assert result == text

    def test_strip_bash_fence(self):
        text = '```bash\necho hello\n```'
        result = _strip_markdown_fences(text)
        assert result == "echo hello"

    def test_normalize_strips_fences_from_file_text(self):
        raw = {
            "command": "create",
            "path": "/workspace/test.py",
            "file_text": '```python\nx = 1\n```',
        }
        result = _normalize_editor_input(raw)
        assert result["file_text"] == "x = 1"


# ── Layer 1c: Path Inference ──


class TestPathInference:
    def test_infer_python_shebang(self):
        result = _infer_path_from_content("#!/usr/bin/env python3\nimport sys")
        assert result == "/workspace/script.py"

    def test_infer_bash_shebang(self):
        result = _infer_path_from_content("#!/bin/bash\nset -e")
        assert result == "/workspace/script.sh"

    def test_infer_powershell_param(self):
        result = _infer_path_from_content("param(\n    [string]$Name\n)")
        assert result == "/workspace/script.ps1"

    def test_infer_python_import(self):
        result = _infer_path_from_content("import os\nimport sys")
        assert result == "/workspace/script.py"

    def test_infer_python_def(self):
        result = _infer_path_from_content("def main():\n    pass")
        assert result == "/workspace/script.py"

    def test_no_inference_for_random_text(self):
        result = _infer_path_from_content("Hello world, this is just text.")
        assert result is None

    def test_no_inference_for_empty(self):
        result = _infer_path_from_content("")
        assert result is None

    def test_normalize_infers_path_when_missing(self):
        raw = {
            "command": "create",
            "file_text": "#!/usr/bin/env python3\nprint('hello')",
        }
        result = _normalize_editor_input(raw)
        assert result["path"] == "/workspace/script.py"

    def test_normalize_does_not_infer_when_path_present(self):
        raw = {
            "command": "create",
            "path": "/workspace/my_file.py",
            "file_text": "#!/usr/bin/env python3\nprint('hello')",
        }
        result = _normalize_editor_input(raw)
        assert result["path"] == "/workspace/my_file.py"


# ── Layer 2a: Dynamic Correction ──


class TestDynamicCorrection:
    def test_missing_path_shows_correct_format(self):
        raw = {"command": "create", "file_text": "x = 1"}
        error = ValueError("1 validation error for TextEditorInput\npath\n  Field required")
        correction = _build_dynamic_correction(raw, "create", error)
        assert "path" in correction.lower()
        assert '"command": "create"' in correction
        assert '"path":' in correction

    def test_missing_file_text_shows_field(self):
        raw = {"command": "create", "path": "/workspace/test.py"}
        error = ValueError("file_text is required")
        correction = _build_dynamic_correction(raw, "create", error)
        assert "file_text" in correction

    def test_wrong_field_shows_correction(self):
        raw = {"command": "create", "content": "x = 1", "path": "/workspace/test.py"}
        error = ValueError("some error")
        correction = _build_dynamic_correction(raw, "create", error)
        assert "content" in correction
        assert "file_text" in correction

    def test_str_replace_empty_old_str_points_to_create(self):
        """Empty old_str is the classic "I want to make a new file" mistake.

        Regression for P8738 — agent called str_replace with old_str="" to
        write a new doc file, got an opaque ValueError, retried with the
        same wrong command. Correction must explicitly mention `create`.
        """
        raw = {
            "command": "str_replace",
            "path": "/workspace/EVA/notes/new.md",
            "old_str": "",
            "new_str": "# Notes\n",
        }
        error = ValueError(
            '`str_replace` requires a non-empty `old_str` (the exact '
            'existing text to find in the file). To create a NEW file, '
            'use `command="create"` with `file_text` instead.'
        )
        correction = _build_dynamic_correction(raw, "str_replace", error)
        assert "create" in correction.lower()
        assert "old_str" in correction
        # Should suggest create, not just dump the raw error
        assert "file_text" in correction.lower()

    def test_str_replace_file_not_found_points_to_create(self):
        """File doesn't exist → str_replace can't work → suggest create.

        Second half of P8738 — agent's retry passed a path that didn't
        exist. The raw FileNotFoundError doesn't tell the agent how to
        recover; the correction must.
        """
        raw = {
            "command": "str_replace",
            "path": "/workspace/EVA/notes/new.md",
            "old_str": "anything",
            "new_str": "# Notes\n",
        }
        error = FileNotFoundError(
            "File does not exist: /workspace/EVA/notes/new.md. "
            'To create a new file at this path, use `command="create"` '
            "with `file_text` instead. `str_replace` only modifies "
            "existing files."
        )
        correction = _build_dynamic_correction(raw, "str_replace", error)
        assert "create" in correction.lower()
        assert "does not exist" in correction.lower()


# ── Smoke test for the full normalization pipeline ──


class TestFullNormalizationPipeline:
    """Test that a completely malformed input gets normalized into something usable."""

    def test_all_aliases_plus_fence_stripping(self):
        raw = {
            "command": "create",
            "filepath": "/workspace/test.py",
            "code": "```python\ndef hello():\n    return 'world'\n```",
        }
        result = _normalize_editor_input(raw)
        assert result["command"] == "create"
        assert result["path"] == "/workspace/test.py"
        assert result["file_text"] == "def hello():\n    return 'world'"
        assert "```" not in result["file_text"]

    def test_missing_path_with_shebang_infers(self):
        raw = {
            "command": "create",
            "body": "#!/usr/bin/env python3\nprint('hello')",
        }
        result = _normalize_editor_input(raw)
        assert result["path"] == "/workspace/script.py"
        assert result["file_text"] == "#!/usr/bin/env python3\nprint('hello')"
