"""Post-write validation: syntax advisory + degeneration hint + path/suffix gap.

api-lookup hardening. `_validate_written_file` already syntax-checks written
.py files and is appended to the str_replace_editor result (edit.py on_tool_use).
This pins the new behavior:
  - the syntax advisory is ENRICHED with a degeneration explanation when the
    syntax error is caused by a degenerate repeated run (so the model rewrites
    the section instead of looping on the whole file);
  - the check no longer silently skips when the path has no .py suffix (a
    create with path=None / a bare content-named path) — it falls back to
    content-based language detection;
  - the loop-break escalation constant exists.

importorskip-guarded for node5 (file/edit.py pulls bs4 / orchestrator runtime);
runs in the cca-tests CI image.
"""
from __future__ import annotations

import pytest

pytest.importorskip(
    "bs4",
    reason="file/edit.py pulls in the orchestrator runtime (bs4); "
           "runs in the cca-tests CI image.",
)

from confucius.orchestrator.extensions.file.edit import (  # noqa: E402
    _validate_written_file,
    _WRITE_FAIL_ESCALATE_AT,
)


_TERNARY = (
    "    num_sockets = max(1, cpu_count) "
    + "if num_sockets > num_cores_per_socket else num_sockets * num_cores_per_socket " * 9
    + "if num_sockets >\n"
)
_BROKEN_PY = (
    "import os\n\ndef create_vm(cpu_count, num_sockets, num_cores_per_socket):\n"
    + _TERNARY
    + "    return num_sockets\n"
)
_VALID_PY = (
    "import os\n\ndef create_vm(cpu_count, num_sockets, num_cores_per_socket):\n"
    "    return max(1, cpu_count // num_cores_per_socket)\n"
)


class TestSyntaxAdvisory:
    def test_broken_py_reports_syntax_error_with_degen_hint(self, tmp_path):
        p = tmp_path / "vm_creator.py"
        p.write_text(_BROKEN_PY)
        w = _validate_written_file(p)
        assert w is not None
        assert "syntax error" in w.lower()
        assert "repeated" in w.lower() or "degenerate" in w.lower(), (
            "syntax error caused by a degenerate run must be explained: " + w
        )

    def test_valid_py_no_warning(self, tmp_path):
        p = tmp_path / "ok.py"
        p.write_text(_VALID_PY)
        assert _validate_written_file(p) is None

    def test_valid_repetitive_py_not_flagged(self, tmp_path):
        # legit code with repeated-but-varying lines + valid syntax → no warning
        p = tmp_path / "cfg.py"
        p.write_text("cfg = {\n" + "".join(f'    "k_{i}": {i},\n' for i in range(40)) + "}\n")
        assert _validate_written_file(p) is None


class TestPathSuffixGap:
    def test_python_content_without_py_suffix_still_validated(self, tmp_path):
        # The 15:17 api-lookup failure: create with path=None resolved to a
        # bare/no-suffix name → the .py check silently skipped. Now content-
        # detected Python is validated regardless of suffix.
        p = tmp_path / "script"          # no .py suffix
        p.write_text(_BROKEN_PY)
        w = _validate_written_file(p)
        assert w is not None and "syntax error" in w.lower(), (
            "Python content without a .py suffix must still be syntax-checked"
        )

    def test_plain_text_unaffected(self, tmp_path):
        p = tmp_path / "notes.txt"
        p.write_text("just some prose, not code at all.\nsecond line.\n")
        assert _validate_written_file(p) is None


class TestLoopBreakConstant:
    def test_escalation_threshold_is_sane(self):
        assert isinstance(_WRITE_FAIL_ESCALATE_AT, int)
        assert _WRITE_FAIL_ESCALATE_AT >= 2
