"""Per-call-site extraction contract tests.

Locks in the parser change that switched call extraction from a
deduplicated-set-of-names model to a per-call-site list of dicts
``[{"name": str, "line": int}, ...]``. The previous model collapsed
30 calls to ``Write-Log`` into a single entry, losing both the count
and the line-number info that the dashboard now needs to render the
"Open-CN ×5 @ lines 47, 102, 218, 305, 412" UX.

Tests run on node5's local venv — no langchain / heavy runtime deps
needed because the parser is pure-Python tree-sitter wrappers + regex.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

# tree_sitter is only installed inside the cca container's image (it
# pulls in compiled language bindings). Skip locally and run via the
# in-container CI bucket.
pytest.importorskip(
    "tree_sitter",
    reason="parser tests need tree-sitter compiled bindings (container only)",
)


# ── Per-language helpers ────────────────────────────────────────────


def _extract_python(code: str) -> List[Dict[str, Any]]:
    from confucius.server.code_intelligence.tree_sitter_parser import (
        TreeSitterParser,
    )
    p = TreeSitterParser()
    funcs = p.extract_functions(code, "python", "/test.py")
    # Tests use a single top-level function per fixture; flatten its calls.
    return [c for f in funcs for c in (f.get("calls") or [])]


def _extract_powershell(code: str) -> List[Dict[str, Any]]:
    from confucius.server.code_intelligence.tree_sitter_parser import (
        TreeSitterParser,
    )
    p = TreeSitterParser()
    funcs = p.extract_functions(code, "powershell", "/test.ps1")
    return [c for f in funcs for c in (f.get("calls") or [])]


def _extract_bash(code: str) -> List[Dict[str, Any]]:
    from confucius.server.code_intelligence.tree_sitter_parser import (
        TreeSitterParser,
    )
    p = TreeSitterParser()
    funcs = p.extract_functions(code, "bash", "/test.sh")
    return [c for f in funcs for c in (f.get("calls") or [])]


# ── Python ──────────────────────────────────────────────────────────


class TestPythonCallSites:
    """Python is the most common language — confirm the dict shape is
    correct and duplicate call sites are preserved."""

    def test_single_call_emits_one_entry_with_line(self):
        code = "def foo():\n    bar()\n"
        calls = _extract_python(code)
        # `bar()` is the only call; it's on line 2 (1-indexed).
        assert len(calls) == 1
        assert calls[0]["name"] == "bar"
        assert calls[0]["line"] == 2

    def test_duplicate_calls_preserved_with_distinct_lines(self):
        """The pre-fix bug — `set()` collapsed all 5 calls to one
        entry. New behavior keeps each call site with its own line."""
        code = (
            "def orchestrate():\n"
            "    Open_CN(vm)\n"      # line 2
            "    Open_CN(vm)\n"      # line 3
            "    Open_CN(vm)\n"      # line 4
            "    Open_CN(vm)\n"      # line 5
            "    Open_CN(vm)\n"      # line 6
        )
        calls = _extract_python(code)
        open_cn_calls = [c for c in calls if c["name"] == "Open_CN"]
        assert len(open_cn_calls) == 5
        assert sorted(c["line"] for c in open_cn_calls) == [2, 3, 4, 5, 6]

    def test_calls_sorted_by_line(self):
        code = (
            "def f():\n"
            "    a()\n"
            "    b()\n"
            "    a()\n"
            "    c()\n"
        )
        calls = _extract_python(code)
        # Sort guarantee — sorted by line ascending so the resolver's
        # collect(line) produces deterministic output.
        lines = [c["line"] for c in calls]
        assert lines == sorted(lines)

    def test_dict_shape_only_name_and_line(self):
        """Caller code (resolver, indexer, tests) relies on the exact
        dict shape — guard against accidental field additions."""
        code = "def f():\n    x()\n"
        calls = _extract_python(code)
        assert len(calls) == 1
        assert set(calls[0].keys()) == {"name", "line"}
        assert isinstance(calls[0]["line"], int)
        assert isinstance(calls[0]["name"], str)


# ── PowerShell ──────────────────────────────────────────────────────


class TestPowerShellCallSites:
    """The motivating use case — JobStart-Standalone.ps1 calling
    `Open-CN` many times. Regex-based extractor; line numbers come
    from match.start() + newline counting."""

    def test_repeated_cmdlet_calls_preserved(self):
        code = (
            "function Invoke-Orchestration {\n"
            "    Open-CN $vm\n"      # line 2
            "    Write-Log 'starting'\n"   # line 3
            "    Open-CN $vm\n"      # line 4
            "    Write-Log 'done'\n"       # line 5
        )
        calls = _extract_powershell(code)
        open_cn = [c for c in calls if c["name"] == "Open-CN"]
        write_log = [c for c in calls if c["name"] == "Write-Log"]
        assert len(open_cn) == 2
        assert len(write_log) == 2

    def test_lines_match_source(self):
        code = (
            "function f {\n"
            "    Get-Process\n"
            "\n"
            "\n"
            "    Set-Item foo bar\n"
        )
        calls = _extract_powershell(code)
        # Get-Process at line 2, Set-Item at line 5 (after 2 blank lines)
        named = {c["name"]: c["line"] for c in calls}
        assert named["Get-Process"] == 2
        assert named["Set-Item"] == 5

    def test_dict_shape(self):
        code = "function f {\n    Get-Item foo\n}\n"
        calls = _extract_powershell(code)
        assert len(calls) >= 1
        assert all(set(c.keys()) == {"name", "line"} for c in calls)


# ── Bash ────────────────────────────────────────────────────────────


class TestBashCallSites:
    def test_repeated_calls_preserved(self):
        code = (
            "do_thing() {\n"
            "    helper foo\n"           # line 2
            "    helper bar\n"           # line 3
            "    other_thing arg\n"      # line 4
            "    helper baz\n"           # line 5
            "}\n"
        )
        calls = _extract_bash(code)
        helper = [c for c in calls if c["name"] == "helper"]
        # 3 helper calls on lines 2, 3, 5
        assert len(helper) == 3
        assert sorted(c["line"] for c in helper) == [2, 3, 5]


# ── Cap behavior ───────────────────────────────────────────────────


class TestCallSiteCap:
    """The 200-call cap protects against degenerate generated code
    that calls one function thousands of times — without it, the
    `_pending_call_names` list could blow past Memgraph's per-property
    size budget."""

    def test_python_cap_applied(self):
        # Generate a function with 250 calls — cap at 200.
        body_lines = ["def big():"]
        for _ in range(250):
            body_lines.append("    f()")
        code = "\n".join(body_lines) + "\n"
        calls = _extract_python(code)
        assert len(calls) == 200
        # Cap keeps the FIRST 200 by line (lowest line numbers).
        # First call is at line 2; 200th is at line 201.
        assert calls[0]["line"] == 2
        assert calls[-1]["line"] == 201
