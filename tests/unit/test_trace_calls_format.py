"""trace_execution tolerates list-of-dict `calls` (per-call-site format).

Regression guard for P22269_eva-full-trace. The per-call-site CALLS work
(commit 15b4d441) changed a function's `calls` representation from
list[str] (`["FuncA"]`) to list[dict] (`[{"name":"FuncA","line":10}]`) in
tree_sitter_parser + the Qdrant payload. trace_extension.py was the missed
consumer — it does set membership (`callee not in visited`), dict-key
lookups, and str joins on `calls` elements, all of which crash with
`TypeError: unhashable type: 'dict'` when an element is a dict.

The fix adds `_call_name()` and normalizes `calls` to list[str] at the two
ingestion points (`_build_func_dict`, `_handle_trace`) plus a defensive
guard in `_bfs_resolve`. These tests pin that behavior.

Pure-Python: exercises `_call_name` + `_bfs_resolve` directly with no
Qdrant / Memgraph / live server. importorskip-guarded for node5 (the
module transitively imports the orchestrator runtime).
"""
from __future__ import annotations

import pytest

pytest.importorskip(
    "langchain_core",
    reason="trace_extension transitively imports orchestrator runtime; "
           "runs in the cca-tests CI image.",
)

from confucius.server.code_intelligence.trace_extension import (  # noqa: E402
    CodeTraceExtension,
    _call_name,
)


# ── _call_name normalizer ─────────────────────────────────────────────


class TestCallNameNormalizer:
    def test_dict_returns_name(self):
        assert _call_name({"name": "Foo", "line": 10}) == "Foo"

    def test_str_returns_itself(self):
        assert _call_name("Bar") == "Bar"

    def test_dict_missing_name_returns_empty(self):
        assert _call_name({"line": 5}) == ""

    def test_dict_null_name_returns_empty(self):
        assert _call_name({"name": None, "line": 5}) == ""

    def test_non_dict_non_str_coerced(self):
        # Defensive — any odd type becomes its str(), never crashes.
        assert _call_name(123) == "123"


# ── _bfs_resolve with list-of-dict calls (the P22269 crash) ───────────


def _ext() -> CodeTraceExtension:
    # backend_clients unused by _bfs_resolve; pass a dummy.
    return CodeTraceExtension(backend_clients=object())


class TestBfsResolveToleratesDictCalls:
    def test_dict_shaped_calls_do_not_crash(self):
        """The exact P22269 reproducer: func_dict with list-of-dict calls.
        Pre-fix this raised `TypeError: unhashable type: 'dict'`."""
        ext = _ext()
        func_dict = {
            "Invoke-Job": {
                "file_path": "/workspace/EVA/code/JobStart.ps1",
                "line_start": 1, "line_end": 50,
                "calls": [
                    {"name": "Open-DB", "line": 10},
                    {"name": "Add-VMFromTemplate", "line": 20},
                ],
            },
            "Open-DB": {
                "file_path": "/workspace/EVA/code/db.ps1",
                "line_start": 1, "line_end": 10,
                "calls": [],
            },
            "Add-VMFromTemplate": {
                "file_path": "/workspace/EVA/code/vcenter.psm1",
                "line_start": 1, "line_end": 30,
                "calls": [{"name": "Open-DB", "line": 5}],
            },
        }
        # entry_calls may itself contain dicts (from tree-sitter) — but the
        # ingestion normalization should have stringified them. Here we
        # pass strings (the post-normalization contract) and assert the
        # BFS resolves the dict-shaped func_dict["calls"] without crashing.
        needed, external = ext._bfs_resolve(
            ["Invoke-Job"], func_dict, max_depth=15,
        )
        assert "Invoke-Job" in needed
        assert "Open-DB" in needed
        assert "Add-VMFromTemplate" in needed
        assert external == set()

    def test_string_shaped_calls_still_work(self):
        """Backward-compat: old list[str] format resolves identically."""
        ext = _ext()
        func_dict = {
            "A": {"file_path": "/w/a.py", "line_start": 1, "line_end": 5,
                  "calls": ["B"]},
            "B": {"file_path": "/w/b.py", "line_start": 1, "line_end": 5,
                  "calls": []},
        }
        needed, external = ext._bfs_resolve(["A"], func_dict, max_depth=15)
        assert needed == {"A", "B"}
        assert external == set()

    def test_external_deps_detected(self):
        """Callees not in func_dict become external deps (no crash on
        dict-shaped calls)."""
        ext = _ext()
        func_dict = {
            "A": {"file_path": "/w/a.py", "line_start": 1, "line_end": 5,
                  "calls": [{"name": "ThirdPartyFunc", "line": 3}]},
        }
        needed, external = ext._bfs_resolve(["A"], func_dict, max_depth=15)
        assert "A" in needed
        assert "ThirdPartyFunc" in external

    def test_mixed_str_and_dict_calls(self):
        """Defensive: a func_dict mixing str and dict call entries
        (e.g. partial reindex) still resolves."""
        ext = _ext()
        func_dict = {
            "A": {"file_path": "/w/a.py", "line_start": 1, "line_end": 5,
                  "calls": ["B", {"name": "C", "line": 2}]},
            "B": {"file_path": "/w/b.py", "line_start": 1, "line_end": 5,
                  "calls": []},
            "C": {"file_path": "/w/c.py", "line_start": 1, "line_end": 5,
                  "calls": []},
        }
        needed, _ = ext._bfs_resolve(["A"], func_dict, max_depth=15)
        assert needed == {"A", "B", "C"}

    def test_empty_name_dict_skipped(self):
        """A dict call entry with no name normalizes to "" and is skipped,
        not enqueued as an empty-string node."""
        ext = _ext()
        func_dict = {
            "A": {"file_path": "/w/a.py", "line_start": 1, "line_end": 5,
                  "calls": [{"line": 9}]},  # no 'name'
        }
        needed, external = ext._bfs_resolve(["A"], func_dict, max_depth=15)
        assert needed == {"A"}
        assert "" not in needed
        assert "" not in external
