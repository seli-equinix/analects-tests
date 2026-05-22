"""Tests for evaluators.eval_tool_errors memory-family recovery.

Verifies the Fix B change to tests/evaluators.py: when edit_memory
fails AND a later iteration of write_memory or delete_memory (or
import_memory) succeeds in the same session, the edit_memory failure
is treated as recovered.

Run:
    pytest tests/unit/test_evaluator_memory_recovery.py -v
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import pytest

from tests.evaluators import eval_tool_errors


@dataclass
class FakeChatResult:
    """Minimal ChatResult standin — only the fields eval_tool_errors reads."""
    content: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def tool_labels(self) -> List[str]:
        return self.raw.get("tool_labels", [])

    @property
    def tool_errors(self) -> List[str]:
        return [
            l for l in self.tool_labels
            if any(k in l.lower() for k in ("failed", "error", "invalid"))
        ]

    @property
    def tool_failures(self) -> List[Dict[str, Any]]:
        return self.raw.get("tool_failures", [])


def _result(content: str, tool_calls: List[Dict[str, Any]],
            tool_labels: List[str],
            tool_failures: List[Dict[str, Any]] | None = None) -> FakeChatResult:
    """Helper to build a synthetic ChatResult."""
    return FakeChatResult(
        content=content,
        raw={
            "tool_labels": tool_labels,
            "tool_failures": tool_failures or [],
        },
        metadata={"tool_calls": tool_calls},
    )


class TestEditMemoryRecoveryViaWriteMemory:
    """The Fix B branch: edit_memory failed + write_memory succeeded later."""

    def test_recovered_when_write_memory_succeeds_after_edit_fail(self):
        """edit_memory fails at iter 5, write_memory succeeds at iter 6 →
        eval should PASS (not flag as unrecovered)."""
        r = _result(
            content="x" * 500,  # substantive
            tool_calls=[
                {"name": "browse_project", "success": True, "iteration": 1},
                {"name": "str_replace_editor", "success": True, "iteration": 2},
                {"name": "edit_memory", "success": False, "iteration": 5},
                {"name": "write_memory", "success": True, "iteration": 6},
            ],
            tool_labels=["browse_project", "Edit memory Failed", "write_memory"],
            tool_failures=[
                {"tool_name": "edit_memory", "label": "Edit memory Failed",
                 "failure_kind": "exception",
                 "exception_type": "ValidationError",
                 "exception_message": "path field missing"},
            ],
        )
        out = eval_tool_errors(r)
        assert out is not None
        assert out["score"] == 1.0, (
            f"expected score=1.0 (PASS) for recovered edit_memory, "
            f"got {out!r}"
        )
        assert "recovered" in out["label"].lower()

    def test_recovered_when_delete_then_write_succeed_after_edit_fail(self):
        """edit_memory fails at iter 5, delete_memory + write_memory
        succeed at iters 6, 7 → eval should PASS."""
        r = _result(
            content="x" * 500,
            tool_calls=[
                {"name": "edit_memory", "success": False, "iteration": 5},
                {"name": "delete_memory", "success": True, "iteration": 6},
                {"name": "write_memory", "success": True, "iteration": 7},
            ],
            tool_labels=["Edit memory Failed", "delete_memory", "write_memory"],
            tool_failures=[
                {"tool_name": "edit_memory", "label": "Edit memory Failed",
                 "failure_kind": "exception",
                 "exception_message": "old_str field missing"},
            ],
        )
        out = eval_tool_errors(r)
        assert out["score"] == 1.0, f"got {out!r}"

    def test_two_edit_memory_failures_recovered_by_later_write(self):
        """The browse-project P21666 pattern exactly: edit_memory fails
        TWICE (iters 13, 14), then delete_memory+write_memory succeed."""
        r = _result(
            content="# JobStart.ps1 Analysis\n" + ("x" * 500),
            tool_calls=[
                {"name": "browse_project", "success": True, "iteration": 1},
                {"name": "edit_memory", "success": False, "iteration": 13},
                {"name": "edit_memory", "success": False, "iteration": 14},
                {"name": "delete_memory", "success": True, "iteration": 15},
                {"name": "write_memory", "success": True, "iteration": 16},
            ],
            tool_labels=[
                "browse_project",
                "Edit memory Failed",
                "Edit memory Failed",
                "delete_memory",
                "write_memory",
            ],
            tool_failures=[
                {"tool_name": "edit_memory", "label": "Edit memory Failed",
                 "failure_kind": "exception",
                 "exception_message": "path field missing"},
                {"tool_name": "edit_memory", "label": "Edit memory Failed",
                 "failure_kind": "exception",
                 "exception_message": "old_str field missing"},
            ],
        )
        out = eval_tool_errors(r)
        assert out["score"] == 1.0, (
            f"P21666 reproducer should PASS after Fix B; got {out!r}"
        )

    def test_not_recovered_when_write_memory_succeeded_BEFORE_edit_fail(self):
        """Order matters: write_memory at iter 1, edit_memory fails at
        iter 5, no later write/delete success → still FAIL.
        The recovery must come AFTER the failure."""
        r = _result(
            content="x" * 500,
            tool_calls=[
                {"name": "write_memory", "success": True, "iteration": 1},
                {"name": "edit_memory", "success": False, "iteration": 5},
            ],
            tool_labels=["write_memory", "Edit memory Failed"],
            tool_failures=[
                {"tool_name": "edit_memory", "label": "Edit memory Failed",
                 "failure_kind": "exception",
                 "exception_message": "path field missing"},
            ],
        )
        out = eval_tool_errors(r)
        # write_memory succeeded BEFORE edit_memory failure — not recovery
        # for THAT failure. Must still be flagged.
        assert out["score"] == 0.0, f"expected FAIL (no recovery), got {out!r}"

    def test_not_recovered_when_no_memory_writes_at_all(self):
        """edit_memory fails, no memory-family success ever → FAIL."""
        r = _result(
            content="x" * 500,
            tool_calls=[
                {"name": "browse_project", "success": True, "iteration": 1},
                {"name": "edit_memory", "success": False, "iteration": 5},
            ],
            tool_labels=["browse_project", "Edit memory Failed"],
            tool_failures=[
                {"tool_name": "edit_memory", "label": "Edit memory Failed",
                 "failure_kind": "exception",
                 "exception_message": "path field missing"},
            ],
        )
        out = eval_tool_errors(r)
        assert out["score"] == 0.0, f"got {out!r}"


class TestExistingRecoveryStillWorks:
    """Fix B must not break the existing same-tool recovery path."""

    def test_same_tool_recovery_still_works(self):
        """edit_memory fails iter 5, edit_memory succeeds iter 6 →
        existing same-tool recovery path PASSES (not Fix B)."""
        r = _result(
            content="x" * 500,
            tool_calls=[
                {"name": "edit_memory", "success": False, "iteration": 5},
                {"name": "edit_memory", "success": True, "iteration": 6},
            ],
            tool_labels=["Edit memory Failed", "edit_memory"],
            tool_failures=[
                {"tool_name": "edit_memory", "label": "Edit memory Failed",
                 "failure_kind": "exception",
                 "exception_message": "path field missing"},
            ],
        )
        out = eval_tool_errors(r)
        assert out["score"] == 1.0, f"got {out!r}"

    def test_clean_run_no_errors(self):
        """No errors at all → PASS with label='clean'."""
        r = _result(
            content="x" * 500,
            tool_calls=[
                {"name": "browse_project", "success": True, "iteration": 1},
                {"name": "write_memory", "success": True, "iteration": 2},
            ],
            tool_labels=["browse_project", "write_memory"],
        )
        out = eval_tool_errors(r)
        assert out["score"] == 1.0
        assert out["label"] == "clean"

    def test_str_replace_editor_unrelated_failure_still_flagged(self):
        """edit_memory failure is recovered (write_memory after), but a
        SEPARATE str_replace_editor failure with no recovery must still
        flag. Tests Fix B narrowly: only edit_memory failures get
        memory-family recovery treatment."""
        r = _result(
            content="x" * 500,
            tool_calls=[
                {"name": "str_replace_editor", "success": False, "iteration": 3},
                {"name": "edit_memory", "success": False, "iteration": 5},
                {"name": "write_memory", "success": True, "iteration": 6},
            ],
            tool_labels=[
                "Str_replace_editor Failed",
                "Edit memory Failed",
                "write_memory",
            ],
            tool_failures=[
                {"tool_name": "str_replace_editor",
                 "label": "Str_replace_editor Failed",
                 "failure_kind": "exception",
                 "command": "str_replace",
                 "exception_message": "Text not found"},
                {"tool_name": "edit_memory", "label": "Edit memory Failed",
                 "failure_kind": "exception",
                 "exception_message": "path field missing"},
            ],
        )
        out = eval_tool_errors(r)
        # str_replace_editor failure remains (not recovered by memory-family).
        # Whether the score is 0 (fail) or PASS via inline-response recovery
        # depends on the substantive-response branch at line 341-347.
        # The substantive (>200 chars) inline-response recovery filters
        # errors containing "command" (lowercase substring). Our label is
        # "Str_replace_editor Failed" which contains no "command" substring,
        # so the str_replace_editor failure stays unrecovered → FAIL.
        # This is correct: tested separately doesn't accidentally pass
        # due to substantive response alone.
        assert out["score"] == 0.0, (
            f"str_replace_editor failure must remain unrecovered when "
            f"only edit_memory has memory-family recovery; got {out!r}"
        )
