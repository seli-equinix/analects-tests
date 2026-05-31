"""Hallucinated-completion guard + research-handoff fix (sweep Fix C).

Regression guard for P22687_python-docs: the 9B research model confabulated
"I've already completed your request in the previous turn by creating the
`mnist_nn.py` file" — with ZERO file-write tool calls in the entire trace — and
the 35B trusted it, so no torch code was ever produced. Two CCA-side levers:

1. ``_unverified_file_claim`` — detects a concrete file-creation claim made
   while NO file-writing tool was called this request (the high-confidence
   hallucination signal) so the orchestrator can nudge the model to actually
   create the file.
2. ``_STRUCTURAL_NAV_TOOLS`` — browse_project / list_projects are
   code-STRUCTURE navigation, not web/knowledge SEARCH; a turn that only
   browsed structure must NOT delegate to the 9B research model (which then
   confabulates), so it stays on the 80B coder.

Driven via a stub ``self`` — no full orchestrator instance needed.
importorskip-guarded for node5; runs in the cca-tests CI image.
"""
from __future__ import annotations

import types

import pytest

pytest.importorskip(
    "langchain_core",
    reason="dual_model_orchestrator imports langchain; runs in cca-tests CI image.",
)

from confucius.server.dual_model_orchestrator import (  # noqa: E402
    DualModelOrchestrator,
    _STRUCTURAL_NAV_TOOLS,
)


def _claim(text, called_tools):
    s = types.SimpleNamespace()
    s._all_called_tools = set(called_tools)
    s._get_last_assistant_text = lambda ctx: text
    return DualModelOrchestrator._unverified_file_claim(s, None)


# ── The proven P22687 hallucination is caught ────────────────────────


class TestHallucinatedClaimCaught:
    def test_claim_without_any_write_tool_is_flagged(self):
        text = (
            "I've already completed your request in the previous turn by "
            "creating the mnist_nn.py file with the MNISTNet class."
        )
        # Only browse_project was called — NO file write happened.
        claimed = _claim(text, {"browse_project"})
        assert claimed == "mnist_nn.py", claimed

    def test_created_file_heading_caught(self):
        text = "Here's the complete solution:\n## Created File: app/models.py\n..."
        assert _claim(text, {"browse_project"}) == "app/models.py"


# ── No false positives ───────────────────────────────────────────────


class TestNoFalsePositives:
    def test_real_write_tool_call_trusted(self):
        text = "I created the mnist_nn.py file with the network definition."
        # str_replace_editor WAS called → the claim is legitimate, no nudge.
        assert _claim(text, {"str_replace_editor"}) is None

    def test_bash_write_trusted(self):
        text = "I wrote build.sh with the deploy steps."
        assert _claim(text, {"bash"}) is None

    def test_claim_without_concrete_filename(self):
        # "created a plan" mentions no code file → don't guess.
        text = "I've created a detailed plan for your CI/CD pipeline."
        assert _claim(text, {"browse_project"}) is None

    def test_filename_without_creation_claim(self):
        text = "The file mnist_nn.py would contain an MNISTNet class. Shall I create it?"
        assert _claim(text, {"browse_project"}) is None

    def test_empty_text(self):
        assert _claim("", set()) is None


# ── Structural-nav tools are NOT search/research tools ───────────────


class TestStructuralNavExclusion:
    def test_browse_tools_are_structural_nav(self):
        assert "browse_project" in _STRUCTURAL_NAV_TOOLS
        assert "list_projects" in _STRUCTURAL_NAV_TOOLS

    def test_search_tools_not_in_structural_nav(self):
        # web/knowledge search tools must still delegate to the 9B research cycle
        for t in ("web_search", "fetch_url_content", "search_codebase", "search_docs"):
            assert t not in _STRUCTURAL_NAV_TOOLS
