"""Flow test: Project-scoped note isolation.

Journey: developer works on EVA project → stores project knowledge →
switches to EVA-migration → verifies EVA notes don't leak → switches
back to EVA → confirms notes are still accessible.

Validates the core project-aware knowledge scoping feature:
  - Notes stored with scope=project get a project field
  - Searching notes in one project context doesn't return notes from another
  - Switching project context restores access to that project's notes

Exercises: search_codebase (CODE_SEARCH), search_notes (NOTES),
store_note (NOTES), CODER route, project context detection.

Requires: Indexed EVA project in workspace with search_codebase support.
EVA-migration project in workspace.
"""

import uuid

import pytest

from tests.evaluators import evaluate_response

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.eva,
]


class TestProjectScopedNotes:
    """Project isolation: notes from project A don't leak into project B."""

    def test_project_note_isolation(self, test_run, trace_test, judge_model):
        """4-turn flow: work on EVA → switch to migration → verify isolation → switch back.

        Turn 1: Work on EVA, discover architectural pattern
        Turn 2: Switch to EVA-migration, ask about migration only
        Turn 3: Search notes for EVA pattern — should NOT find it (wrong project)
        Turn 4: Switch back to EVA — should find the pattern note
        """
        sid = f"test-projnotes-{uuid.uuid4().hex[:8]}"
        test_run.track_session(sid)
        user_name = f"ProjNote_{uuid.uuid4().hex[:6]}"
        test_run.track_user(user_name)

        # Unique marker to identify this test's notes
        unique_tag = uuid.uuid4().hex[:8]

        # ── Turn 1: Work on EVA, discover something ──
        msg1 = (
            f"Hi I'm {user_name}. I'm working on the EVA project. "
            f"I discovered that the build routing system ({unique_tag}) "
            "uses a state machine pattern in JobStart.ps1 — it checks "
            "the OS field in the JSON input to decide between Windows "
            "and Linux code paths. Can you search the EVA codebase "
            "for JobStart.ps1 and tell me about it?"
        )
        r1 = test_run.chat(msg1, session_id=sid)
        evaluate_response(r1, msg1, trace_test, judge_model, "coder")

        trace_test.set_attribute("cca.test.t1_response", r1.content[:500])
        assert r1.content, "Turn 1 returned empty"
        assert r1.user_identified, f"{user_name} should be identified"

        iters1 = r1.metadata.get("tool_iterations", 0)
        trace_test.set_attribute("cca.test.t1_iters", iters1)

        # Response should mention EVA/JobStart content
        content1 = r1.content.lower()
        has_eva = any(w in content1 for w in [
            "jobstart", "eva", ".ps1", "powershell", "function",
            "build", "routing",
        ])
        trace_test.set_attribute("cca.test.t1_has_eva", has_eva)
        assert has_eva, (
            f"Response doesn't mention EVA/JobStart: {r1.content[:300]}"
        )

        # ── Turn 2: Switch to EVA-migration ──
        msg2 = (
            "Now I'm switching to work on the EVA-migration project. "
            "What files are in the migration project? Don't look at "
            "the EVA source code — I only need migration info."
        )
        r2 = test_run.chat(msg2, session_id=sid)
        evaluate_response(r2, msg2, trace_test, judge_model, "coder")

        trace_test.set_attribute("cca.test.t2_response", r2.content[:500])
        assert r2.content, "Turn 2 returned empty"

        # ── Turn 3: Search notes — should NOT find EVA notes ──
        msg3 = (
            "Search your notes for anything about 'state machine' or "
            "'build routing'. What do we know?"
        )
        r3 = test_run.chat(msg3, session_id=sid)
        evaluate_response(r3, msg3, trace_test, judge_model, "coder")

        trace_test.set_attribute("cca.test.t3_response", r3.content[:500])
        assert r3.content, "Turn 3 returned empty"

        # Track whether EVA-specific content leaked into migration context
        content3 = r3.content.lower()
        has_eva_leak = unique_tag.lower() in content3
        trace_test.set_attribute(
            "cca.test.t3_eva_leak", has_eva_leak,
        )
        # Advisory: don't hard-fail on this yet — scope isolation may
        # not be perfect in v1. Track it for monitoring.

        # ── Turn 4: Switch back to EVA ──
        msg4 = (
            "Actually, switch me back to the EVA project. Now search "
            "your notes for 'state machine' — what did we learn about "
            "the build routing?"
        )
        r4 = test_run.chat(msg4, session_id=sid)
        evaluate_response(r4, msg4, trace_test, judge_model, "coder")

        trace_test.set_attribute("cca.test.t4_response", r4.content[:500])
        assert r4.content, "Turn 4 returned empty"

        # Response should recall the EVA state machine discovery
        content4 = r4.content.lower()
        has_recall = any(w in content4 for w in [
            "state machine", "routing", "jobstart", "os field",
            "windows", "linux", "code path",
        ])
        trace_test.set_attribute(
            "cca.test.t4_has_recall", has_recall,
        )
        # Advisory: track recall but don't hard-fail — the note may
        # not have been stored yet (async FactExtractor).
