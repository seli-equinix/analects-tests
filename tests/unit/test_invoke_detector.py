"""Tests for confucius.server.code_intelligence.invoke_detector.

Pure-regex per-language patterns; no tree-sitter or memgraph dependency.
These tests run in any environment with the cca source on sys.path.
"""
from __future__ import annotations

import pytest


def _kinds(edges):
    return [e.kind for e in edges]


def _targets(edges):
    return [e.target_path_hint for e in edges]


# ── Python ────────────────────────────────────────────────────────────


class TestPython:
    def test_subprocess_run_list(self):
        from confucius.server.code_intelligence.invoke_detector import detect_cross_file_invocations
        src = "subprocess.run(['python', 'step_a.py'])\n"
        edges = detect_cross_file_invocations("python", src, "/x.py")
        assert len(edges) == 1
        assert edges[0].kind == "subprocess"
        assert edges[0].target_path_hint == "step_a.py"

    def test_subprocess_run_string(self):
        from confucius.server.code_intelligence.invoke_detector import detect_cross_file_invocations
        src = "subprocess.run('python /tmp/foo.py --x 1')\n"
        edges = detect_cross_file_invocations("python", src, "/x.py")
        assert any(e.target_path_hint == "/tmp/foo.py" for e in edges)

    def test_subprocess_call_check_call(self):
        from confucius.server.code_intelligence.invoke_detector import detect_cross_file_invocations
        src = (
            "subprocess.call(['python', 'a.py'])\n"
            "subprocess.check_call(['python', 'b.py'])\n"
            "subprocess.check_output(['python', 'c.py'])\n"
        )
        edges = detect_cross_file_invocations("python", src, "/x.py")
        targets = _targets(edges)
        assert "a.py" in targets
        assert "b.py" in targets
        assert "c.py" in targets

    def test_os_system(self):
        from confucius.server.code_intelligence.invoke_detector import detect_cross_file_invocations
        src = "os.system('python helper.py')\n"
        edges = detect_cross_file_invocations("python", src, "/x.py")
        assert any(e.kind == "shell" for e in edges)
        assert any(e.target_path_hint == "helper.py" for e in edges)

    def test_runpy_run_path(self):
        from confucius.server.code_intelligence.invoke_detector import detect_cross_file_invocations
        src = "runpy.run_path('scripts/migrate.py')\n"
        edges = detect_cross_file_invocations("python", src, "/x.py")
        assert any(e.kind == "exec" for e in edges)

    def test_exec_open_read(self):
        from confucius.server.code_intelligence.invoke_detector import detect_cross_file_invocations
        src = 'exec(open("setup.py").read())\n'
        edges = detect_cross_file_invocations("python", src, "/x.py")
        assert any(e.target_path_hint == "setup.py" for e in edges)

    def test_no_match_pure_function_call(self):
        """A plain function call (no file path) shouldn't produce edges."""
        from confucius.server.code_intelligence.invoke_detector import detect_cross_file_invocations
        src = "result = some_helper(1, 2, 3)\n"
        edges = detect_cross_file_invocations("python", src, "/x.py")
        assert edges == []


# ── PowerShell ────────────────────────────────────────────────────────


class TestPowerShell:
    def test_dot_slash_invocation(self):
        from confucius.server.code_intelligence.invoke_detector import detect_cross_file_invocations
        src = ".\\helper.ps1 -Name foo\n"
        edges = detect_cross_file_invocations("powershell", src, "/run.ps1")
        assert any(e.target_path_hint == "helper.ps1" for e in edges)

    def test_call_operator(self):
        from confucius.server.code_intelligence.invoke_detector import detect_cross_file_invocations
        src = "& 'deploy.ps1' -Env prod\n"
        edges = detect_cross_file_invocations("powershell", src, "/run.ps1")
        assert any("deploy.ps1" in e.target_path_hint for e in edges)

    def test_dot_source(self):
        from confucius.server.code_intelligence.invoke_detector import detect_cross_file_invocations
        src = ". .\\common.ps1\n"
        edges = detect_cross_file_invocations("powershell", src, "/run.ps1")
        # Dedup keeps the dot_source kind (more semantically specific).
        assert len(edges) == 1
        assert edges[0].kind == "dot_source"
        assert edges[0].target_path_hint == "common.ps1"

    def test_start_process_file(self):
        from confucius.server.code_intelligence.invoke_detector import detect_cross_file_invocations
        src = "Start-Process pwsh -File 'build.ps1' -NoNewWindow\n"
        edges = detect_cross_file_invocations("powershell", src, "/run.ps1")
        assert any(e.kind == "subprocess" and "build.ps1" in e.target_path_hint for e in edges)

    def test_invoke_expression_is_dynamic(self):
        from confucius.server.code_intelligence.invoke_detector import detect_cross_file_invocations
        src = "Invoke-Expression $cmd\n"
        edges = detect_cross_file_invocations("powershell", src, "/run.ps1")
        assert any(e.kind == "dynamic" for e in edges)


# ── Bash ──────────────────────────────────────────────────────────────


class TestBash:
    def test_interpreter_invocation(self):
        from confucius.server.code_intelligence.invoke_detector import detect_cross_file_invocations
        src = "bash setup.sh\nsh init.sh\nzsh post.sh\n"
        edges = detect_cross_file_invocations("bash", src, "/run.sh")
        targets = _targets(edges)
        assert "setup.sh" in targets
        assert "init.sh" in targets
        assert "post.sh" in targets

    def test_source(self):
        from confucius.server.code_intelligence.invoke_detector import detect_cross_file_invocations
        src = "source ./lib/common.sh\n"
        edges = detect_cross_file_invocations("bash", src, "/run.sh")
        assert any(e.kind == "source" for e in edges)

    def test_dot_alias_source(self):
        from confucius.server.code_intelligence.invoke_detector import detect_cross_file_invocations
        src = ". ./helper.sh\n"
        edges = detect_cross_file_invocations("bash", src, "/run.sh")
        assert any(e.kind == "source" for e in edges)

    def test_exec(self):
        from confucius.server.code_intelligence.invoke_detector import detect_cross_file_invocations
        src = "exec other.sh\n"
        edges = detect_cross_file_invocations("bash", src, "/run.sh")
        assert any(e.kind == "exec" for e in edges)


# ── JavaScript / TypeScript ──────────────────────────────────────────


class TestJavaScript:
    def test_child_process_exec(self):
        from confucius.server.code_intelligence.invoke_detector import detect_cross_file_invocations
        src = "child_process.exec('node worker.js')\n"
        edges = detect_cross_file_invocations("javascript", src, "/app.js")
        assert any(e.kind == "subprocess" and "worker.js" in e.target_path_hint for e in edges)

    def test_child_process_spawn(self):
        from confucius.server.code_intelligence.invoke_detector import detect_cross_file_invocations
        src = "child_process.spawn('node', ['server.js'])\n"
        edges = detect_cross_file_invocations("javascript", src, "/app.js")
        # spawn-with-list isn't supported by our regex (requires "node " prefix
        # within a single string), so this is an explicit no-match — captured
        # to document the limitation.
        # (No assert — just verifying no crash.)
        _ = edges

    def test_require_relative(self):
        from confucius.server.code_intelligence.invoke_detector import detect_cross_file_invocations
        src = "const init = require('./init.js');\n"
        edges = detect_cross_file_invocations("javascript", src, "/app.js")
        assert any(e.kind == "require" for e in edges)

    def test_import_side_effect(self):
        from confucius.server.code_intelligence.invoke_detector import detect_cross_file_invocations
        src = "import './side_effects';\n"
        edges = detect_cross_file_invocations("javascript", src, "/app.js")
        assert any(e.kind == "require" for e in edges)

    def test_typescript_dispatches_to_js_scanner(self):
        from confucius.server.code_intelligence.invoke_detector import detect_cross_file_invocations
        src = "child_process.exec('node worker.js')\n"
        edges = detect_cross_file_invocations("typescript", src, "/app.ts")
        assert any(e.kind == "subprocess" for e in edges)


# ── Out-of-scope languages ───────────────────────────────────────────


class TestUnknownLanguage:
    @pytest.mark.parametrize("lang", ["go", "rust", "java", "c", "cpp", "ruby", "php", "perl"])
    def test_no_edges_for_unsupported_language(self, lang):
        from confucius.server.code_intelligence.invoke_detector import detect_cross_file_invocations
        src = "anything goes here"
        assert detect_cross_file_invocations(lang, src, "/x") == []


# ── Path resolution ──────────────────────────────────────────────────


class TestResolveInvocationTarget:
    def test_absolute_path_match(self):
        from confucius.server.code_intelligence.invoke_detector import (
            InvocationEdge, resolve_invocation_target,
        )
        edge = InvocationEdge(
            source_file="/proj/orchestrator.py",
            target_path_hint="/proj/step_a.py",
            kind="subprocess", line=5,
        )
        known = {"/proj/orchestrator.py", "/proj/step_a.py"}
        assert resolve_invocation_target(edge, known) == "/proj/step_a.py"

    def test_relative_path_resolution(self):
        from confucius.server.code_intelligence.invoke_detector import (
            InvocationEdge, resolve_invocation_target,
        )
        edge = InvocationEdge(
            source_file="/proj/orchestrator.py",
            target_path_hint="step_a.py",
            kind="subprocess", line=5,
        )
        known = {"/proj/orchestrator.py", "/proj/step_a.py"}
        assert resolve_invocation_target(edge, known) == "/proj/step_a.py"

    def test_relative_path_with_subdir(self):
        from confucius.server.code_intelligence.invoke_detector import (
            InvocationEdge, resolve_invocation_target,
        )
        edge = InvocationEdge(
            source_file="/proj/orchestrator.py",
            target_path_hint="sub/step_b.py",
            kind="subprocess", line=5,
        )
        known = {"/proj/orchestrator.py", "/proj/sub/step_b.py"}
        assert resolve_invocation_target(edge, known) == "/proj/sub/step_b.py"

    def test_basename_unique_match(self):
        from confucius.server.code_intelligence.invoke_detector import (
            InvocationEdge, resolve_invocation_target,
        )
        edge = InvocationEdge(
            source_file="/proj/orchestrator.py",
            target_path_hint="unique.py",
            kind="subprocess", line=5,
        )
        known = {"/proj/orchestrator.py", "/proj/deep/sub/unique.py"}
        assert resolve_invocation_target(edge, known) == "/proj/deep/sub/unique.py"

    def test_basename_ambiguous_returns_none(self):
        from confucius.server.code_intelligence.invoke_detector import (
            InvocationEdge, resolve_invocation_target,
        )
        edge = InvocationEdge(
            source_file="/proj/orchestrator.py",
            target_path_hint="duplicate.py",
            kind="subprocess", line=5,
        )
        known = {"/proj/a/duplicate.py", "/proj/b/duplicate.py"}
        assert resolve_invocation_target(edge, known) is None

    def test_no_match_returns_none(self):
        from confucius.server.code_intelligence.invoke_detector import (
            InvocationEdge, resolve_invocation_target,
        )
        edge = InvocationEdge(
            source_file="/proj/orchestrator.py",
            target_path_hint="nonexistent.py",
            kind="subprocess", line=5,
        )
        known = {"/proj/other.py"}
        assert resolve_invocation_target(edge, known) is None

    def test_dynamic_kind_unresolvable(self):
        from confucius.server.code_intelligence.invoke_detector import (
            InvocationEdge, resolve_invocation_target,
        )
        edge = InvocationEdge(
            source_file="/proj/x.ps1",
            target_path_hint="",
            kind="dynamic", line=5,
        )
        assert resolve_invocation_target(edge, {"/proj/x.ps1"}) is None
