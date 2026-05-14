"""Tests for confucius.server.code_intelligence.module_entry.

Synthetic per-language source strings exercise detect_module_body.
These tests must run inside the cca container (or any environment with
tree-sitter language grammars built) — the module_entry path through
tree_sitter_parser requires /usr/local/lib/tree-sitter-languages.so.
"""
from __future__ import annotations

import pytest


pytest.importorskip("tree_sitter", reason="needs tree-sitter compiled bindings")


# tree-sitter grammars are only present in container builds; local node5 venv
# can collect this file but can't load the .so. Skip the whole module there.
def _grammars_built() -> bool:
    import os
    return os.path.exists("/usr/local/lib/tree-sitter-languages.so")


pytestmark = pytest.mark.skipif(
    not _grammars_built(),
    reason="tree-sitter grammars not compiled (only present in test-runner / cca containers)",
)


def _call_names(entry):
    return [c["name"] for c in entry["calls"]]


# ── Python ────────────────────────────────────────────────────────────


class TestPython:
    def test_pure_function_defs_returns_none(self):
        """A file with only def + class, no module-level call → no ModuleEntry."""
        from confucius.server.code_intelligence.module_entry import detect_module_body
        src = (
            "def foo():\n"
            "    return 1\n"
            "\n"
            "class Bar:\n"
            "    def baz(self):\n"
            "        return 2\n"
        )
        assert detect_module_body("python", src, "/x/pure.py") is None

    def test_imports_only_returns_none(self):
        from confucius.server.code_intelligence.module_entry import detect_module_body
        src = (
            "import os\n"
            "import sys\n"
            "from typing import List\n"
        )
        assert detect_module_body("python", src, "/x/imports.py") is None

    def test_top_level_call_produces_module_entry(self):
        from confucius.server.code_intelligence.module_entry import detect_module_body
        src = (
            "import subprocess\n"
            "import sys\n"
            "\n"
            "def helper():\n"
            "    return 42\n"
            "\n"
            "subprocess.run(['python', 'other.py'])\n"
            "print('done')\n"
        )
        entry = detect_module_body("python", src, "/orchestrator.py")
        assert entry is not None
        assert entry["name"] == "__module__"
        assert entry["qualified_name"] == "/orchestrator.py::__module__"
        assert entry["is_module"] is True
        # helper() is inside a function — should NOT appear in top-level calls.
        names = _call_names(entry)
        assert any("subprocess.run" in n for n in names), names
        assert any("print" in n for n in names), names
        assert "helper" not in names

    def test_if_main_block_counts_as_executable(self):
        from confucius.server.code_intelligence.module_entry import detect_module_body
        src = (
            "def main():\n"
            "    pass\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )
        entry = detect_module_body("python", src, "/x/script.py")
        assert entry is not None
        assert "main" in _call_names(entry)


# ── PowerShell ───────────────────────────────────────────────────────


class TestPowerShell:
    def test_top_level_pipeline_produces_module_entry(self):
        """A PS file with NO named functions, just a body of commands —
        currently the indexer produces zero Function nodes. This is exactly
        the gap the feature closes."""
        from confucius.server.code_intelligence.module_entry import detect_module_body
        src = (
            "param($Name)\n"
            "Write-Host \"Hello $Name\"\n"
            "$items = Get-ChildItem .\n"
            "$items | ForEach-Object { Write-Output $_.Name }\n"
        )
        entry = detect_module_body("powershell", src, "/scripts/hello.ps1")
        assert entry is not None
        assert entry["name"] == "__module__"
        assert entry["qualified_name"] == "/scripts/hello.ps1::__module__"
        names = _call_names(entry)
        # Different PS grammars yield different surface forms; the
        # important invariant is that at least one of the command verbs
        # we wrote shows up as a top-level call.
        joined = " ".join(names).lower()
        assert ("write-host" in joined or "get-childitem" in joined or
                "foreach-object" in joined), names

    def test_function_only_file_returns_none(self):
        from confucius.server.code_intelligence.module_entry import detect_module_body
        src = (
            "function Get-Foo {\n"
            "    param($X)\n"
            "    Write-Host $X\n"
            "}\n"
            "\n"
            "function Set-Bar {\n"
            "    param($Y)\n"
            "    Set-Variable -Name Bar -Value $Y\n"
            "}\n"
        )
        assert detect_module_body("powershell", src, "/scripts/lib.ps1") is None


# ── Bash ─────────────────────────────────────────────────────────────


class TestBash:
    def test_top_level_command_produces_module_entry(self):
        from confucius.server.code_intelligence.module_entry import detect_module_body
        src = (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "\n"
            "echo 'starting'\n"
            "./other.sh\n"
            "ls -la\n"
        )
        entry = detect_module_body("bash", src, "/scripts/run.sh")
        assert entry is not None
        assert entry["name"] == "__module__"
        names = _call_names(entry)
        joined = " ".join(names).lower()
        # Bash's `command_name` extraction should pick at least one of these
        assert ("echo" in joined or "ls" in joined or "set" in joined), names

    def test_function_only_bash_returns_none(self):
        from confucius.server.code_intelligence.module_entry import detect_module_body
        src = (
            "do_thing() {\n"
            "    echo doing\n"
            "}\n"
            "\n"
            "function helper() {\n"
            "    echo helping\n"
            "}\n"
        )
        assert detect_module_body("bash", src, "/scripts/lib.sh") is None


# ── JavaScript / TypeScript ─────────────────────────────────────────


class TestJavaScript:
    def test_top_level_call_produces_module_entry(self):
        from confucius.server.code_intelligence.module_entry import detect_module_body
        src = (
            "const fs = require('fs');\n"
            "function helper(x) { return x + 1; }\n"
            "console.log('hi');\n"
            "fs.readFileSync('./other.js');\n"
        )
        entry = detect_module_body("javascript", src, "/orch.js")
        assert entry is not None
        names = _call_names(entry)
        assert any("console.log" in n for n in names), names

    def test_function_only_js_returns_none(self):
        from confucius.server.code_intelligence.module_entry import detect_module_body
        src = (
            "function foo() { return 1; }\n"
            "class Bar { baz() { return 2; } }\n"
        )
        assert detect_module_body("javascript", src, "/lib.js") is None


# ── Out-of-scope languages ─────────────────────────────────────────


class TestOutOfScope:
    @pytest.mark.parametrize("lang", ["go", "rust", "java", "c", "cpp"])
    def test_skipped_languages_always_none(self, lang):
        """Go / Rust / Java / C / C++ don't have meaningful module-body
        semantics; detect_module_body must return None even if the source
        contains executable statements at the top level (which is unusual
        but technically possible)."""
        from confucius.server.code_intelligence.module_entry import detect_module_body
        # Source content is irrelevant — short-circuit on language.
        assert detect_module_body(lang, "irrelevant", "/x") is None


class TestUnknownLanguage:
    def test_unknown_language_returns_none(self):
        from confucius.server.code_intelligence.module_entry import detect_module_body
        assert detect_module_body("zsh", "echo hi", "/x") is None
