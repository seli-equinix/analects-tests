"""Unit tests for bash file-write interception guardrails.

Tests that CommandLineExtension blocks bash commands that try to create/edit
files (heredoc, redirect, python -c write, tee) and redirects to str_replace_editor.
"""

import pytest

from confucius.orchestrator.extensions.command_line.base import _check_file_write


class TestFileWriteDetection:
    """Test _check_file_write pattern matching."""

    # ── Should be BLOCKED ──

    @pytest.mark.parametrize(
        "command,label",
        [
            # Heredoc patterns
            ("cat << 'EOF' > /workspace/test.py\nprint('hello')\nEOF", "heredoc"),
            ("cat <<EOF > /tmp/script.sh\necho hi\nEOF", "heredoc"),
            ("cat <<'SCRIPT' > /workspace/deploy.sh\nset -e\nSCRIPT", "heredoc"),
            # Echo/printf redirect
            ("echo 'hello world' > /workspace/test.txt", "redirect to file"),
            ('echo "line1\nline2" > /workspace/output.py', "redirect to file"),
            ("printf '%s\\n' hello > /workspace/test.txt", "redirect to file"),
            # Python inline file write
            ("python3 -c \"with open('/workspace/test.py', 'w') as f: f.write('hello')\"", "python inline file write"),
            ("python -c \"from pathlib import Path; Path('/workspace/x.py').write_text('hi')\"", "python inline file write"),
            # Tee
            ("echo hello | tee /workspace/output.txt", "tee to file"),
            ("tee /workspace/config.yaml", "tee to file"),
        ],
    )
    def test_blocks_file_write(self, command: str, label: str):
        result = _check_file_write(command)
        assert result is not None, f"Expected '{command}' to be blocked as '{label}'"
        assert "str_replace_editor" in result
        assert "BLOCKED" in result

    # ── Should be ALLOWED ──

    @pytest.mark.parametrize(
        "command",
        [
            # Normal commands
            "python3 /workspace/test.py",
            "ls -la /workspace",
            "grep -r 'pattern' /workspace",
            "git status",
            "cat /workspace/test.py",  # Reading, not writing
            "head -20 /workspace/test.py",
            # /dev/ targets should be excluded
            "echo 'debug' > /dev/null",
            "tee /dev/stderr",
            # python3 -c without file write
            "python3 -c \"print('hello')\"",
            "python3 -c \"import sys; print(sys.version)\"",
            # Echo without redirect
            "echo hello",
            "echo 'test' | grep test",
        ],
    )
    def test_allows_safe_commands(self, command: str):
        result = _check_file_write(command)
        assert result is None, f"Expected '{command}' to be allowed, got: {result}"

    def test_rejection_message_has_examples(self):
        result = _check_file_write("cat << EOF > /workspace/test.py\nEOF")
        assert result is not None
        assert '"command": "create"' in result
        assert '"path":' in result
        assert '"file_text":' in result


class TestRedirectFalsePositiveRegression:
    """Regression tests for the redirect-to-file pattern.

    Pinned by an incident on 2026-04-11 where the runtime config DB shipped
    with `>\\s*\\S+` as the redirect pattern, which matched ANY `>` followed
    by anything non-whitespace. That false-positived on `2>/dev/null` (stderr
    discard), broke the api-lookup test, and silently degraded every coder
    test that ever piped stderr to /dev/null.

    The fix: anchor the pattern to a word boundary so the leading file
    descriptor (`2>`, `1>`, `&>`) excludes the match, AND keep the existing
    `(?!/dev/)` negative lookahead so /dev/null and friends pass through.

    These tests pin down the boundary between "looks like a file write" and
    "looks like an fd discard" so a future seed update can't quietly slide
    the regex back to something over-broad without a CI failure.
    """

    # ── False positives that USED to break the api-lookup test ──

    @pytest.mark.parametrize(
        "command",
        [
            # Stderr → /dev/null discard. The leading `2` makes this an
            # fd-2 redirect, NOT a stdout redirect to a file.
            "rm /workspace/script.py 2>/dev/null",
            "rm /workspace/script.py 2>/dev/null; ls /workspace/*.py",
            "curl -s api.example.com 2>/dev/null",
            "git fetch origin main 2>/dev/null",
            # Stdout → /dev/null discard. The /dev/ negative lookahead
            # exempts these.
            "pytest -v >/dev/null",
            "python3 script.py >/dev/null",
            # Both fds discarded.
            "python3 script.py 2>/dev/null >/dev/null",
            "command >/dev/null 2>&1",
            # Stderr append to a log file. The `2` makes it fd-2.
            "grep foo file 2>>/tmp/log",
            # File descriptor merge. `>&N` is not a file write.
            "make 2>&1 | tee /dev/null",
            "echo hi >&2",
            "command 2>&1",
            # Process substitution uses `<(...)` not `>` so it shouldn't match.
            "diff <(sort a) <(sort b)",
            # The exact two commands from incident report P5849:
            "rm /workspace/script.py 2>/dev/null; cat /workspace/test.py",
            "rm -f /workspace/foo.txt 2>/dev/null && ls -la /workspace",
        ],
    )
    def test_stderr_redirects_are_not_blocked(self, command: str):
        """Stderr/stdout discards and fd merges must NOT trigger the
        file-write interceptor."""
        result = _check_file_write(command)
        assert result is None, (
            f"FALSE POSITIVE: {command!r} was blocked as a file write but "
            f"it's only redirecting a file descriptor, not creating a file. "
            f"Rejection: {result}"
        )

    # ── True positives that MUST still be caught ──

    @pytest.mark.parametrize(
        "command",
        [
            # Bare stdout → file (no leading echo/printf)
            "cat data.json > /workspace/copy.json",
            "grep -r pattern dir/ > /tmp/matches.txt",
            "curl https://example.com > /tmp/page.html",
            "command > /tmp/result",
            # Append to file
            "echo content >> /tmp/log.txt",
            "python3 script.py >> /var/log/app.log",
            # Subshell write
            "(echo a; echo b) > /tmp/both.txt",
            # After semicolon
            "rm a.txt; echo done > /workspace/marker",
            # Stdout to relative path
            "echo hi > local.txt",
        ],
    )
    def test_real_writes_still_blocked(self, command: str):
        """Genuine file-creation patterns must still be caught."""
        result = _check_file_write(command)
        assert result is not None, (
            f"FALSE NEGATIVE: {command!r} writes a file but the "
            f"interceptor allowed it through. Pattern needs tightening."
        )
        assert "BLOCKED" in result
        assert "str_replace_editor" in result


class TestPatternConsistency:
    """The seed defaults in the runtime_config package and the code-level
    fallbacks in command_line/base.py must stay in sync. The label
    dictionary uses literal pattern strings as keys, so a drift between
    the two would cause the label lookup to silently fall through to
    "file write" — which makes operator debugging much harder.

    This test catches that drift on every CI run.
    """

    def test_seed_and_code_default_patterns_match(self):
        """The patterns in the seed file MUST be byte-for-byte identical
        to the code-level _DEFAULT_FILE_WRITE_PATTERNS list."""
        from confucius.orchestrator.extensions.command_line.base import (
            _DEFAULT_FILE_WRITE_PATTERNS,
            _PATTERN_LABELS,
        )
        from confucius.server.seeds.runtime_config.extensions_shell import (
            EXTENSION_BASH,
        )

        seed_patterns = EXTENSION_BASH["file_write_patterns"].value
        assert list(seed_patterns) == list(_DEFAULT_FILE_WRITE_PATTERNS), (
            "Seed patterns drifted from code defaults — they MUST stay "
            "identical so _PATTERN_LABELS can resolve labels for both. "
            f"\nseed: {list(seed_patterns)}"
            f"\ncode: {list(_DEFAULT_FILE_WRITE_PATTERNS)}"
        )

    def test_every_default_pattern_has_a_label(self):
        """Every pattern in the code default list must have a corresponding
        entry in _PATTERN_LABELS, otherwise the rejection message uses the
        generic 'file write' fallback instead of the specific label."""
        from confucius.orchestrator.extensions.command_line.base import (
            _DEFAULT_FILE_WRITE_PATTERNS,
            _PATTERN_LABELS,
        )

        for pat in _DEFAULT_FILE_WRITE_PATTERNS:
            assert pat in _PATTERN_LABELS, (
                f"Pattern {pat!r} has no entry in _PATTERN_LABELS. "
                f"Add it so error messages stay specific."
            )
