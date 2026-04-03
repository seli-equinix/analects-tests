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
