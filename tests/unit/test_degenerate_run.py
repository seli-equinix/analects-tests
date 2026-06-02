"""Coverage-independent degenerate-run detection (api-lookup hardening).

Regression guard for the api-lookup failure: the model emitted a Python file
whose line 62 was a degenerate repeated ternary (`… if X else Y if X else Y …`
×10) — a ~600-char broken section inside a ~2500-char file (~24% coverage).
The coverage-gated `is_degenerate` (50% gate) MISSED it, so the broken file was
written and the agent looped. `find_degenerate_run` catches a degenerate
exact-repeated run regardless of overall coverage, while NOT false-positiving on
legitimate code (which never has a long EXACT-repeated semantic unit).

Pure-Python (only confucius.core.quality). Runs on node5 AND in CI.
"""
from __future__ import annotations

from confucius.core.quality import find_degenerate_run, is_degenerate
from confucius.core.quality._detectors import DEGEN_RUN_MIN_CHARS


_TERNARY = (
    "    num_sockets = max(1, cpu_count) "
    + "if num_sockets > num_cores_per_socket else num_sockets * num_cores_per_socket " * 9
    + "if num_sockets >\n"
)
_VALID = (
    "import os\nfrom typing import Optional\n\n"
    "def create_vm(cpu_count, num_sockets, num_cores_per_socket, name, image):\n"
    '    """Create a Nutanix VM."""\n'
    "    client = build_client(os.environ['HOST'])\n"
    "    vm = VmConfig(name=name, sockets=num_sockets, cores=num_cores_per_socket)\n"
    "    return tasks_api.create(vm).ext_id\n"
)


class TestCatchesDegeneration:
    def test_repeated_ternary_run(self):
        f = _VALID + _TERNARY + _VALID
        d = find_degenerate_run(f)
        assert d is not None, "degenerate repeated ternary must be detected"
        assert "num_cores_per_socket" in d[0], d

    def test_coverage_gate_misses_it_but_run_detector_catches(self):
        # The whole point: is_degenerate (50% coverage) misses a minority run;
        # find_degenerate_run catches it.
        f = _VALID + _TERNARY + _VALID
        assert is_degenerate(f) is False
        assert find_degenerate_run(f) is not None

    def test_spew(self):
        assert find_degenerate_run("etcetera etcetera etcetera " * 30) is not None


class TestNoFalsePositives:
    def test_valid_varied_code(self):
        assert find_degenerate_run(_VALID * 3) is None

    def test_varying_dict_entries(self):
        # repeated-but-VARYING lines (different keys/values) are NOT an exact run
        code = "cfg = {\n" + "".join(f'    "k_{i}": "v_{i}",\n' for i in range(40)) + "}\n"
        assert find_degenerate_run(code) is None

    def test_short_text(self):
        assert find_degenerate_run("x" * (DEGEN_RUN_MIN_CHARS - 1)) is None

    def test_whitespace_run_ignored(self):
        # pure indentation / blank lines are not degeneration
        assert find_degenerate_run("\n    " * 60) is None
        assert find_degenerate_run(" " * 500) is None
