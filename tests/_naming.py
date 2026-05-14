"""Canonical test naming — one function, one source of truth.

The standard (docs/testing/writing-tests.md):
    test_<name>.py  ↔  <name>.replace("_", "-")

The FILE is the test. Function names inside the file are sub-checks of
the same test. Parametrized variants are different invocations of the
same test. Every system uses this same name: Phoenix project, GitLab CI
$RUN_TEST, TestDefinition.test_name, TestResult.ci_job_name, report
folder name.

If you find yourself normalizing a test name somewhere new, import
canonical_name() instead — don't reinvent it.
"""
from pathlib import Path


def canonical_name(target) -> str:
    """Return the canonical name for a test file or pytest node.

    Accepts a string path, a pathlib.Path, or a pytest node (anything
    exposing .fspath, .path, or .nodeid). Always derives from the FILE,
    never from the function name.

    Examples:
        canonical_name("tests/user/test_new_user_onboarding.py")
            -> "new-user-onboarding"
        canonical_name(request.node)  # pytest fixture argument
            -> file stem of the test, normalized
        canonical_name("tests/coder/test_api_lookup.py::TestX::test_y")
            -> "api-lookup"  (function names inside the file don't change the name)
    """
    if hasattr(target, "fspath"):
        p = Path(str(target.fspath))
    elif hasattr(target, "path"):
        p = Path(str(target.path))
    elif hasattr(target, "nodeid"):
        p = Path(str(target.nodeid).split("::", 1)[0])
    else:
        p = Path(str(target))
    stem = p.stem
    if stem.startswith("test_"):
        stem = stem[5:]
    return stem.replace("_", "-")
