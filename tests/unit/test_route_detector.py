"""Unit tests for the routes-phase decorator regexes.

Locks in the bugfix that switched the route detector from scanning
``f.signature`` (which never carries decorators) to scanning
``f.decorators`` (the persisted list property). The regexes have to
handle the multi-line decorator strings tree-sitter actually produces
— ``@app.post(`` followed by newlines and arguments before the closing
paren — not just the textbook single-line shape.

Without these tests, a future refactor to either the parser's decorator
extraction or the routes-phase regex would silently regress route
detection back to zero.
"""
from __future__ import annotations

import pytest

from confucius.server.code_intelligence.pipeline.phases.routes import (
    FASTAPI_PATTERN,
    FLASK_PATTERN,
    NESTJS_PATTERN,
    SPRING_PATTERN,
)


class TestFastAPIPattern:
    """Real FastAPI decorator shapes the parser produces in the wild."""

    def test_single_line_get_root(self):
        decos = ['@app.get("/")']
        matches = [m.groups() for d in decos for m in FASTAPI_PATTERN.finditer(d)]
        assert matches == [("get", "/")]

    def test_multiline_post_with_kwargs(self):
        # This is exactly the shape captured from
        # /workspace/fastapi_user_app/main.py — the @app.post decorator
        # spans 6 lines because it has response_model / status_code /
        # summary kwargs. Tree-sitter preserves the newlines.
        deco = (
            '@app.post( \n    "/users", \n    response_model=UserResponse, \n'
            '    status_code=status.HTTP_201_CREATED, \n'
            '    summary="Create a new user" \n)'
        )
        m = FASTAPI_PATTERN.search(deco)
        assert m is not None
        assert m.group(1) == "post"
        assert m.group(2) == "/users"

    def test_router_variant(self):
        # APIRouter is the other common FastAPI entry point.
        deco = '@router.put("/sessions/{id}")'
        m = FASTAPI_PATTERN.search(deco)
        assert m is not None
        assert m.group(1) == "put"
        assert m.group(2) == "/sessions/{id}"

    def test_does_not_match_non_route_decorator(self):
        for d in ["@asynccontextmanager", "@property", "@staticmethod",
                  "@dataclass", "@functools.wraps(fn)"]:
            assert FASTAPI_PATTERN.search(d) is None, d

    def test_all_methods_supported(self):
        for verb in ("get", "post", "put", "delete", "patch", "head", "options"):
            d = f'@app.{verb}("/x")'
            assert FASTAPI_PATTERN.search(d) is not None, verb


class TestFlaskPattern:
    """Flask covers @app.route(...) and the @blueprint/@bp/@router shapes."""

    def test_app_route(self):
        m = FLASK_PATTERN.search('@app.route("/legacy", methods=["GET"])')
        assert m is not None
        assert m.group(1) == "route"
        assert m.group(2) == "/legacy"

    def test_blueprint(self):
        m = FLASK_PATTERN.search('@bp.get("/v1/health")')
        assert m is not None
        assert m.group(1) == "get"
        assert m.group(2) == "/v1/health"

    def test_no_match_on_method_calls(self):
        # `app.route(...)` without the leading `@` is a runtime call,
        # not a decorator — must not be detected as a route.
        assert FLASK_PATTERN.search('app.route("/no")') is None


class TestNestJSPattern:
    def test_get_decorator(self):
        m = NESTJS_PATTERN.search('@Get("/users")')
        assert m is not None
        assert m.group(1) == "Get"
        assert m.group(2) == "/users"

    def test_get_no_path(self):
        # NestJS allows decorators without a path — root of the controller.
        m = NESTJS_PATTERN.search("@Get()")
        # The current regex requires a quoted path; bare @Get() returns no
        # match, which the routes-phase treats as "no route". Documented
        # here so future relaxations don't accidentally over-match.
        assert m is None


class TestSpringPattern:
    def test_getmapping_value(self):
        m = SPRING_PATTERN.search('@GetMapping(value = "/users")')
        assert m is not None
        assert m.group(1) == "GetMapping"
        assert m.group(2) == "/users"

    def test_postmapping_short(self):
        m = SPRING_PATTERN.search('@PostMapping("/login")')
        assert m is not None
        assert m.group(1) == "PostMapping"
        assert m.group(2) == "/login"


class TestDetectorBehaviorAcrossAllRegexes:
    """Cross-cutting checks: a decorator string must be matched by exactly
    the framework patterns it belongs to. Regression catcher for someone
    accidentally widening one of the regexes."""

    @pytest.mark.parametrize("deco,expected", [
        ('@app.get("/")', {"flask", "fastapi"}),  # both regexes share the @app.{verb} shape
        ('@app.route("/x")', {"flask"}),
        ('@router.post("/x")', {"flask", "fastapi"}),
        ('@Get("/x")', {"nestjs"}),
        ('@GetMapping("/x")', {"spring"}),
        ('@asynccontextmanager', set()),
        ('@property', set()),
    ])
    def test_decorator_dispatches_to_expected_frameworks(self, deco, expected):
        hits: set[str] = set()
        for pat, name in (
            (FLASK_PATTERN, "flask"),
            (FASTAPI_PATTERN, "fastapi"),
            (NESTJS_PATTERN, "nestjs"),
            (SPRING_PATTERN, "spring"),
        ):
            if pat.search(deco):
                hits.add(name)
        assert hits == expected, f"{deco!r} → {hits} (expected {expected})"
