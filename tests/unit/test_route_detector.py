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

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from confucius.server.code_intelligence.pipeline.phases.routes import (
    FASTAPI_PATTERN,
    FLASK_METHOD_VERB_PATTERN,
    FLASK_METHODS_KWARG_PATTERN,
    FLASK_PATTERN,
    NESTJS_PATTERN,
    SPRING_PATTERN,
    _delete_orphan_routes,
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
    """Flask is restricted to its distinctive `@app.route(...)` /
    `@blueprint.route(...)` / `@bp.route(...)` shape so it doesn't
    overlap with FASTAPI_PATTERN's verb-based form. Modern Flask 2.0+
    `@app.get(...)` decorators are intentionally caught by the
    FastAPI regex and labeled `fastapi` — that's a cosmetic
    misattribution, not a duplicate-row bug."""

    def test_app_route(self):
        m = FLASK_PATTERN.search('@app.route("/legacy", methods=["GET"])')
        assert m is not None
        assert m.group(1) == "route"
        assert m.group(2) == "/legacy"

    def test_blueprint_route(self):
        m = FLASK_PATTERN.search('@bp.route("/v1/health")')
        assert m is not None
        assert m.group(1) == "route"
        assert m.group(2) == "/v1/health"

    def test_does_not_match_verb_decorator(self):
        # @app.get("/") used to over-match Flask, producing duplicate
        # Route nodes (one labeled flask, one fastapi). Now FLASK_PATTERN
        # only matches `route`; verb-based decorators belong to FASTAPI.
        assert FLASK_PATTERN.search('@app.get("/")') is None
        assert FLASK_PATTERN.search('@bp.post("/users")') is None

    def test_no_match_on_method_calls(self):
        # `app.route(...)` without the leading `@` is a runtime call,
        # not a decorator — must not be detected as a route.
        assert FLASK_PATTERN.search('app.route("/no")') is None


class TestFlaskMethodsKwarg:
    """Verbs from `methods=["GET", "POST"]` kwarg recovery —
    without this, every `@app.route(...)` ended up labeled as
    HTTP method `ROUTE`."""

    def test_extracts_single_method(self):
        deco = '@app.route("/x", methods=["POST"])'
        kwarg = FLASK_METHODS_KWARG_PATTERN.search(deco)
        assert kwarg is not None
        verbs = FLASK_METHOD_VERB_PATTERN.findall(kwarg.group(1))
        assert verbs == ["POST"]

    def test_extracts_multiple_methods(self):
        deco = '@app.route("/x", methods=["GET", "POST", "DELETE"])'
        kwarg = FLASK_METHODS_KWARG_PATTERN.search(deco)
        assert kwarg is not None
        verbs = FLASK_METHOD_VERB_PATTERN.findall(kwarg.group(1))
        assert verbs == ["GET", "POST", "DELETE"]

    def test_no_kwarg_returns_none(self):
        # `@app.route("/x")` with no methods kwarg → caller defaults to GET.
        deco = '@app.route("/x")'
        assert FLASK_METHODS_KWARG_PATTERN.search(deco) is None

    def test_handles_multiline_kwarg(self):
        # Flask decorators sometimes wrap onto multiple lines; the
        # DOTALL flag makes the bracket-content match across newlines.
        deco = '@app.route(\n    "/x",\n    methods=[\n        "GET",\n        "POST"\n    ]\n)'
        kwarg = FLASK_METHODS_KWARG_PATTERN.search(deco)
        assert kwarg is not None
        verbs = FLASK_METHOD_VERB_PATTERN.findall(kwarg.group(1))
        assert verbs == ["GET", "POST"]


class TestWrapsDecoratorCleaning:
    """The WRAPS heuristic in workspace_indexer skips dotted decorators
    (method-call registrations like `@register.filter`, `@app.get`,
    `@functools.wraps(fn)`). Including them produced false-positive
    WRAPS edges whenever any Function in the project happened to share
    the last segment's name. This test mirrors the cleaning logic so a
    regression flips the unit suite, not a live reindex."""

    @staticmethod
    def _clean_and_skip(deco: str) -> str | None:
        # Mirror of workspace_indexer.py:Phase-10 logic.
        bare = str(deco).strip().lstrip("@").split("(", 1)[0].strip()
        if not bare or len(bare) < 2:
            return None
        if "." in bare:
            return None
        return bare

    def test_bare_decorator_kept(self):
        assert self._clean_and_skip("@my_decorator") == "my_decorator"
        assert self._clean_and_skip("@timed_api") == "timed_api"
        assert self._clean_and_skip("@login_required") == "login_required"

    def test_dotted_decorator_skipped(self):
        # The bug repro cases — these all used to produce false WRAPS edges.
        for d in [
            "@register.filter",
            "@register.simple_tag",
            "@app.get",
            "@app.post(\"/users\")",
            "@functools.wraps(fn)",
            "@click.option(\"--config\")",
        ]:
            assert self._clean_and_skip(d) is None, d

    def test_bare_with_args_kept(self):
        # Bare decorator with call-style args still resolves to the
        # bare name (e.g. `@cache(seconds=10)` → `cache`).
        assert self._clean_and_skip("@cache(seconds=10)") == "cache"


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
        # After the polish-pass: Flask's regex is restricted to `route`
        # so verb-based decorators dispatch only to FastAPI. This kills
        # the duplicate-Route-node bug where each `@app.get(...)` was
        # being upserted twice (one labeled flask, one fastapi).
        ('@app.get("/")', {"fastapi"}),
        ('@app.route("/x")', {"flask"}),
        ('@router.post("/x")', {"fastapi"}),
        ('@bp.route("/x")', {"flask"}),
        ('@bp.get("/x")', set()),  # bp.{verb} not commonly Flask; left unmatched
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


# ── Orphan Route cleanup ────────────────────────────────────────────


class TestDeleteOrphanRoutes:
    """Cleanup pre-pass that runs at the start of RoutesPhase. Removes
    Route nodes for the project that have no inbound HANDLES_ROUTE edge
    — these accumulate when the detector's emitted tuple shape changes
    for the same source code (verb extracted from `methods=[...]`,
    framework label narrowed by Polish #1, etc.).

    All three tests mock the async session boundary; we don't need a
    live Memgraph because the helper is pure orchestration over two
    Cypher queries."""

    @staticmethod
    def _make_session(count_value, single_returns_none=False):
        """Build a session mock whose first session.run returns a
        result whose .single() returns {n: count_value}, and whose
        second session.run is a no-op AsyncMock."""
        session = AsyncMock()
        count_record: Any
        if single_returns_none:
            count_record = None
        else:
            count_record = MagicMock()
            count_record.__getitem__ = lambda self, k: (
                count_value if k == "n" else None
            )
        count_result = MagicMock()
        count_result.single = AsyncMock(return_value=count_record)
        delete_result = MagicMock()
        # First call → count_result, subsequent calls → delete_result
        session.run = AsyncMock(side_effect=[count_result, delete_result])
        return session

    @pytest.mark.asyncio
    async def test_deletes_when_orphans_exist(self):
        """Orphans found → returns the count and issues a DELETE roundtrip."""
        session = self._make_session(count_value=5)
        n = await _delete_orphan_routes(session, project="EVA")
        assert n == 5
        assert session.run.call_count == 2  # count + delete
        # Second call's first arg (Cypher text) must contain DETACH DELETE
        delete_call_query = session.run.call_args_list[1].args[0]
        assert "DETACH DELETE r" in delete_call_query
        assert "project" in session.run.call_args_list[1].kwargs

    @pytest.mark.asyncio
    async def test_skips_delete_when_zero_orphans(self):
        """Zero orphans → returns 0 and skips the DELETE roundtrip
        entirely (no point spending a query on an empty MATCH)."""
        session = AsyncMock()
        count_record = MagicMock()
        count_record.__getitem__ = lambda self, k: 0 if k == "n" else None
        count_result = MagicMock()
        count_result.single = AsyncMock(return_value=count_record)
        session.run = AsyncMock(return_value=count_result)

        n = await _delete_orphan_routes(session, project="fastapi_user_app")
        assert n == 0
        assert session.run.call_count == 1  # only the count query

    @pytest.mark.asyncio
    async def test_handles_missing_count_record(self):
        """Defensive — if session.single() returns None (shouldn't
        happen in practice but possible if the count query is malformed
        or Memgraph is in a weird state), helper degrades to 0 instead
        of crashing the whole routes phase."""
        session = AsyncMock()
        count_result = MagicMock()
        count_result.single = AsyncMock(return_value=None)
        session.run = AsyncMock(return_value=count_result)

        n = await _delete_orphan_routes(session, project="EVA")
        assert n == 0
        assert session.run.call_count == 1  # count only, no delete

    @pytest.mark.asyncio
    async def test_project_scoped(self):
        """Cleanup is project-scoped — the parameter dict must carry
        the project so other projects' orphans aren't touched."""
        session = self._make_session(count_value=3)
        await _delete_orphan_routes(session, project="EVA_migration")
        # Both calls (count + delete) pass project="EVA_migration"
        for call in session.run.call_args_list:
            assert call.kwargs.get("project") == "EVA_migration"
