"""HTTP client for the Analects Agent-as-a-Model server.

Uses SSE streaming with idle timeout instead of fixed total timeouts.
This means tests won't fail just because a task takes longer than expected
— they only fail if the server stops sending data entirely.

Only chat() creates OpenTelemetry spans — diagnostic/helper methods
(health, list_users, find_user, cleanup) are untraced to keep Phoenix
traces clean. Each test trace shows: test span → cca.chat → server spans.
W3C traceparent propagation unifies test and server spans into one trace.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

import httpx
from opentelemetry import trace
from opentelemetry.propagate import inject
from opentelemetry.trace import StatusCode

# Timeout defaults (seconds)
TIMEOUT_HEALTH = 10
TIMEOUT_CONNECT = 30
TIMEOUT_IDLE = 120       # No data for 120s → dead
TIMEOUT_DIAGNOSTIC = 15


class ChatResult:
    """Parsed result from a /v1/chat/completions call."""

    def __init__(self, raw: Dict[str, Any], elapsed_ms: float) -> None:
        self.raw = raw
        self.elapsed_ms = elapsed_ms
        self.id: str = raw.get("id", "")
        self.model: str = raw.get("model", "")

        choices = raw.get("choices", [])
        msg = choices[0].get("message", {}) if choices else {}
        self.content: str = msg.get("content", "") or ""
        self.role: str = msg.get("role", "")
        self.reasoning: Optional[str] = msg.get("reasoning")
        self.finish_reason: str = (
            choices[0].get("finish_reason", "") if choices else ""
        )

        self.usage: Dict[str, int] = raw.get("usage", {})
        self.metadata: Dict[str, Any] = raw.get("context_metadata", {}) or {}

    @property
    def tool_labels(self) -> List[str]:
        """All SSE comment labels from the stream (progress + errors)."""
        return self.raw.get("tool_labels", [])

    @property
    def tool_errors(self) -> List[str]:
        """SSE labels that indicate tool failures."""
        error_keywords = ("failed", "error", "invalid", "not allowed", "disallowed")
        return [
            lbl for lbl in self.tool_labels
            if any(kw in lbl.lower() for kw in error_keywords)
        ]

    @property
    def user_identified(self) -> bool:
        return self.metadata.get("user_identified", False)

    @property
    def user_name(self) -> Optional[str]:
        return self.metadata.get("user_name")

    @property
    def user_id(self) -> Optional[str]:
        return self.metadata.get("user_id")

    @property
    def tool_names(self) -> List[str]:
        """Names of all tools called during this request."""
        return [
            tc["name"] for tc in (self.metadata.get("tool_calls") or [])
        ]

    def __repr__(self) -> str:
        preview = self.content[:80] + "..." if len(self.content) > 80 else self.content
        meta_parts = [f"{self.elapsed_ms:.0f}ms", f"{len(self.content)} chars"]
        if self.metadata.get("route"):
            meta_parts.append(f"route={self.metadata['route']}")
        if self.metadata.get("tool_iterations"):
            meta_parts.append(f"iters={self.metadata['tool_iterations']}")
        if self.metadata.get("estimated_steps"):
            meta_parts.append(f"steps={self.metadata['estimated_steps']}")
        if self.metadata.get("nudge_skipped"):
            meta_parts.append("nudge_skipped")
        if self.metadata.get("circuit_breaker_fired"):
            meta_parts.append("CB_FIRED")
        errors = self.tool_errors
        if errors:
            meta_parts.append(f"TOOL_ERRORS={len(errors)}")
        return f"ChatResult({', '.join(meta_parts)}: {preview!r})"


class CCAClient:
    """HTTP client for CCA AAAM server with streaming + idle timeout.

    Uses SSE streaming to avoid fixed total timeouts. The idle_timeout
    resets every time data arrives (content, keepalive, progress comments).
    A task that takes 10 minutes but is actively working will succeed;
    a hung connection that goes silent for idle_timeout seconds will fail.
    """

    def __init__(
        self,
        base_url: str = os.getenv("CCA_BASE_URL", "http://192.168.4.205:8500"),
        tracer: Optional[trace.Tracer] = None,
        idle_timeout: float = TIMEOUT_IDLE,
        project_name: Optional[str] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.tracer = tracer or trace.get_tracer("cca-http")
        self.idle_timeout = idle_timeout
        self.project_name = project_name
        self.api_key = os.getenv("CCA_TEST_API_KEY", "")
        self._session_turns: dict[str, int] = {}  # session_id → turn count
        self._client = httpx.Client(
            timeout=httpx.Timeout(
                connect=TIMEOUT_CONNECT,
                read=idle_timeout,
                write=30.0,
                pool=30.0,
            )
        )

    def close(self) -> None:
        self._client.close()

    # ==================== Core Methods ====================

    def health(self) -> Dict[str, Any]:
        """GET /health — check server status. No tracing (runs every test)."""
        try:
            resp = self._client.get(
                f"{self.base_url}/health", timeout=TIMEOUT_HEALTH
            )
            return resp.json()
        except Exception as e:
            return {"status": "unreachable", "error": str(e)}

    def check_backends(self) -> Dict[str, Any]:
        """Check health of CCA + vLLM backends. Returns errors dict.

        Use after each test turn to detect backend failures early
        instead of waiting for a hardcoded timeout.
        """
        issues: Dict[str, Any] = {}

        # CCA server
        cca_health = self.health()
        if cca_health.get("status") != "healthy":
            issues["cca"] = cca_health.get("error", "unhealthy")

        # vLLM coder (Spark2:8000)
        vllm_url = os.environ.get("VLLM_BASE_URL", "http://192.168.4.208:8000/v1")
        try:
            resp = self._client.get(
                f"{vllm_url[:-3]}/health", timeout=5,
            )
            if resp.status_code != 200:
                issues["vllm_coder"] = f"HTTP {resp.status_code}"
        except Exception as e:
            issues["vllm_coder"] = str(e)

        # vLLM notetaker (Spark1:8400)
        try:
            resp = self._client.get(
                "http://192.168.4.205:8400/health", timeout=5,
            )
            if resp.status_code != 200:
                issues["vllm_notetaker"] = f"HTTP {resp.status_code}"
        except Exception as e:
            issues["vllm_notetaker"] = str(e)

        return issues

    def _start_health_monitor(self) -> tuple:
        """Start a background thread that checks backend health every 15s.

        Returns (stop_event, issues_dict, thread). Call stop_event.set()
        when done. Any backend failures detected during monitoring are
        stored in issues_dict.
        """
        issues: Dict[str, Any] = {}
        stop = threading.Event()

        def _monitor() -> None:
            while not stop.is_set():
                stop.wait(15)
                if stop.is_set():
                    break
                try:
                    found = self.check_backends()
                    if found:
                        issues.update(found)
                        log.warning("Backend issue during stream: %s", found)
                except Exception:
                    pass

        t = threading.Thread(target=_monitor, daemon=True)
        t.start()
        return stop, issues, t

    def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        idle_timeout: Optional[float] = None,
        system: Optional[str] = None,
    ) -> ChatResult:
        """POST /v1/chat/completions — send a message to the Analects agent.

        Uses SSE streaming with idle timeout. The connection stays open as
        long as the server sends data (content, keepalives, progress).
        Only times out if no data arrives for idle_timeout seconds.

        Retries once on transient connection errors (5s backoff) to handle
        network blips without failing the test. The server sends keepalives
        every 8s, so a genuine timeout means the connection is truly dead.

        Args:
            message: The user message to send.
            session_id: Optional session ID for multi-turn conversations.
            idle_timeout: Seconds of silence before timeout (default: 120s).
            system: Optional system message.
        """
        session_id = session_id or f"test-{uuid.uuid4().hex[:12]}"
        read_timeout = idle_timeout or self.idle_timeout
        max_attempts = 2  # 1 try + 1 retry for transient errors

        # Background health monitor — checks backends every 15s during
        # streaming. Detects vLLM/CCA crashes within seconds instead of
        # waiting for a hard timeout.
        monitor_stop, health_issues, monitor_thread = self._start_health_monitor()

        # Capture the parent (test) span BEFORE starting the cca.chat child span.
        # Inside the child span, trace.get_current_span() returns the child,
        # not the test span — so we must save the reference here.
        test_span = trace.get_current_span()

        # Track turn number per session for Phoenix labeling
        turn = self._session_turns.get(session_id, 0) + 1
        self._session_turns[session_id] = turn
        span_name = f"cca.chat [Turn {turn}]"

        with self.tracer.start_as_current_span(span_name) as span:
            span.set_attribute("openinference.span.kind", "CHAIN")
            span.set_attribute("input.value", message)
            span.set_attribute("cca.session_id", session_id)
            span.set_attribute("cca.turn", turn)
            span.set_attribute("cca.message", message[:200])
            span.set_attribute("cca.idle_timeout", read_timeout)

            messages: List[Dict[str, str]] = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": message})

            payload = {
                "model": "cca",
                "messages": messages,
                "stream": True,
                "session_id": session_id,
            }
            if user_id:
                payload["user_id"] = user_id

            last_error: Optional[Exception] = None
            for attempt in range(max_attempts):
                t0 = time.time()
                try:
                    result = self._stream_chat(
                        payload, session_id, read_timeout, span,
                        user_id=user_id,
                    )
                    elapsed_ms = (time.time() - t0) * 1000
                    result.elapsed_ms = elapsed_ms

                    # _stream_chat returns error ChatResult on connection
                    # failures instead of raising. Retry once on these.
                    if result.raw.get("error") and attempt < max_attempts - 1:
                        span.set_attribute(
                            "cca.retry_reason",
                            result.raw["error"][:200],
                        )
                        span.set_attribute("cca.retry_attempt", attempt + 1)
                        time.sleep(5)
                        continue

                    span.set_attribute("cca.status", "success")
                    span.set_attribute("cca.duration_ms", elapsed_ms)
                    span.set_attribute("cca.response_length", len(result.content))
                    span.set_attribute("cca.response_preview", result.content[:500])
                    span.set_attribute("output.value", result.content)
                    span.set_attribute("cca.finish_reason", result.finish_reason)
                    if result.metadata:
                        span.set_attribute(
                            "cca.tool_iterations",
                            result.metadata.get("tool_iterations", 0),
                        )
                        if result.metadata.get("route"):
                            span.set_attribute("cca.route", result.metadata["route"])
                        if result.metadata.get("estimated_steps"):
                            span.set_attribute(
                                "cca.estimated_steps",
                                result.metadata["estimated_steps"],
                            )
                    if result.user_identified:
                        span.set_attribute("cca.user_identified", True)
                        span.set_attribute("cca.user_name", result.user_name or "")

                    # Stash metrics on test (parent) span for per-test reporting.
                    # test_span was captured before the child cca.chat span was
                    # started — trace.get_current_span() here would return the
                    # child span, not the test span.
                    if hasattr(test_span, "_test_metrics"):
                        test_span._test_metrics.update({
                            "tool_iterations": result.metadata.get("tool_iterations", 0),
                            "route": result.metadata.get("route", ""),
                            "estimated_steps": result.metadata.get("estimated_steps", 0),
                            "execution_time_ms": elapsed_ms,
                            "nudge_skipped": result.metadata.get("nudge_skipped", False),
                            "circuit_breaker_fired": result.metadata.get("circuit_breaker_fired", False),
                        })

                    # Accumulate turns so conftest.py can set input.value /
                    # output.value on the root span after ALL turns complete.
                    # Include full metadata for report generation (tool calls,
                    # errors, system events).
                    if hasattr(test_span, "_test_metrics"):
                        test_span._test_metrics.setdefault("_turns", []).append(
                            (message, result.content)
                        )
                        test_span._test_metrics.setdefault("_turn_details", []).append({
                            "tool_calls": result.metadata.get("tool_calls", []),
                            "tool_labels": list(getattr(result, "tool_labels", [])),
                            "tool_errors": list(getattr(result, "tool_errors", [])),
                            "tool_iterations": result.metadata.get("tool_iterations", 0),
                            "tools_escalated": result.metadata.get("tools_escalated", False),
                            "escalated_groups": result.metadata.get("escalated_groups"),
                            "nudge_skipped": result.metadata.get("nudge_skipped", False),
                            "circuit_breaker_fired": result.metadata.get("circuit_breaker_fired", False),
                            "max_iterations": result.metadata.get("max_iterations", 0),
                            "elapsed_ms": elapsed_ms,
                        })

                    span.set_status(StatusCode.OK)

                    # Stop health monitor and attach any issues found
                    monitor_stop.set()
                    monitor_thread.join(timeout=2)
                    if health_issues:
                        result.raw["_backend_issues"] = dict(health_issues)
                        span.set_attribute(
                            "cca.backend_issues", str(health_issues),
                        )

                    return result

                except Exception as e:
                    last_error = e
                    elapsed_ms = (time.time() - t0) * 1000
                    if attempt < max_attempts - 1:
                        span.set_attribute(
                            "cca.retry_reason", str(e)[:200],
                        )
                        span.set_attribute("cca.retry_attempt", attempt + 1)
                        time.sleep(5)
                        continue
                    span.set_attribute("cca.status", "error")
                    span.set_attribute("cca.error", str(e))
                    span.set_attribute("cca.duration_ms", elapsed_ms)
                    span.set_status(StatusCode.ERROR, str(e)[:500])
                    monitor_stop.set()
                    monitor_thread.join(timeout=2)
                    raise

            # Should not reach here, but satisfy type checker
            monitor_stop.set()
            raise last_error  # type: ignore[misc]

    def _stream_chat(
        self,
        payload: Dict[str, Any],
        session_id: str,
        read_timeout: float,
        span: Any,
        user_id: Optional[str] = None,
    ) -> ChatResult:
        """Execute streaming chat and accumulate response.

        Parses SSE events, accumulates content + reasoning, and extracts
        context_metadata from the final metadata event before [DONE].

        Catches httpx transport errors (ReadTimeout, ConnectionReset, etc.)
        and returns a ChatResult with partial content + error info instead
        of propagating raw httpcore tracebacks.
        """
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_labels: list[str] = []  # SSE comment labels (progress, errors)
        completion_id = ""
        model = ""
        finish_reason = ""
        context_metadata: Dict[str, Any] = {}

        timeout = httpx.Timeout(
            connect=TIMEOUT_CONNECT,
            read=read_timeout,
            write=30.0,
            pool=30.0,
        )

        # Build headers with W3C trace context propagation.
        # inject() adds traceparent so server spans become children of this
        # test trace — unified view in Phoenix (one trace per test).
        headers: Dict[str, str] = {"X-Session-Id": session_id}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if user_id:
            headers["X-User-Id"] = user_id
        if self.project_name:
            headers["X-Phoenix-Project"] = self.project_name
        inject(headers)  # adds traceparent header

        try:
            with self._client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=timeout,
            ) as resp:
                if resp.status_code != 200:
                    body = resp.read().decode()
                    return ChatResult(
                        {"error": body, "status_code": resp.status_code}, 0
                    )

                for line in resp.iter_lines():
                    if not line.strip():
                        continue

                    # SSE comments (keepalive, progress) — capture for
                    # tool error reporting, then continue. They also
                    # reset the read timeout, which is the whole point.
                    if line.startswith(":"):
                        label = line[1:].strip()
                        if label:  # skip empty keepalives
                            tool_labels.append(label)
                        continue

                    if not line.startswith("data: "):
                        continue

                    data_str = line[6:]  # strip "data: " prefix

                    # Terminal marker
                    if data_str == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    # Context metadata event (sent before [DONE])
                    if "context_metadata" in chunk and "choices" not in chunk:
                        context_metadata = chunk["context_metadata"]
                        continue

                    completion_id = chunk.get("id", completion_id)
                    model = chunk.get("model", model)

                    choices = chunk.get("choices", [])
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})
                    fr = choices[0].get("finish_reason")
                    if fr:
                        finish_reason = fr

                    if delta.get("content"):
                        content_parts.append(delta["content"])
                    if delta.get("reasoning_content"):
                        reasoning_parts.append(delta["reasoning_content"])

        except (httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError,
                httpx.CloseError) as e:
            # Return partial content with clear error instead of raw traceback
            content = "".join(content_parts)
            partial = f" (partial: {len(content)} chars)" if content else ""
            return ChatResult(
                {
                    "error": f"{type(e).__name__}: {e}{partial}",
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": content,
                            "reasoning": "".join(reasoning_parts) or None,
                        },
                        "finish_reason": "error",
                    }],
                    "context_metadata": context_metadata,
                },
                0,
            )

        # Build a ChatResult that looks like a non-streaming response
        content = "".join(content_parts)
        reasoning = "".join(reasoning_parts) or None

        raw = {
            "id": completion_id,
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "reasoning": reasoning,
                    },
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {},
            "context_metadata": context_metadata,
            "tool_labels": tool_labels,
        }
        return ChatResult(raw, 0)

    # ==================== Diagnostic Endpoints ====================

    def list_users(self) -> Dict[str, Any]:
        """GET /users — list all known user profiles. No tracing."""
        resp = self._client.get(
            f"{self.base_url}/users", timeout=TIMEOUT_DIAGNOSTIC
        )
        return resp.json()

    def list_sessions(self) -> Dict[str, Any]:
        """GET /sessions — list active sessions. No tracing."""
        resp = self._client.get(
            f"{self.base_url}/sessions", timeout=TIMEOUT_DIAGNOSTIC
        )
        return resp.json()

    def get_stats(self) -> Dict[str, Any]:
        """GET /stats — diagnostic statistics. No tracing."""
        resp = self._client.get(
            f"{self.base_url}/stats", timeout=TIMEOUT_DIAGNOSTIC
        )
        return resp.json()

    def list_models(self) -> Dict[str, Any]:
        """GET /v1/models — list available models. No tracing."""
        resp = self._client.get(
            f"{self.base_url}/v1/models", timeout=TIMEOUT_DIAGNOSTIC
        )
        return resp.json()

    # ==================== Helpers ====================

    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """GET /user/{user_id} — full profile with facts, skills, aliases. No tracing."""
        try:
            resp = self._client.get(
                f"{self.base_url}/user/{user_id}", timeout=TIMEOUT_DIAGNOSTIC
            )
            if resp.status_code == 200:
                return resp.json()
            log.warning(
                "get_user_profile(%s) status=%d body=%s",
                user_id, resp.status_code, resp.text[:300],
            )
        except Exception as e:
            log.warning(
                "get_user_profile(%s) exception: %s: %s",
                user_id, type(e).__name__, e,
            )
        return None

    def find_user_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Search /users for a user by display name (case-insensitive). No tracing."""
        data = self.list_users()
        for user in data.get("users", []):
            if user.get("display_name", "").lower() == name.lower():
                return user
            # Also check aliases
            aliases = [a.lower() for a in user.get("aliases", [])]
            if name.lower() in aliases:
                return user
        return None

    def cleanup_test_user(self, name: str, session_id: Optional[str] = None) -> None:
        """Delete a test user profile via REST API. No tracing.

        Uses DELETE /users/{user_id} directly — no LLM round-trip needed.
        Best-effort cleanup — failures are logged but don't raise.
        """
        try:
            user = self.find_user_by_name(name)
            if user is None:
                return
            user_id = user["user_id"]
            self._client.delete(
                f"{self.base_url}/users/{user_id}",
                timeout=TIMEOUT_DIAGNOSTIC,
            )
        except Exception:
            pass  # Best-effort cleanup

    def delete_sessions(self, session_ids: List[str]) -> None:
        """DELETE /sessions — batch delete sessions from Redis. No tracing."""
        if not session_ids:
            return
        try:
            self._client.request(
                "DELETE",
                f"{self.base_url}/sessions",
                json={"session_ids": session_ids},
                timeout=TIMEOUT_DIAGNOSTIC,
            )
        except Exception:
            pass  # Best-effort cleanup

    def tracker(self) -> "TestResourceTracker":
        """Create a resource tracker for per-test cleanup."""
        return TestResourceTracker(self)

    def search_notes(self, query: str, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """GET /v1/notes/search — semantic search over extracted notes. No tracing."""
        params: Dict[str, Any] = {"q": query}
        if user_id:
            params["user_id"] = user_id
        try:
            resp = self._client.get(
                f"{self.base_url}/v1/notes/search",
                params=params,
                timeout=TIMEOUT_DIAGNOSTIC,
            )
            data = resp.json()
            return data.get("notes", [])
        except Exception:
            return []

    def list_workspace_files(self) -> Dict[str, Any]:
        """GET /workspace/files — list files in /workspace. No tracing."""
        resp = self._client.get(
            f"{self.base_url}/workspace/files", timeout=TIMEOUT_DIAGNOSTIC
        )
        return resp.json()

    def clean_workspace_files(self, prefix: str = "") -> Dict[str, Any]:
        """DELETE /workspace/files — remove files from /workspace. No tracing."""
        params = {"prefix": prefix} if prefix else {}
        resp = self._client.request(
            "DELETE",
            f"{self.base_url}/workspace/files",
            params=params,
            timeout=TIMEOUT_DIAGNOSTIC,
        )
        return resp.json()


class TestResourceTracker:
    """Tracks resources created by a single test for targeted cleanup.

    Each test creates one tracker, registers resources as they are created,
    and calls cleanup() in its finally: block. Only cleans what was tracked.

    Usage::

        tracker = cca.tracker()
        tracker.track_user(name)
        tracker.track_session(session_id)
        try:
            ...  # test body
        finally:
            tracker.cleanup()
    """

    def __init__(self, client: CCAClient) -> None:
        self._client = client
        self._user_names: List[str] = []
        self._session_ids: List[str] = []
        self._workspace_prefixes: List[str] = []

    def track_user(self, name: str) -> None:
        """Register a user name created by this test."""
        if name not in self._user_names:
            self._user_names.append(name)

    def track_session(self, session_id: str) -> None:
        """Register a session ID created by this test."""
        if session_id not in self._session_ids:
            self._session_ids.append(session_id)

    def track_workspace_prefix(self, prefix: str) -> None:
        """Register a workspace file prefix for cleanup."""
        if prefix not in self._workspace_prefixes:
            self._workspace_prefixes.append(prefix)

    def cleanup(self) -> None:
        """Delete all tracked resources. Best-effort, logs failures."""
        # 1. Users (cascades to Qdrant profiles, contexts, notes)
        for name in self._user_names:
            try:
                self._client.cleanup_test_user(name)
            except Exception as e:
                log.warning("Tracker cleanup user %s: %s", name, e)

        # 2. Sessions (Redis state + trajectory)
        if self._session_ids:
            try:
                self._client.delete_sessions(self._session_ids)
            except Exception as e:
                log.warning("Tracker cleanup sessions: %s", e)

        # 3. Workspace files
        for prefix in self._workspace_prefixes:
            try:
                self._client.clean_workspace_files(prefix=prefix)
            except Exception as e:
                log.warning("Tracker cleanup workspace %s: %s", prefix, e)


class TestRunContext:
    """Auto-tracking test context that writes a manifest on completion.

    Wraps CCAClient to auto-capture session IDs and user creation.
    On finalize(), writes manifest.json to the reports directory.
    The CI after_script uploads it alongside the report files.
    Django auto-ingests the manifest into the TestRun registry.

    **Never deletes test data** — that's the dashboard's job via
    "Clear One" or "Clear All".

    Usage::

        def test_something(test_run):
            session_id = f"test-xxx-{uuid4().hex[:8]}"
            test_run.track_session(session_id)
            r = test_run.chat("Hello", session_id=session_id)
            # ... assertions ...
            # On completion: fixture calls finalize() automatically
    """

    def __init__(self, client: CCAClient, test_node_name: str) -> None:
        self.client = client  # Direct access for tests that need it
        self._test_name = self._normalize_name(test_node_name)
        self._pipeline_id = os.environ.get("CI_PIPELINE_ID", "local")
        self._user_ids: List[str] = []
        self._user_names: List[str] = []
        self._session_ids: List[str] = []
        self._workspace_prefixes: List[str] = []
        self._started_at = datetime.utcnow()
        self._status = "running"
        self._turns = 0
        self._route = ""

    @property
    def base_url(self) -> str:
        return self.client.base_url

    def chat(self, message: str, session_id: Optional[str] = None, **kwargs) -> "ChatResult":
        """Send a message via CCA and auto-track the session."""
        if session_id and session_id not in self._session_ids:
            self._session_ids.append(session_id)
        result = self.client.chat(message, session_id=session_id, **kwargs)
        self._turns += 1
        if hasattr(result, "route") and result.route and not self._route:
            self._route = result.route
        return result

    def track_user(self, name: str, user_id: Optional[str] = None) -> None:
        """Register a user created by this test."""
        if name not in self._user_names:
            self._user_names.append(name)
        if user_id and user_id not in self._user_ids:
            self._user_ids.append(user_id)

    def track_session(self, session_id: str) -> None:
        """Register a session ID created by this test."""
        if session_id not in self._session_ids:
            self._session_ids.append(session_id)

    def track_workspace_prefix(self, prefix: str) -> None:
        """Register a workspace file prefix created by this test."""
        if prefix not in self._workspace_prefixes:
            self._workspace_prefixes.append(prefix)

    def resolve_user_id(self, name: str) -> Optional[str]:
        """Look up a user's ID by name and track it."""
        user = self.client.find_user_by_name(name)
        if user:
            uid = user.get("user_id")
            if uid and uid not in self._user_ids:
                self._user_ids.append(uid)
            return uid
        return None

    def get_tracked_resources(self) -> dict:
        """Return a summary of all tracked resources."""
        return {
            "user_ids": list(self._user_ids),
            "user_names": list(self._user_names),
            "session_ids": list(self._session_ids),
            "workspace_prefixes": list(self._workspace_prefixes),
        }

    def finalize(self, failed: bool = False) -> None:
        """Write manifest.json to the reports directory.

        Called by the test_run fixture teardown. Does NOT delete any data.
        The manifest is uploaded to CCA by the CI after_script and ingested
        into the Django TestRun registry.
        """
        self._status = "failed" if failed else "passed"
        elapsed = (datetime.utcnow() - self._started_at).total_seconds()

        # Resolve any unresolved user IDs
        for name in self._user_names:
            if not any(name.lower() in (uid or "").lower() for uid in self._user_ids):
                self.resolve_user_id(name)

        manifest = {
            "test_name": self._test_name,
            "pipeline_id": self._pipeline_id,
            "status": self._status,
            "route": self._route,
            "turns": self._turns,
            "duration_ms": elapsed * 1000,
            "user_ids": self._user_ids,
            "user_names": self._user_names,
            "session_ids": self._session_ids,
            "workspace_prefixes": self._workspace_prefixes,
            "phoenix_project": f"test/{self._test_name}",
            "report_folder": f"P{self._pipeline_id}_{self._test_name}",
            "finished_at": datetime.utcnow().isoformat(),
        }

        try:
            from report_generator import REPORTS_DIR
            folder = REPORTS_DIR / f"P{self._pipeline_id}_{self._test_name}"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "manifest.json").write_text(
                json.dumps(manifest, indent=2, default=str)
            )
            log.info("Wrote manifest: %s (%d users, %d sessions)",
                     folder.name, len(self._user_ids), len(self._session_ids))
        except Exception as e:
            log.warning("Failed to write manifest: %s", e)

    @staticmethod
    def _normalize_name(node_name: str) -> str:
        """Convert pytest node name to CI test-name format.

        Examples:
            "TestNewUserOnboarding::test_new_user_onboarding"
            → "new-user-onboarding"
        """
        # Take the method name part (after ::)
        name = node_name.split("::")[-1] if "::" in node_name else node_name
        # Strip test_ prefix
        if name.startswith("test_"):
            name = name[5:]
        # Underscores to hyphens
        return name.replace("_", "-")
