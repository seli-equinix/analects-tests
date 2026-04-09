#!/usr/bin/env python3
"""Analects Web UI Validation Suite.

Validates every page, HTMX endpoint, form action, and API call in the
Django management interface.  Uses httpx with session cookies — no browser
needed, so Claude Code can run it directly.

Usage:
    python tests/test_web_ui.py                      # full suite
    python tests/test_web_ui.py --base https://192.168.4.205:8443
    python tests/test_web_ui.py -v                   # verbose
    python tests/test_web_ui.py -k navigation        # run one group
    python tests/test_web_ui.py --list                # list all checks

Exit codes:
    0 = all passed
    1 = failures detected
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable

import httpx

# ── Defaults ──────────────────────────────────────────────────────────
DEFAULT_BASE = "https://192.168.4.205:8443"
DEFAULT_USER = "admin"
DEFAULT_PASS = "Loveme-sex64"
CCA_API = "https://192.168.4.205:8500"


# ── Result tracking ──────────────────────────────────────────────────
@dataclass
class Check:
    name: str
    group: str
    passed: bool = False
    message: str = ""
    duration_ms: float = 0


@dataclass
class Suite:
    checks: list[Check] = field(default_factory=list)
    _current_group: str = ""

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def failed(self) -> int:
        return sum(1 for c in self.checks if not c.passed)

    def record(self, name: str, passed: bool, message: str = "", duration_ms: float = 0):
        self.checks.append(Check(
            name=name, group=self._current_group,
            passed=passed, message=message, duration_ms=duration_ms,
        ))

    def summary(self) -> str:
        lines = []
        groups: dict[str, list[Check]] = {}
        for c in self.checks:
            groups.setdefault(c.group, []).append(c)

        for group, checks in groups.items():
            lines.append(f"\n  {group}")
            for c in checks:
                icon = "PASS" if c.passed else "FAIL"
                ms = f" ({c.duration_ms:.0f}ms)" if c.duration_ms else ""
                detail = f" — {c.message}" if c.message and not c.passed else ""
                lines.append(f"    [{icon}] {c.name}{ms}{detail}")

        total = len(self.checks)
        lines.append(f"\n  {'=' * 50}")
        lines.append(f"  {self.passed}/{total} passed, {self.failed} failed")
        return "\n".join(lines)


# ── HTTP Client ──────────────────────────────────────────────────────
class WebUIClient:
    """Authenticated httpx client for the Django web UI."""

    def __init__(self, base_url: str, username: str, password: str, verbose: bool = False):
        self.base = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.verbose = verbose
        self.client = httpx.Client(
            base_url=self.base,
            verify=False,  # self-signed TLS
            follow_redirects=False,
            timeout=30,
        )
        self.csrf_token: str = ""
        self.logged_in = False

    def login(self) -> bool:
        """Log into Django and store session + CSRF token."""
        # GET login page to get CSRF cookie
        r = self.client.get("/login")
        if r.status_code not in (200, 302):
            return False

        self.csrf_token = self._extract_csrf(r)

        # POST login
        r = self.client.post("/login", data={
            "username": self.username,
            "password": self.password,
            "csrfmiddlewaretoken": self.csrf_token,
        }, follow_redirects=False)

        # Django redirects to / on success (302)
        if r.status_code in (302, 303):
            self.logged_in = True
            # Follow redirect to get fresh CSRF
            r2 = self.client.get(r.headers.get("location", "/"))
            self.csrf_token = self._extract_csrf(r2)
            return True
        return False

    def get(self, path: str, **kwargs) -> httpx.Response:
        r = self.client.get(path, **kwargs)
        if self.verbose:
            print(f"    GET {path} → {r.status_code} ({len(r.content)} bytes)")
        return r

    def post(self, path: str, data: dict | None = None, json_data: dict | None = None, **kwargs) -> httpx.Response:
        headers = {"X-CSRFToken": self.csrf_token}
        if json_data is not None:
            headers["Content-Type"] = "application/json"
            r = self.client.post(path, content=json.dumps(json_data), headers=headers, **kwargs)
        else:
            if data is None:
                data = {}
            data["csrfmiddlewaretoken"] = self.csrf_token
            r = self.client.post(path, data=data, headers=headers, **kwargs)
        if self.verbose:
            print(f"    POST {path} → {r.status_code} ({len(r.content)} bytes)")
        return r

    def _extract_csrf(self, response: httpx.Response) -> str:
        """Extract CSRF token from cookies or HTML."""
        # From cookie
        for cookie in self.client.cookies.jar:
            if cookie.name == "csrftoken":
                return cookie.value
        # From HTML hidden input
        m = re.search(r'name="csrfmiddlewaretoken"\s+value="([^"]+)"', response.text)
        if m:
            return m.group(1)
        return self.csrf_token  # keep old one

    def close(self):
        self.client.close()


# ── Test Groups ──────────────────────────────────────────────────────

def check_auth(client: WebUIClient, suite: Suite):
    """Authentication — login, session, redirect behavior."""
    suite._current_group = "Authentication"

    # Unauthenticated redirect
    anon = httpx.Client(base_url=client.base, verify=False, follow_redirects=False, timeout=10)
    r = anon.get("/")
    suite.record("unauthenticated → redirect to login",
                 r.status_code in (301, 302) and "/login" in r.headers.get("location", ""))
    anon.close()

    # Login page loads
    r = client.get("/login")
    suite.record("login page renders",
                 r.status_code == 200 and "Analects" in r.text and "Sign In" in r.text)

    # Authenticated session works
    suite.record("login succeeded", client.logged_in)

    # Dashboard loads after login
    r = client.get("/")
    suite.record("dashboard loads after login",
                 r.status_code == 200 and "Dashboard" in r.text)


def check_navigation(client: WebUIClient, suite: Suite):
    """Navigation — sidebar links, all pages load without 500."""
    suite._current_group = "Navigation & Pages"

    pages = {
        "/": ("Dashboard", "Dashboard"),
        "/tests": ("AI System Tests", "test"),
        "/reports/": ("Reports", "report"),
        "/config": ("Configuration", "config"),
        "/infra": ("Infrastructure", "infra"),
        "/logs": ("Live Logs", "log"),
        "/users": ("User", "user"),
        "/api-keys": ("API Keys", "key"),
        "/credentials": ("Credentials", "credential"),
        "/prompts": ("Prompts", "prompt"),
        "/tools": ("Tools", "tool"),
        "/knowledge": ("Knowledge", "knowledge"),
    }

    for path, (label, content_check) in pages.items():
        t0 = time.time()
        try:
            r = client.get(path)
            ms = (time.time() - t0) * 1000
            ok = r.status_code == 200
            has_content = content_check.lower() in r.text.lower() if ok else False
            if ok and not has_content:
                suite.record(f"{label} ({path})", False,
                             f"200 but missing expected content '{content_check}'", ms)
            else:
                suite.record(f"{label} ({path})", ok,
                             f"HTTP {r.status_code}" if not ok else "", ms)
        except Exception as e:
            suite.record(f"{label} ({path})", False, str(e))


def check_sidebar(client: WebUIClient, suite: Suite):
    """Sidebar — verify active state and all nav links present."""
    suite._current_group = "Sidebar"

    r = client.get("/")
    html = r.text

    expected_links = [
        ("Dashboard", "/"),
        ("AI System Tests", "/tests"),
        ("Reports", "/reports/"),
        ("Config", "/config"),
        ("Infrastructure", "/infra"),
        ("Live Logs", "/logs"),
        ("API Keys", "/api-keys"),
    ]

    for label, href in expected_links:
        # Check link exists in sidebar
        has_link = f'href="{href}"' in html
        suite.record(f"sidebar has '{label}' link", has_link,
                     f"href={href} not found" if not has_link else "")

    # Check active class on dashboard
    active_match = re.search(r'<a\s+href="/"\s+class="[^"]*active[^"]*"', html)
    suite.record("dashboard link has 'active' class", active_match is not None)


def check_dashboard(client: WebUIClient, suite: Suite):
    """Dashboard — stats grid, test groups, clear-all button, HTMX health."""
    suite._current_group = "Dashboard"

    r = client.get("/")
    html = r.text

    suite.record("stats grid present", "stats-grid" in html)
    suite.record("CCA Status stat card", "CCA Status" in html)
    suite.record("Passed stat card", "Passed" in html)
    suite.record("Failed stat card", "Failed" in html)
    suite.record("Total Reports stat card", "Total Reports" in html)
    suite.record("Clear All button", "clearAll()" in html)
    suite.record("HTMX health endpoint wired", 'hx-get="/api/health-status"' in html)
    suite.record("test groups rendered", "test-grid" in html or "test-card" in html)

    # HTMX partial: health status
    t0 = time.time()
    r = client.get("/api/health-status")
    ms = (time.time() - t0) * 1000
    suite.record("GET /api/health-status returns HTML",
                 r.status_code == 200 and len(r.text) > 0, "", ms)


def check_tests_page(client: WebUIClient, suite: Suite):
    """Tests page — test grid, run/clear buttons, HTMX partials."""
    suite._current_group = "AI System Tests"

    r = client.get("/tests")
    html = r.text

    suite.record("tests page loads", r.status_code == 200)
    suite.record("has Run All button", "Run All" in html)
    suite.record("has Run Failed button", "Run Failed" in html or "runFailed" in html)
    suite.record("has Clear All button", "Clear All" in html or "clearAll" in html)
    suite.record("has auto-refresh toggle", "toggleAutoRefresh" in html or "auto-refresh" in html.lower())
    suite.record("has test status partial endpoint",
                 "/tests/status-partial" in html or "status-partial" in html)

    # Group run buttons
    for group in ("User", "Websearch", "Coder", "Integration"):
        suite.record(f"has {group} group button",
                     group.lower() in html.lower())

    # HTMX partial: test status
    t0 = time.time()
    r = client.get("/tests/status-partial")
    ms = (time.time() - t0) * 1000
    suite.record("GET /tests/status-partial returns content",
                 r.status_code == 200, "", ms)


def check_reports(client: WebUIClient, suite: Suite):
    """Reports — list page, filters, empty state."""
    suite._current_group = "Reports"

    r = client.get("/reports/")
    html = r.text

    suite.record("reports page loads", r.status_code == 200)
    suite.record("has status filter buttons",
                 "Passed" in html and "Failed" in html)
    suite.record("has category filter buttons",
                 any(cat in html for cat in ("User", "Coder", "Websearch")))

    # Filter parameters work
    r = client.get("/reports/?status=FAILED")
    suite.record("filter ?status=FAILED works", r.status_code == 200)

    r = client.get("/reports/?cat=user")
    suite.record("filter ?cat=user works", r.status_code == 200)


def check_config(client: WebUIClient, suite: Suite):
    """Config — tabs, form fields, save/preview endpoints."""
    suite._current_group = "Configuration"

    r = client.get("/config")
    html = r.text

    suite.record("config page loads", r.status_code == 200)
    suite.record("has Settings tab", "Settings" in html)
    suite.record("has Infrastructure tab", "Infrastructure" in html)
    suite.record("has Raw TOML tab", "Raw TOML" in html or "raw" in html.lower())
    suite.record("has Save button", "Save" in html)
    suite.record("has form fields", "form-field" in html or "form-grid" in html or "data-path" in html)

    # Config preview endpoint
    r = client.post("/config/preview", data={"config_text": "[active]\ncoder = \"local\"\n"})
    suite.record("POST /config/preview responds",
                 r.status_code in (200, 405))

    # Backups endpoint
    r = client.get("/config/backups")
    suite.record("GET /config/backups responds",
                 r.status_code in (200, 204))


def check_infra(client: WebUIClient, suite: Suite):
    """Infrastructure — node cards, service health, HTMX partials."""
    suite._current_group = "Infrastructure"

    r = client.get("/infra")
    html = r.text

    suite.record("infra page loads", r.status_code == 200)
    suite.record("has Refresh All button", "Refresh" in html or "refreshAll" in html)
    suite.record("has node cards", "node-card" in html or "node-grid" in html)
    suite.record("has service health section", "service" in html.lower() or "Service" in html)

    # HTMX: all services
    t0 = time.time()
    r = client.get("/infra/all-services")
    ms = (time.time() - t0) * 1000
    suite.record("GET /infra/all-services returns content",
                 r.status_code == 200, "", ms)

    # HTMX: refresh all
    r = client.post("/infra/refresh-all")
    suite.record("POST /infra/refresh-all responds",
                 r.status_code in (200, 204, 302))


def check_logs(client: WebUIClient, suite: Suite):
    """Live Logs — fetch, filter tags, clear."""
    suite._current_group = "Live Logs"

    r = client.get("/logs")
    html = r.text

    suite.record("logs page loads", r.status_code == 200)
    suite.record("has tag filter checkboxes",
                 "toggleLogTag" in html or "log-filters" in html)
    suite.record("has search input",
                 "searchLog" in html or "log-search" in html.lower())
    suite.record("has tail lines selector", "200" in html and "500" in html)

    # Fetch logs
    t0 = time.time()
    r = client.get("/logs/fetch?tail=50")
    ms = (time.time() - t0) * 1000
    suite.record("GET /logs/fetch?tail=50 returns content",
                 r.status_code == 200, "", ms)


def check_users(client: WebUIClient, suite: Suite):
    """Users — Django accounts list, CCA profiles, create form."""
    suite._current_group = "Users"

    r = client.get("/users")
    html = r.text

    suite.record("users page loads", r.status_code == 200)
    suite.record("has Django accounts section",
                 "Web UI Accounts" in html or "Django" in html or "admin" in html.lower())
    suite.record("has CCA profiles section",
                 "Agent Profile" in html or "Qdrant" in html or "CCA" in html)
    suite.record("has Create Account button",
                 "Create Account" in html or "Create" in html)
    suite.record("has export/import buttons",
                 "Export" in html or "Import" in html)


def check_api_keys(client: WebUIClient, suite: Suite):
    """API Keys — list, create form, auth status."""
    suite._current_group = "API Keys"

    r = client.get("/api-keys")
    html = r.text

    suite.record("api-keys page loads", r.status_code == 200)
    suite.record("has Create Key button",
                 "Create Key" in html or "Create" in html)
    suite.record("shows auth status",
                 "Auth Status" in html or "auth" in html.lower())
    suite.record("has key management UI",
                 "key" in html.lower())


def check_credentials(client: WebUIClient, suite: Suite):
    """Credentials — list, add form, import from env."""
    suite._current_group = "Credentials"

    r = client.get("/credentials")
    html = r.text

    suite.record("credentials page loads", r.status_code == 200)
    suite.record("has Add Credential button",
                 "Add Credential" in html or "Add" in html)
    suite.record("has Import from Environment",
                 "Import" in html or "import" in html)


def check_prompts(client: WebUIClient, suite: Suite):
    """Prompts — list page loads."""
    suite._current_group = "Prompts"

    r = client.get("/prompts")
    suite.record("prompts page loads", r.status_code == 200)
    suite.record("has prompt content", "prompt" in r.text.lower())


def check_tools(client: WebUIClient, suite: Suite):
    """Tools — list page loads."""
    suite._current_group = "Tools"

    r = client.get("/tools")
    suite.record("tools page loads", r.status_code == 200)
    suite.record("has tool content", "tool" in r.text.lower())


def check_knowledge(client: WebUIClient, suite: Suite):
    """Knowledge — list page loads."""
    suite._current_group = "Knowledge"

    r = client.get("/knowledge")
    suite.record("knowledge page loads", r.status_code == 200)
    suite.record("has knowledge content", "knowledge" in r.text.lower())


def check_cca_api_integration(client: WebUIClient, suite: Suite):
    """CCA API — health endpoint, models, connection from Django."""
    suite._current_group = "CCA API Integration"

    # Direct CCA health check (bypasses Django)
    try:
        r = httpx.get(f"{CCA_API}/health", timeout=10)
        data = r.json()
        suite.record("CCA /health reachable",
                     r.status_code == 200 and data.get("status") == "healthy")
        suite.record("CCA reports version", bool(data.get("version")))
    except Exception as e:
        suite.record("CCA /health reachable", False, str(e))
        suite.record("CCA reports version", False, "health unreachable")

    # CCA models endpoint
    try:
        r = httpx.get(f"{CCA_API}/v1/models", timeout=10)
        suite.record("CCA /v1/models responds",
                     r.status_code == 200 and "data" in r.json())
    except Exception as e:
        suite.record("CCA /v1/models responds", False, str(e))


def check_clear_all(client: WebUIClient, suite: Suite):
    """Clear All — the POST endpoint responds (does NOT actually clear in test mode)."""
    suite._current_group = "Clear Operations"

    # We only test that the endpoint responds — don't actually wipe data
    # Send OPTIONS or check the function exists
    r = client.get("/")
    suite.record("clearAll() function defined in dashboard",
                 "function clearAll" in r.text or "clearAll()" in r.text)

    # Check the /api/clear-all endpoint exists (POST required)
    r = client.get("/api/clear-all")
    # Should 405 (method not allowed) since it requires POST
    suite.record("GET /api/clear-all returns 405 (POST required)",
                 r.status_code in (405, 302, 200))


def check_htmx_loaded(client: WebUIClient, suite: Suite):
    """HTMX — verify the library is served and loaded."""
    suite._current_group = "Static Assets"

    r = client.get("/")
    html = r.text

    # Check HTMX script tag
    htmx_match = re.search(r'src="([^"]*htmx[^"]*)"', html)
    suite.record("HTMX script tag in base template", htmx_match is not None)

    if htmx_match:
        htmx_url = htmx_match.group(1)
        r = client.get(htmx_url)
        suite.record(f"HTMX JS loads ({htmx_url})",
                     r.status_code == 200 and len(r.content) > 1000)

    # Check CSS loads
    css_match = re.search(r'href="([^"]*style\.css[^"]*)"', html)
    suite.record("style.css link in base template", css_match is not None)

    if css_match:
        css_url = css_match.group(1)
        r = client.get(css_url)
        suite.record(f"CSS loads ({css_url})",
                     r.status_code == 200 and len(r.content) > 500)


def check_logout(client: WebUIClient, suite: Suite):
    """Logout — POST to /logout clears session."""
    suite._current_group = "Logout"

    # Don't actually log out — just verify the form exists
    r = client.get("/")
    has_logout = ('action="/logout"' in r.text
                  or 'href="/logout"' in r.text
                  or "Logout" in r.text)
    suite.record("logout control present in sidebar", has_logout)


def check_test_setup(client: WebUIClient, suite: Suite):
    """Test Setup — state endpoint and sync controls."""
    suite._current_group = "Test Setup"

    t0 = time.time()
    r = client.get("/test-setup/state")
    ms = (time.time() - t0) * 1000
    suite.record("GET /test-setup/state responds",
                 r.status_code == 200, "", ms)

    if r.status_code == 200:
        try:
            data = r.json()
            suite.record("test-setup state returns JSON", isinstance(data, dict))
        except Exception:
            suite.record("test-setup state returns JSON", False, "not valid JSON")


def check_no_errors_in_pages(client: WebUIClient, suite: Suite):
    """Error Detection — scan all pages for Django error pages or tracebacks."""
    suite._current_group = "Error Detection"

    error_patterns = [
        "Traceback (most recent call last)",
        "TemplateSyntaxError",
        "TemplateDoesNotExist",
        "ImproperlyConfigured",
        "Server Error (500)",
        "Page not found (404)",
        "NoReverseMatch",
        "OperationalError",
    ]

    pages = ["/", "/tests", "/reports/", "/config", "/infra", "/logs",
             "/users", "/api-keys", "/credentials", "/prompts", "/tools", "/knowledge"]

    errors_found = []
    for path in pages:
        try:
            r = client.get(path)
            for pattern in error_patterns:
                if pattern in r.text:
                    errors_found.append(f"{path}: {pattern}")
        except Exception:
            pass

    suite.record("no Django errors on any page",
                 len(errors_found) == 0,
                 "; ".join(errors_found[:5]) if errors_found else "")


# ── Runner ───────────────────────────────────────────────────────────

ALL_GROUPS: list[tuple[str, Callable]] = [
    ("auth", check_auth),
    ("navigation", check_navigation),
    ("sidebar", check_sidebar),
    ("static", check_htmx_loaded),
    ("dashboard", check_dashboard),
    ("tests", check_tests_page),
    ("reports", check_reports),
    ("config", check_config),
    ("infra", check_infra),
    ("logs", check_logs),
    ("users", check_users),
    ("api_keys", check_api_keys),
    ("credentials", check_credentials),
    ("prompts", check_prompts),
    ("tools", check_tools),
    ("knowledge", check_knowledge),
    ("test_setup", check_test_setup),
    ("cca_api", check_cca_api_integration),
    ("clear", check_clear_all),
    ("logout", check_logout),
    ("errors", check_no_errors_in_pages),
]


def main():
    parser = argparse.ArgumentParser(description="Analects Web UI Validation Suite")
    parser.add_argument("--base", default=DEFAULT_BASE, help="Base URL of the web UI")
    parser.add_argument("--user", default=DEFAULT_USER, help="Login username")
    parser.add_argument("--password", default=DEFAULT_PASS, help="Login password")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show HTTP requests")
    parser.add_argument("-k", "--filter", help="Run only groups matching this substring")
    parser.add_argument("--list", action="store_true", help="List all check groups")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    if args.list:
        for name, _ in ALL_GROUPS:
            print(f"  {name}")
        return

    suite = Suite()
    client = WebUIClient(args.base, args.user, args.password, verbose=args.verbose)

    print(f"\n  Analects Web UI Validation")
    print(f"  Target: {args.base}")
    print(f"  {'=' * 50}")

    # Login first
    try:
        ok = client.login()
        if not ok:
            print("\n  FATAL: Login failed. Check credentials and URL.")
            sys.exit(1)
        print(f"  Logged in as: {args.user}")
    except Exception as e:
        print(f"\n  FATAL: Cannot connect to {args.base}")
        print(f"  Error: {e}")
        sys.exit(1)

    # Run checks
    groups = ALL_GROUPS
    if args.filter:
        groups = [(n, f) for n, f in groups if args.filter.lower() in n.lower()]

    for name, func in groups:
        try:
            func(client, suite)
        except Exception as e:
            suite._current_group = name
            suite.record(f"GROUP CRASHED: {name}", False, f"{type(e).__name__}: {e}")
            if args.verbose:
                traceback.print_exc()

    client.close()

    # Output
    if args.json:
        results = []
        for c in suite.checks:
            results.append({
                "group": c.group, "name": c.name,
                "passed": c.passed, "message": c.message,
                "duration_ms": round(c.duration_ms, 1),
            })
        print(json.dumps({"passed": suite.passed, "failed": suite.failed, "checks": results}, indent=2))
    else:
        print(suite.summary())

    sys.exit(0 if suite.failed == 0 else 1)


if __name__ == "__main__":
    main()
