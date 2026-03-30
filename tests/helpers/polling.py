"""Polling helpers for async CCA subsystems.

NoteObserver and FactExtractor are fire-and-forget async tasks.
Instead of fixed sleeps, these helpers poll with backend health
checks — if a backend is down, they fail fast with a clear error
instead of waiting the full timeout.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tests.cca_client import CCAClient

log = logging.getLogger(__name__)


def wait_for_notes(
    cca: CCAClient,
    query: str,
    user_id: str | None = None,
    session_id: str | None = None,
    max_wait: int = 90,
    interval: int = 3,
) -> list[dict]:
    """Wait for NoteObserver to complete, then search for notes.

    Two-phase polling:
      Phase 1: If session_id is provided, poll /v1/notes/status until
               the NoteObserver reports completed/failed for this session.
      Phase 2: Search for notes matching the query.

    This avoids the race condition where tests check for notes before
    the async NoteObserver has finished extracting them.

    Args:
        cca: CCAClient instance.
        query: Search query for notes.
        user_id: Optional user filter.
        session_id: Session to wait for (enables Phase 1 status polling).
        max_wait: Total seconds to wait before giving up.
        interval: Seconds between polls.

    Returns:
        List of note dicts, or empty list on timeout.

    Raises:
        RuntimeError: If a backend is unhealthy during polling.
    """
    elapsed = 0

    # Phase 1: Wait for NoteObserver to complete for this session
    if session_id:
        log.info("Waiting for NoteObserver to complete session %s...", session_id)
        while elapsed < max_wait:
            time.sleep(interval)
            elapsed += interval

            try:
                headers = {}
                if cca.api_key:
                    headers["Authorization"] = f"Bearer {cca.api_key}"
                resp = cca._client.get(
                    f"{cca.base_url}/v1/notes/status",
                    params={"session_id": session_id},
                    headers=headers,
                    timeout=5,
                )
                status = resp.json().get("status", "unknown")
                pending = resp.json().get("pending", 0)

                if status == "completed":
                    log.info(
                        "NoteObserver completed for %s after %ds (pending=%d)",
                        session_id, elapsed, pending,
                    )
                    break
                elif status == "failed":
                    log.warning(
                        "NoteObserver FAILED for %s after %ds",
                        session_id, elapsed,
                    )
                    break
                # Still pending or unknown — keep polling
            except Exception as e:
                log.debug("Status poll error: %s", e)

            # Check backends every 4th poll (~12s)
            if elapsed % (interval * 4) == 0:
                issues = cca.check_backends()
                if issues:
                    raise RuntimeError(
                        f"Backend unhealthy during note polling "
                        f"(waited {elapsed}s): {issues}"
                    )

    # Phase 2: Search for matching notes
    remaining = max(15, max_wait - elapsed)
    search_end = elapsed + remaining
    while elapsed < search_end:
        time.sleep(interval)
        elapsed += interval

        notes = cca.search_notes(query, user_id=user_id)
        if notes:
            log.info("Found %d notes for '%s' after %ds", len(notes), query[:30], elapsed)
            return notes

    log.warning("No notes found for '%s' after %ds total wait", query[:30], elapsed)
    return []
