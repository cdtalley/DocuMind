"""Process-level timestamps for diagnostics (keeps routers free of import cycles with app.main)."""

from __future__ import annotations

from datetime import UTC, datetime

_started_at_utc: datetime | None = None


def mark_process_started() -> None:
    global _started_at_utc
    if _started_at_utc is None:
        _started_at_utc = datetime.now(UTC)


def process_started_at_utc() -> datetime | None:
    return _started_at_utc
