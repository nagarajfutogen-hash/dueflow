"""Injectable clock. Policy code never reads the system time; callers pass `clock.now()`."""

from datetime import UTC, datetime
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class FixedClock:
    """Simulated time for outbound-call simulations and tests."""

    def __init__(self, at: datetime) -> None:
        if at.tzinfo is None:
            raise ValueError("FixedClock needs a timezone-aware datetime")
        self._at = at

    def now(self) -> datetime:
        return self._at


class OffsetClock:
    """Simulated time that advances in real time from a simulated start (e.g. the dial time)."""

    def __init__(self, start: datetime, real: Clock | None = None) -> None:
        if start.tzinfo is None:
            raise ValueError("OffsetClock needs a timezone-aware start")
        self._start = start
        self._real = real or SystemClock()
        self._t0 = self._real.now()

    def now(self) -> datetime:
        return self._start + (self._real.now() - self._t0)
