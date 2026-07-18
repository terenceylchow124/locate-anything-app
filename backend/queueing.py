"""Generic single-worker queue: serializes concurrent async callers through
one lock and reports how long each one waited for its turn.

Decoupled from the inference engine so the serialization/wait-time logic is
unit-testable without a real model call (ticket #02b) -- see test_queueing.py.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


class SingleWorkerQueue:
    """Runs one caller's `fn` at a time; concurrent callers queue behind it.

    `run` returns `(result, queue_wait_ms)` -- `queue_wait_ms` is ~0 when the
    caller didn't have to wait, and reflects how long it waited in line
    otherwise. Position isn't reported live (that would need a
    streaming/polling response, out of scope -- see ticket #02b's grilling);
    the caller finds out how long it waited once its own turn completes.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    async def run(self, fn: Callable[[], Awaitable[T]]) -> tuple[T, int]:
        start = time.perf_counter()
        async with self._lock:
            queue_wait_ms = int((time.perf_counter() - start) * 1000)
            result = await fn()
        return result, queue_wait_ms
