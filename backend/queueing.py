"""Generic bounded-concurrency queue: runs up to `concurrency` callers' `fn`
at once through a semaphore, and reports how long each one waited for a slot.

Decoupled from the inference engine so the serialization/wait-time logic is
unit-testable without a real model call (ticket #02b) -- see test_queueing.py.
Default concurrency=1 preserves the original one-at-a-time behavior (the
class name is a holdover from when that was the only mode -- see app.py's
LA_REQUEST_CONCURRENCY for why higher values are now safe for triton/modal).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


class SingleWorkerQueue:
    """Runs up to `concurrency` callers' `fn` at once; extra callers queue.

    `run` returns `(result, queue_wait_ms)` -- `queue_wait_ms` is ~0 when the
    caller got a free slot immediately, and reflects how long it waited
    otherwise. Position isn't reported live (that would need a
    streaming/polling response, out of scope -- see ticket #02b's grilling);
    the caller finds out how long it waited once its own turn completes.
    """

    def __init__(self, concurrency: int = 1) -> None:
        self._semaphore = asyncio.Semaphore(concurrency)

    async def run(self, fn: Callable[[], Awaitable[T]]) -> tuple[T, int]:
        start = time.perf_counter()
        async with self._semaphore:
            queue_wait_ms = int((time.perf_counter() - start) * 1000)
            result = await fn()
        return result, queue_wait_ms
