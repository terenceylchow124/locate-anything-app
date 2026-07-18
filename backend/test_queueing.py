"""Unit tests for SingleWorkerQueue -- fast, no model/CLI dependency. Uses
asyncio.sleep as a stand-in for real inference work."""

import asyncio

import pytest
from queueing import SingleWorkerQueue

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def test_run_returns_result_and_near_zero_wait_when_uncontended():
    queue = SingleWorkerQueue()

    async def work():
        return "done"

    result, queue_wait_ms = await queue.run(work)

    assert result == "done"
    assert queue_wait_ms < 20  # generous margin; nothing was ahead of it


async def test_concurrent_callers_run_one_at_a_time_not_in_parallel():
    queue = SingleWorkerQueue()
    currently_running = 0
    max_concurrent = 0

    async def work():
        nonlocal currently_running, max_concurrent
        currently_running += 1
        max_concurrent = max(max_concurrent, currently_running)
        await asyncio.sleep(0.05)
        currently_running -= 1
        return "ok"

    await asyncio.gather(queue.run(work), queue.run(work), queue.run(work))

    assert max_concurrent == 1


async def test_second_caller_reports_wait_time_for_first_callers_work():
    queue = SingleWorkerQueue()

    async def slow_work():
        await asyncio.sleep(0.1)
        return "first"

    async def fast_work():
        return "second"

    first_task = asyncio.create_task(queue.run(slow_work))
    await asyncio.sleep(0.01)  # let first_task grab the lock first
    second_task = asyncio.create_task(queue.run(fast_work))

    (first_result, first_wait), (second_result, second_wait) = await asyncio.gather(
        first_task, second_task
    )

    assert first_result == "first"
    assert first_wait < 20  # nothing ahead of it
    assert second_result == "second"
    assert second_wait >= 80  # waited for most of first_work's ~100ms


async def test_a_failing_caller_does_not_break_the_queue_for_the_next_one():
    queue = SingleWorkerQueue()

    async def failing_work():
        raise ValueError("boom")

    async def ok_work():
        return "still fine"

    with pytest.raises(ValueError):
        await queue.run(failing_work)

    result, queue_wait_ms = await queue.run(ok_work)
    assert result == "still fine"
    assert queue_wait_ms < 20
