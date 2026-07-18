"""Fast integration test for the /detect single-worker queue (ticket #02b) --
mocks the engine so it needs no real model/CLI, unlike test_detect.py."""

import asyncio
import time
from pathlib import Path

import app as app_module
import pytest
from httpx import ASGITransport, AsyncClient
from queueing import SingleWorkerQueue

pytestmark = pytest.mark.anyio

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_IMAGE = REPO_ROOT / "locate-anything.cpp" / "benchmarks" / "media" / "bus_in.png"


@pytest.fixture
def anyio_backend():
    return "asyncio"


class FakeEngine:
    """Stands in for LocateAnythingEngine: blocks briefly per call (as if it
    were CPU-bound inference) and records call order to prove serialization."""

    def __init__(self, call_log: list[str], work_seconds: float = 0.1) -> None:
        self.call_log = call_log
        self.work_seconds = work_seconds

    def locate_buffer(self, image_bytes: bytes, prompt: str, mode: str) -> list[dict]:
        self.call_log.append(f"start:{prompt}")
        time.sleep(self.work_seconds)
        self.call_log.append(f"end:{prompt}")
        return []


@pytest.fixture
def fake_engine_setup(monkeypatch):
    call_log: list[str] = []
    engine = FakeEngine(call_log)
    monkeypatch.setattr(app_module, "LA_LIB_PATH", Path(__file__))
    monkeypatch.setattr(app_module, "LA_MODEL_PATH", Path(__file__))
    monkeypatch.setattr(app_module, "get_engine", lambda: engine)
    # The module-level queue's asyncio.Lock binds to whichever event loop
    # first uses it; each test here runs in its own event loop (anyio,
    # function-scoped), so reuse of the same global lock object across tests
    # would raise "bound to a different event loop." Not a real app bug --
    # a real deployment has exactly one event loop for its whole lifetime --
    # just test isolation, fixed with a fresh queue per test.
    monkeypatch.setattr(app_module, "_detect_queue", SingleWorkerQueue())
    return call_log


async def _post_detect(client: AsyncClient, prompt: str):
    with open(SAMPLE_IMAGE, "rb") as f:
        return await client.post(
            "/detect",
            files={"file": ("bus_in.png", f, "image/png")},
            data={"prompt": prompt},
        )


async def test_concurrent_requests_are_serialized_not_interleaved(fake_engine_setup):
    call_log = fake_engine_setup
    transport = ASGITransport(app=app_module.app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response_a, response_b = await asyncio.gather(
            _post_detect(client, "first"),
            _post_detect(client, "second"),
        )

    assert response_a.status_code == 200
    assert response_b.status_code == 200

    # Whichever request went first, its start/end pair must not straddle the
    # other's -- i.e. no [start:a, start:b, end:a, end:b] interleaving. Each
    # tile of bus_in.png is one locate_buffer call, so a fully-serialized run
    # looks like start:X, end:X, start:X, end:X, ... (repeated per tile) then
    # the same for Y -- never a start for the other prompt before this
    # prompt's tiles are all done.
    first_prompt = call_log[0].split(":")[1]
    other_prompt = "second" if first_prompt == "first" else "first"
    switch_index = next(i for i, entry in enumerate(call_log) if other_prompt in entry)
    assert all(first_prompt in entry for entry in call_log[:switch_index])


async def test_second_concurrent_request_reports_nonzero_queue_wait(fake_engine_setup):
    transport = ASGITransport(app=app_module.app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response_a, response_b = await asyncio.gather(
            _post_detect(client, "first"),
            _post_detect(client, "second"),
        )

    wait_times = sorted([response_a.json()["queue_wait_ms"], response_b.json()["queue_wait_ms"]])
    assert wait_times[0] < 50  # whichever ran first waited ~nothing
    assert wait_times[1] > 50  # the other waited for the first's tiles to finish


async def test_single_request_has_near_zero_queue_wait(fake_engine_setup):
    transport = ASGITransport(app=app_module.app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await _post_detect(client, "only one")

    assert response.status_code == 200
    assert response.json()["queue_wait_ms"] < 50
