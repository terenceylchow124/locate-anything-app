"""Thin HTTP wrapper around locate-anything.cpp's C API, exposed as a single
POST /detect endpoint -- the one seam the rest of the system is built against.

Dense/large images are split into overlapping tiles (see tiling.py); each
tile is run through the persistent, in-process model engine (la_capi.py) and
results are merged across tile boundaries by IoU. This is a backend-internal
implementation detail -- the API contract is a single image in, one set of
detections out.
"""

from __future__ import annotations

import io
import os
import threading
import time
from pathlib import Path

from engine import InferenceEngine
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from la_capi import MODE_CODES, LocateAnythingEngine, LocateAnythingError
from modal_engine import ModalEngine
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from queueing import SingleWorkerQueue
from starlette.concurrency import run_in_threadpool
from tiling import (
    DEDUP_IOU_THRESHOLD,
    BoxDetection,
    generate_tiles,
    merge_detections,
    translate_and_clamp,
)
from triton_engine import TritonEngine

REPO_ROOT = Path(__file__).resolve().parent.parent
LA_LIB_PATH = Path(
    os.environ.get(
        "LA_LIB_PATH",
        REPO_ROOT / "locate-anything.cpp" / "build" / "liblocate_anything.so",
    )
)
LA_MODEL_PATH = Path(
    os.environ.get(
        "LA_MODEL_PATH",
        REPO_ROOT / "locate-anything.cpp" / "models" / "locate-anything-q8_0.gguf",
    )
)
VALID_MODES = set(MODE_CODES)

# Which InferenceEngine to construct: "local" (default, locate-anything.cpp
# via ctypes, la_capi.py), "triton" (self-hosted Triton Inference Server on a
# GPU box, triton_engine.py), or "modal" (Modal-hosted GPU deployment,
# modal_engine.py) -- see docs/adr/0005-pluggable-inference-backend-local-or-triton.md
# and modal_app/model_server.py.
LA_INFERENCE_BACKEND = os.environ.get("LA_INFERENCE_BACKEND", "local")
LA_TRITON_URL = os.environ.get("LA_TRITON_URL", "localhost:8000")
LA_TRITON_MODEL_NAME = os.environ.get("LA_TRITON_MODEL_NAME", "locate_anything")
LA_MODAL_URL = os.environ.get("LA_MODAL_URL", "")
LA_MODAL_TOKEN = os.environ.get("LA_MODAL_TOKEN", "")

app = FastAPI(title="LocateAnything-3B detect API")

# Local portfolio demo, no auth/user data -- open CORS so the frontend (a
# separately-served static build, see ADR 0002) can call this from any origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Serializes whole /detect requests (not just individual tile calls) through
# the single loaded engine, and reports how long a queued request waited for
# its turn -- see ticket #02b and backend/queueing.py.
_detect_queue = SingleWorkerQueue()
_engine: InferenceEngine | None = None
_engine_init_lock = threading.Lock()


class Detection(BaseModel):
    label: str
    box: list[float]
    score: float | None = None


class DetectResponse(BaseModel):
    detections: list[Detection]
    count: int
    inference_time_ms: int
    mode: str
    queue_wait_ms: int


def get_engine() -> InferenceEngine:
    global _engine
    if _engine is None:
        with _engine_init_lock:
            if _engine is None:
                if LA_INFERENCE_BACKEND == "triton":
                    _engine = TritonEngine(LA_TRITON_URL, LA_TRITON_MODEL_NAME)
                elif LA_INFERENCE_BACKEND == "modal":
                    _engine = ModalEngine(LA_MODAL_URL, LA_MODAL_TOKEN)
                else:
                    _engine = LocateAnythingEngine(LA_LIB_PATH, LA_MODEL_PATH)
    return _engine


async def run_detection(
    engine: InferenceEngine, image: Image.Image, prompt: str, mode: str
) -> tuple[list[Detection], int]:
    """Tile the image, run each tile through the engine, merge results.

    Runs as the single unit of work handed to `_detect_queue` -- the whole
    thing completes before the next queued request's turn starts (ticket
    #02b), not just each individual tile call.

    Not HTTP-specific: lets `LocateAnythingError` propagate rather than
    raising `HTTPException` itself -- the route handler owns that
    translation (see `docs/CODING_STANDARDS.md`'s Structure rule).
    """
    tiles = generate_tiles(*image.size)
    all_detections: list[BoxDetection] = []
    elapsed_ms = 0

    for tile in tiles:
        crop = image.crop((tile.x0, tile.y0, tile.x1, tile.y1))
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        tile_bytes = buf.getvalue()

        start = time.perf_counter()
        raw_detections = await run_in_threadpool(engine.locate_buffer, tile_bytes, prompt, mode)
        elapsed_ms += int((time.perf_counter() - start) * 1000)
        all_detections.extend(translate_and_clamp(raw_detections, tile))

    merged = merge_detections(all_detections, DEDUP_IOU_THRESHOLD)
    detections = [Detection(label=m.label, box=m.box, score=m.score) for m in merged]
    return detections, elapsed_ms


@app.post("/detect", response_model=DetectResponse)
async def detect(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    mode: str = Form("hybrid"),
) -> DetectResponse:
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="prompt must not be empty")
    if mode not in VALID_MODES:
        raise HTTPException(status_code=400, detail=f"mode must be one of {sorted(VALID_MODES)}")
    if LA_INFERENCE_BACKEND == "local":
        if not LA_LIB_PATH.exists():
            raise HTTPException(
                status_code=500, detail=f"liblocate_anything not found at {LA_LIB_PATH}"
            )
        if not LA_MODEL_PATH.exists():
            raise HTTPException(status_code=500, detail=f"model not found at {LA_MODEL_PATH}")

    try:
        engine = get_engine()
    except LocateAnythingError as exc:
        raise HTTPException(status_code=500, detail=f"failed to load engine: {exc}") from exc

    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="uploaded file is not a valid image") from exc
    image = image.convert("RGB")
    width, height = image.size

    try:
        (detections, elapsed_ms), queue_wait_ms = await _detect_queue.run(
            lambda: run_detection(engine, image, prompt, mode)
        )
    except LocateAnythingError as exc:
        raise HTTPException(status_code=500, detail=f"inference failed: {exc}") from exc

    # Should be unreachable -- every box is clamped to its tile's crop
    # before translation, and tiles never extend past the image (see
    # tiling.generate_tiles) -- kept as a defensive safety net.
    for d in detections:
        x1, y1, x2, y2 = d.box
        if not (0 <= x1 <= width and 0 <= x2 <= width and 0 <= y1 <= height and 0 <= y2 <= height):
            raise HTTPException(
                status_code=500,
                detail=f"detection box {d.box} out of image bounds ({width}x{height})",
            )

    return DetectResponse(
        detections=detections,
        count=len(detections),
        inference_time_ms=elapsed_ms,
        mode=mode,
        queue_wait_ms=queue_wait_ms,
    )
