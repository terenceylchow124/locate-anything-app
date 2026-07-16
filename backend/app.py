"""Thin HTTP wrapper around locate-anything.cpp's C API, exposed as a single
POST /detect endpoint -- the one seam the rest of the system is built against.

Dense/large images are split into overlapping tiles (see tiling.py); each
tile is run through the persistent, in-process model engine (la_capi.py) and
results are merged across tile boundaries by IoU. This is a backend-internal
implementation detail -- the API contract is a single image in, one set of
detections out.
"""

from __future__ import annotations

import asyncio
import io
import os
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from la_capi import LocateAnythingEngine, LocateAnythingError
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from tiling import DEDUP_IOU_THRESHOLD, BoxDetection, generate_tiles, merge_detections

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
VALID_MODES = {"hybrid", "slow", "fast"}

app = FastAPI(title="LocateAnything-3B detect API")

# Guards access to the single loaded engine -- inference is effectively
# single-worker (one CPU-bound call at a time). This is a minimal safety
# gate, not the full queue/wait-state UX; see ticket #02b for that.
_detect_lock = asyncio.Lock()
_engine: LocateAnythingEngine | None = None
_engine_init_lock = threading.Lock()


class Detection(BaseModel):
    label: str
    box: list[float]


class DetectResponse(BaseModel):
    detections: list[Detection]
    count: int
    inference_time_ms: int
    mode: str


def get_engine() -> LocateAnythingEngine:
    global _engine
    if _engine is None:
        with _engine_init_lock:
            if _engine is None:
                _engine = LocateAnythingEngine(LA_LIB_PATH, LA_MODEL_PATH)
    return _engine


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

    tiles = generate_tiles(width, height)
    all_detections: list[BoxDetection] = []
    elapsed_ms = 0

    for tile in tiles:
        crop = image.crop((tile.x0, tile.y0, tile.x1, tile.y1))
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        tile_bytes = buf.getvalue()

        start = time.perf_counter()
        try:
            async with _detect_lock:
                raw_detections = await run_in_threadpool(
                    engine.locate_buffer, tile_bytes, prompt, mode
                )
        except LocateAnythingError as exc:
            raise HTTPException(status_code=500, detail=f"inference failed: {exc}") from exc
        elapsed_ms += int((time.perf_counter() - start) * 1000)

        crop_width, crop_height = tile.x1 - tile.x0, tile.y1 - tile.y0
        for d in raw_detections:
            x1, y1, x2, y2 = d["box"]
            # The model's box regression can slightly overshoot the input
            # image's own edges (observed: a few % on a tile near the true
            # object boundary) -- clamp to the crop actually fed to the
            # model before translating to whole-image coordinates, rather
            # than treating natural coordinate imprecision as an error.
            x1 = max(0.0, min(x1, crop_width))
            x2 = max(0.0, min(x2, crop_width))
            y1 = max(0.0, min(y1, crop_height))
            y2 = max(0.0, min(y2, crop_height))
            all_detections.append(
                BoxDetection(
                    label=d["label"],
                    box=[x1 + tile.x0, y1 + tile.y0, x2 + tile.x0, y2 + tile.y0],
                )
            )

    merged = merge_detections(all_detections, DEDUP_IOU_THRESHOLD)
    detections = [Detection(label=m.label, box=m.box) for m in merged]

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
    )
