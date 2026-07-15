"""Thin HTTP wrapper around locate-anything.cpp's CLI, exposed as a single
POST /detect endpoint -- the one seam the rest of the system is built against.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent
LA_CLI_PATH = Path(
    os.environ.get(
        "LA_CLI_PATH",
        REPO_ROOT / "locate-anything.cpp" / "build" / "examples" / "cli" / "locate-anything-cli",
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


class Detection(BaseModel):
    label: str
    box: list[float]


class DetectResponse(BaseModel):
    detections: list[Detection]
    count: int
    inference_time_ms: int
    mode: str


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
    if not LA_CLI_PATH.exists():
        raise HTTPException(status_code=500, detail=f"locate-anything-cli not found at {LA_CLI_PATH}")
    if not LA_MODEL_PATH.exists():
        raise HTTPException(status_code=500, detail=f"model not found at {LA_MODEL_PATH}")

    image_bytes = await file.read()
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        image_path = tmp_dir_path / "input.png"
        output_path = tmp_dir_path / "output.json"

        image_path.write_bytes(image_bytes)
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except UnidentifiedImageError as exc:
            raise HTTPException(status_code=400, detail="uploaded file is not a valid image") from exc

        cmd = [
            str(LA_CLI_PATH),
            "detect",
            "--model",
            str(LA_MODEL_PATH),
            "--input",
            str(image_path),
            "--prompt",
            prompt,
            "--mode",
            mode,
            "--output",
            str(output_path),
        ]

        start = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"inference failed: {result.stderr.strip() or result.stdout.strip()}",
            )

        try:
            raw = json.loads(output_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=500, detail="inference produced no valid output") from exc

    detections = [Detection(label=d["label"], box=d["box"]) for d in raw.get("detections", [])]
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
