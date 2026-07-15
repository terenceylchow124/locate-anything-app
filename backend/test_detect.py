"""Contract tests for POST /detect -- external behavior only, real inference
(no mock mode; this project is CPU-only end to end). Slow: each request runs
a real model inference (single-digit to tens of seconds on CPU).
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import LA_CLI_PATH, LA_MODEL_PATH, app

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_IMAGE = REPO_ROOT / "locate-anything.cpp" / "benchmarks" / "media" / "bus_in.png"

pytestmark = pytest.mark.skipif(
    not (LA_CLI_PATH.exists() and LA_MODEL_PATH.exists()),
    reason="locate-anything-cli binary or model weights not present",
)

client = TestClient(app)


def test_detect_returns_valid_contract_shape():
    with open(SAMPLE_IMAGE, "rb") as f:
        response = client.post(
            "/detect",
            files={"file": ("bus_in.png", f, "image/png")},
            data={"prompt": "person"},
        )

    assert response.status_code == 200
    body = response.json()

    assert "detections" in body
    assert "count" in body
    assert "inference_time_ms" in body
    assert "mode" in body

    assert body["count"] == len(body["detections"])
    assert body["mode"] == "hybrid"
    assert body["inference_time_ms"] > 0

    with Image.open(SAMPLE_IMAGE) as img:
        width, height = img.size

    for detection in body["detections"]:
        assert set(detection.keys()) == {"label", "box"}
        assert isinstance(detection["label"], str) and detection["label"]
        x1, y1, x2, y2 = detection["box"]
        assert 0 <= x1 <= width
        assert 0 <= x2 <= width
        assert 0 <= y1 <= height
        assert 0 <= y2 <= height


def test_detect_rejects_empty_prompt():
    with open(SAMPLE_IMAGE, "rb") as f:
        response = client.post(
            "/detect",
            files={"file": ("bus_in.png", f, "image/png")},
            data={"prompt": "  "},
        )
    assert response.status_code == 400


def test_detect_rejects_invalid_mode():
    with open(SAMPLE_IMAGE, "rb") as f:
        response = client.post(
            "/detect",
            files={"file": ("bus_in.png", f, "image/png")},
            data={"prompt": "person", "mode": "not-a-real-mode"},
        )
    assert response.status_code == 400


def test_detect_rejects_non_image_file():
    response = client.post(
        "/detect",
        files={"file": ("not-an-image.txt", b"hello world", "text/plain")},
        data={"prompt": "person"},
    )
    assert response.status_code == 400
