"""Contract tests for POST /detect -- external behavior only, real inference
(no mock mode; this project is CPU-only end to end).

Slow: each request tiles the image and runs one real model inference per
tile (~80s/tile on this dev machine, dominated by the vision encoder + LM
decode, not model loading -- see docs/adr/0003). bus_in.png (812x1092) tiles
into 6 tiles, so the contract-shape test alone takes on the order of 8-10
minutes; budget accordingly when running the full suite.
"""

import os
from pathlib import Path

import pytest
from app import LA_LIB_PATH, LA_MODEL_PATH, app
from fastapi.testclient import TestClient
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_IMAGE = REPO_ROOT / "locate-anything.cpp" / "benchmarks" / "media" / "bus_in.png"
CALIBRATION_IMAGE = REPO_ROOT / "backend" / "tests" / "fixtures" / "dense_screws.jpg"

pytestmark = pytest.mark.skipif(
    not (LA_LIB_PATH.exists() and LA_MODEL_PATH.exists()),
    reason="liblocate_anything shared library or model weights not present",
)

client = TestClient(app)


@pytest.mark.slow
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


@pytest.mark.slow
@pytest.mark.calibration
@pytest.mark.skipif(
    os.environ.get("RUN_CALIBRATION") != "1",
    reason="extremely slow (~30 tiles x ~80s/tile, 30-40+ min) -- opt in with RUN_CALIBRATION=1",
)
def test_detect_screws_count_within_calibrated_range():
    """Known-answer regression check: dense_screws.jpg, prompt 'screw', should
    land in the 80-110 expected_count_range settled during ticket #03/#05
    grilling (manual ground truth ~95, see backend/tests/fixtures/SOURCES.md).
    Also asserts the count isn't obviously inflated by cross-tile
    double-counting (the dedup this ticket exists to verify).
    """
    with open(CALIBRATION_IMAGE, "rb") as f:
        response = client.post(
            "/detect",
            files={"file": ("dense_screws.jpg", f, "image/jpeg")},
            data={"prompt": "screw"},
        )

    assert response.status_code == 200
    body = response.json()
    assert 80 <= body["count"] <= 110
