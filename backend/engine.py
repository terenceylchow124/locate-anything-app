"""Shared inference-engine contract.

`app.py`'s `run_detection` calls exactly one method on whichever engine is
configured (`LA_INFERENCE_BACKEND`) -- this Protocol pins that contract so
neither `app.py` nor a new engine implementation needs to depend on the
other's concrete class. Both `LocateAnythingEngine` (la_capi.py, local CPU)
and `TritonEngine` (triton_engine.py, remote GPU) satisfy it structurally.
"""

from __future__ import annotations

from typing import Protocol


class InferenceEngine(Protocol):
    def locate_buffer(self, image_bytes: bytes, prompt: str, mode: str) -> list[dict]:
        """Run detection on an in-memory encoded image (e.g. PNG bytes).

        Returns a list of `{"label": str, "box": [x1, y1, x2, y2]}` dicts in
        the *input image's own* pixel coordinates (tile-local, not the whole
        original image -- translation to whole-image coordinates happens in
        `tiling.translate_and_clamp`). Must raise `la_capi.LocateAnythingError`
        on any failure (connection, inference, or otherwise) so `app.py`'s
        existing error handling catches it.
        """
        ...
