"""Client for a remote Triton Inference Server serving the original NVIDIA
LocateAnything-3B PyTorch checkpoint (GPU) -- an alternative to the local
CPU engine (la_capi.py), selected via LA_INFERENCE_BACKEND.

The Triton-side model (see triton/model_repository/locate_anything/) returns
the exact same {"label", "box"} tile-local-pixel-coordinate shape the local
engine returns, so this class is a drop-in InferenceEngine (engine.py) --
app.py, tiling.py, and queueing.py need no awareness of which backend is
in use.
"""

from __future__ import annotations

import json

import numpy as np
import tritonclient.http as httpclient
from la_capi import LocateAnythingError
from tritonclient.utils import InferenceServerException


class TritonEngine:
    """Calls a Triton Inference Server model over HTTP.

    Unlike LocateAnythingEngine, this holds no local model state -- each
    call is a network request. Still expected to be used from behind
    app.py's single-worker queue like the local engine (this app always
    processes one /detect request at a time; see ticket #02b), not because
    this class itself requires it.
    """

    def __init__(self, url: str, model_name: str) -> None:
        self._model_name = model_name
        try:
            self._client = httpclient.InferenceServerClient(url=url)
        except Exception as exc:  # tritonclient raises plain Exception here
            raise LocateAnythingError(f"failed to create Triton client for {url}: {exc}") from exc

    def locate_buffer(self, image_bytes: bytes, prompt: str, mode: str) -> list[dict]:
        image_input = httpclient.InferInput("IMAGE", [1], "BYTES")
        image_input.set_data_from_numpy(np.array([image_bytes], dtype=object))

        prompt_input = httpclient.InferInput("PROMPT", [1], "BYTES")
        prompt_input.set_data_from_numpy(np.array([prompt.encode()], dtype=object))

        mode_input = httpclient.InferInput("MODE", [1], "BYTES")
        mode_input.set_data_from_numpy(np.array([mode.encode()], dtype=object))

        output = httpclient.InferRequestedOutput("DETECTIONS_JSON")

        try:
            result = self._client.infer(
                self._model_name,
                inputs=[image_input, prompt_input, mode_input],
                outputs=[output],
            )
        except InferenceServerException as exc:
            raise LocateAnythingError(f"Triton inference failed: {exc}") from exc
        except Exception as exc:  # connection errors etc. from the http client
            raise LocateAnythingError(f"Triton request failed: {exc}") from exc

        raw = result.as_numpy("DETECTIONS_JSON")[0]
        detections_json = raw.decode() if isinstance(raw, bytes) else raw
        try:
            return json.loads(detections_json)["detections"]
        except (json.JSONDecodeError, KeyError) as exc:
            raise LocateAnythingError(
                f"Triton returned an unexpected DETECTIONS_JSON payload: {detections_json!r}"
            ) from exc
