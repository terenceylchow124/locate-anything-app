"""Client for a Modal-hosted deployment of the original NVIDIA
LocateAnything-3B PyTorch checkpoint (GPU) -- an alternative to both the
local CPU engine (la_capi.py) and the self-hosted Triton engine
(triton_engine.py), selected via LA_INFERENCE_BACKEND.

The Modal-side model (see modal_app/model_server.py) returns the exact same
{"label", "box"} tile-local-pixel-coordinate shape the other two engines
return, so this class is a drop-in InferenceEngine (engine.py) -- app.py,
tiling.py, and queueing.py need no awareness of which backend is in use.
"""

from __future__ import annotations

import httpx
from la_capi import LocateAnythingError


class ModalEngine:
    """Calls a Modal web endpoint over HTTPS.

    Unlike LocateAnythingEngine, this holds no local model state -- each
    call is a network request. Still expected to be used from behind
    app.py's single-worker queue like the other engines (this app always
    processes one /detect request at a time; see ticket #02b), not because
    this class itself requires it.
    """

    def __init__(self, url: str, token: str) -> None:
        if not url:
            raise LocateAnythingError("LA_MODAL_URL is not set")
        if not token:
            raise LocateAnythingError("LA_MODAL_TOKEN is not set")
        self._url = url
        # Modal cold starts (container spin-up + model load onto a fresh GPU)
        # can take well over a minute; a scaled-down container plus tile
        # fan-out for a dense image makes 5 min the same reasoned ceiling
        # triton_engine.py uses for the same reason.
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {token}"}, timeout=300.0
        )

    def locate_buffer(self, image_bytes: bytes, prompt: str, mode: str) -> list[dict]:
        try:
            response = self._client.post(
                self._url,
                files={"file": ("tile.png", image_bytes, "image/png")},
                data={"prompt": prompt, "mode": mode},
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise LocateAnythingError(
                f"Modal endpoint returned {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.HTTPError as exc:
            raise LocateAnythingError(f"Modal request failed: {exc}") from exc

        try:
            return response.json()["detections"]
        except Exception as exc:
            raise LocateAnythingError(
                f"Modal returned an unexpected or malformed response: {exc}"
            ) from exc
