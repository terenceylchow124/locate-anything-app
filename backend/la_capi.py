"""ctypes binding for locate-anything.cpp's flat C ABI (`liblocate_anything`).

Loads the model once and keeps it resident in-process, instead of shelling
out to the CLI (which reloads the ~5.9GB model on every single invocation --
prohibitive once the tile pipeline turns one /detect call into dozens of
per-tile inferences). See docs/adr/0003-persistent-capi-engine-for-tiling.md.
"""

from __future__ import annotations

import ctypes
import json
import threading
from pathlib import Path

MODE_CODES = {"hybrid": 0, "slow": 1, "fast": 2}


class LocateAnythingError(RuntimeError):
    """Raised when the underlying engine fails to load or run inference."""


class LocateAnythingEngine:
    """Wraps one loaded model context (`la_ctx*`).

    Not safe for concurrent calls from multiple threads -- the engine
    processes one request at a time. Callers must hold `lock` for the
    duration of each `locate_buffer` call.
    """

    def __init__(self, lib_path: Path, model_path: Path, n_threads: int = 0) -> None:
        self._lib = ctypes.CDLL(str(lib_path))
        self._configure_signatures()
        self._ctx = self._lib.la_capi_load(str(model_path).encode(), n_threads)
        if not self._ctx:
            raise LocateAnythingError(f"failed to load model at {model_path}")
        self.lock = threading.Lock()

    def _configure_signatures(self) -> None:
        lib = self._lib
        lib.la_capi_load.restype = ctypes.c_void_p
        lib.la_capi_load.argtypes = [ctypes.c_char_p, ctypes.c_int]
        lib.la_capi_free.argtypes = [ctypes.c_void_p]
        lib.la_capi_locate_buffer.restype = ctypes.c_void_p
        lib.la_capi_locate_buffer.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        lib.la_capi_free_string.argtypes = [ctypes.c_void_p]
        lib.la_capi_last_error.restype = ctypes.c_char_p
        lib.la_capi_last_error.argtypes = [ctypes.c_void_p]

    def locate_buffer(self, image_bytes: bytes, prompt: str, mode: str) -> list[dict]:
        """Run detection on an in-memory encoded image (e.g. PNG bytes).

        Caller must hold `self.lock` for the duration of this call.
        """
        result_ptr = self._lib.la_capi_locate_buffer(
            self._ctx,
            image_bytes,
            len(image_bytes),
            prompt.encode(),
            MODE_CODES[mode],
        )
        if not result_ptr:
            error = self._lib.la_capi_last_error(self._ctx)
            message = error.decode() if error else "unknown inference error"
            raise LocateAnythingError(message)

        raw_json = ctypes.cast(result_ptr, ctypes.c_char_p).value
        self._lib.la_capi_free_string(result_ptr)
        return json.loads(raw_json)["detections"]

    def close(self) -> None:
        self._lib.la_capi_free(self._ctx)
        self._ctx = None
