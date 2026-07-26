"""Unit tests for app.get_engine()'s backend selection (LA_INFERENCE_BACKEND)
-- no real model/CLI/GPU needed, both engine classes are constructed but
never actually called."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import app as app_module
from la_capi import LocateAnythingEngine
from modal_engine import ModalEngine
from triton_engine import TritonEngine


def test_get_engine_defaults_to_local(monkeypatch):
    monkeypatch.setattr(app_module, "_engine", None)
    monkeypatch.setattr(app_module, "LA_INFERENCE_BACKEND", "local")
    with patch.object(app_module, "LocateAnythingEngine") as mock_local:
        app_module.get_engine()
        mock_local.assert_called_once_with(app_module.LA_LIB_PATH, app_module.LA_MODEL_PATH)


def test_get_engine_uses_triton_when_configured(monkeypatch):
    monkeypatch.setattr(app_module, "_engine", None)
    monkeypatch.setattr(app_module, "LA_INFERENCE_BACKEND", "triton")
    monkeypatch.setattr(app_module, "LA_TRITON_URL", "dgx-spark:8000")
    monkeypatch.setattr(app_module, "LA_TRITON_MODEL_NAME", "locate_anything")
    with patch.object(app_module, "TritonEngine") as mock_triton:
        app_module.get_engine()
        mock_triton.assert_called_once_with("dgx-spark:8000", "locate_anything")


def test_get_engine_uses_modal_when_configured(monkeypatch):
    monkeypatch.setattr(app_module, "_engine", None)
    monkeypatch.setattr(app_module, "LA_INFERENCE_BACKEND", "modal")
    monkeypatch.setattr(app_module, "LA_MODAL_URL", "https://example.modal.run/detect")
    monkeypatch.setattr(app_module, "LA_MODAL_TOKEN", "shh-token")
    with patch.object(app_module, "ModalEngine") as mock_modal:
        app_module.get_engine()
        mock_modal.assert_called_once_with(
            "https://example.modal.run/detect", "shh-token"
        )


def test_get_engine_only_constructs_once(monkeypatch):
    monkeypatch.setattr(app_module, "_engine", None)
    monkeypatch.setattr(app_module, "LA_INFERENCE_BACKEND", "local")
    with patch.object(app_module, "LocateAnythingEngine") as mock_local:
        first = app_module.get_engine()
        second = app_module.get_engine()
        assert first is second
        mock_local.assert_called_once()


def test_engine_classes_are_the_expected_concrete_types():
    # Sanity check that the engine classes app.py imports are the real ones
    # (not accidentally shadowed) -- catches an import-path typo that the
    # mocked tests above wouldn't.
    assert app_module.LocateAnythingEngine is LocateAnythingEngine
    assert app_module.TritonEngine is TritonEngine
    assert app_module.ModalEngine is ModalEngine


def test_local_backend_rejects_tile_concurrency_above_one():
    # la_capi.LocateAnythingEngine isn't safe for concurrent calls (single
    # loaded model instance) -- app.py must fail at import/startup time
    # rather than silently letting LA_TILE_CONCURRENCY>1 corrupt local-mode
    # inference. Runs app.py as a subprocess (not monkeypatch) since the
    # check runs at module import time, before any patching could apply.
    backend_dir = Path(__file__).resolve().parent
    result = subprocess.run(
        [sys.executable, "-c", "import app"],
        cwd=backend_dir,
        env={"LA_INFERENCE_BACKEND": "local", "LA_TILE_CONCURRENCY": "4", "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "LA_TILE_CONCURRENCY > 1 is not supported with LA_INFERENCE_BACKEND=local" in result.stderr
