"""Unit tests for app.get_engine()'s backend selection (LA_INFERENCE_BACKEND)
-- no real model/CLI/GPU needed, both engine classes are constructed but
never actually called."""

from unittest.mock import patch

import app as app_module
from la_capi import LocateAnythingEngine
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


def test_get_engine_only_constructs_once(monkeypatch):
    monkeypatch.setattr(app_module, "_engine", None)
    monkeypatch.setattr(app_module, "LA_INFERENCE_BACKEND", "local")
    with patch.object(app_module, "LocateAnythingEngine") as mock_local:
        first = app_module.get_engine()
        second = app_module.get_engine()
        assert first is second
        mock_local.assert_called_once()


def test_engine_classes_are_the_expected_concrete_types():
    # Sanity check that the two engine classes app.py imports are the real
    # ones (not accidentally shadowed) -- catches an import-path typo that
    # the mocked tests above wouldn't.
    assert app_module.LocateAnythingEngine is LocateAnythingEngine
    assert app_module.TritonEngine is TritonEngine
