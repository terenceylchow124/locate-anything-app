"""Unit tests for TritonEngine -- mocks tritonclient, no GPU/Triton server
needed. Verifies request construction, response unpacking, and that every
failure mode is wrapped as la_capi.LocateAnythingError (app.py's error
handling depends on that -- see engine.py's InferenceEngine contract)."""

import json
from unittest.mock import MagicMock, patch

import pytest
from la_capi import LocateAnythingError
from triton_engine import TritonEngine
from tritonclient.utils import InferenceServerException


def _make_engine():
    with patch("triton_engine.httpclient.InferenceServerClient") as mock_client_cls:
        engine = TritonEngine("dgx-spark:8000", "locate_anything")
        mock_client = mock_client_cls.return_value
    return engine, mock_client


def _fake_result(payload: dict):
    result = MagicMock()
    result.as_numpy.return_value = [json.dumps(payload).encode()]
    return result


def test_locate_buffer_returns_detections_from_response():
    engine, mock_client = _make_engine()
    mock_client.infer.return_value = _fake_result(
        {"detections": [{"label": "screw", "box": [1.0, 2.0, 3.0, 4.0]}]}
    )

    result = engine.locate_buffer(b"fake-png-bytes", "screw", "hybrid")

    assert result == [{"label": "screw", "box": [1.0, 2.0, 3.0, 4.0]}]


def test_locate_buffer_sends_correct_model_name_and_output():
    engine, mock_client = _make_engine()
    mock_client.infer.return_value = _fake_result({"detections": []})

    engine.locate_buffer(b"bytes", "hex nut", "fast")

    call = mock_client.infer.call_args
    assert call.args[0] == "locate_anything"
    input_names = [i.name() for i in call.kwargs["inputs"]]
    assert input_names == ["IMAGE", "PROMPT", "MODE"]
    assert call.kwargs["outputs"][0].name() == "DETECTIONS_JSON"


def test_locate_buffer_wraps_inference_server_exception():
    engine, mock_client = _make_engine()
    mock_client.infer.side_effect = InferenceServerException("model not ready")

    with pytest.raises(LocateAnythingError, match="Triton inference failed"):
        engine.locate_buffer(b"bytes", "screw", "hybrid")


def test_locate_buffer_wraps_connection_errors():
    engine, mock_client = _make_engine()
    mock_client.infer.side_effect = ConnectionError("no route to host")

    with pytest.raises(LocateAnythingError, match="Triton request failed"):
        engine.locate_buffer(b"bytes", "screw", "hybrid")


def test_locate_buffer_wraps_malformed_response_payload():
    engine, mock_client = _make_engine()
    result = MagicMock()
    result.as_numpy.return_value = [b"not valid json"]
    mock_client.infer.return_value = result

    with pytest.raises(LocateAnythingError, match="unexpected DETECTIONS_JSON payload"):
        engine.locate_buffer(b"bytes", "screw", "hybrid")


def test_locate_buffer_wraps_response_missing_detections_key():
    engine, mock_client = _make_engine()
    mock_client.infer.return_value = _fake_result({"not_detections": []})

    with pytest.raises(LocateAnythingError, match="unexpected DETECTIONS_JSON payload"):
        engine.locate_buffer(b"bytes", "screw", "hybrid")


def test_constructor_wraps_client_creation_failure():
    with patch("triton_engine.httpclient.InferenceServerClient") as mock_client_cls:
        mock_client_cls.side_effect = ValueError("bad url")
        with pytest.raises(LocateAnythingError, match="failed to create Triton client"):
            TritonEngine("not a url", "locate_anything")
