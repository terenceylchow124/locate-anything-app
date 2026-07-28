"""Unit tests for ModalEngine -- mocks httpx, no live Modal endpoint needed.
Verifies request construction, response unpacking, and that every failure
mode is wrapped as la_capi.LocateAnythingError (app.py's error handling
depends on that -- see engine.py's InferenceEngine contract)."""

from unittest.mock import MagicMock, patch

import httpx
import pytest
from la_capi import LocateAnythingError
from modal_engine import ModalEngine


def _make_engine():
    with patch("modal_engine.httpx.Client") as mock_client_cls:
        engine = ModalEngine("https://example.modal.run/detect", "shh-token")
        mock_client = mock_client_cls.return_value
    return engine, mock_client


def _fake_response(payload: dict, status_code: int = 200):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload
    response.raise_for_status = MagicMock()
    if status_code >= 400:
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=response
        )
    return response


def test_constructor_requires_url_and_token():
    with pytest.raises(LocateAnythingError, match="LA_MODAL_URL"):
        ModalEngine("", "token")
    with pytest.raises(LocateAnythingError, match="LA_MODAL_TOKEN"):
        ModalEngine("https://example.modal.run/detect", "")


def test_locate_buffer_returns_detections_from_response():
    engine, mock_client = _make_engine()
    mock_client.post.return_value = _fake_response(
        {"detections": [{"label": "screw", "box": [1.0, 2.0, 3.0, 4.0]}]}
    )

    result = engine.locate_buffer(b"fake-png-bytes", "screw", "hybrid")

    assert result == [{"label": "screw", "box": [1.0, 2.0, 3.0, 4.0]}]


def test_locate_buffer_sends_correct_fields():
    engine, mock_client = _make_engine()
    mock_client.post.return_value = _fake_response({"detections": []})

    engine.locate_buffer(b"bytes", "hex nut", "fast")

    call = mock_client.post.call_args
    assert call.args[0] == "https://example.modal.run/detect"
    assert call.kwargs["files"]["file"][1] == b"bytes"
    assert call.kwargs["data"] == {"prompt": "hex nut", "mode": "fast"}


def test_locate_buffer_wraps_http_status_error():
    engine, mock_client = _make_engine()
    mock_client.post.return_value = _fake_response({"detail": "bad token"}, status_code=401)

    with pytest.raises(LocateAnythingError, match="Modal endpoint returned 401"):
        engine.locate_buffer(b"bytes", "screw", "hybrid")


def test_locate_buffer_wraps_connection_errors():
    engine, mock_client = _make_engine()
    mock_client.post.side_effect = httpx.ConnectError("no route to host")

    with pytest.raises(LocateAnythingError, match="Modal request failed"):
        engine.locate_buffer(b"bytes", "screw", "hybrid")


def test_locate_buffer_wraps_malformed_response_payload():
    engine, mock_client = _make_engine()
    response = _fake_response({"not_detections": []})
    mock_client.post.return_value = response

    with pytest.raises(LocateAnythingError, match="unexpected or malformed response"):
        engine.locate_buffer(b"bytes", "screw", "hybrid")
