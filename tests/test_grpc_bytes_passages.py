"""Verify ``Passage`` encoding + wire round-trip for text and bytes shapes.

PR2 ships the proto + encoder for bytes passages but does not wire them end
to end; the server rejects bytes with INVALID_ARGUMENT. These tests pin
``_encode_passage`` and the serialize/parse round-trip so PR4/5 can rely on
the wire format without re-litigating it.
"""

import pytest

from routir.client.grpc import GrpcTransport
from routir.proto._generated import routir_pb2 as pb


def test_encode_passage_text():
    msg = GrpcTransport._encode_passage("hello")
    assert msg.WhichOneof("value") == "text"
    assert msg.text == "hello"


def test_encode_passage_single_bytes():
    blob = b"\x89PNG\r\n\x1a\n"
    msg = GrpcTransport._encode_passage(blob)
    assert msg.WhichOneof("value") == "bytes"
    assert list(msg.bytes.parts) == [blob]


def test_encode_passage_multi_bytes():
    blobs = [b"frame1", b"frame2", b"frame3"]
    msg = GrpcTransport._encode_passage(blobs)
    assert msg.WhichOneof("value") == "bytes"
    assert list(msg.bytes.parts) == blobs


def test_encode_passage_rejects_int():
    with pytest.raises(TypeError, match="must be str, bytes, or list"):
        GrpcTransport._encode_passage(42)


def test_encode_passage_rejects_mixed_list():
    with pytest.raises(TypeError, match="must be str, bytes, or list"):
        GrpcTransport._encode_passage([b"ok", "not-bytes"])


def test_score_request_roundtrip_mixed_passages():
    """Build a ScoreRequest with all three Passage shapes and verify the
    serialize -> parse cycle preserves WhichOneof and contents."""
    text_msg = GrpcTransport._encode_passage("hello")
    single_byte = GrpcTransport._encode_passage(b"\x89PNG\x00binary")
    multi_byte = GrpcTransport._encode_passage([b"frame1", b"frame2", b"frame3"])

    req = pb.ScoreRequest(
        service="x",
        query="q",
        passages=[text_msg, single_byte, multi_byte],
    )
    parsed = pb.ScoreRequest.FromString(req.SerializeToString())

    assert parsed.service == "x"
    assert parsed.query == "q"
    assert len(parsed.passages) == 3

    assert parsed.passages[0].WhichOneof("value") == "text"
    assert parsed.passages[0].text == "hello"

    assert parsed.passages[1].WhichOneof("value") == "bytes"
    assert list(parsed.passages[1].bytes.parts) == [b"\x89PNG\x00binary"]

    assert parsed.passages[2].WhichOneof("value") == "bytes"
    assert list(parsed.passages[2].bytes.parts) == [b"frame1", b"frame2", b"frame3"]


def test_content_response_oneof_text_roundtrip():
    """Text content path: oneof selects ``text``; bytes side is empty."""
    resp = pb.ContentResponse(
        collection="c",
        id="d1",
        text="full doc text",
        view="text",
        cached=False,
        timestamp=1.0,
    )
    parsed = pb.ContentResponse.FromString(resp.SerializeToString())
    assert parsed.WhichOneof("content") == "text"
    assert parsed.text == "full doc text"
    assert parsed.HasField("view")
    assert parsed.view == "text"


def test_content_response_oneof_data_roundtrip():
    """Bytes content path: oneof selects ``data``; text side is empty."""
    resp = pb.ContentResponse(
        collection="c",
        id="d2",
        data=pb.BytesParts(parts=[b"frame-a", b"frame-b"]),
        view="keyframes",
        cached=True,
        timestamp=2.0,
    )
    parsed = pb.ContentResponse.FromString(resp.SerializeToString())
    assert parsed.WhichOneof("content") == "data"
    assert list(parsed.data.parts) == [b"frame-a", b"frame-b"]
    assert parsed.view == "keyframes"


def test_content_request_view_optional():
    """ContentRequest.view must be a tracked optional that HasField sees."""
    req_no_view = pb.ContentRequest(collection="c", id="d1")
    assert not req_no_view.HasField("view")

    req_with_view = pb.ContentRequest(collection="c", id="d1", view="asr")
    assert req_with_view.HasField("view")
    assert req_with_view.view == "asr"

    parsed = pb.ContentRequest.FromString(req_with_view.SerializeToString())
    assert parsed.HasField("view")
    assert parsed.view == "asr"
