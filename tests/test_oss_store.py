# Licensed under the Apache License, Version 2.0
"""Native OSS adapter contracts, using an in-memory SDK boundary."""

import io
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from dte.backends import HttpTransport, OSSStore
from dte.core import DeltaTracker
from dte.engine import DeltaEngine
from dte.transport import Payload

oss2 = pytest.importorskip("oss2")


@pytest.fixture
def client():
    objects = {}
    client = Mock(spec=oss2.Bucket)

    def get(key):
        if key not in objects:
            raise oss2.exceptions.NoSuchKey(404, {}, b"", {})
        return io.BytesIO(objects[key])

    def put(key, data):
        objects[key] = data

    client.put_object.side_effect = put
    client.get_object.side_effect = get
    client.head_object.side_effect = lambda key: get(key).close()
    client.delete_object.side_effect = lambda key: objects.pop(key, None)
    client.list_objects_v2.side_effect = lambda **kw: SimpleNamespace(
        object_list=[
            SimpleNamespace(key=k) for k in objects if k.startswith(kw["prefix"])
        ],
        is_truncated=False,
    )
    return client


def test_oss_roundtrip_and_scope(client):
    store = OSSStore(client, prefix="run-a")
    other = OSSStore(client, prefix="run-b")
    other.put_bytes("policy/b", b"untouched")
    store.put_bytes("policy/b", b"first")
    store.put_bytes("policy/b", b"second")
    store.put_bytes("policy/a", b"a")

    assert store.get_bytes("policy/b") == b"second"
    assert store.exists("policy/b")
    assert store.list_keys("policy/") == ["policy/a", "policy/b"]
    assert store.delete_prefix("policy/") == 2
    assert not store.exists("policy/b")
    store.delete("policy/b")
    with pytest.raises(FileNotFoundError):
        store.get_bytes("policy/b")
    assert other.get_bytes("policy/b") == b"untouched"


def test_oss_pagination_sorted_relative_keys(client):
    client.list_objects_v2.side_effect = [
        SimpleNamespace(
            object_list=[SimpleNamespace(key="run/p/b")],
            is_truncated=True,
            next_continuation_token="next",
        ),
        SimpleNamespace(
            object_list=[SimpleNamespace(key="run/p/a")], is_truncated=False
        ),
    ]
    assert OSSStore(client, prefix="run").list_keys("p/") == ["p/a", "p/b"]
    assert client.list_objects_v2.call_args.kwargs["continuation_token"] == "next"


@pytest.mark.parametrize(
    "operation,method",
    [
        ("exists", "head_object"),
        ("get_bytes", "get_object"),
        ("delete", "delete_object"),
    ],
)
def test_oss_denied_and_missing_bucket_are_not_missing_keys(client, operation, method):
    store = OSSStore(client, prefix="run")
    for error in (
        oss2.exceptions.AccessDenied(403, {}, b"", {}),
        oss2.exceptions.NoSuchBucket(404, {}, b"", {}),
    ):
        getattr(client, method).side_effect = error
        with pytest.raises(type(error)):
            getattr(store, operation)("p/key")


def test_oss_response_closed_when_read_fails(client):
    response = Mock()
    response.read.side_effect = OSError("interrupted")
    client.get_object.side_effect = None
    client.get_object.return_value = response
    with pytest.raises(OSError, match="interrupted"):
        OSSStore(client, prefix="run").get_bytes("p/key")
    response.close.assert_called_once()


def test_oss_head_without_error_body_uses_get_to_distinguish_missing_bucket(client):
    client.head_object.side_effect = oss2.exceptions.NotFound(404, {}, b"", {})
    store = OSSStore(client, prefix="run")
    assert not store.exists("p/key")
    client.get_object.side_effect = oss2.exceptions.NoSuchBucket(404, {}, b"", {})
    with pytest.raises(oss2.exceptions.NoSuchBucket):
        store.exists("p/key")


def test_oss_multipart_commits_only_after_all_parts(client):
    client.init_multipart_upload.return_value.upload_id = "upload"
    client.upload_part.side_effect = lambda *args: SimpleNamespace(etag=str(args[2]))
    store = OSSStore(client, prefix="run", multipart_threshold=1, part_size=100 * 1024)
    data = b"x" * (250 * 1024)
    store.put_bytes("p/key", data)
    calls = client.upload_part.call_args_list
    assert b"".join(call.args[3] for call in calls) == data
    assert [call.args[2] for call in calls] == [1, 2, 3]
    assert client.complete_multipart_upload.call_args.args[:2] == (
        "run/p/key",
        "upload",
    )
    client.put_object.assert_not_called()


def test_oss_multipart_failure_does_not_publish_object(client):
    client.init_multipart_upload.return_value.upload_id = "upload"
    client.upload_part.side_effect = OSError("interrupted")
    store = OSSStore(client, prefix="run", multipart_threshold=1)
    with pytest.raises(OSError):
        store.put_bytes("p/key", b"payload")
    client.complete_multipart_upload.assert_not_called()


@pytest.mark.parametrize("key", ["../escape", "/absolute", "a/../b"])
def test_oss_unsafe_keys_never_reach_sdk(client, key):
    with pytest.raises(ValueError):
        OSSStore(client, prefix="run").put_bytes(key, b"data")
    client.put_object.assert_not_called()


def test_oss_anchor_deltas_streaming_and_manual_retention(client):
    transport = HttpTransport(OSSStore(client, prefix="run"), prune_on_anchor=False)
    receiver = DeltaEngine(None)
    tracker = DeltaTracker()
    weights = {
        "a": torch.arange(256, dtype=torch.float32),
        "b": torch.arange(256, dtype=torch.bfloat16),
    }
    target = {}
    for version in range(5):
        anchor = version in (0, 4)
        weights["a"][version] += 2
        weights["b"][version] += 2
        if anchor:
            payloads = [Payload(k, t) for k, t in weights.items()]
            tracker.seed(weights.items(), version)
        else:
            encoded = tracker.encode(weights.items(), version)
            payloads = [Payload(k, t) for k, t in zip(encoded.names, encoded.tensors)]
        transport.publish(version, payloads, anchor=anchor, chunk_bytes=128)
        kind = "anchor" if anchor else "delta"
        for chunk in receiver.reconstruct_stream(
            transport.iter_fetch(version, kind=kind), version
        ):
            target.update(chunk)
        assert receiver.base_version == version
        for key in weights:
            assert torch.equal(
                target[key].view(torch.uint8), weights[key].view(torch.uint8)
            )
    assert transport.list_versions() == ([0, 4], [1, 2, 3])
    client.delete_object.assert_not_called()


def test_oss_remote_disconnect_retries_get(client, monkeypatch):
    from http.client import RemoteDisconnected

    from requests.exceptions import ConnectionError

    sleeps = []
    monkeypatch.setattr("dte.backends.oss_store.time.sleep", sleeps.append)
    error = oss2.exceptions.RequestError(ConnectionError(RemoteDisconnected("closed")))
    response = io.BytesIO(b"complete")
    client.get_object.side_effect = [error, response]
    assert OSSStore(client, prefix="run").get_bytes("p/key") == b"complete"
    assert client.get_object.call_count == 2
    assert response.closed
    assert sleeps == [0.5]


def test_oss_interrupted_body_restarts_and_closes_response(client, monkeypatch):
    from requests.exceptions import ChunkedEncodingError

    monkeypatch.setattr("dte.backends.oss_store.time.sleep", lambda _: None)
    failed = Mock()
    failed.read.side_effect = ChunkedEncodingError("partial response")
    client.get_object.side_effect = [failed, io.BytesIO(b"whole object")]
    assert OSSStore(client, prefix="run").get_bytes("p/key") == b"whole object"
    failed.close.assert_called_once()
    assert client.get_object.call_count == 2


def test_oss_retry_exhaustion_is_bounded(client, monkeypatch):
    from requests.exceptions import Timeout

    sleeps = []
    monkeypatch.setattr("dte.backends.oss_store.time.sleep", sleeps.append)
    client.get_object.side_effect = oss2.exceptions.RequestError(Timeout("timeout"))
    with pytest.raises(oss2.exceptions.RequestError):
        OSSStore(client, prefix="run", read_attempts=3).get_bytes("p/key")
    assert client.get_object.call_count == 3
    assert sleeps == [0.5, 1.0]


@pytest.mark.parametrize(
    "error",
    [
        oss2.exceptions.AccessDenied(403, {}, b"", {}),
        oss2.exceptions.NoSuchKey(404, {}, b"", {}),
        oss2.exceptions.InconsistentError("checksum mismatch"),
    ],
)
def test_oss_permanent_read_failure_is_not_retried(client, error):
    client.get_object.side_effect = error
    with pytest.raises((type(error), FileNotFoundError)):
        OSSStore(client, prefix="run").get_bytes("p/key")
    assert client.get_object.call_count == 1


def test_oss_tls_failure_is_not_retried(client):
    from requests.exceptions import SSLError

    client.get_object.side_effect = oss2.exceptions.RequestError(SSLError("bad cert"))
    with pytest.raises(oss2.exceptions.RequestError):
        OSSStore(client, prefix="run").get_bytes("p/key")
    assert client.get_object.call_count == 1


def test_oss_list_retry_keeps_page_token(client, monkeypatch):
    monkeypatch.setattr("dte.backends.oss_store.time.sleep", lambda _: None)
    client.list_objects_v2.side_effect = [
        SimpleNamespace(
            object_list=[SimpleNamespace(key="run/p/a")],
            is_truncated=True,
            next_continuation_token="next",
        ),
        oss2.exceptions.ServerError(503, {}, b"", {}),
        SimpleNamespace(
            object_list=[SimpleNamespace(key="run/p/b")], is_truncated=False
        ),
    ]
    assert OSSStore(client, prefix="run").list_keys("p/") == ["p/a", "p/b"]
    assert [
        c.kwargs["continuation_token"] for c in client.list_objects_v2.call_args_list
    ] == ["", "next", "next"]


def test_oss_head_retries_transient_service_error(client, monkeypatch):
    monkeypatch.setattr("dte.backends.oss_store.time.sleep", lambda _: None)
    client.head_object.side_effect = [
        oss2.exceptions.ServerError(503, {}, b"", {}),
        None,
    ]
    assert OSSStore(client, prefix="run").exists("p/key")
    assert client.head_object.call_count == 2


def test_oss_put_retries_identical_bytes(client, monkeypatch):
    from requests.exceptions import ConnectionError

    monkeypatch.setattr("dte.backends.oss_store.time.sleep", lambda _: None)
    client.put_object.side_effect = [
        oss2.exceptions.RequestError(ConnectionError("closed")),
        None,
    ]
    OSSStore(client, prefix="run").put_bytes("p/key", b"payload")
    assert [call.args for call in client.put_object.call_args_list] == [
        ("run/p/key", b"payload"),
        ("run/p/key", b"payload"),
    ]


def test_oss_multipart_init_and_part_transients_retry(client, monkeypatch):
    from requests.exceptions import ConnectionError

    monkeypatch.setattr("dte.backends.oss_store.time.sleep", lambda _: None)
    error = oss2.exceptions.RequestError(ConnectionError("closed"))
    client.init_multipart_upload.side_effect = [
        error,
        SimpleNamespace(upload_id="upload"),
    ]
    client.upload_part.side_effect = [error, SimpleNamespace(etag="etag")]
    OSSStore(client, prefix="run", multipart_threshold=1).put_bytes("p/key", b"payload")
    assert client.init_multipart_upload.call_count == 2
    assert client.upload_part.call_args_list[0] == client.upload_part.call_args_list[1]
    client.complete_multipart_upload.assert_called_once()


@pytest.mark.parametrize("matches", [True, False])
def test_oss_completed_upload_without_id_requires_matching_object(client, matches):
    client.init_multipart_upload.return_value.upload_id = "upload"
    client.upload_part.return_value.etag = "etag"
    client.complete_multipart_upload.side_effect = oss2.exceptions.NoSuchUpload(
        404, {}, b"", {}
    )
    client.get_object.side_effect = None
    client.get_object.return_value = io.BytesIO(b"payload" if matches else b"other")
    store = OSSStore(client, prefix="run", multipart_threshold=1)
    if matches:
        store.put_bytes("p/key", b"payload")
    else:
        with pytest.raises(oss2.exceptions.NoSuchUpload):
            store.put_bytes("p/key", b"payload")


def test_oss_write_exhaustion_and_authentication_fail_closed(client, monkeypatch):
    from requests.exceptions import ConnectionError

    monkeypatch.setattr("dte.backends.oss_store.time.sleep", lambda _: None)
    client.put_object.side_effect = oss2.exceptions.RequestError(
        ConnectionError("closed")
    )
    with pytest.raises(oss2.exceptions.RequestError):
        OSSStore(client, prefix="run", write_attempts=2).put_bytes("p/key", b"payload")
    assert client.put_object.call_count == 2
    client.put_object.reset_mock()
    client.put_object.side_effect = oss2.exceptions.AccessDenied(403, {}, b"", {})
    with pytest.raises(oss2.exceptions.AccessDenied):
        OSSStore(client, prefix="run").put_bytes("p/key", b"payload")
    client.put_object.assert_called_once()


@pytest.mark.parametrize("option", ["read_attempts", "write_attempts"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_oss_retry_attempts_invalid_rejected(client, option, value):
    with pytest.raises(ValueError, match="positive integer"):
        OSSStore(client, prefix="run", **{option: value})


@pytest.mark.parametrize("value", [-0.5, 9, float("nan")])
def test_oss_retry_backoff_invalid_rejected(client, value):
    with pytest.raises(ValueError, match="read_backoff"):
        OSSStore(client, prefix="run", read_backoff=value)


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
@pytest.mark.parametrize(
    "operation,method", [("get_bytes", "get_object"), ("put_bytes", "put_object")]
)
def test_oss_transient_status_retries_once(
    client, monkeypatch, status, operation, method
):
    monkeypatch.setattr("dte.backends.oss_store.time.sleep", lambda _: None)
    error = oss2.exceptions.ServerError(status, {}, b"", {})
    result = io.BytesIO(b"payload") if operation == "get_bytes" else None
    getattr(client, method).side_effect = [error, result]
    store = OSSStore(client, prefix="run")
    if operation == "get_bytes":
        assert store.get_bytes("p/key") == b"payload"
    else:
        store.put_bytes("p/key", b"payload")
    assert getattr(client, method).call_count == 2


@pytest.mark.parametrize("attempts,expected_sleeps", [(1, []), (4, [3, 6, 8])])
def test_oss_write_backoff_bounded_without_sensitive_logging(
    client, monkeypatch, caplog, attempts, expected_sleeps
):
    from requests.exceptions import ConnectionError

    sleeps = []
    monkeypatch.setattr("dte.backends.oss_store.time.sleep", sleeps.append)
    client.put_object.side_effect = ConnectionError("sensitive-request-context")
    store = OSSStore(client, prefix="run", write_attempts=attempts, read_backoff=3)
    with pytest.raises(ConnectionError):
        store.put_bytes("private-object-name", b"payload")
    assert client.put_object.call_count == attempts
    assert sleeps == expected_sleeps
    assert "sensitive-request-context" not in caplog.text
    assert "private-object-name" not in caplog.text


def test_oss_complete_lost_response_matching_object_succeeds(client):
    from requests.exceptions import ConnectionError

    client.init_multipart_upload.return_value.upload_id = "upload"
    client.upload_part.return_value.etag = "etag"
    client.complete_multipart_upload.side_effect = ConnectionError("response lost")
    client.get_object.side_effect = [io.BytesIO(b"payload")]

    OSSStore(client, prefix="run", multipart_threshold=1).put_bytes("p/key", b"payload")

    client.complete_multipart_upload.assert_called_once()
    client.get_object.assert_called_once_with("run/p/key")


def test_oss_complete_lost_response_missing_object_retries_same_upload(
    client, monkeypatch
):
    from requests.exceptions import ConnectionError

    monkeypatch.setattr("dte.backends.oss_store.time.sleep", lambda _: None)
    client.init_multipart_upload.return_value.upload_id = "upload"
    client.upload_part.return_value.etag = "etag"
    client.complete_multipart_upload.side_effect = [ConnectionError("closed"), None]

    OSSStore(client, prefix="run", multipart_threshold=1).put_bytes("p/key", b"payload")

    first, second = client.complete_multipart_upload.call_args_list
    assert first == second
    client.init_multipart_upload.assert_called_once()
    client.upload_part.assert_called_once()
