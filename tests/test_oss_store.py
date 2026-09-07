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
