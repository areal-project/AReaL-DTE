# Licensed under the Apache License, Version 2.0
"""S3Store tests over moto's mock S3 (skipped when moto/boto3 are absent)."""

from __future__ import annotations

import pytest

boto3 = pytest.importorskip("boto3")
moto = pytest.importorskip("moto")


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    with moto.mock_aws():
        boto3.client("s3").create_bucket(Bucket="dte-test")
        from dte.backends.http_backend import S3Store

        yield S3Store("dte-test", prefix="weights")


def test_s3_store_roundtrip(store):
    store.put_bytes("a/b/c.bin", b"hello")
    assert store.get_bytes("a/b/c.bin") == b"hello"
    assert store.exists("a/b/c.bin")
    assert not store.exists("a/b/missing.bin")
    assert store.list_keys("a/b/") == ["a/b/c.bin"]


def test_s3_store_overwrite(store):
    store.put_bytes("k.bin", b"v1")
    store.put_bytes("k.bin", b"v2")
    assert store.get_bytes("k.bin") == b"v2"


def test_s3_store_get_missing_raises(store):
    with pytest.raises(FileNotFoundError):
        store.get_bytes("nope.bin")


def test_s3_store_list_sorted_and_prefix_scoped(store):
    store.put_bytes("s/deltas/v000002/p.bin", b"2")
    store.put_bytes("s/deltas/v000001/p.bin", b"1")
    store.put_bytes("other/x.bin", b"x")
    assert store.list_keys("s/deltas/") == [
        "s/deltas/v000001/p.bin",
        "s/deltas/v000002/p.bin",
    ]


def test_http_transport_over_s3(store):
    import torch

    from dte.backends.http_backend import HttpTransport
    from dte.transport import Payload

    t = HttpTransport(store, stream="s", checksum="adler32")
    t.publish(1, [Payload("w", torch.ones(4, dtype=torch.float32))], anchor=True)
    assert t.poll_latest()["version"] == 1
    out = t.fetch(1, kind="anchor")
    assert torch.equal(out[0].values, torch.ones(4))


@pytest.mark.parametrize("operation", ["exists", "get_bytes"])
def test_s3_access_denied_propagates(store, monkeypatch, operation):
    from botocore.exceptions import ClientError

    def denied(**kwargs):
        raise ClientError({"Error": {"Code": "AccessDenied"}}, "GetObject")

    monkeypatch.setattr(store._client, "head_object", denied)
    monkeypatch.setattr(store._client, "get_object", denied)
    with pytest.raises(ClientError, match="AccessDenied"):
        getattr(store, operation)("weights.bin")
