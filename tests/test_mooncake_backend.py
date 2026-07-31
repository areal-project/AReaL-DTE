"""Tests for the Mooncake backend's CPU-testable parts: pack/unpack round-trip,
endpoint descriptor serialization, contract, graceful degradation. The RDMA path
(setup/recv) needs hardware and is not exercised here.
"""

from types import SimpleNamespace

import pytest
import torch

import dte.backends.mooncake_backend as mb
from dte.backends.mooncake_backend import (
    BucketEntry,
    MooncakeEndpoint,
    MooncakeTransport,
    mooncake_available,
    pack_payloads,
    unpack_buffer,
)
from dte.transport import Payload, Transport


def test_mooncake_available_reflects_env():
    assert mooncake_available() is False  # not installed on the CPU dev box


def test_is_transport_subclass():
    assert issubclass(MooncakeTransport, Transport)
    for m in ("build_plan", "send", "recv", "setup", "teardown"):
        assert hasattr(MooncakeTransport, m)


def test_constructor_requires_mooncake_when_absent():
    if mooncake_available():
        pytest.skip("mooncake installed")
    with pytest.raises(RuntimeError, match="mooncake"):
        MooncakeTransport(meta_channel=SimpleNamespace(), bucket_size=1024)


def test_pack_unpack_round_trip():
    """Pure logic: payloads -> flat buffer -> payloads, bitwise-preserving.

    Mirrors dte.core's flat protocol: a header + a sparse idx/val pair + a dense
    tensor, all of mixed dtype, pack and unpack exactly.
    """
    payloads = [
        Payload("__awex_delta_header__", torch.tensor([1, 2, 3], dtype=torch.int64)),
        Payload("w@delta_idx", torch.tensor([0, 5, 9], dtype=torch.int32)),
        Payload("w@delta_val", torch.tensor([1.5, -2.0, 3.25], dtype=torch.bfloat16)),
        Payload("dense.w", torch.randn(4, 4, dtype=torch.bfloat16)),
    ]
    total = sum(p.values.numel() * p.values.element_size() for p in payloads)
    buf = torch.zeros(total, dtype=torch.uint8)

    buckets = pack_payloads(payloads, buf)
    assert len(buckets) == 4
    out = unpack_buffer(buf, buckets)

    assert [p.name for p in out] == [p.name for p in payloads]
    for src, dst in zip(payloads, out):
        assert dst.values.dtype == src.values.dtype
        assert tuple(dst.values.shape) == tuple(src.values.shape)
        assert torch.equal(dst.values, src.values), f"{src.name} not bitwise-equal"


def test_pack_buffer_overflow_raises():
    payloads = [Payload("w", torch.randn(100, dtype=torch.bfloat16))]
    tiny = torch.zeros(8, dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="overflow"):
        pack_payloads(payloads, tiny)


def test_endpoint_descriptor_serialization():
    """Endpoint descriptor survives a JSON-able dict round trip (it travels the
    out-of-band meta channel)."""
    ep = MooncakeEndpoint(
        session_id="host:9000",
        buffer_ptr=0xDEADBEEF,
        total_bytes=128,
        buckets=[
            BucketEntry("w@delta_idx", 0, 12, torch.int32, (3,)),
            BucketEntry("w@delta_val", 12, 6, torch.bfloat16, (3,)),
        ],
    )
    d = ep.to_dict()
    ep2 = MooncakeEndpoint.from_dict(d)
    assert ep2.session_id == ep.session_id
    assert ep2.buffer_ptr == ep.buffer_ptr
    assert len(ep2.buckets) == 2
    assert ep2.buckets[0].dtype == torch.int32
    assert ep2.buckets[1].dtype == torch.bfloat16
    assert ep2.buckets[1].shape == (3,)


def test_send_recv_need_rdma(monkeypatch):
    """Without a registered buffer (no setup/RDMA), send and recv refuse."""
    monkeypatch.setattr(mb, "_MOONCAKE_AVAILABLE", True)
    t = MooncakeTransport(meta_channel=SimpleNamespace(), bucket_size=1024)
    with pytest.raises(NotImplementedError, match="RDMA"):
        t.send(t.build_plan(None, None), [Payload("w", torch.zeros(2))])
    with pytest.raises(NotImplementedError, match="RDMA"):
        t.recv(t.build_plan(None, None))
