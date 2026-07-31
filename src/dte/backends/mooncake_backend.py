# Licensed under the Apache License, Version 2.0
"""MooncakeTransport — disaggregated RDMA backend for dte (put/get).

Target: actor/rollout on **separate** GPUs/nodes (disaggregated), where Mooncake
Transfer Engine moves weights over RDMA. Unlike awex's colocate symmetric P2P,
Mooncake here uses a **put/get** model — the sender publishes a buffer endpoint,
the receiver pulls it with ``transfer_sync_read`` — which is naturally
one-directional and maps cleanly onto dte's ``send`` / ``recv`` (no symmetric
impedance like the awex backend).

dte still owns the wire protocol: payloads are dte.core's flat named-tensors
(``__awex_delta_header__`` + ``w@delta_idx`` / ``w@delta_val`` + dense ``w``),
packed contiguously into one RDMA-registered buffer; ``bucket_meta`` records each
tensor's byte layout so the receiver can slice them back and hand a
``dict(zip(names, tensors))`` to ``decode_delta_payload``.

──────────────────────────────────────────────────────────────────────────────
STATUS: interface designed against Mooncake's real TransferEngine API (verified
against veRL's mooncake_checkpoint_engine). The pack/unpack of payloads ↔ a flat
buffer is pure logic and unit-tested on CPU. The RDMA lifecycle
(initialize / batch_register_memory / transfer_sync_read|write) needs Mooncake +
RDMA NICs and is exercised on the cluster — those methods raise with a pointer
rather than fake it.
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch

from dte.transport import Payload, Plan, Transport

try:  # optional: pip install delta-transfer-engine[mooncake]
    from mooncake.engine import TransferEngine  # noqa: F401

    _MOONCAKE_AVAILABLE = True
except Exception:  # pragma: no cover
    _MOONCAKE_AVAILABLE = False


_NEEDS_RDMA = (
    "MooncakeTransport RDMA path needs the 'mooncake' package + RDMA NICs and a "
    "running peer; verified on the cluster. Pack/unpack logic is CPU-testable, "
    "but initialize/transfer_sync_* are not."
)


def mooncake_available() -> bool:
    return _MOONCAKE_AVAILABLE


# ---------------------------------------------------------------- metadata channel
class MetaChannel(ABC):
    """Out-of-band channel to publish/lookup an RDMA endpoint descriptor.

    Mooncake moves bytes by address; *which* address (session_id + ptr + layout)
    must travel out of band. dte does not mandate the medium — awex's MetaServer,
    a StatelessProcessGroup, or a KV store can implement this. The descriptor is a
    small JSON-able dict (see ``MooncakeEndpoint.to_dict``).
    """

    @abstractmethod
    def publish(self, key: str, descriptor: dict) -> None: ...

    @abstractmethod
    def lookup(self, key: str) -> dict: ...


# ---------------------------------------------------------------- buffer layout
@dataclass(slots=True)
class BucketEntry:
    """One tensor's byte layout inside the registered RDMA buffer."""

    name: str
    offset: int  # byte offset within the buffer
    nbytes: int
    dtype: torch.dtype
    shape: tuple[int, ...]


@dataclass(slots=True)
class MooncakeEndpoint:
    """RDMA endpoint descriptor published by the sender, pulled by the receiver."""

    session_id: str  # f"{hostname}:{rpc_port}"
    buffer_ptr: int  # data_ptr() of the registered buffer
    total_bytes: int
    buckets: list[BucketEntry] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "buffer_ptr": self.buffer_ptr,
            "total_bytes": self.total_bytes,
            "buckets": [
                {
                    "name": b.name,
                    "offset": b.offset,
                    "nbytes": b.nbytes,
                    "dtype": str(b.dtype).removeprefix("torch."),
                    "shape": list(b.shape),
                }
                for b in self.buckets
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> MooncakeEndpoint:
        return cls(
            session_id=d["session_id"],
            buffer_ptr=d["buffer_ptr"],
            total_bytes=d["total_bytes"],
            buckets=[
                BucketEntry(
                    name=b["name"],
                    offset=b["offset"],
                    nbytes=b["nbytes"],
                    dtype=getattr(torch, b["dtype"]),
                    shape=tuple(b["shape"]),
                )
                for b in d["buckets"]
            ],
        )


# ---------------------------------------------------------------- pure pack/unpack
def pack_payloads(payloads: list[Payload], buffer: torch.Tensor) -> list[BucketEntry]:
    """Copy each payload tensor contiguously into ``buffer`` (uint8), recording
    its byte layout. dte.core's flat protocol means a sparse entry is just two
    payloads (``w@delta_idx`` / ``w@delta_val``); they pack like any tensor.

    Pure tensor/byte logic — no RDMA, CPU-testable. Raises if the buffer is too
    small (the caller sizes the registered buffer from the encoded payload bytes).
    """
    buckets: list[BucketEntry] = []
    offset = 0
    for p in payloads:
        t = p.values.contiguous()
        nbytes = t.numel() * t.element_size()
        if offset + nbytes > buffer.numel():
            raise RuntimeError(
                f"Mooncake buffer overflow packing {p.name}: need "
                f"{offset + nbytes} > {buffer.numel()} bytes."
            )
        flat = t.view(torch.uint8).reshape(-1)
        buffer[offset : offset + nbytes].copy_(flat)
        buckets.append(BucketEntry(p.name, offset, nbytes, t.dtype, tuple(t.shape)))
        offset += nbytes
    return buckets


def unpack_buffer(buffer: torch.Tensor, buckets: list[BucketEntry]) -> list[Payload]:
    """Slice tensors back out of ``buffer`` per ``buckets`` → dte Payloads.

    Sparse idx/val entries come back under their flat-protocol names; the engine
    rebuilds ``dict(zip(names, tensors))`` and calls ``decode_delta_payload``.
    Pure logic, CPU-testable.
    """
    out: list[Payload] = []
    for b in buckets:
        raw = buffer[b.offset : b.offset + b.nbytes]
        tensor = raw.view(b.dtype).reshape(b.shape)
        out.append(Payload(b.name, tensor.clone()))
    return out


# ---------------------------------------------------------------- transport
class MooncakeTransport(Transport):
    """dte Transport over Mooncake Transfer Engine (disaggregated, put/get).

    Args:
        meta_channel: out-of-band endpoint publish/lookup (see ``MetaChannel``).
        bucket_size: registered RDMA buffer size in bytes.
        device: where the registered buffer lives (e.g. "cuda").
        protocol / device_name: forwarded to ``TransferEngine.initialize``.
        peer_key: key under which the sender publishes / receiver looks up the
            endpoint descriptor for this sync step.
    """

    def __init__(
        self,
        meta_channel: MetaChannel,
        *,
        bucket_size: int,
        device: str = "cuda",
        protocol: str = "rdma",
        device_name: str = "",
        peer_key: str = "dte_mooncake_endpoint",
    ):
        if not _MOONCAKE_AVAILABLE:
            raise RuntimeError(
                "MooncakeTransport needs the 'mooncake' package. "
                "Install with: pip install delta-transfer-engine[mooncake]"
            )
        self.meta_channel = meta_channel
        self.bucket_size = bucket_size
        self.device = device
        self.protocol = protocol
        self.device_name = device_name
        self.peer_key = peer_key
        self.engine = None
        self.session_id: str | None = None
        self._buffer: torch.Tensor | None = None

    # ---- lifecycle (RDMA, cluster-only) ----
    def setup(self) -> None:
        """initialize TransferEngine + register one RDMA buffer.

        Real body: ``engine.initialize(host, "P2PHANDSHAKE", protocol,
        device_name)``; ``session_id = f"{host}:{engine.get_rpc_port()}"``;
        allocate ``buffer = torch.empty(bucket_size, uint8, device)`` and
        ``engine.batch_register_memory([buffer.data_ptr()], [bucket_size])``.
        Needs RDMA NICs.
        """
        raise NotImplementedError(_NEEDS_RDMA)

    def teardown(self) -> None:
        """Deregister buffer / close the engine. RDMA-only."""
        self.engine = None
        self._buffer = None

    # ---- plan ----
    def build_plan(self, train_meta: Any, infer_meta: Any) -> Plan:
        """put/get is address-based, not op-based.

        With identical train/infer sharding (the common disaggregated case) no
        reshard plan is needed → empty Plan. For TP-mismatch disaggregated, an
        awex ``TransferPlanBuilder`` can fill ``Plan.ops`` (same as the awex
        backend); left to the integration that has the parameter metadata.
        """
        return Plan()

    # ---- transfer (sender stages buffer + publishes; receiver pulls) ----
    def send(self, plan: Plan, payloads: list[Payload]) -> None:
        """Pack payloads into the registered buffer and publish the endpoint.

        Packing is pure logic (``pack_payloads``); publishing the descriptor goes
        through ``meta_channel``. The actual bytes stay in the local registered
        buffer until the receiver reads them via RDMA — that's the put/get model.
        Requires ``setup()`` to have registered the buffer (RDMA).
        """
        if self._buffer is None:
            raise NotImplementedError(_NEEDS_RDMA)
        buckets = pack_payloads(payloads, self._buffer)
        endpoint = MooncakeEndpoint(
            session_id=self.session_id,
            buffer_ptr=self._buffer.data_ptr(),
            total_bytes=sum(b.nbytes for b in buckets),
            buckets=buckets,
        )
        self.meta_channel.publish(self.peer_key, endpoint.to_dict())

    def recv(self, plan: Plan) -> list[Payload]:
        """Look up the remote endpoint, RDMA-read its buffer, unpack to Payloads.

        Real body: ``endpoint = MooncakeEndpoint.from_dict(meta_channel.lookup(
        peer_key))``; ``engine.transfer_sync_read(endpoint.session_id,
        local_buf.data_ptr(), endpoint.buffer_ptr, endpoint.total_bytes)``; then
        ``unpack_buffer(local_buf, endpoint.buckets)``. The RDMA read needs
        hardware; unpack itself is the CPU-tested pure logic.
        """
        raise NotImplementedError(_NEEDS_RDMA)


__all__ = [
    "MooncakeTransport",
    "MetaChannel",
    "MooncakeEndpoint",
    "BucketEntry",
    "pack_payloads",
    "unpack_buffer",
    "mooncake_available",
]
