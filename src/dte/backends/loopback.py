# Licensed under the Apache License, Version 2.0
"""LoopbackTransport — a dependency-free in-process transport backend.

The simplest possible ``Transport``: ``send`` stashes the payloads, ``recv``
hands them back. It performs no resharding (train and infer shardings are
assumed identical) and no real communication, so it runs anywhere on CPU.

Purpose: prove the dte layering is self-consistent end to end (engine → core
codec → transport → core decode) without any awex/NCCL/GPU dependency. Real
backends (awex NCCL, future mooncake) live alongside this file and implement the
same contract.
"""

from __future__ import annotations

from dte.transport import Payload, Plan, Transport

__all__ = ["LoopbackTransport"]


class LoopbackTransport(Transport):
    """Single-process passthrough: send buffers what recv then returns.

    Tensors are cloned on send so the receiver cannot alias sender storage
    (mirrors a real transport's copy semantics; keeps bitwise checks honest).
    """

    def __init__(self) -> None:
        self._buffer: list[Payload] = []

    def build_plan(self, train_meta, infer_meta) -> Plan:
        # No resharding in loopback: an empty plan (engine forwards it untouched).
        return Plan()

    def send(self, plan: Plan, payloads: list[Payload]) -> None:
        self._buffer = [
            Payload(
                name=p.name,
                values=p.values.detach().clone(),
                indices=None if p.indices is None else p.indices.detach().clone(),
                header=None if p.header is None else p.header.detach().clone(),
            )
            for p in payloads
        ]

    def recv(self, plan: Plan) -> list[Payload]:
        out = self._buffer
        self._buffer = []
        return out
