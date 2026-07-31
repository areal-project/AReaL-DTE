# Licensed under the Apache License, Version 2.0
"""dte transport contract — the interface dte *owns*.

dte is the main subject: the engine drives the incremental algorithm
(``dte.core``) and calls a ``Transport`` to actually move bytes. Transport
engines (awex NCCL, future mooncake) are pluggable backends that *implement*
this contract under ``dte.backends``. The engine never knows which backend is
underneath — same layering as vLLM's sparse transfer sitting on top of NCCL.

Three concerns, kept backend-neutral here:

- ``TransferOp`` / ``Plan`` — *who sends what to whom* (the reshard geometry).
  A backend computes the geometry (e.g. awex's ``TransferPlanBuilder``) and
  expresses it as these dte-owned structures.
- ``Payload`` — *what travels* for one parameter: a full tensor, or a sparse
  ``(indices, values)`` delta. Mirrors ``dte.core.SparseWeightPatch`` but is the
  unit the transport ships (optionally carrying a version-chain header).
- ``Transport`` — *how it moves*: ``build_plan`` / ``send`` / ``recv``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch

__all__ = ["TransferOp", "Plan", "Payload", "Transport"]


@dataclass(slots=True)
class TransferOp:
    """One backend-neutral reshard operation: send a parameter region from a
    training rank to an inference rank.

    ``train_slices`` / ``inf_slices`` describe the overlapping region in the
    source (train) shard and destination (infer) shard respectively — the same
    geometry ``dte.core.remap`` consumes to remap sparse indices across shards.
    ``backend_op`` lets a backend stash its native op (e.g. awex
    ``CommunicationOperation``) without dte depending on its type.
    """

    send_rank: int
    recv_rank: int
    param_name: str
    train_slices: tuple[slice, ...] = ()
    inf_slices: tuple[slice, ...] = ()
    backend_op: Any = None


@dataclass(slots=True)
class Plan:
    """This rank's reshard plan: the send/recv ops it participates in.

    ``backend_plan`` may hold the backend's native plan object; dte code only
    iterates ``ops``.
    """

    ops: list[TransferOp] = field(default_factory=list)
    backend_plan: Any = None


@dataclass(slots=True)
class Payload:
    """One parameter's transfer unit.

    - full  : ``indices is None`` — ``values`` is the whole (sliced) tensor.
    - delta : ``indices`` is a 1-D int32 flat-index tensor, ``values`` the
      corresponding new elements (sparse in-place patch).

    ``header`` optionally carries a version-chain header tensor (see
    ``dte.core.DeltaHeader``) so the receiver can validate the delta base.

    .. note::
        Two ways to express sparsity coexist by design. ``DeltaEngine`` currently
        speaks ``dte.core``'s **flat named-tensor protocol**, where a sparse
        parameter ``w`` rides as two separate full payloads named ``w@delta_idx``
        and ``w@delta_val`` (each with ``indices=None``). On that path the
        ``indices`` / ``is_delta`` / ``nnz`` fields here are **not used by the
        engine** — they are reserved for a future backend that prefers to carry
        sparsity in one structured payload (e.g. an RDMA backend pre-sizing
        buffers from ``nnz``). Don't assume ``is_delta`` reflects whether the
        sync is incremental; it only reflects this single payload's shape.
    """

    name: str
    values: torch.Tensor
    indices: torch.Tensor | None = None
    header: torch.Tensor | None = None

    @property
    def is_delta(self) -> bool:
        """Whether THIS payload carries structured sparsity (``indices`` set).

        Not a signal of whether the overall sync is delta vs full — under the
        flat protocol sparse entries travel as ``indices=None`` named tensors.
        """
        return self.indices is not None

    @property
    def nnz(self) -> int:
        """Number of changed elements (delta), or full element count."""
        return (
            int(self.indices.numel())
            if self.indices is not None
            else int(self.values.numel())
        )


class Transport(ABC):
    """The contract a transport backend implements; dte's engine calls it.

    Lifecycle: ``setup`` once, then per sync ``build_plan`` → ``send``/``recv``,
    ``teardown`` at the end. ``setup``/``teardown`` default to no-ops.
    """

    def setup(self) -> None:
        """Establish process groups / register buffers. Optional."""
        return None

    @abstractmethod
    def build_plan(self, train_meta: Any, infer_meta: Any) -> Plan:
        """Compute the reshard geometry for this rank → a dte ``Plan``.

        ``train_meta`` / ``infer_meta`` are backend-specific parameter metadata
        (e.g. awex ``ParameterMeta`` lists). The backend owns how geometry is
        computed; it must return dte-owned ``TransferOp``/``Plan``.
        """

    @abstractmethod
    def send(self, plan: Plan, payloads: list[Payload]) -> None:
        """Send this rank's payloads (full and/or delta) per ``plan``."""

    @abstractmethod
    def recv(self, plan: Plan) -> list[Payload]:
        """Receive payloads destined for this rank per ``plan``."""

    def colocate_apply(
        self,
        plan: Plan,
        full_params: dict,
        masks: dict | None,
        recv_params: dict,
        *,
        step_id: int = -1,
    ) -> int:
        """Colocate reshard hook: scatter reconstructed weights into live params.

        For the *symmetric colocate* reshard (each rank both sends its changed
        elements cross-rank and receives peers' into its live inference shard),
        the simple ``send``/``recv`` split does not fit — the engine has already
        reconstructed the full train-shard + change ``masks`` and needs the
        backend to drive the cross-rank two-round exchange + local self-copy.

        ``full_params``/``masks`` come from the engine's reconstruct;
        ``recv_params`` is the live inference param dict (write-through).
        ``masks`` is ``None`` on a dense/full step (no sparse projection).
        Returns the number of non-empty cross-rank patches applied.

        Default: unsupported (loopback/mooncake use plain ``send``/``recv``).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement colocate_apply"
        )

    def teardown(self) -> None:
        """Tear down process groups / release buffers. Optional."""
        return None
