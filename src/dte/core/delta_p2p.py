# Licensed under the Apache License, Version 2.0
"""Variable-length sparse payload protocol for the colocate P2P transport.

The dense colocate transport (``nccl_stream_batch.update_weights_in_colocate_mode``)
sends a fixed-size sliced tensor per CommunicationOperation. A delta payload is
*variable length* — each op carries only the elements that changed, so the
receiver cannot pre-size its recv buffers from the static transfer plan alone.

Following vLLM PR #40096, we split control plane from data plane:

- **control plane**: a per-op ``nnz`` (number of changed elements) is exchanged
  first (one int32 per op — fixed size, derivable from the plan's op count, so
  it rides the existing symmetric recursive-partition round without breaking the
  deadlock invariant).
- **data plane**: for each op the sender broadcasts ``indices`` (int32, in the
  *inference* shard's flat space — already remapped) and ``values``; the
  receiver, now knowing ``nnz``, pre-allocates and scatters into its live
  parameter view.

This module holds the **pure, CPU-testable** packing/unpacking logic. The actual
NCCL send/recv is wired in ``nccl_stream_batch`` and is not tested here (no
distributed context on CPU); what is tested is the round-trip:
``build_send_patches -> (transmit) -> allocate_recv_buffers -> scatter`` equals a
dense copy, including the ``nnz == 0`` symmetry that is the #1 deadlock hazard.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import torch

from dte.core.codec import DecodedDelta, apply_sparse_patch_
from dte.core.patch import SparseWeightPatch
from dte.core.remap import (
    remap_delta_indices_for_ops,
    remap_mask_for_op,
)

logger = logging.getLogger(__name__)


def _profile_delta_build_enabled() -> bool:
    return os.environ.get("DTE_DELTA_REMAP_PROFILE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@dataclass
class OpDeltaPayload:
    """Per-op delta payload, parallel to one CommunicationOperation.

    ``nnz == 0`` is a valid, required state: an op whose overlap region had no
    changed element this version still occupies a slot (empty index/value
    tensors), so both peers iterate the *same* op sequence and stay symmetric.
    Dropping a zero-nnz op on one side only would desync the recursive-partition
    schedule and hang.
    """

    op: object  # CommunicationOperation
    indices: torch.Tensor  # int32, inference-shard flat space (possibly empty)
    values: torch.Tensor  # param dtype (possibly empty)

    @property
    def nnz(self) -> int:
        return int(self.indices.numel())


def build_send_patches(
    ops: list,
    masks: dict[str, torch.Tensor],
    send_params: dict[str, torch.Tensor],
) -> list[OpDeltaPayload]:
    """Build one OpDeltaPayload per op (in input order), preserving zero-nnz ops.

    Args:
        ops: CommunicationOperations for one peer (send direction).
        masks: ``{hf_name: bool change mask}`` over the train-shard param;
            computed once per param, shared across that param's ops.
        send_params: ``{hf_name: train-shard tensor}`` (value source).

    Returns:
        List parallel to ``ops``; every op gets a payload, empty if no overlap.
    """
    payloads: list[OpDeltaPayload] = []
    for op in ops:
        name = op.send_shard_meta.name
        mask = masks.get(name)
        src = send_params.get(name)
        patch: SparseWeightPatch | None = None
        if mask is not None and src is not None:
            patch = remap_mask_for_op(name, mask, src, tuple(src.shape), op)
        if patch is None:
            # zero-nnz slot: keep the op so both peers stay symmetric.
            dtype = send_params[name].dtype if name in send_params else torch.bfloat16
            payloads.append(
                OpDeltaPayload(
                    op=op,
                    indices=torch.empty(0, dtype=torch.int32),
                    values=torch.empty(0, dtype=dtype),
                )
            )
        else:
            payloads.append(
                OpDeltaPayload(op=op, indices=patch.indices, values=patch.values)
            )
    return payloads


def _empty_payload(op) -> OpDeltaPayload:
    dtype = getattr(op.recv_shard_meta, "dtype", None) or getattr(
        op.send_shard_meta, "dtype", torch.bfloat16
    )
    return OpDeltaPayload(
        op=op,
        indices=torch.empty(0, dtype=torch.int32),
        values=torch.empty(0, dtype=dtype),
    )


def _slice_flat_indices(
    shape: tuple[int, ...],
    slices: tuple[slice, ...],
    *,
    device: torch.device,
) -> torch.Tensor:
    """Return flat indices for ``tensor[slices]`` in the full tensor space."""
    if len(shape) == 0:
        return torch.zeros(1, dtype=torch.int64, device=device)
    grids = []
    for dim, s in enumerate(slices):
        step = s.step if s.step is not None else 1
        if step != 1:
            raise NotImplementedError(
                f"dense delta fallback does not support strided slice step={step} "
                f"on dim {dim}"
            )
        start = s.start if s.start is not None else 0
        stop = s.stop if s.stop is not None else shape[dim]
        grids.append(torch.arange(start, stop, device=device, dtype=torch.int64))
    if len(grids) == 1:
        multi = (grids[0],)
    else:
        multi = torch.meshgrid(*grids, indexing="ij")
    flat = torch.zeros_like(multi[0], dtype=torch.int64)
    stride = 1
    for dim in range(len(shape) - 1, -1, -1):
        flat += multi[dim] * stride
        stride *= shape[dim]
    return flat.reshape(-1)


def _dense_payload_for_op(op, tensor: torch.Tensor) -> OpDeltaPayload:
    recv_dtype = getattr(op.recv_shard_meta, "dtype", None)
    values = tensor[op.train_slices].reshape(-1).contiguous()
    if recv_dtype is not None and values.dtype != recv_dtype:
        values = values.to(recv_dtype)
    indices = _slice_flat_indices(
        tuple(op.recv_shard_meta.shape),
        tuple(op.inf_slices),
        device=values.device,
    )
    return OpDeltaPayload(op=op, indices=indices.to(torch.int32), values=values)


def build_send_patches_from_delta(
    ops: list,
    decoded: DecodedDelta,
    *,
    profile_label: str = "",
) -> list[OpDeltaPayload]:
    """Build per-op payloads directly from a decoded delta payload.

    Unlike ``build_send_patches`` this path does not require a reconstructed
    full train-shard or a bool mask. Sparse entries are remapped from the
    train-shard flat index space carried on the wire; dense fallback entries are
    expanded into an all-index patch for the op overlap. Unchanged params still
    emit zero-nnz slots to preserve the symmetric P2P schedule.
    """
    profile = _profile_delta_build_enabled()
    profile_t0 = time.perf_counter() if profile else 0.0
    payloads: list[OpDeltaPayload | None] = [None] * len(ops)
    sparse_groups: dict[tuple[str, tuple[int, ...]], list[tuple[int, object]]] = {}
    sparse_ops = 0
    dense_ops = 0
    empty_ops = 0
    dense_ms = 0.0

    for i, op in enumerate(ops):
        name = op.send_shard_meta.name
        if name in decoded.sparse:
            train_shape = tuple(op.send_shard_meta.shape)
            sparse_groups.setdefault((name, train_shape), []).append((i, op))
            sparse_ops += 1
        elif name in decoded.dense:
            if profile:
                t_dense = time.perf_counter()
            payloads[i] = _dense_payload_for_op(op, decoded.dense[name])
            if profile:
                dense_ms += (time.perf_counter() - t_dense) * 1000
            dense_ops += 1
        else:
            payloads[i] = _empty_payload(op)
            empty_ops += 1

    first_pass_ms = (time.perf_counter() - profile_t0) * 1000 if profile else 0.0
    sparse_remap_ms = 0.0
    dtype_cast_ms = 0.0
    sparse_input_nnz = 0
    sparse_output_nnz = 0
    sparse_zero_ops = 0
    group_sizes: list[int] = []
    for (name, train_shape), entries in sparse_groups.items():
        indices, values = decoded.sparse[name]
        patch = SparseWeightPatch(name=name, indices=indices, values=values)
        if profile:
            sparse_input_nnz += int(indices.numel())
            group_sizes.append(len(entries))
            t_remap = time.perf_counter()
        try:
            remapped = remap_delta_indices_for_ops(
                patch,
                train_shape=train_shape,
                operations=[op for _, op in entries],
                assume_sorted=True,
            )
        except TypeError as exc:
            # Long-running colocate workers may have imported an older remap
            # module before this optimization was deployed. Fresh processes use
            # the fast path; mixed-version processes fall back safely.
            if "assume_sorted" not in str(exc):
                raise
            remapped = remap_delta_indices_for_ops(
                patch,
                train_shape=train_shape,
                operations=[op for _, op in entries],
            )
        if profile:
            sparse_remap_ms += (time.perf_counter() - t_remap) * 1000
        for (i, op), op_patch in zip(entries, remapped, strict=True):
            if op_patch is None:
                payloads[i] = _empty_payload(op)
                sparse_zero_ops += 1
                continue
            recv_dtype = getattr(op.recv_shard_meta, "dtype", None)
            patch_values = op_patch.values
            if recv_dtype is not None and patch_values.dtype != recv_dtype:
                if profile:
                    t_cast = time.perf_counter()
                patch_values = patch_values.to(recv_dtype)
                if profile:
                    dtype_cast_ms += (time.perf_counter() - t_cast) * 1000
            if profile:
                sparse_output_nnz += int(op_patch.indices.numel())
            payloads[i] = OpDeltaPayload(
                op=op,
                indices=op_patch.indices,
                values=patch_values,
            )

    missing = [i for i, payload in enumerate(payloads) if payload is None]
    if missing:
        raise RuntimeError(f"delta payload build missed op slots: {missing[:5]}")
    if profile:
        group_count = len(group_sizes)
        avg_group_size = (sum(group_sizes) / group_count) if group_count else 0.0
        max_group_size = max(group_sizes) if group_sizes else 0
        logger.warning(
            "[dte-perf][delta-build] %s ops=%d sparse_ops=%d dense_ops=%d "
            "empty_ops=%d sparse_groups=%d avg_ops_per_group=%.2f "
            "max_ops_per_group=%d sparse_input_nnz=%d sparse_output_nnz=%d "
            "sparse_zero_ops=%d first_pass_ms=%.1f dense_ms=%.1f "
            "sparse_remap_ms=%.1f dtype_cast_ms=%.1f total_ms=%.1f",
            profile_label,
            len(ops),
            sparse_ops,
            dense_ops,
            empty_ops,
            group_count,
            avg_group_size,
            max_group_size,
            sparse_input_nnz,
            sparse_output_nnz,
            sparse_zero_ops,
            first_pass_ms,
            dense_ms,
            sparse_remap_ms,
            dtype_cast_ms,
            (time.perf_counter() - profile_t0) * 1000,
        )
    return [payload for payload in payloads if payload is not None]


def nnz_vector(payloads: list[OpDeltaPayload]) -> torch.Tensor:
    """Control-plane vector: one int32 nnz per op, in op order (fixed size)."""
    return torch.tensor([p.nnz for p in payloads], dtype=torch.int32)


def op_key(op):
    """Stable per-op key for payload lookup, matching the transport layer.

    A param may have multiple ops (different overlap / recv rank); key by
    (send name, recv name, recv_rank, train_slices) so each op maps to its own
    remapped payload. Must stay in sync with ``nccl_stream_batch._op_key``.
    """
    return (
        op.send_shard_meta.name,
        op.recv_shard_meta.name,
        op.recv_rank,
        tuple((s.start, s.stop, s.step) for s in op.train_slices),
    )


def build_send_payloads_by_op(
    ops: list,
    masks: dict[str, torch.Tensor],
    send_params: dict[str, torch.Tensor],
) -> dict:
    """Same as build_send_patches but keyed by op_key for transport lookup."""
    return {
        op_key(op): p for op, p in zip(ops, build_send_patches(ops, masks, send_params))
    }


def build_send_payloads_by_op_from_delta(
    ops: list,
    decoded: DecodedDelta,
    *,
    profile_label: str = "",
) -> dict:
    """Same as build_send_patches_from_delta but keyed for transport lookup."""
    return {
        op_key(op): p
        for op, p in zip(
            ops,
            build_send_patches_from_delta(
                ops,
                decoded,
                profile_label=profile_label,
            ),
        )
    }


def allocate_recv_buffers(
    nnz_list: list[int],
    value_dtype: torch.dtype,
    device="cpu",
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Receiver pre-allocates (idx, val) buffers from the exchanged nnz vector.

    One (possibly empty) buffer pair per op, in the same order the sender packs,
    so the data-plane recv stays aligned and symmetric.
    """
    buffers: list[tuple[torch.Tensor, torch.Tensor]] = []
    for nnz in nnz_list:
        buffers.append(
            (
                torch.empty(nnz, dtype=torch.int32, device=device),
                torch.empty(nnz, dtype=value_dtype, device=device),
            )
        )
    return buffers


@torch.no_grad()
def scatter_recv_into(
    recv_params: dict[str, torch.Tensor],
    ops: list,
    recv_buffers: list[tuple[torch.Tensor, torch.Tensor]],
    slice_fn,
) -> int:
    """Scatter received (idx, val) buffers into the live inference params.

    Args:
        recv_params: ``{hf_name: inference param}`` (write-through view).
        ops: CommunicationOperations (recv direction), parallel to recv_buffers.
        recv_buffers: (idx, val) per op, as filled by the data-plane recv.
        slice_fn: kept for API compatibility with the transport wrapper. The
            payload indices are already in the full inference-shard flat space,
            so the patch is applied to the full live param, not to
            ``tensor[op.inf_slices]``.

    Returns:
        Number of ops that actually applied a non-empty patch (for logging).
    """
    applied = 0
    for op, (idx, val) in zip(ops, recv_buffers, strict=True):
        if idx.numel() == 0:
            continue  # zero-nnz op: nothing to write, but it still had a slot
        name = op.recv_shard_meta.name
        apply_sparse_patch_(recv_params[name], idx, val)
        applied += 1
    return applied
