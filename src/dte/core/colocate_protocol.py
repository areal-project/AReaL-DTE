# Licensed under the Apache License, Version 2.0
"""Transport-agnostic two-round colocate delta protocol orchestration.

This module owns the *control flow* of the sparse colocate weight exchange that
previously lived inline in awex ``nccl_stream_batch.transfer_delta_in_colocate_mode``
+ ``apply_delta_colocate``. Moving it here makes the two-round protocol portable:
a backend supplies only the low-level primitives via injected callbacks —

- ``schedule_fn``: the deadlock-safe P2P schedule that actually moves the built
  ``dist.P2POp`` lists. Signature mirrors awex's
  ``execute_recursive_partition_stream_transfer(transfer_rank, world_size,
  send_p2p, recv_p2p, group, rank_coordinate, step_id)``. ``send_p2p`` /
  ``recv_p2p`` are ``Dict[peer_rank] -> List[(op, dist.P2POp)]``.
- ``slice_fn``: reshard geometry, mirrors awex ``slice_tensor(tensor, op,
  is_train, slice_context=...)`` — returns the op's overlap region of a tensor.
- ``selfcopy_fn``: the local dense self-copy, mirrors awex
  ``execute_tensors_to_copy(tensors, copy_ops, recv_params, stage)``.

The transport-agnostic payload primitives (``nnz_vector`` /
``allocate_recv_buffers`` / ``scatter_recv_into`` / ``build_send_payloads_by_op``)
come from ``dte.core.delta_p2p``. The body is a faithful (bit-for-bit intended)
port of the awex implementation; see the inline ``awex parity`` notes where a
detail matters for correctness (device, ``copy=True``, zero-nnz slots, mixed-
dtype group order).
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable

import torch
import torch.distributed as dist

from dte.core.codec import DecodedDelta, apply_sparse_patch_
from dte.core.delta_p2p import (
    allocate_recv_buffers,
    build_send_payloads_by_op,
    build_send_payloads_by_op_from_delta,
    op_key,
    scatter_recv_into,
)

logger = logging.getLogger(__name__)

__all__ = [
    "two_round_delta_exchange",
    "apply_delta_colocate",
    "apply_decoded_delta_colocate",
]


def two_round_delta_exchange(**kwargs) -> int:
    """Cross-rank sparse delta over P2P (the bandwidth-critical path).

    The default protocol coalesces one peer's many sparse op payloads into a
    small number of flat tensors. Set ``DTE_DELTA_P2P_COALESCE=0`` to use the
    original per-op protocol for diagnostics or rollback.
    """
    env = os.environ.get("DTE_DELTA_P2P_COALESCE", "1").strip().lower()
    if env in {"0", "false", "no", "off"}:
        return _two_round_delta_exchange_per_op(**kwargs)
    return _two_round_delta_exchange_coalesced(**kwargs)


def _two_round_delta_exchange_coalesced(
    *,
    transfer_rank: int,
    world_size: int,
    send_plan,
    recv_plan,
    train_to_infer_device_mapping: dict,
    weights_update_group,
    send_payloads_by_op: dict,
    recv_params: dict,
    value_dtype: torch.dtype,
    device: torch.device,
    schedule_fn: Callable,
    slice_fn: Callable,
    rank_coordinate: str = "",
    step_id: int = -1,
) -> int:
    """Coalesced sparse P2P protocol.

    Round 1 sends one int32 nnz vector per peer. Round 2 sends one flat index
    tensor and one flat value tensor per non-empty peer. The receiver splits
    the flat buffers back into per-op views using the nnz vector, preserving the
    original scatter semantics while reducing P2P op count from O(params) to
    O(peers).
    """
    perf_start = time.perf_counter()
    send_ops = dict(send_plan.operations)
    recv_ops = dict(recv_plan.operations)
    total_send_ops = sum(len(ops) for ops in send_ops.values())
    total_recv_ops = sum(len(ops) for ops in recv_ops.values())
    total_send_nnz = sum(payload.nnz for payload in send_payloads_by_op.values())

    # ---- Round 1: exchange nnz vectors (one int32 vector per peer) ----
    nnz_send_p2p: dict = {}
    nnz_recv_p2p: dict = {}
    recv_nnz_buf: dict = {}
    round1_send_p2p_ops = 0
    round1_recv_p2p_ops = 0

    for peer_rank, ops in send_ops.items():
        mapped_peer = train_to_infer_device_mapping.get(peer_rank, peer_rank)
        if mapped_peer == transfer_rank or not ops:
            continue
        nnz_t = torch.tensor(
            [send_payloads_by_op[op_key(op)].nnz for op in ops],
            dtype=torch.int32,
            device=device,
        )
        recv_rank = train_to_infer_device_mapping.get(
            ops[0].recv_rank, ops[0].recv_rank
        )
        nnz_send_p2p[mapped_peer] = [
            (
                ops[0],
                dist.P2POp(dist.isend, nnz_t, recv_rank, group=weights_update_group),
            )
        ]
        round1_send_p2p_ops += 1

    for send_rank, ops in recv_ops.items():
        recv_from = train_to_infer_device_mapping[send_rank]
        if recv_from == transfer_rank or not ops:
            continue
        nnz_t = torch.empty(len(ops), dtype=torch.int32, device=device)
        nnz_recv_p2p[recv_from] = [
            (
                ops[0],
                dist.P2POp(dist.irecv, nnz_t, recv_from, group=weights_update_group),
            )
        ]
        recv_nnz_buf[recv_from] = (ops, nnz_t)
        round1_recv_p2p_ops += 1

    t_round1 = time.perf_counter()
    schedule_fn(
        transfer_rank,
        world_size,
        nnz_send_p2p,
        nnz_recv_p2p,
        weights_update_group,
        rank_coordinate,
        step_id,
    )
    round1_ms = (time.perf_counter() - t_round1) * 1000

    # ---- Allocate flat recv idx/val buffers from received nnz vectors ----
    t_alloc = time.perf_counter()
    recv_payload_bufs: dict = {}
    total_recv_nnz = 0
    for peer, (ops, nnz_t) in recv_nnz_buf.items():
        nnz_list = [int(v) for v in nnz_t.cpu().tolist()]
        total_nnz = sum(nnz_list)
        total_recv_nnz += total_nnz
        idx_buf = torch.empty(total_nnz, dtype=torch.int32, device=device)
        val_buf = torch.empty(total_nnz, dtype=value_dtype, device=device)
        recv_payload_bufs[peer] = (ops, nnz_list, idx_buf, val_buf)
    alloc_ms = (time.perf_counter() - t_alloc) * 1000

    # ---- Round 2: exchange coalesced idx + val buffers ----
    t_build_round2 = time.perf_counter()
    pay_send_p2p: dict = {}
    pay_recv_p2p: dict = {}
    round2_send_p2p_ops = 0
    round2_recv_p2p_ops = 0

    for peer_rank, ops in send_ops.items():
        mapped_peer = train_to_infer_device_mapping.get(peer_rank, peer_rank)
        if mapped_peer == transfer_rank or not ops:
            continue
        idx_chunks = []
        val_chunks = []
        for op in ops:
            payload = send_payloads_by_op[op_key(op)]
            if payload.nnz == 0:
                continue
            # torch.cat below creates the independent peer-level send buffer.
            # Avoid an extra per-op copy before that concat.
            idx_chunks.append(
                payload.indices.to(device=device, dtype=torch.int32)
                .reshape(-1)
                .contiguous()
            )
            val_chunks.append(
                payload.values.to(device=device, dtype=value_dtype)
                .reshape(-1)
                .contiguous()
            )
        if not idx_chunks:
            continue
        idx = torch.cat(idx_chunks, dim=0)
        val = torch.cat(val_chunks, dim=0)
        recv_rank = train_to_infer_device_mapping.get(
            ops[0].recv_rank, ops[0].recv_rank
        )
        pay_send_p2p[mapped_peer] = [
            (
                ops[0],
                dist.P2POp(dist.isend, idx, recv_rank, group=weights_update_group),
            ),
            (
                ops[0],
                dist.P2POp(dist.isend, val, recv_rank, group=weights_update_group),
            ),
        ]
        round2_send_p2p_ops += 2

    for peer, (ops, _nnz_list, idx_buf, val_buf) in recv_payload_bufs.items():
        if idx_buf.numel() == 0:
            continue
        pay_recv_p2p[peer] = [
            (
                ops[0],
                dist.P2POp(dist.irecv, idx_buf, peer, group=weights_update_group),
            ),
            (
                ops[0],
                dist.P2POp(dist.irecv, val_buf, peer, group=weights_update_group),
            ),
        ]
        round2_recv_p2p_ops += 2
    build_round2_ms = (time.perf_counter() - t_build_round2) * 1000

    t_round2 = time.perf_counter()
    schedule_fn(
        transfer_rank,
        world_size,
        pay_send_p2p,
        pay_recv_p2p,
        weights_update_group,
        rank_coordinate,
        step_id,
    )
    round2_ms = (time.perf_counter() - t_round2) * 1000

    # ---- Scatter received flat buffers into live inference params ----
    t_scatter = time.perf_counter()
    recv_ops_flat = []
    recv_bufs_flat = []
    for ops, nnz_list, idx_buf, val_buf in recv_payload_bufs.values():
        offset = 0
        for op, nnz in zip(ops, nnz_list, strict=True):
            recv_ops_flat.append(op)
            recv_bufs_flat.append(
                (
                    idx_buf.narrow(0, offset, nnz),
                    val_buf.narrow(0, offset, nnz),
                )
            )
            offset += nnz
    applied = scatter_recv_into(
        recv_params,
        recv_ops_flat,
        recv_bufs_flat,
        lambda t, op: slice_fn(t, op, False),
    )
    scatter_ms = (time.perf_counter() - t_scatter) * 1000
    logger.info(
        "[%s] delta transfer step %s: applied %d non-empty patches",
        rank_coordinate,
        step_id,
        applied,
    )
    logger.warning(
        "[dte-perf][delta-p2p] rank=%s step=%s dtype=%s "
        "send_ops=%d recv_ops=%d send_nnz=%d recv_nnz=%d "
        "round1_p2p_send=%d round1_p2p_recv=%d "
        "round2_p2p_send=%d round2_p2p_recv=%d "
        "round1_ms=%.1f alloc_ms=%.1f build_round2_ms=%.1f "
        "round2_ms=%.1f scatter_ms=%.1f total_ms=%.1f mode=coalesced",
        rank_coordinate,
        step_id,
        value_dtype,
        total_send_ops,
        total_recv_ops,
        total_send_nnz,
        total_recv_nnz,
        round1_send_p2p_ops,
        round1_recv_p2p_ops,
        round2_send_p2p_ops,
        round2_recv_p2p_ops,
        round1_ms,
        alloc_ms,
        build_round2_ms,
        round2_ms,
        scatter_ms,
        (time.perf_counter() - perf_start) * 1000,
    )
    return applied


def _two_round_delta_exchange_per_op(
    *,
    transfer_rank: int,
    world_size: int,
    send_plan,
    recv_plan,
    train_to_infer_device_mapping: dict,
    weights_update_group,
    send_payloads_by_op: dict,
    recv_params: dict,
    value_dtype: torch.dtype,
    device: torch.device,
    schedule_fn: Callable,
    slice_fn: Callable,
    rank_coordinate: str = "",
    step_id: int = -1,
) -> int:
    """Original per-op sparse P2P protocol kept as a diagnostic fallback.

    Faithful port of awex ``transfer_delta_in_colocate_mode``. Local self-copy
    ops (peer maps to self) are NOT handled here — the caller does them via the
    dense path. Returns the number of non-empty patches applied.
    """
    perf_start = time.perf_counter()
    send_ops = dict(send_plan.operations)
    recv_ops = dict(recv_plan.operations)
    total_send_ops = sum(len(ops) for ops in send_ops.values())
    total_recv_ops = sum(len(ops) for ops in recv_ops.values())
    total_send_nnz = sum(payload.nnz for payload in send_payloads_by_op.values())

    # ---- Round 1: exchange nnz (one int32 per op, symmetric) ----
    nnz_send_p2p: dict = {}
    nnz_recv_p2p: dict = {}
    recv_nnz_buf: dict = {}

    for peer_rank, ops in send_ops.items():
        mapped_peer = train_to_infer_device_mapping.get(peer_rank, peer_rank)
        if mapped_peer == transfer_rank:
            continue  # self-copy handled by caller
        p2p = []
        for op in ops:
            payload = send_payloads_by_op[op_key(op)]
            nnz_t = torch.tensor([payload.nnz], dtype=torch.int32, device=device)
            recv_rank = train_to_infer_device_mapping.get(op.recv_rank, op.recv_rank)
            p2p.append(
                (
                    op,
                    dist.P2POp(
                        dist.isend, nnz_t, recv_rank, group=weights_update_group
                    ),
                )
            )
        nnz_send_p2p[mapped_peer] = p2p

    for send_rank, ops in recv_ops.items():
        recv_from = train_to_infer_device_mapping[send_rank]
        if recv_from == transfer_rank:
            continue
        p2p = []
        bufs = []
        for op in ops:
            nnz_t = torch.empty(1, dtype=torch.int32, device=device)
            p2p.append(
                (
                    op,
                    dist.P2POp(
                        dist.irecv, nnz_t, recv_from, group=weights_update_group
                    ),
                )
            )
            bufs.append((op, nnz_t))
        nnz_recv_p2p[recv_from] = p2p
        recv_nnz_buf[recv_from] = bufs

    t_round1 = time.perf_counter()
    schedule_fn(
        transfer_rank,
        world_size,
        nnz_send_p2p,
        nnz_recv_p2p,
        weights_update_group,
        rank_coordinate,
        step_id,
    )
    round1_ms = (time.perf_counter() - t_round1) * 1000

    # ---- Allocate recv idx/val buffers from received nnz ----
    t_alloc = time.perf_counter()
    recv_payload_bufs: dict = {}
    total_recv_nnz = 0
    for peer, bufs in recv_nnz_buf.items():
        entries = []
        nnz_list = [int(nnz_t.item()) for _, nnz_t in bufs]
        total_recv_nnz += sum(nnz_list)
        idx_val = allocate_recv_buffers(nnz_list, value_dtype, device=device)
        for (op, _nnz_t), (idx_buf, val_buf) in zip(bufs, idx_val):
            entries.append((op, idx_buf, val_buf))
        recv_payload_bufs[peer] = entries
    alloc_ms = (time.perf_counter() - t_alloc) * 1000

    # ---- Round 2: exchange idx + val (sizes now known both sides) ----
    t_build_round2 = time.perf_counter()
    pay_send_p2p: dict = {}
    pay_recv_p2p: dict = {}
    for peer_rank, ops in send_ops.items():
        mapped_peer = train_to_infer_device_mapping.get(peer_rank, peer_rank)
        if mapped_peer == transfer_rank:
            continue
        p2p = []
        for op in ops:
            payload = send_payloads_by_op[op_key(op)]
            recv_rank = train_to_infer_device_mapping.get(op.recv_rank, op.recv_rank)
            # awex parity: copy=True guarantees an independent buffer for the
            # async isend (.to(device) is a no-op when already on-device).
            idx = payload.indices.to(device=device, copy=True).contiguous()
            val = payload.values.to(
                device=device, dtype=value_dtype, copy=True
            ).contiguous()
            p2p.append(
                (
                    op,
                    dist.P2POp(dist.isend, idx, recv_rank, group=weights_update_group),
                )
            )
            p2p.append(
                (
                    op,
                    dist.P2POp(dist.isend, val, recv_rank, group=weights_update_group),
                )
            )
        pay_send_p2p[mapped_peer] = p2p

    for peer, entries in recv_payload_bufs.items():
        p2p = []
        for op, idx_buf, val_buf in entries:
            p2p.append(
                (op, dist.P2POp(dist.irecv, idx_buf, peer, group=weights_update_group))
            )
            p2p.append(
                (op, dist.P2POp(dist.irecv, val_buf, peer, group=weights_update_group))
            )
        pay_recv_p2p[peer] = p2p
    build_round2_ms = (time.perf_counter() - t_build_round2) * 1000

    t_round2 = time.perf_counter()
    schedule_fn(
        transfer_rank,
        world_size,
        pay_send_p2p,
        pay_recv_p2p,
        weights_update_group,
        rank_coordinate,
        step_id,
    )
    round2_ms = (time.perf_counter() - t_round2) * 1000

    # ---- Scatter received patches into live inference params ----
    t_scatter = time.perf_counter()
    recv_ops_flat = []
    recv_bufs_flat = []
    for entries in recv_payload_bufs.values():
        for op, idx_buf, val_buf in entries:
            recv_ops_flat.append(op)
            recv_bufs_flat.append((idx_buf, val_buf))
    applied = scatter_recv_into(
        recv_params,
        recv_ops_flat,
        recv_bufs_flat,
        lambda t, op: slice_fn(t, op, False),
    )
    scatter_ms = (time.perf_counter() - t_scatter) * 1000
    logger.info(
        "[%s] delta transfer step %s: applied %d non-empty patches",
        rank_coordinate,
        step_id,
        applied,
    )
    logger.warning(
        "[dte-perf][delta-p2p] rank=%s step=%s dtype=%s "
        "send_ops=%d recv_ops=%d send_nnz=%d recv_nnz=%d "
        "round1_ms=%.1f alloc_ms=%.1f build_round2_ms=%.1f "
        "round2_ms=%.1f scatter_ms=%.1f total_ms=%.1f mode=per_op",
        rank_coordinate,
        step_id,
        value_dtype,
        total_send_ops,
        total_recv_ops,
        total_send_nnz,
        total_recv_nnz,
        round1_ms,
        alloc_ms,
        build_round2_ms,
        round2_ms,
        scatter_ms,
        (time.perf_counter() - perf_start) * 1000,
    )
    return applied


class _PlanView:
    """Lightweight filtered transfer-plan view (only ``.operations`` is read)."""

    def __init__(self, operations):
        self.operations = operations


def _filter_plan_by_dtype(plan, dtype, *, is_send):
    """Keep only ops whose wire value dtype matches.

    Delta payload values are received directly into inference parameters, so the
    wire dtype is the receiver shard dtype. This matters for Flash/Bailing cases
    where a training shard is bf16 but the SGLang-side weight is fp32.
    ``is_send`` is kept for API compatibility with older callers.
    """
    del is_send
    meta_attr = "recv_shard_meta"
    filtered = {}
    for peer_rank, ops in plan.operations.items():
        kept = [op for op in ops if getattr(op, meta_attr).dtype == dtype]
        if kept:
            filtered[peer_rank] = kept
    return _PlanView(filtered)


def _ops_by_recv_dtype(ops: list) -> dict[torch.dtype, list]:
    grouped: dict[torch.dtype, list] = {}
    for op in ops:
        grouped.setdefault(getattr(op.recv_shard_meta, "dtype"), []).append(op)
    return grouped


def _self_apply_payloads(payloads, recv_params: dict) -> int:
    applied = 0
    for payload in payloads:
        if payload.nnz == 0:
            continue
        name = payload.op.recv_shard_meta.name
        apply_sparse_patch_(recv_params[name], payload.indices, payload.values)
        applied += 1
    return applied


def apply_delta_colocate(
    *,
    transfer_rank: int,
    world_size: int,
    send_plan,
    recv_plan,
    train_to_infer_device_mapping: dict,
    infer_to_train_device_mapping: dict,
    weights_update_group,
    full_params: dict,
    masks: dict,
    recv_params: dict,
    device: torch.device,
    schedule_fn: Callable,
    slice_fn: Callable,
    selfcopy_fn: Callable,
    rank_coordinate: str = "",
    step_id: int = -1,
) -> int:
    """Reader-facing: local self-copy (full, dense) + cross-rank sparse delta.

    Faithful port of awex ``apply_delta_colocate``. ``full_params`` is the full
    reconstructed train-shard (``_delta_base`` + delta applied), used for the
    local self-copy; ``masks`` drives the per-op sparse projection for the
    cross-rank ops. Returns total non-empty patches applied cross-rank.
    """
    send_ops = dict(send_plan.operations)

    # --- Self-copy segment (local, full, dense logic) ---
    train_slice_context: dict = {}
    tensors_to_copy = []
    for peer_rank, ops in send_ops.items():
        mapped_peer = train_to_infer_device_mapping.get(peer_rank, peer_rank)
        if mapped_peer != transfer_rank:
            continue  # cross-rank handled below
        for op in ops:
            send_tensor = full_params[op.send_shard_meta.name]
            tensors_to_copy.append(
                slice_fn(send_tensor, op, True, slice_context=train_slice_context)
            )
    if tensors_to_copy:
        local_send_rank = infer_to_train_device_mapping[transfer_rank]
        selfcopy_fn(
            tensors_to_copy,
            recv_plan.operations[local_send_rank],
            recv_params,
            f"delta self-copy for {rank_coordinate}-{step_id}",
        )

    # --- Cross-rank segment (sparse delta over P2P), grouped by dtype ---
    # Round 1 carries only nnz (not per-op dtype), so the recv side pre-allocates
    # val buffers from ONE dtype. Partition cross-rank ops by param dtype and run
    # one full two-round transfer per uniform-dtype group. Deadlock symmetry: the
    # group set must be identical and same-order on every rank, so derive it from
    # recv_params (full live inference view) and iterate sorted(by str); every
    # rank enters every group unconditionally (empty -> zero-nnz, still symmetric).
    cross_ops = []
    for peer_rank, ops in send_ops.items():
        mapped_peer = train_to_infer_device_mapping.get(peer_rank, peer_rank)
        if mapped_peer == transfer_rank:
            continue
        cross_ops.extend(ops)

    ops_by_dtype = _ops_by_recv_dtype(cross_ops)

    all_dtypes = sorted({t.dtype for t in recv_params.values()}, key=str)

    applied = 0
    for dt in all_dtypes:
        group_ops = ops_by_dtype.get(dt, [])
        send_payloads_by_op = build_send_payloads_by_op(group_ops, masks, full_params)
        sub_send_plan = _filter_plan_by_dtype(send_plan, dt, is_send=True)
        sub_recv_plan = _filter_plan_by_dtype(recv_plan, dt, is_send=False)
        applied += two_round_delta_exchange(
            transfer_rank=transfer_rank,
            world_size=world_size,
            send_plan=sub_send_plan,
            recv_plan=sub_recv_plan,
            train_to_infer_device_mapping=train_to_infer_device_mapping,
            weights_update_group=weights_update_group,
            send_payloads_by_op=send_payloads_by_op,
            recv_params=recv_params,
            value_dtype=dt,
            device=device,
            schedule_fn=schedule_fn,
            slice_fn=slice_fn,
            rank_coordinate=rank_coordinate,
            step_id=step_id,
        )
    return applied


def apply_decoded_delta_colocate(
    *,
    transfer_rank: int,
    world_size: int,
    send_plan,
    recv_plan,
    train_to_infer_device_mapping: dict,
    infer_to_train_device_mapping: dict,
    weights_update_group,
    decoded: DecodedDelta,
    recv_params: dict,
    device: torch.device,
    schedule_fn: Callable,
    slice_fn: Callable,
    rank_coordinate: str = "",
    step_id: int = -1,
) -> int:
    """Apply a decoded delta payload directly to live inference weights.

    This is the receiver path that does not materialize a CPU/GPU full model
    base. Each rank holds the delta payload from its paired training rank:
    local mapped ops are patched in place, and cross-rank ops use the same
    two-round sparse P2P protocol as ``apply_delta_colocate``.
    """
    del infer_to_train_device_mapping
    perf_start = time.perf_counter()
    send_ops = dict(send_plan.operations)

    self_ops = []
    cross_ops = []
    for peer_rank, ops in send_ops.items():
        mapped_peer = train_to_infer_device_mapping.get(peer_rank, peer_rank)
        if mapped_peer == transfer_rank:
            self_ops.extend(ops)
        else:
            cross_ops.extend(ops)

    applied = 0
    self_payload_ms = 0.0
    self_apply_ms = 0.0
    self_nnz = 0
    if self_ops:
        t_payload = time.perf_counter()
        self_payloads = build_send_payloads_by_op_from_delta(
            self_ops,
            decoded,
            profile_label=f"rank={rank_coordinate} step={step_id} phase=self",
        )
        self_payload_ms = (time.perf_counter() - t_payload) * 1000
        self_nnz = sum(payload.nnz for payload in self_payloads.values())
        t_apply = time.perf_counter()
        applied += _self_apply_payloads(
            self_payloads.values(),
            recv_params,
        )
        self_apply_ms = (time.perf_counter() - t_apply) * 1000

    ops_by_dtype = _ops_by_recv_dtype(cross_ops)
    all_dtypes = sorted({t.dtype for t in recv_params.values()}, key=str)
    cross_build_ms = 0.0
    cross_exchange_ms = 0.0
    dtype_summaries = []
    for dt in all_dtypes:
        group_ops = ops_by_dtype.get(dt, [])
        t_build = time.perf_counter()
        send_payloads_by_op = build_send_payloads_by_op_from_delta(
            group_ops,
            decoded,
            profile_label=(
                f"rank={rank_coordinate} step={step_id} phase=cross dtype={dt}"
            ),
        )
        build_ms = (time.perf_counter() - t_build) * 1000
        cross_build_ms += build_ms
        sub_send_plan = _filter_plan_by_dtype(send_plan, dt, is_send=True)
        sub_recv_plan = _filter_plan_by_dtype(recv_plan, dt, is_send=False)
        t_exchange = time.perf_counter()
        applied += two_round_delta_exchange(
            transfer_rank=transfer_rank,
            world_size=world_size,
            send_plan=sub_send_plan,
            recv_plan=sub_recv_plan,
            train_to_infer_device_mapping=train_to_infer_device_mapping,
            weights_update_group=weights_update_group,
            send_payloads_by_op=send_payloads_by_op,
            recv_params=recv_params,
            value_dtype=dt,
            device=device,
            schedule_fn=schedule_fn,
            slice_fn=slice_fn,
            rank_coordinate=rank_coordinate,
            step_id=step_id,
        )
        exchange_ms = (time.perf_counter() - t_exchange) * 1000
        cross_exchange_ms += exchange_ms
        dtype_summaries.append(
            f"{dt}:ops={len(group_ops)},nnz="
            f"{sum(payload.nnz for payload in send_payloads_by_op.values())},"
            f"build_ms={build_ms:.1f},exchange_ms={exchange_ms:.1f}"
        )
    logger.info(
        "[%s] decoded delta apply step %s: applied %d non-empty patches",
        rank_coordinate,
        step_id,
        applied,
    )
    logger.warning(
        "[dte-perf][decoded-apply] rank=%s step=%s "
        "self_ops=%d self_nnz=%d self_payload_ms=%.1f self_apply_ms=%.1f "
        "cross_ops=%d cross_build_ms=%.1f cross_exchange_ms=%.1f "
        "total_ms=%.1f dtype_groups=%s",
        rank_coordinate,
        step_id,
        len(self_ops),
        self_nnz,
        self_payload_ms,
        self_apply_ms,
        len(cross_ops),
        cross_build_ms,
        cross_exchange_ms,
        (time.perf_counter() - perf_start) * 1000,
        ";".join(dtype_summaries),
    )
    return applied
