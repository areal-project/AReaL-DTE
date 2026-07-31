# Licensed under the Apache License, Version 2.0
"""CPU contract tests for dte.core.colocate_protocol.

The cross-rank two-round protocol normally needs a live NCCL context. Here we
inject a synchronous in-process ``schedule_fn`` that loops back each ``isend``
P2POp tensor into the parallel ``irecv`` P2POp tensor — constructing a
``dist.P2POp`` does NOT require an initialized process group (only
``batch_isend_irecv`` does, which the loopback never calls). This pins the
orchestration (nnz round → idx/val round → scatter, zero-nnz symmetry, mixed-
dtype grouping, self-copy) bit-for-bit on CPU; the real NCCL path is verified
A/B on the cluster.
"""

import logging
from types import SimpleNamespace

import torch

from dte.core import colocate_protocol


def _op(
    name,
    train_slices,
    inf_slices,
    recv_rank,
    dtype,
    infer_shape=(4,),
    train_shape=None,
):
    """Mock CommunicationOperation with the fields the protocol reads."""
    return SimpleNamespace(
        send_shard_meta=SimpleNamespace(
            name=name,
            dtype=dtype,
            shape=tuple(train_shape or infer_shape),
        ),
        recv_shard_meta=SimpleNamespace(
            name=name, dtype=dtype, shape=tuple(infer_shape)
        ),
        recv_rank=recv_rank,
        train_slices=tuple(train_slices),
        inf_slices=tuple(inf_slices),
    )


class _Plan:
    def __init__(self, operations):
        self.operations = operations


def _loopback_schedule(
    transfer_rank, world_size, send_p2p, recv_p2p, group, coord, step
):
    """Move each isend tensor into the parallel irecv tensor (in-process).

    The test builds send/recv plans that are mirror-parallel (this rank sends to
    peer P exactly what it receives from peer P), so flattening both sides in the
    same deterministic (sorted-peer, op-order) order pairs them up.
    """
    send_t = [p2p.tensor for peer in sorted(send_p2p) for _, p2p in send_p2p[peer]]
    recv_t = [p2p.tensor for peer in sorted(recv_p2p) for _, p2p in recv_p2p[peer]]
    assert len(send_t) == len(recv_t), (len(send_t), len(recv_t))
    for s, r in zip(send_t, recv_t):
        r.copy_(s)


def _slice_fn(tensor, op, is_train, **kw):
    return tensor[op.train_slices if is_train else op.inf_slices]


class TestTwoRoundExchange:
    def test_sparse_patch_round_trip(self):
        """A cross-rank delta applies the changed elements into recv params."""
        # rank 0 sends its [4] shard to rank 1's inf param, and (mirror) receives
        # the same from rank 1. Mapping: train rank 0 -> infer 0, train 1 -> infer 1.
        dtype = torch.float32
        full = torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=dtype)  # train-shard full
        # mask: elements 1 and 3 changed
        mask = torch.tensor([False, True, False, True])
        op = _op("w", (slice(0, 4),), (slice(0, 4),), recv_rank=1, dtype=dtype)
        send_plan = _Plan({1: [op]})  # peer_rank 1 (train) -> mapped infer 1
        recv_plan = _Plan({1: [op]})  # receive from train rank 1
        recv_params = {"w": torch.zeros(4, dtype=dtype)}

        applied = colocate_protocol.two_round_delta_exchange(
            transfer_rank=0,
            world_size=2,
            send_plan=send_plan,
            recv_plan=recv_plan,
            train_to_infer_device_mapping={0: 0, 1: 1},
            weights_update_group=None,
            send_payloads_by_op=__import__(
                "dte.core.delta_p2p", fromlist=["build_send_payloads_by_op"]
            ).build_send_payloads_by_op([op], {"w": mask}, {"w": full}),
            recv_params=recv_params,
            value_dtype=dtype,
            device=torch.device("cpu"),
            schedule_fn=_loopback_schedule,
            slice_fn=_slice_fn,
            step_id=1,
        )

        # Only the masked positions (1,3) are written; others stay 0.
        assert applied == 1
        expected = torch.tensor([0.0, 20.0, 0.0, 40.0], dtype=dtype)
        assert torch.equal(recv_params["w"], expected)

    def test_zero_nnz_symmetry(self):
        """An op whose overlap had no change still occupies a slot (no crash)."""
        dtype = torch.float32
        full = torch.tensor([1.0, 2.0], dtype=dtype)
        mask = torch.tensor([False, False])  # nothing changed
        op = _op("w", (slice(0, 2),), (slice(0, 2),), recv_rank=1, dtype=dtype)
        recv_params = {"w": torch.tensor([7.0, 8.0], dtype=dtype)}
        applied = colocate_protocol.two_round_delta_exchange(
            transfer_rank=0,
            world_size=2,
            send_plan=_Plan({1: [op]}),
            recv_plan=_Plan({1: [op]}),
            train_to_infer_device_mapping={0: 0, 1: 1},
            weights_update_group=None,
            send_payloads_by_op=__import__(
                "dte.core.delta_p2p", fromlist=["build_send_payloads_by_op"]
            ).build_send_payloads_by_op([op], {"w": mask}, {"w": full}),
            recv_params=recv_params,
            value_dtype=dtype,
            device=torch.device("cpu"),
            schedule_fn=_loopback_schedule,
            slice_fn=_slice_fn,
            step_id=1,
        )
        assert applied == 0  # zero-nnz: nothing applied
        assert torch.equal(recv_params["w"], torch.tensor([7.0, 8.0]))  # unchanged

    def test_coalesced_multi_op_round_trip_preserves_zero_slot(self):
        """Peer-level coalescing still splits received buffers by per-op nnz."""
        dtype = torch.float32
        full = torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=dtype)
        # op0 has one changed element, op1 is intentionally zero-nnz.
        mask = torch.tensor([False, True, False, False])
        op0 = _op("w", (slice(0, 2),), (slice(0, 2),), recv_rank=1, dtype=dtype)
        op1 = _op("w", (slice(2, 4),), (slice(2, 4),), recv_rank=1, dtype=dtype)
        send_plan = _Plan({1: [op0, op1]})
        recv_plan = _Plan({1: [op0, op1]})
        recv_params = {"w": torch.zeros(4, dtype=dtype)}
        schedule_op_counts = []
        send_payloads_by_op = __import__(
            "dte.core.delta_p2p", fromlist=["build_send_payloads_by_op"]
        ).build_send_payloads_by_op([op0, op1], {"w": mask}, {"w": full})
        original_payload_ptrs = {
            tensor.data_ptr()
            for payload in send_payloads_by_op.values()
            for tensor in (payload.indices, payload.values)
            if tensor.numel() > 0
        }
        round2_send_ptrs = []

        def _recording_loopback_schedule(
            transfer_rank, world_size, send_p2p, recv_p2p, group, coord, step
        ):
            counts = (
                sum(len(v) for v in send_p2p.values()),
                sum(len(v) for v in recv_p2p.values()),
            )
            schedule_op_counts.append(counts)
            if counts == (2, 2):
                round2_send_ptrs.extend(
                    p2p.tensor.data_ptr()
                    for peer in sorted(send_p2p)
                    for _, p2p in send_p2p[peer]
                )
            _loopback_schedule(
                transfer_rank, world_size, send_p2p, recv_p2p, group, coord, step
            )

        applied = colocate_protocol.two_round_delta_exchange(
            transfer_rank=0,
            world_size=2,
            send_plan=send_plan,
            recv_plan=recv_plan,
            train_to_infer_device_mapping={0: 0, 1: 1},
            weights_update_group=None,
            send_payloads_by_op=send_payloads_by_op,
            recv_params=recv_params,
            value_dtype=dtype,
            device=torch.device("cpu"),
            schedule_fn=_recording_loopback_schedule,
            slice_fn=_slice_fn,
            step_id=1,
        )

        assert applied == 1
        assert torch.equal(
            recv_params["w"], torch.tensor([0.0, 20.0, 0.0, 0.0], dtype=dtype)
        )
        # Round1 uses one nnz vector per peer; round2 uses one idx and one val
        # tensor for the non-empty peer, instead of two tensors per op.
        assert schedule_op_counts == [(1, 1), (2, 2)]
        assert round2_send_ptrs
        assert original_payload_ptrs.isdisjoint(round2_send_ptrs)

    def test_scatter_uses_full_inference_shard_indices(self):
        """Non-zero inf_slices must write the full live shard, not the slice view."""
        dtype = torch.float32
        recv_params = {"w": torch.zeros(8, dtype=dtype)}
        op = _op(
            "w",
            (slice(4, 8),),
            (slice(4, 8),),
            recv_rank=0,
            dtype=dtype,
            infer_shape=(8,),
            train_shape=(8,),
        )
        applied = __import__(
            "dte.core.delta_p2p", fromlist=["scatter_recv_into"]
        ).scatter_recv_into(
            recv_params,
            [op],
            [(torch.tensor([5, 7], dtype=torch.int32), torch.tensor([50.0, 70.0]))],
            _slice_fn,
        )
        assert applied == 1
        assert torch.equal(
            recv_params["w"],
            torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 50.0, 0.0, 70.0]),
        )


class TestApplyDeltaColocate:
    def test_self_copy_only(self):
        """All ops map to self -> dense self-copy, no cross-rank schedule call."""
        dtype = torch.float32
        full = torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=dtype)
        op = _op("w", (slice(0, 4),), (slice(0, 4),), recv_rank=0, dtype=dtype)
        recv_params = {"w": torch.zeros(4, dtype=dtype)}

        def _selfcopy(tensors_to_copy, copy_ops, recv_parameters, stage):
            for t, cop in zip(tensors_to_copy, copy_ops):
                recv_parameters[cop.recv_shard_meta.name][cop.inf_slices].copy_(t)

        def _schedule_must_be_empty(tr, ws, send_p2p, recv_p2p, g, c, s):
            assert not any(send_p2p.values()) and not any(recv_p2p.values())

        applied = colocate_protocol.apply_delta_colocate(
            transfer_rank=0,
            world_size=2,
            send_plan=_Plan({0: [op]}),  # peer 0 -> maps to self (transfer_rank 0)
            recv_plan=_Plan({0: [op]}),
            train_to_infer_device_mapping={0: 0},
            infer_to_train_device_mapping={0: 0},
            weights_update_group=None,
            full_params={"w": full},
            masks={"w": torch.ones(4, dtype=torch.bool)},
            recv_params=recv_params,
            device=torch.device("cpu"),
            schedule_fn=_schedule_must_be_empty,
            slice_fn=_slice_fn,
            selfcopy_fn=_selfcopy,
            step_id=1,
        )
        assert applied == 0  # all self-copy, no cross-rank patches
        assert torch.equal(recv_params["w"], full)  # self-copied full

    def test_filter_plan_by_dtype(self):
        """Mixed-dtype ops are partitioned; each group sees only its dtype."""
        op_bf16 = _op("a", (slice(0, 2),), (slice(0, 2),), 1, torch.bfloat16)
        op_fp32 = _op("b", (slice(0, 2),), (slice(0, 2),), 1, torch.float32)
        plan = _Plan({1: [op_bf16, op_fp32]})
        v = colocate_protocol._filter_plan_by_dtype(plan, torch.float32, is_send=True)
        assert v.operations == {1: [op_fp32]}
        v2 = colocate_protocol._filter_plan_by_dtype(plan, torch.bfloat16, is_send=True)
        assert v2.operations == {1: [op_bf16]}

    def test_decoded_delta_self_sparse_without_full_base(self):
        from dte.core.codec import DecodedDelta, DeltaHeader

        dtype = torch.float32
        op = _op(
            "w",
            (slice(4, 8),),
            (slice(4, 8),),
            recv_rank=0,
            dtype=dtype,
            infer_shape=(8,),
            train_shape=(8,),
        )
        decoded = DecodedDelta(
            header=DeltaHeader(payload_version=2, base_version=1, num_sparse=1),
            sparse={
                "w": (
                    torch.tensor([5, 7], dtype=torch.int32),
                    torch.tensor([50.0, 70.0], dtype=dtype),
                )
            },
        )
        recv_params = {"w": torch.zeros(8, dtype=dtype)}

        applied = colocate_protocol.apply_decoded_delta_colocate(
            transfer_rank=0,
            world_size=2,
            send_plan=_Plan({0: [op]}),
            recv_plan=_Plan({0: [op]}),
            train_to_infer_device_mapping={0: 0},
            infer_to_train_device_mapping={0: 0},
            weights_update_group=None,
            decoded=decoded,
            recv_params=recv_params,
            device=torch.device("cpu"),
            schedule_fn=_loopback_schedule,
            slice_fn=_slice_fn,
            step_id=2,
        )

        assert applied == 1
        assert torch.equal(
            recv_params["w"],
            torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 50.0, 0.0, 70.0]),
        )

    def test_decoded_delta_cross_sparse_multi_op_preserves_zero_slot(self):
        from dte.core.codec import DecodedDelta, DeltaHeader

        dtype = torch.float32
        op0 = _op(
            "w",
            (slice(0, 2),),
            (slice(0, 2),),
            recv_rank=1,
            dtype=dtype,
            infer_shape=(6,),
            train_shape=(6,),
        )
        op1 = _op(
            "w",
            (slice(2, 4),),
            (slice(2, 4),),
            recv_rank=1,
            dtype=dtype,
            infer_shape=(6,),
            train_shape=(6,),
        )
        op2 = _op(
            "w",
            (slice(4, 6),),
            (slice(4, 6),),
            recv_rank=1,
            dtype=dtype,
            infer_shape=(6,),
            train_shape=(6,),
        )
        decoded = DecodedDelta(
            header=DeltaHeader(payload_version=2, base_version=1, num_sparse=1),
            sparse={
                "w": (
                    torch.tensor([1, 5], dtype=torch.int32),
                    torch.tensor([20.0, 60.0], dtype=dtype),
                )
            },
        )
        recv_params = {"w": torch.zeros(6, dtype=dtype)}

        applied = colocate_protocol.apply_decoded_delta_colocate(
            transfer_rank=0,
            world_size=2,
            send_plan=_Plan({1: [op0, op1, op2]}),
            recv_plan=_Plan({1: [op0, op1, op2]}),
            train_to_infer_device_mapping={0: 0, 1: 1},
            infer_to_train_device_mapping={0: 0},
            weights_update_group=None,
            decoded=decoded,
            recv_params=recv_params,
            device=torch.device("cpu"),
            schedule_fn=_loopback_schedule,
            slice_fn=_slice_fn,
            step_id=2,
        )

        assert applied == 2
        assert torch.equal(
            recv_params["w"],
            torch.tensor([0.0, 20.0, 0.0, 0.0, 0.0, 60.0]),
        )

    def test_decoded_delta_build_profile_preserves_payload(self, monkeypatch, caplog):
        from dte.core.codec import DecodedDelta, DeltaHeader
        from dte.core.delta_p2p import build_send_patches_from_delta

        monkeypatch.setenv("DTE_DELTA_REMAP_PROFILE", "1")
        caplog.set_level(logging.WARNING, logger="dte.core.delta_p2p")

        dtype = torch.float32
        op0 = _op(
            "w",
            (slice(0, 2),),
            (slice(0, 2),),
            recv_rank=1,
            dtype=dtype,
            infer_shape=(6,),
            train_shape=(6,),
        )
        op1 = _op(
            "w",
            (slice(2, 4),),
            (slice(2, 4),),
            recv_rank=1,
            dtype=dtype,
            infer_shape=(6,),
            train_shape=(6,),
        )
        op2 = _op(
            "w",
            (slice(4, 6),),
            (slice(4, 6),),
            recv_rank=1,
            dtype=dtype,
            infer_shape=(6,),
            train_shape=(6,),
        )
        decoded = DecodedDelta(
            header=DeltaHeader(payload_version=2, base_version=1, num_sparse=1),
            sparse={
                "w": (
                    torch.tensor([1, 5], dtype=torch.int32),
                    torch.tensor([20.0, 60.0], dtype=dtype),
                )
            },
        )

        payloads = build_send_patches_from_delta(
            [op0, op1, op2],
            decoded,
            profile_label="unit",
        )

        assert [payload.nnz for payload in payloads] == [1, 0, 1]
        assert torch.equal(payloads[0].indices, torch.tensor([1], dtype=torch.int32))
        assert torch.equal(payloads[2].indices, torch.tensor([5], dtype=torch.int32))
        assert "[dte-perf][delta-build] unit" in caplog.text
        assert "sparse_groups=1" in caplog.text

    def test_decoded_delta_self_dense_fallback_without_full_base(self):
        from dte.core.codec import DecodedDelta, DeltaHeader

        dtype = torch.float32
        op = _op(
            "w",
            (slice(4, 8),),
            (slice(4, 8),),
            recv_rank=0,
            dtype=dtype,
            infer_shape=(8,),
            train_shape=(8,),
        )
        dense = torch.arange(8, dtype=dtype) * 10
        decoded = DecodedDelta(
            header=DeltaHeader(payload_version=2, base_version=1, num_dense=1),
            dense={"w": dense},
        )
        recv_params = {"w": torch.zeros(8, dtype=dtype)}

        applied = colocate_protocol.apply_decoded_delta_colocate(
            transfer_rank=0,
            world_size=2,
            send_plan=_Plan({0: [op]}),
            recv_plan=_Plan({0: [op]}),
            train_to_infer_device_mapping={0: 0},
            infer_to_train_device_mapping={0: 0},
            weights_update_group=None,
            decoded=decoded,
            recv_params=recv_params,
            device=torch.device("cpu"),
            schedule_fn=_loopback_schedule,
            slice_fn=_slice_fn,
            step_id=2,
        )

        assert applied == 1
        assert torch.equal(
            recv_params["w"],
            torch.tensor([0.0, 0.0, 0.0, 0.0, 40.0, 50.0, 60.0, 70.0]),
        )
