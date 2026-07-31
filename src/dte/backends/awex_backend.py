# Licensed under the Apache License, Version 2.0
"""AwexTransport — the awex NCCL backend for dte.

This is the *dirty-work isolation file*: every dependency on awex lives here and
nowhere else in dte. The engine and core never import awex. awex is reused, not
modified and not monkey-patched — we compose an awex
``NCCLWorkerWeightsReader`` and call its public surface.

Layering (mirrors vLLM sparse-transfer on top of NCCL):
- dte owns the contract (``Transport``) and the incremental algorithm
  (``dte.core``);
- awex provides the reshard geometry (``TransferPlanBuilder``) and the
  deadlock-safe P2P schedule (``execute_recursive_partition_stream_transfer``),
  which already ships on upstream main.

──────────────────────────────────────────────────────────────────────────────
STATUS: skeleton wired to awex's *real* signatures (verified against
asystem-awex feat-delta-transfer). The cross-rank sparse transfer body and the
bitwise parity check require a 2-GPU + MetaServer + megatron/sglang cluster and
CANNOT be exercised on CPU. See docs/M3-gpu-verification.md. Methods that need
that runtime raise NotImplementedError with a pointer rather than pretend.
──────────────────────────────────────────────────────────────────────────────

Impedance note (real design point, not hand-waving): awex's colocate transfer
is a *single symmetric* operation — each rank sends and receives in the same
``apply_delta_colocate`` call (local self-copy + dtype-grouped cross-rank P2P via
``execute_recursive_partition_stream_transfer``). dte's contract splits this into
``send`` / ``recv``. AwexTransport therefore exposes a combined
``exchange`` and maps the split API onto it (``send`` stages, ``recv`` drives the
one symmetric call), rather than forcing awex's symmetric P2P into two halves.
"""

from __future__ import annotations

from typing import Any

from dte.transport import Payload, Plan, TransferOp, Transport

# awex is an optional dependency (extras = ["awex"]). Import lazily so importing
# this module never hard-fails; the constructor is what requires awex present.
try:  # pragma: no cover - availability depends on the host environment
    import awex  # noqa: F401

    _AWEX_AVAILABLE = True
except Exception:  # pragma: no cover
    _AWEX_AVAILABLE = False


_NEEDS_CLUSTER = (
    "AwexTransport requires a live awex runtime (NCCL process group, MetaServer, "
    "megatron/sglang reader state) — a 2-GPU cluster. This path is verified on "
    "the cluster, see docs/M3-gpu-verification.md. It cannot run on CPU."
)


def awex_available() -> bool:
    """Whether the awex package can be imported in this environment."""
    return _AWEX_AVAILABLE


def install_dte_delta_apply(colocate_transport) -> None:
    """Route an awex StreamBatch transport's two-round delta protocol through dte.

    Instance-level method swap (awex *source* unchanged, not monkey-patched at
    class/module level): replaces ``colocate_transport.apply_delta_colocate``
    with a shim that forwards to ``dte.core.colocate_protocol`` while injecting
    awex's own primitives (``execute_recursive_partition_stream_transfer`` for
    the deadlock-safe P2P schedule, ``slice_tensor`` / ``execute_tensors_to_copy``
    for reshard geometry). The native reader keeps owning IPC-get, reconstruct,
    version-chain base, and the writer-coordination driver protocol — only the
    two-round exchange now lives in dte. Idempotent.

    The shim mirrors awex ``apply_delta_colocate``'s positional signature
    (nccl_stream_batch.py:943) so the native ``_update_weights_in_colocate_mode``
    call site (line 596) needs no change.
    """
    if getattr(colocate_transport, "_dte_delta_installed", False):
        return
    from awex.transfer.nccl_comm import execute_tensors_to_copy
    from awex.transfer.transfer_plan import slice_tensor

    from dte.core import colocate_protocol

    def _shim(
        train_to_infer_device_mapping,
        infer_to_train_device_mapping,
        transfer_rank,
        rank_coordinate,
        world_size,
        send_transfer_plan,
        recv_transfer_plan,
        weights_update_group,
        send_full_params,
        masks,
        recv_parameters,
        value_dtype,
        *,
        step_id=-1,
    ):
        device = next(iter(recv_parameters.values())).device
        return colocate_protocol.apply_delta_colocate(
            transfer_rank=transfer_rank,
            world_size=world_size,
            send_plan=send_transfer_plan,
            recv_plan=recv_transfer_plan,
            train_to_infer_device_mapping=train_to_infer_device_mapping,
            infer_to_train_device_mapping=infer_to_train_device_mapping,
            weights_update_group=weights_update_group,
            full_params=send_full_params,
            masks=masks,
            recv_params=recv_parameters,
            device=device,
            schedule_fn=colocate_transport.execute_recursive_partition_stream_transfer,
            slice_fn=slice_tensor,
            selfcopy_fn=execute_tensors_to_copy,
            rank_coordinate=rank_coordinate,
            step_id=step_id,
        )

    colocate_transport.apply_delta_colocate = _shim
    colocate_transport._dte_delta_installed = True


class AwexTransport(Transport):
    """dte ``Transport`` backed by awex's reshard plan + NCCL P2P schedule.

    Composition, not inheritance: holds an awex ``NCCLWorkerWeightsReader`` (or a
    reader-like object exposing the same state) and calls its public surface. We
    never subclass or patch awex.

    Args:
        reader: an awex ``NCCLWorkerWeightsReader`` already ``initialize()``-d
            (it owns the process group, device mappings, transfer plans,
            ``colocate_transport`` and MetaServer client). On a cluster this is
            the object AReaL/awex already constructs; in tests it can be a stub
            exposing the same attributes.
        train_world_size / infer_world_size / num_infer_engines: forwarded to
            awex ``TransferPlanBuilder`` when (re)building plans here.
    """

    def __init__(
        self,
        reader: Any,
        *,
        train_world_size: int | None = None,
        infer_world_size: int | None = None,
        num_infer_engines: int = 1,
    ):
        if not _AWEX_AVAILABLE:
            raise RuntimeError(
                "AwexTransport needs the 'awex' package. "
                "Install with: pip install delta-transfer-engine[awex]"
            )
        self.reader = reader
        self.train_world_size = train_world_size
        self.infer_world_size = infer_world_size
        self.num_infer_engines = num_infer_engines
        self._staged: list[Payload] = []

    # ------------------------------------------------------------------ plan
    def build_plan(self, train_meta: Any, infer_meta: Any) -> Plan:
        """Wrap awex's local transfer plan as a dte ``Plan``.

        Reuses awex ``TransferPlanBuilder.build_local_transfer_plan`` (the
        reshard geometry: train_slices/inf_slices/recv_rank). We do not recompute
        geometry — awex owns that. The awex plan is stashed in
        ``Plan.backend_plan``; each awex ``CommunicationOperation`` is surfaced as
        a dte ``TransferOp`` (with ``backend_op`` carrying the native op) so dte
        code can iterate without importing awex types.
        """
        if self.train_world_size is None or self.infer_world_size is None:
            # Prefer plans the reader already built during initialize()
            # (no awex import needed on this path — useful for contract tests).
            awex_plan = getattr(self.reader, "transfer_plan", None)
            if awex_plan is None:
                raise RuntimeError(
                    "build_plan needs train/infer world sizes or a reader with a "
                    "prebuilt transfer_plan."
                )
        else:
            from awex.transfer.transfer_plan import TransferPlanBuilder

            builder = TransferPlanBuilder(
                infer_world_size=self.infer_world_size,
                train_world_size=self.train_world_size,
                num_infer_engines=self.num_infer_engines,
            )
            awex_plan = builder.build_local_transfer_plan(
                infer_meta, train_meta, self.reader.transfer_rank
            )

        ops: list[TransferOp] = []
        for peer_rank, co_list in awex_plan.operations.items():
            for co in co_list:
                ops.append(
                    TransferOp(
                        send_rank=co.send_rank,
                        recv_rank=co.recv_rank,
                        param_name=co.send_shard_meta.name,
                        train_slices=tuple(co.train_slices),
                        inf_slices=tuple(co.inf_slices),
                        backend_op=co,
                    )
                )
        return Plan(ops=ops, backend_plan=awex_plan)

    # -------------------------------------------------------------- transfer
    def send(self, plan: Plan, payloads: list[Payload]) -> None:
        """Stage payloads for the symmetric exchange.

        awex colocate transfer is symmetric (one ``apply_delta_colocate`` both
        sends and receives), so ``send`` only stages; ``recv`` drives the actual
        collective. See the impedance note in the module docstring.
        """
        self._staged = payloads

    def recv(self, plan: Plan) -> list[Payload]:
        """Not used for colocate (symmetric reshard goes through
        ``colocate_apply``). Kept for the Transport contract; the put/get
        backends (mooncake) are where ``recv`` is the natural primitive.
        """
        raise NotImplementedError(_NEEDS_CLUSTER)

    # ------------------------------------------------------- colocate reshard
    def colocate_apply(
        self,
        plan: Plan,
        full_params: dict,
        masks: dict | None,
        recv_params: dict,
        *,
        step_id: int = -1,
    ) -> int:
        """Drive awex's cross-rank two-round delta exchange + local self-copy.

        The two-round *protocol* now lives in dte
        (``dte.core.colocate_protocol``); awex supplies only the deadlock-safe
        P2P schedule (``execute_recursive_partition_stream_transfer``) and the
        reshard geometry (``slice_tensor`` / ``execute_tensors_to_copy``), all
        injected as callbacks. The reader (``self.reader``) owns the runtime
        state (rank, device mappings, transfer plans, process group, live
        inference params).

        ``masks is None`` ⇒ dense/full step: nothing sparse to scatter here (the
        full weights are written by the dense colocate path, not this hook).
        """
        from awex.transfer.nccl_comm import execute_tensors_to_copy
        from awex.transfer.transfer_plan import slice_tensor

        from dte.core import colocate_protocol

        if masks is None:
            return 0  # dense step handled by the dense colocate path

        r = self.reader
        device = next(iter(recv_params.values())).device
        schedule_fn = r.colocate_transport.execute_recursive_partition_stream_transfer
        return colocate_protocol.apply_delta_colocate(
            transfer_rank=r.transfer_rank,
            world_size=r.infer_world_size,
            send_plan=r.send_transfer_plan,
            recv_plan=r.transfer_plan,
            train_to_infer_device_mapping=r.train_to_infer_device_mapping,
            infer_to_train_device_mapping=r.infer_to_train_device_mapping,
            weights_update_group=r.weights_update_group,
            full_params=full_params,
            masks=masks,
            recv_params=recv_params,
            device=device,
            schedule_fn=schedule_fn,
            slice_fn=slice_tensor,
            selfcopy_fn=execute_tensors_to_copy,
            rank_coordinate=getattr(r, "rank_coordinate", ""),
            step_id=step_id,
        )
