# Licensed under the Apache License, Version 2.0
"""DeltaEngine — the orchestration layer that owns the control flow.

The engine drives the incremental algorithm (``dte.core``) and *calls* a
``Transport`` to move bytes. It is never called by a transport — that inversion
is what makes dte the upper layer (same relationship as TRL→vLLM, vLLM sparse
transfer→NCCL).

Wire protocol: the engine speaks ``dte.core``'s flat named-tensor payload. A
delta payload carries a header tensor plus ``w@delta_idx`` / ``w@delta_val``
pairs (sparse) and plain ``w`` entries (dense fallback); a full payload is just
the plain tensors. The receiver reconstructs via ``decode_delta_payload`` +
``reconstruct_against_base``. This is byte-identical to the verified awex path,
so swapping the transport cannot change the result.

Two roles (separate processes in production; co-located in CPU tests):

- sender   : ``push(named_params, version)`` — decides full vs delta, encodes,
             calls ``transport.send``.
- receiver : ``pull(target, version)`` — ``transport.recv``, decodes, applies.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

import torch

from dte.core import (
    DELTA_HEADER_NAME,
    DELTA_IDX_SUFFIX,
    DELTA_VAL_SUFFIX,
    DecodedDelta,
    DeltaHeader,
    DeltaTracker,
    apply_sparse_patch_,
    decode_delta_payload,
    is_delta_payload,
    reconstruct_against_base,
)
from dte.transport import Payload, Plan, Transport

logger = logging.getLogger(__name__)

__all__ = ["DeltaEngine", "PushResult", "DeltaChainBroken"]


class DeltaChainBroken(RuntimeError):
    """Raised by the receiver when a delta cannot be applied to the current base.

    Either there is no full-sync base yet, or the delta's ``base_version`` does
    not match the receiver's base (stale / reordered / dropped payload). The
    caller should fall back to requesting a fresh full sync. Subclasses
    ``RuntimeError`` so existing ``except RuntimeError`` paths still catch it.
    """


@dataclass
class PushResult:
    """Outcome of a sender-side ``push``."""

    version: int
    full_sync: bool
    reason: str | None
    num_payloads: int


class DeltaEngine:
    """Orchestrates detect → encode → transport.send / recv → decode → apply.

    A single instance can act as sender (holds a ``DeltaTracker``) and/or
    receiver (holds the CPU base). In production the two roles live in different
    processes; in CPU tests one engine pair shares a ``LoopbackTransport``.
    """

    def __init__(
        self,
        transport: Transport | None = None,
        *,
        mode: str = "delta",
        anchor_interval: int = 0,
        sparse_bytes_ratio: float = 0.9,
        device: str | torch.device = "cpu",
    ):
        """
        Args:
            transport: the byte-mover. Optional for a receiver-only instance that
                only calls ``reconstruct`` (IPC-fed; no transport.recv/send) —
                e.g. the awex colocate reader, where awex owns the IPC-get and the
                cross-rank scatter. ``push``/``pull`` require a transport.
            mode: ``"delta"`` (default) — incremental, with full syncs at seed /
                anchor / chain break (``anchor_interval`` controls re-anchoring).
                ``"full"`` — every push is a full sync; the detector/snapshot is
                never touched (equivalent to awex ``AWEX_DELTA_TRANSFER=0``).
            anchor_interval: in delta mode, force a full sync every N deltas
                (0 = never; rely on seed + chain-break only). Ignored in full mode.
            sparse_bytes_ratio: per-tensor delta-vs-dense fallback threshold.
        """
        if mode not in ("delta", "full"):
            raise ValueError(f"mode must be 'delta' or 'full', got {mode!r}")
        self.transport = transport
        self.mode = mode
        self.tracker = DeltaTracker(anchor_interval, sparse_bytes_ratio)
        self.device = torch.device(device)
        # receiver-side base: {name: tensor}, refreshed in place by reconstruct.
        self._base: dict[str, torch.Tensor] = {}
        self._base_version: int | None = None

    # ------------------------------------------------------------------ sender
    def push(
        self,
        named_params: Iterable[tuple[str, torch.Tensor]],
        version: int,
        plan: Plan | None = None,
    ) -> PushResult:
        """Encode the current weights for ``version`` and send them.

        ``mode="full"`` ships a full payload every time (no detection).
        ``mode="delta"`` ships a full sync when the tracker says so (not seeded /
        anchor / requested) and a delta otherwise. ``plan`` is forwarded to the
        transport (None is fine for backends that don't reshard, e.g. loopback).
        """
        params = list(named_params)
        plan = plan if plan is not None else Plan()

        if self.mode == "full":
            reason = "full_mode"
        else:
            reason = self.tracker.full_sync_reason(version)

        if reason is not None:
            # Full sync: ship plain tensors. In delta mode (re)seed the snapshot
            # baseline; in full mode the detector/snapshot is never used.
            payloads = [Payload(name, t.detach()) for name, t in params]
            if self.mode == "delta":
                self.tracker.seed(params, version)
            self.transport.send(plan, payloads)
            return PushResult(version, True, reason, len(payloads))

        # Delta: encode against snapshot -> flat (names, tensors) payload.
        encoded = self.tracker.encode(params, version)
        payloads = [Payload(name, t) for name, t in zip(encoded.names, encoded.tensors)]
        self.transport.send(plan, payloads)
        return PushResult(version, False, None, len(payloads))

    # ---------------------------------------------------------------- receiver
    def pull(
        self,
        target_params: dict[str, torch.Tensor],
        version: int,
        plan: Plan | None = None,
    ) -> dict[str, torch.Tensor]:
        """Receive a payload, reconstruct full weights, write into ``target``.

        Returns the reconstructed ``{name: tensor}`` (also written in place into
        ``target_params`` entries that exist).

        Version trust model (important — full and delta are asymmetric):

        - **delta** payloads are self-describing: the header carries
          ``base_version``/``payload_version``, so the receiver independently
          verifies the chain and raises ``RuntimeError`` on any mismatch (the
          primary safety net against desync).
        - **full-sync** payloads ship plain tensors with NO header (kept this way
          for bitwise parity with the awex wire format), so the receiver cannot
          self-verify and trusts the caller-supplied ``version``. The caller MUST
          pass the version the sender used. A wrong ``version`` here is caught one
          step later by the next delta's chain check, but with a misleading
          message — so two guards below catch the obvious cases eagerly:
          (1) a full sync must not move the version backwards past an established
          base (stale/reordered payload), and (2) it adopts ``version`` as base.

        Raises on a broken version chain (delta whose ``base_version`` != receiver
        base, or a backwards full sync) — caller should request a fresh full sync.
        """
        plan = plan if plan is not None else Plan()
        payloads = self.transport.recv(plan)
        named = {p.name: p.values for p in payloads}

        if is_delta_payload(named):
            if self._base_version is None:
                raise RuntimeError(
                    "Received a delta before any full-sync base; broken chain."
                )
            decoded = decode_delta_payload(named)
            if decoded.header.base_version != self._base_version:
                raise RuntimeError(
                    f"Delta base mismatch: payload base="
                    f"{decoded.header.base_version}, receiver base="
                    f"{self._base_version}; request full sync."
                )
            result, _counts = reconstruct_against_base(self._base, decoded, self.device)
            self._base_version = decoded.header.payload_version
        else:
            # Full sync: trusts the caller's version (payload is header-less).
            # Guard the one case we *can* detect without a header: a full sync
            # must not move the receiver's version backwards (stale/reordered).
            if self._base_version is not None and version < self._base_version:
                raise RuntimeError(
                    f"Full-sync version {version} is older than the receiver's "
                    f"current base {self._base_version} (stale/reordered payload)."
                )
            result = {name: t for name, t in named.items()}
            self._base = {
                name: t.detach().to("cpu").clone() for name, t in named.items()
            }
            self._base_version = version

        for name, tensor in result.items():
            if name in target_params:
                target_params[name].copy_(tensor.to(target_params[name].device))
        return result

    # ------------------------------------------------ receiver (IPC-fed)
    @torch.no_grad()
    def reconstruct(
        self,
        named_tensors: dict[str, torch.Tensor],
        version: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None]:
        """Decode + version-chain check + reconstruct, for an IPC-fed payload.

        Same decode/version-chain/reconstruct as ``pull`` but WITHOUT a
        ``transport.recv`` (the bytes already arrived via the backend's own
        IPC-get) and WITHOUT writing a target (the backend's cross-rank scatter
        does that). Used by the awex colocate reader: awex owns IPC-get +
        cross-rank apply + writer-coordination; dte owns decode + version-chain
        base + the per-param change masks.

        Returns ``(full_params, masks)``:
          - **delta** payload: ``full_params`` is the reconstructed full
            train-shard; ``masks`` is the per-param bool change mask, built
            byte-identically to the awex reader's old logic — ``dense -> ones``,
            ``sparse -> scattered True``, and **unchanged params get NO entry**
            (mask absent == nothing changed, which the cross-rank builder reads
            as a zero-nnz op).
          - **full-sync** (dense) payload: seeds/refreshes the base and returns
            ``(named_tensors, None)`` — ``None`` masks signals the caller to take
            the dense apply path (mirrors awex ``_delta_masks = None``).

        Raises ``DeltaChainBroken`` on a delta with no base / mismatched base so
        the caller can request a fresh full sync. ``ValueError`` on a corrupt
        delta (e.g. a sparse patch for a name absent from the base).
        """
        if not is_delta_payload(named_tensors):
            # Dense full sync: (re)seed the base; dense apply path downstream.
            self._base = {
                name: t.detach().to("cpu").clone() for name, t in named_tensors.items()
            }
            self._base_version = version
            return named_tensors, None

        if self._base_version is None or not self._base:
            raise DeltaChainBroken(
                f"Delta at version {version} but receiver has no base "
                f"(base_version={self._base_version}); request full sync."
            )
        decoded = decode_delta_payload(named_tensors)
        if decoded.header.base_version != self._base_version:
            raise DeltaChainBroken(
                f"Delta base mismatch: payload base="
                f"{decoded.header.base_version}, receiver base="
                f"{self._base_version}; request full sync."
            )
        # Reconstruct onto the device the payload already lives on (matches the
        # awex reader's old behavior of using the deserialized tensors' device,
        # which under CUDA_VISIBLE_DEVICES isolation is the *visible* current
        # device, not necessarily ``self.device``'s physical index).
        device = next(iter(named_tensors.values())).device
        result, counts = reconstruct_against_base(self._base, decoded, device)
        base_v = decoded.header.base_version
        self._base_version = decoded.header.payload_version
        logger.info(
            "dte reconstructed step %d (base=%d) sparse=%d dense=%d unchanged=%d",
            version,
            base_v,
            counts["sparse"],
            counts["dense"],
            counts["unchanged"],
        )

        # Per-param change masks (byte-identical to the awex reader's old build):
        # dense -> all True; sparse -> scattered True; unchanged -> NO entry.
        masks: dict[str, torch.Tensor] = {}
        for name, full in result.items():
            if name in decoded.dense:
                masks[name] = torch.ones(
                    full.shape, dtype=torch.bool, device=full.device
                )
            elif name in decoded.sparse:
                indices, _ = decoded.sparse[name]
                m = torch.zeros(full.numel(), dtype=torch.bool, device=full.device)
                if indices.numel() > 0:
                    m[indices.to(full.device).long()] = True
                masks[name] = m.view(full.shape)
        return result, masks

    def reconstruct_stream(
        self,
        chunks: Iterable[dict[str, torch.Tensor] | Iterable[Payload]],
        version: int,
    ):
        """Chunk-at-a-time ``reconstruct`` for staged payloads (http tensor mode).

        Consumes ``HttpTransport.iter_fetch``-shaped chunks (each a
        ``{name: tensor}`` dict or a ``Payload`` list) and yields
        ``{name: full_tensor}`` per chunk, covering ONLY the params present in
        that chunk — unchanged params are never materialized or yielded, and no
        masks are built, so receiver peak memory is base + ~one chunk instead of
        base + full result + masks.

        Chunking contract (guaranteed by ``HttpTransport``): a sparse pair
        (``w@delta_idx``/``w@delta_val``) never splits across chunks, and each
        writer's header rides in its first chunk. Multiple writers may each
        contribute a header; their sparse/dense counts are summed and verified
        against the stream total on completion.

        TODO(agent): Keep each sparse index/value pair in one chunk unless this
        method gains cross-chunk pair buffering.

        Version-chain safety mirrors ``reconstruct`` with one stream-specific
        rule: the chain check runs eagerly (before anything is yielded), and the
        receiver's ``base_version`` is invalidated while chunks are being
        applied — it only advances to the payload version after the stream
        completes and the counts match. A mid-stream failure therefore leaves
        the receiver chain-broken (next delta raises ``DeltaChainBroken`` and
        the caller requests a fresh full sync).

        Raises ``DeltaChainBroken`` on no-base / base-mismatch / stale full
        sync; ``ValueError`` on corrupt or incomplete streams (unpaired sparse
        entries, sparse param absent from base, count mismatch, empty stream).
        """
        it = iter(chunks)
        try:
            first = self._chunk_dict(next(it))
        except StopIteration:
            raise ValueError("reconstruct_stream got an empty chunk stream")

        if not is_delta_payload(first):
            if self._base_version is not None and version < self._base_version:
                raise DeltaChainBroken(
                    f"Full-sync version {version} is older than the receiver's "
                    f"current base {self._base_version} (stale/reordered payload)."
                )
            return self._stream_full_sync(first, it, version)

        header = DeltaHeader.from_tensor(first[DELTA_HEADER_NAME].cpu())
        if header.payload_version != version:
            raise DeltaChainBroken(
                "Stream payload version does not match requested version"
            )
        if self._base_version is None or not self._base:
            raise DeltaChainBroken(
                f"Delta at version {version} but receiver has no base "
                f"(base_version={self._base_version}); request full sync."
            )
        if header.base_version != self._base_version:
            raise DeltaChainBroken(
                f"Delta base mismatch: payload base={header.base_version}, "
                f"receiver base={self._base_version}; request full sync."
            )
        return self._stream_delta(header, first, it, version)

    @staticmethod
    def _chunk_dict(chunk) -> dict[str, torch.Tensor]:
        if isinstance(chunk, dict):
            return chunk
        return {p.name: p.values for p in chunk}

    def _stream_full_sync(self, first, rest, version: int):
        self._base = {}
        self._base_version = None
        for chunk in self._iter_chunks(first, rest):
            if is_delta_payload(chunk) or self._base.keys() & chunk.keys():
                raise ValueError(
                    "Full stream requires unique parameter names and no delta headers"
                )
            for name, tensor in chunk.items():
                self._base[name] = tensor.detach().to("cpu").clone()
            yield chunk
        self._base_version = version

    @torch.no_grad()
    def _stream_delta(self, header, first, rest, version: int):
        expected = {"sparse": 0, "dense": 0}
        counts = {"sparse": 0, "dense": 0}
        base_version = self._base_version
        self._base_version = None
        seen = set()
        for chunk in self._iter_chunks(first, rest):
            result = self._apply_delta_chunk(chunk, header, expected, counts)
            if seen & result.keys():
                raise ValueError(
                    "Delta stream requires unique parameter names across writers"
                )
            seen.update(result)
            yield result
        if counts != expected:
            raise ValueError(
                f"Delta stream count mismatch: headers promised {expected}, "
                f"stream carried {counts}; incomplete or corrupt chunking."
            )
        self._base_version = header.payload_version
        logger.info(
            "dte stream-reconstructed step %d (base=%s) sparse=%d dense=%d",
            version,
            base_version,
            counts["sparse"],
            counts["dense"],
        )

    def _iter_chunks(self, first, rest):
        yield first
        for chunk in rest:
            yield self._chunk_dict(chunk)

    def _apply_delta_chunk(
        self,
        chunk: dict[str, torch.Tensor],
        header,
        expected: dict[str, int],
        counts: dict[str, int],
    ) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        pending_idx: dict[str, torch.Tensor] = {}
        pending_val: dict[str, torch.Tensor] = {}
        for name, tensor in chunk.items():
            if name == DELTA_HEADER_NAME:
                extra = DeltaHeader.from_tensor(tensor.cpu())
                if (extra.base_version, extra.payload_version) != (
                    header.base_version,
                    header.payload_version,
                ):
                    raise ValueError(
                        f"Delta stream header disagreement: expected "
                        f"base={header.base_version}->v{header.payload_version}, "
                        f"got base={extra.base_version}->v{extra.payload_version}."
                    )
                # One header per writer; summing them yields the stream total.
                expected["sparse"] += extra.num_sparse
                expected["dense"] += extra.num_dense
                continue
            if name.endswith(DELTA_IDX_SUFFIX):
                pending_idx[name[: -len(DELTA_IDX_SUFFIX)]] = tensor
            elif name.endswith(DELTA_VAL_SUFFIX):
                pending_val[name[: -len(DELTA_VAL_SUFFIX)]] = tensor
            else:
                full = tensor
                base_cpu = self._base.get(name)
                if base_cpu is not None:
                    base_cpu.copy_(full.detach().to("cpu"))
                else:
                    self._base[name] = full.detach().to("cpu").clone()
                out[name] = full
                counts["dense"] += 1

        if set(pending_idx) != set(pending_val):
            missing = set(pending_idx).symmetric_difference(pending_val)
            raise ValueError(
                f"Corrupt delta chunk, unpaired sparse entries: {missing} "
                "(idx/val pairs must never split across chunks)."
            )
        for name, indices in pending_idx.items():
            values = pending_val[name]
            base_cpu = self._base.get(name)
            if base_cpu is None:
                raise ValueError(
                    f"Delta sparse param absent from base, cannot reconstruct "
                    f"(unknown full shape): ['{name}']"
                )
            full = base_cpu.to(values.device, copy=True)
            apply_sparse_patch_(full, indices, values)
            base_cpu.copy_(full.detach().to("cpu"))
            out[name] = full
            counts["sparse"] += 1
        return out

    @torch.no_grad()
    def decode_for_live_apply(
        self,
        named_tensors: dict[str, torch.Tensor],
        version: int,
    ):
        """Decode a payload for direct live-weight apply without a CPU base.

        Full-sync payloads carry no header, so they only advance the receiver's
        base version and return ``None``; the caller should use its dense apply
        path with the original ``named_tensors``. Delta payloads are decoded and
        version-chain checked, then returned as ``DecodedDelta`` for a sparse
        live apply path. No full parameter copy is materialized or stored.
        """
        if not is_delta_payload(named_tensors):
            if self._base_version is not None and version < self._base_version:
                raise DeltaChainBroken(
                    f"Full-sync version {version} is older than receiver base "
                    f"{self._base_version}."
                )
            self._base_version = version
            return None

        decoded = decode_delta_payload(named_tensors)
        if self._base_version is None:
            raise DeltaChainBroken(
                f"Delta at version {version} but receiver has no base; "
                "request full sync."
            )
        if decoded.header.base_version != self._base_version:
            raise DeltaChainBroken(
                f"Delta base mismatch: payload base="
                f"{decoded.header.base_version}, receiver base="
                f"{self._base_version}; request full sync."
            )
        logger.info(
            "dte decoded live delta step %d (base=%d) sparse=%d dense=%d",
            version,
            decoded.header.base_version,
            len(decoded.sparse),
            len(decoded.dense),
        )
        return decoded

    def commit_live_apply(self, decoded: DecodedDelta) -> None:
        """Advance the live-apply version after the caller applied a delta.

        ``decode_for_live_apply`` intentionally does not advance the receiver's
        base version: the sparse apply path can still fail after decoding. If we
        moved the version first, a partial/failed apply could make the next
        delta pass the version-chain check against weights that are not actually
        at the expected base.
        """
        if self._base_version is None:
            raise DeltaChainBroken(
                "Cannot commit live delta because receiver has no base."
            )
        if decoded.header.base_version != self._base_version:
            raise DeltaChainBroken(
                f"Delta commit base mismatch: payload base="
                f"{decoded.header.base_version}, receiver base="
                f"{self._base_version}; request full sync."
            )
        self._base_version = decoded.header.payload_version

    @property
    def base_version(self) -> int | None:
        return self._base_version
