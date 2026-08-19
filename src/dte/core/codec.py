# Licensed under the Apache License, Version 2.0
"""Delta payload codec for colocate weight transfer.

Encodes per-parameter sparse updates into a flat ``(names, tensors)`` list that
rides the existing transfer pipeline unchanged::

    group_tensors_by_shape_and_dtype -> share_memory_ -> cuda_ipc_serialize
        -> MetaServer -> cuda_ipc_deserialize -> reconstruct_tensors_from_groups
        -> dict(zip(names, tensors))

Encoding scheme (writer side, ``DeltaTracker.encode``):

- header:    one int64 tensor under the reserved name ``__awex_delta_header__``
             carrying ``[magic, codec_version, payload_version, base_version,
             num_sparse, num_dense]`` for version-chain validation.
- sparse:    parameter ``w`` becomes two 1-D tensors ``w@delta_idx`` (int32 flat
             indices) and ``w@delta_val`` (new values, param dtype).
- dense:     parameters whose change density exceeds the break-even threshold
             fall back to a full tensor under the plain name ``w`` (identical to
             the non-delta payload entry).
- unchanged: parameters with zero changed elements are omitted entirely; the
             reader keeps its restored base values.

Change detection is *bitwise* (integer view comparison), so NaN payload bits
and +0.0/-0.0 are handled exactly: the transfer reproduces the training-side
bf16 bit pattern losslessly.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

import torch

logger = logging.getLogger(__name__)

DELTA_HEADER_NAME = "__awex_delta_header__"
DELTA_IDX_SUFFIX = "@delta_idx"
DELTA_VAL_SUFFIX = "@delta_val"

DELTA_MAGIC = 0x41574558  # "AWEX"
CODEC_VERSION = 1

# int32 flat indices: single param must stay below 2**31 elements.
_MAX_INT32_NUMEL = 2**31


def _check_reserved_names(names: Iterable[str]) -> None:
    """Guard the flat-protocol namespace.

    The wire protocol distinguishes sparse entries purely by name suffix
    (``@delta_idx`` / ``@delta_val``) and the header by ``__awex_delta_header__``.
    A real parameter named with one of these would corrupt decoding, so callers
    must not use them. Real HF parameter names never contain ``@``; this is a
    defensive assertion, not an expected condition.
    """
    for name in names:
        if (
            name == DELTA_HEADER_NAME
            or name.endswith(DELTA_IDX_SUFFIX)
            or name.endswith(DELTA_VAL_SUFFIX)
        ):
            raise ValueError(
                f"Parameter name {name!r} collides with a reserved delta-protocol "
                f"name ({DELTA_HEADER_NAME!r}, *{DELTA_IDX_SUFFIX!r}, "
                f"*{DELTA_VAL_SUFFIX!r}). Rename the parameter."
            )


_INT_VIEW_DTYPE = {
    1: torch.int8,
    2: torch.int16,
    4: torch.int32,
    8: torch.int64,
}
_BIT_OFFSET_LOOKUP: dict[str, torch.Tensor] = {}


def _bit_offset_lookup(device: torch.device) -> torch.Tensor:
    """Return byte -> sorted set-bit offsets table for ``device``."""
    key = str(device)
    table = _BIT_OFFSET_LOOKUP.get(key)
    if table is None:
        rows = []
        for value in range(256):
            offsets = [bit for bit in range(8) if value & (1 << bit)]
            offsets.extend([8] * (8 - len(offsets)))
            rows.append(offsets)
        table = torch.tensor(rows, dtype=torch.uint8, device=device)
        _BIT_OFFSET_LOOKUP[key] = table
    return table


def int_view(tensor: torch.Tensor) -> torch.Tensor:
    """Reinterpret a tensor's storage as a same-width integer tensor.

    Used for bitwise (not floating-point) comparison: NaN != NaN under float
    semantics even when the bit patterns are identical, while -0.0 == +0.0
    even though the bits differ. Bitwise comparison gives exact change
    detection in both cases.
    """
    if not tensor.dtype.is_floating_point:
        return tensor
    itype = _INT_VIEW_DTYPE.get(tensor.element_size())
    if itype is None:
        raise TypeError(f"Unsupported element size for bitwise view: {tensor.dtype}")
    return tensor.contiguous().view(itype)


def bitwise_changed_mask(current: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
    """Element-wise bool mask of bit-pattern differences between two tensors."""
    if current.dtype != baseline.dtype or current.shape != baseline.shape:
        raise ValueError(
            f"Mismatched tensors for bitwise compare: "
            f"{current.dtype}/{tuple(current.shape)} vs "
            f"{baseline.dtype}/{tuple(baseline.shape)}"
        )
    return int_view(current) != int_view(baseline)


def payload_changed_mask_from_pre_post(
    before: torch.Tensor,
    after: torch.Tensor,
    *,
    payload_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Reference dirty-mask semantics for optimizer-time change detection.

    Delta payloads are keyed by the dtype consumed by the inference engine.  For
    Qwen3 colocate delta this is bf16, so the exact sparse mask is not
    ``before != after`` in fp32.  It is the bitwise difference after both values
    are rounded to the payload dtype:

        bitpattern(after.to(payload_dtype)) != bitpattern(before.to(payload_dtype))

    Optimizer-fused detectors should match this helper before remapping masks to
    HF payload space.
    """
    if before.shape != after.shape:
        raise ValueError(
            f"Mismatched tensors for payload compare: "
            f"{tuple(before.shape)} vs {tuple(after.shape)}"
        )
    before_payload = before.to(payload_dtype)
    after_payload = after.to(payload_dtype)
    return bitwise_changed_mask(after_payload, before_payload)


def pack_bool_mask_to_uint8(mask: torch.Tensor) -> torch.Tensor:
    """Pack a bool dirty mask into little-endian uint8 bitset bytes.

    Optimizer-time detectors should not keep one byte per element for dirty
    state at 30B scale. This helper defines the compact bitset convention used
    by B1/B2 prototypes: element ``i`` is stored at ``packed[i // 8]`` bit
    ``i % 8``. The caller stores the original element count separately.
    """
    flat = mask.reshape(-1).to(torch.bool)
    numel = flat.numel()
    if numel == 0:
        return torch.empty(0, dtype=torch.uint8, device=mask.device)

    padded_numel = ((numel + 7) // 8) * 8
    if padded_numel != numel:
        pad = torch.zeros(padded_numel - numel, dtype=torch.bool, device=flat.device)
        flat = torch.cat((flat, pad), dim=0)

    bits = flat.view(-1, 8).to(torch.uint8)
    return (
        bits[:, 0]
        | (bits[:, 1] << 1)
        | (bits[:, 2] << 2)
        | (bits[:, 3] << 3)
        | (bits[:, 4] << 4)
        | (bits[:, 5] << 5)
        | (bits[:, 6] << 6)
        | (bits[:, 7] << 7)
    )


def unpack_bool_mask_from_uint8(packed: torch.Tensor, numel: int) -> torch.Tensor:
    """Unpack a little-endian uint8 bitset back to a bool dirty mask."""
    if numel < 0:
        raise ValueError(f"numel must be non-negative, got {numel}")
    if packed.dtype != torch.uint8:
        raise TypeError(f"packed mask must be torch.uint8, got {packed.dtype}")
    needed = (numel + 7) // 8
    flat = packed.reshape(-1)
    if flat.numel() < needed:
        raise ValueError(
            f"Packed mask is too short for {numel} bits: "
            f"need {needed} bytes, got {flat.numel()}"
        )
    if numel == 0:
        return torch.empty(0, dtype=torch.bool, device=packed.device)

    bytes_i16 = flat[:needed].to(torch.int16).view(-1, 1)
    shifts = torch.arange(8, dtype=torch.int16, device=packed.device).view(1, 8)
    bits = ((bytes_i16 >> shifts) & 1).to(torch.bool).reshape(-1)
    return bits[:numel].contiguous()


def payload_changed_bitset_from_pre_post(
    before: torch.Tensor,
    after: torch.Tensor,
    *,
    payload_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, int]:
    """Return compact dirty bitset using payload dtype bit-pattern semantics."""
    mask = payload_changed_mask_from_pre_post(
        before,
        after,
        payload_dtype=payload_dtype,
    )
    return pack_bool_mask_to_uint8(mask), mask.numel()


def packed_bool_mask_to_indices(
    packed: torch.Tensor,
    numel: int,
    *,
    dtype: torch.dtype = torch.int64,
    chunk_bytes: int = 1 << 20,
) -> torch.Tensor:
    """Return sorted set-bit indices from a packed dirty bitset.

    The common Qwen3 dirty-mask density is only a few percent. Avoid expanding
    every byte into eight bools; scan nonzero bytes and use a tiny lookup table
    to emit only their set-bit offsets. Padding bits beyond ``numel`` are
    ignored even if the caller passes nonzero garbage in the final packed byte.
    """
    if dtype not in {torch.int32, torch.int64}:
        raise TypeError(f"indices dtype must be int32 or int64, got {dtype}")
    if chunk_bytes <= 0:
        raise ValueError(f"chunk_bytes must be positive, got {chunk_bytes}")
    if numel < 0:
        raise ValueError(f"numel must be non-negative, got {numel}")
    if packed.dtype != torch.uint8:
        raise TypeError(f"packed mask must be torch.uint8, got {packed.dtype}")

    needed = (numel + 7) // 8
    flat = packed.reshape(-1)
    if flat.numel() < needed:
        raise ValueError(
            f"Packed mask is too short for {numel} bits: "
            f"need {needed} bytes, got {flat.numel()}"
        )
    if numel == 0:
        return torch.empty(0, dtype=dtype, device=packed.device)

    pieces: list[torch.Tensor] = []
    lookup: torch.Tensor | None = None
    shifts: torch.Tensor | None = None
    for byte_start in range(0, needed, chunk_bytes):
        byte_end = min(byte_start + chunk_bytes, needed)
        chunk = flat[byte_start:byte_end]
        nonzero_bytes = chunk.nonzero(as_tuple=False).squeeze(1)
        if nonzero_bytes.numel() == 0:
            continue
        if nonzero_bytes.numel() > chunk.numel() // 2:
            # Dense chunks are faster with the straightforward vectorized
            # unpack+nonzero path. The lookup path expands every nonzero byte
            # to eight candidate offsets, which is wasteful when most bytes
            # have at least one dirty element.
            if shifts is None:
                shifts = torch.arange(
                    8,
                    dtype=torch.int16,
                    device=packed.device,
                ).view(1, 8)
            bits = (
                ((chunk.to(torch.int16).view(-1, 1) >> shifts) & 1)
                .to(torch.bool)
                .reshape(-1)
            )
            base = byte_start * 8
            valid_bits = min(bits.numel(), numel - base)
            if valid_bits <= 0:
                break
            local = bits[:valid_bits].nonzero(as_tuple=False).squeeze(1)
            if local.numel() > 0:
                pieces.append(local + base)
            continue

        if lookup is None:
            lookup = _bit_offset_lookup(flat.device)
        byte_values = chunk.index_select(0, nonzero_bytes).to(torch.long)
        offsets = lookup.index_select(0, byte_values)
        valid = offsets < 8
        byte_indices = nonzero_bytes + byte_start
        candidates = byte_indices.view(-1, 1).to(torch.long) * 8 + offsets.to(
            torch.long
        )
        valid &= candidates < numel
        selected = candidates[valid]
        if selected.numel() > 0:
            pieces.append(selected)

    if not pieces:
        return torch.empty(0, dtype=dtype, device=packed.device)
    return torch.cat(pieces, dim=0).to(dtype)


@torch.no_grad()
def invert_adamw(
    theta_t: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: float,
    lr: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
    eps: float,
) -> torch.Tensor:
    """Reconstruct pre-step weights theta_{t-1} from one decoupled-AdamW step.

    Inverse of the torch ``AdamW`` update (decoupled weight decay), used by the
    AReaL AdamW-inversion change detector to recover the previous weights from
    the optimizer's resident moments without storing a snapshot:

        theta_t      = theta_{t-1}·(1 - lr·wd) - (lr/bc1)·m / (sqrt(v)/sqrt(bc2) + eps)
        theta_{t-1}  = (theta_t + (lr/bc1)·m / (sqrt(v)/sqrt(bc2) + eps)) / (1 - lr·wd)

    with ``m=exp_avg``, ``v=exp_avg_sq``, ``bc1 = 1 - beta1^step``,
    ``bc2 = 1 - beta2^step``. Computed in fp32; the result is fp32 regardless of
    the input dtype. ``step`` is the 1-based optimizer step count for these
    moments.
    """
    theta = theta_t.to(torch.float32)
    m = exp_avg.to(torch.float32)
    v = exp_avg_sq.to(torch.float32)
    bc1 = 1.0 - beta1**step
    bc2 = 1.0 - beta2**step
    denom = (v / bc2).sqrt().add_(eps)
    update = (lr / bc1) * m / denom
    return (theta + update) / (1.0 - lr * weight_decay)


def adamw_payload_changed_bitset_from_post_step(
    theta_t: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: float,
    lr: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
    eps: float,
    *,
    payload_dtype: torch.dtype = torch.bfloat16,
    update_successful: bool = True,
) -> tuple[torch.Tensor, int]:
    """CPU oracle for an optimizer-fused AdamW dirty-bit implementation.

    A fused kernel should mark exactly the elements whose payload-dtype bit
    pattern changed during the latest AdamW step.  This helper reconstructs
    ``theta_{t-1}`` from the post-step parameter and resident AdamW moments,
    then applies the same payload bit-pattern oracle used by snapshot diffing.
    """
    numel = theta_t.numel()
    if not update_successful:
        empty = torch.zeros(
            (numel + 7) // 8,
            dtype=torch.uint8,
            device=theta_t.device,
        )
        return empty, numel
    if step <= 0:
        raise ValueError(f"step must be positive after a successful update, got {step}")

    theta_prev = invert_adamw(
        theta_t,
        exp_avg,
        exp_avg_sq,
        step,
        lr,
        weight_decay,
        beta1,
        beta2,
        eps,
    )
    return payload_changed_bitset_from_pre_post(
        theta_prev,
        theta_t,
        payload_dtype=payload_dtype,
    )


@dataclass
class DeltaHeader:
    """Version-chain header carried inside the payload as an int64 tensor."""

    payload_version: int
    base_version: int
    num_sparse: int = 0
    num_dense: int = 0
    codec_version: int = CODEC_VERSION

    def to_tensor(self, device: torch.device | None = None) -> torch.Tensor:
        return torch.tensor(
            [
                DELTA_MAGIC,
                self.codec_version,
                self.payload_version,
                self.base_version,
                self.num_sparse,
                self.num_dense,
            ],
            dtype=torch.int64,
            device=device or "cpu",
        )

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> DeltaHeader:
        if tensor.numel() != 6 or tensor.dtype != torch.int64:
            raise ValueError(
                f"Invalid delta header tensor: dtype={tensor.dtype}, "
                f"numel={tensor.numel()}"
            )
        vals = tensor.flatten().tolist()
        if vals[0] != DELTA_MAGIC:
            raise ValueError(f"Bad delta header magic: {vals[0]:#x}")
        if vals[1] != CODEC_VERSION:
            raise ValueError(
                f"Unsupported delta codec version {vals[1]} (expected {CODEC_VERSION})"
            )
        return cls(
            payload_version=vals[2],
            base_version=vals[3],
            num_sparse=vals[4],
            num_dense=vals[5],
            codec_version=vals[1],
        )


@dataclass
class EncodedDelta:
    """Result of ``DeltaTracker.encode``: payload + transfer statistics.

    ``names``/``tensors`` (header entry included) feed directly into
    ``group_tensors_by_shape_and_dtype`` + ``cuda_ipc_serialize`` on the
    writer side, replacing the dense names/tensors lists.
    """

    names: list[str] = field(default_factory=list)
    tensors: list[torch.Tensor] = field(default_factory=list)
    header: DeltaHeader | None = None
    total_elements: int = 0
    changed_elements: int = 0
    num_sparse: int = 0
    num_dense_fallback: int = 0
    num_unchanged: int = 0
    payload_bytes: int = 0
    dense_bytes: int = 0

    @property
    def changed_ratio(self) -> float:
        if self.total_elements == 0:
            return 0.0
        return self.changed_elements / self.total_elements

    def summary(self) -> str:
        return (
            f"EncodedDelta(v{self.header.payload_version} base=v{self.header.base_version}, "
            f"changed={self.changed_elements}/{self.total_elements} "
            f"({self.changed_ratio:.2%}), "
            f"sparse={self.num_sparse} dense_fb={self.num_dense_fallback} "
            f"unchanged={self.num_unchanged}, "
            f"payload={self.payload_bytes / 1e6:.1f}MB "
            f"vs dense={self.dense_bytes / 1e6:.1f}MB "
            f"({self.payload_bytes / max(self.dense_bytes, 1):.2%})"
        )


class DeltaTracker:
    """Writer-side delta state: CPU baseline snapshot + version chain.

    Lifecycle (driven by the writer integration)::

        tracker = DeltaTracker(anchor_interval=N)
        for version in steps:
            if tracker.full_sync_reason(version):
                <existing dense transfer path>
                tracker.seed(named_params, version)
            else:
                encoded = tracker.encode(named_params, version)
                <group + serialize encoded.names / encoded.tensors>

    The snapshot lives in CPU pinned memory (one bf16 copy of the HF-converted
    weights). Parameters sharing storage (tied embeddings) are deduplicated:
    the snapshot holds a single CPU tensor and ``encode`` computes the delta
    once, emitting it under every aliased name.
    """

    def __init__(
        self,
        anchor_interval: int = 0,
        sparse_bytes_ratio: float = 0.9,
    ):
        """
        Args:
            anchor_interval: Force a full sync after this many consecutive
                delta payloads (0 = never force).
            sparse_bytes_ratio: Per-tensor fallback threshold. Use the sparse
                encoding only if its size is below ``ratio * dense_size``;
                bf16 break-even is at ~1/3 changed elements, the default 0.9
                keeps a safety margin against scatter overhead.
        """
        self._anchor_interval = anchor_interval
        self._sparse_bytes_ratio = sparse_bytes_ratio
        self._snapshot: dict[str, torch.Tensor] = {}
        # Inversion mode (seed(store_snapshot=False)) keeps no CPU baseline; it
        # only records seen names so encode can tell known params from unknown.
        self._snapshot_names: set[str] = set()
        self._snapshot_backed = False
        self._base_version: int | None = None
        self._deltas_since_anchor = 0
        self._force_full = False
        self._force_full_reason = ""

    @property
    def seeded(self) -> bool:
        return self._base_version is not None

    @property
    def base_version(self) -> int | None:
        return self._base_version

    @property
    def snapshot_size_bytes(self) -> int:
        seen_ptrs = set()
        total = 0
        for t in self._snapshot.values():
            ptr = t.data_ptr()
            if ptr not in seen_ptrs:
                seen_ptrs.add(ptr)
                total += t.numel() * t.element_size()
        return total

    def request_full_sync(self, reason: str = "external") -> None:
        """Force the next payload to be a full sync (e.g. reader requested)."""
        self._force_full = True
        self._force_full_reason = reason

    def full_sync_reason(self, version: int) -> str | None:
        """Return why ``version`` must be a full sync, or None if delta is OK."""
        if not self.seeded:
            return "not_seeded"
        if self._force_full:
            return f"requested:{self._force_full_reason}"
        if (
            self._anchor_interval > 0
            and self._deltas_since_anchor >= self._anchor_interval
        ):
            return f"anchor_interval:{self._anchor_interval}"
        return None

    def seed(
        self,
        named_parameters: Iterable[tuple[str, torch.Tensor]],
        version: int,
        *,
        store_snapshot: bool = True,
    ) -> None:
        """(Re)build the baseline after a full dense transfer.

        Args:
            store_snapshot: when True (default, snapshot detector) build the CPU
                bf16 baseline used by ``encode(masks=None)``. When False
                (inversion detector) keep no baseline tensors — change masks come
                from AdamW inversion, so we only record the seen names so
                ``encode(masks=...)`` can distinguish known from unknown params.
                The detector mode is fixed until the next ``seed`` call.
        """
        start = time.time()
        self._snapshot.clear()
        self._snapshot_names.clear()
        self._snapshot_backed = store_snapshot
        count = 0
        unique = 0
        if store_snapshot:
            # Dedup tied parameters: aliased names share one CPU tensor object.
            by_storage: dict[tuple[int, int], torch.Tensor] = {}
            pin = torch.cuda.is_available()
            for name, param in named_parameters:
                _check_reserved_names((name,))
                data = param.detach()
                key = (data.data_ptr(), data.numel())
                cpu_tensor = by_storage.get(key)
                if cpu_tensor is None:
                    cpu_tensor = data.contiguous().cpu().clone()
                    if pin:
                        cpu_tensor = cpu_tensor.pin_memory()
                    by_storage[key] = cpu_tensor
                self._snapshot[name] = cpu_tensor
                count += 1
            unique = len(by_storage)
        else:
            for name, _ in named_parameters:
                _check_reserved_names((name,))
                self._snapshot_names.add(name)
                count += 1
        self._base_version = version
        self._deltas_since_anchor = 0
        self._force_full = False
        self._force_full_reason = ""
        logger.info(
            "DeltaTracker: seeded at version %d with %d params "
            "(snapshot=%s, %d unique storages, %.1fMB, took %.3fs)",
            version,
            count,
            store_snapshot,
            unique,
            self.snapshot_size_bytes / 1e6,
            time.time() - start,
        )

    def mark_delta_committed(
        self,
        version: int,
    ) -> None:
        """Advance the version chain after an externally encoded delta succeeds.

        Integrations which build and transfer sparse payloads without calling
        :meth:`encode` must call this only after the payload is durably applied by
        the receiver. This API belongs exclusively to external/inversion mode
        (``seed(..., store_snapshot=False)``). Snapshot-backed trackers advance
        through ``encode(masks=None)`` so their baseline and version stay atomic.
        Failed transfers must not call this method.
        """
        if not self.seeded:
            raise RuntimeError("DeltaTracker not seeded; run a full sync first.")
        if self._snapshot_backed:
            raise RuntimeError(
                "mark_delta_committed() is only valid for a tracker seeded with "
                "store_snapshot=False; snapshot-backed trackers must advance "
                "through encode(masks=None)."
            )
        reason = self.full_sync_reason(version)
        if reason is not None:
            raise RuntimeError(
                f"Delta version {version} requires a full sync ({reason})."
            )
        assert self._base_version is not None
        expected_version = self._base_version + 1
        if version != expected_version:
            raise ValueError(
                "Committed delta version must be contiguous with the base: "
                f"base={self._base_version}, expected={expected_version}, "
                f"version={version}."
            )

        self._base_version = version
        self._deltas_since_anchor += 1
        logger.info(
            "DeltaTracker: externally committed delta version %d (%d since anchor)",
            version,
            self._deltas_since_anchor,
        )

    @torch.no_grad()
    def encode(
        self,
        named_parameters: Iterable[tuple[str, torch.Tensor]],
        version: int,
        *,
        masks: dict[str, torch.Tensor] | None = None,
    ) -> EncodedDelta:
        """Diff current weights against the snapshot and build a delta payload.

        Args:
            named_parameters: ``(hf_name, tensor)`` pairs in the same naming as
                ``seed``. Names absent from the snapshot fall back to dense.
            version: The payload (target) version; becomes the new base.
            masks: optional ``{hf_name: bool change mask}`` from an external
                detector (AdamW inversion). When provided, the change mask comes
                from ``masks`` instead of an internal snapshot diff, and the CPU
                snapshot is NOT refreshed (inversion mode keeps no baseline). A
                name missing from ``masks`` falls back to dense. When ``masks``
                is None the snapshot detector is used. The selected path must
                match the ``store_snapshot`` mode of the latest ``seed`` call.

        The snapshot is refreshed in place (snapshot mode only), so after this
        call the tracker's base is ``version``.
        """
        if not self.seeded:
            raise RuntimeError("DeltaTracker not seeded; run a full sync first.")

        external = masks is not None
        if self._snapshot_backed and external:
            raise RuntimeError(
                "External masks require a tracker seeded with "
                "store_snapshot=False; snapshot-backed trackers must use "
                "encode(masks=None)."
            )
        if not self._snapshot_backed and not external:
            raise RuntimeError(
                "Snapshot diff requires a tracker seeded with "
                "store_snapshot=True; external/inversion trackers must provide "
                "masks or be reseeded."
            )
        start = time.time()
        result = EncodedDelta()
        # Per-call dedup for tied params: same storage -> compute once, emit
        # the same idx/val tensors under every aliased name.
        computed: dict[
            tuple[int, int], tuple[str, tuple[torch.Tensor, torch.Tensor] | None]
        ] = {}
        header_device: torch.device | None = None

        for name, param in named_parameters:
            _check_reserved_names((name,))
            cur = param.detach().contiguous()
            if header_device is None:
                header_device = cur.device
            numel = cur.numel()
            if numel == 0:
                continue
            elem_size = cur.element_size()
            dense_bytes = numel * elem_size
            result.total_elements += numel
            result.dense_bytes += dense_bytes

            # Resolve the change mask source: external detector vs snapshot.
            ext_indices: torch.Tensor | None = None
            if external:
                known = name in self._snapshot_names
                ext_mask = masks.get(name)
                bad = ext_mask is None
                if ext_mask is not None and not bad:
                    if ext_mask.dtype == torch.bool:
                        bad = ext_mask.numel() != numel
                    elif ext_mask.dtype in {torch.int32, torch.int64}:
                        ext_indices = ext_mask.to(cur.device).reshape(-1).to(torch.long)
                        bad = bool(
                            ext_indices.numel() > 0
                            and (
                                ext_indices.min().item() < 0
                                or ext_indices.max().item() >= numel
                            )
                        )
                    else:
                        bad = True
                if not known or bad:
                    # Unknown/missing mask: send dense, adopt name as known.
                    logger.warning(
                        "DeltaTracker: param %s has no usable external mask/indices, "
                        "sending dense",
                        name,
                    )
                    self._snapshot_names.add(name)
                    self._emit_dense(result, name, cur, numel, dense_bytes)
                    continue
            else:
                snap = self._snapshot.get(name)
                if snap is None or snap.dtype != cur.dtype or snap.numel() != numel:
                    # Unknown/reshaped param: send dense, adopt into snapshot.
                    logger.warning(
                        "DeltaTracker: param %s missing from snapshot or "
                        "mismatched, sending dense",
                        name,
                    )
                    self._adopt_snapshot(name, cur)
                    self._emit_dense(result, name, cur, numel, dense_bytes)
                    continue

            key = (param.detach().data_ptr(), numel)
            if key in computed:
                # Tied parameter alias: reuse the canonical result.
                canonical, sparse = computed[key]
                if sparse is None:
                    self._emit_dense(result, name, cur, numel, dense_bytes)
                else:
                    indices, values = sparse
                    if indices.numel() == 0:
                        result.num_unchanged += 1
                    else:
                        self._emit_sparse(result, name, indices, values, elem_size)
                logger.debug(
                    "DeltaTracker: %s aliases %s, reused delta", name, canonical
                )
                continue

            if external and ext_indices is not None:
                indices = ext_indices
            elif external:
                mask = ext_mask.to(cur.device).reshape(-1)
                indices = mask.nonzero(as_tuple=False).squeeze(1)
            else:
                old = snap.to(cur.device, non_blocking=False)
                mask = bitwise_changed_mask(cur, old).view(-1)
                indices = mask.nonzero(as_tuple=False).squeeze(1)
            changed = indices.numel()
            result.changed_elements += changed

            if changed == 0:
                result.num_unchanged += 1
                computed[key] = (name, (indices.to(torch.int32), cur.new_empty(0)))
                continue

            sparse_bytes = changed * (4 + elem_size)
            if (
                numel < _MAX_INT32_NUMEL
                and sparse_bytes <= self._sparse_bytes_ratio * dense_bytes
            ):
                values = cur.view(-1)[indices]
                indices = indices.to(torch.int32)
                self._emit_sparse(result, name, indices, values, elem_size)
                computed[key] = (name, (indices, values))
                if not external:
                    # Refresh snapshot in place: scatter only changed elements.
                    snap.view(-1)[indices.cpu().long()] = values.cpu()
            else:
                self._emit_dense(result, name, cur, numel, dense_bytes)
                computed[key] = (name, None)
                if not external:
                    snap.copy_(cur, non_blocking=False)

        result.header = DeltaHeader(
            payload_version=version,
            base_version=self._base_version,
            num_sparse=result.num_sparse,
            num_dense=result.num_dense_fallback,
        )
        result.names.insert(0, DELTA_HEADER_NAME)
        result.tensors.insert(0, result.header.to_tensor(device=header_device))

        self._base_version = version
        self._deltas_since_anchor += 1
        logger.info(
            "DeltaTracker: %s, took %.3fs", result.summary(), time.time() - start
        )
        return result

    def _adopt_snapshot(self, name: str, cur: torch.Tensor) -> None:
        cpu_tensor = cur.cpu().clone()
        if torch.cuda.is_available():
            cpu_tensor = cpu_tensor.pin_memory()
        self._snapshot[name] = cpu_tensor

    @staticmethod
    def _emit_sparse(
        result: EncodedDelta,
        name: str,
        indices: torch.Tensor,
        values: torch.Tensor,
        elem_size: int,
    ) -> None:
        result.names.append(name + DELTA_IDX_SUFFIX)
        result.tensors.append(indices)
        result.names.append(name + DELTA_VAL_SUFFIX)
        result.tensors.append(values)
        result.num_sparse += 1
        result.payload_bytes += indices.numel() * (4 + elem_size)

    @staticmethod
    def _emit_dense(
        result: EncodedDelta,
        name: str,
        cur: torch.Tensor,
        numel: int,
        dense_bytes: int,
    ) -> None:
        result.names.append(name)
        result.tensors.append(cur)
        result.num_dense_fallback += 1
        result.payload_bytes += dense_bytes


# ---------------------------------------------------------------------------
# Reader side
# ---------------------------------------------------------------------------


@dataclass
class DecodedDelta:
    """Reader-side view of a delta payload split by apply mode."""

    header: DeltaHeader
    dense: dict[str, torch.Tensor] = field(default_factory=dict)
    sparse: dict[str, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)

    def iter_sparse(self) -> Iterator[tuple[str, torch.Tensor, torch.Tensor]]:
        for name, (indices, values) in self.sparse.items():
            yield name, indices, values


def is_delta_payload(names: Iterable[str]) -> bool:
    """Whether a deserialized names list carries a delta payload."""
    return DELTA_HEADER_NAME in names


def decode_delta_payload(named_tensors: dict[str, torch.Tensor]) -> DecodedDelta:
    """Split ``dict(zip(names, tensors))`` into header / dense / sparse parts.

    Raises:
        ValueError: missing/invalid header, or an idx tensor without its
            matching val tensor (corrupt payload).
    """
    header_tensor = named_tensors.get(DELTA_HEADER_NAME)
    if header_tensor is None:
        raise ValueError("Not a delta payload: header tensor missing")
    decoded = DecodedDelta(header=DeltaHeader.from_tensor(header_tensor.cpu()))

    pending_idx: dict[str, torch.Tensor] = {}
    pending_val: dict[str, torch.Tensor] = {}
    for name, tensor in named_tensors.items():
        if name == DELTA_HEADER_NAME:
            continue
        if name.endswith(DELTA_IDX_SUFFIX):
            pending_idx[name[: -len(DELTA_IDX_SUFFIX)]] = tensor
        elif name.endswith(DELTA_VAL_SUFFIX):
            pending_val[name[: -len(DELTA_VAL_SUFFIX)]] = tensor
        else:
            decoded.dense[name] = tensor

    if set(pending_idx) != set(pending_val):
        missing = set(pending_idx).symmetric_difference(pending_val)
        raise ValueError(f"Corrupt delta payload, unpaired sparse entries: {missing}")
    for name, indices in pending_idx.items():
        values = pending_val[name]
        if indices.numel() != values.numel():
            raise ValueError(
                f"Corrupt sparse entry {name}: "
                f"{indices.numel()} indices vs {values.numel()} values"
            )
        decoded.sparse[name] = (indices, values)

    if decoded.header.num_sparse != len(decoded.sparse) or (
        decoded.header.num_dense != len(decoded.dense)
    ):
        raise ValueError(
            f"Delta payload count mismatch: header says "
            f"sparse={decoded.header.num_sparse}/dense={decoded.header.num_dense}, "
            f"payload has sparse={len(decoded.sparse)}/dense={len(decoded.dense)}"
        )
    return decoded


@torch.no_grad()
def apply_sparse_patch_(
    target: torch.Tensor, indices: torch.Tensor, values: torch.Tensor
) -> None:
    """Scatter sparse values into ``target`` in place (flat int32/int64 indices).

    ``target`` is typically a live inference weight (write-through view); a
    non-contiguous target is handled via a contiguous staging copy so the
    writes are not silently dropped on a temporary.
    """
    if indices.numel() == 0:
        return
    if values.dtype != target.dtype:
        values = values.to(target.dtype)
    idx = indices.to(device=target.device, dtype=torch.int64)
    values = values.to(device=target.device)
    if target.is_contiguous():
        target.view(-1)[idx] = values
    else:
        staged = target.contiguous()
        staged.view(-1)[idx] = values
        target.copy_(staged)


@torch.no_grad()
def reconstruct_against_base(
    base: dict[str, torch.Tensor],
    decoded: DecodedDelta,
    device,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Rebuild full tensors from a CPU ``base`` and a decoded delta.

    Single source of truth for the reader's reconstruction (also unit-tested
    directly at ``device='cpu'``). The returned tensors are keyed and shaped
    exactly like a dense payload so a downstream transport runs unchanged;
    ``base`` is refreshed in place for the next version.

    - dense  entry -> use the full tensor, refresh base.
    - sparse entry -> scatter onto a *copy* of base, refresh base.
    - absent entry -> restore base value (param unchanged this version).

    A sparse patch for a name absent from the base cannot be reconstructed
    (its full shape is unknown — never dead-reckon it from ``indices.max()``);
    that is a broken chain and raises. A dense entry absent from the base is a
    first-seen full param and is adopted.

    Returns ``(result, counts)`` where counts has sparse/dense/unchanged keys.
    """
    result: dict[str, torch.Tensor] = {}
    counts = {"sparse": 0, "dense": 0, "unchanged": 0}

    absent_sparse = [n for n in decoded.sparse if n not in base]
    if absent_sparse:
        raise ValueError(
            f"Delta sparse params absent from base, cannot reconstruct "
            f"(unknown full shape): {absent_sparse}"
        )

    for name, base_cpu in base.items():
        if name in decoded.dense:
            full = decoded.dense[name].to(device)
            base_cpu.copy_(full.detach().to("cpu", non_blocking=False))
            counts["dense"] += 1
        elif name in decoded.sparse:
            indices, values = decoded.sparse[name]
            # copy=True: never alias the base (matters on the CPU test path;
            # a H2D copy on the real path is unconditional anyway).
            full = base_cpu.to(device, copy=True)
            apply_sparse_patch_(full, indices, values)
            base_cpu.copy_(full.detach().to("cpu", non_blocking=False))
            counts["sparse"] += 1
        else:
            full = base_cpu.to(device, copy=True)
            counts["unchanged"] += 1
        result[name] = full

    # First-seen dense params (not yet in base): adopt them.
    for name in decoded.dense:
        if name in base:
            continue
        full = decoded.dense[name].to(device)
        result[name] = full
        base[name] = full.detach().to("cpu", copy=True)
        counts["dense"] += 1

    return result, counts
