# Licensed under the Apache License, Version 2.0
"""BF16 change detector for delta weight sync.

Detects which bf16 weight elements changed after an optimizer step by
maintaining a CPU pinned bf16 snapshot and comparing element-wise.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator

import torch

from dte.core.codec import bitwise_changed_mask
from dte.core.patch import DeltaResult, SparseWeightPatch

logger = logging.getLogger(__name__)


class DeltaWeightDetector:
    """Standalone bf16 change detector (snapshot diff).

    Maintains a CPU pinned bf16 snapshot. After each optimizer step, compares
    current weights against the snapshot to find changed elements.

    .. note::
        ``DeltaEngine`` does NOT use this class — it drives ``DeltaTracker``
        (``codec.py``), which is a superset: snapshot diff **plus** version chain,
        per-tensor dense fallback, tied-param dedup, and the flat payload codec.
        ``DeltaWeightDetector`` is the minimal, self-contained detector kept for
        (a) direct unit testing of the bitwise-diff core and (b) reuse by
        callers that only want change detection without the full encode pipeline.
        For end-to-end sync use ``DeltaTracker``.

    Usage:
        detector = DeltaWeightDetector()
        detector.init_snapshot(named_params_iter)
        # ... training step ...
        result = detector.compute_delta(named_params_iter)
        if result.should_use_delta():
            # use sparse transfer
    """

    def __init__(self):
        self._snapshot: dict[str, torch.Tensor] = {}
        self._step_count: int = 0

    @property
    def initialized(self) -> bool:
        return len(self._snapshot) > 0

    def init_snapshot(
        self, named_parameters: Iterator[tuple[str, torch.Tensor]]
    ) -> None:
        """Initialize baseline snapshot from current weights.

        Call this after the first full weight transfer (anchor step).
        """
        self._snapshot.clear()
        count = 0
        for name, param in named_parameters:
            cpu_tensor = param.data.detach().to(torch.bfloat16).cpu().clone()
            if torch.cuda.is_available():
                cpu_tensor = cpu_tensor.pin_memory()
            self._snapshot[name] = cpu_tensor
            count += 1
        logger.info("DeltaWeightDetector: initialized snapshot with %d params", count)

    def compute_delta(
        self, named_parameters: Iterator[tuple[str, torch.Tensor]]
    ) -> DeltaResult:
        """Compare current weights against snapshot, return sparse delta.

        Args:
            named_parameters: Iterator of (hf_name, gpu_tensor) pairs.
                Must use the same naming as init_snapshot.

        Returns:
            DeltaResult containing SparseWeightPatch for each changed param.
        """
        if not self._snapshot:
            raise RuntimeError("Snapshot not initialized. Call init_snapshot first.")

        self._step_count += 1
        start_time = time.time()

        patches: list[SparseWeightPatch] = []
        total_elements = 0
        changed_elements = 0

        for name, param in named_parameters:
            if name not in self._snapshot:
                logger.warning("Parameter %s not in snapshot, skipping", name)
                continue

            current_bf16 = param.data.detach().to(torch.bfloat16)
            old_bf16 = self._snapshot[name].to(param.device, non_blocking=False)

            # Bitwise compare (NaN bits and signed zero handled exactly).
            mask = bitwise_changed_mask(current_bf16, old_bf16)
            numel = current_bf16.numel()
            total_elements += numel

            if mask.any():
                indices = mask.flatten().nonzero(as_tuple=False).squeeze(1)
                if int(indices.max()) >= 2**31:
                    raise ValueError(
                        f"Parameter {name} too large for int32 indices: "
                        f"max_idx={int(indices.max())}, numel={numel}"
                    )
                indices = indices.to(torch.int32)
                values = current_bf16.flatten()[indices.long()]
                patches.append(
                    SparseWeightPatch(name=name, indices=indices, values=values)
                )
                changed_elements += indices.numel()

            # Always update snapshot (prevents drift on corruption)
            self._snapshot[name].copy_(current_bf16.cpu())

        duration = time.time() - start_time
        result = DeltaResult(
            patches=patches,
            total_elements=total_elements,
            changed_elements=changed_elements,
            step=self._step_count,
        )

        logger.info(
            "DeltaWeightDetector step %d: %s, took %.3fs",
            self._step_count,
            result.summary(),
            duration,
        )
        return result

    def update_snapshot_full(
        self, named_parameters: Iterator[tuple[str, torch.Tensor]]
    ) -> None:
        """Force-update snapshot with current weights (after a full transfer)."""
        for name, param in named_parameters:
            if name in self._snapshot:
                self._snapshot[name].copy_(param.data.detach().to(torch.bfloat16).cpu())

    @property
    def snapshot_size_bytes(self) -> int:
        return sum(t.numel() * 2 for t in self._snapshot.values())

    @property
    def num_params(self) -> int:
        return len(self._snapshot)
