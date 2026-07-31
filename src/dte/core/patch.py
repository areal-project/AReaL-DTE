# Licensed under the Apache License, Version 2.0
"""Delta weight patch data structures and serialization.

Aligned with vLLM's SparseWeightPatch format (name + flat indices int32 + values).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import torch


@dataclass
class SparseWeightPatch:
    """A sparse in-place patch for one parameter.

    Aligned with vLLM's SparseWeightPatch format.
    """

    name: str
    indices: torch.Tensor  # int32, flat indices into the parameter
    values: torch.Tensor  # same dtype as parameter (typically bf16)

    @property
    def num_updates(self) -> int:
        return self.indices.numel()

    @property
    def size_bytes(self) -> int:
        return (
            self.indices.numel() * 4 + self.values.numel() * self.values.element_size()
        )


@dataclass
class DeltaResult:
    """Result of delta detection: sparse patches + statistics."""

    patches: list[SparseWeightPatch] = field(default_factory=list)
    total_elements: int = 0
    changed_elements: int = 0
    step: int = 0

    @property
    def sparsity(self) -> float:
        if self.total_elements == 0:
            return 1.0
        return 1.0 - self.changed_elements / self.total_elements

    @property
    def delta_size_bytes(self) -> int:
        return sum(p.size_bytes for p in self.patches)

    @property
    def full_size_bytes(self) -> int:
        return self.total_elements * 2  # assume bf16

    def should_use_delta(self, threshold: float = 0.5) -> bool:
        """Use delta when delta_size < threshold * full_size."""
        return self.delta_size_bytes < threshold * self.full_size_bytes

    def iter_patches(self) -> Iterator[SparseWeightPatch]:
        return iter(self.patches)

    @property
    def num_changed_params(self) -> int:
        return len(self.patches)

    def summary(self) -> str:
        return (
            f"DeltaResult(step={self.step}, "
            f"changed={self.changed_elements}/{self.total_elements}, "
            f"sparsity={self.sparsity:.4f}, "
            f"delta_size={self.delta_size_bytes / 1e6:.1f}MB, "
            f"full_size={self.full_size_bytes / 1e6:.1f}MB, "
            f"params_changed={self.num_changed_params})"
        )


def patches_to_dict(
    patches: list[SparseWeightPatch],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Convert patches to {name: (indices, values)} dict for serialization.

    Merges patches with the same name by concatenating indices and values.
    """
    merged: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    for p in patches:
        if p.name not in merged:
            merged[p.name] = []
        merged[p.name].append((p.indices, p.values))

    result = {}
    for name, parts in merged.items():
        if len(parts) == 1:
            result[name] = parts[0]
        else:
            result[name] = (
                torch.cat([idx for idx, _ in parts]),
                torch.cat([val for _, val in parts]),
            )
    return result


def dict_to_patches(
    d: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> list[SparseWeightPatch]:
    """Convert {name: (indices, values)} dict back to patches."""
    return [
        SparseWeightPatch(name=name, indices=indices, values=values)
        for name, (indices, values) in d.items()
    ]
