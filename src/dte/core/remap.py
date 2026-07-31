# Licensed under the Apache License, Version 2.0
"""Delta index remapping for TP mismatch scenarios.

When training TP != inference TP, flat delta indices computed on the training
shard must be filtered and remapped to the inference shard's coordinate space.
Uses AWEX's CommunicationOperation (train_slices / inf_slices) to define
the overlap region for each (train_rank, infer_rank) pair.
"""

from __future__ import annotations

import logging

import torch

from dte.core.patch import SparseWeightPatch

logger = logging.getLogger(__name__)

# Per-row searchsorted avoids repeated nnz scans for small/medium row spans, but
# can lose to the vectorized mask fallback on very tall matrices.
_MAX_RECT_FAST_PATH_ROWS = 8192


def remap_delta_indices(
    patch: SparseWeightPatch,
    train_shape: tuple[int, ...],
    train_slices: tuple[slice, ...],
    inf_slices: tuple[slice, ...],
    infer_shape: tuple[int, ...],
) -> SparseWeightPatch | None:
    """Remap flat delta indices from training shard space to inference shard space.

    Uses the overlap region defined by CommunicationOperation to:
    1. Unflatten training flat indices → multi-dim
    2. Filter: keep only indices within train_slices
    3. Remap: translate from train_slices coords to inf_slices coords
    4. Flatten for inference shard shape

    Args:
        patch: sparse patch with indices in training shard flat space
        train_shape: shape of the training shard tensor
        train_slices: slice ranges defining the overlap in training shard
        inf_slices: slice ranges defining where to write in inference shard
        infer_shape: shape of the inference shard tensor

    Returns:
        New SparseWeightPatch with remapped indices, or None if no overlap.
    """
    indices = patch.indices.long()
    values = patch.values
    ndim = len(train_shape)

    if ndim == 0 or indices.numel() == 0:
        return None

    # Step 1: unflatten to multi-dim indices
    if ndim == 1:
        multi_idx = (indices,)
    elif ndim == 2:
        multi_idx = (indices // train_shape[1], indices % train_shape[1])
    else:
        multi_idx = _unravel_index(indices, train_shape)

    # Step 2: filter — keep indices within train_slices
    mask = torch.ones(indices.numel(), dtype=torch.bool, device=indices.device)
    for dim in range(ndim):
        s = train_slices[dim]
        if s == slice(None):
            continue
        _assert_unit_step(s, dim, patch.name)
        start = s.start or 0
        stop = s.stop if s.stop is not None else train_shape[dim]
        mask &= (multi_idx[dim] >= start) & (multi_idx[dim] < stop)

    if not mask.any():
        return None

    # Step 3: remap — train_slices space → inf_slices space
    remapped = []
    for dim in range(ndim):
        dim_idx = multi_idx[dim][mask]
        t_start = (
            (train_slices[dim].start or 0) if train_slices[dim] != slice(None) else 0
        )
        i_start = (inf_slices[dim].start or 0) if inf_slices[dim] != slice(None) else 0
        if train_slices[dim] != slice(None):
            _assert_unit_step(train_slices[dim], dim, patch.name)
        if inf_slices[dim] != slice(None):
            _assert_unit_step(inf_slices[dim], dim, patch.name)
        remapped.append(dim_idx - t_start + i_start)

    # Step 4: flatten for inference shard
    new_flat = _ravel_multi_index(remapped, infer_shape)
    _assert_int32_safe(new_flat, infer_shape, patch.name)

    return SparseWeightPatch(
        name=patch.name,
        indices=new_flat.to(torch.int32),
        values=values[mask],
    )


def remap_delta_indices_for_ops(
    patch: SparseWeightPatch,
    train_shape: tuple[int, ...],
    operations: list,
    *,
    assume_sorted: bool = False,
) -> list[SparseWeightPatch | None]:
    """Remap one train-shard sparse patch for many CommunicationOperations.

    ``remap_delta_indices`` is the simple single-op entry point. The colocate
    receiver path may have several cross-rank ops for the same parameter, and
    calling the single-op function repeatedly redoes the same flat-index
    unravelling each time. This helper shares that decode once, then applies
    each op's train/inference slice filter independently.

    Set ``assume_sorted`` only for internal codec-produced sparse patches. It
    skips an O(nnz) sortedness scan and GPU sync before trying range fast paths.
    """
    indices = patch.indices.long()
    values = patch.values
    ndim = len(train_shape)

    if ndim == 0 or indices.numel() == 0:
        return [None for _ in operations]

    indices_are_sorted = assume_sorted or _is_sorted_ascending(indices)
    multi_idx: tuple[torch.Tensor, ...] | None = None

    if indices_are_sorted:
        handled, batched = _try_remap_sorted_contiguous_ranges_for_ops(
            patch,
            indices,
            values,
            train_shape,
            operations,
        )
        if handled:
            return batched
        handled, batched = _try_remap_sorted_rectangular_ranges_for_ops(
            patch,
            indices,
            values,
            train_shape,
            operations,
        )
        if handled:
            return batched

    remapped_patches: list[SparseWeightPatch | None] = []
    for op in operations:
        train_slices = op.train_slices
        inf_slices = op.inf_slices
        infer_shape = tuple(op.recv_shard_meta.shape)

        if indices_are_sorted:
            handled, fast_patch = _try_remap_sorted_contiguous_range(
                patch,
                indices,
                values,
                train_shape,
                train_slices,
                inf_slices,
                infer_shape,
            )
            if handled:
                remapped_patches.append(fast_patch)
                continue

        if multi_idx is None:
            if ndim == 1:
                multi_idx = (indices,)
            elif ndim == 2:
                multi_idx = (indices // train_shape[1], indices % train_shape[1])
            else:
                multi_idx = _unravel_index(indices, train_shape)

        mask = torch.ones(indices.numel(), dtype=torch.bool, device=indices.device)
        for dim in range(ndim):
            s = train_slices[dim]
            if s == slice(None):
                continue
            _assert_unit_step(s, dim, patch.name)
            start = s.start or 0
            stop = s.stop if s.stop is not None else train_shape[dim]
            mask &= (multi_idx[dim] >= start) & (multi_idx[dim] < stop)

        selected = mask.nonzero(as_tuple=True)[0]
        if selected.numel() == 0:
            remapped_patches.append(None)
            continue

        remapped = []
        for dim in range(ndim):
            dim_idx = multi_idx[dim].index_select(0, selected)
            t_start = (
                (train_slices[dim].start or 0)
                if train_slices[dim] != slice(None)
                else 0
            )
            i_start = (
                (inf_slices[dim].start or 0) if inf_slices[dim] != slice(None) else 0
            )
            if train_slices[dim] != slice(None):
                _assert_unit_step(train_slices[dim], dim, patch.name)
            if inf_slices[dim] != slice(None):
                _assert_unit_step(inf_slices[dim], dim, patch.name)
            remapped.append(dim_idx - t_start + i_start)

        new_flat = _ravel_multi_index(remapped, infer_shape)
        _assert_int32_safe(new_flat, infer_shape, patch.name)
        remapped_patches.append(
            SparseWeightPatch(
                name=patch.name,
                indices=new_flat.to(torch.int32),
                values=values.index_select(0, selected),
            )
        )

    return remapped_patches


def _try_remap_sorted_contiguous_ranges_for_ops(
    patch: SparseWeightPatch,
    indices: torch.Tensor,
    values: torch.Tensor,
    train_shape: tuple[int, ...],
    operations: list,
) -> tuple[bool, list[SparseWeightPatch | None]]:
    """Batched fast path for row-major-contiguous op slices.

    The single-op fast path uses scalar ``searchsorted`` and then reads the
    bounds with ``.item()``. On GPU colocate runs that becomes one device sync
    per transfer-plan op. For the common 1D and 2D row-slice cases, compute all
    bounds for one parameter in a single vectorized ``searchsorted`` and perform
    one host transfer for the small boundary vectors.
    """
    if not operations:
        return True, []

    ndim = len(train_shape)
    name = patch.name

    starts: list[int] = []
    stops: list[int] = []
    row_starts: list[int] = []
    inf_row_starts: list[int] = []
    inf_starts: list[int] = []
    infer_shapes: list[tuple[int, ...]] = []

    if ndim == 1:
        for op in operations:
            infer_shape = tuple(op.recv_shard_meta.shape)
            start, stop = _slice_bounds(op.train_slices[0], train_shape[0], 0, name)
            inf_start, _ = _slice_bounds(op.inf_slices[0], infer_shape[0], 0, name)
            starts.append(start)
            stops.append(stop)
            inf_starts.append(inf_start)
            infer_shapes.append(infer_shape)
    elif ndim == 2:
        rows, cols = train_shape
        for op in operations:
            infer_shape = tuple(op.recv_shard_meta.shape)
            if not _is_full_slice(op.train_slices[1], cols, 1, name):
                return False, []
            if not _is_full_slice(op.inf_slices[1], infer_shape[1], 1, name):
                return False, []
            row_start, row_stop = _slice_bounds(op.train_slices[0], rows, 0, name)
            inf_row_start, _ = _slice_bounds(op.inf_slices[0], infer_shape[0], 0, name)
            starts.append(row_start * cols)
            stops.append(row_stop * cols)
            row_starts.append(row_start)
            inf_row_starts.append(inf_row_start)
            infer_shapes.append(infer_shape)
    else:
        return False, []

    starts_t = torch.tensor(starts, dtype=indices.dtype, device=indices.device)
    stops_t = torch.tensor(stops, dtype=indices.dtype, device=indices.device)
    lo_t = torch.searchsorted(indices, starts_t, right=False)
    hi_t = torch.searchsorted(indices, stops_t, right=False)
    lo_list = lo_t.cpu().tolist()
    hi_list = hi_t.cpu().tolist()

    remapped: list[SparseWeightPatch | None] = []
    for i, (lo, hi) in enumerate(zip(lo_list, hi_list, strict=True)):
        if lo == hi:
            remapped.append(None)
            continue
        selected = indices[lo:hi]
        infer_shape = infer_shapes[i]
        if ndim == 1:
            new_flat = selected - starts[i] + inf_starts[i]
        else:
            cols = train_shape[1]
            infer_cols = infer_shape[1]
            row_start = row_starts[i]
            inf_row_start = inf_row_starts[i]
            if cols == infer_cols:
                new_flat = selected - starts[i] + inf_row_start * infer_cols
            else:
                row = selected // cols
                col = selected % cols
                new_flat = (row - row_start + inf_row_start) * infer_cols + col
        _assert_int32_safe(new_flat, infer_shape, name)
        remapped.append(
            SparseWeightPatch(
                name=name,
                indices=new_flat.to(torch.int32),
                values=values[lo:hi],
            )
        )
    return True, remapped


def _try_remap_sorted_rectangular_ranges_for_ops(
    patch: SparseWeightPatch,
    indices: torch.Tensor,
    values: torch.Tensor,
    train_shape: tuple[int, ...],
    operations: list,
) -> tuple[bool, list[SparseWeightPatch | None]]:
    """Batched fast path for 2D rectangular op slices.

    Column or general rectangle slices are a union of per-row contiguous flat
    ranges. The single-op rectangle fast path repeats row-wise ``searchsorted``
    for every peer op. In colocate cross apply, one sparse parameter commonly
    has several peer ops, so batch those row ranges once and then split the
    resulting selected positions back into per-op payloads.
    """
    if not operations:
        return True, []

    if len(train_shape) != 2:
        return False, []

    name = patch.name
    rows, cols = train_shape
    row_chunks: list[torch.Tensor] = []
    col_start_chunks: list[torch.Tensor] = []
    col_stop_chunks: list[torch.Tensor] = []
    op_id_chunks: list[torch.Tensor] = []
    op_meta: list[tuple[int, int, int, int, tuple[int, ...]]] = []
    row_counts: list[int] = []

    for op_index, op in enumerate(operations):
        infer_shape = tuple(op.recv_shard_meta.shape)
        if len(infer_shape) != 2:
            return False, []
        infer_rows, infer_cols = infer_shape
        row_start, row_stop = _slice_bounds(op.train_slices[0], rows, 0, name)
        col_start, col_stop = _slice_bounds(op.train_slices[1], cols, 1, name)
        inf_row_start, _ = _slice_bounds(op.inf_slices[0], infer_rows, 0, name)
        inf_col_start, _ = _slice_bounds(op.inf_slices[1], infer_cols, 1, name)

        row_count = row_stop - row_start
        if row_count > _MAX_RECT_FAST_PATH_ROWS:
            return False, []

        op_meta.append(
            (row_start, col_start, inf_row_start, inf_col_start, infer_shape)
        )
        if row_count <= 0 or col_start >= col_stop:
            row_counts.append(0)
            continue

        row_counts.append(row_count)
        row_chunks.append(
            torch.arange(
                row_start,
                row_stop,
                device=indices.device,
                dtype=indices.dtype,
            )
        )
        col_start_chunks.append(
            torch.full(
                (row_count,),
                col_start,
                device=indices.device,
                dtype=indices.dtype,
            )
        )
        col_stop_chunks.append(
            torch.full(
                (row_count,),
                col_stop,
                device=indices.device,
                dtype=indices.dtype,
            )
        )
        op_id_chunks.append(
            torch.full(
                (row_count,),
                op_index,
                device=indices.device,
                dtype=torch.long,
            )
        )

    if not row_chunks:
        return True, [None for _ in operations]

    row_ids = torch.cat(row_chunks) if len(row_chunks) > 1 else row_chunks[0]
    col_starts = (
        torch.cat(col_start_chunks)
        if len(col_start_chunks) > 1
        else col_start_chunks[0]
    )
    col_stops = (
        torch.cat(col_stop_chunks) if len(col_stop_chunks) > 1 else col_stop_chunks[0]
    )
    op_ids = torch.cat(op_id_chunks) if len(op_id_chunks) > 1 else op_id_chunks[0]

    range_starts = row_ids * cols + col_starts
    range_stops = row_ids * cols + col_stops
    lo = torch.searchsorted(indices, range_starts, right=False)
    hi = torch.searchsorted(indices, range_stops, right=False)
    counts = hi - lo

    op_counts_t = torch.zeros(
        len(operations),
        dtype=counts.dtype,
        device=indices.device,
    )
    op_counts_t.scatter_add_(0, op_ids, counts)
    op_counts = [int(v) for v in op_counts_t.cpu().tolist()]
    total = sum(op_counts)
    if total == 0:
        return True, [None for _ in operations]

    non_empty = counts > 0
    lo = lo[non_empty]
    counts = counts[non_empty]
    out_starts = torch.cumsum(counts, dim=0) - counts
    base = torch.repeat_interleave(lo, counts)
    repeated_out_starts = torch.repeat_interleave(out_starts, counts)
    offsets = (
        torch.arange(total, device=indices.device, dtype=lo.dtype) - repeated_out_starts
    )
    selected_positions_all = base + offsets

    remapped: list[SparseWeightPatch | None] = []
    offset = 0
    for op_count, meta in zip(op_counts, op_meta, strict=True):
        if op_count == 0:
            remapped.append(None)
            continue
        selected_positions = selected_positions_all[offset : offset + op_count]
        offset += op_count
        selected = indices.index_select(0, selected_positions)
        row_start, col_start, inf_row_start, inf_col_start, infer_shape = meta
        infer_cols = infer_shape[1]
        row = selected // cols
        col = selected % cols
        new_flat = (row - row_start + inf_row_start) * infer_cols + (
            col - col_start + inf_col_start
        )
        _assert_int32_safe(new_flat, infer_shape, name)
        remapped.append(
            SparseWeightPatch(
                name=name,
                indices=new_flat.to(torch.int32),
                values=values.index_select(0, selected_positions),
            )
        )

    return True, remapped


def _is_sorted_ascending(indices: torch.Tensor) -> bool:
    if indices.numel() <= 1:
        return True
    return bool((indices[:-1] <= indices[1:]).all().item())


def _slice_bounds(s: slice, dim_size: int, dim: int, name: str) -> tuple[int, int]:
    _assert_unit_step(s, dim, name)
    start = s.start if s.start is not None else 0
    stop = s.stop if s.stop is not None else dim_size
    return start, stop


def _is_full_slice(s: slice, dim_size: int, dim: int, name: str) -> bool:
    start, stop = _slice_bounds(s, dim_size, dim, name)
    return start == 0 and stop == dim_size


def _searchsorted_range(
    indices: torch.Tensor,
    start: int,
    stop: int,
) -> tuple[int, int]:
    start_t = torch.tensor(start, dtype=indices.dtype, device=indices.device)
    stop_t = torch.tensor(stop, dtype=indices.dtype, device=indices.device)
    lo = torch.searchsorted(indices, start_t, right=False)
    hi = torch.searchsorted(indices, stop_t, right=False)
    return int(lo.item()), int(hi.item())


def _try_remap_sorted_contiguous_range(
    patch: SparseWeightPatch,
    indices: torch.Tensor,
    values: torch.Tensor,
    train_shape: tuple[int, ...],
    train_slices: tuple[slice, ...],
    inf_slices: tuple[slice, ...],
    infer_shape: tuple[int, ...],
) -> tuple[bool, SparseWeightPatch | None]:
    """Fast path for sorted indices and row-major-contiguous op slices.

    Most sparse payload indices are produced by ``nonzero`` or packed-bitset
    scans, so they are sorted. For 1D slices and 2D row slices with full columns,
    an op overlap is a single flat index range; ``searchsorted`` selects it
    without building a full boolean mask for every op.
    """
    ndim = len(train_shape)
    name = patch.name

    if ndim == 1:
        start, stop = _slice_bounds(train_slices[0], train_shape[0], 0, name)
        inf_start, _ = _slice_bounds(inf_slices[0], infer_shape[0], 0, name)
        lo, hi = _searchsorted_range(indices, start, stop)
        if lo == hi:
            return True, None
        selected = indices[lo:hi]
        new_flat = selected - start + inf_start
        _assert_int32_safe(new_flat, infer_shape, name)
        return True, SparseWeightPatch(
            name=name,
            indices=new_flat.to(torch.int32),
            values=values[lo:hi],
        )

    if ndim == 2 and _is_full_slice(train_slices[1], train_shape[1], 1, name):
        if not _is_full_slice(inf_slices[1], infer_shape[1], 1, name):
            return False, None
        cols = train_shape[1]
        row_start, row_stop = _slice_bounds(train_slices[0], train_shape[0], 0, name)
        inf_row_start, _ = _slice_bounds(inf_slices[0], infer_shape[0], 0, name)
        lo, hi = _searchsorted_range(
            indices,
            row_start * cols,
            row_stop * cols,
        )
        if lo == hi:
            return True, None
        selected = indices[lo:hi]
        infer_cols = infer_shape[1]
        if cols == infer_cols:
            new_flat = selected - row_start * cols + inf_row_start * infer_cols
        else:
            row = selected // cols
            col = selected % cols
            new_flat = (row - row_start + inf_row_start) * infer_cols + col
        _assert_int32_safe(new_flat, infer_shape, name)
        return True, SparseWeightPatch(
            name=name,
            indices=new_flat.to(torch.int32),
            values=values[lo:hi],
        )

    if ndim == 2:
        return _try_remap_sorted_2d_rectangle(
            patch,
            indices,
            values,
            train_shape,
            train_slices,
            inf_slices,
            infer_shape,
        )

    return False, None


def _try_remap_sorted_2d_rectangle(
    patch: SparseWeightPatch,
    indices: torch.Tensor,
    values: torch.Tensor,
    train_shape: tuple[int, ...],
    train_slices: tuple[slice, ...],
    inf_slices: tuple[slice, ...],
    infer_shape: tuple[int, ...],
) -> tuple[bool, SparseWeightPatch | None]:
    """Fast path for sorted 2D indices over a rectangular slice.

    A column slice or general 2D rectangle is not one contiguous flat range, but
    it is a union of per-row contiguous ranges. For cross-rank transfer plans
    this avoids scanning the full sparse index tensor once per peer op.
    """
    name = patch.name
    rows, cols = train_shape
    infer_rows, infer_cols = infer_shape

    row_start, row_stop = _slice_bounds(train_slices[0], rows, 0, name)
    col_start, col_stop = _slice_bounds(train_slices[1], cols, 1, name)
    inf_row_start, _ = _slice_bounds(inf_slices[0], infer_rows, 0, name)
    inf_col_start, _ = _slice_bounds(inf_slices[1], infer_cols, 1, name)

    if row_start >= row_stop or col_start >= col_stop:
        return True, None
    if row_stop - row_start > _MAX_RECT_FAST_PATH_ROWS:
        return False, None

    row_ids = torch.arange(
        row_start,
        row_stop,
        device=indices.device,
        dtype=indices.dtype,
    )
    range_starts = row_ids * cols + col_start
    range_stops = row_ids * cols + col_stop
    lo = torch.searchsorted(indices, range_starts, right=False)
    hi = torch.searchsorted(indices, range_stops, right=False)
    counts = hi - lo
    total = int(counts.sum().item())
    if total == 0:
        return True, None

    non_empty = counts > 0
    lo = lo[non_empty]
    counts = counts[non_empty]
    out_starts = torch.cumsum(counts, dim=0) - counts
    base = torch.repeat_interleave(lo, counts)
    repeated_out_starts = torch.repeat_interleave(out_starts, counts)
    offsets = (
        torch.arange(total, device=indices.device, dtype=indices.dtype)
        - repeated_out_starts
    )
    selected_positions = base + offsets
    selected = indices.index_select(0, selected_positions)

    row = selected // cols
    col = selected % cols
    new_flat = (row - row_start + inf_row_start) * infer_cols + (
        col - col_start + inf_col_start
    )
    _assert_int32_safe(new_flat, infer_shape, name)
    return True, SparseWeightPatch(
        name=name,
        indices=new_flat.to(torch.int32),
        values=values.index_select(0, selected_positions),
    )


def remap_patches_for_operation(
    patches: list[SparseWeightPatch],
    train_shapes: dict[str, tuple[int, ...]],
    infer_shapes: dict[str, tuple[int, ...]],
    operations: list,
) -> list[SparseWeightPatch]:
    """Remap all patches for a set of CommunicationOperations.

    Args:
        patches: sparse patches from delta detection (in training shard space)
        train_shapes: {param_name: train_shard_shape}
        infer_shapes: {param_name: infer_shard_shape}
        operations: list of CommunicationOperation defining overlaps

    Returns:
        List of remapped patches for the target inference rank.
    """
    patch_map = {p.name: p for p in patches}
    remapped = []

    for op in operations:
        name = op.send_shard_meta.name
        if name not in patch_map:
            continue
        if name not in train_shapes:
            continue

        result = remap_delta_indices(
            patch=patch_map[name],
            train_shape=train_shapes[name],
            train_slices=op.train_slices,
            inf_slices=op.inf_slices,
            infer_shape=infer_shapes.get(name, train_shapes[name]),
        )
        if result is not None and result.num_updates > 0:
            remapped.append(result)

    return remapped


def _unravel_index(
    flat_indices: torch.Tensor, shape: tuple[int, ...]
) -> tuple[torch.Tensor, ...]:
    """Convert flat indices to multi-dimensional indices."""
    result = []
    remaining = flat_indices
    for dim in range(len(shape) - 1):
        stride = 1
        for s in shape[dim + 1 :]:
            stride *= s
        dim_idx = remaining // stride
        remaining = remaining % stride
        result.append(dim_idx)
    result.append(remaining)
    return tuple(result)


def _ravel_multi_index(
    multi_idx: list[torch.Tensor], shape: tuple[int, ...]
) -> torch.Tensor:
    """Convert multi-dimensional indices to flat indices."""
    flat = torch.zeros_like(multi_idx[0], dtype=torch.long)
    stride = 1
    for dim in range(len(shape) - 1, -1, -1):
        flat += multi_idx[dim] * stride
        stride *= shape[dim]
    return flat


# int32 flat index ceiling: a single inference shard must stay below 2**31
# elements, otherwise the int32 patch indices overflow.
_MAX_INT32_NUMEL = 2**31


def _assert_unit_step(s: slice, dim: int, name: str) -> None:
    """Remap math assumes contiguous (step==1) slices. AWEX transfer plans only
    build ``slice(start, stop)`` (step is None == 1); a strided slice would
    silently miscompute the overlap, so fail loud instead."""
    if s.step not in (None, 1):
        raise NotImplementedError(
            f"remap_delta_indices: strided slice step={s.step} on dim {dim} of "
            f"'{name}' is unsupported (transfer plans are expected contiguous)."
        )


def _assert_int32_safe(flat: torch.Tensor, shape: tuple[int, ...], name: str) -> None:
    numel = 1
    for d in shape:
        numel *= d
    if numel >= _MAX_INT32_NUMEL:
        raise ValueError(
            f"remap_delta_indices: inference shard '{name}' has {numel} elements "
            f">= 2**31; int32 flat indices overflow. Fall back to dense for this param."
        )


def remap_mask_for_op(
    name: str,
    changed_mask: torch.Tensor,
    values_source: torch.Tensor,
    train_shape: tuple[int, ...],
    op,
) -> SparseWeightPatch | None:
    """Per-op entry point for the transport layer.

    Computes the sparse patch for a single CommunicationOperation directly from
    a boolean change mask, without first materializing a full-shard patch. This
    is what the P2P send path calls once per op: the same param's mask is shared
    across all ops (computed once), each op projects its own overlap sub-region
    into inference-shard index space.

    Args:
        name: HF parameter name (``op.send_shard_meta.name``).
        changed_mask: bool tensor, same numel/shape as the train-shard param,
            True where the bf16 element changed this step.
        values_source: the train-shard param tensor (new values are gathered
            from it at the changed positions).
        train_shape: shape of the train-shard tensor.
        op: CommunicationOperation, provides ``train_slices`` / ``inf_slices``
            and ``recv_shard_meta.shape`` (inference-shard shape).

    Returns:
        SparseWeightPatch with indices in the inference-shard flat space and the
        gathered values, or None if no changed element falls in this op's
        overlap region.
    """
    flat_mask = changed_mask.reshape(-1)
    idx = flat_mask.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return None
    vals = values_source.reshape(-1).index_select(0, idx)
    patch = SparseWeightPatch(name=name, indices=idx.to(torch.int32), values=vals)
    return remap_delta_indices(
        patch=patch,
        train_shape=tuple(train_shape),
        train_slices=op.train_slices,
        inf_slices=op.inf_slices,
        infer_shape=tuple(op.recv_shard_meta.shape),
    )
