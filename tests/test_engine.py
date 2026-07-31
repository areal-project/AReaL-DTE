"""End-to-end integration tests for DeltaEngine over LoopbackTransport (CPU).

M2 ironclad proof: the dte layering (engine → core codec → transport → core
decode) is self-consistent and bitwise-correct, with zero awex/NCCL/GPU
dependency. Covers: full-sync seed, delta step, multi-step chain, and
version-chain break → full-sync recovery.
"""

import pytest
import torch

from dte.backends import LoopbackTransport
from dte.engine import DeltaEngine


def _params(values: dict[str, torch.Tensor]):
    return list(values.items())


def _clone(d):
    return {k: v.detach().clone() for k, v in d.items()}


def _make_pair(**kw):
    """A sender and a receiver engine sharing one loopback transport."""
    transport = LoopbackTransport()
    sender = DeltaEngine(transport, **kw)
    receiver = DeltaEngine(transport, **kw)
    return sender, receiver, transport


def test_full_sync_seed_then_exact():
    sender, receiver, _ = _make_pair()
    w = {
        "a": torch.randn(4, 8, dtype=torch.bfloat16),
        "b": torch.randn(16, dtype=torch.bfloat16),
    }
    target = {k: torch.zeros_like(v) for k, v in w.items()}

    res = sender.push(_params(w), version=0)
    assert res.full_sync and res.reason == "not_seeded"

    receiver.pull(target, version=0)
    for k in w:
        assert torch.equal(target[k], w[k]), f"{k} mismatch after full seed"
    assert receiver.base_version == 0


def test_delta_step_bitwise_equal():
    sender, receiver, _ = _make_pair()
    w = {"a": torch.randn(4, 8, dtype=torch.bfloat16)}
    target = {"a": torch.zeros(4, 8, dtype=torch.bfloat16)}

    # step 0: full seed
    sender.push(_params(w), 0)
    receiver.pull(target, 0)

    # step 1: change a few elements -> delta
    w2 = _clone(w)
    w2["a"].view(-1)[3] = 1.5
    w2["a"].view(-1)[10] = -2.25
    res = sender.push(_params(w2), 1)
    assert not res.full_sync, "should be a delta payload"

    receiver.pull(target, 1)
    assert torch.equal(target["a"], w2["a"]), "delta reconstruction not bitwise-equal"
    assert receiver.base_version == 1


def test_multi_step_chain():
    sender, receiver, _ = _make_pair()
    w = {
        "a": torch.randn(8, 8, dtype=torch.bfloat16),
        "b": torch.randn(32, dtype=torch.bfloat16),
    }
    target = {k: torch.zeros_like(v) for k, v in w.items()}

    sender.push(_params(w), 0)
    receiver.pull(target, 0)

    cur = _clone(w)
    for step in range(1, 6):
        # mutate a handful of elements each step
        cur["a"].view(-1)[step] = float(step)
        cur["b"].view(-1)[step * 2] = float(-step)
        sender.push(_params(cur), step)
        receiver.pull(target, step)
        for k in cur:
            assert torch.equal(target[k], cur[k]), f"step {step} {k} mismatch"


def test_no_change_step_is_noop_equal():
    sender, receiver, _ = _make_pair()
    w = {"a": torch.randn(4, 4, dtype=torch.bfloat16)}
    target = {"a": torch.zeros(4, 4, dtype=torch.bfloat16)}
    sender.push(_params(w), 0)
    receiver.pull(target, 0)

    # identical weights -> delta with zero changes
    sender.push(_params(_clone(w)), 1)
    receiver.pull(target, 1)
    assert torch.equal(target["a"], w["a"])


def test_anchor_interval_forces_full():
    sender, receiver, _ = _make_pair(anchor_interval=2)
    w = {"a": torch.randn(4, 4, dtype=torch.bfloat16)}
    target = {"a": torch.zeros(4, 4, dtype=torch.bfloat16)}

    r0 = sender.push(_params(w), 0)
    receiver.pull(target, 0)
    assert r0.full_sync  # not_seeded
    cur = _clone(w)
    cur["a"].view(-1)[0] = 9.0
    r1 = sender.push(_params(cur), 1)
    receiver.pull(target, 1)
    assert not r1.full_sync
    cur["a"].view(-1)[1] = 8.0
    r2 = sender.push(_params(cur), 2)
    receiver.pull(target, 2)
    assert not r2.full_sync
    # third consecutive delta would exceed anchor_interval=2 -> forced full
    cur["a"].view(-1)[2] = 7.0
    r3 = sender.push(_params(cur), 3)
    receiver.pull(target, 3)
    assert r3.full_sync and "anchor_interval" in r3.reason
    assert torch.equal(target["a"], cur["a"])


def test_version_chain_break_raises():
    """A delta whose base doesn't match the receiver's base must raise."""
    sender, receiver, _ = _make_pair()
    w = {"a": torch.randn(4, 4, dtype=torch.bfloat16)}
    target = {"a": torch.zeros(4, 4, dtype=torch.bfloat16)}

    sender.push(_params(w), 0)
    receiver.pull(target, 0)  # receiver base = 0

    # sender makes a delta for v1 (base 0)
    cur = _clone(w)
    cur["a"].view(-1)[0] = 3.0
    sender.push(_params(cur), 1)
    receiver.pull(target, 1)  # receiver base = 1

    # craft a stale delta: sender base is now 1, but force a payload that the
    # receiver (already at base 1) would accept only if bases line up. Simulate
    # a break by pushing a delta then dropping the receiver to an older base.
    receiver._base_version = 99  # corrupt the chain
    cur["a"].view(-1)[1] = 4.0
    sender.push(_params(cur), 2)
    with pytest.raises(RuntimeError, match="base mismatch"):
        receiver.pull(target, 2)


def test_delta_before_base_raises():
    """Receiving a delta with no prior full sync is a broken chain."""
    sender, receiver, _ = _make_pair()
    w = {"a": torch.randn(4, 4, dtype=torch.bfloat16)}
    target = {"a": torch.zeros(4, 4, dtype=torch.bfloat16)}
    # seed sender only; manually encode a delta and feed receiver with no base
    sender.push(_params(w), 0)
    receiver.pull(target, 0)
    receiver._base_version = None  # wipe receiver base
    cur = _clone(w)
    cur["a"].view(-1)[0] = 1.0
    sender.push(_params(cur), 1)
    with pytest.raises(RuntimeError, match="broken chain"):
        receiver.pull(target, 1)


# ---- mode switch: full vs delta (mirrors awex AWEX_DELTA_TRANSFER on/off) ----


def test_mode_full_every_step_is_full_no_detector():
    """mode='full': every push is a full sync; the snapshot is never seeded."""
    sender, receiver, _ = _make_pair(mode="full")
    w = {"a": torch.randn(4, 8, dtype=torch.bfloat16)}
    target = {"a": torch.zeros(4, 8, dtype=torch.bfloat16)}

    for step in range(3):
        if step:
            w["a"].view(-1)[step] = float(step)
        res = sender.push(_params(w), step)
        receiver.pull(target, step)
        assert res.full_sync and res.reason == "full_mode"
        assert torch.equal(target["a"], w["a"]), f"step {step} not exact in full mode"
    # detector/snapshot untouched in full mode
    assert not sender.tracker.seeded


def test_mode_invalid_raises():
    with pytest.raises(ValueError, match="mode must be"):
        DeltaEngine(LoopbackTransport(), mode="bogus")


def test_default_mode_is_delta():
    sender, receiver, _ = _make_pair()
    assert sender.mode == "delta"
    w = {"a": torch.randn(4, 4, dtype=torch.bfloat16)}
    target = {"a": torch.zeros(4, 4, dtype=torch.bfloat16)}
    sender.push(_params(w), 0)
    receiver.pull(target, 0)
    w2 = _clone(w)
    w2["a"].view(-1)[0] = 5.0
    res = sender.push(_params(w2), 1)
    receiver.pull(target, 1)
    assert not res.full_sync  # delta by default after seed
    assert torch.equal(target["a"], w2["a"])


# ---- review fixes: real chain-break, stale full-sync, reserved names ----


def test_missing_intermediate_delta_raises():
    """Real desync (not a hand-poked field): receiver skips v2 and pulls v3's
    delta directly. v3 has base_version=2 but receiver base is still 1 -> raise."""
    sender, receiver, transport = _make_pair()
    w = {"a": torch.randn(4, 4, dtype=torch.bfloat16)}
    target = {"a": torch.zeros(4, 4, dtype=torch.bfloat16)}

    sender.push(_params(w), 0)
    receiver.pull(target, 0)  # both at base 0

    cur = _clone(w)
    cur["a"].view(-1)[0] = 1.0
    sender.push(_params(cur), 1)
    receiver.pull(target, 1)  # both at base 1

    # sender advances to v2 (delta base=1 -> payload_version 2) but the receiver
    # NEVER pulls v2 (e.g. it was dropped). transport buffer now holds v2.
    cur["a"].view(-1)[1] = 2.0
    sender.push(_params(cur), 2)
    # receiver, still at base 1, instead receives the NEXT delta (v3, base=2).
    # Overwrite the staged payload with v3 to simulate v2 lost in flight.
    cur["a"].view(-1)[2] = 3.0
    sender.push(_params(cur), 3)  # v3 delta has base_version=2
    with pytest.raises(RuntimeError, match="base mismatch"):
        receiver.pull(target, 3)  # receiver base=1, payload base=2 -> chain break


def test_stale_full_sync_version_raises():
    """A full sync must not move the receiver's version backwards (H1 guard)."""
    sender, receiver, _ = _make_pair(mode="full")
    w = {"a": torch.randn(4, 4, dtype=torch.bfloat16)}
    target = {"a": torch.zeros(4, 4, dtype=torch.bfloat16)}
    sender.push(_params(w), 5)
    receiver.pull(target, 5)  # base = 5
    sender.push(_params(w), 3)  # stale/reordered full sync
    with pytest.raises(RuntimeError, match="older than"):
        receiver.pull(target, 3)


def test_reserved_param_name_rejected():
    """A parameter colliding with the delta protocol namespace is rejected."""
    sender, _, _ = _make_pair()
    bad = {"layer@delta_idx": torch.randn(4, dtype=torch.bfloat16)}
    # full seed path goes through tracker.seed -> _check_reserved_names
    with pytest.raises(ValueError, match="reserved delta-protocol"):
        sender.push(_params(bad), 0)


# ---------------------------------------------------------------------------
# DeltaEngine.reconstruct (receiver-only, IPC-fed; used by the awex colocate
# reader in Round-C/C2). Verifies the full reconstruction + mask semantics are
# byte-identical to the old awex _maybe_reconstruct_delta logic.
# ---------------------------------------------------------------------------
from dte.engine import DeltaChainBroken  # noqa: E402
from dte.transport import Plan  # noqa: E402


def _wire(sender, transport, w, version):
    """push weights through loopback, return the wire named_tensors dict."""
    sender.push(_params(w), version)
    payloads = transport.recv(Plan())
    return {p.name: p.values for p in payloads}


def test_reconstruct_full_sync_returns_none_masks():
    sender, _, transport = _make_pair()
    recv = DeltaEngine(transport=None)  # receiver-only, no transport needed
    w = {
        "a": torch.randn(4, 8, dtype=torch.bfloat16),
        "b": torch.randn(16, dtype=torch.bfloat16),
    }
    named = _wire(sender, transport, w, 0)

    full, masks = recv.reconstruct(named, 0)
    assert masks is None, "full-sync must return None masks (-> dense apply path)"
    assert recv.base_version == 0
    for k in w:
        assert torch.equal(full[k], w[k])


def test_decode_for_live_apply_keeps_no_cpu_base():
    sender, _, transport = _make_pair()
    recv = DeltaEngine(transport=None)
    w = {"a": torch.randn(4, 8, dtype=torch.bfloat16)}

    named = _wire(sender, transport, w, 0)
    assert recv.decode_for_live_apply(named, 0) is None
    assert recv.base_version == 0
    assert recv._base == {}

    w2 = _clone(w)
    w2["a"].view(-1)[3] = 1.5
    decoded = recv.decode_for_live_apply(_wire(sender, transport, w2, 1), 1)
    assert decoded is not None
    assert decoded.header.base_version == 0
    assert recv.base_version == 0
    recv.commit_live_apply(decoded)
    assert recv.base_version == 1
    assert recv._base == {}


def test_reconstruct_delta_full_and_mask_semantics():
    sender, _, transport = _make_pair()
    recv = DeltaEngine(transport=None)
    w = {
        "a": torch.randn(4, 8, dtype=torch.bfloat16),
        "b": torch.randn(16, dtype=torch.bfloat16),
    }  # b will stay unchanged
    # seed base
    recv.reconstruct(_wire(sender, transport, w, 0), 0)

    # change two elements of "a" only; "b" untouched
    w2 = _clone(w)
    w2["a"].view(-1)[3] = 1.5
    w2["a"].view(-1)[10] = -2.25
    named = _wire(sender, transport, w2, 1)
    full, masks = recv.reconstruct(named, 1)

    # full reconstruction bitwise-equal
    assert torch.equal(full["a"], w2["a"])
    assert torch.equal(full["b"], w2["b"])
    assert recv.base_version == 1

    # mask semantics (byte-identical to awex): sparse param -> True exactly at
    # changed flat positions; UNCHANGED param -> NO entry (key absent).
    assert "a" in masks
    assert masks["a"].shape == w2["a"].shape and masks["a"].dtype == torch.bool
    flat = masks["a"].view(-1)
    assert flat[3].item() and flat[10].item()
    assert flat.sum().item() == 2, "only the 2 changed positions are True"
    assert "b" not in masks, "unchanged param must have NO mask entry"


def test_reconstruct_delta_before_base_raises_chain_broken():
    sender, _, transport = _make_pair()
    recv = DeltaEngine(transport=None)
    w = {"a": torch.randn(4, dtype=torch.bfloat16)}
    recv.reconstruct(_wire(sender, transport, w, 0), 0)  # seed
    w2 = _clone(w)
    w2["a"][1] = 9.0
    named = _wire(sender, transport, w2, 1)

    fresh = DeltaEngine(transport=None)  # no base
    with pytest.raises(DeltaChainBroken):
        fresh.reconstruct(named, 1)
