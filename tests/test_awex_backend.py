"""Local contract + graceful-degradation tests for AwexTransport.

These run on CPU with NO awex/GPU. They verify:
- import degrades gracefully when awex is absent;
- AwexTransport satisfies the dte Transport contract;
- build_plan converts an awex transfer_plan into dte Plan/TransferOp (pure data
  conversion, mockable);
- the symmetric-exchange impedance is handled (send stages, recv needs cluster).

The real bitwise parity vs the fork's in-awex delta path needs a 2-GPU cluster
and lives in docs/M3-gpu-verification.md — not here.
"""

from types import SimpleNamespace

import pytest

import dte.backends.awex_backend as ab
from dte.backends.awex_backend import AwexTransport, awex_available
from dte.transport import Payload, Plan, Transport


def _fake_comm_op(name, send_rank, recv_rank):
    """Mimic awex CommunicationOperation's attribute surface."""
    return SimpleNamespace(
        send_rank=send_rank,
        recv_rank=recv_rank,
        send_shard_meta=SimpleNamespace(name=name),
        train_slices=(slice(None),),
        inf_slices=(slice(None),),
    )


def _fake_reader():
    """A reader stub exposing only what build_plan reads on the no-builder path."""
    plan = SimpleNamespace(
        operations={
            1: [_fake_comm_op("layer.0.w", send_rank=2, recv_rank=1)],
            0: [_fake_comm_op("layer.1.w", send_rank=3, recv_rank=0)],
        }
    )
    return SimpleNamespace(transfer_rank=0, transfer_plan=plan)


def test_awex_available_reflects_env():
    # On this CPU dev box awex is not installed.
    assert awex_available() is False


def test_constructor_requires_awex_when_absent():
    if awex_available():
        pytest.skip("awex is installed in this environment")
    with pytest.raises(RuntimeError, match="awex"):
        AwexTransport(_fake_reader())


def test_is_transport_subclass():
    """Contract: AwexTransport implements the dte Transport interface."""
    assert issubclass(AwexTransport, Transport)
    for m in ("build_plan", "send", "recv"):
        assert hasattr(AwexTransport, m)


def test_build_plan_converts_awex_plan(monkeypatch):
    """build_plan maps awex CommunicationOperation -> dte TransferOp (no GPU)."""
    monkeypatch.setattr(ab, "_AWEX_AVAILABLE", True)
    t = AwexTransport(_fake_reader())  # world sizes None -> uses reader.transfer_plan

    plan = t.build_plan(train_meta=None, infer_meta=None)
    assert isinstance(plan, Plan)
    assert len(plan.ops) == 2
    names = {op.param_name for op in plan.ops}
    assert names == {"layer.0.w", "layer.1.w"}
    op0 = next(op for op in plan.ops if op.param_name == "layer.0.w")
    assert op0.send_rank == 2 and op0.recv_rank == 1
    assert op0.backend_op is not None  # native awex op carried through
    assert plan.backend_plan is not None


def test_send_stages_recv_needs_cluster(monkeypatch):
    """send only stages (awex transfer is symmetric); recv needs the cluster."""
    monkeypatch.setattr(ab, "_AWEX_AVAILABLE", True)
    t = AwexTransport(_fake_reader())
    plan = Plan()
    t.send(plan, [Payload("w", values=None)])  # staging is a pure assignment
    assert t._staged and t._staged[0].name == "w"
    with pytest.raises(NotImplementedError, match="cluster"):
        t.recv(plan)
