"""Contract tests for dte.transport (the interface dte owns). CPU-only."""

import pytest
import torch

from dte.transport import Payload, Plan, TransferOp, Transport


def test_transport_is_abstract():
    """Transport cannot be instantiated without implementing the contract."""
    with pytest.raises(TypeError):
        Transport()


def test_partial_impl_still_abstract():
    """Missing any of build_plan/send/recv keeps the class abstract."""

    class Half(Transport):
        def build_plan(self, train_meta, infer_meta):
            return Plan()

    with pytest.raises(TypeError):
        Half()


def test_minimal_backend_satisfies_contract():
    """A backend implementing the 3 abstract methods is instantiable and
    composes the dte-owned data structures."""

    class DummyTransport(Transport):
        def build_plan(self, train_meta, infer_meta):
            return Plan(ops=[TransferOp(0, 1, "w")])

        def send(self, plan, payloads):
            self.sent = payloads

        def recv(self, plan):
            return [Payload("w", torch.zeros(4))]

    t = DummyTransport()
    t.setup()
    plan = t.build_plan(None, None)
    assert plan.ops[0].param_name == "w"
    t.send(
        plan,
        [Payload("w", torch.ones(2), indices=torch.tensor([0, 3], dtype=torch.int32))],
    )
    out = t.recv(plan)
    t.teardown()
    assert out[0].name == "w"


def test_payload_full_vs_delta():
    full = Payload("w", torch.arange(6.0))
    assert not full.is_delta
    assert full.nnz == 6

    delta = Payload(
        "w",
        torch.tensor([1.0, 2.0]),
        indices=torch.tensor([0, 5], dtype=torch.int32),
    )
    assert delta.is_delta
    assert delta.nnz == 2


def test_plan_defaults_empty():
    p = Plan()
    assert p.ops == []
    assert p.backend_plan is None
