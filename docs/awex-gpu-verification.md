# M3 GPU Verification — AwexTransport bitwise parity

`AwexTransport` (`src/dte/backends/awex_backend.py`) is wired to awex's real signatures,
but its sparse cross-rank transfer needs a **live awex runtime**: NCCL process group,
MetaServer, and megatron/sglang reader state built by
`NCCLWorkerWeightsReader.initialize()`. None of that runs on CPU, so the parity proof
below is a **cluster task**, not a local unit test.

What IS proven locally (no GPU): `tests/test_awex_backend.py` — the contract
(AwexTransport is a `Transport`), graceful import degradation, and the `build_plan`
conversion of awex `CommunicationOperation` → dte `TransferOp`.

## Goal

Show that delta sync through **upstream (unmodified) awex + dte** produces the
**bitwise-identical** received weights as the fork's in-awex delta path
(`feat-delta-transfer`'s `apply_delta_colocate`). That proves the layering introduces
zero semantic drift — dte only relocated the algorithm, it didn't change it.

## Environment

- 2 GPUs (1 train rank → 1 infer rank minimum; ideally TP-mismatch e.g. TP4→TP2)
- Upstream awex installed (NOT the fork's delta branch): `pip install awex`
- `pip install -e .[awex]` for dte in the same env
- MetaServer reachable; megatron (train) + sglang (infer) as in a normal awex run

## Steps

1. **Baseline (fork, in-awex delta)** — run the verified fork path, dump the received
   inference weights at steps `{0 (seed), 1, 2, ... , anchor}`:

   - `AWEX_DELTA_TRANSFER=1 ANCHOR_INTERVAL=10` on `feat-delta-transfer`.
   - Save each step's `{name: tensor}` (e.g. `torch.save`) as the golden set.

1. **dte path (upstream awex)** — same model/config/seed, drive sync via
   `DeltaEngine(transport=AwexTransport(reader), mode="delta", anchor_interval=10)`
   where `reader` is the awex `NCCLWorkerWeightsReader` the run already builds. Dump the
   received weights at the same steps.

1. **Compare** — assert `torch.equal(golden[name], dte[name])` for every name at every
   step (bitwise, not allclose — delta is lossless by design). Any mismatch is a porting
   bug in `awex_backend.py`, not in `dte.core` (core is already proven by the 76
   migrated unit tests).

## Mode-switch parity (mirrors awex `AWEX_DELTA_TRANSFER`)

Also verify the full/delta switch end-to-end on GPU:

- `DeltaEngine(..., mode="full")` ≡ awex `AWEX_DELTA_TRANSFER=0` (pure dense, detector
  untouched). Received weights must equal a plain full sync.
- `DeltaEngine(..., mode="delta", anchor_interval=N)` ≡ awex
  `AWEX_DELTA_TRANSFER=1 + ANCHOR_INTERVAL=N`.

## Pass criteria

- All steps bitwise-equal to the fork golden set (delta mode).
- `mode="full"` equals a dense full sync; `mode="delta"` matches the anchored delta run.
- Upstream awex source is unmodified (no delta code committed to awex).

## Known open point

awex colocate transfer is **symmetric** (one `apply_delta_colocate` both sends and
receives); dte's `Transport.send`/`recv` split is bridged inside `AwexTransport` (send
stages, recv drives the one symmetric call). Validate that the staging/exchange bridge
preserves the deadlock-safe recursive-partition symmetry across ranks (every rank enters
every dtype group unconditionally).
