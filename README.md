# AReaL-DTE

Delta Transfer Engine (`dte`) provides incremental weight synchronization for
distributed reinforcement-learning training and inference.

[English](README.md) | [简体中文](README.zh-CN.md)

> Sparse, versioned weight deltas for RL training -> rollout sync, even when training
> and inference use different TP/PP/EP shard layouts.

`dte` is an incremental weight-transfer engine for online RL training. It keeps the
rollout side in sync with the trainer by sending full weights only when a new base is
needed, then sending sparse patches for later steps.

## Why dte

Dense weight sync is easy to reason about, but it burns bandwidth every RL step. For
bf16 weights, a sparse delta element costs 6 bytes: 4 bytes for the flat index and 2
bytes for the value. If 2% of a tensor changes, the raw payload is about 6% of the dense
tensor before transport overhead. Less payload usually means shorter trainer-to-rollout
handoff: the rollout side can start using the new policy sooner, and the RL loop spends
less time waiting on weight sync. `dte` adds the machinery needed to make that safe in
distributed training:

- **Faster weight handoff.** Normal steps move sparse deltas instead of full tensors.
  When a tensor is not sparse enough, `dte` falls back to dense for that tensor, so the
  fast path does not become a penalty.
- **Less data on the wire.** For bf16, 2% changed elements produce a raw sparse payload
  around 6% of dense size. The exact speedup depends on backend, topology, and scatter
  cost, but the transfer work scales with changed elements instead of total parameters
  on sparse steps.
- **Bitwise reconstruction.** Change detection compares integer views of the
  floating-point storage. NaN payload bits and signed zero are preserved.
- **Shard-aware sparse transfer.** Sparse indices are remapped through `train_slices`
  and `inf_slices`, so deltas can cross TP/PP/EP layout mismatches instead of assuming
  aligned shards.
- **Version-safe apply.** Every delta carries `base_version` and `payload_version`; the
  receiver rejects a patch if it no longer has the right base.
- **Backend isolation.** The delta algorithm is independent of NCCL, RDMA, awex,
  Mooncake, and runtime process management. Backends move bytes; `dte` defines what
  those bytes mean.

The transfer model is short:

1. Send a full checkpoint once to create a receiver-side base.
1. For later steps, detect which weight elements changed.
1. Encode the changed flat indices and new values.
1. Remap those indices when training and inference shards do not line up.
1. Apply the patch only if the receiver still has the exact base version.

## Architecture

<p align="center">
  <img src="./docs/images/dte_architecture.svg" alt="Delta Transfer Engine architecture" width="100%">
</p>

Diagram source: [docs/images/dte_architecture.svg](docs/images/dte_architecture.svg)

The diagram keeps only the main boundary: runtimes own model state, DTE owns delta
semantics, and backends own byte movement. The smaller modules are:

| Module                                       | Role                                                                                                                                                                                                        |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DeltaEngine`                                | Public orchestration layer. On the sender it chooses full vs delta, calls the tracker, and sends payloads. On the receiver it checks the version chain and writes reconstructed tensors into target params. |
| `DeltaTracker`                               | Sender-side delta state. It seeds a base after full sync, computes bitwise masks from a snapshot or accepts external masks, and emits sparse/dense/unchanged entries.                                       |
| Delta codec                                  | Flat named-tensor protocol: `__awex_delta_header__`, `w@delta_idx`, `w@delta_val`, and dense fallback `w`. This keeps the payload compatible with existing tensor grouping and IPC paths.                   |
| `reconstruct_against_base`                   | Receiver-side reconstruction. It applies sparse patches on top of the stored base, adopts dense fallback tensors, and refreshes the base for the next version.                                              |
| `remap_delta_indices`                        | Converts sparse flat indices from a training shard into the corresponding inference-shard indices using `train_slices` and `inf_slices`.                                                                    |
| `delta_p2p` / `colocate_protocol`            | Two-round protocol for variable-length sparse P2P: first exchange `nnz`, then exchange `idx` and `val`. This keeps cross-rank scheduling symmetric.                                                         |
| `Transport`, `Plan`, `TransferOp`, `Payload` | Backend contract. `Plan` describes rank pairs and shard slices; `Payload` is what moves; `Transport` defines `build_plan`, `send`, and `recv`.                                                              |
| Backends                                     | `loopback` is the local CPU backend; `awex_backend` reuses awex reshard plans and NCCL scheduling; `mooncake_backend` provides the RDMA-oriented path.                                                      |

For a normal delta step the data path is:

```text
detect -> encode -> transport.send/recv -> decode -> reconstruct -> apply
```

With this split, bitwise diff, sparse encoding, version checks, and index remap can be
tested without a cluster. NCCL, RDMA, MetaServer, and runtime integration stay inside
backends.

## Algorithm Overview

`dte` treats weight sync as a versioned state transition:

```text
theta_v --Delta(v -> v+1)--> theta_{v+1}
```

The receiver stores a full base `{name: tensor}` for version `v`. A delta payload is
valid only if its header says `base_version == v`; otherwise the receiver rejects it and
the caller must run a full sync. This is the main correctness guard: a sparse patch is
not a complete model state.

### 1. Detect changed elements

For each parameter `w`, the sender builds a change set:

```text
C_w = { i | bits(theta_t[w][i]) != bits(theta_base[w][i]) }
```

The comparison is bitwise, not floating-point `!=`. `NaN` payload bits, signed zero,
bf16 rounding artifacts, and fp32 router weights are handled by comparing same-width
integer views of the tensor storage.

Delta mode accepts several change-detection inputs:

- **Snapshot diff:** keep a CPU copy of the previous transferred weights and compare
  current weights against it. This is the standalone `DeltaTracker` fallback when
  `encode(..., masks=None)` is used.
- **AdamW inversion mask:** reconstruct the pre-step weights from AdamW moments, then
  compare against the current weights. AReaL's DTE example launchers use this as their
  default delta configuration because it avoids a full extra snapshot. For decoupled
  AdamW:

```text
u_t = (lr / (1 - beta1^step)) * m_t / (sqrt(v_t / (1 - beta2^step)) + eps)
theta_{t-1} = (theta_t + u_t) / (1 - lr * weight_decay)
```

The inversion path lets callers avoid storing a full snapshot inside `DeltaTracker`;
they pass `{name: bool_mask}` into `encode(...)`.

- **External masks and indices:** callers may also pass boolean masks, sorted integer
  changed indices, or indices decoded from packed optimizer dirty bits.

`DeltaEngine.mode` remains either `"delta"` or `"full"`. Snapshot, AdamW inversion, and
dirty-bit detection are strategies used inside delta mode, not additional engine modes.
The AReaL integration recommends AdamW inversion for normal runs and keeps snapshot diff
available for fallback and verification, including MoE validation.

### 2. Choose sparse or dense per tensor

For a tensor with `n` elements, element size `s`, and `k = |C_w|` changed elements:

```text
dense_bytes(w)  = n * s
sparse_bytes(w) = k * (4 + s)       # int32 flat index + value
```

`DeltaTracker` emits sparse entries only when:

```text
k > 0
n < 2^31
sparse_bytes(w) <= sparse_bytes_ratio * dense_bytes(w)
```

Otherwise the tensor is sent dense for that step. This makes a payload heterogeneous by
design:

```text
__awex_delta_header__ -> [magic, codec, payload_version, base_version, counts]
w@delta_idx           -> int32[k]
w@delta_val           -> dtype[k]
w                     -> dense fallback tensor
```

Unchanged tensors are omitted. Dense fallback tensors refresh the receiver base just
like a full sync for that tensor.

### 3. Remap sparse indices across shard layouts

Sparse indices are computed in the training shard's flat index space. If the rollout
side uses a different TP/PP/EP layout, those indices must be projected through the
transfer plan.

For each `TransferOp`, `dte` uses the source and destination overlap:

```text
train_slices = op.train_slices
inf_slices   = op.inf_slices
```

The remap is:

```text
p  = training-shard flat index
c  = unravel(p, train_shape)
keep c only if c is inside train_slices
c' = c - train_start + inf_start
p' = ravel(c', infer_shape)
```

The value tensor is filtered with the same mask, so the receiver scatters `values[j]`
into `p'[j]` in its own inference shard. The remap function is pure tensor geometry;
NCCL, RDMA, and IPC are still backend concerns.

### 4. Reconstruct on the receiver

After transport, the receiver decodes the payload and rebuilds full tensors against its
base:

```text
if tensor is dense:
    full = payload[name]
elif tensor is sparse:
    full = clone(base[name])
    full[delta_idx] = delta_val
else:
    full = clone(base[name])
```

The reconstructed tensors are written into live inference parameters, and the CPU base
is refreshed to `payload_version`. In colocate NCCL paths, variable-size sparse patches
use a two-round protocol: first exchange `nnz` for every op, then exchange `(idx, val)`
buffers. Zero-`nnz` ops still participate so every rank walks the same schedule.

## Repository Layout

- `src/dte/core/`: delta algorithms: bitwise diff, AdamW inversion helper, sparse codec,
  reconstruction, shard index remap, and colocate P2P protocol logic.
- `src/dte/engine.py`: `DeltaEngine`, the sender/receiver control flow.
- `src/dte/transport.py`: backend-neutral `Transport`, `Plan`, `TransferOp`, and
  `Payload` types.
- `src/dte/backends/loopback.py`: CPU in-memory backend used by tests.
- `src/dte/backends/awex_backend.py`: awex adapter for transfer plans and NCCL
  scheduling.
- `src/dte/backends/mooncake_backend.py`: Mooncake/RDMA-oriented packing path and
  transport interface.
- `src/dte/backends/http_backend.py`: staged shared-filesystem or S3-compatible blob
  transport with atomic manifests, checksums, and bounded-memory streaming.
- `docs/design.md`: full design, equations, protocol invariants, and validation notes.
- `docs/awex-gpu-verification.md`: GPU parity checklist for `AwexTransport`.

## Install

Requirements:

- Python >= 3.11 and \< 3.13
- PyTorch >= 2.9.1 and \< 2.11 on the primary Linux stack

Install from a source checkout:

```bash
python -m pip install -e .
```

For development:

```bash
python -m pip install -e ".[dev]"
```

Optional transport backends:

```bash
python -m pip install -e ".[awex]"      # dingzhiqiang/asystem-awex + NCCL runtime
python -m pip install -e ".[mooncake]"  # Mooncake Transfer Engine
python -m pip install -e ".[http]"      # Shared-filesystem staged transport
python -m pip install -e ".[http,s3]"   # Add the S3-compatible store provider
python -m pip install -e ".[http,oss]"  # Add the native Alibaba Cloud OSS provider
```

The default install is enough for the core algorithm and the loopback backend. Cluster
backends need their runtime dependencies and hardware environment. After a PyPI release,
the same extras can be installed from `delta-transfer-engine[...]`.

## Quickstart

```python
from dte import DeltaEngine
from dte.backends import LoopbackTransport

engine = DeltaEngine(
    transport=LoopbackTransport(),
    mode="delta",
    anchor_interval=10,
)

# Trainer side.
engine.push(model.named_parameters(), version=step)

# Rollout side.
engine.pull(target_params, version=step)
```

`mode="delta"` seeds with a full sync, uses sparse deltas for normal steps, and can
force periodic full anchors. `mode="full"` skips detection and sends dense weights every
step. The standalone engine uses snapshot diff unless an integration supplies external
masks; the AReaL examples supply AdamW-inversion masks by default.

For processes that do not share a live communication fabric, the HTTP backend stages
each version as immutable blobs on a shared filesystem or S3-compatible object store:

```python
from dte import DeltaEngine
from dte.backends import HttpTransport, SharedFSStore

store = SharedFSStore("/mnt/weights")

# Trainer process. Full payloads are headerless, so begin() supplies their version.
sender = DeltaEngine(HttpTransport(store, stream="run-1"))
sender.transport.begin(step)
sender.push(model.named_parameters(), version=step)

# Rollout process on any host that can read the same store.
receiver = DeltaEngine(HttpTransport(store, stream="run-1"))
receiver.pull(target_params, version=step)
```

Payloads use zstd-compressed safetensors with file- and tensor-level checksums. A
manifest commits the complete version atomically; chunked publish/fetch and
`DeltaEngine.reconstruct_stream` keep peak staging memory bounded for large models.

Keep sender and receiver instances alive across steps. The example uses a single writer
and assumes the caller waits for publication and consumes each version in order.
`recv()` polls the latest version once; it does not replay skipped deltas. For multiple
writers, use `publish` / `write_manifest` / `iter_fetch`. The application owns shard
routing; one `reconstruct_stream` requires unique parameter names across writers. Early
writer-marker reads still require waiting for the final manifest before exposing the new
model to inference. New anchors prune older versions, so readers racing retention must
reload the current anchor.

For native OSS, inject an initialized `oss2.Bucket` into `OSSStore`. Authentication,
region, endpoint, credentials, and SDK connection settings remain application-owned:

```python
from dte.backends import HttpTransport, OSSStore

# oss_bucket is an application-configured oss2.Bucket with V2 or V4 authentication.
store = OSSStore(bucket_client=oss_bucket, prefix="dte/run-1")
transport = HttpTransport(store, stream="policy", prune_on_anchor=False)
```

`prune_on_anchor=False` retains older versions for application-managed cleanup. The
default remains `True` for compatibility. Native OSS reads and writes use memory
buffers, with no local staging files. Blobs of at least 64 MiB use sequential multipart
uploads (configurable); multipart does not remove the need to hold the packed blob in
memory. Incomplete multipart uploads require separate cleanup; a lost completion
response does not imply that the object was not committed. Credentials and
deployment-specific endpoints never belong in code.

OSS GET (including response-body reads), HEAD, and listing pages retry transient
connection/time-out errors and HTTP 429/500/502/503/504 up to four attempts by default.
Backoff starts at 0.5 seconds and doubles, capped at 8 seconds; configure
`read_attempts` and `read_backoff` on `OSSStore`. Attempt counts include the initial
request; setting an attempt count to one disables retries for that operation class.
Failed GET responses are closed and retried from the beginning. Authentication, missing
objects, TLS verification, and checksum errors remain failures. Uploads also retry
transient failures (controlled by `write_attempts`, default four, using the same
`read_backoff` setting). PUT and individual parts replay identical bytes. Ambiguous
multipart completion is accepted only after reading back identical object bytes;
initialization retries can leave empty upload sessions for separate cleanup.
Applications must serialize updates to mutable keys such as `latest.json`. DELETE is not
automatically retried.

## Development

```bash
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
mdformat --check README.md README.zh-CN.md CONTRIBUTING.md docs
python -m pytest tests -q
python -m build
```

The CPU test suite covers the core algorithm, `DeltaEngine`, loopback transport, and
backend contracts that do not require a live cluster. GPU parity for awex is tracked
separately in `docs/awex-gpu-verification.md`.

## Current Status

| Area                 | State                                                                                                                                                                                            |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Core delta algorithm | CPU tests cover snapshot diff, external masks and sorted indices, packed dirty bits, sparse/dense fallback, tied-storage snapshot dedup, remap fast paths, AdamW inversion, and version chains.  |
| `DeltaEngine`        | CPU end-to-end tests cover full/delta sync, anchors, no-op deltas, broken chains, and transactional `decode_for_live_apply` / `commit_live_apply`.                                               |
| `loopback` backend   | Local test backend.                                                                                                                                                                              |
| `awex` backend       | Adapter uses `dingzhiqiang/asystem-awex`; the colocated sparse path supports coalesced two-round metadata/data exchange. Live cross-rank parity still needs NCCL + MetaServer + megatron/sglang. |
| `mooncake` backend   | Interface and pack/unpack path are present. RDMA execution needs a Mooncake runtime.                                                                                                             |
| `http` backend       | Shared-filesystem behavior, multi-writer manifests, chunk streaming, retention, corruption handling, and S3 contracts have CPU tests.                                                            |

## License

This project is licensed under the [Apache License 2.0](LICENSE).
