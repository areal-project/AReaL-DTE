# Contributing

Thanks for taking a look at Delta Transfer Engine. The core package is designed to be
testable on CPU; cluster backends have separate verification notes.

## Setup

```bash
python -m pip install -e ".[dev]"
```

## Checks

Run the same quality checks as CI before opening a PR:

```bash
ruff check .
ruff format --check .
mdformat --check README.md README.zh-CN.md CONTRIBUTING.md docs
python -m pytest tests -q
python -m build
```

If your change touches the awex backend, also update or run the GPU parity steps in
`docs/awex-gpu-verification.md`. CPU tests cannot prove NCCL process-group, MetaServer,
or colocate runtime behavior.

For staged HTTP transport changes, run the focused CPU suite as well:

```bash
python -m pytest tests/test_http_backend*.py tests/test_engine.py -q
```

Keep tests centered on transport contracts: publication visibility, writer-scoped
checksums, bounded fetching, retention, and reconstruction after partial failure. Prefer
shared scenarios over duplicate happy-path tests, and assert specific failure types
rather than accepting any exception. CPU fixtures do not validate live provider
authentication, network performance, or an inference engine's reload/ACK lifecycle.

Native OSS changes additionally require `python -m pytest tests/test_oss_store.py -q`.
These tests use an in-memory SDK boundary, without cloud credentials or live writes.
Provider tests cover SDK errors and resource handling; generic reconstruction and
retention scenarios belong in the shared HTTP/engine suite.

## Pull requests

- Keep algorithm changes in `src/dte/core` independent of transport backends unless the
  change is explicitly about a backend contract.
- Add focused tests for version-chain, reconstruction, or remap behavior when changing
  those paths.
- Document any new runtime assumption in `docs/design.md`.
- Changes to the core protocol and engine contracts require review from the owners in
  `.github/CODEOWNERS`.
