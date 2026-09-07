# Licensed under the Apache License, Version 2.0
"""http backend tests — staged blob transport over shared fs / object stores."""

from __future__ import annotations

import pytest
import torch

from dte.backends.http_backend import (
    ChecksumMismatch,
    ManifestMissing,
    SharedFSStore,
    compute_checksum,
    pack_payloads,
    unpack_payloads,
)
from dte.transport import Payload


def test_shared_fs_store_roundtrip(tmp_path):
    store = SharedFSStore(str(tmp_path))
    store.put_bytes("a/b/c.bin", b"hello")
    assert store.get_bytes("a/b/c.bin") == b"hello"
    assert store.exists("a/b/c.bin")
    assert not store.exists("a/b/missing.bin")
    assert store.list_keys("a/b/") == ["a/b/c.bin"]


def test_shared_fs_store_overwrite_and_no_tmp_leftovers(tmp_path):
    store = SharedFSStore(str(tmp_path))
    store.put_bytes("k.bin", b"v1")
    store.put_bytes("k.bin", b"v2")
    assert store.get_bytes("k.bin") == b"v2"
    leftovers = [p.name for p in tmp_path.rglob("*") if ".tmp" in p.name]
    assert leftovers == []


def test_shared_fs_store_get_missing_raises(tmp_path):
    store = SharedFSStore(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        store.get_bytes("nope.bin")


def test_shared_fs_store_list_keys_sorted_and_prefix_scoped(tmp_path):
    store = SharedFSStore(str(tmp_path))
    store.put_bytes("s/deltas/v000002/p.bin", b"2")
    store.put_bytes("s/deltas/v000001/p.bin", b"1")
    store.put_bytes("other/x.bin", b"x")
    assert store.list_keys("s/deltas/") == [
        "s/deltas/v000001/p.bin",
        "s/deltas/v000002/p.bin",
    ]


@pytest.mark.parametrize("key", ["../escape", "/absolute", "a/../b", "a//b", "a\\b"])
def test_shared_fs_store_rejects_unsafe_keys(tmp_path, key):
    store = SharedFSStore(str(tmp_path))

    with pytest.raises(ValueError, match="invalid blob key"):
        store.put_bytes(key, b"data")


# ---------------------------------------------------------------------------
# Task 2: payload pack/unpack (safetensors + zstd + per-tensor checksums)
# ---------------------------------------------------------------------------


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    return torch.equal(
        a.view(torch.int16) if a.dtype == torch.bfloat16 else a,
        b.view(torch.int16) if b.dtype == torch.bfloat16 else b,
    )


def _payloads():
    return [
        Payload("w1", torch.arange(8, dtype=torch.bfloat16)),
        Payload("w2@delta_idx", torch.tensor([1, 5], dtype=torch.int32)),
        Payload("w2@delta_val", torch.tensor([9.0, 7.0], dtype=torch.bfloat16)),
    ]


def test_pack_unpack_roundtrip_bitwise():
    data, sums = pack_payloads(_payloads(), checksum="adler32")
    assert isinstance(data, bytes) and len(data) > 0
    assert set(sums) == {"w1", "w2@delta_idx", "w2@delta_val"}
    out = unpack_payloads(data, sums, checksum="adler32")
    got = {p.name: p.values for p in out}
    for p in _payloads():
        assert _bitwise_equal(got[p.name], p.values)


def test_pack_unpack_2d_tensor_preserves_shape():
    payloads = [Payload("m", torch.randn(3, 4, dtype=torch.float32))]
    data, sums = pack_payloads(payloads, checksum="adler32")
    (out,) = unpack_payloads(data, sums, checksum="adler32")
    assert out.values.shape == (3, 4)
    assert torch.equal(out.values, payloads[0].values)


def test_unpack_detects_payload_corruption():
    import zstandard

    data, sums = pack_payloads(_payloads(), checksum="adler32")
    bad = bytearray(data)
    bad[len(bad) // 2] ^= 0xFF
    with pytest.raises((ChecksumMismatch, zstandard.ZstdError, Exception)):
        unpack_payloads(bytes(bad), sums, checksum="adler32")


def test_unpack_detects_wrong_checksum_map():
    data, sums = pack_payloads(_payloads(), checksum="adler32")
    tampered = dict(sums)
    tampered["w1"] = "00000000"
    with pytest.raises(ChecksumMismatch):
        unpack_payloads(data, tampered, checksum="adler32")


def test_compute_checksum_algorithms():
    buf = b"delta-transfer-engine"
    adler = compute_checksum("adler32", buf)
    assert adler == compute_checksum("adler32", buf)  # deterministic
    assert len(adler) == 8
    xxh = compute_checksum("xxh3-128", buf)
    assert len(xxh) == 32 and xxh != adler
    with pytest.raises(KeyError):
        compute_checksum("md5-nope", buf)


# ---------------------------------------------------------------------------
# Task 3: HttpTransport — staged publish/fetch + manifest chain + engine compat
# ---------------------------------------------------------------------------


def _mk(tmp_path, **kw):
    from dte.backends.http_backend import HttpTransport

    return HttpTransport(
        SharedFSStore(str(tmp_path)), stream="s", checksum="adler32", **kw
    )


def test_publish_anchor_then_fetch_roundtrip(tmp_path):
    t = _mk(tmp_path)
    res = t.publish(1, [Payload("w", torch.ones(4, dtype=torch.bfloat16))], anchor=True)
    assert (res.kind, res.version) == ("anchor", 1)
    assert res.wire_bytes > 0 and res.num_tensors == 1
    latest = t.poll_latest()
    assert latest["version"] == 1 and latest["kind"] == "anchor"
    out = t.fetch(1, kind="anchor")
    assert torch.equal(out[0].values.float(), torch.ones(4))


def test_poll_latest_on_empty_store_is_none(tmp_path):
    assert _mk(tmp_path).poll_latest() is None


def test_fetch_missing_version_raises(tmp_path):
    from dte.backends.http_backend import ManifestMissing

    with pytest.raises(ManifestMissing):
        _mk(tmp_path).fetch(7, kind="delta")


def test_manifest_schema_fields(tmp_path):
    import json

    t = _mk(tmp_path)
    t.publish(3, [Payload("w", torch.zeros(2, dtype=torch.float32))], anchor=True)
    store = SharedFSStore(str(tmp_path))
    m = json.loads(store.get_bytes("s/anchors/v000003/manifest.json"))
    assert m["schema"] == 1
    assert m["version"] == 3 and m["base_version"] is None
    assert m["kind"] == "anchor" and m["compression"] == "zstd"
    assert m["checksum_format"] == "adler32" and m["num_writers"] == 1
    writer = m["writers"][0]
    assert writer["writer_rank"] == 0
    assert "payload-00000-0000.st.zst" in writer["files"]
    f = writer["files"]["payload-00000-0000.st.zst"]
    assert f["bytes"] > 0 and len(f["checksum"]) == 8
    assert "w" in writer["tensor_checksums"]
    latest = json.loads(store.get_bytes("s/latest.json"))
    assert latest == {
        "schema": 1,
        "version": 3,
        "kind": "anchor",
        "dir": "anchors/v000003",
    }


def test_file_level_corruption_detected_on_fetch(tmp_path):
    from dte.backends.http_backend import ChecksumMismatch as CM

    t = _mk(tmp_path)
    t.publish(1, [Payload("w", torch.ones(8, dtype=torch.float32))], anchor=True)
    key = "s/anchors/v000001/payload-00000-0000.st.zst"
    store = SharedFSStore(str(tmp_path))
    blob = bytearray(store.get_bytes(key))
    blob[len(blob) // 2] ^= 0xFF
    store.put_bytes(key, bytes(blob))
    with pytest.raises(CM):
        t.fetch(1, kind="anchor")


def test_multi_writer_publish_single_manifest(tmp_path):
    w0 = _mk(tmp_path, writer_rank=0, num_writers=2)
    w1 = _mk(tmp_path, writer_rank=1, num_writers=2)
    m1 = w1.publish(1, [Payload("b", torch.ones(2, dtype=torch.bfloat16))], anchor=True)
    m0 = w0.publish(1, [Payload("a", torch.ones(2, dtype=torch.bfloat16))], anchor=True)
    assert (
        w0.poll_latest() is None
    )  # multi-writer: nothing visible before write_manifest
    w0.write_manifest(1, anchor=True, per_writer=[m0.writer_meta, m1.writer_meta])
    assert w0.poll_latest()["version"] == 1
    assert {p.name for p in w0.fetch(1, kind="anchor")} == {"a", "b"}


def test_multi_writer_same_tensor_name_keeps_checksums_scoped(tmp_path):
    w0 = _mk(tmp_path, writer_rank=0, num_writers=2)
    w1 = _mk(tmp_path, writer_rank=1, num_writers=2)
    m0 = w0.publish(1, [Payload("w", torch.tensor([1.0]))], anchor=True)
    m1 = w1.publish(1, [Payload("w", torch.tensor([2.0]))], anchor=True)
    w0.write_manifest(1, anchor=True, per_writer=[m0.writer_meta, m1.writer_meta])

    values = [p.values.item() for p in w0.fetch(1, kind="anchor")]

    assert values == [1.0, 2.0]


def test_manifest_requires_every_writer_once(tmp_path):
    w0 = _mk(tmp_path, writer_rank=0, num_writers=2)
    meta = w0.publish(1, [Payload("w", torch.tensor([1.0]))], anchor=True)

    with pytest.raises(ValueError, match="expected metadata from 2 writers"):
        w0.write_manifest(1, anchor=True, per_writer=[meta.writer_meta])


def test_iter_fetch_reads_writer_files_in_parallel_and_yields_in_order(
    tmp_path, monkeypatch
):
    import threading
    import time

    writers = [_mk(tmp_path, writer_rank=i, num_writers=4) for i in range(4)]
    metas = [
        writer.publish(
            1,
            [Payload(chr(ord("a") + i), torch.ones(2, dtype=torch.bfloat16))],
            anchor=True,
        ).writer_meta
        for i, writer in enumerate(writers)
    ]
    writers[0].write_manifest(1, anchor=True, per_writer=metas)

    store = writers[0].store
    original_get = store.get_bytes
    state = {"active": 0, "max_active": 0}
    lock = threading.Lock()

    def slow_get(key):
        if "payload-" in key:
            with lock:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
            time.sleep(0.05)
            try:
                return original_get(key)
            finally:
                with lock:
                    state["active"] -= 1
        return original_get(key)

    monkeypatch.setattr(store, "get_bytes", slow_get)
    monkeypatch.setenv("DTE_HTTP_FETCH_WORKERS", "4")

    chunks = list(writers[0].iter_fetch(1, kind="anchor"))

    assert state["max_active"] > 1
    assert [chunk[0].name for chunk in chunks] == ["a", "b", "c", "d"]


def test_writer_meta_carries_full_state_checksums(tmp_path):
    t = _mk(tmp_path)
    res = t.publish(
        1,
        [
            Payload("w@delta_idx", torch.tensor([0], dtype=torch.int32)),
            Payload("w@delta_val", torch.tensor([2.0], dtype=torch.float32)),
        ],
        anchor=False,
        full_state_checksums={"w": "cafebabe"},
    )
    t.write_manifest(1, anchor=False, per_writer=[res.writer_meta])
    import json

    m = json.loads(
        SharedFSStore(str(tmp_path)).get_bytes("s/deltas/v000001/manifest.json")
    )
    assert m["writers"][0]["full_state_checksums"] == {"w": "cafebabe"}


def test_engine_push_pull_e2e_over_http_backend(tmp_path):
    from dte.engine import DeltaEngine

    send = DeltaEngine(transport=_mk(tmp_path), mode="delta", anchor_interval=100)
    recv = DeltaEngine(transport=_mk(tmp_path), mode="delta", anchor_interval=100)
    w = {"w": torch.randn(64)}
    send.transport.begin(0)
    send.push(w.items(), version=0)  # full seed
    tgt = {"w": torch.zeros(64)}
    recv.pull(tgt, version=0)
    assert torch.equal(tgt["w"], w["w"])
    w["w"][3] += 1.0
    send.transport.begin(1)
    send.push(w.items(), version=1)  # delta
    recv.pull(tgt, version=1)
    assert torch.equal(tgt["w"], w["w"])


def test_recv_with_no_new_version_raises_nothing_to_fetch(tmp_path):
    from dte.backends.http_backend import NothingToFetch
    from dte.transport import Plan

    t = _mk(tmp_path)
    with pytest.raises(NothingToFetch):
        t.recv(Plan())
    t.begin(1)
    t.send(Plan(), [Payload("w", torch.ones(2, dtype=torch.float32))])
    assert {p.name for p in t.recv(Plan())} == {"w"}
    with pytest.raises(NothingToFetch):  # same version again: nothing new
        t.recv(Plan())


def test_send_without_begin_raises_for_headerless_full(tmp_path):
    from dte.transport import Plan

    t = _mk(tmp_path)
    with pytest.raises(RuntimeError):
        t.send(Plan(), [Payload("w", torch.ones(2, dtype=torch.float32))])


def test_delta_publish_stats_density(tmp_path):
    from dte.core import DeltaTracker

    t = _mk(tmp_path)
    tracker = DeltaTracker(0, 0.9)
    params = [("w", torch.zeros(100, dtype=torch.float32))]
    tracker.seed(params, 0)
    t.publish(0, [Payload("w", params[0][1])], anchor=True)
    params[0][1][7] = 5.0
    enc = tracker.encode(params, 1)
    res = t.publish(
        1,
        [Payload(n, v) for n, v in zip(enc.names, enc.tensors)],
        anchor=False,
        stats={"density": enc.changed_ratio},
    )
    assert res.kind == "delta"
    assert res.density == enc.changed_ratio
    assert 0 < res.density < 0.5


# ---------------------------------------------------------------------------
# Chunked publish + retention (large-model anchor path)
# ---------------------------------------------------------------------------


def test_chunked_publish_roundtrip_and_multiple_files(tmp_path):
    t = _mk(tmp_path)
    payloads = [
        Payload(f"w{i}", torch.full((256,), float(i), dtype=torch.float32))
        for i in range(8)
    ]  # 1KB each; 2.5KB budget -> 3+ chunks
    res = t.publish(1, iter(payloads), anchor=True, chunk_bytes=2500)
    assert res.num_tensors == 8
    assert len(res.writer_meta["files"]) >= 3
    out = {p.name: p.values for p in t.fetch(1, kind="anchor")}
    assert set(out) == {f"w{i}" for i in range(8)}
    for i in range(8):
        assert torch.equal(out[f"w{i}"], payloads[i].values)


def test_oversized_single_tensor_forms_own_chunk(tmp_path):
    t = _mk(tmp_path)
    payloads = [
        Payload("big", torch.zeros(4096, dtype=torch.float32)),  # 16KB > budget
        Payload("small", torch.ones(4, dtype=torch.float32)),
    ]
    res = t.publish(1, payloads, anchor=True, chunk_bytes=1024)
    assert res.num_tensors == 2
    out = {p.name for p in t.fetch(1, kind="anchor")}
    assert out == {"big", "small"}


def test_anchor_commit_prunes_older_versions(tmp_path):
    t = _mk(tmp_path)
    store = SharedFSStore(str(tmp_path))
    t.publish(1, [Payload("w", torch.ones(4, dtype=torch.float32))], anchor=True)
    for v in (2, 3):
        t.publish(
            v,
            [
                Payload("w@delta_idx", torch.tensor([0], dtype=torch.int32)),
                Payload("w@delta_val", torch.tensor([float(v)], dtype=torch.float32)),
            ],
            anchor=False,
        )
    assert t.list_versions() == ([1], [2, 3])
    # New anchor at v4 -> v1..v3 pruned, only v4 remains.
    t.publish(
        4, [Payload("w", torch.full((4,), 4.0, dtype=torch.float32))], anchor=True
    )
    assert t.list_versions() == ([4], [])
    assert not store.list_keys("s/anchors/v000001/")
    assert not store.list_keys("s/deltas/")
    assert t.poll_latest()["version"] == 4
    out = t.fetch(4, kind="anchor")
    assert torch.equal(out[0].values, torch.full((4,), 4.0))


def test_delta_commit_does_not_prune(tmp_path):
    t = _mk(tmp_path)
    t.publish(1, [Payload("w", torch.ones(4, dtype=torch.float32))], anchor=True)
    t.publish(
        2,
        [
            Payload("w@delta_idx", torch.tensor([0], dtype=torch.int32)),
            Payload("w@delta_val", torch.tensor([2.0], dtype=torch.float32)),
        ],
        anchor=False,
    )
    assert t.list_versions() == ([1], [2])  # anchor v1 must survive delta commits


def test_iter_fetch_chunks_drive_engine_reconstruct_stream(tmp_path):
    from dte.core import DeltaTracker
    from dte.engine import DeltaEngine

    t = _mk(tmp_path)
    w0 = {
        "a": torch.randn(64, dtype=torch.float32),
        "b": torch.randn(32, dtype=torch.float32),
        "c": torch.randn(16, dtype=torch.float32),
    }
    t.publish(0, [Payload(k, v) for k, v in w0.items()], anchor=True, chunk_bytes=128)

    recv = DeltaEngine(None)
    got = {}
    for chunk in recv.reconstruct_stream(t.iter_fetch(0, kind="anchor"), 0):
        got.update(chunk)
    assert set(got) == set(w0)
    for k in w0:
        assert torch.equal(got[k], w0[k])
    assert recv.base_version == 0

    tracker = DeltaTracker(0, 0.9)
    tracker.seed(list(w0.items()), 0)
    w1 = {k: v.clone() for k, v in w0.items()}
    w1["a"][3] = 1.5
    w1["b"][7] = -2.0
    enc = tracker.encode(list(w1.items()), 1)
    t.publish(
        1,
        [Payload(n, x) for n, x in zip(enc.names, enc.tensors)],
        anchor=False,
        chunk_bytes=64,
    )

    got1 = {}
    for chunk in recv.reconstruct_stream(t.iter_fetch(1, kind="delta"), 1):
        got1.update(chunk)
    assert set(got1) == {"a", "b"}, "unchanged c never crosses the stream"
    assert torch.equal(got1["a"], w1["a"])
    assert torch.equal(got1["b"], w1["b"])
    assert recv.base_version == 1


def test_writer_markers_let_readers_apply_before_manifest(tmp_path):
    """A reader can consume writer 0's chunks while writer 1 is still absent,
    and the manifest is not required to do so."""
    w0 = _mk(tmp_path, writer_rank=0, num_writers=2)
    w1 = _mk(tmp_path, writer_rank=1, num_writers=2)
    m0 = w0.publish(
        1, [Payload("a", torch.ones(2, dtype=torch.bfloat16))], anchor=True
    ).writer_meta
    w0.publish_writer_marker(1, anchor=True, meta=m0, num_writers=2)

    markers = w0.read_writer_markers(1, kind="anchor")
    assert [m["writer_rank"] for m in markers] == [0]
    assert markers[0]["num_writers"] == 2 and markers[0]["empty"] is False
    got = [
        p.name
        for chunk in w0.fetch_writer(1, kind="anchor", marker=markers[0])
        for p in chunk
    ]
    assert got == ["a"]
    with pytest.raises(ManifestMissing):
        w0.read_manifest(1, kind="anchor")

    m1 = w1.publish(
        1, [Payload("b", torch.ones(2, dtype=torch.bfloat16))], anchor=True
    ).writer_meta
    w1.publish_writer_marker(1, anchor=True, meta=m1, num_writers=2)
    markers = w0.read_writer_markers(1, kind="anchor")
    assert [m["writer_rank"] for m in markers] == [0, 1]
    w0.write_manifest(1, anchor=True, per_writer=[m0, m1])
    consumed = {
        p.name
        for m in markers
        for chunk in w0.fetch_writer(1, kind="anchor", marker=m)
        for p in chunk
    }
    manifest = w0.read_manifest(1, kind="anchor")
    committed = {
        name for writer in manifest["writers"] for name in writer["tensor_checksums"]
    }
    assert consumed == committed


def test_empty_writer_marker_is_distinguishable_from_pending(tmp_path):
    w = _mk(tmp_path, writer_rank=0, num_writers=2)
    w.publish_writer_marker(1, anchor=False, meta=None, num_writers=2)
    markers = w.read_writer_markers(1, kind="delta")
    assert len(markers) == 1
    assert markers[0]["empty"] is True
    assert markers[0]["base_version"] == 0
    assert list(w.fetch_writer(1, kind="delta", marker=markers[0])) == []


def test_fetch_writer_rejects_corrupted_chunk(tmp_path):
    w = _mk(tmp_path, writer_rank=0, num_writers=1)
    meta = w.publish(
        1, [Payload("a", torch.ones(4, dtype=torch.bfloat16))], anchor=True
    ).writer_meta
    w.publish_writer_marker(1, anchor=True, meta=meta, num_writers=1)
    fname = next(iter(meta["files"]))
    w.store.put_bytes(f"{w.stream}/anchors/v000001/{fname}", b"corrupted")
    marker = w.read_writer_markers(1, kind="anchor")[0]
    with pytest.raises(ChecksumMismatch):
        list(w.fetch_writer(1, kind="anchor", marker=marker))


def test_iter_fetch_slow_consumer_bounds_prefetch(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    t = _mk(tmp_path)
    t.publish(
        0,
        [Payload(f"w{i}", torch.ones(8)) for i in range(20)],
        anchor=True,
        chunk_bytes=32,
    )
    submitted = []
    original = ThreadPoolExecutor.submit

    def record(pool, fn, *args, **kwargs):
        submitted.append(args)
        return original(pool, fn, *args, **kwargs)

    monkeypatch.setattr(ThreadPoolExecutor, "submit", record)
    monkeypatch.setenv("DTE_HTTP_FETCH_WORKERS", "2")
    chunks = t.iter_fetch(0, kind="anchor")
    try:
        next(chunks)
        assert len(submitted) == 2
        next(chunks)
        assert len(submitted) == 3
    finally:
        chunks.close()


@pytest.mark.parametrize(
    "field,value", [("version", 2), ("schema", 99), ("kind", "delta")]
)
def test_manifest_rejects_wrong_identity(tmp_path, field, value):
    import json

    t = _mk(tmp_path)
    t.publish(0, [Payload("w", torch.ones(8))], anchor=True)
    key = "s/anchors/v000000/manifest.json"
    manifest = json.loads(t.store.get_bytes(key))
    manifest[field] = value
    t.store.put_bytes(key, json.dumps(manifest).encode())

    with pytest.raises(ValueError, match="invalid anchor manifest"):
        t.fetch(0, kind="anchor")


def test_writer_marker_rejects_wrong_version(tmp_path):
    t = _mk(tmp_path)
    meta = t.publish(0, [Payload("w", torch.ones(8))], anchor=True).writer_meta
    t.publish_writer_marker(0, anchor=True, meta=meta, num_writers=1)
    marker = t.read_writer_markers(0, kind="anchor")[0]
    marker["version"] = 1

    with pytest.raises(ValueError, match="invalid writer marker"):
        list(t.fetch_writer(0, kind="anchor", marker=marker))


def test_shared_fs_symlink_escape_is_rejected(tmp_path):
    root = tmp_path / "store"
    root.mkdir()
    (root / "escape").symlink_to(tmp_path, target_is_directory=True)
    store = SharedFSStore(str(root))

    with pytest.raises(ValueError, match="outside the store root"):
        store.put_bytes("escape/payload.bin", b"data")
    assert not (tmp_path / "payload.bin").exists()
