# Licensed under the Apache License, Version 2.0
"""http backend — staged blob transport over shared filesystems / object stores.

Unlike the peer backends (awex NCCL, mooncake RDMA) which move tensors over a
live fabric, the http backend *stages* each version as immutable blobs in a
    ``BlobStore`` (a shared filesystem or an S3-compatible object store),
so sender and receiver never need connectivity beyond the store — the design
point for cross-datacenter weight sync.

Layout per stream (one stream per train→infer pair)::

    {stream}/anchors/v{N:06d}/payload-{writer:05d}-{chunk:04d}.st.zst
    {stream}/deltas/v{N:06d}/payload-{writer:05d}-{chunk:04d}.st.zst
    {stream}/{anchors|deltas}/v{N:06d}/manifest.json   # written last per version
    {stream}/latest.json                               # atomic publication point

SharedFS uses the standard library; S3 imports boto3 and payload packing imports
safetensors and zstd lazily.
"""

from __future__ import annotations

import io
import json
import os
import uuid
import zlib
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from dte.transport import Payload, Plan, Transport

__all__ = [
    "BlobStore",
    "SharedFSStore",
    "S3Store",
    "ChecksumMismatch",
    "ManifestMissing",
    "NothingToFetch",
    "PublishResult",
    "HttpTransport",
    "compute_checksum",
    "pack_payloads",
    "unpack_payloads",
]


class ChecksumMismatch(RuntimeError):
    """Payload checksum verification failed (corrupted wire or storage)."""


class _Adler32:
    """adler32 behind the incremental .update/.hexdigest interface hashers expose."""

    def __init__(self) -> None:
        self._value = 1

    def update(self, data) -> None:
        self._value = zlib.adler32(data, self._value)

    def hexdigest(self) -> str:
        return f"{self._value:08x}"


def _new_hasher(algorithm: str):
    if algorithm == "xxh3-128":
        import xxhash

        return xxhash.xxh3_128()
    if algorithm == "blake3":
        import blake3

        return blake3.blake3()
    if algorithm == "adler32":
        return _Adler32()
    raise KeyError(f"unknown checksum algorithm {algorithm!r}")


def compute_checksum(algorithm: str, buf) -> str:
    hasher = _new_hasher(algorithm)
    hasher.update(buf)
    return hasher.hexdigest()


def _tensor_bytes(values) -> bytes:
    import torch

    flat = values.detach().contiguous().view(-1)
    if flat.dtype == torch.bfloat16:
        flat = flat.view(torch.int16)
    return flat.numpy().tobytes()


def _validate_blob_key(key: str, *, prefix: bool = False) -> str:
    """Validate a store-relative POSIX key before it reaches a provider."""
    candidate = key[:-1] if prefix and key.endswith("/") else key
    if (
        not candidate
        or candidate.startswith("/")
        or "\\" in candidate
        or any(part in {"", ".", ".."} for part in candidate.split("/"))
    ):
        raise ValueError(f"invalid blob key {key!r}")
    return key


def pack_payloads(
    payloads: list[Payload], checksum: str
) -> tuple[bytes, dict[str, str]]:
    """Pack payload tensors into one zstd-compressed safetensors blob.

    Returns ``(compressed_bytes, {name: checksum_of_uncompressed_tensor_bytes})``.
    Checksums are computed on the raw storage bytes of each tensor so the
    receiver can verify integrity after decompression, independent of the
    container framing. Tensors may live on any device; staging serializes
    from host memory (GPU tensors are copied to CPU here).
    """
    import zstandard
    from safetensors.torch import save

    tensors = {p.name: p.values.detach().contiguous().cpu() for p in payloads}
    if len(tensors) != len(payloads):
        raise ValueError("Duplicate tensor names within a payload chunk")
    if any(p.indices is not None or p.header is not None for p in payloads):
        raise ValueError("HTTP transport requires flat named-tensor payloads")
    sums = {
        name: compute_checksum(checksum, _tensor_bytes(t))
        for name, t in tensors.items()
    }
    blob = save(tensors)
    return zstandard.ZstdCompressor(level=1, threads=-1).compress(blob), sums


def unpack_payloads(
    data: bytes, tensor_checksums: dict[str, str], checksum: str
) -> list[Payload]:
    """Inverse of :func:`pack_payloads`; raises :class:`ChecksumMismatch` on any
    per-tensor digest that differs from the manifest's expectation."""
    import zstandard
    from safetensors.torch import load

    blob = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(data)).read()
    tensors = load(blob)
    mismatches = [
        name
        for name, t in tensors.items()
        if compute_checksum(checksum, _tensor_bytes(t)) != tensor_checksums.get(name)
    ]
    if mismatches:
        raise ChecksumMismatch(
            f"checksum mismatch for {len(mismatches)} tensors: {sorted(mismatches)[:20]}"
        )
    return [Payload(name, t) for name, t in tensors.items()]


class BlobStore(ABC):
    """Minimal blob contract the transport stages through.

    ``put_bytes`` must be atomic per key: a reader either sees the previous
    complete value or the new complete value, never a torn write. Keys are
    ``/``-separated paths; ``list_keys`` returns full keys sorted ascending.
    """

    @abstractmethod
    def put_bytes(self, key: str, data: bytes) -> None:
        """Store ``data`` under ``key`` atomically (create or overwrite)."""

    @abstractmethod
    def get_bytes(self, key: str) -> bytes:
        """Return the value at ``key``; raise ``FileNotFoundError`` if absent."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Whether ``key`` holds a committed value."""

    @abstractmethod
    def list_keys(self, prefix: str) -> list[str]:
        """All committed keys under ``prefix``, sorted ascending."""

    def delete_prefix(self, prefix: str) -> int:
        """Delete every key under ``prefix``; returns the number deleted.
        Missing keys are not an error (idempotent, retry-safe)."""
        deleted = 0
        for key in self.list_keys(prefix):
            self.delete(key)
            deleted += 1
        return deleted

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove ``key`` if present (no error when absent)."""


class SharedFSStore(BlobStore):
    """BlobStore over a POSIX filesystem (local disk or shared volume).

    Writes go to a unique temp file then ``os.replace`` onto the final path —
    the same tmp+fsync+rename discipline slime's disk-delta publisher uses, so
    a concurrent reader never observes a torn blob. Note that shared volumes
    may lack cross-host read-after-write consistency; callers coordinate
    *visibility* out of band (the transport's manifest-last protocol), this
    class only guarantees per-file atomicity.
    """

    def __init__(self, root: str) -> None:
        self._root = os.path.abspath(root)

    def _path(self, key: str) -> str:
        _validate_blob_key(key)
        path = os.path.join(self._root, key)
        root = os.path.realpath(self._root)
        if os.path.commonpath([root, os.path.realpath(path)]) != root:
            raise ValueError("Blob key resolves outside the store root")
        return path

    def put_bytes(self, key: str, data: bytes) -> None:
        path = self._path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp.{uuid.uuid4().hex}"
        try:
            with open(tmp, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):  # replace failed; don't leak temp files
                os.unlink(tmp)

    def get_bytes(self, key: str) -> bytes:
        with open(self._path(key), "rb") as f:
            return f.read()

    def exists(self, key: str) -> bool:
        return os.path.isfile(self._path(key))

    def list_keys(self, prefix: str) -> list[str]:
        _validate_blob_key(prefix, prefix=True)
        keys: list[str] = []
        for dirpath, _, filenames in os.walk(self._root):
            for name in filenames:
                rel = os.path.relpath(os.path.join(dirpath, name), self._root)
                key = rel.replace(os.sep, "/")
                if key.startswith(prefix):
                    keys.append(key)
        return sorted(keys)

    def delete(self, key: str) -> None:
        try:
            os.unlink(self._path(key))
        except FileNotFoundError:
            pass
        # Drop now-empty parent dirs so pruned versions vanish entirely.
        parent = os.path.dirname(self._path(key))
        while parent.startswith(self._root) and parent != self._root:
            try:
                os.rmdir(parent)
            except OSError:
                break
            parent = os.path.dirname(parent)


class ManifestMissing(FileNotFoundError):
    """No manifest for the requested version (never published or not committed)."""


class S3Store(BlobStore):
    """BlobStore over any S3-compatible object store (AWS S3, OSS, MinIO).

    ``endpoint_url`` overrides the endpoint for non-AWS deployments; credentials
    come from the standard AWS environment/config chain (never passed here).
    S3 PUTs are atomic per key, satisfying the ``put_bytes`` contract natively.
    """

    def __init__(self, bucket: str, prefix: str = "", endpoint_url: str | None = None):
        import boto3

        self._bucket = bucket
        self._prefix = prefix.strip("/")
        if self._prefix:
            _validate_blob_key(self._prefix)
        self._client = boto3.client("s3", endpoint_url=endpoint_url or None)

    def _key(self, key: str, *, prefix: bool = False) -> str:
        _validate_blob_key(key, prefix=prefix)
        return f"{self._prefix}/{key}" if self._prefix else key

    @staticmethod
    def _is_not_found(exc) -> bool:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        return code in {"404", "NoSuchKey", "NotFound"} or status == 404

    def put_bytes(self, key: str, data: bytes) -> None:
        self._client.put_object(Bucket=self._bucket, Key=self._key(key), Body=data)

    def get_bytes(self, key: str) -> bytes:
        from botocore.exceptions import ClientError

        try:
            resp = self._client.get_object(Bucket=self._bucket, Key=self._key(key))
        except ClientError as exc:
            if not self._is_not_found(exc):
                raise
            raise FileNotFoundError(f"s3://{self._bucket}/{self._key(key)}") from exc
        return resp["Body"].read()

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self._client.head_object(Bucket=self._bucket, Key=self._key(key))
            return True
        except ClientError as exc:
            if self._is_not_found(exc):
                return False
            raise

    def list_keys(self, prefix: str) -> list[str]:
        full_prefix = self._key(prefix, prefix=True)
        strip = len(self._prefix) + 1 if self._prefix else 0
        keys: list[str] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=full_prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"][strip:])
        return sorted(keys)

    def delete(self, key: str) -> None:
        self._client.delete_object(Bucket=self._bucket, Key=self._key(key))


class NothingToFetch(RuntimeError):
    """``recv`` found no version newer than the last one pulled."""


@dataclass
class PublishResult:
    """Outcome of a writer-side ``publish``; ``writer_meta`` feeds ``write_manifest``."""

    version: int
    kind: str
    wire_bytes: int
    num_tensors: int
    density: float | None
    writer_meta: dict


class HttpTransport(Transport):
    """Staged blob transport: versions are published to a ``BlobStore`` and
    fetched by version, decoupling sender and receiver in time and space.

    Two API levels coexist:

    - staging API (production): ``publish`` / ``write_manifest`` / ``fetch`` /
      ``poll_latest`` — multi-writer aware; the caller coordinates writers
      (e.g. over gloo) and rank 0 commits the manifest last, which is the
      atomic publication point (readers only ever see committed versions).
    - ``Transport`` ABC (single writer): ``send`` auto-publishes (delta version
      read from the payload header, anchor version from ``begin``); ``recv``
      polls ``latest.json`` and fetches anything new, so ``DeltaEngine``'s
      ``push``/``pull`` work over this backend unchanged.
    """

    def __init__(
        self,
        store: BlobStore,
        stream: str = "default",
        writer_rank: int = 0,
        num_writers: int = 1,
        checksum: str = "xxh3-128",
    ) -> None:
        _validate_blob_key(stream)
        if num_writers < 1:
            raise ValueError(f"num_writers must be positive, got {num_writers}")
        if not 0 <= writer_rank < num_writers:
            raise ValueError(
                f"writer_rank must be in [0, {num_writers}), got {writer_rank}"
            )
        if checksum not in {"xxh3-128", "blake3", "adler32"}:
            raise ValueError(f"unsupported checksum algorithm {checksum!r}")
        self.store = store
        self.stream = stream
        self.writer_rank = writer_rank
        self.num_writers = num_writers
        self.checksum = checksum
        self._begin_version: int | None = None
        self._pulled_version: int | None = None

    # ------------------------------------------------------------- staging API
    @staticmethod
    def _anchor_from_kind(kind: str) -> bool:
        if kind not in {"anchor", "delta"}:
            raise ValueError(f"kind must be 'anchor' or 'delta', got {kind!r}")
        return kind == "anchor"

    def _version_dir(self, version: int, *, anchor: bool) -> str:
        return f"{'anchors' if anchor else 'deltas'}/v{version:06d}"

    @staticmethod
    def _payload_nbytes(payload: Payload) -> int:
        t = payload.values
        return t.numel() * t.element_size()

    def _chunk_payloads(
        self, payloads: Iterable[Payload], chunk_bytes: int | None
    ) -> Iterator[list[Payload]]:
        """Greedily group payloads into chunks of ~chunk_bytes (uncompressed).

        Adjacent payloads sharing a base name (``w``, ``w@delta_idx``, ``w@delta_val``)
        never split across chunks, so every chunk file is self-contained and
        independently appliable (streamed fetch depends on this). A single
        group larger than the budget still forms its own chunk.
        """
        if chunk_bytes is None:
            batch = list(payloads)
            if batch:
                yield batch
            return
        if chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be positive")

        def groups() -> Iterator[list[Payload]]:
            group: list[Payload] = []
            key: str | None = None
            for p in payloads:
                base = p.name.split("@", 1)[0]
                if group and base != key:
                    yield group
                    group = []
                key = base
                group.append(p)
            if group:
                yield group

        batch: list[Payload] = []
        size = 0
        for group in groups():
            gsize = sum(self._payload_nbytes(p) for p in group)
            if batch and size + gsize > chunk_bytes:
                yield batch
                batch, size = [], 0
            batch.extend(group)
            size += gsize
        if batch:
            yield batch

    def publish(
        self,
        version: int,
        payloads: Iterable[Payload],
        *,
        anchor: bool,
        stats: dict | None = None,
        full_state_checksums: dict[str, str] | None = None,
        chunk_bytes: int | None = None,
    ) -> PublishResult:
        """Stage this writer's payloads for ``version``.

        ``payloads`` may be a list or a lazy iterator; with ``chunk_bytes`` set
        it is consumed chunk by chunk — pack, upload, release. Memory scales with
        the largest chunk plus serialization/compression buffers. A tensor group
        larger than the budget is not split. Sparse index/value pairs must be
        adjacent, as emitted by the DTE codec.
        """
        vdir = self._version_dir(version, anchor=anchor)
        files: dict[str, dict] = {}
        tensor_sums: dict[str, str] = {}
        wire_bytes = 0
        num_tensors = 0
        for idx, chunk in enumerate(self._chunk_payloads(payloads, chunk_bytes)):
            data, sums = pack_payloads(chunk, checksum=self.checksum)
            if tensor_sums.keys() & sums.keys():
                raise ValueError("Duplicate tensor names across writer chunks")
            fname = f"payload-{self.writer_rank:05d}-{idx:04d}.st.zst"
            self.store.put_bytes(f"{self.stream}/{vdir}/{fname}", data)
            files[fname] = {
                "bytes": len(data),
                "checksum": compute_checksum(self.checksum, data),
            }
            tensor_sums.update(sums)
            wire_bytes += len(data)
            num_tensors += len(chunk)
            del data, chunk
        kind = "anchor" if anchor else "delta"
        stats = dict(stats or {})
        stats.setdefault("wire_bytes", wire_bytes)
        stats.setdefault("num_tensors", num_tensors)
        writer_meta = {
            "writer_rank": self.writer_rank,
            "files": files,
            "tensor_checksums": tensor_sums,
            "full_state_checksums": dict(full_state_checksums or {}),
            "stats": stats,
        }
        if self.num_writers == 1 and files:
            self.write_manifest(version, anchor=anchor, per_writer=[writer_meta])
        return PublishResult(
            version=version,
            kind=kind,
            wire_bytes=wire_bytes,
            num_tensors=num_tensors,
            density=stats.get("density"),
            writer_meta=writer_meta,
        )

    def _writer_marker_key(self, version: int, *, anchor: bool, writer: int) -> str:
        vdir = self._version_dir(version, anchor=anchor)
        return f"{self.stream}/{vdir}/writers/w{writer:05d}.json"

    def publish_writer_marker(
        self, version: int, *, anchor: bool, meta: dict | None, num_writers: int
    ) -> None:
        """Announce that THIS writer's payload files are all committed.

        Lets a reader start applying this writer's self-contained chunks while
        peers are still uploading, instead of waiting for the whole version.
        ``meta=None`` publishes an empty marker, which is mandatory: readers
        wait for exactly ``num_writers`` markers and cannot otherwise tell
        "nothing to send" from "not finished yet". The manifest is still
        written last and remains the version's completion criterion.
        """
        if num_writers != self.num_writers:
            raise ValueError(
                f"marker num_writers={num_writers} does not match transport "
                f"num_writers={self.num_writers}"
            )
        if meta is not None and meta.get("writer_rank") != self.writer_rank:
            raise ValueError(
                f"marker metadata belongs to writer {meta.get('writer_rank')}, "
                f"not writer {self.writer_rank}"
            )
        payload = {
            "schema": 1,
            "version": version,
            "base_version": None if anchor else version - 1,
            "kind": "anchor" if anchor else "delta",
            "writer_rank": self.writer_rank,
            "num_writers": num_writers,
            "checksum_format": self.checksum,
            "empty": meta is None,
            "meta": meta,
        }
        self.store.put_bytes(
            self._writer_marker_key(version, anchor=anchor, writer=self.writer_rank),
            json.dumps(payload).encode(),
        )

    def read_writer_markers(self, version: int, *, kind: str) -> list[dict]:
        """Return the writer markers committed so far for this version."""
        anchor = self._anchor_from_kind(kind)
        vdir = self._version_dir(version, anchor=anchor)
        prefix = f"{self.stream}/{vdir}/writers/"
        out = []
        for key in sorted(self.store.list_keys(prefix)):
            if not key.endswith(".json"):
                continue
            try:
                marker = json.loads(self.store.get_bytes(key))
            except FileNotFoundError:
                continue
            self._validate_marker(marker, version=version, kind=kind)
            out.append(marker)
        return out

    def _validate_marker(self, marker: dict, *, version: int, kind: str) -> None:
        expected_base = None if kind == "anchor" else version - 1
        expected = {
            "schema": 1,
            "version": version,
            "base_version": expected_base,
            "kind": kind,
            "num_writers": self.num_writers,
        }
        mismatched = {
            key: (marker.get(key), value)
            for key, value in expected.items()
            if marker.get(key) != value
        }
        writer = marker.get("writer_rank")
        if not isinstance(writer, int) or not 0 <= writer < self.num_writers:
            mismatched["writer_rank"] = (writer, f"[0, {self.num_writers})")
        meta = marker.get("meta")
        if marker.get("empty") is not (meta is None):
            mismatched["empty"] = (marker.get("empty"), meta is None)
        if meta is not None and meta.get("writer_rank") != writer:
            mismatched["meta.writer_rank"] = (meta.get("writer_rank"), writer)
        if mismatched:
            raise ValueError(f"invalid writer marker: {mismatched}")

    def fetch_writer(
        self, version: int, *, kind: str, marker: dict
    ) -> Iterator[list[Payload]]:
        """Yield one committed writer's payload chunks, verified against the
        checksums recorded in ITS marker (per-writer scope, not the merged
        manifest, so no manifest read is needed to start applying)."""
        self._validate_marker(marker, version=version, kind=kind)
        meta = marker.get("meta") or {}
        files = meta.get("files") or {}
        tensor_sums = meta.get("tensor_checksums") or {}
        checksum = marker.get("checksum_format", self.checksum)
        vdir = self._version_dir(version, anchor=self._anchor_from_kind(kind))
        for fname, info in sorted(files.items()):
            data = self.store.get_bytes(f"{self.stream}/{vdir}/{fname}")
            digest = compute_checksum(checksum, data)
            if len(data) != info["bytes"] or digest != info["checksum"]:
                raise ChecksumMismatch(
                    f"file {fname} of {kind} v{version}: bytes/checksum differ "
                    f"from writer marker (got {len(data)}B/{digest})"
                )
            yield unpack_payloads(data, tensor_sums, checksum=checksum)

    def write_manifest(
        self, version: int, *, anchor: bool, per_writer: list[dict]
    ) -> None:
        """Aggregate writers' metadata into the version manifest, then commit
        ``latest.json`` — strictly in that order: the manifest must be complete
        before any reader can discover the version. Committing an anchor then
        prunes every older version (see :meth:`prune_versions_before`)."""
        if self.writer_rank != 0:
            raise RuntimeError(
                f"write_manifest must run on writer_rank 0, not {self.writer_rank}"
            )
        if len(per_writer) != self.num_writers:
            raise ValueError(
                f"expected metadata from {self.num_writers} writers, "
                f"got {len(per_writer)}"
            )
        writers = sorted(per_writer, key=lambda meta: meta.get("writer_rank", -1))
        ranks = [meta.get("writer_rank") for meta in writers]
        if ranks != list(range(self.num_writers)):
            raise ValueError(
                f"writer metadata ranks must be 0..{self.num_writers - 1}, got {ranks}"
            )

        kind = "anchor" if anchor else "delta"
        vdir = self._version_dir(version, anchor=anchor)
        wire_bytes = num_tensors = 0
        densities = []
        for meta in writers:
            wire_bytes += meta["stats"].get("wire_bytes", 0)
            num_tensors += meta["stats"].get("num_tensors", 0)
            if meta["stats"].get("density") is not None:
                densities.append(meta["stats"]["density"])
        manifest = {
            "schema": 1,
            "version": version,
            "base_version": None if anchor else version - 1,
            "kind": kind,
            "encoding": "dte_flat_v1",
            "compression": "zstd",
            "checksum_format": self.checksum,
            "num_writers": self.num_writers,
            "writers": writers,
            "stats": {
                "wire_bytes": wire_bytes,
                "num_tensors": num_tensors,
                "density": (sum(densities) / len(densities)) if densities else None,
            },
        }
        self.store.put_bytes(
            f"{self.stream}/{vdir}/manifest.json", json.dumps(manifest).encode()
        )
        latest = {"schema": 1, "version": version, "kind": kind, "dir": vdir}
        self.store.put_bytes(f"{self.stream}/latest.json", json.dumps(latest).encode())
        if anchor:
            # A committed anchor supersedes the whole chain before it; prune so
            # store usage stays bounded at ~1 anchor + one interval of deltas.
            # Readers mid-fetch on a pruned version hit ManifestMissing and
            # recover via the chain_broken -> fresh-anchor path.
            self.prune_versions_before(version)

    def list_versions(self) -> tuple[list[int], list[int]]:
        """Committed (anchor_versions, delta_versions), each sorted ascending."""

        def versions(kind_dir: str) -> list[int]:
            prefix = f"{self.stream}/{kind_dir}/"
            found = set()
            for key in self.store.list_keys(prefix):
                rest = key[len(prefix) :]
                if rest.endswith("/manifest.json"):
                    name = rest.split("/", 1)[0]
                    if name.startswith("v") and name[1:].isdigit():
                        found.add(int(name[1:]))
            return sorted(found)

        return versions("anchors"), versions("deltas")

    def prune_versions_before(self, version: int) -> int:
        """Delete every committed version older than ``version``; returns the
        number of versions removed. Manifests are deleted first so a pruned
        version is never half-visible to a reader."""
        anchors, deltas = self.list_versions()
        removed = 0
        for v in anchors:
            if v < version:
                vdir = self._version_dir(v, anchor=True)
                self.store.delete(f"{self.stream}/{vdir}/manifest.json")
                self.store.delete_prefix(f"{self.stream}/{vdir}/")
                removed += 1
        for v in deltas:
            if v < version:
                vdir = self._version_dir(v, anchor=False)
                self.store.delete(f"{self.stream}/{vdir}/manifest.json")
                self.store.delete_prefix(f"{self.stream}/{vdir}/")
                removed += 1
        return removed

    def read_manifest(self, version: int, *, kind: str) -> dict:
        vdir = self._version_dir(version, anchor=self._anchor_from_kind(kind))
        key = f"{self.stream}/{vdir}/manifest.json"
        if not self.store.exists(key):
            raise ManifestMissing(f"no manifest for {kind} version {version} at {key}")
        manifest = json.loads(self.store.get_bytes(key))
        expected_base = None if kind == "anchor" else version - 1
        expected = {
            "schema": 1,
            "version": version,
            "base_version": expected_base,
            "kind": kind,
            "encoding": "dte_flat_v1",
            "compression": "zstd",
        }
        mismatched = {
            field: (manifest.get(field), value)
            for field, value in expected.items()
            if manifest.get(field) != value
        }
        writers = manifest.get("writers")
        if not isinstance(writers, list) or len(writers) != manifest.get("num_writers"):
            mismatched["writers"] = (
                len(writers) if isinstance(writers, list) else type(writers).__name__,
                manifest.get("num_writers"),
            )
        elif any(not isinstance(writer, dict) for writer in writers):
            mismatched["writers"] = ("non-object entry", "objects")
        else:
            ranks = [writer.get("writer_rank") for writer in writers]
            if ranks != list(range(len(writers))):
                mismatched["writer_ranks"] = (ranks, list(range(len(writers))))
            for writer in writers:
                if not isinstance(writer.get("files"), dict) or not isinstance(
                    writer.get("tensor_checksums"), dict
                ):
                    mismatched[f"writer_{writer.get('writer_rank')}"] = (
                        "invalid files/checksums",
                        "mapping",
                    )
        if manifest.get("checksum_format") not in {"xxh3-128", "blake3", "adler32"}:
            mismatched["checksum_format"] = (
                manifest.get("checksum_format"),
                "supported algorithm",
            )
        if mismatched:
            raise ValueError(
                f"invalid {kind} manifest for version {version}: {mismatched}"
            )
        return manifest

    def fetch(self, version: int, *, kind: str) -> list[Payload]:
        return [p for chunk in self.iter_fetch(version, kind=kind) for p in chunk]

    def iter_fetch(self, version: int, *, kind: str) -> Iterator[list[Payload]]:
        """Yield the version's payloads one committed chunk file at a time —
        the receiver-side memory bound for large anchors (peak ~= workers
        in-flight chunks). Each chunk verifies file- and tensor-level
        checksums before yielding.

        Chunk files (one per writer x chunk) are independent, so read +
        checksum + zstd decode run in a bounded pool; results still yield in
        manifest order. DTE_HTTP_FETCH_WORKERS=1 restores the serial path.
        """
        manifest = self.read_manifest(version, kind=kind)
        vdir = self._version_dir(version, anchor=self._anchor_from_kind(kind))

        def fetch_one(item: tuple[str, dict, dict[str, str]]) -> list[Payload]:
            fname, info, tensor_sums = item
            data = self.store.get_bytes(f"{self.stream}/{vdir}/{fname}")
            digest = compute_checksum(manifest["checksum_format"], data)
            if len(data) != info["bytes"] or digest != info["checksum"]:
                raise ChecksumMismatch(
                    f"file {fname} of {kind} v{version}: bytes/checksum differ "
                    f"from manifest (got {len(data)}B/{digest})"
                )
            return unpack_payloads(
                data,
                tensor_sums,
                checksum=manifest["checksum_format"],
            )

        files = [
            (fname, info, writer["tensor_checksums"])
            for writer in manifest["writers"]
            for fname, info in sorted(writer["files"].items())
        ]
        workers = min(
            max(int(os.environ.get("DTE_HTTP_FETCH_WORKERS", "8")), 1), len(files) or 1
        )
        if workers == 1:
            for item in files:
                yield fetch_one(item)
            return

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers) as pool:
            remaining = iter(files)
            pending = deque(
                pool.submit(fetch_one, next(remaining)) for _ in range(workers)
            )
            try:
                while pending:
                    yield pending.popleft().result()
                    item = next(remaining, None)
                    if item is not None:
                        pending.append(pool.submit(fetch_one, item))
            finally:
                for future in pending:
                    future.cancel()

    def poll_latest(self) -> dict | None:
        key = f"{self.stream}/latest.json"
        if not self.store.exists(key):
            return None
        latest = json.loads(self.store.get_bytes(key))
        kind = latest.get("kind")
        version = latest.get("version")
        if (
            latest.get("schema") != 1
            or kind not in {"anchor", "delta"}
            or not isinstance(version, int)
            or latest.get("dir")
            != self._version_dir(version, anchor=(kind == "anchor"))
        ):
            raise ValueError(f"invalid latest marker: {latest}")
        return latest

    # -------------------------------------------------- Transport ABC (1 writer)
    def begin(self, version: int) -> None:
        """Set the version ``send`` publishes a header-less full sync under."""
        self._begin_version = version

    def build_plan(self, train_meta, infer_meta) -> Plan:
        return Plan()

    def send(self, plan: Plan, payloads: list[Payload]) -> None:
        from dte.core.codec import DELTA_HEADER_NAME, DeltaHeader

        if self.num_writers != 1:
            raise ValueError("send/recv require one writer; use the staging API")
        header = next((p for p in payloads if p.name == DELTA_HEADER_NAME), None)
        if header is not None:
            version = DeltaHeader.from_tensor(header.values).payload_version
            self.publish(version, payloads, anchor=False)
            return
        if self._begin_version is None:
            raise RuntimeError(
                "send() got a header-less full payload with no version context; "
                "call begin(version) first"
            )
        self.publish(self._begin_version, payloads, anchor=True)

    def recv(self, plan: Plan) -> list[Payload]:
        latest = self.poll_latest()
        if latest is None or (
            self._pulled_version is not None
            and latest["version"] <= self._pulled_version
        ):
            raise NothingToFetch(
                f"no version newer than {self._pulled_version} in stream "
                f"{self.stream!r}"
            )
        manifest = self.read_manifest(latest["version"], kind=latest["kind"])
        if manifest["num_writers"] != 1:
            raise ValueError("send/recv require one writer; use the staging API")
        payloads = self.fetch(latest["version"], kind=latest["kind"])
        self._pulled_version = latest["version"]
        return payloads
