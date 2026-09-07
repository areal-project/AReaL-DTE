# Licensed under the Apache License, Version 2.0
"""Native OSS blob storage with application-owned authentication and sessions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dte.backends.http_backend import BlobStore, _validate_blob_key

if TYPE_CHECKING:
    import oss2


class OSSStore(BlobStore):
    """Stage bytes without local files using an initialized ``oss2.Bucket``.

    The application supplies V2/V4 authentication, credentials, endpoint,
    timeouts, and connection pool settings through ``bucket_client``. Concurrent
    readers must not mutate that client's configuration. Use a dedicated prefix
    for each run. Writes replace existing keys, as required for latest pointers.
    Payload keys must be immutable by the application's publication convention.

    Large blobs use sequential multipart uploads; this bounds the number of
    outstanding parts, but the caller still owns the complete input byte buffer.
    Failed uploads are left uncommitted; their upload IDs are not persisted here.
    The application must arrange cleanup of incomplete multipart uploads.
    """

    def __init__(
        self,
        bucket_client: oss2.Bucket,
        *,
        prefix: str,
        multipart_threshold: int = 64 * 1024**2,
        part_size: int = 16 * 1024**2,
    ) -> None:
        _validate_blob_key(prefix)
        if not 1 <= multipart_threshold <= 5 * 1024**3:
            raise ValueError("multipart_threshold must be between 1 byte and 5 GiB")
        if not 100 * 1024 <= part_size <= 5 * 1024**3:
            raise ValueError("part_size must be between 100 KiB and 5 GiB")
        self._bucket = bucket_client
        self._prefix = prefix + "/"
        self._multipart_threshold = multipart_threshold
        self._part_size = part_size

    def _key(self, key: str, *, prefix: bool = False) -> str:
        _validate_blob_key(key, prefix=prefix)
        return self._prefix + key

    def put_bytes(self, key: str, data: bytes) -> None:
        import oss2

        full_key = self._key(key)
        if len(data) < self._multipart_threshold:
            self._bucket.put_object(full_key, data)
            return
        size = oss2.determine_part_size(len(data), preferred_size=self._part_size)
        upload_id = self._bucket.init_multipart_upload(full_key).upload_id
        parts = []
        for number, offset in enumerate(range(0, len(data), size), start=1):
            result = self._bucket.upload_part(
                full_key, upload_id, number, data[offset : offset + size]
            )
            parts.append(oss2.models.PartInfo(number, result.etag))
        self._bucket.complete_multipart_upload(full_key, upload_id, parts)

    def get_bytes(self, key: str) -> bytes:
        import oss2

        try:
            response = self._bucket.get_object(self._key(key))
        except oss2.exceptions.NoSuchKey as exc:
            raise FileNotFoundError(key) from exc
        try:
            return response.read()
        finally:
            response.close()

    def exists(self, key: str) -> bool:
        import oss2

        try:
            self._bucket.head_object(self._key(key))
        except oss2.exceptions.NoSuchKey:
            return False
        except oss2.exceptions.NoSuchBucket:
            raise
        except oss2.exceptions.NotFound:
            # HEAD can omit the error body. GET distinguishes a missing bucket
            # from a missing key without treating every 404 as absence.
            try:
                response = self._bucket.get_object(self._key(key))
            except oss2.exceptions.NoSuchKey:
                return False
            response.close()
        return True

    def list_keys(self, prefix: str) -> list[str]:
        full_prefix = self._key(prefix, prefix=True)
        token = ""
        keys = []
        while True:
            page = self._bucket.list_objects_v2(
                prefix=full_prefix, continuation_token=token, max_keys=1000
            )
            for item in page.object_list:
                if not item.key.startswith(full_prefix):
                    raise ValueError(
                        "OSS listing returned an object outside the prefix"
                    )
                keys.append(item.key[len(self._prefix) :])
            if not page.is_truncated:
                return sorted(keys)
            next_token = page.next_continuation_token
            if not next_token or next_token == token:
                raise RuntimeError("OSS pagination did not advance")
            token = next_token

    def delete(self, key: str) -> None:
        import oss2

        try:
            self._bucket.delete_object(self._key(key))
        except oss2.exceptions.NoSuchKey:
            pass
