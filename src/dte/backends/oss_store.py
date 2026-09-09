# Licensed under the Apache License, Version 2.0
"""Native OSS blob storage with application-owned authentication and sessions."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, TypeVar

from dte.backends.http_backend import BlobStore, _validate_blob_key

if TYPE_CHECKING:
    import oss2

logger = logging.getLogger(__name__)
_T = TypeVar("_T")


class OSSStore(BlobStore):
    """Stage bytes without local files using an initialized ``oss2.Bucket``.

    The application supplies V2/V4 authentication, credentials, endpoint,
    timeouts, and connection pool settings through ``bucket_client``. Concurrent
    readers must not mutate that client's configuration. Use a dedicated prefix
    for each run. Writes replace existing keys, as required for latest pointers.
    Payload keys must be immutable by the application's publication convention.

    Large blobs use sequential multipart uploads; this bounds the number of
    outstanding parts, but the caller still owns the complete input byte buffer.
    Upload IDs are not persisted here. The application must arrange cleanup of
    incomplete multipart uploads. A lost response does not imply a failed commit.

    GET bodies, HEAD, listing pages and uploads retry transient failures with
    bounded backoff. Callers must serialize writes to mutable keys such as latest.
    Lost multipart completion responses are reconciled against exact object bytes.
    Retried initialization may leave empty upload sessions for application cleanup.
    ``read_attempts`` and ``write_attempts`` include the initial request; the
    historical ``read_backoff`` option controls backoff for both reads and writes.
    """

    def __init__(
        self,
        bucket_client: oss2.Bucket,
        *,
        prefix: str,
        multipart_threshold: int = 64 * 1024**2,
        part_size: int = 16 * 1024**2,
        read_attempts: int = 4,
        read_backoff: float = 0.5,
        write_attempts: int = 4,
    ) -> None:
        _validate_blob_key(prefix)
        if not 1 <= multipart_threshold <= 5 * 1024**3:
            raise ValueError("multipart_threshold must be between 1 byte and 5 GiB")
        if not 100 * 1024 <= part_size <= 5 * 1024**3:
            raise ValueError("part_size must be between 100 KiB and 5 GiB")
        if type(read_attempts) is not int or read_attempts < 1:
            raise ValueError("read_attempts must be a positive integer")
        if type(write_attempts) is not int or write_attempts < 1:
            raise ValueError("write_attempts must be a positive integer")
        if not 0 <= read_backoff <= 8:
            raise ValueError("read_backoff must be between 0 and 8 seconds")
        self._bucket = bucket_client
        self._prefix = prefix + "/"
        self._multipart_threshold = multipart_threshold
        self._part_size = part_size
        self._read_attempts = read_attempts
        self._retry_backoff = read_backoff
        self._write_attempts = write_attempts

    @staticmethod
    def _retryable_error(exc: Exception) -> bool:
        import oss2
        from requests import exceptions as request_errors

        underlying = (
            exc.exception if isinstance(exc, oss2.exceptions.RequestError) else exc
        )
        if isinstance(underlying, request_errors.SSLError):
            return False
        if isinstance(exc, oss2.exceptions.RequestError):
            return isinstance(
                underlying, (request_errors.ConnectionError, request_errors.Timeout)
            )
        if isinstance(
            exc,
            (
                request_errors.ConnectionError,
                request_errors.Timeout,
                request_errors.ChunkedEncodingError,
            ),
        ):
            return True
        return isinstance(exc, oss2.exceptions.OssError) and exc.status in (
            429,
            500,
            502,
            503,
            504,
        )

    def _read_with_retry(self, operation: Callable[[], _T]) -> _T:
        return self._with_retry(operation, self._read_attempts, "read")

    def _write_with_retry(self, operation: Callable[[], _T]) -> _T:
        return self._with_retry(operation, self._write_attempts, "write")

    def _with_retry(
        self, operation: Callable[[], _T], attempts: int, kind: Literal["read", "write"]
    ) -> _T:
        for attempt in range(attempts):
            try:
                return operation()
            except Exception as exc:
                if not self._retryable_error(exc) or attempt + 1 == attempts:
                    raise
                logger.warning(
                    "OSS %s retry %d/%d after %s",
                    kind,
                    attempt + 1,
                    attempts - 1,
                    type(exc).__name__,
                )
                time.sleep(min(self._retry_backoff * 2**attempt, 8))

    def _key(self, key: str, *, prefix: bool = False) -> str:
        _validate_blob_key(key, prefix=prefix)
        return self._prefix + key

    def put_bytes(self, key: str, data: bytes) -> None:
        import oss2

        full_key = self._key(key)
        if len(data) < self._multipart_threshold:
            self._write_with_retry(lambda: self._bucket.put_object(full_key, data))
            return
        size = oss2.determine_part_size(len(data), preferred_size=self._part_size)
        upload_id = self._write_with_retry(
            lambda: self._bucket.init_multipart_upload(full_key)
        ).upload_id
        parts = []
        for number, offset in enumerate(range(0, len(data), size), start=1):
            result = self._write_with_retry(
                lambda: self._bucket.upload_part(
                    full_key, upload_id, number, data[offset : offset + size]
                )
            )
            parts.append(oss2.models.PartInfo(number, result.etag))

        def complete():
            try:
                return self._bucket.complete_multipart_upload(
                    full_key, upload_id, parts
                )
            except Exception as exc:
                if not self._retryable_error(exc) and not isinstance(
                    exc, oss2.exceptions.NoSuchUpload
                ):
                    raise
                # A successful Complete may lose its response. A subsequent
                # retry can return NoSuchUpload; verify the exact object bytes
                # rather than assuming either success or failure from that code.
                try:
                    matches = self.get_bytes(key) == data
                except FileNotFoundError:
                    matches = False
                if matches:
                    return None
                raise

        self._write_with_retry(complete)

    def get_bytes(self, key: str) -> bytes:
        # Retry the entire GET, including body consumption. Partial bytes never
        # reach the applier, and each failed response is closed before retrying.
        return self._read_with_retry(lambda: self._get_bytes_once(key))

    def _get_bytes_once(self, key: str) -> bytes:
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
        return self._read_with_retry(lambda: self._exists_once(key))

    def _exists_once(self, key: str) -> bool:
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
            page = self._read_with_retry(
                lambda: self._bucket.list_objects_v2(
                    prefix=full_prefix, continuation_token=token, max_keys=1000
                )
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
