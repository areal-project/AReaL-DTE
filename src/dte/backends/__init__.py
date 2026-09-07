# Licensed under the Apache License, Version 2.0
"""dte.backends — pluggable Transport implementations.

loopback is always available (CPU, zero deps). The awex backend is imported
lazily and degrades gracefully: if the optional ``awex`` dependency (or anything
it needs) is missing, ``AwexTransport`` is ``None`` instead of raising at import.
"""

from dte.backends.loopback import LoopbackTransport
from dte.backends.oss_store import OSSStore

__all__ = ["LoopbackTransport", "OSSStore"]

try:  # optional: pip install delta-transfer-engine[awex]
    from dte.backends.awex_backend import AwexTransport

    __all__ += ["AwexTransport"]
except Exception:  # pragma: no cover - missing awex or its deps
    AwexTransport = None

try:  # optional: pip install delta-transfer-engine[mooncake]
    from dte.backends.mooncake_backend import MooncakeTransport

    __all__ += ["MooncakeTransport"]
except Exception:  # pragma: no cover - missing mooncake or its deps
    MooncakeTransport = None

try:  # optional: pip install delta-transfer-engine[http]
    from dte.backends.http_backend import HttpTransport, S3Store, SharedFSStore

    __all__ += ["HttpTransport", "SharedFSStore", "S3Store"]
except Exception:  # pragma: no cover - missing serialization or store dependencies
    HttpTransport = None
    SharedFSStore = None
    S3Store = None
