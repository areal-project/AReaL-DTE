# Licensed under the Apache License, Version 2.0
"""Delta Transfer Engine (dte): incremental weight sync for RL training.

dte is the main subject — it owns the incremental algorithm (``dte.core``) and
defines its own transport contract (``dte.transport``). Transport engines such
as awex are pluggable backends under ``dte.backends``.
"""

__version__ = "0.0.1"

from dte.engine import DeltaEngine, PushResult
from dte.transport import Payload, Plan, TransferOp, Transport

__all__ = [
    "DeltaEngine",
    "PushResult",
    "Transport",
    "TransferOp",
    "Plan",
    "Payload",
]
