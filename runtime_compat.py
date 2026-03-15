from __future__ import annotations

import os


def ensure_runtime_compatibility() -> None:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
