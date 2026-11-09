"""Small shared persistence helpers (atomic file writes)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_write(path: str | os.PathLike, data: bytes) -> None:
    """Write `data` to `path` via temp-file + os.replace (same directory),
    so a crash never leaves a truncated/partial manifest behind."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_text(path: str | os.PathLike, text: str) -> None:
    atomic_write(path, text.encode("utf-8"))