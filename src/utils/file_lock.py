"""Small cross-process advisory locks for immutable file publication."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
from pathlib import Path
from typing import Iterator


@contextmanager
def exclusive_file_lock(path: str | Path) -> Iterator[None]:
    """Hold an exclusive lock on ``path`` until the publication is complete."""

    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


__all__ = ["exclusive_file_lock"]
