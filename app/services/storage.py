import uuid
from pathlib import Path
from typing import BinaryIO

from app.core.exceptions import PayloadTooLargeError

_COPY_CHUNK_BYTES = 1024 * 1024


class LocalStorage:
    """Stores dataset files on the local filesystem (or a mounted volume).

    Keys are relative paths; every resolved path is checked to stay inside the root.
    Swap this class for an object-store implementation (e.g. S3) with the same interface.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.tmp_dir = self.root / ".tmp"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"Storage key escapes storage root: {key!r}")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def delete(self, key: str) -> None:
        self.path_for(key).unlink(missing_ok=True)

    def new_temp_path(self, suffix: str) -> Path:
        return self.tmp_dir / f"{uuid.uuid4().hex}{suffix}"

    def is_writable(self) -> bool:
        probe = self.new_temp_path(".probe")
        try:
            probe.write_bytes(b"ok")
            probe.unlink()
            return True
        except OSError:
            return False


def copy_with_limit(source: BinaryIO, destination: Path, max_bytes: int) -> int:
    """Copy a file object to disk, aborting once it exceeds max_bytes. Returns bytes written."""
    written = 0
    try:
        with destination.open("wb") as out:
            while chunk := source.read(_COPY_CHUNK_BYTES):
                written += len(chunk)
                if written > max_bytes:
                    raise PayloadTooLargeError(
                        f"File exceeds the maximum upload size of {max_bytes // (1024 * 1024)} MB"
                    )
                out.write(chunk)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return written
