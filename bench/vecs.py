"""Read and write .fvecs / .ivecs files with NumPy.

Each row in the file is: [int32 dim][dim values].
"""

from pathlib import Path

import numpy as np


def mmap_vecs(path: Path, dtype=np.float32) -> np.ndarray:
    """Return a (rows, dim) view of the file. Data is loaded from disk only when used."""
    raw = np.memmap(path, dtype=np.int32, mode="r")
    if raw.size == 0:
        return np.empty((0, 0), dtype=dtype)
    dim = int(raw[0])
    if dim <= 0 or raw.size % (dim + 1) != 0:
        raise ValueError(f"{path}: broken vecs file")
    rows = raw.reshape(-1, dim + 1)
    if not (rows[:, 0] == dim).all():
        raise ValueError(f"{path}: rows have different dims")
    # Drop the dim column and read the remaining bytes as the real type.
    return rows[:, 1:].view(dtype)


class VecsWriter:
    """Writes rows to a .fvecs/.ivecs file a chunk at a time."""

    def __init__(self, path: Path, dtype):
        self.path = path
        self.dtype = np.dtype(dtype)
        self.rows = 0
        self.dim = None
        self._f = open(path, "wb")

    def write(self, chunk: np.ndarray) -> None:
        chunk = np.ascontiguousarray(chunk, dtype=self.dtype)
        if chunk.ndim != 2:
            raise ValueError("chunk must be 2-D")
        n, d = chunk.shape
        if self.dim is None:
            self.dim = d
        elif d != self.dim:
            raise ValueError(f"dim changed from {self.dim} to {d}")
        out = np.empty((n, d + 1), dtype=np.int32)
        out[:, 0] = d
        out[:, 1:] = chunk.view(np.int32)
        self._f.write(out.tobytes())
        self.rows += n

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
