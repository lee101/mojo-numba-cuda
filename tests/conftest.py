import pathlib
import sys

import numpy as np
import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))

_LIB = _ROOT / "dist" / "libmojo-numba-cuda.so"

if not _LIB.exists():
    pytest.skip(
        "libmojo-numba-cuda.so not built; run `bash build/build.sh`",
        allow_module_level=True,
    )


@pytest.fixture(scope="session")
def high_address_f64():
    """A float64 array whose address is above 4 GiB.

    A ctypes ``argtypes`` list one entry short is silent: ctypes zero-fills the
    missing slot, the address is truncated, and the kernel then reads and writes
    the low 4 GiB of the heap.  Multi-megabyte allocations come back from mmap
    on the high half of the address space.
    """
    a = np.zeros(8 * 1024 * 1024, dtype=np.float64)  # 64 MiB
    if a.ctypes.data < (1 << 32):
        pytest.skip("allocator returned a low address; cannot test truncation")
    return a
