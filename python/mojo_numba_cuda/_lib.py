"""ctypes bridge to the compiled Mojo kernels.

Buffers cross the C ABI as 64-bit addresses; ``c_int`` truncates them.  The
arity of every declaration is checked at import, because a short ``argtypes``
list is silently zero-filled by ctypes rather than rejected.

The shim owns every buffer, including the ones the kernel would otherwise have
to allocate: Mojo 1.2.0 has no ``Pointer.alloc``, so the 128 lane slots of the
block reduction and the transpose tile are allocated here and passed in.
"""

import ctypes
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-numba-cuda.so"

_I = ctypes.c_int64
_F64 = ctypes.c_double
_F32 = ctypes.c_float
_U32 = ctypes.c_uint32

# The block geometry of numba/cuda/kernels/reduction.py.
WARPSIZE = 32
NUMWARPS = 4
MAX_BLOCKSIZE = WARPSIZE * NUMWARPS  # 128
MAX_FULL_BLOCKS = WARPSIZE * 2  # 64

_SIGNATURES = {
    "ncu_launch_shape": (None, [_I, _I, _I], 3),
    "ncu_tile_shape": (None, [_I, _I], 2),
    "ncu_reduce_f64": (None, [_I, _I, _I, _F64, _I, _I, _I], 7),
    "ncu_reduce_f32": (None, [_I, _I, _I, _F32, _I, _I, _I], 7),
    "ncu_reduce_i64": (None, [_I, _I, _I, ctypes.c_int64, _I, _I, _I], 7),
    "ncu_reduce_final_f64": (None, [_I, _I, _I, _I], 4),
    "ncu_reduce_final_f32": (None, [_I, _I, _I, _I], 4),
    "ncu_reduce_final_i64": (None, [_I, _I, _I, _I], 4),
    "ncu_transpose_f64": (None, [_I, _I, _I, _I, _I, _I, _I], 7),
    "ncu_transpose_f32": (None, [_I, _I, _I, _I, _I, _I, _I], 7),
}


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"{_LIB_PATH} not found; run `bash build/build.sh` first"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    for name, (restype, argtypes, arity) in _SIGNATURES.items():
        try:
            fn = getattr(lib, name)
        except AttributeError as exc:  # pragma: no cover - build mismatch
            raise RuntimeError(f"{_LIB_PATH} is missing symbol {name!r}") from exc
        if len(argtypes) != arity:  # pragma: no cover - build mismatch
            raise RuntimeError(
                f"arity table for {name!r} says {arity}, argtypes has "
                f"{len(argtypes)}"
            )
        fn.restype = restype
        fn.argtypes = argtypes
    return lib


lib = _load()


def _addr(a) -> int:
    if not a.flags["C_CONTIGUOUS"]:
        raise ValueError("buffer must be C-contiguous")
    return a.ctypes.data


def scratch(dtype) -> np.ndarray:
    """The 128 lane slots a block reduces in, owned here."""
    return np.zeros(MAX_BLOCKSIZE, dtype=dtype)
