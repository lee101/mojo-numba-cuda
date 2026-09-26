"""The two kernels numba-cuda ships, compiled to Mojo and run on the CPU.

numba.cuda 0.30.4 is a JIT compiler, a driver binding, a simulator and a
dispatch layer.  The arithmetic a user runs on the GPU lives in their own
kernel, not in this package.  The only kernels the package itself ships are in
``numba/cuda/kernels/``: ``Reduce`` (block-strided, 32-lane butterfly, four-warp
combine, one partial per block, a final pass) and ``transpose`` (the tiled
shared-memory transpose).  Those are what this module ports, together with the
host-side launch arithmetic that decides the grid.
"""

from __future__ import annotations

import ctypes
import numpy as np

from ._lib import (
    MAX_BLOCKSIZE,
    MAX_FULL_BLOCKS,
    NUMWARPS,
    WARPSIZE,
    _addr,
    lib,
    scratch,
)

__all__ = [
    "MAX_BLOCKSIZE",
    "MAX_FULL_BLOCKS",
    "NUMWARPS",
    "WARPSIZE",
    "launch_shape",
    "reduce_sum",
    "tile_shape",
    "transpose",
]

_CTYPES = {"float64": ctypes.c_double, "float32": ctypes.c_float,
           "int64": ctypes.c_int64}
_KERNEL = {"float64": "ncu_reduce_f64", "float32": "ncu_reduce_f32",
           "int64": "ncu_reduce_i64"}
_FINAL = {"float64": "ncu_reduce_final_f64", "float32": "ncu_reduce_final_f32",
          "int64": "ncu_reduce_final_i64"}

# A reduction is memory-bound: one pass over the array with a serial dependency
# chain of additions per lane.  The toolchain reference measured a threaded
# 4M-element axpy as *slower* than serial, and that is the same shape as this.
# So the grid runs in one call and no thread pool is used.


def launch_shape(n: int, blocksize: int = MAX_BLOCKSIZE) -> tuple:
    """The grid ``Reduce.__call__`` would choose for ``n`` elements.

    Returns ``(size_full, size_partial, full_blockct, partials_size)`` with the
    upstream formulas: the array is split into a whole number of blocks plus a
    ragged tail, at most ``_WARPSIZE * 2`` full blocks are launched, and the
    tail gets one block of its own.
    """
    out = np.empty(4, dtype=np.int32)
    lib.ncu_launch_shape(n, blocksize, _addr(out))
    return tuple(int(v) for v in out)


def tile_shape(max_threads_per_block: int = 1024) -> tuple:
    """The tile ``transpose`` derives from the device's thread limit.

    ``tile_width = 2 ** (log2(tpb) / 2)``, ``tile_height = tpb // tile_width``,
    and the shared tile is padded by one column, which on a GPU is what keeps
    the transposed stores off the same bank.
    """
    out = np.empty(3, dtype=np.int32)
    lib.ncu_tile_shape(max_threads_per_block, _addr(out))
    return tuple(int(v) for v in out)


def reduce_sum(arr: np.ndarray, init=0) -> "np.generic":
    """Sum ``arr`` the way ``numba.cuda.kernels.Reduce(lambda a, b: a + b)`` does.

    Same two phases and the same tree order: each block strides through the
    array with ``blocksize * nblocks`` as the step, folds its lanes with a
    halving butterfly, combines its four warps pairwise, and leaves one partial;
    a final pass folds the partials.  A tree is deterministic, so with the
    default ``init=0`` the float result is the one the device kernel produces
    for the same launch shape -- which is *not* ``np.sum``, whose pairwise order
    differs.  With a non-zero ``init`` the value is added at the end rather than
    folded into block 0's tree as upstream does, so the last bits may differ.

    Integer sums are exact and order-independent either way.
    """
    arr = np.ascontiguousarray(arr)
    name = arr.dtype.name
    if name not in _KERNEL:
        raise TypeError(
            f"dtype {arr.dtype} is not one of float64, float32, int64"
        )
    if arr.ndim != 1:
        raise TypeError("only support 1D array")
    n = arr.size
    if n < 1:
        # Reduce returns the init value for an empty array without launching.
        return arr.dtype.type(init)

    size_full, size_partial, full_blockct, partials_size = launch_shape(n)
    partials = np.zeros(partials_size, dtype=arr.dtype)
    lanes = scratch(arr.dtype)
    reduce_fn = getattr(lib, _KERNEL[name])
    ctype = _CTYPES[name]
    item = arr.itemsize
    base = _addr(arr)

    if size_full:
        # one call for the whole full-block region; the kernel loops over the
        # blocks itself.  No thread pool: a reduction is memory-bound, and the
        # toolchain's own guidance is that chunking those makes them slower.
        reduce_fn(size_full, MAX_BLOCKSIZE, full_blockct, ctype(0), base,
                  _addr(partials), _addr(lanes))
    if size_partial:
        # the tail: one block, sized to the remainder, launched on its own
        reduce_fn(size_partial, size_partial, 1, ctype(0),
                  base + item * size_full,
                  _addr(partials) + item * full_blockct, _addr(lanes))

    out = np.zeros(1, dtype=arr.dtype)
    getattr(lib, _FINAL[name])(partials_size, _addr(partials), _addr(out),
                               _addr(lanes))
    return out[0] + arr.dtype.type(init)


def transpose(a: np.ndarray, max_threads_per_block: int = 1024) -> np.ndarray:
    """``numba.cuda.kernels.transpose``: the tiled transpose, on the CPU.

    The tile geometry comes from the device thread limit exactly as upstream
    derives it.  The pad column that avoids bank conflicts on a GPU has no
    meaning here, so it is not in the kernel; the ragged edges are, and they are
    where the bounds checks live.
    """
    a = np.ascontiguousarray(a)
    if a.ndim != 2:
        raise TypeError("transpose takes a 2D array")
    if a.dtype.name not in ("float32", "float64"):
        raise TypeError("transpose is ported for float32 and float64")
    rows, cols = a.shape
    tile_width, tile_height, _padded = tile_shape(max_threads_per_block)
    b = np.empty((cols, rows), dtype=a.dtype)
    if rows and cols:
        tile = np.zeros(tile_width * tile_height, dtype=a.dtype)
        symbol = "ncu_transpose_f64" if a.dtype.name == "float64" else \
            "ncu_transpose_f32"
        getattr(lib, symbol)(rows, cols, tile_width, tile_height, _addr(a),
                             _addr(b), _addr(tile))
    return b
