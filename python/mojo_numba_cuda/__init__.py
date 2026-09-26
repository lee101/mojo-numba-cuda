"""mojo-numba-cuda: numba-cuda's own two kernels, compiled to Mojo.

Importable alongside the real ``numba.cuda``; nothing here shadows it.
"""

from . import core
from ._lib import lib
from .core import (
    MAX_BLOCKSIZE,
    MAX_FULL_BLOCKS,
    NUMWARPS,
    WARPSIZE,
    launch_shape,
    reduce_sum,
    tile_shape,
    transpose,
)

__version__ = "0.1.0"

__all__ = [
    "MAX_BLOCKSIZE",
    "MAX_FULL_BLOCKS",
    "NUMWARPS",
    "WARPSIZE",
    "core",
    "launch_shape",
    "lib",
    "reduce_sum",
    "tile_shape",
    "transpose",
]
