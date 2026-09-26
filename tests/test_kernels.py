"""Tests for mojo-numba-cuda's two kernels.

The parity target is analytic truth, not the upstream package, and the reason
is recorded in the README: numba-cuda 0.30.4 cannot execute its own kernels in
this environment (it imports ``np.row_stack``, removed in NumPy 2.5), and its
CUDA simulator cannot either (``numba.cuda`` exposes no ``threadIdx`` there).
The algorithms are transcribed from ``numba/cuda/kernels/reduction.py`` and
``transpose.py``; where the upstream host-side arithmetic is the thing under
test -- the launch shape and the tile geometry -- the expected values are the
upstream expressions, written out in the tests.
"""

import math

import numpy as np
import pytest

import mojo_numba_cuda as mc

WARPSIZE = 32
MAX_BLOCKSIZE = 128
MAX_FULL_BLOCKS = 64


def upstream_launch_shape(n, blocksize=MAX_BLOCKSIZE):
    """Reduce.__call__, verbatim from numba/cuda/kernels/reduction.py."""
    size_full = (n // blocksize) * blocksize
    size_partial = n - size_full
    full_blockct = min(size_full // blocksize, WARPSIZE * 2)
    partials_size = full_blockct + (1 if size_partial else 0)
    return size_full, size_partial, full_blockct, partials_size


def upstream_tile_shape(tpb):
    """transpose(), verbatim from numba/cuda/kernels/transpose.py."""
    tile_width = int(2 ** (math.log(tpb, 2) / 2))
    tile_height = int(tpb / tile_width)
    return tile_width, tile_height, tile_width + 1


# ------------------------------------------------------------- reduction


@pytest.mark.parametrize("n", [1, 2, 5, 31, 32, 33, 63, 64, 65, 127, 128, 129,
                               255, 256, 1000, 8192, 9000, 100000, 1 << 20])
def test_integer_reduction_is_exact(n):
    """Integer addition is associative, so the tree order cannot change the
    answer: this is an exact comparison, and a dropped lane or a missed tail
    block shows up immediately."""
    a = np.arange(n, dtype=np.int64)
    assert int(mc.reduce_sum(a)) == int(a.sum())


@pytest.mark.parametrize("n", [1, 5, 33, 127, 128, 129, 1000, 9000, 100000,
                               1 << 20])
def test_float64_reduction_matches_the_exact_sum(n):
    """A tree of doubles is accurate to well under 1e-12 relative, so the
    comparison is against the exactly-rounded sum with a tight tolerance."""
    rng = np.random.default_rng(20260926)
    a = rng.standard_normal(n)
    got = float(mc.reduce_sum(a))
    want = math.fsum(a.tolist())
    assert got == pytest.approx(want, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("n", [1, 33, 129, 1000, 9000, 100000])
def test_float32_reduction_tolerance_is_the_summation_order(n):
    """float32 is the one place a tolerance is needed, and it is not the
    kernel's rounding: it is that a tree sum and numpy's pairwise sum round
    differently.  1e-6 relative is ~8 float32 ulps over this range."""
    rng = np.random.default_rng(7)
    a = rng.standard_normal(n).astype(np.float32)
    got = float(mc.reduce_sum(a))
    want = float(np.float64(a.astype(np.float64)).sum())
    assert got == pytest.approx(want, rel=1e-6, abs=1e-6)


def test_empty_reduction_returns_the_init():
    """Reduce returns init without launching for an empty array."""
    for init in (0, 7):
        assert int(mc.reduce_sum(np.array([], dtype=np.int64), init=init)) == init


def test_init_is_added_once():
    a = np.arange(1000, dtype=np.int64)
    assert int(mc.reduce_sum(a, init=7)) == int(a.sum()) + 7
    # not once per block: a 8192-element array runs 64 full blocks
    b = np.arange(8192, dtype=np.int64)
    assert int(mc.reduce_sum(b, init=7)) == int(b.sum()) + 7


def test_ragged_tail_block_is_folded_sequentially():
    """A partially populated warp is summed in order, not with a halving tree:
    a tree over 5 lanes would drop lanes 4 and 5.  n = 130 is 128 + a tail of
    2, n = 8195 a tail of 3, and 131 a tail of 3 as well."""
    for n, tail in ((129, 1), (130, 2), (131, 3), (8195, 3), (9000, 40),
                    (9001, 41), (100001, 33)):
        _, size_partial, _, _ = mc.launch_shape(n)
        assert size_partial == tail % MAX_BLOCKSIZE
        a = np.arange(n, dtype=np.int64)
        assert int(mc.reduce_sum(a)) == int(a.sum())


@pytest.mark.parametrize("n", [0, 1, 31, 32, 127, 128, 129, 896, 897, 1000,
                               8191, 8192, 8193, 100000, 1 << 20, 1 << 24])
def test_launch_shape_matches_the_upstream_arithmetic(n):
    assert mc.launch_shape(n) == upstream_launch_shape(n)


def test_launch_shape_caps_the_block_count():
    # 64 full blocks is the cap, so a huge array gets a ragged-free grid of 64
    # plus nothing: every block strides further
    assert mc.launch_shape(1 << 20) == (1 << 20, 0, 64, 64)
    assert mc.launch_shape((1 << 20) + 5) == (1 << 20, 5, 64, 65)
    # the partials buffer is at most 65 entries, which is what lets the final
    # pass run in a single 128-thread block
    assert mc.launch_shape(1 << 24)[3] <= 65


def test_reduction_rejects_what_upstream_rejects():
    with pytest.raises(TypeError):
        mc.reduce_sum(np.zeros((4, 4), dtype=np.float64))
    with pytest.raises(TypeError):
        mc.reduce_sum(np.arange(10, dtype=np.int32))


# ------------------------------------------------------------- transpose


@pytest.mark.parametrize("shape", [(1, 1), (2, 3), (3, 4), (5, 7), (16, 16),
                                  (31, 33), (32, 32), (33, 31), (64, 65),
                                  (127, 129), (255, 256), (17, 1), (1, 17),
                                  (1024, 3)])
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_transpose_is_exact(shape, dtype):
    a = np.arange(shape[0] * shape[1], dtype=dtype).reshape(shape)
    assert np.array_equal(mc.transpose(a), a.T)


def test_transpose_of_a_ragged_tile_region():
    """129x127 with a 32x32 tile: the last tile row and column are both
    partial, which is where the bounds checks are."""
    rows, cols = 129, 127
    a = np.random.default_rng(3).standard_normal((rows, cols))
    b = mc.transpose(a)
    assert b.shape == (cols, rows)
    assert np.array_equal(b, a.T)


@pytest.mark.parametrize("tpb", [32, 64, 128, 256, 512, 1024])
def test_tile_shape_matches_the_upstream_derivation(tpb):
    assert mc.tile_shape(tpb) == upstream_tile_shape(tpb)


def test_tile_shape_for_the_real_device_limit():
    # every CUDA device since Fermi reports 1024 max threads per block
    assert mc.tile_shape(1024) == (32, 32, 33)


def test_transpose_rejects_what_it_cannot_do():
    with pytest.raises(TypeError):
        mc.transpose(np.arange(10, dtype=np.float64))
    with pytest.raises(TypeError):
        mc.transpose(np.arange(10, dtype=np.int64).reshape(2, 5))


def test_transpose_is_its_own_inverse():
    rng = np.random.default_rng(11)
    a = rng.standard_normal((37, 53))
    assert np.array_equal(mc.transpose(mc.transpose(a)), a)


# ----------------------------------------------------------------- sizes


def test_buffers_above_four_gib(high_address_f64):
    a = high_address_f64
    assert a.ctypes.data >= (1 << 32)
    n = a.size - (a.size % MAX_BLOCKSIZE)
    view = a[:n]
    view[:] = 1.0
    got = int(mc.reduce_sum(view))
    assert got == n
    square = view[: 4096].reshape(64, 64)
    assert np.array_equal(mc.transpose(square), square.T)
