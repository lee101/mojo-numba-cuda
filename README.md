# mojo-numba-cuda

[Numba CUDA](https://numba.readthedocs.io/en/stable/cuda/index.html) 0.30.4 is
a JIT compiler, a CUDA driver binding, a fake-CUDA simulator and a dispatch
layer. It has almost no arithmetic of its own — the work a user runs on the GPU
lives in *their* kernel. This port takes the two kernels the package actually
ships, in `numba/cuda/kernels/`, and compiles them to Mojo.

```
PYTHONPATH=python python -c "
import numpy as np, mojo_numba_cuda as mc
a = np.arange(1 << 20, dtype=np.int64)
assert int(mc.reduce_sum(a)) == int(a.sum())
assert np.array_equal(mc.transpose(np.arange(12.0).reshape(3, 4)), np.arange(12.0).reshape(3, 4).T)
print('ok')"
```

## What is and is not here

**The compiler is not a kernel.** `compiler.py`, `cudaimpl.py`, `lowering.py`,
`typing/`, `debuginfo.py`, `itanium_mangler.py`, `cudadrv/`, `descriptor.py`,
`dispatcher.py`, `serialize.py`, the simulator, `libdevicefuncs.py` and
`vector_types.py` are all plumbing: they turn Python into PTX, or they move
bytes to the device. There is no array loop in any of them to move into Mojo,
and porting an NVVM code generator would be a new compiler, not a port.

**What the package ships as kernels is two files**, and both are ported here:

| Upstream (`numba/cuda/kernels/`) | Ported | Kernel |
| --- | --- | --- |
| `reduction.py` — `Reduce`: block-strided loop, 32-lane butterfly, four-warp combine, one partial per block, final pass | yes, f64/f32/i64 | `ncu_reduce_*`, `ncu_reduce_final_*` |
| `reduction.py` — the host-side launch arithmetic (`size_full`, `size_partial`, `full_blockct`, `partials_size`) | yes | `ncu_launch_shape` |
| `transpose.py` — the tiled transpose and its tile geometry from the device thread limit | yes, f64/f32 | `ncu_transpose_*`, `ncu_tile_shape` |
| `bf16.py`, `fp16.py`, `fp8.py`, `vector_types.py` | no | thin wrappers over device intrinsics; `bf16` cannot even be imported here (below) |
| `cudamath.py`, `libdevicefuncs.py`, `intrinsics.py` | no | signatures for device intrinsics |
| `launchconfig.py`, `cudadecl.py`, `codegen.py`, `cudadrv/`, `simulator/`, `typing/`, `debuginfo.py` | no | plumbing |
| `deviceufunc.py` | no | builds a ufunc from a *user's* device function |

## Parity target: analytic, not the package — and why

**numba-cuda 0.30.4 cannot execute its own kernels in this environment.** Two
independent reasons, both verified here:

```
>>> from numba.cuda.kernels.reduction import Reduce   # imports fine
>>> Reduce(lambda a, b: a + b)(np.arange(10))
AttributeError: module 'numpy' has no attribute 'row_stack'
```

`numba/cuda/bf16.py` → `fp16.py` → `_internal/cuda_bf16.py` →
`typing/builtins.py` → `cudaimpl.py` → `np/npyimpl.py` → `np/arrayobj.py`, which
calls `np.row_stack` at import time. NumPy 2.5.3 is installed and removed it.
The failure is in numba-cuda's own import chain, not in this port. The
simulator path fails differently: under `NUMBA_ENABLE_CUDASIM=1` the kernel
compiles and launches, then dies on `module 'numba.cuda' has no attribute
'threadIdx'`.

So the parity targets are:

* **integer reductions, exactly** — integer addition is associative, so the tree
  order cannot change the answer, and a dropped lane or a missed tail block
  shows up immediately;
* **float64 reductions against `math.fsum`**, the exactly-rounded sum, at
  `rel=1e-12`;
* **float32 reductions at `rel=1e-6`** — the one place a tolerance is needed,
  and it is not the kernel's rounding: a tree sum and NumPy's pairwise sum
  round differently;
* **transposes, exactly**, against `a.T`;
* **the launch-shape and tile-geometry arithmetic against the upstream
  expressions themselves**, transcribed into the tests from
  `reduction.py` and `transpose.py` with the source lines named.

The tree order of the reduction matches the device kernel's by construction —
each lane accumulates sequentially, warps fold through a halving butterfly, and
the four warp results combine pairwise — so for `init=0` the float result is
the one the GPU would produce for the same launch shape. That is a claim about
the algorithm, not a measurement: upstream could not be run to confirm it.

## Deliberate differences

* **`init` is added at the end** rather than folded into block 0's tree as
  upstream does, so with a non-zero `init` the last bits of a float sum may
  differ. Integer sums are exact either way, and that is what the tests check.
* **The pad column is gone.** `transpose.py` allocates a
  `(tile_height, tile_width + 1)` shared tile; the extra column is what keeps
  the transposed stores off the same bank on a GPU. On a CPU it means nothing,
  so the kernel has no pad. The tile geometry, the ragged edges and the bounds
  checks are all still there.
* **No thread pool.** A reduction is memory-bound with a serial dependency
  chain per lane; the toolchain reference measured a threaded 4M-element axpy
  as *slower* than serial, which is the same shape. The grid runs in one call.
* **float32/int32/int16 reductions are not ported** — three dtypes were enough
  to pin the integer exactness and the float order, and each extra width is
  another export to keep honest.

## Two bugs the tests caught while porting

**One loop per lane, not one loop per block.** Upstream's thread `tid` owns
`start = tid + blocksize * block` and then strides by `blocksize * nblocks`. A
single loop from the block base visits only lane 0 and writes one partial per
block; with 64 blocks that is 64 elements out of 8192. The kernel now loops
over lanes and assigns the first element before accumulating, which is what
`tmp = arr[start]` followed by `tmp = reduce_op(tmp, arr[i])` does.

**A partially populated warp cannot use the halving tree.** The tree halves
`width` — over 5 lanes it folds 0+2, 1+3, then 0+1, and lanes 4 and 5 are
never added. Upstream's partially-populated branch sums sequentially, and so
does this, for every warp that is not exactly 32 lanes wide. The ragged-tail
sizes 1, 2, 3, 40, 41 and 33 are each in the tests.

## Not implemented

Everything in the "not here" table above, with the reason given. There is no
GPU path: the box has an RTX 5090 and Mojo's `DeviceContext` would compile
these two kernels for it, but the package's own kernels could not be run here
to check against, and a GPU port of a port with no verifiable reference is not
worth the risk on a shared device.

## Install and run

```bash
source /nvme0n1-disk/mojo-toolchain/activate.sh   # mojo 1.2.0.dev2026092605
bash build/build.sh                               # -> dist/libmojo-numba-cuda.so
PYTHONPATH=python /nvme0n1-disk/mojo-toolchain/testvenv/bin/python -m pytest tests -q
```

`tests/conftest.py` skips the module if the shared library has not been built.

## Tests

95 tests, all against analytic truth or against the upstream expressions.

* **Integer reductions, exact**, at 19 sizes chosen to straddle every boundary
  in the launch arithmetic: 1, 2, 5, 31/32/33, 63/64/65, 127/128/129, 255/256,
  1000, 8192, 9000, 100000, 2^20.
* **float64 against `math.fsum`** at `rel=1e-12`, and float32 at `rel=1e-6`
  with the reason written down.
* **The launch shape at 17 sizes**, including the two that matter most: 2^20
  gives 64 blocks and no tail, 2^20 + 5 gives 64 blocks and a tail of 5, and
  the partials buffer never exceeds 65 entries — which is what lets the final
  pass fit in one 128-thread block.
* **The tile geometry at six thread limits**, and specifically 32x32x33 for the
  1024 that every CUDA device reports.
* **Transposes at 14 shapes x 2 dtypes**, including 1x1, single-row and
  single-column, prime dimensions, and shapes whose last tile row *and* column
  are both partial (129x127 with a 32x32 tile).
* **A buffer above 4 GiB in address space**, because a ctypes `argtypes` list
  one entry short is silent: ctypes zero-fills the missing slot, the address is
  truncated, and the kernel writes to the low 4 GiB of the heap instead.

## Benchmarks

Xeon E5-2697 v4, shared with other work, min of 7 runs. The gate compares an
exact integer reduction over 1 Mi elements and a 1024² transpose against NumPy
before anything is timed.

| kernel | Mojo | NumPy | ratio |
| --- | ---: | ---: | ---: |
| reduce_sum f64, n = 1 024 | 0.08 ms | 0.00 ms | **0.03x** |
| reduce_sum f64, n = 65 536 | 0.27 ms | 0.04 ms | **0.14x** |
| reduce_sum f64, n = 1 048 576 | 6.61 ms | 0.56 ms | **0.08x** |
| reduce_sum f64, n = 16 777 216 | 334 ms | 17.5 ms | **0.05x** |
| transpose f64, 256² | 0.47 ms | 0.22 ms | **0.46x** |
| transpose f64, 1024² | 11.1 ms | 7.7 ms | **0.70x** |
| transpose f64, 4096² | 273 ms | 347 ms | **1.27x** |

**The reduction is a loss, and it is not close.** `np.sum` is a vectorised
pairwise reduction — eight accumulators, SIMD, one pass — and this is a serial
scalar tree that exists to mirror a GPU kernel whose whole point is the
parallelism. Running the GPU algorithm's structure on one core was always going
to lose; the honest result is the table above. The transpose reaches parity
only once the working set is large enough for the tiling to pay for its extra
pass through the tile.

Small inputs are dominated by the boundary: `reduce_sum` costs 31–54 us against
`np.sum`'s 1.5–2.4 us, because the shim makes three ctypes calls and allocates
the 128 lane slots each time.

## Layout

```
build/build.sh              src/kernels.mojo -> dist/libmojo-numba-cuda.so
src/kernels.mojo           one compilation unit, 10 exports
python/mojo_numba_cuda/    _lib.py (ctypes), core.py (API), __init__.py
tests/                      test_kernels.py, conftest.py
bench/bench.py              correctness-gated timing
```

Licence: BSD-2-Clause, as upstream (NVIDIA).
