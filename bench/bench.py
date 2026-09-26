"""Benchmark the two ported kernels against the fastest NumPy formulation.

Correctness is gated first: an integer reduction, which is exact and therefore
order-independent, and a transpose, are compared against NumPy before any
timing is printed.
"""

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import mojo_numba_cuda as mc  # noqa: E402


def timeit(fn, repeats=7):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    rng = np.random.default_rng(20260926)
    a = rng.integers(0, 1000, size=1 << 20, dtype=np.int64)
    assert int(mc.reduce_sum(a)) == int(a.sum()), "reduction gate failed"
    m = rng.standard_normal((1024, 1024))
    assert np.array_equal(mc.transpose(m), m.T), "transpose gate failed"
    print("gate ok: reduction exact on 1 Mi int64, transpose exact on 1024^2")

    print()
    print(f"{'kernel':<36}{'mojo':>11}{'numpy':>11}{'ratio':>9}{'GiB/s':>9}")
    for n in (1 << 10, 1 << 16, 1 << 20, 1 << 24):
        x = rng.standard_normal(n)
        tm = timeit(lambda: mc.reduce_sum(x))
        tn = timeit(lambda: x.sum())
        print(f"{'reduce_sum f64, n=' + str(n):<36}{tm * 1e3:>9.2f}ms"
              f"{tn * 1e3:>9.2f}ms{tn / tm:>8.2f}x{x.nbytes / tm / (1 << 30):>9.2f}")
    for shape in ((256, 256), (1024, 1024), (4096, 4096)):
        m = rng.standard_normal(shape)
        tm = timeit(lambda: mc.transpose(m))
        tn = timeit(lambda: m.T.copy())
        print(f"{'transpose f64, ' + str(shape):<36}{tm * 1e3:>9.2f}ms"
              f"{tn * 1e3:>9.2f}ms{tn / tm:>8.2f}x{m.nbytes / tm / (1 << 30):>9.2f}")

    print()
    print("small inputs, where the ctypes call is the whole cost:")
    for n in (1, 128, 1024):
        x = rng.standard_normal(n)
        tm = timeit(lambda: mc.reduce_sum(x), 200)
        tn = timeit(lambda: x.sum(), 200)
        print(f"  reduce_sum n={n:<8}{tm * 1e6:>8.2f}us{tn * 1e6:>8.2f}us"
              f"{tn / tm:>7.2f}x")


if __name__ == "__main__":
    main()
