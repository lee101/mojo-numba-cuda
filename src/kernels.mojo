from std.math import floor, log2

# numba-cuda -- the two kernels the package actually ships, in Mojo.
#
# What numba-cuda 0.30.4 is: a JIT compiler (NVVM + libdevice), a CUDA driver
# binding, a fake-CUDA simulator, a type-inference and dispatch layer, and a
# vector-type / intrinsic surface.  All of that is plumbing: the arithmetic a
# user runs on the GPU lives in *their* kernel, not in this package.  There is
# no numeric core to port in a compiler.
#
# What numba-cuda does ship is exactly two kernels of its own, in
# numba/cuda/kernels/:
#
#   reduction.py  Reduce -- a block-strided reduction with a 32-lane butterfly
#                tree inside each block, a four-warp block combine, one partial
#                per block, and a final pass over the partials.  The host side
#                also carries real arithmetic: the blocksize, the split of the
#                array into a full-block part and a ragged tail, and the number
#                of blocks.
#   transpose.py  transpose -- the shared-memory tiled transpose, with the
#                tile width derived from the device's max threads per block and
#                a pad column that avoids bank conflicts.
#
# Those are ported here, algorithm for algorithm, including the launch-shape
# arithmetic and the order in which the tree adds its terms (a tree reduction
# is deterministic, so the float results are the same ones the GPU kernel
# produces for the same launch shape).
#
# The GPU-shaped parts stay GPU-shaped: a block is 4 warps of 32 lanes, the
# butterfly halves the lane count 32 -> 1, and the block combine folds the four
# warp results pairwise.  On a CPU those are just loops, and this port says so
# rather than pretending otherwise.

comptime WARPSIZE: Int = 32
comptime NUMWARPS: Int = 4
comptime MAX_BLOCKSIZE: Int = WARPSIZE * NUMWARPS  # 128, as in reduction.py
comptime MAX_FULL_BLOCKS: Int = WARPSIZE * 2         # 64, as in reduction.py


@export("ncu_launch_shape")
def ncu_launch_shape(n: Int, blocksize: Int, out_ptr: Int) abi("C"):
    # The host-side arithmetic of Reduce.__call__, verbatim:
    #     size_full    = (arr.size // blocksize) * blocksize
    #     size_partial = arr.size - size_full
    #     full_blockct = min(size_full // blocksize, _WARPSIZE * 2)
    #     partials_size = full_blockct + (1 if size_partial else 0)
    # out = [size_full, size_partial, full_blockct, partials_size]
    var o = Pointer[Int32, AnyOrigin[mut=True]](unsafe_from_address=out_ptr)
    var size_full = (n // blocksize) * blocksize
    var size_partial = n - size_full
    var full_blockct = size_full // blocksize
    if full_blockct > MAX_FULL_BLOCKS:
        full_blockct = MAX_FULL_BLOCKS
    var partials_size = full_blockct
    if size_partial > 0:
        partials_size += 1
    o[unsafe_offset=0] = Int32(size_full)
    o[unsafe_offset=1] = Int32(size_partial)
    o[unsafe_offset=2] = Int32(full_blockct)
    o[unsafe_offset=3] = Int32(partials_size)


@export("ncu_tile_shape")
def ncu_tile_shape(max_threads: Int, out_ptr: Int) abi("C"):
    # transpose.py:
    #     tile_width  = int(2 ** (log2(tpb) / 2))
    #     tile_height = tpb // tile_width
    #     tile_shape  = (tile_height, tile_width + 1)   # +1 pad column
    # out = [tile_width, tile_height, padded_width]
    var o = Pointer[Int32, AnyOrigin[mut=True]](unsafe_from_address=out_ptr)
    var tile_width = Int32(2.0 ** (log2(Float64(max_threads)) / 2.0))
    var tile_height = max_threads // Int(tile_width)
    o[unsafe_offset=0] = tile_width
    o[unsafe_offset=1] = Int32(tile_height)
    o[unsafe_offset=2] = tile_width + Int32(1)


# -------------------------------------------------------------- reduction


@always_inline
def _tree_reduce_f64(v: Pointer[Float64, AnyOrigin[mut=True]], base: Int, count: Int) -> Float64:
    # Halving tree over `count` slots starting at `base`, exactly the shape of
    # inner_warp_reduction (width = 32 -> 1) followed by the four-warp combine.
    var width = count
    while width > 1:
        width = width // 2
        var t = 0
        while t < width:
            v[unsafe_offset=base + t] = v[unsafe_offset=base + t] + v[
                unsafe_offset=base + t + width
            ]
            t += 1
    return v[unsafe_offset=base]


@always_inline
def _tree_reduce_f32(v: Pointer[Float32, AnyOrigin[mut=True]], base: Int, count: Int) -> Float32:
    var width = count
    while width > 1:
        width = width // 2
        var t = 0
        while t < width:
            v[unsafe_offset=base + t] = v[unsafe_offset=base + t] + v[
                unsafe_offset=base + t + width
            ]
            t += 1
    return v[unsafe_offset=base]


@always_inline
def _tree_reduce_i64(v: Pointer[Int64, AnyOrigin[mut=True]], base: Int, count: Int) -> Int64:
    var width = count
    while width > 1:
        width = width // 2
        var t = 0
        while t < width:
            v[unsafe_offset=base + t] = v[unsafe_offset=base + t] + v[
                unsafe_offset=base + t + width
            ]
            t += 1
    return v[unsafe_offset=base]


@always_inline
def _block_f64(
    a: Pointer[Float64, AnyOrigin[mut=True]],
    lanes: Pointer[Float64, AnyOrigin[mut=True]],
    n: Int, block: Int, blocksize: Int, nblocks: Int, init: Float64,
    partials: Pointer[Float64, AnyOrigin[mut=True]],
) -> None:
    for t in range(MAX_BLOCKSIZE):
        lanes[unsafe_offset=t] = init
    var step = blocksize * nblocks
    var base = block * blocksize
    # One loop per lane, not one loop over the block: upstream's thread `tid`
    # owns start = tid + blocksize * block and then strides by `step`, so lane 0
    # alone does not cover the block.  The first element is assigned and the
    # rest accumulated, which is what `tmp = arr[start]` then
    # `tmp = reduce_op(tmp, arr[i])` does.
    var tid = 0
    while tid < blocksize:
        var i = base + tid
        var have = False
        while i < n:
            if have:
                lanes[unsafe_offset=tid] = lanes[unsafe_offset=tid] + a[unsafe_offset=i]
            else:
                lanes[unsafe_offset=tid] = a[unsafe_offset=i]
                have = True
            i += step
        tid += 1
    var w = 0
    while w < NUMWARPS:
        var active = blocksize - w * WARPSIZE
        if active > WARPSIZE:
            active = WARPSIZE
        if active == WARPSIZE:
            _ = _tree_reduce_f64(lanes, w * WARPSIZE, active)
        elif active > 1:
            # a partially populated warp: upstream sums it sequentially, and a
            # halving tree would drop lanes when the count is not a power of two
            var acc = lanes[unsafe_offset=w * WARPSIZE]
            var k = 1
            while k < active:
                acc = acc + lanes[unsafe_offset=w * WARPSIZE + k]
                k += 1
            lanes[unsafe_offset=w * WARPSIZE] = acc
        w += 1
    if blocksize > 2 * WARPSIZE:
        lanes[unsafe_offset=0] = lanes[unsafe_offset=0] + lanes[unsafe_offset=2 * WARPSIZE]
        lanes[unsafe_offset=WARPSIZE] = lanes[unsafe_offset=WARPSIZE] + lanes[
            unsafe_offset=3 * WARPSIZE
        ]
    var result = lanes[unsafe_offset=0]
    if blocksize > WARPSIZE:
        result = result + lanes[unsafe_offset=WARPSIZE]
    partials[unsafe_offset=block] = result


@always_inline
def _block_f32(
    a: Pointer[Float32, AnyOrigin[mut=True]],
    lanes: Pointer[Float32, AnyOrigin[mut=True]],
    n: Int, block: Int, blocksize: Int, nblocks: Int, init: Float32,
    partials: Pointer[Float32, AnyOrigin[mut=True]],
) -> None:
    for t in range(MAX_BLOCKSIZE):
        lanes[unsafe_offset=t] = init
    var step = blocksize * nblocks
    var base = block * blocksize
    # One loop per lane, not one loop over the block: upstream's thread `tid`
    # owns start = tid + blocksize * block and then strides by `step`, so lane 0
    # alone does not cover the block.  The first element is assigned and the
    # rest accumulated, which is what `tmp = arr[start]` then
    # `tmp = reduce_op(tmp, arr[i])` does.
    var tid = 0
    while tid < blocksize:
        var i = base + tid
        var have = False
        while i < n:
            if have:
                lanes[unsafe_offset=tid] = lanes[unsafe_offset=tid] + a[unsafe_offset=i]
            else:
                lanes[unsafe_offset=tid] = a[unsafe_offset=i]
                have = True
            i += step
        tid += 1
    var w = 0
    while w < NUMWARPS:
        var active = blocksize - w * WARPSIZE
        if active > WARPSIZE:
            active = WARPSIZE
        if active == WARPSIZE:
            _ = _tree_reduce_f32(lanes, w * WARPSIZE, active)
        elif active > 1:
            # a partially populated warp: upstream sums it sequentially, and a
            # halving tree would drop lanes when the count is not a power of two
            var acc = lanes[unsafe_offset=w * WARPSIZE]
            var k = 1
            while k < active:
                acc = acc + lanes[unsafe_offset=w * WARPSIZE + k]
                k += 1
            lanes[unsafe_offset=w * WARPSIZE] = acc
        w += 1
    if blocksize > 2 * WARPSIZE:
        lanes[unsafe_offset=0] = lanes[unsafe_offset=0] + lanes[unsafe_offset=2 * WARPSIZE]
        lanes[unsafe_offset=WARPSIZE] = lanes[unsafe_offset=WARPSIZE] + lanes[
            unsafe_offset=3 * WARPSIZE
        ]
    var result = lanes[unsafe_offset=0]
    if blocksize > WARPSIZE:
        result = result + lanes[unsafe_offset=WARPSIZE]
    partials[unsafe_offset=block] = result


@always_inline
def _block_i64(
    a: Pointer[Int64, AnyOrigin[mut=True]],
    lanes: Pointer[Int64, AnyOrigin[mut=True]],
    n: Int, block: Int, blocksize: Int, nblocks: Int, init: Int64,
    partials: Pointer[Int64, AnyOrigin[mut=True]],
) -> None:
    for t in range(MAX_BLOCKSIZE):
        lanes[unsafe_offset=t] = init
    var step = blocksize * nblocks
    var base = block * blocksize
    # One loop per lane, not one loop over the block: upstream's thread `tid`
    # owns start = tid + blocksize * block and then strides by `step`, so lane 0
    # alone does not cover the block.  The first element is assigned and the
    # rest accumulated, which is what `tmp = arr[start]` then
    # `tmp = reduce_op(tmp, arr[i])` does.
    var tid = 0
    while tid < blocksize:
        var i = base + tid
        var have = False
        while i < n:
            if have:
                lanes[unsafe_offset=tid] = lanes[unsafe_offset=tid] + a[unsafe_offset=i]
            else:
                lanes[unsafe_offset=tid] = a[unsafe_offset=i]
                have = True
            i += step
        tid += 1
    var w = 0
    while w < NUMWARPS:
        var active = blocksize - w * WARPSIZE
        if active > WARPSIZE:
            active = WARPSIZE
        if active == WARPSIZE:
            _ = _tree_reduce_i64(lanes, w * WARPSIZE, active)
        elif active > 1:
            # a partially populated warp: upstream sums it sequentially, and a
            # halving tree would drop lanes when the count is not a power of two
            var acc = lanes[unsafe_offset=w * WARPSIZE]
            var k = 1
            while k < active:
                acc = acc + lanes[unsafe_offset=w * WARPSIZE + k]
                k += 1
            lanes[unsafe_offset=w * WARPSIZE] = acc
        w += 1
    if blocksize > 2 * WARPSIZE:
        lanes[unsafe_offset=0] = lanes[unsafe_offset=0] + lanes[unsafe_offset=2 * WARPSIZE]
        lanes[unsafe_offset=WARPSIZE] = lanes[unsafe_offset=WARPSIZE] + lanes[
            unsafe_offset=3 * WARPSIZE
        ]
    var result = lanes[unsafe_offset=0]
    if blocksize > WARPSIZE:
        result = result + lanes[unsafe_offset=WARPSIZE]
    partials[unsafe_offset=block] = result


@export("ncu_reduce_f64")
def ncu_reduce_f64(
    n: Int, blocksize: Int, nblocks: Int, init: Float64, in_ptr: Int,
    partials_ptr: Int, scratch_ptr: Int
) abi("C"):
    # The whole grid in one call: every block's block-strided loop, its
    # 32-lane butterfly and its four-warp combine, writing one partial per
    # block.  `n` is the size of this chunk; the shim splits the array the way
    # upstream splits it, into a full-block part and a ragged tail.
    var a = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=in_ptr)
    var partials = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=partials_ptr)
    var lanes = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=scratch_ptr)
    for b in range(nblocks):
        _block_f64(a, lanes, n, b, blocksize, nblocks, init, partials)


@export("ncu_reduce_f32")
def ncu_reduce_f32(
    n: Int, blocksize: Int, nblocks: Int, init: Float32, in_ptr: Int,
    partials_ptr: Int, scratch_ptr: Int
) abi("C"):
    var a = Pointer[Float32, AnyOrigin[mut=True]](unsafe_from_address=in_ptr)
    var partials = Pointer[Float32, AnyOrigin[mut=True]](unsafe_from_address=partials_ptr)
    var lanes = Pointer[Float32, AnyOrigin[mut=True]](unsafe_from_address=scratch_ptr)
    for b in range(nblocks):
        _block_f32(a, lanes, n, b, blocksize, nblocks, init, partials)


@export("ncu_reduce_i64")
def ncu_reduce_i64(
    n: Int, blocksize: Int, nblocks: Int, init: Int64, in_ptr: Int,
    partials_ptr: Int, scratch_ptr: Int
) abi("C"):
    var a = Pointer[Int64, AnyOrigin[mut=True]](unsafe_from_address=in_ptr)
    var partials = Pointer[Int64, AnyOrigin[mut=True]](unsafe_from_address=partials_ptr)
    var lanes = Pointer[Int64, AnyOrigin[mut=True]](unsafe_from_address=scratch_ptr)
    for b in range(nblocks):
        _block_i64(a, lanes, n, b, blocksize, nblocks, init, partials)


@export("ncu_reduce_final_f64")
def ncu_reduce_final_f64(
    npartials: Int, partials_ptr: Int, out_ptr: Int, scratch_ptr: Int
) abi("C"):
    # The final pass: one block, the same tree, over at most 65 partials.
    var partials = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=partials_ptr)
    var o = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=out_ptr)
    var lanes = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=scratch_ptr)
    for t in range(MAX_BLOCKSIZE):
        lanes[unsafe_offset=t] = partials[unsafe_offset=t] if t < npartials else 0.0
    var count = 1
    while count < npartials:
        count = count * 2
    o[unsafe_offset=0] = _tree_reduce_f64(lanes, 0, count)


@export("ncu_reduce_final_f32")
def ncu_reduce_final_f32(
    npartials: Int, partials_ptr: Int, out_ptr: Int, scratch_ptr: Int
) abi("C"):
    var partials = Pointer[Float32, AnyOrigin[mut=True]](unsafe_from_address=partials_ptr)
    var o = Pointer[Float32, AnyOrigin[mut=True]](unsafe_from_address=out_ptr)
    var lanes = Pointer[Float32, AnyOrigin[mut=True]](unsafe_from_address=scratch_ptr)
    for t in range(MAX_BLOCKSIZE):
        lanes[unsafe_offset=t] = partials[unsafe_offset=t] if t < npartials else Float32(0.0)
    var count = 1
    while count < npartials:
        count = count * 2
    o[unsafe_offset=0] = _tree_reduce_f32(lanes, 0, count)


@export("ncu_reduce_final_i64")
def ncu_reduce_final_i64(
    npartials: Int, partials_ptr: Int, out_ptr: Int, scratch_ptr: Int
) abi("C"):
    var partials = Pointer[Int64, AnyOrigin[mut=True]](unsafe_from_address=partials_ptr)
    var o = Pointer[Int64, AnyOrigin[mut=True]](unsafe_from_address=out_ptr)
    var lanes = Pointer[Int64, AnyOrigin[mut=True]](unsafe_from_address=scratch_ptr)
    for t in range(MAX_BLOCKSIZE):
        lanes[unsafe_offset=t] = partials[unsafe_offset=t] if t < npartials else 0
    var count = 1
    while count < npartials:
        count = count * 2
    o[unsafe_offset=0] = _tree_reduce_i64(lanes, 0, count)


# -------------------------------------------------------------- transpose


@export("ncu_transpose_f64")
def ncu_transpose_f64(
    rows: Int, cols: Int, tile_width: Int, tile_height: Int, in_ptr: Int,
    out_ptr: Int, tile_ptr: Int
) abi("C"):
    # The tiled transpose of kernels/transpose.py, with the index arithmetic
    # kept intact: one "block" per tile, gathered into a tile_height x
    # tile_width buffer and written back transposed.  The pad column upstream
    # adds to dodge bank conflicts means nothing on a CPU, so it is not here;
    # the ragged edges are, and they are where the bounds checks live.
    var a = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=in_ptr)
    var o = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=out_ptr)
    var tw = tile_width
    var th = tile_height
    var tile = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=tile_ptr)
    var nby = (rows + th - 1) // th
    var nbx = (cols + tw - 1) // tw
    var by = 0
    while by < nby:
        var bx = 0
        while bx < nbx:
            var ty = 0
            while ty < th:
                var tx = 0
                while tx < tw:
                    var r = by * th + ty
                    var c = bx * tw + tx
                    if r < rows and c < cols:
                        tile[unsafe_offset=ty * tw + tx] = a[unsafe_offset=r * cols + c]
                    tx += 1
                ty += 1
            # write the tile back transposed
            ty = 0
            while ty < th:
                var tx = 0
                while tx < tw:
                    var r = by * th + ty
                    var c = bx * tw + tx
                    if r < rows and c < cols:
                        o[unsafe_offset=c * rows + r] = tile[unsafe_offset=ty * tw + tx]
                    tx += 1
                ty += 1
            bx += 1
        by += 1


@export("ncu_transpose_f32")
def ncu_transpose_f32(
    rows: Int, cols: Int, tile_width: Int, tile_height: Int, in_ptr: Int,
    out_ptr: Int, tile_ptr: Int
) abi("C"):
    var a = Pointer[Float32, AnyOrigin[mut=True]](unsafe_from_address=in_ptr)
    var o = Pointer[Float32, AnyOrigin[mut=True]](unsafe_from_address=out_ptr)
    var tw = tile_width
    var th = tile_height
    var tile = Pointer[Float32, AnyOrigin[mut=True]](unsafe_from_address=tile_ptr)
    var nby = (rows + th - 1) // th
    var nbx = (cols + tw - 1) // tw
    var by = 0
    while by < nby:
        var bx = 0
        while bx < nbx:
            var ty = 0
            while ty < th:
                var tx = 0
                while tx < tw:
                    var r = by * th + ty
                    var c = bx * tw + tx
                    if r < rows and c < cols:
                        tile[unsafe_offset=ty * tw + tx] = a[unsafe_offset=r * cols + c]
                    tx += 1
                ty += 1
            ty = 0
            while ty < th:
                var tx = 0
                while tx < tw:
                    var r = by * th + ty
                    var c = bx * tw + tx
                    if r < rows and c < cols:
                        o[unsafe_offset=c * rows + r] = tile[unsafe_offset=ty * tw + tx]
                    tx += 1
                ty += 1
            bx += 1
        by += 1
