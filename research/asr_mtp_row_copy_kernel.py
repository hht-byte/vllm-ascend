# SPDX-License-Identifier: Apache-2.0
"""Sparse row-copy kernel imported only by the NPU startup writer factory."""

from vllm.triton_utils import tl, triton


@triton.jit
def row_copy_kernel(
    cache,
    addresses,
    updates,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    DEST_ROWS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    column = tl.arange(0, BLOCK_COLS)
    destination = tl.load(addresses + row, mask=row < ROWS, other=0).to(tl.int64)
    valid = (row[:, None] < ROWS) & (column[None, :] < WIDTH)
    value = tl.load(
        updates + row[:, None] * WIDTH + column[None, :], mask=valid, other=0
    )
    valid = valid & (destination[:, None] >= 0) & (destination[:, None] < DEST_ROWS)
    tl.store(cache + destination[:, None] * WIDTH + column[None, :], value, mask=valid)
