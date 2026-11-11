# SPDX-License-Identifier: Apache-2.0
"""Startup-selected sparse auxiliary row writes, without destination scans.

The caller guarantees unique, in-range addresses, including distinct padding
sinks. Only tensor metadata is checked on the host. No per-row device reads,
destination clone/initialization, synchronization, or runtime exception fallback
is added. Warm this writer through the existing profile/capture preparation.
"""

import importlib

import torch


def _geometry(cache, addresses, updates):
    if (
        cache.ndim != 2
        or updates.ndim != 2
        or cache.shape[1] != updates.shape[1]
        or cache.shape[1] < 1
        or addresses.ndim not in (1, 2)
        or (addresses.ndim == 2 and addresses.shape[1] != 1)
        or addresses.numel() != updates.shape[0]
        or addresses.dtype not in (torch.int32, torch.int64)
        or updates.dtype != cache.dtype
        or updates.device != cache.device
        or addresses.device != cache.device
        or not all(value.is_contiguous() for value in (cache, addresses, updates))
    ):
        raise ValueError(
            "MTP row writer requires aligned contiguous cache/addresses/updates"
        )
    return updates.shape[0], updates.shape[1]


def _load_triton_writer():
    utilities = importlib.import_module("vllm.triton_utils")
    if not getattr(utilities, "HAS_TRITON", False):
        raise ImportError("vLLM Triton backend is unavailable")
    name = ".asr_mtp_row_copy_kernel" if __package__ else "asr_mtp_row_copy_kernel"
    implementation = importlib.import_module(name, package=__package__ or None)
    kernel, triton = implementation.row_copy_kernel, utilities.triton

    def write(cache, addresses, updates):
        rows, width = _geometry(cache, addresses, updates)
        if not rows:
            return
        columns = triton.next_power_of_2(width)
        tile_rows = max(1, min(16, 4096 // columns))
        kernel[(triton.cdiv(rows, tile_rows),)](
            cache,
            addresses,
            updates,
            ROWS=rows,
            WIDTH=width,
            DEST_ROWS=cache.shape[0],
            BLOCK_ROWS=tile_rows,
            BLOCK_COLS=columns,
        )

    return write


def prepare_row_writer(device, *, backend="auto"):
    """Return ``(writer, name)``; CPU keeps the caller's index_copy_ reference.

    ``auto`` keeps the native ScatterNd writer without importing Triton. The
    opt-in ``triton`` backend fails at startup when unavailable. Import selection
    is performed once, not inside the writer or graph replay. A runtime/JIT error
    propagates instead of silently switching semantics or adding synchronization.
    """
    if backend not in ("auto", "triton", "scatter_nd"):
        raise ValueError("MTP row write backend must be auto, triton or scatter_nd")
    kind = device.split(":", 1)[0] if isinstance(device, str) else device.type
    if kind == "cpu":
        return None, "index_copy"
    if kind != "npu":
        raise ValueError("MTP row writer requires a CPU or NPU device")
    if backend == "triton":
        try:
            return _load_triton_writer(), "triton"
        except ImportError as error:
            raise RuntimeError("MTP triton row writer is unavailable") from error
    api = importlib.import_module("torch_npu")
    operation = getattr(api, "npu_scatter_nd_update_", None)
    if operation is None:
        raise RuntimeError("MTP ScatterNd row writer is unavailable")
    if not callable(operation):
        raise TypeError("MTP ScatterNd row writer API must be callable")

    def write(cache, addresses, updates):
        rows, _ = _geometry(cache, addresses, updates)
        if rows:
            operation(cache, addresses.reshape(-1, 1), updates)

    return write, "scatter_nd"
