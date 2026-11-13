# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A/B only position metadata preparation; does not measure end-to-end ASR latency."""

import argparse
import importlib.util
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "npu"], default="npu")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--query", type=int, default=6)
    parser.add_argument("--capacity", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=300)
    args = parser.parse_args()
    n, t = args.requests, args.requests * args.query
    if min(n, args.query, args.iterations) < 1 or t > args.capacity:
        parser.error("positive sizes required and requests*query must fit capacity")
    if args.device == "npu":
        import torch_npu  # noqa: F401

        sync = torch.npu.synchronize
        event = torch.npu.Event()
    else:
        sync = lambda: None
        event = None
    src = Path(__file__).resolve().parents[2] / "vllm_ascend/worker/sync_input_metadata.py"
    spec = importlib.util.spec_from_file_location("sync_input_metadata", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    pinned = args.device == "npu"

    def host(values, dtype):
        result = torch.empty(len(values), dtype=dtype, pin_memory=pinned)
        result.copy_(torch.as_tensor(values, dtype=dtype))
        return result

    counts = host(np.arange(n) + 30, torch.int32)
    ids = host(np.repeat(np.arange(n), args.query), torch.int64)
    query = host(np.tile(np.arange(args.query), n), torch.int64)
    scheduled = host([args.query] * n, torch.int32)
    d_counts = torch.empty(n, device=args.device, dtype=torch.int32)
    d_ids = torch.empty(t, device=args.device, dtype=torch.int64)
    d_query = torch.empty(t, device=args.device, dtype=torch.int64)
    d_sched = torch.empty(n, device=args.device, dtype=torch.int32)
    pos = torch.zeros(args.capacity, device=args.device, dtype=torch.int64)
    seq = torch.zeros(n + 8, device=args.device, dtype=torch.int32)
    packed = mod.SyncInputMetadata(args.capacity, n + 8, args.device, pin_memory=pinned, copy_event=event)
    pos_np = counts.numpy()[ids.numpy()] + query.numpy()
    seq_np = counts.numpy() + scheduled.numpy()

    def baseline():
        d_counts.copy_(counts, non_blocking=True)
        d_ids.copy_(ids, non_blocking=True)
        d_query.copy_(query, non_blocking=True)
        d_sched.copy_(scheduled, non_blocking=True)
        pos[:t] = d_counts[d_ids].to(torch.int64) + d_query
        seq[:n] = d_counts + d_sched
        seq[n:].fill_(0)

    def optimized():
        packed.prepare(counts.numpy(), seq_np, pos_np)

    baseline()
    optimized()
    sync()
    torch.testing.assert_close(pos[:t], packed.positions[:t])
    torch.testing.assert_close(seq, packed.seq_lens)
    results = {"device": args.device, "requests": n, "tokens": t, "packed_copy_bytes": packed.last_copy_bytes}
    # Measure both orders to reduce one-sided warmup/order bias. Synchronization
    # brackets the interval, not each operator; host reuse is protected by helper.
    for name, fn in [
        ("baseline", baseline),
        ("packed", optimized),
        ("packed_repeat", optimized),
        ("baseline_repeat", baseline),
    ]:
        for _ in range(20):
            fn()
        sync()
        start = perf_counter()
        for _ in range(args.iterations):
            fn()
        sync()
        results[name + "_ms"] = (perf_counter() - start) * 1000 / args.iterations
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
