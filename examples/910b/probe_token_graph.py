# SPDX-License-Identifier: Apache-2.0
"""910B PA feasibility probe: startup capture, device metadata inplace replay.

This isolates attention with a pre-populated ND KV cache; it does not validate
model integration, cache-write kernels, speculative acceptance, or performance.
"""
import argparse
import json
from pathlib import Path

import torch


def expand(bucket, lengths, contexts, blocks):
    if (not lengths or len(lengths) != len(contexts) or sum(lengths) > bucket
            or blocks.shape[0] < len(lengths)
            or any(q <= 0 or c < q for q, c in zip(lengths, contexts))):
        raise ValueError("Invalid token partition or context lengths")
    lens = torch.ones(bucket, dtype=torch.int32)
    tables = torch.zeros(bucket, blocks.shape[1], dtype=torch.int32)
    start = 0
    for row, (q, c) in enumerate(zip(lengths, contexts)):
        lens[start:start + q] = torch.arange(c - q + 1, c + 1, dtype=torch.int32)
        tables[start:start + q] = blocks[row]
        start += q
    return lens, tables


def reference(query, key, value, blocks, lengths, contexts):
    """Independent request-level causal GQA reference on logical ND cache."""
    output, offset = [], 0
    repeat = query.shape[1] // key.shape[2]
    for row, (q, c) in enumerate(zip(lengths, contexts)):
        pos = torch.arange(c)
        ids = blocks[row, pos // key.shape[1]].long()
        k = key[ids, pos % key.shape[1]].float().repeat_interleave(repeat, dim=1)
        v = value[ids, pos % key.shape[1]].float().repeat_interleave(repeat, dim=1)
        scores = torch.einsum("qhd,khd->hqk", query[offset:offset + q].float(), k)
        scores *= query.shape[-1] ** -0.5
        visible = pos[None, :] <= torch.arange(c - q, c)[:, None]
        scores.masked_fill_(~visible[None], float("-inf"))
        output.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), v))
        offset += q
    return torch.cat(output)


def cases(bucket, control):
    dense = [1] * min(bucket, 20)
    if control != "all":
        if control == "qlens":
            rows = min(8, bucket)
            q = [bucket // rows] * rows
            q[-1] += bucket - sum(q)
            return [q, q[1:] + q[:1], [1] * (rows - 1) + [bucket - rows + 1], q]
        return [dense] * 4
    known = {20: [[7, 7, 1, 1, 1, 1, 1, 1], [2] * 10,
                  [1, 1, 1, 1, 1, 5], [1, 1, 1, 1, 7, 7]],
             80: [[1, 1, 1, 1, 5, 70], [20] * 4],
             192: [[70, 40, 70], [192]]}
    return [dense] + known.get(bucket, [[bucket]]) + [dense]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buckets", type=int, nargs="+", default=[20, 80, 192])
    parser.add_argument("--control", choices=["inputs", "contexts", "blocks", "qlens", "all"], default="all")
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--max-context", type=int, default=512)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--graph-pool", choices=["private", "shared"], default="private")
    parser.add_argument("--eager-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("910b-token-graph.json"))
    args = parser.parse_args()
    if (min(args.buckets + [args.heads, args.kv_heads, args.head_size, args.block_size,
                           args.max_context, args.repeats]) <= 0
            or len(set(args.buckets)) != len(args.buckets) or args.heads % args.kv_heads
            or args.head_size % 16 or max(args.buckets) > args.max_context):
        parser.error("Require positive unique buckets <= max-context, heads divisible by kv-heads, D aligned to 16")
    import torch_npu  # Device dependency is deliberately lazy for CPU reference tests.

    report = {"passed": False, "config": {**vars(args), "output": str(args.output)},
              "captures": 0, "cases": [], "stage": "init", "operator": "_npu_paged_attention"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.set_num_threads(1)
        torch.manual_seed(910)
        torch.npu.set_device(args.device)
        device, dtype = f"npu:{args.device}", getattr(torch, args.dtype)
        report.update(torch=torch.__version__, torch_npu=torch_npu.__version__,
                      device=torch.npu.get_device_name(args.device))
        print("[ENV]", json.dumps(report), flush=True)
        width = (args.max_context + args.block_size - 1) // args.block_size
        shape = (1 + 20 * width, args.block_size, args.kv_heads, args.head_size)
        key_cpu, value_cpu = (torch.randn(shape, dtype=dtype) for _ in range(2))
        key, value = key_cpu.to(device), value_cpu.to(device)
        base_blocks = torch.arange(1, 1 + 20 * width, dtype=torch.int32).reshape(20, width)
        stream = torch.npu.Stream(device=args.device)
        torch.npu.synchronize()
        pool = torch.npu.graph_pool_handle() if args.graph_pool == "shared" else None
        states = {}

        def prepare(bucket, qlens, iteration):
            state = states[bucket]
            torch.npu.synchronize()
            change_inputs = args.control in ("all", "inputs")
            generator = torch.Generator().manual_seed(910 + (iteration if change_inputs else 0))
            q = torch.randn(bucket, args.heads, args.head_size, dtype=dtype, generator=generator)
            if change_inputs:
                key_cpu.copy_(torch.randn(shape, dtype=dtype, generator=generator))
                value_cpu.copy_(torch.randn(shape, dtype=dtype, generator=generator))
                key.copy_(key_cpu)
                value.copy_(value_cpu)
            blocks = base_blocks.roll(iteration if args.control in ("all", "blocks") else 0, dims=0)
            if args.control in ("all", "contexts"):
                choices = [1, args.block_size, min(args.max_context, args.block_size + 1), args.max_context]
                contexts = [max(n, choices[(i + iteration) % 4]) for i, n in enumerate(qlens)]
            else:
                contexts = [args.max_context] * len(qlens)
            lens, tables = expand(bucket, qlens, contexts, blocks)
            state["q"].copy_(q)
            state["lens"].copy_(lens)
            state["blocks"].copy_(tables)
            expected = reference(q, key_cpu, value_cpu, blocks, qlens, contexts)
            report["inputs"] = {"bucket": bucket, "scheduled": qlens, "contexts": contexts,
                                "operator_contexts": lens.tolist()}
            return expected

        def run(state):
            torch_npu._npu_paged_attention(
                query=state["q"], key_cache=key, value_cache=value,
                num_heads=args.heads, num_kv_heads=args.kv_heads, scale_value=args.head_size ** -0.5,
                block_table=state["blocks"], context_lens=state["lens"], out=state["out"])

        def check(state, expected, label):
            actual = state["out"][:len(expected)].float().cpu().clone()
            if not torch.isfinite(actual).all():
                raise AssertionError(f"{label}: non-finite output")
            try:
                torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
            except AssertionError:
                path = args.output.with_suffix(".failure.pt")
                torch.save({"actual": actual, "expected": expected, "inputs": report["inputs"],
                            "label": label}, path)
                report["failure_file"] = str(path)
                raise
            return (actual - expected).abs().max().item()

        with torch.npu.stream(stream):
            for bucket in sorted(args.buckets, reverse=True):
                state = {"q": torch.empty(bucket, args.heads, args.head_size, dtype=dtype, device=device),
                         "out": torch.empty(bucket, args.heads, args.head_size, dtype=dtype, device=device),
                         "lens": torch.ones(bucket, dtype=torch.int32, device=device),
                         "blocks": torch.zeros(bucket, width, dtype=torch.int32, device=device)}
                states[bucket] = state
                expected = prepare(bucket, cases(bucket, args.control)[0], 0)
                report["stage"] = "eager_warmup"
                run(state)
                torch.npu.synchronize()
                check(state, expected, "startup eager")
                if not args.eager_only:
                    report["stage"] = "capture"
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph, stream=stream, pool=pool):
                        run(state)
                    state["graph"] = graph
                    report["captures"] += 1
                    print("[CAPTURE]", bucket, flush=True)
            # All captures finish before the first dynamic test. Revisit buckets.
            for repeat in range(args.repeats):
                for bucket in args.buckets:
                    for index, qlens in enumerate(cases(bucket, args.control)):
                        iteration = repeat * len(cases(bucket, args.control)) + index
                        state = states[bucket]
                        expected = prepare(bucket, qlens, iteration)
                        if not args.eager_only:
                            state["out"].fill_(float("nan"))
                            torch.npu.synchronize()
                            report["stage"] = "replay"
                            state["graph"].replay()
                            torch.npu.synchronize()
                            # Check replay before eager can refresh any ATB setup state.
                            diff = check(state, expected, "replay")
                        report["stage"] = "eager_reference"
                        run(state)
                        torch.npu.synchronize()
                        eager_diff = check(state, expected, "dynamic eager")
                        if args.eager_only:
                            diff = eager_diff
                        row = {**report["inputs"], "repeat": repeat, "max_abs_diff": diff,
                               "graph_id": id(state["graph"]) if "graph" in state else None,
                               "context_ptr": state["lens"].data_ptr(), "blocks_ptr": state["blocks"].data_ptr()}
                        report["cases"].append(row)
                        print("[PASS]", json.dumps(row), flush=True)
        report.update(passed=True, stage="complete")
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"PASS: {report['captures']} graphs, {len(report['cases'])} cases. Report: {args.output}")


if __name__ == "__main__":
    main()
