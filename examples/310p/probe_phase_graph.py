# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture native FA/PA buckets up front, then test in-place metadata replay.

Run each family in a fresh process. A failed NPU capture may poison the process.
No task_update, lazy capture, recapture or operator fallback is used.
"""

import argparse
import ast
import json
from pathlib import Path

import torch
import torch_npu
from probe_token_graph import (
    BLOCK_SIZE,
    CASES,
    HEAD_SIZE,
    HEADS,
    KV_HEADS,
    MAX_CONTEXT,
    MAX_REQS,
    NZ_FORMAT,
    check,
    reference,
    sync,
    tg,
)


def native_operator():
    """Execute the real backend methods without importing the whole vLLM engine."""
    source = Path(__file__).resolve().parents[2] / "vllm_ascend/_310p/attention/attention_v1.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    backend = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "AscendAttentionBackendImpl310")
    names = {"_flash_attention", "forward_prefill_310", "forward_paged_attention"}
    methods = [node for node in backend.body if isinstance(node, ast.FunctionDef) and node.name in names]
    cls = ast.ClassDef(name="NativeOperator", bases=[], keywords=[], body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    import __future__

    env = dict(torch=torch, torch_npu=torch_npu, MASK_TYPE_NORM_COMPRESS_SELF_ATTENTION=3)
    exec(compile(module, str(source), "exec", flags=__future__.annotations.compiler_flag), env)
    op = env["NativeOperator"]()
    op.support_compressed_mask = True
    op.num_heads, op.num_kv_heads, op.scale = HEADS, KV_HEADS, HEAD_SIZE**-0.5
    return op


def flash_reference(query, key, value, qlens):
    outputs = []
    offset = 0
    for length in qlens:
        k = key[offset:offset + length].float().repeat_interleave(HEADS // KV_HEADS, dim=1)
        v = value[offset:offset + length].float().repeat_interleave(HEADS // KV_HEADS, dim=1)
        for i in range(length):
            scores = torch.einsum("hd,lhd->hl", query[offset + i].float(), k[:i + 1]) * HEAD_SIZE**-0.5
            outputs.append(torch.einsum("hl,lhd->hd", scores.softmax(-1), v[:i + 1]))
        offset += length
    return torch.stack(outputs)


def tensor_descriptor(tensor):
    return dict(shape=list(tensor.shape), dtype=str(tensor.dtype), device=str(tensor.device),
                stride=list(tensor.stride()), storage_offset=tensor.storage_offset(), ptr=tensor.data_ptr(),
                npu_format=torch_npu.get_npu_format(tensor) if tensor.device.type != "cpu" else None)


def main():
    from types import SimpleNamespace

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("prefill", "decode"), required=True)
    parser.add_argument("--buckets", type=int, nargs="+", default=[20, 80, 192])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--eager-only", action="store_true", help="Validate native inputs without any graph capture")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or len(set(args.buckets)) != len(args.buckets) or any(b not in CASES for b in args.buckets):
        parser.error("Use unique buckets from 20,80,192 and positive repeats")
    torch.set_num_threads(1)
    torch.manual_seed(310)
    torch.npu.set_device(0)
    device = "npu:0"
    max_blocks = MAX_CONTEXT // BLOCK_SIZE
    num_blocks = 1 + MAX_REQS * max_blocks
    arena = tg.NativeGraphArena(max(args.buckets), MAX_REQS, max_blocks, MAX_CONTEXT, device, args.family)
    op = native_operator()
    shape = (num_blocks, KV_HEADS * HEAD_SIZE // 16, BLOCK_SIZE, 16)
    op.key_cache = torch_npu.empty_with_format(size=shape, dtype=torch.float16, device=device, acl_format=NZ_FORMAT)
    op.value_cache = torch_npu.empty_with_format(size=shape, dtype=torch.float16, device=device, acl_format=NZ_FORMAT)
    key_nd = torch.randn(num_blocks, BLOCK_SIZE, KV_HEADS, HEAD_SIZE, dtype=torch.float16)
    value_nd = torch.randn_like(key_nd)
    torch_npu._npu_reshape_and_cache(
        key_nd.flatten(0, 1).to(device), value_nd.flatten(0, 1).to(device), op.key_cache, op.value_cache,
        torch.arange(num_blocks * BLOCK_SIZE, dtype=torch.int32, device=device),
    )
    blocks = torch.arange(1, num_blocks, dtype=torch.int32).reshape(MAX_REQS, max_blocks)
    pool = torch.npu.graph_pool_handle()
    buffers, graphs = {}, {}
    report = dict(family=args.family, torch=torch.__version__, torch_npu=torch_npu.__version__,
                  operator="_npu_flash_attention" if args.family == "prefill" else "_npu_paged_attention",
                  device=torch.npu.get_device_name(0), eager_only=args.eager_only,
                  captures=0, eager_checks=[], cases=[], passed=False)
    for bucket in args.buckets:
        buffers[bucket] = (
            torch.zeros(bucket, HEADS, HEAD_SIZE, device=device, dtype=torch.float16),
            torch.zeros(bucket, KV_HEADS, HEAD_SIZE, device=device, dtype=torch.float16),
            torch.zeros(bucket, KV_HEADS, HEAD_SIZE, device=device, dtype=torch.float16),
            torch.empty(bucket, HEADS, HEAD_SIZE, device=device, dtype=torch.float16),
        )

    def prepare(bucket, qlens, iteration):
        sync()
        q, k, v, output = buffers[bucket]
        contexts = qlens if args.family == "prefill" else [
            (1, 64, 128, 130)[(i + iteration) % 4] for i in range(len(qlens))
        ]
        tables = blocks.roll(iteration % MAX_REQS, dims=0)
        slots = [int(tables[row, p // BLOCK_SIZE]) * BLOCK_SIZE + p % BLOCK_SIZE
                 for row, (length, end) in enumerate(zip(qlens, contexts)) for p in range(end - length, end)]
        state = arena.prepare(bucket, qlens, contexts, tables.to(device),
                              torch.tensor(slots, device=device, dtype=torch.int32))
        metadata = SimpleNamespace()
        state.attach(metadata)
        q_cpu = torch.randn(tuple(q.shape), dtype=torch.float16)
        k_cpu = torch.randn(tuple(k.shape), dtype=torch.float16)
        v_cpu = torch.randn(tuple(v.shape), dtype=torch.float16)
        q.copy_(q_cpu)
        k.copy_(k_cpu)
        v.copy_(v_cpu)
        for index, slot in enumerate(slots):
            key_nd[slot // BLOCK_SIZE, slot % BLOCK_SIZE] = k_cpu[index]
            value_nd[slot // BLOCK_SIZE, slot % BLOCK_SIZE] = v_cpu[index]
        expected = (flash_reference(q_cpu, k_cpu, v_cpu, qlens) if args.family == "prefill"
                    else reference(q_cpu, key_nd, value_nd, tables, qlens, contexts))

        def forward(diagnose=False):
            if diagnose:
                report["stage"] = "reshape_and_cache"
            torch_npu._npu_reshape_and_cache(k, v, op.key_cache, op.value_cache, metadata.slot_mapping)
            if diagnose:
                sync()
                report["stage"] = "attention"
            if args.family == "prefill":
                op.forward_prefill_310(q, k, v, metadata, output)
            else:
                op.forward_paged_attention(q, metadata, output)

        report["inputs"] = dict(bucket=bucket, scheduled=qlens, operator=report["operator"],
                                stream=str(torch.npu.current_stream()),
                                query=tensor_descriptor(q), key=tensor_descriptor(k), value=tensor_descriptor(v))
        if args.family == "prefill":
            report["inputs"].update(seq_len=tensor_descriptor(state.flash_seq_lens),
                                    seq_len_values=state.flash_seq_lens.tolist(),
                                    mask=tensor_descriptor(arena.prefill_mask))
        return forward, expected, state

    try:
        # All captures precede all measured replay cases, largest-first.
        for bucket in sorted(args.buckets, reverse=True):
            rows = min(bucket, MAX_REQS)
            qlens = [1] * rows if args.family == "decode" else [bucket // rows] * rows
            if args.family == "prefill":
                qlens[-1] += bucket % rows
            forward, expected, state = prepare(bucket, qlens, 0)
            print("[EAGER]", args.family, bucket, flush=True)
            print("[INPUTS]", json.dumps(report["inputs"]), flush=True)
            forward(diagnose=True)
            sync()
            diff = check(buffers[bucket][3][:sum(qlens)], expected, "eager vs CPU")
            report["eager_checks"].append(dict(bucket=bucket, max_abs_diff=diff))
            if args.eager_only:
                continue
            graph = torch.npu.NPUGraph()
            print("[CAPTURE]", args.family, bucket, flush=True)
            report["stage"] = "capture"
            with torch.npu.graph(graph, pool=pool):
                forward()
            sync()
            graphs[bucket] = graph
            report["captures"] += 1
        if args.eager_only:
            report["passed"] = True
            report["stage"] = "eager_complete"
            print(f"EAGER PASS: {len(report['eager_checks'])} buckets; no graphs captured.", flush=True)
            return
        for repeat in range(args.repeats):
            for bucket in args.buckets:
                cases = CASES[bucket] if args.family == "prefill" else [[1] * n for n in (1, 8, 10, 20)]
                for index, qlens in enumerate(cases):
                    forward, expected, state = prepare(bucket, qlens, repeat + index)
                    output = buffers[bucket][3]
                    output.fill_(float("nan"))
                    sync()
                    report["stage"] = "replay"
                    graphs[bucket].replay()
                    sync()
                    actual = output[:sum(qlens)].cpu().clone()
                    # Snapshot graph output BEFORE eager overwrites the same output.
                    forward(diagnose=True)
                    sync()
                    check(output[:sum(qlens)], expected, "changed-layout eager vs CPU")
                    diff = check(actual, expected, "graph vs CPU")
                    event = dict(bucket=bucket, repeat=repeat, scheduled=qlens,
                                 graph_id=id(graphs[bucket]), seq_ptr=arena.context.data_ptr(),
                                 mask_ptr=arena.prefill_mask.data_ptr() if arena.prefill_mask is not None else None,
                                 max_abs_diff=diff)
                    report["cases"].append(event)
                    print("[PASS]", json.dumps(event), flush=True)
        report["passed"] = True
        report["stage"] = "complete"
    except Exception as exc:
        report["error"] = repr(exc)
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"PASS: {report['captures']} graphs, {len(report['cases'])} cases. Report: {args.output}")


if __name__ == "__main__":
    main()
