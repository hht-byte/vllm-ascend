# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""310P splitfuse capture-once probe, using the same arena/task code as runtime.

Run each layout/update mode in a fresh process. Any failure exits nonzero; no
recapture fallback or silent change to another attention operator is allowed.
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch
import torch_npu

SOURCE = Path(__file__).resolve().parents[2] / "vllm_ascend/_310p/token_graph.py"
SPEC = importlib.util.spec_from_file_location("token_graph_probe_runtime", SOURCE)
tg = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tg
SPEC.loader.exec_module(tg)

BLOCK_SIZE = 128
MAX_CONTEXT = 512
MAX_REQS = 20
HEADS = 8
KV_HEADS = 2
HEAD_SIZE = 64
NZ_FORMAT = 29
CASES = {
    20: [
        [7, 7, 1, 1, 1, 1, 1, 1],
        [1, 1, 7, 1, 1, 7, 1, 1],
        [3, 3, 3, 3, 2, 2, 2, 2],
        [7, 5, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 5],
        [1, 1, 1, 1, 7, 7],
        [1] * 20,
        [7, 7, 1, 1, 1, 1, 1, 1],
    ],
    80: [[1, 1, 1, 1, 5, 70], [1] * 20, [20, 20, 20, 20]],
    192: [[70, 40, 70], [192], [1] * 20, [70, 40, 70]],
}


def sync():
    torch.npu.synchronize()


def check(actual, expected, label):
    actual = actual.detach().float().cpu().clone()
    expected = expected.detach().float().cpu().clone()
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError(f"{label}: non-finite values")
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    return float((actual - expected).abs().max())


def reference(query, key, value, blocks, qlens, contexts):
    """Independent CPU causal GQA reference over the logical ND cache."""
    outputs = []
    offset = 0
    for row, (length, context) in enumerate(zip(qlens, contexts)):
        for index in range(length):
            visible = context - length + index + 1
            positions = torch.arange(visible)
            block_ids = blocks[row, positions // BLOCK_SIZE].long()
            k = key[block_ids, positions % BLOCK_SIZE].float().repeat_interleave(HEADS // KV_HEADS, dim=1)
            v = value[block_ids, positions % BLOCK_SIZE].float().repeat_interleave(HEADS // KV_HEADS, dim=1)
            scores = torch.einsum("hd,lhd->hl", query[offset + index].float(), k) * HEAD_SIZE**-0.5
            outputs.append(torch.einsum("hl,lhd->hd", scores.softmax(-1), v))
        offset += length
    return torch.stack(outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", choices=("fixed", "active"), default="fixed")
    parser.add_argument("--update-mode", choices=("inplace", "task_update"), default="task_update")
    parser.add_argument("--buckets", type=int, nargs="+", default=[20, 80, 192])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("token-graph-probe.json"))
    args = parser.parse_args()
    if args.layout == "active" and args.update_mode != "task_update":
        parser.error("active layout requires task_update")
    if args.repeats < 1 or len(set(args.buckets)) != len(args.buckets) or any(b not in CASES for b in args.buckets):
        parser.error("Choose unique buckets from 20,80,192 and positive repeats")
    torch.set_num_threads(1)
    torch.manual_seed(310)
    torch.npu.set_device(0)
    device = "npu:0"
    max_blocks = MAX_CONTEXT // BLOCK_SIZE
    num_blocks = 1 + MAX_REQS * max_blocks
    storage = tg.TokenGraphArena(
        max(args.buckets),
        MAX_REQS,
        max_blocks,
        MAX_CONTEXT,
        device,
        tg.TokenGraphConfig(True, args.update_mode, args.layout),
    )
    shape = (num_blocks, KV_HEADS * HEAD_SIZE // 16, BLOCK_SIZE, 16)
    key_cache = torch_npu.empty_with_format(size=shape, dtype=torch.float16, device=device, acl_format=NZ_FORMAT)
    value_cache = torch_npu.empty_with_format(size=shape, dtype=torch.float16, device=device, acl_format=NZ_FORMAT)
    key_nd = torch.randn(num_blocks, BLOCK_SIZE, KV_HEADS, HEAD_SIZE, dtype=torch.float16)
    value_nd = torch.randn_like(key_nd)
    all_slots = torch.arange(num_blocks * BLOCK_SIZE, dtype=torch.int32, device=device)
    torch_npu._npu_reshape_and_cache(
        key_nd.flatten(0, 1).to(device), value_nd.flatten(0, 1).to(device), key_cache, value_cache, all_slots
    )
    sync()
    base_blocks = torch.arange(1, num_blocks, dtype=torch.int32).reshape(MAX_REQS, max_blocks)
    mask = torch.triu(torch.ones(2048, 2048, dtype=torch.float16, device=device), diagonal=1) * -10000
    report = dict(
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        device=torch.npu.get_device_name(0),
        layout=args.layout,
        update_mode=args.update_mode,
        operator="_npu_paged_attention_splitfuse_v2",
        cases=[],
        captures=0,
    )
    graphs = {}  # Keep every bucket's graph alive, as the model runner does.
    graph_pool = torch.npu.graph_pool_handle()
    buffers = {}
    for bucket in args.buckets:
        query = torch.zeros(bucket, HEADS, HEAD_SIZE, dtype=torch.float16, device=device)
        key = torch.zeros(bucket, KV_HEADS, HEAD_SIZE, dtype=torch.float16, device=device)
        value = torch.zeros_like(key)
        output = torch.empty_like(query)
        eager_output = torch.empty_like(query)
        buffers[bucket] = (query, key, value, output, eager_output)
    # Revisit old graphs after other buckets have overwritten the shared arena.
    # Retaining graph objects alone does not exercise this lifetime boundary.
    visits = [(repeat, bucket) for repeat in range(args.repeats) for bucket in args.buckets]
    for repeat, bucket in visits:
        query, key, value, output, eager_output = buffers[bucket]
        graph = graphs.get(bucket)
        last_reference = None
        for case_index, qlens in enumerate(CASES[bucket]):
            iteration = repeat * len(CASES[bucket]) + case_index
            sync()  # Do not overwrite host qLens while a replay may still read it.
            contexts = [n + (0, 63, 127, 129)[(row + iteration) % 4] for row, n in enumerate(qlens)]
            blocks = base_blocks.roll(iteration % MAX_REQS, dims=0)
            slots = []
            for row, (n, c) in enumerate(zip(qlens, contexts)):
                slots.extend(int(blocks[row, p // BLOCK_SIZE]) * BLOCK_SIZE + p % BLOCK_SIZE for p in range(c - n, c))
            n = sum(qlens)
            q_cpu = torch.randn(bucket, HEADS, HEAD_SIZE, dtype=torch.float16)
            k_cpu = torch.randn(bucket, KV_HEADS, HEAD_SIZE, dtype=torch.float16)
            v_cpu = torch.randn_like(k_cpu)
            query.copy_(q_cpu)
            key.copy_(k_cpu)
            value.copy_(v_cpu)
            for i, slot in enumerate(slots):
                key_nd[slot // BLOCK_SIZE, slot % BLOCK_SIZE] = k_cpu[i]
                value_nd[slot // BLOCK_SIZE, slot % BLOCK_SIZE] = v_cpu[i]
            expected = reference(q_cpu[:n], key_nd, value_nd, blocks, qlens, contexts)
            if (
                last_reference is not None
                and last_reference.shape == expected.shape
                and torch.equal(last_reference, expected)
            ):
                raise AssertionError("Test inputs failed to produce a distinct reference")
            last_reference = expected.clone()
            state = storage.prepare(bucket, qlens, contexts, blocks.to(device), torch.tensor(slots, device=device))
            # Snapshot cache to detect any writes outside real slots, including block 0.
            before_k = key_cache.cpu().clone()
            before_v = value_cache.cpu().clone()

            def forward(key=key, value=value, bucket=bucket, state=state, query=query, output=output):
                torch_npu._npu_reshape_and_cache(key, value, key_cache, value_cache, storage.slots[:bucket])
                state.attention(
                    torch_npu,
                    "attention",
                    query,
                    key_cache,
                    value_cache,
                    mask,
                    output,
                    HEADS,
                    KV_HEADS,
                    HEAD_SIZE**-0.5,
                )

            if graph is None:
                forward()  # Eager operator warmup.
                sync()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph, pool=graph_pool):
                    forward()
                sync()
                report["captures"] += 1
                graphs[bucket] = graph
            output.fill_(float("nan"))
            sync()
            start = time.perf_counter()
            state.update(torch_npu)
            sync()
            update_ms = (time.perf_counter() - start) * 1000
            # A task update may launch work on some stacks. Poison AFTER it so
            # only a successful graph replay can produce the compared output.
            output.fill_(float("nan"))
            sync()
            start = time.perf_counter()
            graph.replay()
            sync()
            replay_ms = (time.perf_counter() - start) * 1000
            actual = output[:n].cpu().clone()
            # Independent eager invocation, with exact positive request lengths
            # (no zero rows and no dummy) to expose padding mistakes.
            torch_npu._npu_paged_attention_splitfuse_v2(
                query=query[:n],
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=blocks[: len(qlens)].to(device),
                context_lens=torch.tensor(contexts, dtype=torch.int32, device=device),
                seq_len=torch.tensor(qlens, dtype=torch.int32),
                mask=mask,
                num_heads=HEADS,
                num_kv_heads=KV_HEADS,
                scale_value=HEAD_SIZE**-0.5,
                mask_type=5,
                out=eager_output[:n],
            )
            sync()
            diff = check(actual, expected, "graph vs independent CPU causal reference")
            check(actual, eager_output[:n], "graph vs exact-request eager")
            untouched = torch.ones(num_blocks, BLOCK_SIZE, dtype=torch.bool)
            for slot in slots:
                untouched[slot // BLOCK_SIZE, slot % BLOCK_SIZE] = False
            # Physical NZ dimensions: [block, hidden/16, token, 16].
            untouched_nz = untouched[:, None, :, None].expand(shape)
            if not torch.equal(before_k[untouched_nz], key_cache.cpu()[untouched_nz]):
                raise AssertionError("K cache changed outside real slots")
            if not torch.equal(before_v[untouched_nz], value_cache.cpu()[untouched_nz]):
                raise AssertionError("V cache changed outside real slots")
            case = dict(
                bucket=bucket,
                repeat=repeat,
                graph_id=id(graph),
                scheduled=qlens,
                contexts=contexts,
                max_abs_diff=diff,
                update_ms=update_ms,
                replay_ms=replay_ms,
                context_ptr=storage.context.data_ptr(),
                qlens_ptr=storage.qlens.data_ptr(),
            )
            report["cases"].append(case)
            print("[PASS]", json.dumps(case), flush=True)
        if len(state.tasks) != 1:
            raise AssertionError("Expected one captured attention task per bucket")
    report["peak_memory_bytes"] = torch.npu.max_memory_allocated()
    if report["captures"] != len(args.buckets) or len(graphs) != len(args.buckets):
        raise AssertionError("Unexpected recapture")
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {report['captures']} graphs, {len(report['cases'])} cases. Report: {args.output}")


if __name__ == "__main__":
    main()
