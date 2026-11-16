"""Isolated NPU handoff replay simulation; not an end-to-end ASR benchmark."""

import argparse
import ast
import hashlib
import importlib
import json
import logging
import sys
import time
from types import MethodType, ModuleType
from types import SimpleNamespace as NS

import numpy as np
import torch
import torch_npu
from asr_mtp_fused import prepare_fused_heads
from asr_mtp_graph import GraphOptions
from asr_mtp_heads import BatchedMTPHeads
from asr_mtp_metrics import AcceptanceTracker
from asr_mtp_paged_cache import PagedCacheArena
from asr_mtp_token_graph import PackedMTPProgram, TokenMTPGraphs
from asr_mtp_worker import MTPWorker
from vllm.v1.sample.rejection_sampler import RejectionSampler


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--repeats", type=int, default=60)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--checkpoint")
    parser.add_argument("--model")
    parser.add_argument("--baseline")
    parser.add_argument("--profile-dir")
    parser.add_argument("--runner-source")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(912)
    torch.npu.set_device(0)
    logging.disable(logging.CRITICAL)
    original_bookkeeping = None
    if args.runner_source:
        with open(args.runner_source) as f:
            tree = ast.parse(f.read())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "NPUModelRunner"
        )
        fn = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "_bookkeeping_sync"
        )
        fn.decorator_list = []
        namespace = {"RejectionSampler": RejectionSampler}
        # Execute only the requested method from the trusted checkout supplied
        # by the benchmark operator; do not import or mutate the live runner.
        exec("from __future__ import annotations\n" + ast.unparse(fn), namespace)  # noqa: S102
        original_bookkeeping = namespace["_bookkeeping_sync"]
    cfg = NS(
        hidden_size=args.width,
        intermediate_size=args.width * 2,
        num_attention_heads=args.width // 128,
        num_key_value_heads=1,
        head_dim=128,
        attention_bias=False,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        rope_theta=10000,
        rope_scaling=None,
    )
    vocab = 256
    if args.checkpoint:
        ckpt = torch.load(
            args.checkpoint, weights_only=False, map_location="cpu", mmap=True
        )
        with open(args.model + "/config.json") as f:
            cfg = NS(**json.load(f)["thinker_config"]["text_config"])
        vocab = cfg.vocab_size
    heads = BatchedMTPHeads(cfg, args.heads)
    if args.checkpoint:
        weights = {k: v for k, v in ckpt["mtp"].items() if k in heads.state_dict()}
        heads.load_state_dict(weights, strict=True, assign=True)
    heads = heads.to("npu", torch.float16).eval().requires_grad_(False)
    prepare_fused_heads(heads)
    from asr_mtp_qkv_rope import prepare_qkv_rope

    heads.qkv_rope = prepare_qkv_rope(heads, max_context=512, backend="native")
    embed = torch.nn.Embedding(vocab, cfg.hidden_size).to("npu", torch.float16).eval()
    logits = (
        torch.nn.Linear(cfg.hidden_size, vocab, bias=False)
        .to("npu", torch.float16)
        .eval()
    )
    if args.model:
        from safetensors import safe_open

        with open(args.model + "/model.safetensors.index.json") as f:
            index = json.load(f)["weight_map"]
        for name, target in [
            ("thinker.model.embed_tokens.weight", embed.weight),
            ("thinker.lm_head.weight", logits.weight),
        ]:
            with safe_open(
                args.model + "/" + index[name], framework="pt", device="cpu"
            ) as f:
                target.copy_(f.get_tensor(name))
    options = GraphOptions(8, (), 160, tokens=(32, 128, 512), head_counts=(args.heads,))
    arena = PagedCacheArena(
        cfg,
        args.heads,
        max_context=512,
        slots=8,
        block_size=128,
        dummy_tokens=513,
        device="npu",
        dtype=torch.float16,
        scratch_elements=options.scratch_elements(cfg),
    )
    graphs = TokenMTPGraphs(options, heads, arena, embed, logits)
    if args.baseline:
        package = ModuleType("baseline")
        package.__path__ = [args.baseline]
        sys.modules["baseline"] = package
        baseline_worker = importlib.import_module("baseline.asr_mtp_worker").MTPWorker
        baseline_readback = importlib.import_module(
            "baseline.asr_mtp_device_rejection"
        ).RejectionPrelude.readback
        baseline_stage = importlib.import_module(
            "baseline.asr_mtp_token_graph"
        ).PackedMTPProgram.stage
    result = {
        "scope": "synthetic NPU handoff + actual CPU rejection parser + state/proposal update; no target decoder/audio/scheduler",
        "width": cfg.hidden_size,
        "heads": args.heads,
        "vocab": vocab,
        "checkpoint": args.checkpoint,
        "model": args.model,
        "qkv_rope": "native",
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "common_graphs": True,
        "stage_dispatch": "per variant, shared paged device computation",
        "reset_timing_excluded": True,
        "order": "ABBA",
        "profiled": bool(args.profile_dir),
        "device": torch.npu.get_device_name(0),
        "cases": [],
    }
    for batch_size, cold in ((1, False), (4, False), (8, False), (1, True), (4, True)):
        ids = [f"stream_bench_{i}_0" for i in range(batch_size)]
        length = 59 if cold else 96
        scheduled = length if cold else args.heads + 1
        params = NS(
            extra_args={"ascend_mtp": True},
            temperature=0.0,
            n=1,
            max_tokens=256,
            min_tokens=0,
            logprobs=None,
        )
        batch = NS(
            req_ids=ids,
            num_tokens_no_spec=np.full(batch_size, length + int(not cold)),
            num_computed_tokens_cpu=np.full(batch_size, 0 if cold else length),
            max_model_len=512,
            token_ids_cpu=np.zeros((batch_size, 512), dtype=np.int32),
            spec_decode_unsupported_reqs=set(),
            vocab_size=vocab,
        )
        batch.token_ids_cpu[:, : length + 1] = np.arange(length + 1) % vocab
        runner = NS(
            input_batch=batch,
            num_discarded_requests=0,
            use_async_scheduling=False,
            routed_experts_initialized=False,
            speculative_config=NS(num_speculative_tokens=args.heads),
            query_start_loc=NS(np=np.arange(batch_size + 1) * scheduled),
            requests={r: NS(sampling_params=params, output_token_ids=[]) for r in ids},
        )
        runner.max_model_len = 512
        runner.discard_request_indices = NS(np=np.array([], dtype=np.int64))
        runner._get_prompt_logprobs_dict = lambda *args: {}
        runner._rejection_output_buffer = None
        batch.generators = {}
        batch.req_id_to_index = {r: i for i, r in enumerate(ids)}
        batch.is_token_ids = np.ones_like(batch.token_ids_cpu, dtype=bool)
        batch.num_tokens = batch.num_tokens_no_spec.copy()
        worker = MTPWorker(
            runner,
            heads,
            arena,
            embed,
            logits,
            set(),
            graphs=graphs,
            device_rejection=True,
            acceptance_stats=True,
            cold_heads=args.heads,
            warm_heads=args.heads,
        )
        graphs.prepare()
        if args.baseline:
            for program, _ in graphs.entries.values():
                program.rejection.readback = MethodType(
                    baseline_readback, program.rejection
                )
            old = baseline_worker(
                runner,
                heads,
                arena,
                embed,
                logits,
                set(),
                graphs=graphs,
                device_rejection=True,
                acceptance_stats=True,
                cold_heads=args.heads,
                warm_heads=args.heads,
            )
        states = [arena.acquire() for _ in ids]
        worker.states = dict(zip(ids, states))
        workers = {"b": worker}
        if args.baseline:
            old.states = worker.states
            workers["a"] = old
        for s in states:
            s.target.normal_(0, 0.1)
            for h in s.heads:
                h.key.normal_(0, 0.1)
                h.value.normal_(0, 0.1)
                h.hidden.normal_(0, 0.1)
        draft = list(range(21, 21 + args.heads))
        scheduler = NS(
            num_scheduled_tokens=dict.fromkeys(ids, scheduled),
            scheduled_spec_decode_tokens={} if cold else dict.fromkeys(ids, draft),
            num_invalid_spec_tokens={},
        )
        samples = [
            [31]
            if cold
            else draft[: i % (args.heads + 1)]
            + [31]
            + [-1] * (args.heads - i % (args.heads + 1))
            for i in range(batch_size)
        ]
        sampler = NS(
            sampled_token_ids=torch.tensor(samples, device="npu", dtype=torch.int32),
            logprobs_tensors=None,
        )
        hidden = torch.randn(
            batch_size * scheduled, cfg.hidden_size, device="npu", dtype=torch.float16
        )
        times = {k: [] for k in workers}
        digests = {k: hashlib.sha256() for k in workers}
        trace = None
        sequence = "abba" if args.baseline else "b"
        for sample in range((args.repeats + 15) * len(sequence)):
            iteration = sample // len(sequence)
            label = sequence[sample % len(sequence)]
            worker = workers[label]
            # Shared capture removes graph-pool/order noise, but host staging
            # must still dispatch the selected source version on every step.
            stage = baseline_stage if label == "a" else PackedMTPProgram.stage
            for program, _ in graphs.entries.values():
                program.stage = MethodType(stage, program)
            for r, s in worker.states.items():
                s.length = 0 if cold else length
                for d, h in enumerate(s.heads):
                    h.valid = 0 if cold else length - d - 1
                runner.requests[r].output_token_ids = [] if cold else [4] * 20
            batch.num_tokens_no_spec[:] = length + int(not cold)
            worker.proposed = set() if cold else set(ids)
            worker.legacy_submitted = set(ids)
            worker.acceptance = AcceptanceTracker(args.heads)
            if not cold:
                for r in ids:
                    worker.acceptance.offer(r, draft, length, fresh_positions=0)
            # No asynchronous preparation from reset may escape into the measurement.
            torch.npu.synchronize()
            if args.profile_dir and iteration == 15 and sample % len(sequence) == 0:
                trace = torch_npu.profiler.profile(
                    activities=[
                        torch_npu.profiler.ProfilerActivity.CPU,
                        torch_npu.profiler.ProfilerActivity.NPU,
                    ],
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                        args.profile_dir + f"/b{batch_size}-cold{int(cold)}"
                    ),
                    record_shapes=True,
                    profile_memory=False,
                    with_stack=False,
                )
                trace.start()
            t0 = time.perf_counter()
            cpu = worker.device_handoff.prepare(scheduler, sampler, hidden)
            if cpu is None:
                raise RuntimeError(worker.device_handoff.stats)
            t1 = time.perf_counter()
            if original_bookkeeping is not None:
                runner._rejection_output_buffer = getattr(
                    worker.device_handoff, "cpu_parser", None
                )
                rows = original_bookkeeping(
                    runner,
                    scheduler,
                    NS(sampled_token_ids=cpu, logprobs_tensors=None),
                    None,
                    hidden,
                    batch_size * scheduled,
                    None,
                )[1]
            elif hasattr(worker.device_handoff, "cpu_parser"):
                rows, _ = worker.device_handoff.cpu_parser.parse(
                    cpu,
                    vocab,
                    (),
                    parser=RejectionSampler.parse_output,
                    logprobs_tensors=None,
                )
            else:
                rows, _ = RejectionSampler.parse_output(
                    cpu, vocab, (), logprobs_tensors=None
                )
            if original_bookkeeping is None:
                for i, (r, row) in enumerate(zip(ids, rows)):
                    c = int(batch.num_tokens_no_spec[i])
                    batch.token_ids_cpu[i, c : c + len(row)] = row
                    batch.num_tokens_no_spec[i] += len(row)
                    runner.requests[r].output_token_ids.extend(row)
            t2 = time.perf_counter()
            output = worker.propose(rows, scheduler, hidden)
            t3 = time.perf_counter()
            if trace is not None:
                trace.step()
                if sample % len(sequence) == len(sequence) - 1:
                    trace.stop()
                    trace = None
            if iteration >= 15:
                times[label].append(
                    [
                        (t1 - t0) * 1000,
                        (t2 - t1) * 1000,
                        (t3 - t2) * 1000,
                        (t3 - t0) * 1000,
                    ]
                )
                digests[label].update(json.dumps(output).encode())
        for label in workers:
            result["cases"].append(
                {
                    "variant": label,
                    "batch": batch_size,
                    "cold": cold,
                    "history": length,
                    "median_ms": dict(
                        zip(
                            ("prepare", "bookkeeping", "late", "total"),
                            np.median(times[label], axis=0).tolist(),
                        )
                    ),
                    "round_ms": times[label],
                    "output_digest": digests[label].hexdigest(),
                }
            )
        worker.finish(ids)
    with open(args.out, "w") as f:
        result["bookkeeping"] = (
            "delivery NPUModelRunner._bookkeeping_sync"
            if original_bookkeeping
            else "parser and append simulation"
        )
        json.dump(result, f, indent=2)
    print(
        json.dumps(
            [{k: v for k, v in c.items() if k != "round_ms"} for c in result["cases"]]
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
