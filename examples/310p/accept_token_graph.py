# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline full-model acceptance: isolated eager/graph processes, exact tokens,
actual replay audit, memory and end-to-end batch latency. No device imports in
the parent or report comparison. User supplies model arguments and raw prompts.
"""

import argparse
import copy
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def compare_reports(eager, graph, buckets):
    errors = []
    left = {run["key"]: run for run in eager["runs"]}
    right = {run["key"]: run for run in graph["runs"]}
    if not left or left.keys() != right.keys() or len(left) != len(eager["runs"]) or len(right) != len(graph["runs"]):
        errors.append("Missing or duplicate workload results")
    for key in sorted(left.keys() & right.keys()):
        a, b = left[key]["outputs"], right[key]["outputs"]
        if not a or len(a) != len(b):
            errors.append(f"{key}: output count mismatch or empty batch")
            continue
        for i, (x, y) in enumerate(zip(a, b)):
            if (
                x["token_ids"] != y["token_ids"]
                or x["finish_reason"] != y["finish_reason"]
                or x.get("text") != y.get("text")
            ):
                errors.append(f"{key} request {i}: generated tokens or finish reason differ")
    initial, final = graph["initial_graphs"], graph["final_graphs"]
    if initial != final:
        errors.append("Graph entries/identities changed during measured workloads")
    families = ("token", "prefill", "decode") if graph.get("phase_routing", False) else ("token",)
    actual_keys = sorted((entry.get("family", "token"), entry["bucket"]) for entry in final)
    expected_keys = sorted((family, bucket) for family in families for bucket in buckets)
    if actual_keys != expected_keys:
        errors.append("Expected exactly one FULL model graph per family and configured token bucket")
    if any(entry["num_reqs"] is not None or entry["uniform"] for entry in final):
        errors.append("Graph keys still depend on request count or uniform decode layout")
    replay = [event for event in graph["audit"] if event["event"] == "replay"]
    if not replay:
        errors.append("No actual FULL graph replay observed; eager fallback cannot pass acceptance")
    if any(event["event"] == "capture" for event in graph["audit"]):
        errors.append("Unexpected capture during measured workloads")
    if graph.get("backend") == "910b":
        if graph["startup"]["graphs"] != initial:
            errors.append("910B graphs were not all captured at startup")
        if any(event.get("operator") != "_npu_paged_attention" or event.get("task_updates", 0) < 1
               for event in replay):
            errors.append("910B replay did not use updated PA tasks")
    coverage = {
        str(bucket): {
            "replays": sum(event["bucket"] == bucket for event in replay),
            "request_counts": sorted({event["actual_reqs"] for event in replay if event["bucket"] == bucket}),
            "multiquery_replays": sum(
                event["bucket"] == bucket and max(event["scheduled"], default=0) > 1 for event in replay
            ),
        }
        for bucket in buckets
    }
    # Report incomplete workload coverage explicitly, separate from correctness.
    missing = [f"bucket {bucket}: no replay" for bucket in buckets if not coverage[str(bucket)]["replays"]]
    family_coverage = {family: sum(event.get("family", "token") == family for event in replay) for family in families}
    for family, count in family_coverage.items():
        if not count:
            missing.append(f"family {family}: no actual replay")
    if not any(max(event["scheduled"], default=0) > 1 for event in replay):
        missing.append("No mixed/prefill query length > 1 executed through FULL replay")
    if not any(event.get("phase") == "decode" and event.get("decode_only") for event in replay):
        missing.append("No actual decode-only FULL replay observed; requests may have ended during prefill")
    if 20 in buckets and not {8, 10}.issubset(coverage["20"]["request_counts"]):
        missing.append("Bucket 20 did not replay both 8 and 10 real requests")
    timings = {}
    for key in left.keys() & right.keys():
        workload = key.rsplit("/r", 1)[0]
        timings.setdefault(workload, {"eager_seconds": [], "graph_seconds": []})
        timings[workload]["eager_seconds"].append(left[key]["seconds"])
        timings[workload]["graph_seconds"].append(right[key]["seconds"])
    for result in timings.values():
        a, b = statistics.median(result["eager_seconds"]), statistics.median(result["graph_seconds"])
        result.update(eager_median_seconds=a, graph_median_seconds=b, eager_over_graph=a / b if b > 0 else None)
    return dict(
        passed=not errors,
        errors=errors,
        coverage=coverage,
        family_coverage=family_coverage,
        missing_coverage=missing,
        full_acceptance=not errors and not missing,
        timings=timings,
    )


def load_inputs(path):
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("cases JSON must be a nonempty array")
    inputs, manifest = [], []
    for index, item in enumerate(records):
        prompt = item.get("prompt")
        tokens = item.get("prompt_token_ids")
        if (prompt is None) == (tokens is None):
            raise ValueError("Each case requires exactly one of prompt / prompt_token_ids")
        value = {"prompt": prompt} if prompt is not None else {"prompt_token_ids": tokens}
        record = dict(id=str(item.get("id", index)), prompt=prompt, prompt_token_ids=tokens)
        if "audio" in item:
            import soundfile as sf

            audio_path = (path.parent / item["audio"]).resolve()
            audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
            if audio.ndim != 1:
                raise ValueError(f"Use mono audio, not implicit channel mixing: {audio_path}")
            value["multi_modal_data"] = {"audio": (audio, sample_rate)}
            record.update(
                audio=str(audio_path),
                audio_sha256=hashlib.sha256(audio_path.read_bytes()).hexdigest(),
                sample_rate=sample_rate,
                audio_samples=len(audio),
            )
        inputs.append(value)
        manifest.append(record)
    return inputs, manifest


def run_child(args):
    import torch
    import torch_npu
    import vllm
    from vllm import LLM, SamplingParams

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or "model" not in config:
        raise ValueError("config JSON must contain LLM keyword arguments including model")
    config = copy.deepcopy(config)
    extension = "vllm_ascend._310p.token_graph_acceptance.TokenGraphAcceptanceWorker"
    if config.get("worker_extension_cls") not in (None, "", extension):
        raise ValueError(
            "This script requires its own worker_extension_cls; do not replace an existing extension silently"
        )
    config["worker_extension_cls"] = extension
    config.update(async_scheduling=False, seed=310, enforce_eager=args.child == "eager")
    config.setdefault("max_num_seqs", max(args.batch_sizes))
    config.setdefault("max_num_batched_tokens", max(args.buckets))
    config.setdefault("enable_prefix_caching", False)
    config.setdefault("enable_chunked_prefill", True)
    if config["max_num_seqs"] < max(args.batch_sizes):
        raise ValueError("max_num_seqs is smaller than the acceptance batch sizes")
    for name in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size"):
        if config.get(name, 1) != 1:
            raise ValueError("This acceptance script supports one device only")
    additional = config.setdefault("additional_config", {})
    if args.backend == "910b":
        additional["token_graph_310p"] = dict(enabled=False)
        additional["token_graph_910b"] = dict(enabled=args.child == "graph")
    else:
        additional["token_graph_910b"] = dict(enabled=False)
        additional["token_graph_310p"] = dict(enabled=args.child == "graph", request_layout="token",
                                             update_mode="inplace", phase_routing=args.phase_routing)
    compilation = config.setdefault("compilation_config", {})
    if not isinstance(compilation, dict):
        raise ValueError("compilation_config must be a JSON object")
    compilation.update(
        cudagraph_mode="FULL" if args.child == "graph" else "NONE",
        cudagraph_capture_sizes=args.buckets,
        max_cudagraph_capture_size=max(args.buckets),
    )
    inputs, manifest = load_inputs(args.cases)
    report = dict(
        mode=args.child,
        backend=args.backend,
        phase_routing=args.phase_routing,
        config=config,
        manifest=manifest,
        runs=[],
        audit=[],
        versions=dict(torch=torch.__version__, torch_npu=torch_npu.__version__, vllm=vllm.__version__),
    )
    destination = args.out / f"{args.child}.json"
    write_json(destination, report)
    start = time.perf_counter()
    llm = LLM(**config)
    report["startup_seconds"] = time.perf_counter() - start

    def rpc(action="snapshot", phase=""):
        snapshots = llm.collective_rpc("token_graph_acceptance", args=(action, phase))
        if len(snapshots) != 1:
            raise RuntimeError("Expected one worker snapshot")
        return snapshots[0]

    report["startup"] = rpc("install")
    phases = [("prefill", 1), ("decode", args.max_tokens)]
    for phase, limit in phases:
        for batch_size in args.batch_sizes:
            prompts = [inputs[i % len(inputs)] for i in range(batch_size)]
            params = SamplingParams(temperature=0, seed=310, max_tokens=limit)
            if config["enable_prefix_caching"] and not llm.reset_prefix_cache():
                raise RuntimeError("Could not reset prefix cache for a cold-prefill workload")
            llm.generate(prompts, params, use_tqdm=False)
    report["warmup"] = rpc()
    report["initial_graphs"] = report["warmup"]["graphs"]
    for repeat in range(args.repeats):
        for phase, limit in phases:
            for batch_size in args.batch_sizes:
                key = f"{phase}/b{batch_size}/r{repeat}"
                prompts = [inputs[i % len(inputs)] for i in range(batch_size)]
                params = SamplingParams(temperature=0, seed=310, max_tokens=limit)
                if config["enable_prefix_caching"] and not llm.reset_prefix_cache():
                    raise RuntimeError("Could not reset prefix cache")
                rpc("reset", phase)
                start = time.perf_counter()
                outputs = llm.generate(prompts, params, use_tqdm=False)
                elapsed = time.perf_counter() - start
                snapshot = rpc()
                if len(outputs) != batch_size:
                    raise RuntimeError("Model returned an unexpected number of request outputs")
                serialized = []
                for output in outputs:
                    if not output.finished or len(output.outputs) != 1:
                        raise RuntimeError("Expected one finished greedy completion per request")
                    completion = output.outputs[0]
                    serialized.append(
                        dict(
                            token_ids=list(completion.token_ids),
                            text=completion.text,
                            finish_reason=completion.finish_reason,
                            prompt_tokens=len(output.prompt_token_ids or []),
                        )
                    )
                run = dict(
                    key=key,
                    seconds=elapsed,
                    outputs=serialized,
                    memory=snapshot["memory"],
                    generated_tokens=sum(len(item["token_ids"]) for item in serialized),
                )
                report["runs"].append(run)
                report["audit"].extend(snapshot["audit"])
                report["final_graphs"] = snapshot["graphs"]
                write_json(destination, report)
                replays = sum(event["event"] == "replay" for event in snapshot["audit"])
                print(f"[MODEL RUN] {key} seconds={elapsed:.4f} replays={replays}", flush=True)
    report["completed"] = True
    write_json(destination, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="JSON object of vLLM LLM arguments")
    parser.add_argument(
        "--cases", type=Path, required=True, help="JSON array of raw prompts and optional mono audio paths"
    )
    parser.add_argument("--out", type=Path, default=Path("model-acceptance"))
    parser.add_argument("--buckets", type=int, nargs="+", default=[20, 80, 192])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 10, 20])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--backend", choices=("310p", "910b"), default="310p")
    parser.add_argument("--phase-routing", action="store_true", help="Audit separate prefill, decode and token graphs")
    parser.add_argument(
        "--max-slowdown", type=float, help="Optional upper bound on graph/eager median latency per workload"
    )
    parser.add_argument("--max-peak-gib", type=float, help="Optional upper bound on graph worker peak reserved GiB")
    parser.add_argument("--child", choices=("eager", "graph"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1 or args.max_tokens < 2 or min(args.buckets + args.batch_sizes) < 1:
        parser.error("positive sizes/repeats and max-tokens >= 2 required")
    if len(set(args.buckets)) != len(args.buckets) or len(set(args.batch_sizes)) != len(args.batch_sizes):
        parser.error("duplicate bucket/batch sizes are not allowed")
    if any(value is not None and value <= 0 for value in (args.max_slowdown, args.max_peak_gib)):
        parser.error("performance/memory bounds must be positive")
    if args.backend == "910b" and args.phase_routing:
        parser.error("910B currently supports token PA graphs only")
    args.out = args.out.resolve()
    if args.child:
        run_child(args)
        return
    if args.out.exists():
        parser.error("Use a new output directory to avoid mixing acceptance runs")
    args.out.mkdir(parents=True)
    env = os.environ.copy()
    if env.get("ASCEND_LAUNCH_BLOCKING") == "1":
        parser.error("Unset ASCEND_LAUNCH_BLOCKING before graph acceptance")
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    for mode in ("eager", "graph"):
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--child", mode]
        print(f"[START] {mode}; log: {args.out / (mode + '.log')}", flush=True)
        with (args.out / f"{mode}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            write_json(args.out / "summary.json", dict(passed=False, failed_process=mode, returncode=result.returncode))
            raise SystemExit(f"{mode} failed; inspect {args.out / (mode + '.log')}")
    eager = json.loads((args.out / "eager.json").read_text(encoding="utf-8"))
    graph = json.loads((args.out / "graph.json").read_text(encoding="utf-8"))
    summary = compare_reports(eager, graph, args.buckets)
    summary["memory_bytes"] = {
        mode: dict(
            startup=report["startup"]["memory"],
            max_run_peak_allocated=max(run["memory"]["peak_allocated"] for run in report["runs"]),
            max_run_peak_reserved=max(run["memory"]["peak_reserved"] for run in report["runs"]),
            max_peak_reserved=max(
                report["startup"]["memory"]["peak_reserved"],
                report["warmup"]["memory"]["peak_reserved"],
                *(run["memory"]["peak_reserved"] for run in report["runs"]),
            ),
        )
        for mode, report in (("eager", eager), ("graph", graph))
    }
    summary["dispatch_counts"] = {
        mode: {
            path: sum(event["event"] == "dispatch" and event.get("mode") == path for event in report["audit"])
            for path in ("NONE", "FULL", "PIECEWISE")
        }
        for mode, report in (("eager", eager), ("graph", graph))
    }
    if eager["manifest"] != graph["manifest"]:
        summary["errors"].append("Input files changed between processes")
        summary["passed"] = summary["full_acceptance"] = False
    summary["warnings"] = [
        f"{mode}: storage_offset warning remains unresolved"
        for mode in ("eager", "graph")
        if "storage_offset" in (args.out / f"{mode}.log").read_text(encoding="utf-8", errors="replace")
    ]
    if summary["warnings"]:
        summary["full_acceptance"] = False
    summary["pending_reviews"] = []
    if args.max_slowdown is None:
        summary["pending_reviews"].append("Latency ratios reported; no performance acceptance bound supplied")
    else:
        for workload, result in summary["timings"].items():
            if result["graph_median_seconds"] > args.max_slowdown * result["eager_median_seconds"]:
                summary["errors"].append(f"{workload}: graph/eager latency exceeds {args.max_slowdown}")
    if args.max_peak_gib is None:
        summary["pending_reviews"].append("Memory reported; no peak-reserved GiB acceptance bound supplied")
    elif summary["memory_bytes"]["graph"]["max_peak_reserved"] > args.max_peak_gib * 1024**3:
        summary["errors"].append("Graph peak reserved memory exceeds configured GiB bound")
    if summary["pending_reviews"]:
        summary["full_acceptance"] = False
    if summary["errors"]:
        summary["passed"] = summary["full_acceptance"] = False
    write_json(args.out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not summary["passed"]:
        raise SystemExit(1)
    if not summary["full_acceptance"]:
        raise SystemExit(2)  # Correctness passed, but coverage/warnings still need review.


if __name__ == "__main__":
    main()
