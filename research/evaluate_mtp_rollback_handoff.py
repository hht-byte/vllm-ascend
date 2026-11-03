r"""Run the spec-asr checkpoint through rollback-first streaming verification.

Example on the model host::

    python research/evaluate_mtp_rollback_handoff.py \
      --spec-asr-root /home/work/spec-asr --model /models/Qwen3-ASR-1.7B \
      --checkpoint /home/work/spec-asr/runs/pt-scale-100h-mtp5/best.pt \
      --teacher-manifest /home/work/spec-asr/data/teacher/merged/test.jsonl \
      --dataset fleurs --output /secure-validation/mtp-handoff-fleurs.json

Only the current chunk's cumulative audio is loaded. Each arm rolls its own
actual prior output. The baseline is greedy; the MTP arm first verifies the
old unstable tail, then drafts from the confirmed target state.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __package__:
    from .mtp_rollback_handoff import run_handoff
    from .mtp_streaming_replay import replay_utterance
else:
    from mtp_rollback_handoff import run_handoff
    from mtp_streaming_replay import replay_utterance


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec-asr-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-manifest", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--target-language", default="Portuguese")
    parser.add_argument("--max-utterances", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--draft-heads", type=int, default=5)
    parser.add_argument("--kmax", type=int, default=15)
    parser.add_argument("--rollback-tokens", type=int, default=5)
    parser.add_argument("--unfixed-chunks", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    parser.add_argument("--fail-on-mismatch", action="store_true")
    args = parser.parse_args(argv)
    for name in ("max_utterances", "max_new_tokens", "draft_heads", "kmax"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("rollback_tokens", "unfixed_chunks", "bootstrap_samples"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")
    if args.rollback_tokens > args.kmax:
        parser.error("--rollback-tokens must fit --kmax")
    return args


def main(argv=None):
    args = parse_args(argv)
    root = args.spec_asr_root.resolve()
    if not (root / "spec_asr" / "mtp.py").is_file():
        raise ValueError("--spec-asr-root must contain spec_asr/mtp.py")
    sys.path.insert(0, str(root))

    import soundfile as sf
    import torch
    from scripts.evaluate_pt_unguarded_streaming import gather_utterances
    from spec_asr.asr_metrics import compare_transcripts
    from spec_asr.checkpoint import base_signature
    from spec_asr.data import ChunkCollator
    from spec_asr.decoding import generate, prompt_only
    from spec_asr.mtp import SequentialMTP
    from spec_asr.runtime import (
        eos_ids,
        load_backbone,
        move_batch,
        select_device,
        synchronize,
    )

    if __package__:
        from .mtp_rollback_torch import TorchHandoffBackend
    else:
        from mtp_rollback_torch import TorchHandoffBackend

    records = []
    with args.teacher_manifest.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
    def duration_seconds(path):
        info = sf.info(path)
        return info.frames / info.samplerate

    utterances = gather_utterances(records, args.dataset, duration_seconds)
    utterances = utterances[:args.max_utterances]
    device = select_device(args.device)
    dtype = getattr(torch, args.dtype)
    model, processor = load_backbone(args.model, device, dtype=dtype)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    signature = base_signature(args.model)
    if metadata.get("base_signature") != signature:
        raise ValueError("MTP checkpoint backbone differs from the target model")
    if args.draft_heads > metadata["num_heads"]:
        raise ValueError("--draft-heads exceeds checkpoint num_heads")
    mtp = SequentialMTP(model.thinker.model, metadata["num_heads"],
                        metadata["decay"])
    mtp.load_state_dict(checkpoint["mtp"], strict=True)
    mtp = mtp.to(device=device, dtype=dtype).eval().requires_grad_(False)
    del checkpoint
    collator = ChunkCollator(processor)
    stops = eos_ids(model, processor)
    if not stops:
        raise ValueError("target model has no EOS token")

    def decode(record, arm, prefix, rollback):
        batch = move_batch(prompt_only(collator([record])), device, dtype)
        synchronize(device)
        started = time.perf_counter()
        if arm == "baseline":
            result = generate(model, batch, max_new_tokens=args.max_new_tokens,
                              eos_token_ids=stops)
        else:
            active_mtp = mtp if (args.target_language == "all" or
                                 record.get("language") == args.target_language) else None
            backend = TorchHandoffBackend(model, batch, active_mtp, stops)
            result = run_handoff(
                backend, rollback, max_new_tokens=args.max_new_tokens,
                kmax=args.kmax, draft_heads=args.draft_heads, eos_token_ids=stops)
            synchronize(device)
            result["seconds"] = time.perf_counter() - started
            result["draft_enabled"] = active_mtp is not None
        return result

    rows = []
    with torch.inference_mode():
        for index, utterance in enumerate(utterances, 1):
            row = replay_utterance(
                utterance, decode,
                lambda tokens: processor.tokenizer.decode(tokens, skip_special_tokens=True),
                stops, rollback_tokens=args.rollback_tokens,
                unfixed_chunks=args.unfixed_chunks)
            rows.append(row)
            print(json.dumps({"completed": index, "utterance_id": row["utterance_id"],
                              "token_exact_match": row["token_exact_match"]}), flush=True)

    report = {
        "schema_version": 1, "mode": "spec_asr_reference_rollback_then_mtp",
        "model": str(args.model.resolve()), "checkpoint": str(args.checkpoint.resolve()),
        "base_signature": signature, "dataset": args.dataset,
        "settings": {"kmax": args.kmax, "draft_heads": args.draft_heads,
                     "effective_draft_heads": min(args.draft_heads, args.kmax - 1),
                     "rollback_tokens": args.rollback_tokens,
                     "unfixed_chunks": args.unfixed_chunks,
                     "max_new_tokens": args.max_new_tokens,
                     "target_language": args.target_language},
        "utterances": len(rows), "exact_match_utterances":
            sum(row["token_exact_match"] for row in rows),
        "exact_match_chunks": sum(chunk["token_exact_match"]
                                  for row in rows for chunk in row["chunks"]),
        "chunk_count": sum(len(row["chunks"]) for row in rows),
        "rollback_accepted_tokens": sum(chunk["mtp"]["rollback_accepted_tokens"]
                                        for row in rows for chunk in row["chunks"]),
        "mtp_accepted_draft_tokens": sum(chunk["mtp"]["mtp_accepted_draft_tokens"]
                                         for row in rows for chunk in row["chunks"]),
        "metrics": compare_transcripts(rows, bootstrap_samples=args.bootstrap_samples),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    if args.fail_on_mismatch and report["exact_match_chunks"] != report["chunk_count"]:
        raise RuntimeError("reference MTP output differs from greedy target; see report")
    return report


if __name__ == "__main__":
    main()
