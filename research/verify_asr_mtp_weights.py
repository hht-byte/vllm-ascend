"""Verify trained MTP component parity on synthetic inputs, not ASR quality or speed."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

CHECKS = ("initial_proposal", "partial_rejection", "confirmed_cache")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_inputs(checkpoint, model):
    import torch
    from safetensors import safe_open
    root = Path(model).resolve()
    cfg = SimpleNamespace(**json.loads((root / "config.json").read_text(
        encoding="utf-8"))["thinker_config"]["text_config"])
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if type(saved["metadata"]["num_heads"]) is not int or saved["metadata"]["num_heads"] != 5:
        raise ValueError("five trained heads required")
    prefixes = tuple(f"blocks.{i}." for i in range(5))
    weights = {k: v for k, v in saved["mtp"].items() if k.startswith(prefixes)}
    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    tensors = []
    for suffix in ("embed_tokens.weight", "lm_head.weight"):
        keys = [k for k in index if k == suffix or k.endswith("." + suffix)]
        if len(keys) != 1:
            raise ValueError("unique embedding and logits tensors required")
        shard = (root / index[keys[0]]).resolve()
        if not shard.is_relative_to(root) or not shard.is_file():
            raise ValueError("model shard must stay inside model directory")
        with safe_open(str(shard), framework="pt", device="cpu") as reader:
            tensor = reader.get_tensor(keys[0])
        if tuple(tensor.shape) != (cfg.vocab_size, cfg.hidden_size) or cfg.vocab_size < 64:
            raise ValueError("embedding/logits geometry differs from text config")
        tensors.append(tensor)
    return cfg, weights, *tensors


def _make_heads(cfg, weights, *, device, dtype):
    import torch
    from asr_mtp_heads import BatchedMTPHeads
    with torch.device("meta"):
        heads = BatchedMTPHeads(cfg, 5)
    heads.load_state_dict(weights, strict=True, assign=True)
    return heads.to(device=device, dtype=dtype).eval().requires_grad_(False)


def _check_cache(left, right, length):
    import torch
    for depth, (a, b) in enumerate(zip(left.heads, right.heads)):
        if depth < 4:
            torch.testing.assert_close(a.hidden[:length], b.hidden[:length], atol=0.03, rtol=0.03)
        for x, y in ((a.key, b.key), (a.value, b.value)):
            torch.testing.assert_close(x[:, :length], y[:, :length], atol=0.03, rtol=0.03)


def verify(checkpoint, model):
    import torch
    import torch_npu  # noqa: F401
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
    from asr_mtp_device_rejection import optimistic_states
    from asr_mtp_graph import GraphOptions
    from asr_mtp_heads import CacheArena
    from asr_mtp_paged_cache import PagedCacheArena
    from asr_mtp_token_graph import TokenMTPGraphs
    if not torch.npu.is_available():
        raise RuntimeError("NPU required")
    init_device_properties_triton()
    torch.manual_seed(123)
    cfg, weights, embedding, head = _load_inputs(checkpoint, model)
    device, dtype = "npu", torch.bfloat16
    # Independent modules load the same frozen weights without copying optimizer state.
    eager = _make_heads(cfg, weights, device=device, dtype=dtype)
    fixed = _make_heads(cfg, weights, device=device, dtype=dtype)
    embedding = torch.nn.Embedding.from_pretrained(embedding.to(device=device, dtype=dtype), freeze=True)
    logits = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False, device="meta")
    logits.weight = torch.nn.Parameter(head.to(device=device, dtype=dtype), requires_grad=False)
    embedding.eval()
    logits.eval()
    del weights, head
    with torch.inference_mode():
        old = CacheArena(cfg, 5, max_context=32, slots=1, device=device, dtype=dtype)
        new = PagedCacheArena(cfg, 5, max_context=32, slots=1, block_size=128,
            dummy_tokens=33, device=device, dtype=dtype, scratch_elements=33 * cfg.hidden_size)
        graphs = TokenMTPGraphs(GraphOptions(1, (), 32, tokens=(32,), head_counts=(5,)),
                                fixed, new, embedding, logits)
        graphs.device_rejection = (6, cfg.vocab_size)
        graphs.prepare()  # Real NPUGraph, FIA task updates and captured prelude.
        left, right = old.acquire(), new.acquire()
        try:
            ids = list(range(3, 10))
            left.length = right.length = len(ids)
            hidden = torch.randn(len(ids), cfg.hidden_size, device=device, dtype=dtype)
            left.target[:len(ids)] = right.target[:len(ids)] = hidden
            expected = eager.propose([left], [ids + [4]], embedding, logits, [5])
            actual = graphs.propose([right], [ids + [4]], [5])
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            _check_cache(left, right, left.length)
            draft = expected[0, :2].cpu().tolist()
            accepted = [draft[0], (draft[1] + 1) % 64]
            raw = torch.tensor([accepted + [-1] * 4], device=device, dtype=torch.int32)
            hidden = torch.randn(6, cfg.hidden_size, device=device, dtype=dtype)
            previous = len(ids) + 1
            program, replay = graphs.entries[5, 32]
            program.stage(optimistic_states([right], [previous], [6]),
                          [ids + [4] + [0] * 6], [5])
            program.rejection.stage([previous], [6], [20], [20], [0], [set()], raw, hidden)
            replay(prelude=program.rejection.replay)
            actual = program.rejection.readback(1)[:, 6:].clone()
            left.length = previous + len(accepted) - 1
            left.crop_heads(left.length)
            left.target[previous - 1:left.length] = hidden[:len(accepted)]
            expected = eager.propose([left], [ids + [4] + accepted], embedding, logits, [5]).cpu()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(program.lengths[:1].cpu(), torch.tensor([left.length]))
            _check_cache(left, right, left.length)
            torch.npu.synchronize()
        finally:
            old.release(left)
            new.release(right)
    return dict.fromkeys(CHECKS, True)


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        self.exit(2, "Invalid verifier arguments.\n")


def main(argv=None):
    parser = _Parser(description=__doc__)
    for name in ("checkpoint", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("source-commit", "checkpoint-commit", "training-checkpoint-sha256", "model-commit"):
        parser.add_argument("--" + name)
    args = parser.parse_args(argv)
    provenance = {}
    for name in ("source_commit", "checkpoint_commit", "training_checkpoint_sha256", "model_commit"):
        value = getattr(args, name)
        if value is not None:
            if re.fullmatch(r"[0-9a-fA-F]{" + ("64" if name.endswith("sha256") else "40") + "}", value) is None:
                parser.error("invalid provenance")
            provenance[name] = value.lower()
    result = dict(scope="synthetic_component", heads=5, dtype="bfloat16", passed=False,
                  checks=dict.fromkeys(CHECKS, False), provided_provenance=provenance,
                  input_checkpoint_sha256=None)
    try:
        if args.output.exists():
            sys.stderr.write("MTP component output already exists; choose a new attempt file.\n")
            return 1
        if args.output.resolve() == args.checkpoint.resolve() or args.output.resolve().is_relative_to(args.model.resolve()):
            sys.stderr.write("MTP component output must be separate from input weights.\n")
            return 1
        result["input_checkpoint_sha256"] = _sha256(args.checkpoint)
        checked = verify(args.checkpoint, args.model)
        result["checks"] = {key: checked.get(key) is True for key in CHECKS}
        result["passed"] = all(result["checks"].values())
    except Exception:
        sys.stderr.write("MTP component verification failed.\n")
    try:
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(result, sort_keys=True, indent=2) + "\n")
    except Exception:
        sys.stderr.write("MTP component result could not be written.\n")
        return 1
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
