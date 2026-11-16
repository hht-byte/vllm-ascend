"""Paired synthetic-component benchmark, never a product ASR measurement."""
import argparse
import json
import statistics
import time
from types import SimpleNamespace

import torch
import torch_npu

from asr_mtp_heads import BatchedMTPHeads
from asr_mtp_paged_cache import PagedCacheArena
from asr_mtp_token_graph import PackedMTPProgram


def measure(fn, rounds=7, repeat=100):
    for _ in range(10):
        fn()
    torch.npu.synchronize()
    values = []
    for _ in range(rounds):
        torch.npu.synchronize()
        start = time.perf_counter()
        for _ in range(repeat):
            fn()
        torch.npu.synchronize()
        values.append((time.perf_counter() - start) * 1e3 / repeat)
    return {'median_ms': statistics.median(values), 'round_mean_ms': values,
            'rounds': rounds, 'iterations_per_round': repeat}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile-dir')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(123)
    cfg = SimpleNamespace(hidden_size=256, intermediate_size=512,
        num_attention_heads=2, num_key_value_heads=1, head_dim=128,
        attention_bias=False, rms_norm_eps=1e-6, hidden_act='silu',
        rope_theta=10000, rope_scaling=None)
    heads = BatchedMTPHeads(cfg, 3).to('npu', torch.float16).eval()
    embed = torch.nn.Embedding(64, 256).to('npu', torch.float16).eval()
    logits = torch.nn.Linear(256, 64).to('npu', torch.float16).eval()
    arena = PagedCacheArena(cfg, 3, max_context=4096, slots=8,
        block_size=128, dummy_tokens=65, device='npu', dtype=torch.float16,
        scratch_elements=65*256)
    program = PackedMTPProgram(heads, arena, embed, logits,
        max_batch=8, token_size=64, context_size=4096)
    states = [arena.acquire() for _ in range(4)]
    result = {'scope': 'synthetic_component', 'device': torch.npu.get_device_name(0),
              'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
              'capacity': 8, 'context': 4096, 'heads': 3, 'stage': []}
    for length in (64, 512, 2048):
        for state in states:
            state.length = length
            for head in state.heads:
                head.valid = length-1
        for batch in (1, 4):
            rows = [[i % 64 for i in range(length+1)] for _ in range(batch)]
            fn = lambda: program.stage(states[:batch], rows, [3]*batch)
            result['stage'].append({'batch': batch, 'history': length,
                                    **measure(fn)})
    # Isolate the redundant late host materialization with identical NumPy input.
    import numpy as np
    tokens = np.arange(8*4096, dtype=np.int32).reshape(8,4096)
    result['host_conversion'] = {}
    for name, length in (('full_history',2048), ('sampled_suffix',4)):
        result['host_conversion'][name] = measure(
            lambda: [tokens[i,2048-length:2048].tolist() for i in range(8)], repeat=1000)
    if args.profile_dir:
        # Trace collection is a separate run; no profiled sample enters timing.
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU,
                        torch_npu.profiler.ProfilerActivity.NPU],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(args.profile_dir),
            record_shapes=False, profile_memory=False, with_stack=False):
            for _ in range(10):
                program.stage(states, rows, [3]*4)
            torch.npu.synchronize()
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
