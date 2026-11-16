"""Bounded public-ASR baseline, independent of the missing private async service.

This re-decodes cumulative audio at each simulated chunk boundary. It does not
implement rollback, pre-speculation, MTP, engine KV reuse, or real-time arrival.
The controller's standard-library environment does not import model packages.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time


def chunk_ends(total_samples, sample_rate, chunk_seconds, max_audio_seconds):
    """Return unique cumulative cutoffs, including one final partial chunk."""
    if (type(total_samples) is not int or total_samples <= 0 or
            type(sample_rate) is not int or sample_rate <= 0):
        raise ValueError('positive integer sample count and rate required')
    if any(not math.isfinite(value) or value <= 0
           for value in (chunk_seconds, max_audio_seconds)):
        raise ValueError('finite positive chunk and duration limits required')
    step = round(sample_rate * chunk_seconds)
    limit = min(total_samples, round(sample_rate * max_audio_seconds))
    if min(step, limit) <= 0:
        raise ValueError('timing must contain at least one audio sample')
    return list(range(step, limit, step)) + [limit]


def simulate_cumulative(audio, sample_rate, decode, *, chunk_seconds=2.0,
                        max_audio_seconds=12.0):
    """Decode real cumulative inputs; returned durations are observations only."""
    ends = chunk_ends(len(audio), sample_rate, chunk_seconds, max_audio_seconds)
    rows = []
    for index, end in enumerate(ends):
        started = time.perf_counter_ns()
        result = decode(audio[:end], sample_rate)
        elapsed = (time.perf_counter_ns() - started) / 1e9
        rows.append({
            'simulation': 'cumulative_audio_redecode',
            'mtp_enabled': False,
            'chunk_index': index,
            'audio_end_samples': end,
            'audio_end_seconds': end / sample_rate,
            'final': index == len(ends) - 1,
            'decode_wall_seconds': elapsed,
            'language': result.get('language'),
            'text': result['text'],
            'token_ids': list(result['token_ids']),
        })
    return rows


class QwenTransformersDecoder:
    """Use public Qwen model/processor classes with local-only safetensors."""
    def __init__(self, model_path, *, device='npu:0', max_new_tokens=128,
                 language='English'):
        import torch
        if device.startswith('npu:'):
            import torch_npu  # noqa: F401; registers NPU with torch
            torch.npu.set_device(device)
        from qwen_asr.core.transformers_backend import (
            Qwen3ASRForConditionalGeneration, Qwen3ASRProcessor,
        )

        self.torch, self.device = torch, device
        self.max_new_tokens, self.language = max_new_tokens, language
        self.processor = Qwen3ASRProcessor.from_pretrained(
            model_path, local_files_only=True, fix_mistral_regex=True)
        self.model = Qwen3ASRForConditionalGeneration.from_pretrained(
            model_path, local_files_only=True, use_safetensors=True,
            torch_dtype=torch.bfloat16, attn_implementation='eager',
        ).eval().to(device)
        messages = [
            {'role': 'system', 'content': ''},
            {'role': 'user', 'content': [{'type': 'audio', 'audio': ''}]},
        ]
        self.prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        ) + f'language {language}<asr_text>'

    def synchronize(self):
        if self.device.startswith('npu:'):
            self.torch.npu.synchronize()

    def __call__(self, audio, sample_rate):
        inputs = self.processor(text=self.prompt, audio=audio,
                                sampling_rate=sample_rate, return_tensors='pt')
        inputs = inputs.to(self.device)
        if 'input_features' in inputs:
            inputs['input_features'] = inputs['input_features'].to(self.model.dtype)
        with self.torch.inference_mode():
            result = self.model.generate(**inputs, do_sample=False,
                                         max_new_tokens=self.max_new_tokens)
        self.synchronize()
        token_ids = result[0, inputs['input_ids'].shape[1]:].tolist()
        text = self.processor.tokenizer.decode(token_ids, skip_special_tokens=True)
        return {'language': self.language, 'text': text, 'token_ids': token_ids}


def vllm_engine_options(model_path, max_audio_seconds):
    """Keep the live native engine small, serial and fully declared."""
    return {
        'model': str(model_path), 'dtype': 'bfloat16', 'seed': 0,
        'tensor_parallel_size': 1, 'max_model_len': 2048,
        'max_num_seqs': 1, 'max_num_batched_tokens': 2048,
        'block_size': 128,
        'gpu_memory_utilization': 0.25, 'enforce_eager': True,
        'async_scheduling': False, 'enable_prefix_caching': False,
        'enable_chunked_prefill': False,
        'limit_mm_per_prompt': {'audio': {
            'count': 1, 'length': round(max_audio_seconds * 16000)}},
    }


class QwenVLLMDecoder:
    """Use vLLM's installed native Qwen3-ASR implementation, without registration."""
    def __init__(self, model_path, *, device='npu:0', max_new_tokens=128,
                 language='English', max_audio_seconds=12.0):
        if device != 'npu:0':
            raise ValueError('native engine uses logical npu:0; select physical card in environment')
        from vllm import LLM, SamplingParams
        from vllm.transformers_utils.processors.qwen3_asr import Qwen3ASRProcessor

        self.language = language
        self.options = vllm_engine_options(model_path, max_audio_seconds)
        self.processor = Qwen3ASRProcessor.from_pretrained(
            model_path, local_files_only=True, fix_mistral_regex=True)
        messages = [
            {'role': 'system', 'content': ''},
            {'role': 'user', 'content': [{'type': 'audio', 'audio': ''}]},
        ]
        self.prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        ) + f'language {language}<asr_text>'
        self.sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens,
                                       seed=0, top_p=1.0)
        self.model = LLM(**self.options)

    def synchronize(self):
        # generate() returns only after the native engine has completed output.
        pass

    def __call__(self, audio, sample_rate):
        result = self.model.generate([
            {'prompt': self.prompt,
             'multi_modal_data': {'audio': [(audio, sample_rate)]}},
        ], self.sampling, use_tqdm=False)[0].outputs[0]
        return {'language': self.language, 'text': result.text,
                'token_ids': list(result.token_ids)}

    def close(self):
        self.model.llm_engine.engine_core.shutdown()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--model-revision', required=True)
    parser.add_argument('--audio', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--backend', choices=('vllm', 'transformers'), default='vllm')
    parser.add_argument('--chunk-seconds', type=float, default=2.0)
    parser.add_argument('--max-audio-seconds', type=float, default=12.0)
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--language', default='English')
    args = parser.parse_args(argv)
    if (len(args.model_revision) != 40 or
            any(ch not in '0123456789abcdef' for ch in args.model_revision)):
        parser.error('model revision must be the frozen full lowercase commit SHA')
    if not 1 <= args.max_new_tokens <= 512:
        parser.error('max-new-tokens must be between 1 and 512')
    if not math.isfinite(args.max_audio_seconds) or not 0 < args.max_audio_seconds <= 120:
        parser.error('max-audio-seconds must be in (0, 120]')
    if not args.model.is_dir() or not args.audio.is_file():
        parser.error('local model directory and audio file are required')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        return _execute(args)
    except Exception:
        # Developer diagnostics remain in this explicitly chosen attempt dir.
        # They are not part of the CLI error or a versioned report.
        import traceback
        (args.output_dir / 'failure.log').write_text(traceback.format_exc(), encoding='utf-8')
        raise


def _execute(args):
    import soundfile
    audio, rate = soundfile.read(args.audio, dtype='float32', always_2d=False)
    if audio.ndim != 1 or rate != 16000:
        raise ValueError('public baseline requires mono 16 kHz audio')
    # Validate all boundaries before loading any model or allocating on NPU.
    chunk_ends(len(audio), rate, args.chunk_seconds, args.max_audio_seconds)
    started = time.perf_counter_ns()
    decoder_type = QwenVLLMDecoder if args.backend == 'vllm' else QwenTransformersDecoder
    options = {'max_audio_seconds': args.max_audio_seconds} if args.backend == 'vllm' else {}
    decoder = decoder_type(str(args.model), device=args.device,
                           max_new_tokens=args.max_new_tokens,
                           language=args.language, **options)
    decoder.synchronize()
    startup = (time.perf_counter_ns() - started) / 1e9
    try:
        rows = simulate_cumulative(audio, rate, decoder, chunk_seconds=args.chunk_seconds,
                                   max_audio_seconds=args.max_audio_seconds)
    finally:
        if hasattr(decoder, 'close'):
            decoder.close()
    with (args.output_dir / 'chunks.jsonl').open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    versions = {}
    for name in ('torch', 'torch-npu', 'transformers', 'qwen-asr', 'soundfile',
                 'vllm', 'vllm-ascend'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    manifest = {
        'schema_version': 1,
        'simulation': 'cumulative_audio_redecode',
        'mtp_enabled': False,
        'model_revision': args.model_revision,
        'audio_sha256': sha256_file(args.audio),
        'script_sha256': sha256_file(__file__),
        'sample_rate': rate,
        'audio_samples_available': len(audio),
        'chunk_seconds': args.chunk_seconds,
        'max_audio_seconds': args.max_audio_seconds,
        'max_new_tokens': args.max_new_tokens,
        'device': args.device,
        'backend': args.backend,
        'engine_options': getattr(decoder, 'options', None),
        'attention': 'engine-default' if args.backend == 'vllm' else 'eager',
        'startup_wall_seconds': startup,
        'chunks_completed': len(rows),
        'versions': versions,
        'comparison': None,
        'performance_claim': None,
    }
    (args.output_dir / 'manifest.json').write_text(
        json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'completed', 'chunks_completed': len(rows),
                      'simulation': manifest['simulation'], 'mtp_enabled': False}))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        # Paths, model output, external API bodies and raw logs stay in scratch.
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__}),
              file=sys.stderr)
        raise SystemExit(1)
