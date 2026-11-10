# SPDX-License-Identifier: Apache-2.0
"""Worker-local external draft -> MTP handoff for the v0.23 sync runner."""
import hashlib
import logging
from collections import Counter
from pathlib import Path

import torch

try:
    from .asr_mtp_graph import MTPGraphs, graph_options, validate_graph_runner
    from .asr_mtp_heads import BatchedMTPHeads, CacheArena
except ImportError:
    from asr_mtp_graph import MTPGraphs, graph_options, validate_graph_runner
    from asr_mtp_heads import BatchedMTPHeads, CacheArena

logger = logging.getLogger(__name__)
MTP_KEY = 'ascend_mtp'


def request_enabled(params):
    extra = params.extra_args or {}
    marker = extra.get(MTP_KEY, extra.get('ascend_external_draft_first_verify', False))
    if type(marker) is not bool:
        raise TypeError('ascend_mtp must be a bool')
    return marker


def query_buckets(states, limits, graphs=None):
    buckets = {}
    if graphs is not None and getattr(graphs.options, 'tokens', ()):
        # Keep one batch across fresh-query lengths. Split only at token/request
        # limits; use per-head totals because requests may have different budgets.
        depth = graphs.heads.num_heads
        token_limit = graphs.options.tokens[-1]
        totals, group = [0] * depth, []
        group_index = 0
        for i, (state, limit) in enumerate(zip(states, limits)):
            spans = [state.length - head.valid if d < limit else 0
                     for d, head in enumerate(state.heads)]
            span = max(spans)
            if state.length > graphs.options.context or span > token_limit:
                buckets.setdefault(('eager', (span - 1).bit_length()), []).append(i)
                continue
            combined = [old + fresh for old, fresh in zip(totals, spans)]
            if group and (len(group) >= graphs.options.max_batch or
                          max(combined) > token_limit):
                buckets['tokens', group_index] = group
                group_index += 1
                totals, group = [0] * depth, []
                combined = spans
            group.append(i)
            totals = combined
        if group:
            buckets['tokens', group_index] = group
        return buckets
    for i, (state, limit) in enumerate(zip(states, limits)):
        span = max(state.length - head.valid for head in state.heads[:limit])
        graph_query = (next((q for q in graphs.options.queries if span <= q), None)
                       if graphs is not None and state.length <= graphs.options.context else None)
        key = ('graph', graph_query) if graph_query is not None else ('eager', (span - 1).bit_length())
        buckets.setdefault(key, []).append(i)
    return buckets


def stop_ids(params, model_eos):
    ids = set(getattr(params, 'stop_token_ids', None) or ())
    if hasattr(params, 'eos_token_id'):
        if params.eos_token_id is not None:
            ids.add(params.eos_token_id)
    elif not getattr(params, 'ignore_eos', False):
        ids.update(model_eos)
    return ids


class MTPWorker:
    def __init__(self, runner, heads, arena, embedding, logits, eos, *, graphs=None):
        self.runner, self.heads, self.arena = runner, heads, arena
        self.embedding, self.logits, self.eos = embedding, logits, set(eos)
        self.states, self.disabled = {}, {}
        self.legacy_submitted = set()
        self.stats = Counter(target_forwards=0)
        self.graphs, self.profiled = graphs, False

    def finish(self, request_ids):
        for req_id in request_ids:
            state = self.states.pop(req_id, None)
            if state is not None:
                self.arena.release(state)
            self.disabled.pop(req_id, None)
            self.legacy_submitted.discard(req_id)

    def _fallback(self, req_id, reason):
        state = self.states.pop(req_id, None)
        if state is not None:
            self.arena.release(state)
        self.disabled[req_id] = reason
        self.stats['fallback_' + reason] += 1
        logger.debug('ASR_MTP_FALLBACK request_id=%s reason=%s', req_id, reason)

    def _legacy_draft(self, req_id, params, sampled, confirmed):
        extra = params.extra_args or {}
        if (not sampled or extra.get('ascend_external_draft_first_verify', False)
                or req_id in self.legacy_submitted):
            return None
        self.legacy_submitted.add(req_id)
        draft = extra.get('ascend_external_draft')
        if not draft:
            return None  # No old rollback: MTP may start after the first sample.
        if (not isinstance(draft, (list, tuple)) or
                any(type(token) is not int or token < 0 for token in draft)):
            raise ValueError('legacy external draft must contain nonnegative token IDs')
        request = self.runner.requests[req_id]
        limit = min(self.runner.speculative_config.num_speculative_tokens,
                    self.runner.input_batch.max_model_len - confirmed,
                    params.max_tokens - len(request.output_token_ids))
        offered = list(draft[:max(0, limit)])
        self.stats['legacy_draft_tokens'] += len(offered)
        return offered

    @torch.inference_mode()
    def profile(self):
        if self.states or self.arena.leased:
            raise RuntimeError('MTP memory profiling must precede live requests')
        if self.profiled:
            return
        # Graph buffers must already be resident when the eager fallback's
        # activation peak is measured. Private pool reservation is accounted
        # separately by the startup-only worker memory hook.
        if self.graphs is not None:
            self.graphs.prepare()
        before = self.heads.processed_positions, self.heads.reused_positions
        states = []
        try:
            while len(states) < getattr(self.runner, 'max_num_reqs', self.arena.storage.shape[0]):
                state = self.arena.acquire()
                if state is None:
                    break
                state.length = self.arena.max_context
                state.target.zero_()
                states.append(state)
            # Profile the worst padded full-context head batch, not only weights.
            # Temporary tensors are accounted before the engine sizes target KV.
            self.heads.propose(states, [[0] * (state.length + 1) for state in states],
                               self.embedding, self.logits, [self.heads.num_heads] * len(states))
        finally:
            for state in states:
                self.arena.release(state)
            self.heads.processed_positions, self.heads.reused_positions = before
        self.profiled = True

    @torch.inference_mode()
    def propose(self, sampled, scheduler_output, hidden):
        runner, batch = self.runner, self.runner.input_batch
        outputs = [[] for _ in sampled]
        states, inputs, limits, indices = [], [], [], []
        for i, (req_id, tokens) in enumerate(zip(batch.req_ids, sampled)):
            request = runner.requests[req_id]
            params = request.sampling_params
            if not request_enabled(params):
                continue
            if params.temperature != 0 or params.n != 1 or req_id in batch.spec_decode_unsupported_reqs:
                if req_id not in self.disabled:
                    self._fallback(req_id, 'unsupported_sampling')
                continue
            stops = stop_ids(params, self.eos)
            previous_output = len(request.output_token_ids) - len(tokens)
            minimum = getattr(params, 'min_tokens', 0)
            # A rejection-sampler block may contain accepted EOS plus a bonus.
            # Scheduler will stop at EOS, so no auxiliary work can be consumed.
            if any(token in stops and previous_output + j + 1 >= minimum
                   for j, token in enumerate(tokens)):
                continue
            confirmed = int(batch.num_tokens_no_spec[i])
            legacy = self._legacy_draft(req_id, params, tokens, confirmed)
            if legacy is not None:
                outputs[i] = legacy
            if req_id in self.disabled:
                continue
            start = int(batch.num_computed_tokens_cpu[i])
            count = scheduler_output.num_scheduled_tokens[req_id]
            # Bookkeeping has appended accepted draft IDs + correction/bonus.
            # The last one is a drive token, not yet represented by any hidden row.
            end = confirmed - 1 if tokens else start + count
            if end > self.arena.max_context:
                self._fallback(req_id, 'context_cap')
                continue
            if not start <= end <= start + count:
                raise RuntimeError('ASR MTP requires post-bookkeeping v0.23 hidden/token offsets')
            state = self.states.get(req_id)
            if state is None:
                if start:
                    self._fallback(req_id, 'missing_hidden_prefix')
                    continue
                state = self.arena.acquire()
                if state is None:
                    self._fallback(req_id, 'cache_capacity')
                    continue
                self.states[req_id] = state
            if start > state.length:
                self._fallback(req_id, 'missing_hidden_prefix')
                continue
            if start < state.length:
                state.crop_heads(start)  # recomputation/preemption overwrites old rows
            query_start = int(runner.query_start_loc.np[i])
            state.target[start:end].copy_(hidden[query_start:query_start + end - start])
            state.length = end
            state.crop_heads(end)
            if not tokens or legacy is not None:
                # Keep confirmed prefill hidden while the legacy rollback goes
                # through the target once. Do not skip it by immediately drafting MTP.
                continue
            remaining = params.max_tokens - len(request.output_token_ids)
            limit = min(self.heads.num_heads, runner.speculative_config.num_speculative_tokens,
                        batch.max_model_len - confirmed, remaining)
            if limit <= 0:
                continue
            row = batch.token_ids_cpu[i, :confirmed].tolist()
            if row[-len(tokens):] != tokens:
                raise RuntimeError('ASR MTP must run after confirmed tokens are bookkept')
            states.append(state)
            inputs.append(row)
            limits.append(limit)
            indices.append(i)
        if states:
            before = self.heads.processed_positions
            buckets = query_buckets(states, limits, self.graphs)
            drafts = torch.zeros((len(states), self.heads.num_heads),
                                 device=states[0].target.device, dtype=torch.long)
            for group in buckets.values():
                batch_states = [states[i] for i in group]
                batch_inputs = [inputs[i] for i in group]
                batch_limits = [limits[i] for i in group]
                proposal = (None if self.graphs is None else
                            self.graphs.propose(batch_states, batch_inputs, batch_limits))
                if proposal is None:
                    proposal = self.heads.propose(batch_states, batch_inputs,
                        self.embedding, self.logits, batch_limits)
                    self.stats['eager_head_batches'] += 1
                else:
                    self.stats['graph_head_batches'] += 1
                drafts[group] = proposal
            rows = drafts.tolist()  # One device-to-host boundary for the whole batch.
            mtp_tokens = 0
            for i, row, limit in zip(indices, rows, limits):
                row = row[:limit]
                params = runner.requests[batch.req_ids[i]].sampling_params
                stops = stop_ids(params, self.eos)
                output_count = len(runner.requests[batch.req_ids[i]].output_token_ids)
                for j, token in enumerate(row):
                    if token in stops and output_count + j + 1 >= getattr(params, 'min_tokens', 0):
                        row = row[:j + 1]
                        break
                outputs[i] = row
                mtp_tokens += len(row)
            self.stats['proposal_batches'] += 1
            self.stats['head_batches'] += len(buckets)
            self.stats['draft_tokens'] += mtp_tokens
            self.stats['head_positions'] += self.heads.processed_positions - before
            logger.debug('ASR_MTP_STEP requests=%d draft_tokens=%d new_head_positions=%d '
                         'graph_batches_total=%d eager_batches_total=%d graph_stats=%s',
                         len(states), mtp_tokens, self.heads.processed_positions - before,
                         self.stats['graph_head_batches'], self.stats['eager_head_batches'],
                         self.graphs.stats if self.graphs is not None else {})
        return outputs


def validate_engine(runner):
    config = runner.vllm_config
    parallel = config.parallel_config
    names = ('tensor_parallel_size', 'pipeline_parallel_size',
             'prefill_context_parallel_size', 'decode_context_parallel_size')
    if any(getattr(parallel, name, 1) != 1 for name in names):
        raise ValueError('ASR MTP currently requires TP=PP=PCP=DCP=1')
    if runner.use_async_scheduling:
        raise ValueError('ASR MTP requires async_scheduling=False')
    if config.cache_config.enable_prefix_caching:
        raise ValueError('ASR MTP requires enable_prefix_caching=False: KV hits lack target hidden history')
    if config.quant_config is not None or getattr(config, 'lora_config', None) is not None:
        raise ValueError('ASR MTP currently requires an unquantized target without LoRA')


def base_signature(model_path):
    root = Path(model_path)
    value = hashlib.sha256((root / 'config.json').read_bytes())
    for path in sorted(root.glob('*.safetensors')):
        stat = path.stat()
        value.update(f'{path.name}:{stat.st_size}:{stat.st_mtime_ns}'.encode())
    return value.hexdigest()


def check_base_signature(metadata, model_path, settings):
    """Optionally check the training fingerprint, which includes shard mtimes."""
    verify = settings.get('verify_base_signature', False)
    if type(verify) is not bool:
        raise TypeError('asr_mtp.verify_base_signature must be a bool')
    if not verify:
        return None
    actual = base_signature(model_path)
    saved = metadata.get('base_signature')
    if saved != actual:
        raise ValueError('MTP checkpoint backbone signature differs from the target: '
                         f'saved_signature={saved} target_signature={actual} '
                         f'training_model={metadata.get("base_model")} target_model={model_path}. '
                         'The legacy fingerprint includes config bytes, shard names, sizes and mtimes.')
    return actual


def load_worker(runner, target, settings):
    validate_engine(runner)
    allowed = {'checkpoint', 'num_heads', 'max_context_tokens', 'cache_mib', 'max_requests',
               'verify_base_signature', 'graph_mode', 'graph_context_tokens',
               'graph_query_sizes', 'graph_token_sizes', 'graph_max_batch_size'}
    if not isinstance(settings, dict) or set(settings) - allowed or not settings.get('checkpoint'):
        raise ValueError('invalid additional_config.asr_mtp settings')
    for name, default in (('num_heads', 5), ('max_context_tokens', 512),
                          ('cache_mib', 512), ('max_requests', 64)):
        if type(settings.get(name, default)) is not int or settings.get(name, default) < 1:
            raise ValueError(f'asr_mtp.{name} must be a positive integer')
    config = runner.model_config.hf_config
    if getattr(config, 'model_type', None) != 'qwen3_asr':
        raise ValueError('ASR MTP requires the Qwen3-ASR target architecture')
    text = config.thinker_config.text_config
    if not getattr(text, 'head_dim', None):
        raise ValueError('ASR MTP text config is missing head_dim')
    # This is the user-supplied trusted training checkpoint. mmap avoids loading
    # unused optimizer tensor storage into RAM; no remote download occurs here.
    checkpoint = torch.load(settings['checkpoint'], map_location='cpu', mmap=True, weights_only=False)
    metadata, weights = checkpoint['metadata'], checkpoint['mtp']
    check_base_signature(metadata, runner.model_config.model, settings)
    trained = metadata['num_heads']
    if type(trained) is not int or trained < 1:
        raise ValueError('invalid checkpoint num_heads')
    heads = min(settings.get('num_heads', trained), trained, runner.speculative_config.num_speculative_tokens)
    selected = {name: value for name, value in weights.items()
                if any(name.startswith(f'blocks.{i}.') for i in range(heads))}
    with torch.device('meta'):
        model = BatchedMTPHeads(text, heads)
    model.load_state_dict(selected, strict=True, assign=True)
    model = model.to(device=runner.device, dtype=runner.dtype).eval().requires_grad_(False)
    cap = settings.get('max_context_tokens', 512)
    per_slot = CacheArena.elements_per_position(text, heads) * cap * torch.empty((), dtype=runner.dtype).element_size()
    slots = min(settings.get('max_requests', 64), settings.get('cache_mib', 512) * 1024 ** 2 // per_slot)
    if slots < 1:
        raise ValueError('asr_mtp.cache_mib cannot hold one request slot')
    options = graph_options(settings, context=cap, slots=slots,
                            max_batch=getattr(runner, 'max_num_reqs', slots))
    if options is not None:
        validate_graph_runner(runner)
    arena = CacheArena(text, heads, max_context=cap, slots=slots,
                       device=runner.device, dtype=runner.dtype,
                       scratch_elements=options.scratch_elements(text) if options else 0)
    language = target.get_language_model() if hasattr(target, 'get_language_model') else target.language_model
    eos = getattr(text, 'eos_token_id', None)
    eos = set(eos if isinstance(eos, (tuple, list)) else [eos]) - {None}
    graphs = None
    if options is not None:
        graph_class = MTPGraphs
        if options.tokens:
            try:
                from .asr_mtp_token_graph import TokenMTPGraphs
            except ImportError:
                from asr_mtp_token_graph import TokenMTPGraphs
            graph_class = TokenMTPGraphs
        graphs = graph_class(options, model, arena, language.embed_input_ids,
                             language.compute_logits)
    worker = MTPWorker(runner, model, arena, language.embed_input_ids, language.compute_logits,
                       eos, graphs=graphs)
    logger.info('ASR_MTP_CONFIG heads=%d context_cap=%d cache_slots=%d cache_mib=%.1f checkpoint=%s '
                'verify_base_signature=%s graph_mode=%s graph_scratch_mib=%.2f graph_layout=%s',
                heads, cap, slots, arena.storage.numel() * arena.storage.element_size() / 1024 ** 2,
                settings['checkpoint'], settings.get('verify_base_signature', False),
                settings.get('graph_mode', 'off'),
                (arena.graph_storage.numel() - arena.scratch_offset) *
                arena.storage.element_size() / 1024 ** 2,
                'tokens' if options is not None and options.tokens else 'rectangular')
    return worker
