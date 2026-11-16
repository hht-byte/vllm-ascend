# SPDX-License-Identifier: Apache-2.0
"""Opt-in rejection/prefill handoff, without a host sample read before MTP.

FIA retains CPU-known optimistic Q/KV lengths. Causal queries at the actual
confirmed end cannot see the extra computed tail. A separate small graph writes
device-derived lengths, drive/sequence, and last-row indices into stable buffers.
"""
import logging
from collections import Counter
from dataclasses import replace

import torch

try:
    from .asr_mtp_staging import UploadBatch
except ImportError:
    from asr_mtp_staging import UploadBatch

MAX_STOPS = 8
logger = logging.getLogger(__name__)


class RejectionPrelude:
    def __init__(self, program, *, sample_width, vocab_size):
        if not program.paged_kv:
            raise ValueError('device rejection requires paged token graphs')
        self.program, self.width, self.vocab = program, sample_width, vocab_size
        self.compute_limits = program.limits.clone()
        b, h = program.max_batch, program.heads.num_heads
        device = program.output.device
        self.raw = torch.full((b, sample_width), -1, device=device, dtype=torch.int32)
        # Allocate outside the private capture pool: both graphs share these
        # outputs; pool temporaries must not alias values crossing graph boundaries.
        self.combined = torch.zeros((b, sample_width+h), device=device, dtype=torch.long)
        program.output = self.combined[:, sample_width:]
        self.host = torch.empty_like(self.combined, device='cpu', pin_memory=device.type == 'npu')
        self.event = torch.npu.Event() if device.type == 'npu' else None
        self.hidden = torch.zeros((program.query_capacity, program.heads.config.hidden_size),
                                  device=device, dtype=program.arena.storage.dtype)
        size = b*(5+MAX_STOPS)
        self.metadata = torch.zeros(size+program.query_capacity, device=device, dtype=torch.long)
        self.config = self.metadata[:size].view(b, 5+MAX_STOPS)
        self.addresses = self.metadata[size:]
        self.uploads = UploadBatch(device)
        self.metadata_upload = self.uploads.buffer(self.metadata)
        self.column = torch.arange(sample_width, device=device)[None, :]
        self.position = torch.arange(program.sequence_width, device=device)[None, :]
        self.replay = self.run

    def stage(self, previous, counts, remaining, model_remaining, minimum_left, stops,
              sampled, hidden, *, starts=None, sample_counts=None):
        p, b = self.program, len(previous)
        total = sum(counts)
        if (not 0 < b <= p.max_batch or total > p.tokens or
                sampled.ndim != 2 or sampled.shape[0] != b or sampled.shape[1] > self.width or
                hidden.shape[0] < total or any(len(s) > MAX_STOPS for s in stops)):
            raise ValueError('device rejection staging exceeds captured bounds')
        payload = []
        addresses = []
        starts = [c-1 for c in previous] if starts is None else starts
        sample_counts = counts if sample_counts is None else sample_counts
        stride = p.arena.storage.shape[1] // p.row_width
        # Slots were CPU-staged by program.stage; never fetch p.slots from device.
        slots = p.metadata_upload.array.reshape(-1)
        slot_offset = 5 * p.heads.num_heads * (p.max_batch+1)
        for i, (c, count) in enumerate(zip(previous, counts)):
            payload.extend([c, remaining[i], model_remaining[i], minimum_left[i], sample_counts[i],
                            *sorted(stops[i]), *([-1]*(MAX_STOPS-len(stops[i])))])
            base = int(slots[slot_offset+i]) * stride
            addresses.extend(range(base+starts[i], base+starts[i]+count))
        payload.extend([0]*(self.config.numel()-len(payload)))
        sink = p.arena.scratch_offset // p.row_width
        addresses.extend(range(sink+total, sink+p.query_capacity))
        with self.uploads:
            self.metadata_upload.copy(payload+addresses)
        self.raw[:b, :sampled.shape[1]].copy_(sampled)
        self.hidden[:total].copy_(hidden[:total])

    @torch.inference_mode()
    def run(self):
        p, b = self.program, self.program.max_batch
        previous, remaining, model_remaining, minimum, scheduled = self.config[:, :5].unbind(-1)
        valid = ((self.raw >= 0) & (self.raw < self.vocab) &
                 (self.column < scheduled[:, None]) & (previous[:, None] > 0))
        n = valid.sum(-1)
        # Discarded intermediate prefill has no sampled append. Its hidden
        # prefix still advances to the CPU-known end of target execution.
        end = torch.where(scheduled > 0, previous+n-1, p.lengths[:b])
        live = (previous > 0) & (n > 0)
        stop = (self.raw[:, :, None] == self.config[:, None, 5:]).any(-1)
        terminal = (stop & valid & (self.column+1 >= minimum[:, None])).any(-1)
        limits = torch.minimum(p.limits[:b], torch.minimum(remaining-n, model_remaining-n))
        limits = torch.where(live & ~terminal, limits.clamp_min(0), 0)
        delta = p.lengths[:b]-end
        # last[] is in packed Q coordinates; tail length is the same at every head.
        p.last[:, :b].copy_(torch.where(p.last[:, :b] >= delta[None, :],
                                       p.last[:, :b]-delta[None, :], 0))
        p.lengths[:b].copy_(end.clamp_min(0))
        p.limits[:b].copy_(limits)
        relative = self.position-previous[:, None]
        new_ids = self.raw.gather(1, relative.clamp(0, self.width-1))
        new_ids = torch.where((relative >= 0) & (relative < n[:, None]), new_ids, 0)
        p.sequence[:b].copy_(torch.where(relative >= 0, new_ids, p.sequence[:b]))
        if p.npu_row_update is not None:
            p.npu_row_update(p.cache_rows, self.addresses[:, None], self.hidden)
        else:
            p.cache_rows.index_copy_(0, self.addresses, self.hidden)
        self.combined[:, :self.width].copy_(torch.where(valid, self.raw, -1))

    def capture(self, pool):
        # Startup only. Existing FULL MTP graph and FIA tasks are unchanged.
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            self.run()
        torch.npu.current_stream().wait_stream(stream)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, pool=pool):
            self.run()
        torch.npu.synchronize()
        self.replay = graph.replay
        self.graph = graph

    def readback(self, count):
        # Only wait after MTP. Scheduler still needs CPU samples, but they and
        # drafts now cross the boundary together, not on either side of MTP.
        self.host.copy_(self.combined, non_blocking=True)
        if self.event is not None:
            self.event.record(torch.npu.current_stream())
            self.event.synchronize()
        # This single completion fence also covers both preceding H2D uploads.
        # Their next reuse no longer needs an Event::query / synchronize.
        self.uploads.pending = self.program.uploads.pending = False
        return self.host[:count]


def optimistic_states(states, previous, counts, starts=None):
    starts = [c-1 for c in previous] if starts is None else starts
    return [replace(s, length=start+count,
                    heads=[replace(h, valid=min(h.valid, max(0, c-d)))
                           for d, h in enumerate(s.heads, 1)])
            for s, c, count, start in zip(states, previous, counts, starts)]


class DeviceRejectionHandoff:
    """One pending serial batch; guards never inspect device tensor values."""
    def __init__(self, worker):
        self.worker, self.pending, self.stats = worker, None, Counter()

    def miss(self, reason):
        self.stats['fallback_'+reason] += 1

    @torch.inference_mode()
    def prepare(self, scheduler, sampler, hidden):
        try:
            from .asr_mtp_graph import captured_depth, depth_key
            from .asr_mtp_heads import HeadHistory, RequestHistory
            from .asr_mtp_token_graph import required_tokens
            from .asr_mtp_worker import request_enabled, request_type, stop_ids
        except ImportError:
            from asr_mtp_graph import captured_depth, depth_key
            from asr_mtp_heads import HeadHistory, RequestHistory
            from asr_mtp_token_graph import required_tokens
            from asr_mtp_worker import request_enabled, request_type, stop_ids
        w, r = self.worker, self.worker.runner
        batch, graphs = r.input_batch, w.graphs
        if self.pending is not None:
            raise RuntimeError('previous device rejection handoff was not consumed')
        samples = sampler.sampled_token_ids
        n = len(batch.req_ids)
        if (not graphs.ready or not 0 < n <= graphs.options.max_batch or
                samples.ndim != 2 or samples.shape[0] != n or samples.shape[1] < 1 or
                samples.dtype not in (torch.int32, torch.int64) or sampler.logprobs_tensors is not None or
                getattr(r, 'routed_experts_initialized', False) or r.use_async_scheduling):
            return self.miss('batch')
        previous, counts, states, rows, limits, remaining, capacity, minimum, stops = (
            [] for _ in range(9))
        starts, sample_counts, missing = [], [], []
        discarded = set(r.discard_request_indices.np[:r.num_discarded_requests]) if (
            getattr(r, 'num_discarded_requests', 0)) else set()
        free_slots = list(reversed(w.arena.free))
        cursor = 0
        for i, req in enumerate(batch.req_ids):
            state, request = w.states.get(req), r.requests[req]
            params = request.sampling_params
            if (not request_enabled(params) or
                    req in w.disabled or params.temperature != 0 or params.n != 1 or
                    req in batch.spec_decode_unsupported_reqs):
                return self.miss('request')
            c = int(batch.num_tokens_no_spec[i])
            start = int(batch.num_computed_tokens_cpu[i])
            count = scheduler.num_scheduled_tokens.get(req, 0)
            draft = scheduler.scheduled_spec_decode_tokens.get(req, ())
            end = start+count
            partial = end < c
            sample_count = 0 if partial else len(draft)+1
            if (c < 1 or start < 0 or count < 1 or start >= c or
                    (state is None and start != 0) or (state is not None and state.length != start) or
                    (partial and (draft or i not in discarded)) or
                    (not partial and (end != c+len(draft) or i in discarded)) or
                    sample_count > samples.shape[1] or int(r.query_start_loc.np[i]) != cursor or
                    (getattr(scheduler, 'num_invalid_spec_tokens', None) or {}).get(req, 0)):
                return self.miss('geometry')
            if end > graphs.options.context:
                return self.miss('context')
            if state is None:
                if len(missing) >= len(free_slots):
                    return self.miss('slots')
                # Geometry/capacity preflight is CPU-only. Lease real arena
                # views only after the complete mixed batch is known to fit.
                state = RequestHistory(free_slots[len(missing)], None,
                                       [HeadHistory(None, None, None) for _ in range(w.heads.num_heads)])
                missing.append(req)
            policy = w.request_heads.get(request_type(req), (w.cold_heads, w.warm_heads))
            left = params.max_tokens-len(request.output_token_ids)
            cap = batch.max_model_len-c
            limit = max(0, min(policy[req in w.proposed],
                               r.speculative_config.num_speculative_tokens, left-1, cap-1))
            extra = params.extra_args or {}
            legacy = (req not in w.legacy_submitted and extra.get('ascend_external_draft') and
                      not extra.get('ascend_external_draft_first_verify'))
            if partial or legacy:
                limit = 0
            stop = stop_ids(params, w.eos)
            if len(stop) > MAX_STOPS:
                return self.miss('budget')
            previous.append(c)
            counts.append(count)
            starts.append(start)
            sample_counts.append(sample_count)
            states.append(state)
            known = min(c, end+1)
            rows.append(batch.token_ids_cpu[i, :known].tolist()+[0]*(end+1-known))
            limits.append(limit)
            remaining.append(left)
            capacity.append(cap)
            minimum.append(getattr(params, 'min_tokens', 0)-len(request.output_token_ids))
            stops.append(stop)
            cursor += count
        if hidden.shape[0] < cursor:
            return self.miss('hidden')
        upper = optimistic_states(states, previous, counts, starts)
        needed = max(cursor, required_tokens(upper, limits, w.heads.num_heads))
        depth = captured_depth(graphs.options, w.heads.num_heads, max(1, max(limits)))
        bucket = next((t for t in graphs.options.tokens if needed <= t), None)
        if bucket is None or depth is None:
            return self.miss('tokens')
        program, replay = graphs.entries[depth_key(graphs.options, depth, bucket)]
        if not hasattr(program, 'rejection') or samples.shape[1] > program.rejection.width:
            return self.miss('capture')
        leased = []
        try:
            for req in missing:
                state = w.arena.acquire()
                if state is None:
                    raise RuntimeError('MTP arena capacity changed after preflight')
                leased.append((req, state))
                w.states[req] = state
            # No valid head span for cache-only rows. These stay in original
            # batch order for sample readback and late external-draft routing.
            program.stage(upper, rows, limits, allow_inactive=True)
            prelude = program.rejection
            prelude.stage(previous, counts, remaining, capacity, minimum, stops, samples, hidden,
                          starts=starts, sample_counts=sample_counts)
        except BaseException:
            for req, state in reversed(leased):
                del w.states[req]
                w.arena.release(state)
            raise
        if any(limits):
            replay(prelude=prelude.replay)
        else:
            # Chunked prefill / pending rollback: cache target hidden without
            # running any auxiliary heads or charging a fake proposal round.
            prelude.replay()
        combined = prelude.readback(n)
        # These lists are CPU data; no more device reads in late propose().
        proposal = combined[:, prelude.width:].tolist()
        costs = [sum(s.length-h.valid for h in s.heads[:limit])
                 for s, limit in zip(upper, limits)]
        self.pending = (scheduler, tuple(batch.req_ids), proposal, upper, costs)
        active = sum(limit > 0 for limit in limits)
        if active:
            graphs.stats['replays'] += 1
            graphs.stats[f'replays_h{depth}'] += 1
            graphs.stats['requests'] += active
            w.stats['graph_head_batches'] += 1
        # Execution happened even if CPU bookkeeping subsequently drops every
        # proposal at EOS/length. Account it once, independently of consumption.
        w.heads.processed_positions += sum(costs)
        w.heads.reused_positions += sum(h.valid for s, limit in zip(upper, limits)
                                       for h in s.heads[:limit])
        w.stats['head_positions'] += sum(costs)
        w.stats['head_batches'] += bool(active)
        self.stats['batches'] += 1
        self.stats['requests'] += n
        self.stats['cold_requests'] += sum(bool(limit) and req not in w.proposed
                                           for req, limit in zip(batch.req_ids, limits))
        self.stats['warm_requests'] += sum(bool(limit) and req in w.proposed
                                           for req, limit in zip(batch.req_ids, limits))
        self.stats['cache_only_requests'] += n-active
        self.stats['optimistic_positions'] += sum(costs)
        if self.stats['batches'] == 1 or self.stats['batches'] % 100 == 0:
            logger.info('ASR_MTP_DEVICE_REJECTION batches=%d requests=%d bucket=%d '
                        'readbacks=%d optimistic_positions=%d cold_requests=%d '
                        'warm_requests=%d cache_only_requests=%d fallbacks=%s',
                        self.stats['batches'], self.stats['requests'], bucket,
                        self.stats['batches'], self.stats['optimistic_positions'],
                        self.stats['cold_requests'], self.stats['warm_requests'],
                        self.stats['cache_only_requests'],
                        {k: v for k, v in self.stats.items() if k.startswith('fallback_')})
        # Always width >= 2: even non-spec prefill must take the original CPU
        # rejection parser, not _to_list's unconditional device event wait.
        return combined[:, :prelude.width]

    def take(self, scheduler):
        pending, self.pending = self.pending, None
        if pending is None:
            return None
        old, ids, rows, upper, costs = pending
        if old is not scheduler or ids != tuple(self.worker.runner.input_batch.req_ids):
            raise RuntimeError('device rejection batch changed before bookkeeping completed')
        return rows, upper, costs
