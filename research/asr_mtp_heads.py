# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-compatible shifted MTP heads, batched inside the engine worker.

No target model, HF generate loop, paged-KV modification or spec_asr dependency.
The caller supplies confirmed target hidden states and the last sampled drive
token. All auxiliary IDs stay on device until the worker's one batch D2H.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

try:
    from .asr_mtp_fused import add_rms_norm, mlp_forward, project_qkv
except ImportError:
    from asr_mtp_fused import add_rms_norm, mlp_forward, project_qkv


@dataclass
class HeadHistory:
    hidden: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    valid: int = 0


@dataclass
class RequestHistory:
    slot: int
    target: torch.Tensor
    heads: list[HeadHistory]
    length: int = 0

    def crop_heads(self, confirmed_input_length):
        # Depth d's anchor p consumes token p+d. Only executed, confirmed input
        # tokens are reusable; the new correction/bonus drive is not executed.
        for depth, head in enumerate(self.heads, 1):
            head.valid = max(0, min(head.valid, confirmed_input_length - depth))


class CacheArena:
    """Allocate bounded request storage at model load, before target KV sizing."""
    @staticmethod
    def elements_per_position(config, heads):
        return ((heads + 1) * config.hidden_size +
                2 * heads * config.num_key_value_heads * config.head_dim)

    def __init__(self, config, heads, *, max_context, slots, device, dtype,
                 scratch_elements=0):
        if min(max_context, slots, heads) < 1:
            raise ValueError('cache dimensions must be positive')
        self.config, self.num_heads = config, heads
        width = self.elements_per_position(config, heads)
        if type(scratch_elements) is not int or scratch_elements < 0:
            raise ValueError('scratch_elements must be a nonnegative integer')
        self.scratch_offset = slots * max_context * width
        self.graph_storage = torch.empty(self.scratch_offset + scratch_elements,
                                         device=device, dtype=dtype)
        self.storage = self.graph_storage[:self.scratch_offset].view(slots, max_context * width)
        self.max_context = max_context
        self.free = list(reversed(range(slots)))
        self.leased = set()

    def acquire(self):
        if not self.free:
            return None
        slot = self.free.pop()
        self.leased.add(slot)
        data, offset = self.storage[slot], 0
        def view(shape):
            nonlocal offset
            size = 1
            for extent in shape:
                size *= extent
            result = data[offset:offset + size].view(shape)
            offset += size
            return result
        cap, cfg = self.max_context, self.config
        target = view((cap, cfg.hidden_size))
        histories = []
        for _ in range(self.num_heads):
            histories.append(HeadHistory(
                view((cap, cfg.hidden_size)),
                view((cfg.num_key_value_heads, cap, cfg.head_dim)),
                view((cfg.num_key_value_heads, cap, cfg.head_dim))))
        return RequestHistory(slot, target, histories)

    def release(self, state):
        if state.slot not in self.leased:
            raise ValueError('MTP cache slot was already released')
        self.leased.remove(state.slot)
        self.free.append(state.slot)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        width, dim, bias = cfg.hidden_size, cfg.head_dim, cfg.attention_bias
        self.q_proj = nn.Linear(width, cfg.num_attention_heads * dim, bias=bias)
        self.k_proj = nn.Linear(width, cfg.num_key_value_heads * dim, bias=bias)
        self.v_proj = nn.Linear(width, cfg.num_key_value_heads * dim, bias=bias)
        self.o_proj = nn.Linear(cfg.num_attention_heads * dim, width, bias=bias)
        self.q_norm, self.k_norm = (nn.RMSNorm(dim, eps=cfg.rms_norm_eps) for _ in range(2))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, hidden):
        return mlp_forward(self, hidden)


class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.self_attn, self.mlp = Attention(cfg), MLP(cfg)
        self.input_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.hidden_norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.token_norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.fusion = nn.Linear(cfg.hidden_size * 2, cfg.hidden_size, bias=False)
        self.decoder = Decoder(cfg)
        self.output_norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)


def rotate_half(tensor):
    left, right = tensor.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


class BatchedMTPHeads(nn.Module):
    def __init__(self, config, num_heads):
        super().__init__()
        if (config.hidden_act != 'silu' or config.head_dim % 2 or
                config.num_attention_heads % config.num_key_value_heads):
            raise ValueError('unsupported MTP decoder geometry/activation')
        rope = getattr(config, 'rope_scaling', None) or {}
        if rope.get('rope_type', rope.get('type', 'default')) != 'default':
            raise ValueError('online MTP currently requires default RoPE')
        self.config = config
        self.blocks = nn.ModuleList([Block(config) for _ in range(num_heads)])
        self.num_heads = num_heads
        self.processed_positions = self.reused_positions = 0
        self._inverse_frequency = None

    @torch.inference_mode()
    def propose(self, states, inputs, embedding, logits, limits):
        """Return [batch, heads] auxiliary IDs on device, excluding drive tokens."""
        cfg, batch = self.config, len(states)
        if not batch or len(inputs) != batch or len(limits) != batch:
            raise ValueError('aligned nonempty MTP batch required')
        if any(len(row) != state.length + 1 or not 0 < count <= self.num_heads
               for state, row, count in zip(states, inputs, limits)):
            raise ValueError('MTP history/drive/budget mismatch')
        length = max(state.length for state in states)
        device, dtype = states[0].target.device, states[0].target.dtype
        sequence = torch.tensor([row + [0] * (length + self.num_heads + 1 - len(row))
                                 for row in inputs], device=device, dtype=torch.long)
        outputs = torch.zeros((batch, self.num_heads), device=device, dtype=torch.long)
        if self._inverse_frequency is None or self._inverse_frequency.device != device:
            self._inverse_frequency = 1.0 / (cfg.rope_theta ** (torch.arange(
                0, cfg.head_dim, 2, device=device, dtype=torch.float32) / cfg.head_dim))
        inverse = self._inverse_frequency
        for depth, block in enumerate(self.blocks, 1):
            rows = [i for i, limit in enumerate(limits) if depth <= limit]
            if not rows:
                break
            starts = [states[i].heads[depth - 1].valid for i in rows]
            lengths = [states[i].length for i in rows]
            sizes = [end - start for start, end in zip(starts, lengths)]
            if min(sizes) <= 0:
                raise ValueError('MTP cached prefix must leave a fresh prediction anchor')
            queries, keys, count = max(sizes), max(lengths), len(rows)
            source = torch.zeros((count, queries, cfg.hidden_size), device=device, dtype=dtype)
            for j, i in enumerate(rows):
                old = states[i].target if depth == 1 else states[i].heads[depth - 2].hidden
                source[j, :sizes[j]].copy_(old[starts[j]:lengths[j]])
            anchors = torch.tensor(starts, device=device)[:, None] + torch.arange(queries, device=device)
            indices = (anchors + depth).clamp_max(sequence.shape[1] - 1)
            shifted = sequence[rows].gather(1, indices)
            token_embeddings = embedding(shifted.reshape(-1)).reshape(count, queries, cfg.hidden_size)
            fused = block.fusion(torch.cat((block.hidden_norm(source),
                                           block.token_norm(token_embeddings)), -1))
            normed = block.decoder.input_layernorm(fused)
            attn = block.decoder.self_attn
            q_raw, k_raw, v_raw = project_qkv(attn, normed)
            q = attn.q_norm(q_raw.view(count, queries, cfg.num_attention_heads,
                                                     cfg.head_dim)).transpose(1, 2)
            k = attn.k_norm(k_raw.view(count, queries, cfg.num_key_value_heads,
                                                     cfg.head_dim)).transpose(1, 2)
            v = v_raw.view(count, queries, cfg.num_key_value_heads,
                                        cfg.head_dim).transpose(1, 2)
            # All three MRoPE axes in SequentialMTP use the same depth+anchor;
            # their interleaving therefore reduces to ordinary default RoPE.
            frequencies = (anchors + depth).float().unsqueeze(-1) * inverse
            angles = torch.cat((frequencies, frequencies), -1)
            cos, sin = angles.cos().to(dtype)[:, None], angles.sin().to(dtype)[:, None]
            q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
            past_k = torch.zeros((count, cfg.num_key_value_heads, keys, cfg.head_dim),
                                 device=device, dtype=dtype)
            past_v = torch.zeros_like(past_k)
            for j, i in enumerate(rows):
                history, start, end, size = states[i].heads[depth - 1], starts[j], lengths[j], sizes[j]
                history.key[:, start:end].copy_(k[j, :, :size])
                history.value[:, start:end].copy_(v[j, :, :size])
                past_k[j, :, :end].copy_(history.key[:, :end])
                past_v[j, :, :end].copy_(history.value[:, :end])
            key_index = torch.arange(keys, device=device)
            allowed = ((key_index[None, None, :] <= anchors[:, :, None]) &
                       (key_index[None, None, :] < torch.tensor(lengths, device=device)[:, None, None]))
            groups = cfg.num_attention_heads // cfg.num_key_value_heads
            attended = F.scaled_dot_product_attention(
                q, past_k.repeat_interleave(groups, 1), past_v.repeat_interleave(groups, 1),
                attn_mask=allowed[:, None], dropout_p=0.0, scale=cfg.head_dim ** -0.5)
            normed_residual, residual = add_rms_norm(
                block.decoder.post_attention_layernorm, fused,
                attn.o_proj(attended.transpose(1, 2).reshape(count, queries, -1)))
            fresh, _ = add_rms_norm(block.output_norm, residual,
                                    block.decoder.mlp(normed_residual))
            last = torch.stack([fresh[j, size - 1] for j, size in enumerate(sizes)])
            predicted = logits(last).argmax(-1)
            outputs[rows, depth - 1] = predicted
            sequence[rows, torch.tensor(lengths, device=device) + depth] = predicted
            for j, i in enumerate(rows):
                history = states[i].heads[depth - 1]
                history.hidden[starts[j]:lengths[j]].copy_(fresh[j, :sizes[j]])
                history.valid = lengths[j]
            self.processed_positions += sum(sizes)
            self.reused_positions += sum(starts)
        return outputs
