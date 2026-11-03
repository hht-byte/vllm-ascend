"""Torch backend for rollback verification followed by spec-asr MTP.

This reference backend deliberately recomputes MTP histories through
``SequentialMTP.propose``. It measures correctness and acceptance, not vLLM
serving latency. Import it only after adding the spec-asr checkout to sys.path.
"""

from __future__ import annotations

import torch
from spec_asr.runtime import decode_tokens, prefill

if __package__:
    from .mtp_rollback_handoff import Verification, confirm_draft
else:
    from mtp_rollback_handoff import Verification, confirm_draft


class TorchHandoffBackend:
    def __init__(self, model, batch, mtp, eos_token_ids):
        if batch["input_ids"].shape[0] != 1 or not batch["attention_mask"].all():
            raise ValueError("reference backend requires one unpadded prompt")
        self.model = model
        self.mtp = mtp
        self.eos = set(eos_token_ids)
        self.head = model.thinker.lm_head
        self.text = model.thinker.model
        self.ids = batch["input_ids"].clone()
        output = prefill(model, batch, use_cache=True)
        self.cache = output.past_key_values
        self.hidden = output.last_hidden_state

    @torch.no_grad()
    def verify(self, draft: tuple[int, ...]) -> Verification:
        if not draft:
            raise ValueError("verification requires a draft")
        proposed = torch.tensor([draft], device=self.ids.device, dtype=torch.long)
        previous_length = self.cache.get_seq_length()
        verified = decode_tokens(self.model, proposed, self.cache)
        expected = [int(self.head(self.hidden[:, -1]).argmax(-1).item())]
        if len(draft) > 1:
            expected.extend(self.head(verified.last_hidden_state[:, :-1])
                            .argmax(-1)[0].tolist())
        confirmed, accepted = confirm_draft(draft, expected)
        calls = 1
        if accepted == len(draft):
            self.cache = verified.past_key_values
            self.hidden = torch.cat((self.hidden, verified.last_hidden_state), dim=1)
        else:
            self.cache = verified.past_key_values
            self.cache.crop(previous_length + accepted)
            self.hidden = torch.cat(
                (self.hidden, verified.last_hidden_state[:, :accepted]), dim=1)
            correction = confirmed[-1]
            if correction not in self.eos:
                corrected = torch.tensor([[correction]], device=self.ids.device,
                                         dtype=torch.long)
                corrected_output = decode_tokens(self.model, corrected, self.cache)
                self.cache = corrected_output.past_key_values
                self.hidden = torch.cat(
                    (self.hidden, corrected_output.last_hidden_state), dim=1)
                calls += 1
        self.ids = torch.cat(
            (self.ids, torch.tensor([confirmed], device=self.ids.device,
                                    dtype=torch.long)), dim=1)
        return Verification(confirmed, accepted, calls)

    @torch.no_grad()
    def propose_mtp(self, max_heads: int) -> tuple[int, ...]:
        first = self.head(self.hidden[:, -1]).argmax(-1, keepdim=True)
        if int(first.item()) in self.eos or self.mtp is None or max_heads == 0:
            return (int(first.item()),)
        proposal = self.mtp.propose(
            self.hidden, self.ids, first, self.text.embed_tokens,
            self.head, self.text.rotary_emb, max_heads=max_heads)
        return tuple(int(token) for token in proposal[0].tolist())
