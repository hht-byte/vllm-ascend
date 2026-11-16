"""Bounded first-verification scheduler port for the frozen blue 0.23 engine.

This module only generates source/diffs. It never imports or changes an engine.
The base port retains the existing first-verify contract while accounting for
the 0.23 remote-KV receive path, which allocates slots without executing tokens.
"""

from __future__ import annotations

import difflib
from pathlib import Path

from build_external_draft_first_verify_patch import (
    base_scheduler as _original_base_scheduler,
    recompute_scheduler as _original_recompute_scheduler,
    replace_once,
)


BASE_SCHEDULER = "vllm/v1/core/sched/scheduler.py"
RECOMPUTE_SCHEDULER = "vllm_ascend/core/recompute_scheduler.py"


def base_scheduler(source: str) -> str:
    """Adapt the standard scheduler; reject changed or already-patched anchors."""
    source = source.replace("\r\n", "\n")
    source = _original_base_scheduler(source)
    source = replace_once(
        source,
        '        draft = extra.get("ascend_external_draft")\n',
        '        if self.scheduler_config.async_scheduling:\n'
        '            raise ValueError("first-verify external draft does not support async scheduling")\n'
        '        draft = extra.get("ascend_external_draft")\n',
        "blue first-verify sync contract",
    )
    return _reject_native_streaming(_retain_remote_draft(source))


def _reject_native_streaming(source: str) -> str:
    marker = "    def add_request(self, request: Request) -> None:\n"
    guard = '''        if ((getattr(request.sampling_params, "extra_args", None) or {})
                .get("ascend_external_draft_first_verify", False) and (
                    request.resumable or getattr(
                        self.requests.get(request.request_id), "resumable", False))):
            raise ValueError("first-verify external draft does not support resumable streaming")
'''
    return replace_once(source, marker, marker + guard,
                        "blue native streaming first-verify boundary")


def _retain_remote_draft(source: str) -> str:
    # Remote KV admission has no local forward pass. Keep the seeded candidate
    # until the receive finishes and the waiting request can execute locally.
    return replace_once(
        source,
        '                if (request.spec_token_ids and\n'
        '                        (getattr(request.sampling_params, "extra_args", None) or {})\n'
        '                        .get("ascend_external_draft_first_verify", False) and\n'
        '                        num_computed_tokens + num_new_tokens < request.num_tokens_with_spec):\n',
        '                if (not load_kv_async and request.spec_token_ids and\n'
        '                        (getattr(request.sampling_params, "extra_args", None) or {})\n'
        '                        .get("ascend_external_draft_first_verify", False) and\n'
        '                        num_computed_tokens + num_new_tokens < request.num_tokens_with_spec):\n',
        "blue remote-KV first-draft retention",
    )


def make_scheduler_patch(vllm_root: Path, ascend_root: Path | None = None) -> str:
    """Generate the base patch, plus recompute only when explicitly supplied.

    Pass no ascend_root when the active engine uses only the base scheduler.
    A supplied ascend_root is strict: a missing recompute file is an error.
    """
    sources = [(Path(vllm_root) / BASE_SCHEDULER, BASE_SCHEDULER, base_scheduler)]
    if ascend_root is not None:
        sources.append((Path(ascend_root) / RECOMPUTE_SCHEDULER,
                        RECOMPUTE_SCHEDULER, recompute_scheduler))
    chunks = []
    for path, name, transform in sources:
        old = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        new = transform(old)
        chunks.extend(difflib.unified_diff(
            old.splitlines(keepends=True), new.splitlines(keepends=True),
            fromfile="a/" + name, tofile="b/" + name,
        ))
    return "".join("+\n" if line.startswith("+") and not line[1:].strip()
                   else line for line in chunks)


def recompute_scheduler(source: str) -> str:
    """Adapt recompute; the patched base supplies validation and async rejection."""
    source = _original_recompute_scheduler(source.replace("\r\n", "\n"))
    return _reject_native_streaming(_retain_remote_draft(source))
