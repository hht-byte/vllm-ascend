import asyncio
import os
from pathlib import Path

import pytest

from benchmarks.benchmark_310p import (
    assert_equivalent,
    load_manifest,
    run_equivalence_validation,
)


@pytest.mark.npu
def test_310p_cache_reuse_matches_full_recompute_after_lifecycle_events(
    tmp_path: Path,
) -> None:
    model = os.environ.get("QWEN3_ASR_310P_MODEL")
    manifest = os.environ.get("QWEN3_ASR_310P_MANIFEST")
    if not model or not manifest:
        pytest.skip(
            "set QWEN3_ASR_310P_MODEL and QWEN3_ASR_310P_MANIFEST "
            "to run target-machine equivalence"
        )

    pairs = asyncio.run(
        run_equivalence_validation(
            model=model,
            manifest_path=Path(manifest),
            window_seconds=(2, 4, 8),
            max_tokens=int(os.environ.get("QWEN3_ASR_310P_MAX_TOKENS", "128")),
            lru_pressure_requests=int(
                os.environ.get("QWEN3_ASR_310P_LRU_PRESSURE_REQUESTS", "32")
            ),
        )
    )
    assert pairs
    records = load_manifest(Path(manifest))
    full_checkpoint_scenarios = {
        "steady",
        "after_cache_reset",
        "after_lru_pressure",
        "after_session_recreate",
    }
    for window_seconds in (2, 4, 8):
        for record in records:
            observed = {
                (reuse.scenario, reuse.checkpoint_seconds)
                for _, reuse in pairs
                if reuse.window_seconds == window_seconds and reuse.record_id == record.id
            }
            assert {
                (scenario, checkpoint)
                for scenario in full_checkpoint_scenarios
                for checkpoint in record.checkpoints_seconds
            } <= observed
            assert ("exact_final_retry", record.checkpoints_seconds[-1]) in observed
    assert_equivalent(
        pairs,
        reproducer_path=tmp_path / "310p-equivalence-reproducer.json",
    )
