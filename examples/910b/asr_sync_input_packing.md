# ASR synchronous input metadata packing (910B)

This opt-in V1 runner optimization replaces four metadata H2D submissions and
redundant device indexing/cast/add/fill with one packed metadata transfer. It
reuses CPU positions and sequence lengths already calculated by the scheduler
input path. It does not change MTP heads, acceptance, slot mapping, embeddings,
audio encoder or decoder FULL graph keys.

## Enable / disable

Keep your existing MTP settings. Add a sibling key to `asr_mtp`:

```python
additional = dict(base_config.get("additional_config") or {})
additional["asr_sync_input_packing"] = True  # False or omit to disable
base_config["additional_config"] = additional
# Continue calling your existing mtp_engine_config(..., mtp_enabled=True, ...).
```

This is a startup option; restart the engine when switching it. It is deliberately
not inside `additional_config.asr_mtp` (the MTP loader rejects unknown keys).
The startup log must contain `ASR_SYNC_INPUT_PACKING enabled=True capacity_bytes=...`.

Supported: synchronous dense ASR MTP, TP/PP/DP/PCP/DCP=1, standard V1 preparation
method. Async scheduling/speculation, hybrid/GDN/sparse/compressed models and
subclasses overriding `_prepare_inputs` fail early when enabled. In particular,
the 310P runner has its own preparation method and is not supported by this option,
even though the delivery branch is named `310P-graph`.

## Install

Pull `310P-graph` to obtain the runner and new helper. For a checkout at the prior
`eb99b4de7` delivery, the focused patch can be applied instead:

```bash
git apply --check examples/910b/asr_sync_input_packing_v023.patch
git apply examples/910b/asr_sync_input_packing_v023.patch
```

Do not reapply the patch after pulling a commit that already contains the change.
The existing MTP patches remain unchanged; this patch modifies the runner and
adds `vllm_ascend/worker/sync_input_metadata.py`.

## Validation

```bash
python tests/ut/worker/test_sync_input_metadata_standalone.py
python examples/910b/benchmark_sync_input_metadata.py --requests 20 --query 6 --capacity 4096
```

The test file runs CPU equivalence tests without vLLM installed, and includes an
NPU-only fixed-address graph replay test with shrinking/reordered batches. The
benchmark compares only the replaced preparation region, in both timing orders;
it excludes scheduler, slot mapping, embedding and the actual model. CPU mode
(`--device cpu`) is a smoke test, not NPU performance evidence.

After NPU tests, run the same ASR load with the switch OFF and ON. Compare exact
output tokens, graph replay/fallback counts, queue wait, prefill/decode and commit
mean/P95. Re-profile the preparation gap: this change removes three H2D submissions
from this block, not all 21 copies in the observed interval. A separate ~2 ms gap
after MTP may remain in readback, postprocessing or scheduling.

## Buffer ownership and graph compatibility

Positions (int64), sequence lengths and computed counts (int32) are contiguous
views of one persistent allocation. Installed before model loading/capture, their
addresses never change. The H2D copy includes both request arrays and only active
positions: `8 * request_capacity + 8 * active_tokens` bytes. Sequence-length
padding is zeroed on CPU on every step. An event recorded immediately after the
copy protects reuse of pinned host storage; no whole-device synchronization is
added. If the previous copy is still in flight, host reuse waits for that copy.
This host-event cost is included in the benchmark and must be measured on 910B.

No speedup or full-service NPU correctness claim is made from CPU tests.
