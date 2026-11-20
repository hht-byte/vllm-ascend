# ASR engine delivery validation

Validation status as of 2026-11-20. This record covers the integrated engine
sources and their scheduler prerequisite. Chinese actual-engine correctness
acceptance completed successfully as detailed below.

## Scope and platform

All external-draft, first-verify/rejoin and optional MTP runtime code, worker
hooks and tests are retained. Business acceptance is Chinese-only and excludes
evaluation with real MTP weights. Tiny MTP fixtures exercise real device kernels
and graph replay; they do not measure trained-checkpoint business accuracy or
performance.

The device checks used Atlas A3, raw TorchNPU SoC `Ascend910_9362`, CANN family
`Ascend910_93`, with 20 Cube and 40 Vector cores. Installed CANN metadata and the
[primary project mapping](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/device/hardware.py#L22-L38)
identify the A3 family. The runtime was Python 3.12.13, Torch 2.10.0+cpu,
torch_npu 2.10.0.post4 and CANN 9.1.0. These are A3 results, with no 910B
performance claim.

Checks ran in an isolated temporary deployment without changing installed
runtime sources or restarting production services.

## Completed evidence

| Check | Observed result |
| --- | --- |
| Final formatted-tree local standalone contracts | 314 methods: 289 passed, 25 device-only skips; all 29 per-file processes exited successfully |
| Device standalone coverage | 317 distinct methods passed; zero skips after the two fixture corrections below |
| Patched upstream scheduler contracts | Nine source contracts passed against actual vLLM v0.23.0 source |
| Upstream patch lifecycle | Fresh apply/reverse checks passed; reversal restored the original scheduler bytes |
| Scoped static checks and review | Ruff check/format (27 Python files), forbidden-import, package-init, whitespace and five-file Markdownlint passed; independent reviews completed |

Ruff reformatted fourteen imported runtime files. Recorded pre/post-format ASTs
are identical for all 27 staged Python files, so payload parity means AST
equivalence rather than byte identity. Formatting does not change graph math or
behavior. Before/after hashes and AST results are retained in the task artifacts.
Long-function, boolean-context and logger checks also exited successfully.

Final local coverage comprises worker/preparation 62, external drafts 18,
recompute scheduler 15 and MTP examples 219. The earlier device coverage used 17
external-draft methods, added four attention query-start reuse methods and
enabled the device-only cases. Real-device checks
include changed-input graph replay, rejection/cold-prefill handoff, paged KV,
sparse row writes, joint QKV/RoPE, and mixed request/bucket behavior.

The initial device run recorded three failure events across two joint-Triton
test methods. Their standalone fixtures omitted
`init_device_properties_triton()`, which the real worker performs after device
selection. Only those two joint test methods were corrected. Their files then
passed all 31 methods through normal per-file commands, without an external
initialization prelude. At that fixture-correction stage, the other 286 methods
and their sources were unchanged.
Initial failure logs, intermediate bootstrap reproductions and final passing
logs are retained in the task artifacts.

Actual engine startup also exposed a custom-proposer factory integration gap:
upstream first constructs a transient config-only instance before the Ascend
runner installs its bound drafter. A new regression failed before the
constructor fix; the external-draft file then passed all 18 methods, including
its normal command in the final local batch. The earlier device count remains
317; this added regression is recorded as local coverage, without claiming an
unrun 318-method device suite.

The final per-file commands use the delivered runtime directories on
`PYTHONPATH`:

```bash
export PYTHONPATH="$PWD/vllm_ascend/spec_decode:$PWD/vllm_ascend/patch/worker:$PWD/examples/910b/tests${PYTHONPATH:+:$PYTHONPATH}"
python -m unittest discover -s examples/910b/tests -p test_asr_mtp_qkv_rope.py -v
python -m unittest discover -s examples/910b/tests -p test_asr_mtp_token_graph.py -v
```

The clean/reversed upstream scheduler SHA256 is
`9ae1491f57a6294e32416b93ef2ea68107fa0d65c3a4d7f745f1261012f735e9`.
Deployment manifests and before/after hashes retain the exact tested sources.
Full repository pytest was unavailable in the local environment
(`No module named pytest`); the standalone/source results above do not imply a
full repository pytest run.

The aggregate `format.sh ci` check could not complete because `pre-commit` was
unavailable. The forbidden-import check passed after its `regex` dependency was
provided in the task-local environment. The unavailable aggregate check is
separate from the completed scoped checks above.

## Chinese actual-engine correctness acceptance

The real Qwen3-ASR-1.7B model used revision
`7278e1e70fe206f11671096ffdd38061171dd6e5`; both downloaded safetensor shards
passed their published SHA256 checks. The Chinese input was the official
[Qwen3-ASR example audio](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav),
using the same deterministic four-second prefix in every mode. Its expanded
prompt had 70 tokens. This is a single-sample correctness check, not a business
quality or performance benchmark.

| Engine mode | Case groups | Completed requests | Result |
| --- | --- | --- | --- |
| Ordinary autoregressive, MTP disabled | 2 | 3 | Single and batch-two greedy outputs identical |
| External drafts, eager | 8 | 11 | Every final token identical to the baseline |
| External drafts, native FULL | 8 | 11 | Every final token identical to the baseline; actual graph replay confirmed |

All 25 outputs were `甚至出现交易几乎停滞的情况。`, with the same eight greedy
tokens including EOS:

```text
[100636, 100347, 99886, 100740, 118083, 102072, 1773, 151645]
```

Both external-draft modes exercised legacy one-shot all/zero acceptance,
first-prefill all/zero acceptance, an inserted-token mismatch followed by
`ngram_2` rejoining, variable draft lengths, staggered prefill/decode batching,
and unmarked ordinary requests. The observed legacy accepted lengths were six
and zero; first-verification accepted lengths were seven, zero and three.
The shifted candidate produced the correct ngram continuation, different from
the same-offset aligned continuation.

The FULL run used compilation mode `VLLM_COMPILE` (3), `cudagraph_mode=FULL`,
`enforce_eager=False`, synchronous scheduling and disabled prefix caching.
It captured nine target graph entries and executed 37 successful target graph
replays. Actual replay traces included:

| Batch | Query lengths | Computed lengths before the step | Unpadded → padded tokens |
| --- | --- | --- | --- |
| Variable decode queries | `[1, 6]` | `[73, 70]` | `7 → 8` |
| Mixed decode and fresh prefill | `[3, 70]` | `[70, 0]` | `73 → 128` |

The request limit was two, model length 1024, batch token budget 2048 and manual
target KV budget 1 GiB. The actual variable decode trace was `[1, 6]`, rather
than the nominal candidate lengths; the scheduler controls the executed query
lengths. These checks establish actual native FULL execution for the observed
mixed and variable-length batches without inferring speedup from correctness.

Worker traces used exact internal request IDs, including their generated
suffixes, to check verification and cleanup. Previously completed request IDs
were absent from resident request/draft maps in subsequent cases. All three
engines shut down normally. The diagnostic runs that exposed the constructor
gap and temporary harness RPC/ID-mapping issues remain archived alongside the
final passing runs.

In every business mode, `asr_mtp` was absent and actual worker probes reported
`mtp_proposer=False` and `mtp_online=False`. No trained MTP checkpoint was loaded
or evaluated. The complete MTP implementation and tests remain delivered.
