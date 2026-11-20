# Complete the ASR engine delivery

The user authorized completing the missing engine dependencies identified in
the 310P-graph audit. Streaming service code is excluded.

The branch will track the external-draft proposer/transport, first-verification
metadata and rejoin logic, all current MTP modules and their late worker hooks.
The existing native FULL/preparation path remains intact. Optional ASR behavior
stays disabled by default; ordinary requests and engines must retain their
existing dispatch. Importing the worker must not access a partially initialized
runner. A standalone upstream-vLLM scheduler patch supplies the part outside this
repository, with an explicit v0.23 prerequisite and installation instructions.

Existing validated MTP payloads are reused, rather than changing graph math or
adding another execution backend. Only storing another patch would leave the
audited gap in place, so runtime sources and registrations are integrated.
Replacing upstream vLLM or changing service parameters is outside this change.

Validation covers a clean checkout/import closure, default-off behavior,
transport validation/lifecycle, first-prefill draft scheduling and safe fallback,
rejoin plus MTP handoff, and AST-equivalent source parity with the validated graph
implementation after Ruff formatting. Recorded pre/post-format ASTs match for
all 27 staged Python files; formatting does not change graph math or behavior.
Run local CPU contracts and isolated native Ascend device checks without changing
production services. The verified device platform is Atlas A3: raw TorchNPU SoC
`Ascend910_9362`, CANN family `Ascend910_93`. These results do not establish 910B
performance.

Business acceptance is Chinese-only and excludes evaluation with real MTP
weights. All MTP runtime code, hooks and tests remain part of the delivery;
tiny real-NPU MTP graph checks do not substitute for trained-checkpoint business
accuracy or performance evaluation.

Completed evidence includes 314 local test methods (289 passed, 25 device-only
skips), 317 distinct device test methods passing without skips after two scoped
standalone fixture corrections, and nine patched upstream source contracts.
The added custom-proposer constructor regression is included in the final local
total and passed with the 18-method external-draft file; it is not counted as an
additional device test execution.
Preserve the initial fixture-failure logs and the passing normal-command reruns.
Chinese actual-engine correctness acceptance completed with identical greedy
tokens across 25 ordinary/eager-external/native-FULL requests. Actual FULL
replays covered variable decode queries and mixed prefill/decode batches.
`examples/910b/asr_engine_validation.md` records the observed execution and its
single-sample, non-performance scope.

Preserve all existing work. The user authorized committing and pushing after
validation and final review; the coordinating agent owns those final actions.
