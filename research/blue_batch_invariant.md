# Optional batch-invariant target execution

Greedy decoding can change with batch shape. The strict ASR comparison keeps
exact token and text checks at C1 and C20; a failed gate prevents benchmarking.
`batch_invariant: true` adds a separately pinned execution configuration. It
requires `target_graph_mode: "piecewise"` while auxiliary MTP graphs can retain
`graph_mode: "full"`. Default configuration remains unchanged.

The [official Ascend 0.23 guide](https://docs.vllm.ai/projects/ascend/en/v0.23.0/user_guide/feature_guide/batch_invariance.html)
describes target PIECEWISE execution. Its listed model coverage does not
establish ASR or custom MTP graph compatibility. The common FIA adapter maps
the extension's separate default, output-buffer, workspace and output-inference
operators to the interface used by MTP. It preserves supplied tensor aliases
and workspace; missing APIs fail before changing the operator binding.

The official [build script](https://github.com/vllm-project/vllm-ascend/blob/v0.23.0/csrc/build_batch_invariant_ops.sh)
references these exact artifacts:

| Artifact | SHA256 |
| --- | --- |
| `cann-ops-batch_invariant-910b-1.0.0-linux.aarch64.run` | `6d6454da000164aee9e330582f3b7b6f7c2a38f668c2bd6d837d3b0aebf27f98` |
| `batch_invariant-torch_ops_extension-1.0.0.zip` | `96f28c91d209759d8369c5cb1bf161df17adfee41d401b114a4d3539058e8ac5` |

Keep these binaries and build outputs outside Git. Build the Python extension
against the registered interpreter, Torch and TorchNPU into a private prefix.
The upstream installation script changes global packages and is unsuitable for
this isolated workflow. The operator package reports CANN compiler 9.0.0;
another runtime version needs actual operator and service validation.

The read-only asset tree contains `runtime-deps/python/` and a flattened vendor
root at `runtime-deps/opp/`, including `opp/op_api/lib/`. Its `tree.json` is an
exact relative-file-to-SHA256 map. The common configuration pins
`batch_invariant_vendor_sha256`, the SHA256 of sorted compact JSON for that map,
without a trailing newline. The tree excludes its own manifest. All files must
match; additional files, symlinks, escapes and incomplete layouts are rejected
before environment or Python search paths change.

The trusted server enables the flag before engine imports. Each worker reports
actual operator registration, AscendC and Triton availability, and FIA bridge
state. All must be confirmed for this opt-in mode. This check prevents a partial
Triton-only initialization from being labelled as the full requested mode.
Model output never selects code, dependencies, loader paths or permissions.

Correctness gates and benchmarks use fresh experiment IDs and their complete
configuration pins. Earlier failed attempts and original AR references remain
available for audit. CPU ABI tests alone establish neither device compatibility,
batch invariance nor a measured performance gain.
