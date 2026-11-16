# Optional batch-invariant target execution

Greedy decoding can change with batch shape. The strict ASR comparison keeps
exact token and text checks at C1 and C20; a failed gate prevents benchmarking.
`batch_invariant: true` adds a separately pinned execution configuration. It
requires `target_graph_mode: "piecewise"` while auxiliary MTP graphs can retain
`graph_mode: "full"`. Default configuration remains unchanged.

`batch_invariant_matmul_backend` explicitly selects `ascendc` (default) or
`triton` before operator registration. The Triton selection uses the existing
batch-invariant implementations for `aten.mm` and `aten.matmul`; AscendC
attention/reduction and the other stock registrations remain in place. It
adds no linear override. Invalid or unavailable selections fail before
registration. A kernel failure never changes the selected backend.

The default `Tensor.sum()` registration preserves PyTorch's dtype and all-axis
contract through the untouched dimensioned overload, including scalar and empty
tensors. The AscendC primitive requires an explicit dimension and cannot be
registered directly as that default overload. Single-dimension row reductions
retain the stock implementation.

The [official Ascend 0.23 guide](https://docs.vllm.ai/projects/ascend/en/v0.23.0/user_guide/feature_guide/batch_invariance.html)
describes target PIECEWISE execution. Its listed model coverage does not
establish ASR or custom MTP graph compatibility. The common FIA adapter maps
the extension's separate default, output-buffer, workspace and output-inference
operators to the interface used by MTP. It preserves supplied tensor aliases
and workspace; missing APIs fail before changing the operator binding.

The official [build script](https://github.com/vllm-project/vllm-ascend/blob/main/csrc/build_batch_invariant_ops.sh)
references these exact artifacts:

| Artifact | SHA256 |
| --- | --- |
| `cann-ops-batch_invariant-910b-2.0.0-linux.aarch64.run` | `8be2b339ccaaf550aa083d9d2a0c0b6e8cc1c66cbca8b3ef5487b76feb1bccec` |
| `batch_invariant-torch_ops_extension-2.0.0.zip` | `88c0a69d5a9487ccc54bc71ef4e3581756098f47d87188da52c299e792ab76c2` |

Keep these binaries and build outputs outside Git. Build the Python extension
against the registered interpreter, Torch and TorchNPU into a private prefix.
The upstream installation script changes global packages and is unsuitable for
this isolated workflow. The operator package reports CANN compiler 9.1.0.
The adapter requires the explicit 2.0 operator names; it rejects a 1.0-only
namespace before changing the binding. Matching version metadata still needs
actual operator and service validation.

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
The actual registered matrix backend must also match the frozen configuration.
Model output never selects code, dependencies, loader paths or permissions.

Correctness gates and benchmarks use fresh experiment IDs and their complete
configuration pins. Earlier failed attempts and original AR references remain
available for audit. CPU ABI tests alone establish neither device compatibility,
batch invariance nor a measured performance gain.
