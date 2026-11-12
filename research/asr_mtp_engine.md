# ASR 服务内的外部草稿 → MTP 接力（vLLM-Ascend 0.23）

本实现运行在 **vLLM worker 内**。加载用户提供的 `asr-mtp` checkpoint 中的辅助 heads，共享已加载目标模型的 embedding 和 logits 计算；不加载第二个 ASR 主干，不使用 HF `generate()`，不重新编码音频。回滚和 Prespec 的完整草稿仍先由已有 first-verify scheduler 补丁排入首次目标执行。随后，无论部分失配还是草稿耗尽，worker 从实际确认的目标上下文进入 MTP，后续草稿继续交给原有目标模型 / rejection sampler 验证。MTP 启用的请求不再进行旧 aligned/ngram 重接。

新增代码包含十三个可部署模块和 `worker.py` 中的启动注册：[补丁](asr_mtp_v023.patch)。不重写 `model_runner_v1.py`、目标模型、scheduler 或 paged KV 管理。安装必须在现有 [first-verify 补丁](external_draft_first_verify_v023.md)之上。

## 1. 安装到引擎

先在服务停止更新代码的窗口应用文件，再按原方式重启服务。这里不会自动连接服务器或重启进程。

```bash
# 本分支仓库根目录；根据实际位置修改路径
python research/build_asr_mtp_patch.py \
  --ascend-root /vllm-workspace/vllm-ascend \
  --out /tmp/asr-mtp.patch

cd /vllm-workspace/vllm-ascend
git apply --check /tmp/asr-mtp.patch
git apply /tmp/asr-mtp.patch
```

导出器只读引擎：核对 worker 的真实函数签名、CPU token 表更新与 post-bookkeeping 提案顺序，再生成 diff。重复安装同版本会导出空 diff。还可直接审阅本仓的 `research/asr_mtp_v023.patch` 后应用；不同 worker 版本优先重新导出并检查。文件对应关系：

| 本仓文件 | 安装位置 |
| --- | --- |
| `asr_mtp_heads.py` | `vllm_ascend/spec_decode/asr_mtp_heads.py` |
| `asr_mtp_fused.py` | `vllm_ascend/spec_decode/asr_mtp_fused.py` |
| `asr_mtp_qkv_rope.py` | `vllm_ascend/spec_decode/asr_mtp_qkv_rope.py` |
| `asr_mtp_row_copy.py` | `vllm_ascend/spec_decode/asr_mtp_row_copy.py` |
| `asr_mtp_row_copy_kernel.py` | `vllm_ascend/spec_decode/asr_mtp_row_copy_kernel.py` |
| `asr_mtp_graph.py` | `vllm_ascend/spec_decode/asr_mtp_graph.py` |
| `asr_mtp_token_graph.py` | `vllm_ascend/spec_decode/asr_mtp_token_graph.py` |
| `asr_mtp_token_attention.py` | `vllm_ascend/spec_decode/asr_mtp_token_attention.py` |
| `asr_mtp_paged_cache.py` | `vllm_ascend/spec_decode/asr_mtp_paged_cache.py` |
| `asr_mtp_paged_attention.py` | `vllm_ascend/spec_decode/asr_mtp_paged_attention.py` |
| `asr_mtp_worker.py` | `vllm_ascend/spec_decode/asr_mtp_worker.py` |
| `asr_mtp_metrics.py` | `vllm_ascend/spec_decode/asr_mtp_metrics.py` |
| `patch_external_draft_mtp.py` | `vllm_ascend/patch/worker/patch_external_draft_mtp.py` |
| 注册 import | `vllm_ascend/worker/worker.py`，紧接 `NPUModelRunner` 的导入；受 0.23 版本条件保护 |

runner 的四个 monkey patch 分别负责 drafter 选择、带 hidden states 的提案分派、finished/abort 状态清理，以及 FULL 图初始化时保留原始 token buckets。另在 `NPUWorker` 类定义结束后注册一次启动显存计量 hook，预留辅助图池；不修改运行时目标 forward。重复 import 幂等。未配置 MTP 时委托原方法。

### 已安装版本出现循环导入

若启动报 `cannot import name 'NPUModelRunner' from partially initialized module`，并且调用链经过 `patch/worker/__init__.py -> patch_external_draft_mtp.py`，原因是旧版把 MTP import 放在了会被 runner 自身提前导入的注册表。这个问题发生在加载引擎配置之前，单独加 `--no-mtp` 不能消除该导入错误。

更新导出器后，针对**服务器实际已安装目录**重新导出并按上方命令 `git apply --check` / `git apply`。升级 diff 会移除注册表中的旧 MTP import，并在 `worker.py` 原来的 runner 导入后添加：

```python
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner  # 原有行

# Install optional ASR MTP hooks only after the runner is fully imported.
if vllm_version_is("0.23.0"):
    import vllm_ascend.patch.worker.patch_external_draft_mtp  # noqa
```

`patch/worker/__init__.py` 中原有 external-draft / first-verify import 保留，**仅删除 MTP import**。手动修复时也必须同时改这两处。只把 `patch_external_draft_mtp.py` 内的类 import 移进 `install()`，但仍在导入时执行 `install()`，仍然会循环。

修复需要按原方式重启 worker；这里不会自动重启服务。新增回归使用实际 Python package 导入顺序复现旧异常，再检查新注册在 0.23 上安装、其他版本跳过；CPU 测试不等于 910B 服务已通过启动验收。

## 2. 引擎配置

服务启动开关的完整接线见 [服务 MTP 开关](streaming_mtp_service.md)：`--mtp / --no-mtp` 控制是否加载 MTP 及正式请求，`--prespec-mtp / --no-prespec-mtp` 控制后台请求。默认总开关关闭；仅配置引擎字典并不能代替服务参数传递。以下是开启时的底层配置。

在服务构造 `AsyncEngineArgs` 的 `base_config` 中合并以下配置。checkpoint 是已在服务器上的训练产物路径，不是 HF 仓库 URL；运行时不会下载权重，也不需要安装 `spec_asr`。

```python
base_config["speculative_config"] = {
    "method": "custom_class",
    "model": "vllm_ascend.spec_decode.external_draft_proposer.AscendExternalDraftProposer",
    "num_speculative_tokens": 15,  # 保持现有完整外部草稿预算
}
base_config["additional_config"] = {
    **base_config.get("additional_config", {}),
    "asr_mtp": {
        "checkpoint": "/home/work/spec-asr/runs/pt-scale-100h-mtp5/best.pt",
        "num_heads": 5,
        "cold_heads": 5,  # 每个引擎请求首次实际 MTP proposal
        "warm_heads": 3,  # 同一请求的后续 MTP proposal
        "max_context_tokens": 512,
        "cache_mib": 512,
        "max_requests": 64,
    },
}
base_config["async_scheduling"] = False
base_config["enable_prefix_caching"] = False
```

### FULL 图与 token bucket 归并

启用上述 MTP 扩展并使用 `compilation_config.cudagraph_mode="FULL"` 时，图初始化按用户配置的总 token buckets 建图。实现参考 `310P-graph` 的 [46fa2fb8f](https://github.com/hht-byte/vllm-ascend/commit/46fa2fb8f558845cb07748a7e440e148fcd691b1)：仅在 `_check_and_update_cudagraph_mode()` 内临时使用 query length 1，初始化结束或异常时恢复真实值。`num_speculative_tokens=15`、运行时 query length 16、实际逐请求草稿长度均保留；不修改 scheduler、采样或目标 attention 实现。

原生 Ascend FULL dispatcher 使用 mixed descriptor，并动态更新各请求的 attention metadata；不同长度的草稿仍可同批验证。此前上游按 `K+1=16` 向上取整并去重，会把用户这组 31 个 capture sizes 归并为 17 个，导致 `6→16`、`2→16` 等 padding。保留小 bucket 后，这组配置的 6-token batch 可匹配 bucket 6，2-token batch 可匹配 bucket 2。FULL 的对齐规则与每条草稿实际长度是两个不同的概念。

此初始化 hook 仅对本扩展的 MTP drafter、精确的 `FULL` 模式及全部 attention 后端均报告 `ALWAYS` 图支持时生效。后端能力不足时，requested FULL 可能降级为 decode-only；这类情况继续使用真实的 `K+1` 进行原始安全检查和 bucket 归并。关闭 MTP 或使用 `FULL_DECODE_ONLY`、`FULL_AND_PIECEWISE`、`PIECEWISE`、`NONE` 时也沿用原始配置解析。导出器会检查图初始化 API，防止把 hook 安装到不支持的 runner 上。整个修改不在每次 decode 中执行，也不增加设备同步。

用户提供的 12:37:25 聚合表中，30 次目标执行合计 1152 个有效 token、1440 个 padded token，padding 占执行 token 的 20%；所列各行均为 FULL。这支持减少 padding 的方向，不能换算为同比例时延收益，也不能用该聚合表逐条判定请求的 prefill 阶段。更多 bucket 会增加启动捕获数量和图内存，部署后须对照实际 bucket、峰值内存及完整服务时延。

这只优化**目标 decoder 图的粒度**。辅助 MTP 默认仍为 eager；下面的独立开关才会启用辅助图。音频 encoder 和目标模型自身的 LM head 不在本次辅助图范围内。

### 独立 MTP 整图执行

本次支持 910B 的 `NPUWorker` 显存定容路径。`310P-graph` 是交付分支名；310P 的 `NPUWorker310` 覆盖了显存计量方法，当前会显式拒绝其辅助 `graph_mode=full`，避免漏计图池。其原有 MTP eager 路径不受影响。

沿用现有 `mtp_engine_config(base_config, *, mtp_enabled=False, mtp_checkpoint=None)` 接口。前文已初始化 `base_config["additional_config"]["asr_mtp"]`；在调用该函数之前追加图配置：

```python
base_config["additional_config"]["asr_mtp"].update({
    "graph_mode": "full",          # 默认 off；off 恢复原 MTP eager 实现
    "graph_max_batch_size": 8,      # 精确 batch 1..8，避免 padding LM head
    "graph_query_sizes": [8, 64],   # 分开处理增量 fresh query 与首次 hidden 历史
    "graph_context_tokens": 128,   # KV/hidden 容量，不是当前真实序列长度
})
base_config = mtp_engine_config(
    base_config,
    mtp_enabled=mtp_enabled,
    mtp_checkpoint=mtp_checkpoint,
)
```

现有函数会保留这些字典字段，CLI checkpoint 仍只覆盖路径，MTP 总开关 OFF 仍移除整个 `asr_mtp`。无需修改函数签名，也无需给服务工厂增加 `mtp_graph_mode`。

完整辅助链在一张图中执行：逐 head 的 shifted embedding、融合、attention/MLP、共享目标 LM head、argmax，以及辅助 KV/hidden 回写。目标 verifier、目标 KV、音频 encoder 和 scheduler 保持原流程。不同请求的真实长度、每层有效前缀、drive token 和草稿预算都是设备输入，不会按 request ID 或精确序列长度建图。

只在启动 `profile_run` 中预捕获，最多 64 张，运行时只 replay。默认配置为 8 种精确 batch × 2 种 query 容量，共 16 张（受实际辅助槽数和 `max_num_seqs` 限制）。同 query 图档位的请求合并执行，首次长 query 与增量 query 分开。batch 超过上限、context 超过 128、fresh query 超过 64，或者所有请求的剩余 head 预算不足完整深度时，直接走原 eager。不会在线建图或让请求等待其他 batch。`enforce_eager=True` 与显式 `graph_mode=full` 冲突，会在启动报错。

辅助 arena 追加有限 scratch 区，padding 写入独立地址，不会覆盖 live cache 的最后一行。辅助 token 仍在 worker 末尾统一执行一次 batch D2H。共享图池只容纳临时中间张量，输入、输出、arena 和共享权重均在池外；replay 与输出消费保持同一当前流上的顺序。

图路径的缓存搬运使用 `gcd(hidden_size, head_dim)` 宽度的对齐行，保持原 arena 布局与图 key。读取使用二维行 `index_select`，NPU 写入显式调用 `torch_npu.npu_scatter_nd_update_`；CPU 核对路径使用 `index_copy_`，NPU 不回退到可能逐索引同步的兼容实现。缺少设备更新接口会在图初始化时明确报错。以 hidden 2048、head_dim 128 为例，每行 128 个元素，地址索引数量比逐标量寻址减少 128 倍；这不表示数据搬运量或时延同比下降。

全局最后一个辅助 head 仍保留 K/V 和有效长度，但省去没有后续消费者的 hidden 写回和局部 sequence 更新。五头图程序由 15 次标量缓存 scatter 加 5 次 token scatter，变为 14 次行缓存更新加 4 次 token scatter；15 次缓存读取保持次数不变、改为行索引。尚未融合 K/V 更新与 attention 输入整理，eager 回退路径保持原实现。部署后应检查 `aclnnScatterNdUpdate` 和行级 Gather 的实际耗时，并按同负载比较输出、接受率与 commit P95/P99；本机 CPU 等价测试不能证明 910B 性能提升。

已安装 context 诊断版辅助图的服务，可从引擎 `310P-graph` 交付目录应用聚焦升级：

```bash
git apply --check examples/910b/asr_mtp_cache_rows_v023.patch
git apply examples/910b/asr_mtp_cache_rows_v023.patch
```

该补丁只升级 MTP 图模块，不增加启动参数。完整新安装补丁和累计图升级补丁已经包含此改动，不要再叠加聚焦升级；旧版模块不同则对实际安装目录重新导出。按原流程重启 worker 后生效。

显存：启动 memory hook 先捕获 MTP 图，再进入原始目标模型 profiling 上下文。
这样 profiling 的基线已包含常驻图张量，activation peak 增量测量图常驻时的运行激活；
结束后再从自动 KV 预算单独扣除一次完整图池 reserved 增量（含静态缓冲和私有池空闲段）。
旧版在 profiling 内捕获图，静态张量既进入峰值增量，又被整笔 reserved 预留重复扣减，
在 3/5 两套图时可能触发虚假的预算不足。仅去掉内存检查、或减掉 live allocated 后继续使用
旧顺序都不可靠，因为目标模型先前的激活峰值未必与图张量同时存在。
`cache_mib` 不包含该图池、scratch 和辅助权重。显式指定 `kv_cache_memory_bytes` 时保留
用户手工预算，需自行为图池留出空间。捕获失败直接报启动错误；运行时图已执行部分操作后
不会冒险重试 eager。

启动和首次实际命中应看到：

```text
ASR_MTP_CONFIG ... graph_mode=full graph_scratch_mib=...
ASR_MTP_GRAPH_READY graphs=16 ... reserved_mib=...
ASR_MTP_GRAPH_MEMORY reserved_mib=... allocated_mib=... capture_before_profile=True base_available_mib=... adjusted_available_mib=... manual_kv=False
ASR_MTP_GRAPH_REPLAY batch=... query=... context=... heads=...
```

`ASR_MTP_STEP` DEBUG 日志另给出累计 graph/eager batch 数和 `miss_batch`、`miss_query`、`miss_context`、`miss_budget`。辅助图不会加入目标 decoder 的 `CUDAGraph Stats` 统计；应结合这些日志和 profiling 的辅助 LM head / attention static 事件确认。图能减少下发开销，不会消除 LM head 的矩阵计算。

每次由 context 超限回退时，新增一条 INFO 日志，无需打开 DEBUG：

```text
ASR_MTP_CONTEXT_MISS batch_size=2 max_context=180 context_limit=128 max_fresh_query=79 query_limit=128 miss_count=57
```

`max_context` 是本批最大 hidden 历史长度，`max_fresh_query` 是本批有效 head 的最大新增补算长度，两者可能来自不同请求；`batch_size` 是该辅助批次的请求数。`context_limit` / `query_limit` 给出当前图上限，`miss_count` 与同一 worker 的累计 `miss_context` 对应。历史长度包含 prompt；新增 query 不是输出草稿数量，也不是 `new_head_positions` 的跨 head 总和。

该日志只在原有 context miss 分支记录，长度与 head 有效前缀均为已有 CPU 整数，不读取设备张量、不新增设备同步，也不改变图选择、回退优先级或缓存。INFO 被禁用时不额外计算诊断用的最大 query。

```bash
rg -n 'ASR_MTP_GRAPH_READY|ASR_MTP_CONTEXT_MISS' "$LOG"
```

已部署辅助图的服务可更新单个 `vllm_ascend/spec_decode/asr_mtp_graph.py`，或针对实际安装目录使用前述导出器生成升级差异。按原方式重启 worker 后生效，不需要修改服务参数。

若使用 `--mtp-graph-mode full/off`，在入口先将其值写入上述 `graph_mode` 字段，再走原来的工厂和 `mtp_engine_config()` 调用；不向这些调用新增图参数。入口映射示例见 [服务启动接线](streaming_mtp_service.md)。

CPU 检查覆盖数学路径和生命周期，不能证明 NPU 捕获和时延。设备上的集成测试会实际调用 `torch.npu.NPUGraph`，可在已安装 Torch-NPU 的环境执行：

```bash
python -m unittest discover -s research -p 'test_asr_mtp_graph.py' -v
```

NPU 用例不得显示 skipped；之后按相同 checkpoint、head 数、Prespec、并发与负载对照 `graph_mode=off/full`，比较完整目标输出、接受率、commit P95/P99、prefill/decode 和峰值显存。这里没有本地 NPU 性能验收结果。

### Packed token 图：合并不同补算长度

新增 `graph_token_sizes` 后，辅助图按总 token 容量捕获；不设置该字段时继续使用原来的 `(batch, query)` 图。两种配置同时存在时，token 配置优先。仍使用现有三参数服务工厂，合并配置不会覆盖 checkpoint 等原有字段：

```python
additional = dict(base_config.get("additional_config") or {})
mtp_settings = dict(additional.get("asr_mtp") or {})
mtp_settings.update({
    "graph_mode": "full",
    "graph_token_sizes": [8, 32, 64, 128, 256, 512],
    "graph_max_batch_size": 8,
    "graph_context_tokens": 128,
})
additional["asr_mtp"] = mtp_settings
base_config["additional_config"] = additional
base_config = mtp_engine_config(
    base_config,
    mtp_enabled=mtp_enabled,
    mtp_checkpoint=mtp_checkpoint,
)
```

令请求 i 在辅助头 d 上的补算量为 `q[i,d] = history_length[i] - head_valid[i,d]`，超出该请求 head 预算的项取 0。图容量取能容纳 `max_d sum_i q[i,d]` 的最小 token 档位，五个头按顺序复用这一静态容量，不把五头 token 数相加。每个 head 各自紧凑打包为 `[T,H]`，通过请求边界、slot 和 anchor 元数据保持缓存和 attention 隔离。例如补算量 6、80、45 的三条请求可合并为一个 256-token 图。

同一目标 step 中执行深度相同的请求不再按单请求 Q 分桶；仅在总 token 超过最大档位或请求数超过固定槽上限时拆批。cold/warm 深度不同时另按深度分组。历史超过 context 容量、单请求补算超过最大 token 桶以及原有采样条件继续使用 eager 回退；预算不足时选择能容纳实际深度的最小图。`graph_context_tokens=128` 表示每请求历史容量；总 token 桶可以为 512，不要求两者相等，也不跟随主模型本轮 FULL 图的大小。

投影、MLP、RoPE 和行缓存操作使用 packed token；attention 使用单次 FIA TND 调用处理所有活动请求，并在 replay 前更新各头的 CPU 累积 query/KV 长度。保留固定请求槽的 LM head，只从各请求最后一个补算位置抽取 hidden，不对全部 T 行做词表投影。未使用请求槽仍有 LM 矩阵计算，需在同负载对比中计入。

所选 910B FIA 入口要求 FP16/BF16 和受支持的 head dimension（当前校验 128/192）；NPU 图启动需要 FIA out、最大 workspace 和 graph task update API。workspace 在启动分配，replay 不分配 workspace、不读取设备 token 或长度到 CPU。矩形图与 MTP eager 路径不受此 FIA 几何限制。每个 head 的累计 Q/KV 长度或输入指针改变时才进行主机 task update；相同绑定复用，但每轮仍发送所有 head 的外部事件通知。固定请求槽的开销需要与减少 replay/补齐计算的收益一起评估。

非 PA TND 在已部署 CANN 上要求最后一个累积 KV 长度严格等于 K/V 的物理 T。每个逻辑 token 桶额外预留 1 个物理 query：512 桶仍可容纳 512 个真实补算位置，物理 query 容量为 513。始终存在的正长度 dummy query 段承接所有剩余零填充 KV；dummy KV 长度不小于 dummy query，避免全 masked 行。这样不依赖零 query 分段，不在 task update 中改变 Tensor shape，也不会让恰好填满桶的请求跳到下一档。启动日志显示 `token_sizes` 与 `query_capacities`；首次 replay 还显示 `bucket`、`query_capacity`、`kv_capacity`。

已安装最初 token 图版、遇到 `keyT/valueT must be equal to ... actualSeqenceLengthKV` 的服务，只应用聚焦修复：

```bash
git apply --check examples/910b/asr_mtp_tnd_kv_coverage_v023.patch
git apply examples/910b/asr_mtp_tnd_kv_coverage_v023.patch
```

随后按原流程重启 worker，现有 token 配置无需更改。完整新安装、累计图升级和行缓存版到 token 版的补丁均已包含修复，不要连续叠加；其他安装版本应针对实际源码重新导出。

已安装行缓存优化版的服务可应用 `examples/910b/asr_mtp_token_graph_v023.patch` 聚焦升级；完整安装和累计图补丁已包含此升级，不要连续叠加。当前实现仍需真实 910B 捕获、变长 replay 和时延验收，CPU/FIA 接口模拟测试不能证明设备性能收益。

### 原生辅助头融合与图外接力

新版默认 `native_fusion=True`，仅在 MTP 辅助头内启用原生 RMSNorm、残差 Add+RMSNorm 和 SwiGLU；Q/K/V 以及 gate/up 的权重在启动时各打包一次。原始 Parameter 保留 checkpoint 名称，并共享非持久 packed buffer 的存储，避免常驻两套权重。顺序为严格加载原 checkpoint → 最终设备与 dtype 放置 → 冻结 → 打包 → profile/捕图；运行时不拼权重。eager、矩形图和 token 图共用这些实现，目标模型不变。需要原生 API 而设备软件栈不提供时会在启动报错，可用 `native_fusion=False` 恢复原辅助数学路径。

沿用现有服务工厂，在调用前合并设置，不要求 `base_config` 预先含有 `asr_mtp`：

```python
additional = dict(base_config.get("additional_config") or {})
mtp_settings = dict(additional.get("asr_mtp") or {})
mtp_settings.update({
    "native_fusion": True,   # 默认 True；False 用于同图配置下消融融合
    "profile_stages": False, # 默认 False；定位主机空泡时短期开 True
})
additional["asr_mtp"] = mtp_settings
base_config["additional_config"] = additional
base_config = mtp_engine_config(
    base_config,
    mtp_enabled=mtp_enabled,
    mtp_checkpoint=mtp_checkpoint,
)
```

这些都是启动配置，工厂签名与 CLI checkpoint 覆盖规则不变。只切换 `native_fusion` 时应保持 head 数、token 桶、槽数和负载相同。五头的线性矩阵调用从 45 次减到 30 次；这是算子调用数核对，尚不是 910B 时延收益。五个依赖前一预测的 LM head 和固定请求槽计算仍在。

图外接力同时省去两类可避免工作：所有请求属于一个完整提案组时，直接对图输出执行一次 `tolist()`，省去中间零张量和高级索引汇总；多组仍立即复制持久图输出，再统一转到 CPU，防止下一次 replay 覆写。FIA 的累计 Q/KV 长度、Q/K/V 指针和 scale 均未变时复用原 task 绑定，但每轮仍记录所有外部事件。变参仍先校验全部 head，再重绑变化项。不会新增等待凑批、目标前向或每 head 的设备同步。

已安装严格 TND KV 覆盖修复版（源版本 `46d2537e1`）时，只应用本次聚焦升级：

```bash
git apply --check examples/910b/asr_mtp_native_fusion_v023.patch
git apply examples/910b/asr_mtp_native_fusion_v023.patch
```

新安装用 `research/asr_mtp_v023.patch`；旧 pre-graph MTP 版用累计 `asr_mtp_graph_v023.patch`；行缓存版用 `asr_mtp_token_graph_v023.patch`。后三者均已包含本次改动，不与聚焦升级叠加。版本不同则针对实际源码重新导出，检查通过后按正常部署流程重启 worker。

`profile_stages=True` 时额外输出 INFO 主机阶段日志；关闭时不读时钟、不创建诊断事件、不增加同步。eager/矩形图/打包图都有 worker 级 `ASR_MTP_HOST_STAGE`，打包图另有 `ASR_MTP_GRAPH_HOST_STAGE`，细分 stage、FIA task update、replay 入队、commit，并显示 `task_updates` / `task_update_reuses`。筛选命令：

```bash
rg -n 'ASR_MTP_CONFIG|ASR_MTP_HOST_STAGE|ASR_MTP_GRAPH_HOST_STAGE' "$LOG"
```

这些是主机墙钟阶段，`to_cpu_wait_ms` 包含此前排队 NPU 工作的等待，不能视为纯传输时间，也不能与设备执行时长重复相加。细分图日志属于 worker 提案阶段的内部时间；诊断日志本身也会增加开销，正式性能对比应关闭。下一轮目标输入准备、IPC 或调度空档不在这些 MTP 内部阶段内，不能据此声称已消除所有空泡。

设备测试已包含原生融合与矩形图、packed FIA 同图 replay，覆盖变输入、请求重排、桶切换和拒绝后的缓存裁剪。本机无 NPU，相关用例只能跳过；需在 910B 上不跳过地执行，再核对完整输出、接受率、commit P95/P99 和图外时间线。

### 分页辅助 KV：合并写入与直接读取

新增可选 `kv_cache_backend="paged"`，要求已启用 `graph_mode="full"` 和
`graph_token_sizes`。默认 `"tnd"` 保留原布局用于对照。沿用现有工厂接口，
在调用 `mtp_engine_config()` 之前合并：

```python
additional = dict(base_config.get("additional_config") or {})
settings = dict(additional.get("asr_mtp") or {})
settings.update({
    "graph_mode": "full",
    "graph_token_sizes": [8, 32, 64, 128, 256, 512],
    "kv_cache_backend": "paged",   # tnd 恢复原连续 KV 整理路径
    "kv_block_size": 128,           # 允许 128、256、512
    "kv_write_backend": "auto",    # auto / scatter_pa / scatter_nd
})
additional["asr_mtp"] = settings
base_config["additional_config"] = additional
base_config = mtp_engine_config(
    base_config, mtp_enabled=mtp_enabled, mtp_checkpoint=mtp_checkpoint,
)
```

每个辅助请求槽固定占用若干独立 KV blocks，不使用目标模型块分配器。
持久 hidden 与 KV 分开布局；eager 回退通过共享存储视图读写同一份分页 KV，
不会在图与 eager 之间复制整段历史，也不保留第二份有效 KV。失配仍只裁剪
各 head 的 `valid`，随后覆盖尾部；释放槽时重置有效长度，不清零整块显存。

每个 head 将新增 K/V 合并写入，FIA 通过 `block_table` 直接读取持久页面。
五头图的历史 K/V 显式 gather 从 10 次降为 0，K/V 写入从 10 次调用降为 5 次。
hidden 的 5 次读取、4 次写入，以及 token 更新、最后 hidden 抽取仍然存在；
这些计数是程序级调用，不能等同于设备 kernel 数或同比时延改善。
eager 回退仍采用原有连续 attention 整理；上述读写消融针对分页图执行。

`auto` 在启动检查 `npu_scatter_pa_kv_cache` 是否支持 ND 的
`cache_mode="Norm"`；支持则采用专用 K/V 联合写入。不支持时使用一次合并
K/V 的 `npu_scatter_nd_update_`，仍直接用 PA 读取缓存。显式 `scatter_pa`
要求该接口存在，不能使用旧版 NZ-only 算子解释 ND 缓存；显式 `scatter_nd`
可用于同一分页布局下消融写入方式。旧 [7.2.0 算子 schema](https://gitcode.com/Ascend/op-plugin/blob/7.2.0/op_plugin/config/op_plugin_functions.yaml)
没有 `cache_mode`，因此不能仅凭算子名称判断支持。

FIA 的 Q 仍为 TND 累计边界，PA 的 KV 长度传每请求长度，而非历史长度累加。
各 head 的 block table 根据当前请求顺序与预算紧凑排列，一次上传所有 head，
表未变化时省去重复上传；长度变化继续走图 task update。块表只使用已有 CPU
slot 元数据，不读取设备 token/长度。padding 使用独立 scratch 页和唯一地址，
不覆盖 live cache；正长度 dummy 段保留，PA 不要求 KV 总物理容量等于序列长度和。
接口依据见 [Ascend FIA 文档](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_fused_infer_attention_score.md)。

`cache_mib` 包含按 block size 向上取整的 live KV、持久 hidden 和 dummy KV 页，
不含权重、图工作区和图元数据。同预算下可用槽数可能减少，须核对启动的
`ASR_MTP_CONFIG kv_cache_backend=... kv_write_backend=... cache_slots=...`。
图仍只按 token 桶捕获，块表不按 request ID 建图，不等待凑批。

已安装原生融合版（源码 `a5e591b9c`）时应用引擎交付目录中的聚焦升级：

```bash
git apply --check examples/910b/asr_mtp_paged_kv_v023.patch
git apply examples/910b/asr_mtp_paged_kv_v023.patch
```

新安装、累计图升级、行缓存到 token 升级补丁已包含本次改动，不叠加聚焦补丁。
其他源码版本请对实际目录重新导出。部署后按原流程重启 worker；本次实现
没有连接或重启服务。设备测试覆盖分页 writer/FIA 同图、变 batch/边界、混合预算
与失配；本机无 NPU，尚未验证 910B 时延和数值。

```bash
rg -n 'ASR_MTP_CONFIG|ASR_MTP_GRAPH_READY|ASR_MTP_HOST_STAGE' "$LOG"
PYTHONPATH=research python -m unittest test_asr_mtp_paged -v
PYTHONPATH=research python -m unittest test_asr_mtp_token_graph.ActualPackedNPUReplayTests -v
```

保持 checkpoint、head 数、图档位、并发和输入相同，比较 `tnd/paged` 的
完整输出、接受率、图外时间、prefill/decode、commit P95/P99 及显存。
显式历史搬运减少并不保证 PA 算子在短上下文下更快。

### QKV/RoPE 融合与稀疏 hidden 行写入

这次优化只调整辅助头内部算子和 flat arena 的行写入。eager、矩形图、
packed token 图共用 QKV/RoPE helper；不改变目标模型、scheduler、head 数、
候选预算或已有分页 KV 的 `scatter_pa` writer。仍在三参数工厂调用之前合并：

```python
additional = dict(base_config.get("additional_config") or {})
settings = dict(additional.get("asr_mtp") or {})
settings.update({
    "qkv_rope_backend": "auto",   # auto / torch / native / triton
    "row_write_backend": "scatter_nd",  # 本轮建议；auto 同样选择 ScatterND
})
additional["asr_mtp"] = settings
base_config["additional_config"] = additional
base_config = mtp_engine_config(
    base_config, mtp_enabled=mtp_enabled, mtp_checkpoint=mtp_checkpoint,
)
```

| `qkv_rope_backend` | 行为与比较边界 |
| --- | --- |
| `auto`（默认） | 有 `npu_rotary_mul` 时选择 native RoPE；否则选 Torch。保留分开的 Q/K RMSNorm，不自动选择联合 Triton kernel。 |
| `torch` | 保留 Q/K norm 和原 multiply/add RoPE 的低精度舍入顺序，用于对照。 |
| `native` | 在 Q/K norm 后用 `npu_rotary_mul(rotary_mode="half")`，接口缺失则启动失败。 |
| `triton` | 显式选择已注册的 `triton_split_qkv_rmsnorm_mrope`，联合 split QKV、Q/K RMSNorm 和 RoPE；接口或形状不兼容则启动失败。 |

联合 kernel 使用 FP32 完成 norm/rotation 后再 cast，可能与原分阶段低精度
舍入不同，必须显式开启并核对输出和接受率。不能用旧的硬编码 BF16
split-QKV kernel 处理当前 FP16 服务。projection bias 在 packed Linear 中只加
一次，联合接口的 norm affine bias 保持 `None`，避免重复偏置。

位置表在最终设备和 dtype 上启动时预算、预计算，不在 replay 中做 sin/cos、
设备标量读取或 CPU 长度同步。表容量覆盖 `context_cap + head 数 + 1`。
padding anchor 先按 arena 的 `context_cap` 裁剪，再加 head depth；不能用
token bucket 容量作为历史上限，因 1024-token 桶可大于 512-token cache cap。
`native_fusion=False` 时 `auto/torch` 保留旧数学路径且不安装 helper；显式
`native/triton` 则拒绝启动，避免消融配置实际仍启用融合。

`row_write_backend` 允许 `auto/triton/scatter_nd`，独立控制 flat arena
`_write`，分页模式下主要是 hidden 写入。当前 `auto` 在 NPU 直接选择
`npu_scatter_nd_update_`，不导入 Triton；CPU 继续用 `index_copy_`。
显式 `triton` 保留为实验后端，缺失则启动失败，运行时/JIT 错误直接报出，
不吞掉错误后换算子。用户在 910B 观测到 `row_copy_kernel` 单次约 1.377 ms，
比此前原生 writer 更慢，因此本轮先用 `scatter_nd`，保持 native helper
以及其余图配置、head 数和预算不变。两次补算量仅接近，不能据此给出严格
同 shape 的稳定倍率或把原因确定为低效编译 IR。

实验 copy kernel 只读取更新行的索引和值，并写入相应 destination 行，不读取、
扫描、克隆或清零整个 destination。live 和 padding scratch 地址唯一性由调用
方保证。用户这轮 `(1097744,128)` 的 FP16 destination 对应 hidden arena：

```text
22 slots × 512 context × 6 hidden stages × 2048 / 128
  + (1024 + 1) scratch positions × 2048 / 128 = 1,097,744 rows
2320 update rows / (2048 / 128) = 145 positions
```

约 268 MiB destination 只更新 580 KiB；145 个位置对应 144-token 桶加一个
dummy。这些形状能定位 hidden 写入，不能证明旧 ScatterND 内部全量扫描或
复制了 destination。新版分页 token 图把 hidden 的行宽由 128 改为完整
`hidden_size=2048`：同一存储 view 从 `(1097744,128)` 变为
`(68609,2048)`，更新从 `(2320,128)` 变为 `(145,2048)`。hidden gather
也按一个 token 一行读取，减少索引行数，字节量、padding 隔离和缓存有效性
不变。混合 hidden/KV 的 TND 与矩形图仍使用 gcd 行宽；分页 K/V 已走专用
`scatter_pa`，不会被该开关替换。[官方 VectorCore 优化指南](https://github.com/Ascend/triton-ascend/blob/main/docs/en/programming_guide/vector_operator.md)可指导后续实验，
但不能据此证明当前 kernel 的瓶颈或保证新的时延收益。

已安装上一轮 native-ops 版（ASR 源码 `637b6f878`、引擎交付
`5d757257c`）应用新的聚焦修复：

```bash
git apply --check examples/910b/asr_mtp_native_rows_v023.patch
git apply examples/910b/asr_mtp_native_rows_v023.patch
```

更早的分页版（源码 `f9bf28b42`）则应用已累计包含该修复的 native-ops 升级：

```bash
git apply --check examples/910b/asr_mtp_native_ops_v023.patch
git apply examples/910b/asr_mtp_native_ops_v023.patch
```

上述两条聚焦升级按已安装基线二选一，不叠加。新安装的完整补丁、累计图
升级和行缓存到 token 升级也已包含修复，不与聚焦升级连续叠加。
其他版本对实际安装目录重新导出；新增三个 helper 模块也必须一起部署。
`ASR_MTP_CONFIG` 中 `qkv_rope_backend` 是实际选择，`row_write_backend`
是请求配置；`ASR_MTP_GRAPH_READY row_write_backend` 是图实际 writer，
分页 token 图的 `row_width=2048` 可核对整 token 行布局：

```bash
rg -n 'ASR_MTP_CONFIG|ASR_MTP_GRAPH_READY' "$LOG"
PYTHONPATH=research python -m unittest test_asr_mtp_qkv_rope test_asr_mtp_row_copy -v
PYTHONPATH=research python -m unittest test_asr_mtp_token_graph.ActualPackedNPUReplayTests -v
```

helper 数学/分派测试和整链 NPU 条件测试覆盖融合、行写入与分页 replay。
本机没有 NPU，CPU 通过不能代替 910B 数值、捕图和性能验收。对比保持负载、
checkpoint、head 数、桶和预算一致，分别切换 RoPE 和 row writer，核对完整
输出、接受率、prefill/decode、commit P95/P99 和图外时间线。

### 支持范围与缓存

初版支持 TP=PP=PCP=DCP=1、无量化 / LoRA 的 Qwen3-ASR、同步调度和贪心 `n=1`。不支持的引擎配置在权重加载时拒绝，不能自动退成另一种投机算法。禁用 prefix caching 是因为仅有命中的目标 KV 无法恢复 MTP 所需的完整目标 hidden 历史；不重新 prefill 来弥补。chunked prefill 可以保留每次已计算的 hidden 行。其他模型或 RoPE scaling 不在此版本范围。

checkpoint 按 supplied `asr-mtp` 的 `{"metadata": ..., "mtp": ...}` 格式读取，只装载所用辅助 heads，严格校验辅助头 state_dict。**默认关闭主模型的 `base_signature` 指纹校验**；Qwen3-ASR 架构与辅助头参数形状检查保留。MTP 训练冻结主模型，运行时共享服务已加载的目标模型，所有辅助草稿仍交由目标 verifier 验证。

缓存 arena 在 `load_model` 中按字节预算预分配，纳入引擎目标 KV 定容之前的显存统计；profile 阶段执行最坏辅助 batch 来统计临时内存。`cache_mib` **不包含辅助权重和临时 attention 工作区**，实际总增量显存大于该数值。实际槽数是 `min(max_requests, floor((cache_mib - 固定 scratch 页字节数) / 每槽字节数))`；原 TND 布局的固定 scratch 页预算为 0。启动日志 `ASR_MTP_CONFIG` 给出实际值。

用户提供的是葡语训练 checkpoint；其他语言的接受率和净收益需要单独测量，不能从葡语结果外推。

移除 `additional_config["asr_mtp"]` 即恢复原有外部草稿 proposer。不要改成原生 `method="mtp"`：它不会识别这份定制 checkpoint。

### 可选主模型指纹校验

训练仓库的旧签名包含 config 原始字节、safetensors 文件名、大小、mtime，**不包含权重内容哈希或目录路径**；复制时仅 mtime 变化也可能导致相同权重的签名不同。默认 `verify_base_signature=False` 完全跳过该检查，不读取指纹文件。需要严格复现训练环境时，在现有引擎配置中设置：

```python
base_config["additional_config"]["asr_mtp"]["verify_base_signature"] = True
```

省略该字段或显式设置 `False` 都表示跳过；字符串 `"false"`、数字等非法值会在启动时报错。此配置通过现有 `additional_config` 传递，不是 `AsyncLLM()` 新形参；服务的 `mtp_engine_config()` 会保留它，CLI checkpoint 覆盖也不会删除该设置。

升级 worker 后重新导出、应用补丁并按原方式重启。`ASR_MTP_CONFIG` 日志记录 `verify_base_signature=False/True`。开启时沿用旧算法，并在不匹配错误中打印训练和服务签名 / 模型路径；该模式也只在启动时执行，不增加 decode 阶段开销。checkpoint 文件保持原样。架构、辅助头 state_dict、引擎并行 / 缓存约束独立检查，关闭指纹校验不会影响这些检查。

## 3. ASR 请求接线与 A/B

引擎开启 MTP 后，已有 `ascend_external_draft_first_verify=True` 的正式请求自动接力，两种草稿来源相同。正式 chunk/finish 使用 first-verify；更新后的 Prespec helper 可另外给 shadow 设置 `ascend_mtp=True`，保留其原有回滚验证之后再接 MTP。两条路径的首轮草稿合同不同。

要让**无外部草稿**的正式请求也在首次目标采样后进入 MTP，或明确做请求级 A/B，把本仓更新后的 `streaming_prespec_formal_args.py` 复制到服务的 `qwen_asr/inference/`，在 **chunk 和 finish 两处**现有 helper 调用都传入：

```python
request_kwargs["sampling_params"].extra_args = formal_external_draft_extra_args(
    request_kwargs["sampling_params"].extra_args,
    candidate_tokens=(candidate.formal_draft_token_ids if candidate is not None else None),
    rollback_tokens=rollback_draft_token_ids,
    rollback_full_tokens=rollback_full_token_ids,
    kmax=self.kmax,
    rejoin_policy=self.prespec_rejoin_policy,
    rollback_first_verify=True,
    prespec_first_mismatch_ngram=self.prespec_first_mismatch_ngram,
    mtp_enabled=self.mtp_enabled,
)
```

`mtp_enabled=True` 注入 `ascend_mtp=True`；`False` 注入显式 opt-out；helper 自身的兼容默认值 `None` 不注入，由 first-verify 标记决定。服务必须按 [启动接线](streaming_mtp_service.md) 显式保存并传入 `self.mtp_enabled`。此参数仅在正式 helper 中传值，**不要在 `_pack_stream_request()` 全局注入，也不要把正式 first-verify helper 用于 shadow**。OFF/ON 均保留回滚首轮验证和相同初始草稿。MTP ON 需要 `rollback_first_verify=True`，防止误把旧的 `C[1:]` 合同当成完整草稿。MTP OFF 仍使用原重接策略，因此这是两套完整后续策略的对照，不是只衡量辅助头 kernel。

### 后台 Prespec 验证回滚后接 MTP

同时升级引擎补丁和服务的 `qwen_asr/inference/streaming_prespec_helpers.py`。若服务复制了 helper 方法，应同步复制更新后的 `_build_stream_request()`，不能只增加 `ascend_mtp=True`：旧版 worker 会直接跳过 legacy 草稿。新版 worker 会先返回一次旧回滚草稿，再在下一次验证返回后转入 MTP。

通过 [完整服务接线](streaming_mtp_service.md) 将 `args.prespec_mtp_enabled` 传到 `AsyncLLM()`，再传到 `enable_prespec()`，初始化 `self.prespec_mtp_enabled = mtp_enabled and prespec_mtp_enabled`。不能只依赖请求 builder 的 `getattr(..., True)`，也不能只修改实例中的一处赋值。`--mtp --no-prespec-mtp` 仅关闭后台 MTP，保留正式 MTP；`--no-mtp` 会同时关闭两者并移除引擎 MTP 配置。

- 有旧回滚：目标先采样首 token，worker 提交原来的 `C[1:]` **一次**；接受 0 个、部分或全部之后，均从已确认前缀和修正 / bonus drive 进入 MTP。不会 first-verify `C[1:]`，不会重复拼接回滚草稿。
- 没有旧回滚：首次目标采样后直接用 MTP，直到候选结束、预算用尽或被取消。
- shadow 请求从专用 SamplingParams 克隆上清除 formal first-verify/rejoin 标记；原音频、prefix、candidate budget、CUMULATIVE 部分结果、deadline、abort/aclose 流程保持原规则。部分候选仍必须满足已有的长于旧回滚范围等领取条件，正式请求继续验证候选。
- 容量不足时仍先保留原来的旧回滚验证，后续退回普通自回归；生命周期结束时清理 legacy 已提交标记与辅助缓存。shadow 与 formal 共用 NPU 和缓存池，性能需按同负载分别测试正式 ON / shadow OFF 与正式 ON / shadow ON，尤其检查 deadline 命中率和正式 P95。

## 4. 执行时序与开销

设采样完成后 CPU token 表中的完整已确认序列长 N。最后一个刚采样的修正 / bonus token 尚未被目标前向执行，目标 hidden 只保留前 N−1 行；拒绝草稿的其他行丢弃。MTP 把最后采样 token 作为 drive，只返回其后辅助 token。下一次目标执行同时处理 drive 和辅助草稿，**不增加修正 token 的单独目标前向**。

每个 head 的 shifted KV / hidden 独立缓存，深度 d 最多保留 `confirmed_input_length - d` 个锚点，避免保留依赖被拒绝 token 的状态。请求按 query 长度的 2 次幂分桶，避免新 prefill 与增量 decode 混批后的大量 padding；辅助 token 在设备上逐 head 生成，整次提案只有一次 batch `tolist()`。CPU 上已有的 token IDs 不算额外 GPU 同步。目标 KV 归引擎所有，辅助缓存不读取其物理布局。

采样块中 EOS 后即使附有 bonus，也会跳过这条请求的 MTP；同时遵守请求的 `ignore_eos`、显式 stop token 和 `min_tokens`，避免在即将停止的请求上增加辅助计算。

达到 context cap、缓存槽耗尽或请求采样设置不支持时，该请求后续返回空草稿，正常目标解码；不重算全历史、不重新排队等待缓存，也不重新执行音频编码。`ASR_MTP_FALLBACK` DEBUG 日志和 worker 计数记录原因。finished / abort 清理状态；暂时未调度、batch 重排不释放仍存活请求的缓存。重计算覆盖旧行时裁剪辅助状态；缺少 hidden 前缀则安全退回普通解码。辅助草稿受 head 数、引擎 kmax、max_tokens 剩余输出数、模型上下文余量共同约束。

## 5. 验证边界

本机检查包括：加载真实格式的小型 checkpoint、与 supplied SequentialMTP 比较批量辅助输出、初始草稿接受 0/1/全部后的真实 tiny Qwen3 目标循环、EOS/预算、缓存与清理、混合 shadow/formal 分派、真实 runner API 和补丁应用。测试不依赖独立 replay 脚本。

```bash
python -m unittest discover -s research -p 'test_asr_mtp_engine.py' -v
python -m unittest discover -s research -p 'test_patch_external_draft_mtp.py' -v
python -m unittest discover -s research -p 'test_build_asr_mtp_patch.py' -v
python -m unittest discover -s research -p 'test_streaming_prespec_formal_args.py' -v
python -m unittest discover -s research -p 'test_prespec_mtp.py' -v
python -m unittest discover -s research -p 'test_streaming_mtp_config.py' -v
```

前一组需要 Torch、transformers 4.57.6 及 `SPEC_ASR_TEST_ROOT` 对应的参考源码（当前测试默认使用 `.tmp/asr-mtp-reference`）；这些只用于 CPU 核对，不是线上模块的运行依赖。完整 910B checkpoint 的 BF16/NPU 数值和性能尚未实测。部署后先核对 `ASR_MTP_CONFIG`、正式 `INITIAL_DRAFT` 和后续 `ASR_VERIFY_STEP`，以相同音频 / 并发对照完整目标输出、decode 轮次、commit P50/P95/P99、队列等待和显存；辅助输出匹配或减少 head 计算位置不等于已经提高在线性能。单独 NPU 辅助 kernel 的数值差异可能改变接受率，所有输出仍由目标 verifier 决定。

## MTP acceptance statistics

See [MTP acceptance and head-count ablation](asr_mtp_acceptance.md). Enable `additional_config.asr_mtp.acceptance_stats=True` for MTP-only INFO summaries; the default is disabled and adds no device synchronization. Install the metrics module together with the updated worker.

## Cold / warm head depth

默认 `cold_heads=实际加载的 num_heads`、`warm_heads=min(3, 实际加载的 num_heads)`。
加载 5 heads 时即 cold MTP5 / warm MTP3；两项都设为 5 可恢复固定 MTP5。
两项必须是正整数，且不超过实际加载深度（还受 checkpoint 与引擎投机预算限制）。
冷暖阶段按每个引擎请求首次/后续实际提交的 MTP 草稿划分，与 chunk 编号无关，
也不依赖 `acceptance_stats`。prefill-only、旧 rollback 草稿和已结束请求不会消费 cold；
结束清理后复用请求 ID 会重新 cold。失配、补算不会重置为 cold。

普通 token 图按 `(执行 head 数, token 桶)` 捕获；旧 B/Q 图也增加执行深度。
混合 cold/warm 批次按深度分组，结果仍在整个 worker 批次完成后只回传一次。
warm 的图只捕获前三个 head，后两个 head 的数学计算、LM head、FIA task update 均不执行；
其缓存有效前缀仍按确认上下文裁剪，不会被虚假推进。EOS 仍在返回草稿时截断并交目标模型验证，
不使用条件节点或图内 EOS 提前结束；cold 遇到早期 EOS 时仍可能执行完所选 5 heads。

图数量是原 token/BQ 桶数乘以不同执行深度数，上限仍为 64。
例如 31 个 token 桶配 3/5 两个深度共 62 张图；额外常驻图内存会计入启动内存预算。
剩余输出预算不足时返回更少草稿，图选择能容纳该预算的最小已捕获深度。

日志检查：

```bash
rg -n 'ASR_MTP_CONFIG|ASR_MTP_GRAPH_READY|ASR_MTP_STEP' "$LOG"
```

配置日志包含 `cold_heads=5 warm_heads=3`，图就绪日志包含 `head_counts=(3, 5)`。
step 中的 `cold_requests` / `warm_requests` 是本批请求数；
`graph_stats` 中 `replays_h3` / `replays_h5` 是各深度累计图执行次数。
先比较固定 5/5 与 5/3 的端到端时延、各阶段实际补算与接受长度；本机 CPU 测试不代表 910B 收益。

更新已有 acceptance 版本可使用引擎交付目录的 `asr_mtp_phase_depth_v023.patch`，
或重新运行导出器按服务器实际源码生成增量补丁。不要叠加已撤回的 EOS 条件图补丁。

服务器验证普通 3/5 深度图和 FIA 更新（真实 NPU 用例不能 skipped）：

```bash
PYTHONPATH="$PWD/vllm_ascend/spec_decode${PYTHONPATH:+:$PYTHONPATH}" \
  python -m unittest discover -s examples/910b/tests -p test_asr_mtp_phase.py -v
PYTHONPATH="$PWD/vllm_ascend/spec_decode${PYTHONPATH:+:$PYTHONPATH}" \
  python -m unittest discover -s examples/910b/tests -p test_asr_mtp_token_graph.py -v
```

## 图内存预算不足

从已安装 cold/warm 版本更新显存计量，使用引擎交付目录的
`asr_mtp_graph_memory_v023.patch`，包含两个图模块、worker 和启动 memory hook。
其他累计安装路径也已包含修正；选择与实际安装版本匹配的一条，不叠加。

```bash
rg -n 'ASR_MTP_GRAPH_READY|ASR_MTP_GRAPH_MEMORY|Available KV cache memory' "$LOG"
```

`reserved_mib` 是图的实际 retained pool 预留，`allocated_mib` 是捕获新增的 live tensor
分配。`base_available_mib` 是图已常驻后的原始 profiling 结果；自动预算使用
`adjusted_available_mib = base_available_mib - reserved_mib`。结果仍不足则继续拒绝启动。
这次修正不减少实际图内存，也不保证所有配置都能启动。

例如用户日志 31 个 token 桶 × 两个深度 = 62 张图，占 15658 MiB（约 15.29 GiB）。
若修正后实际容量仍不足，可先保留最大桶 1024、缩减中间档位试验：

```python
mtp_settings["graph_token_sizes"] = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
```

此时共 18 张图；实际显存变化不是按图数严格线性缩放，较粗桶会增加 padding，
须比较同负载下的显存、接受长度和 commit 时延。

## 按 chunk / finish 分别选择深度

可选设置 `chunk_cold_heads`、`chunk_warm_heads`、`finish_cold_heads`、
`finish_warm_heads` 分别覆盖该类请求的深度；未设置的值继承全局 `cold_heads` / `warm_heads`。
默认配置行为不变。shadow 和未识别类型始终使用全局值。

在已有 `additional_config.asr_mtp` 内合并下面配置，保留 checkpoint 等原有参数：

```python
mtp_settings.update({
    "num_heads": 5,
    "cold_heads": 3,
    "warm_heads": 3,
    "finish_cold_heads": 5,
})
```

结果为 chunk 3/3、finish 5/3、shadow/other 3/3。
`num_heads` 必须仍加载五头，不能设为 3 后再请求 finish 五头；所有覆盖项都会按实际
加载深度校验，不接受 bool、字符串、None 或越界数值。设置其他组合时可显式指定四项覆盖值。

请求类型识别与接受率统计一致：`stream_` 是 chunk，`finish_` 是 finish，`prespec-`
是 shadow，允许后续引擎 request ID 后缀。cold/warm 生命周期、EOS 截断、实际剩余预算
和缓存有效边界保持原规则，策略也不依赖是否开启接受率统计。

启动日志显示全局 `cold_heads=3 warm_heads=3`，以及解析后的
`chunk_heads=(3, 3) finish_heads=(5, 3)`。STEP 中的全局 depth 字段仍是回退值；
实际图执行深度查看 `replays_h3` / `replays_h5`。接受率汇总的 `num_heads=5` 仍指最大加载深度。

图捕获使用全局与请求覆盖项所需深度的去重集合。本例仍只捕获 3、5 两套，
不会按 chunk/finish 再翻倍；31 个 token 桶对应 62 张图，九个桶对应 18 张图。
混合请求复用原有按深度分组和单次整批 D2H。此前的图显存计量修正仍保留。

已安装显存计量修正版可使用 `asr_mtp_request_depth_v023.patch` 聚焦更新；
累计安装路径也已包含新设置。服务侧 `mtp_engine_config()` 会透传这些引擎配置，
无需增加服务命令行参数。若其他服务封装自行过滤字段，需同步透传以上四项。

这是一项消融策略能力，不能根据接受率报告宣称已改善时延；需按同负载分别比较
chunk、finish 的耗时、目标验证轮次、commit 时延与显存占用。
