# Ascend 310P 正确性与性能验收

本地开发环境没有 310P；本仓库只完成可执行工具与非 NPU 门禁。下列 NPU 项均须在目标机真实运行后才能标记通过，不能从 Fake Pipeline 或论文数据推断收益。

## 1. 环境门禁

在干净环境安装本项目 benchmark extra、`vllm==0.23.0` 和 `vllm-ascend==0.23.0`，确认模型目录是待部署的 Qwen3-ASR-1.7B 制品：

```bash
python -m pip install -e '.[benchmark]'
python -c 'import vllm, vllm_ascend; print(vllm.__version__, vllm_ascend.__version__)'
npu-smi info
bash scripts/verify_upstream_clean.sh
```

每个 Engine 只绑定一张 310P。记录 CANN、驱动、固件、Python、dtype、量化配置、模型制品摘要和运行命令；不要在报告中保存真实 PCM、Session ID 或敏感路径。

## 2. 数据与矩阵

manifest 是 UTF-8 JSONL，每行只能包含以下字段：

```json
{"id":"zh-001","audio_npy":"/data/zh-001.float32.npy","sample_rate":16000,"checkpoints_seconds":[6,8,10],"language":"zh","reference":"测试文本"}
```

`.npy` 必须是一维 C-contiguous `float32`、mono 16 kHz、总长 6–10 秒；checkpoint 严格递增且位于 6 秒至总时长。数据至少覆盖业务中的中文、英文及其他目标语言，并覆盖 6/8/10 秒、2/4/8 秒窗口、并发 1 和业务典型并发。不要把窗口策略间的差异当作缓存误差：每个窗口策略都只和自身 cache-off 全量重算基线比较。

## 3. 正确性门禁

先设置显式输入再运行 NPU 测试；未设置这两个变量是唯一允许的 skip 条件。变量已设置后，模型/manifest 错误、Engine 初始化失败和任何输出不一致都必须失败：

```bash
export QWEN3_ASR_310P_MODEL=/models/Qwen3-ASR-1.7B
export QWEN3_ASR_310P_MANIFEST=/data/asr-validation.jsonl
python -m pytest -m npu tests/integration/test_310p_equivalence.py -vv
```

测试逐条检查 2/4/8 秒窗口下 cache-off 与 reuse 的 greedy token IDs、语言字段和文本，并覆盖稳定累计请求、显式 Encoder/Prefix Cache reset、唯一 Session 的 LRU 压力、`release_session` 后同 Session/epoch 重建。失败时仅保存 record ID、窗口、checkpoint、场景与两侧输出的最小复现 JSON，不保存音频或 Session ID。

cache-off 与 reuse 使用两个独立 Engine 生命周期。两者模型、prompt、窗口、dtype、量化、`temperature=0`、`top_p=1` 和 `max_tokens` 相同；cache-off 关闭 Prefix Cache，并为每个请求的每个 Adapter UUID 生成含 request ID 的一次性 SHA-256；reuse 原样使用 Adapter UUID/cache salt，并通过 `prepare_vllm_config(..., hash_block_size=32)` 启动。

## 4. 性能测量

先以与正式测量完全相同的参数执行 3 轮预热并丢弃输出，再执行 20 轮测量。预热必须在每个新 Engine 和每个矩阵配置上完成；若使用本脚本之外的编排器，不得把 Engine 初始化计入 TTFT/尾字时延。

```bash
python benchmarks/benchmark_310p.py \
  --model /models/Qwen3-ASR-1.7B \
  --manifest /data/asr-validation.jsonl \
  --window-seconds 2 4 8 \
  --concurrency 1 4 8 \
  --warmup-iterations 3 \
  --iterations 20 \
  --output benchmark-results/310p.jsonl
```

输出每请求 JSONL 与每个 mode/window/concurrency 的 P50/P95 汇总，不内置“提升至少 X%”阈值。每请求包含音频/窗口/并发、request ID、sealed/open 窗口数、token IDs、文本、`RequestOutput.num_cached_tokens`、TTFT、最终时延、四个 cache counter delta 和峰值 NPU 内存。

Prometheus counter 按完整 metric family 汇总所有 label sample。before/after 缺失时写 `null` 和 warning；after 小于 before 视为进程/counter reset，同样写 `null`，不伪造 0 或负数。并发测量中的进程级 counter delta 可能包含同一时段其他请求，应结合整组汇总解释。

峰值 NPU 内存优先来自 `torch.npu.max_memory_allocated`，字段同时写 provenance；运行时不提供 sampler 时写 `null` 和 warning。不得用主机 RSS 或常量代替 NPU 内存。

## 5. Audio Tower 与 LLM prefill 分段

`RequestOutput` 只直接提供最终输出和 `num_cached_tokens`，不提供可信的 Audio Tower/prefill 拆分时长。分段时间使用两条证据链：

1. 保存 vLLM `/metrics` 或进程 Prometheus registry 的请求、multimodal cache、prefix cache 和 prefill 相关指标快照；按同一测量区间做 counter/histogram 差分。
2. 对代表性的 6/8/10 秒、并发 1 与业务典型并发运行 `msprof`，在 timeline 中按 Qwen3-ASR Audio Tower/Fbank/CNN/AuT 调用和 LLM prefill kernel 区间聚合。记录 msprof/CANN 版本、采集命令、时间范围与 warmup 排除规则。

不要从 TTFT 减法反推并宣称 Audio Tower 时间。msprof 会扰动绝对时延：功能/分段采集与无 profiler 的正式 P50/P95 分开运行。

## 6. 多语言质量评估入口

将两种模式 JSONL 按 `record_id/window_seconds/checkpoint_seconds` 对齐，用业务现有 scorer 对空格语言计算 WER、对中日韩等语言计算 CER。缓存正确性要求同窗口 cache-off/reuse token IDs、文本、语言和逐语言 CER/WER 完全一致；窗口 2/4/8 之间允许不同，但必须分别对 reference 报告，供业务选择默认窗口。

## 7. 成功标准

- 所有 manifest 行、窗口、checkpoint、reset/LRU/recreate 场景的 greedy token IDs、语言与文本完全一致。
- 报告覆盖 cache-off/reuse、2/4/8 秒、6/8/10 秒及并发 1/业务典型并发，无缺失矩阵项。
- cache counter、`num_cached_tokens`、TTFT、最终时延、P50/P95、峰值内存都带真实值或明确的 `null` provenance/warning。
- vLLM、vLLM-Ascend 和 checkpoint 零修改；Session/PCM 身份校验始终开启。
- 性能结论只来自目标机测量；根据尾字时延、吞吐、内存与多语言质量共同选择窗口，不设置预设收益阈值。

## 8. 32 → 128 回退

只有 `hash_block_size=32` 在 vLLM-Ascend/310P 初始化失败，或完整 token 等价性验收失败时，才改为 128。保留物理 `block_size=128`、Prefix Cache 和所有 PCM/Session/epoch 身份校验，重新运行完整正确性、质量和性能矩阵。若吞吐或尾字时延收益下降，解释为 prefix 命中粒度从 32 token 变粗到 128 token；不能改用 Session ID 强制命中，也不能绕过内容哈希。

## 9. 故障定位顺序

1. 先确认版本、模型 fingerprint、dtype/量化、prompt、窗口、sampling 参数和 manifest slice 完全相同。
2. 再比较 Adapter audio item、anchor、UUID 与 cache salt；验证 cache-off UUID 每请求唯一、reuse UUID 内容稳定。
3. 检查 `num_cached_tokens` 与四个 Prometheus counter；缺失指标先修采集，不把 null 当 0。
4. 显式 reset Encoder/Prefix Cache 后复测；若恢复一致，检查 UUID/namespace 生命周期和并发串行约束。
5. 以 `hash_block_size=128` 跑完整矩阵判断 32 粒度兼容性；不得只跑单条样本后决定回退。
6. 最后使用 msprof 定位 Audio Tower、prefill、调度或 NPU kernel；profiler 数据不与无 profiler 的时延报告混用。
