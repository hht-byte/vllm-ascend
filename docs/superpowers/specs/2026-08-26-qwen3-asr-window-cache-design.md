# Qwen3-ASR 稳定窗口 Embedding 与 KVCache 复用设计

## 1. 背景与目标

当前流式 ASR 服务使用 Qwen3-ASR-1.7B、vLLM 0.23.0 和 vLLM-Ascend 0.23.0。上层 Session API 服务已经负责：

- VAD 语句切分；
- Session 生命周期；
- 累计完整音频保存；
- 根据负载合并连续音频块并触发推理；
- 语言识别和 3～6 个文本 token 的自适应尾部回滚；
- 调用 vLLM `AsyncLLM` 并消费识别结果。

每个 VAD 语句的累计音频通常为 6～10 秒。当前每轮把整段累计音频作为一个多模态项提交，音频内容变化会导致整个 `mm_hash` 变化，无法复用历史音频 embedding，包含音频的 LLM KV 前缀也无法命中。

本设计的目标是：

1. 将累计音频按可配置的 2 秒、4 秒或 8 秒固定窗口切分；
2. 复用已经封口窗口的 Qwen3-ASR Audio Tower embedding；
3. 复用稳定音频前缀对应的 LLM KVCache；
4. 缓存开启后的结果与使用相同固定窗口、关闭缓存后从头全量计算的 greedy 结果一致；
5. 不修改 vLLM 和 vLLM-Ascend 上游源码，不使用 monkey patch；
6. 保留现有 Session API 服务的调度、VAD 和文本回滚逻辑。

## 2. 非目标

第一版不实现：

- 新的 Session HTTP API；
- 服务端音频块合并或负载调度；
- VAD、语言识别或文本尾部回滚；
- 跨 vLLM Engine 的 Session 迁移；
- KVCache 外部存储或跨 NPU 传输；
- 修改 Qwen3-ASR 权重、Attention 算子或 vLLM Scheduler；
- 与 Qwen3-ASR 默认累计 8 秒编码语义保持等价。

精度基线明确为“使用同一个固定窗口配置，从头全量计算”。固定窗口会改变默认累计编码的上下文范围，因此 2 秒、4 秒和 8 秒窗口之间需要单独评估识别质量。

## 3. 术语

- **累计音频**：现有 Session API 服务保存的、从当前 VAD 语句开始到当前时刻的完整 PCM。
- **稳定窗口**：长度达到 `window_sec` 的固定音频区间，后续追加音频不会再改变其内容。
- **开放窗口**：累计音频尾部不足 `window_sec` 的区间，下一轮可能继续增长。
- **最终窗口**：收到 `is_final=true` 时不足 `window_sec` 的尾部区间；它在语句结束时封口。
- **Encoder Cache**：vLLM 按多模态项标识缓存的 Audio Tower 输出。
- **Prefix Cache**：vLLM 按 token、父块哈希和多模态项标识缓存的 LLM KV 块。

## 4. 版本与部署边界

目标版本固定为：

- Python `>=3.12,<3.13`；
- `vllm==0.23.0`；
- `vllm-ascend==0.23.0`；
- Qwen3-ASR-1.7B；
- 单个 vLLM Engine 绑定一张 Ascend 310P；
- 多个用户 Session 可并发，但同一个 `(session_id, utterance_epoch)` 的推理请求串行执行。

Python 边界按 2026-08-28 用户确认的生产环境收敛到 3.12；本交付不声明 Python 3.11 兼容性。

实现以独立 Python 包 `qwen3-asr-window-cache` 交付。上游源码只用于接口契约检查和目标环境安装。

## 5. 总体架构

```text
现有 Session API 服务
  ├── VAD / Session / 音频累计
  ├── 请求合并与推理触发
  ├── 文本尾部回滚
  └── WindowedRequestAdapter.build_request(...)
          │
          ▼
Windowed Request Adapter
  ├── 校验累计音频和 Session 元数据
  ├── 按 2s / 4s / 8s 切分窗口
  ├── 生成窗口级不可变标识
  ├── 构造单一连续音频区间的 prompt
  └── 构造 multi_modal_data / multi_modal_uuids
          │
          ▼
vLLM 0.23 原生 Qwen3-ASR Processor 与模型
  ├── 每个音频项独立执行 Fbank / CNN / AuT
  ├── 按顺序连续拼接窗口 embedding
  ├── Encoder Cache 复用稳定窗口 embedding
  └── Prefix Cache 复用稳定音频前缀 KV
          │
          ▼
vLLM-Ascend 0.23 / Ascend 310P
```

源码检查表明，vLLM 0.23 原生 Qwen3-ASR Processor 已支持多个 audio item，原生模型会分别编码、按长度拆分 embedding，并根据多模态位置连续生成位置编码。因此第一版只实现 Request Adapter，不注册自定义模型，不替换原生 Processor。

## 6. Adapter 输入与输出契约

### 6.1 输入

Adapter 在现有服务初始化时接收不可变配置，其中包含模型、音频预处理和缓存 schema 指纹。单次调用接口为：

```python
build_request(
    *,
    session_id: str,
    utterance_epoch: int,
    accumulated_audio: numpy.ndarray,
    sample_rate: int,
    window_sec: int,
    is_final: bool,
    prompt: str,
) -> dict
```

约束：

- `session_id` 必须是精确的 `str`，不得为空或仅由空白字符组成；非空值中的前后空白保留并参与身份计算，不做隐式归一化；
- `utterance_epoch` 必须是精确的非负 `int`，`bool` 不作为整数接受；
- `accumulated_audio` 是 mono、16 kHz、C-contiguous `float32`；
- `window_sec` 只能是 2、4 或 8；
- 同一个 `(session_id, utterance_epoch)` 内 `window_sec` 不可改变；Adapter 的模型指纹在进程生命周期内不可改变；
- 累计采样数只能单调增加或在幂等重试时保持不变；
- `prompt` 必须恰好包含一个原生 Qwen3-ASR 音频占位区间：
  `<|audio_start|><|audio_pad|><|audio_end|>`；
- 最大累计音频为 10 秒；超过上限时拒绝请求，由现有服务开始新的 VAD 语句。

Adapter 不重新采样、不做 VAD、不修改现有文本前缀。

现有服务收到最终推理结果并完成 Session 清理时调用：

```python
release_session(session_id: str, utterance_epoch: int) -> None
```

### 6.2 输出

输出保持 vLLM `AsyncLLM.generate` 接受的 PromptType：

```python
{
    "prompt": windowed_prompt,
    "multi_modal_data": {
        "audio": [window_0, window_1, open_or_final_window],
    },
    "multi_modal_uuids": {
        "audio": [window_0_id, window_1_id, tail_window_id],
    },
    "cache_salt": session_namespace,
}
```

当没有开放尾部时，列表只包含完整稳定窗口。

## 7. 窗口切分

设：

```text
window_samples = window_sec * 16000
full_count     = total_samples // window_samples
remainder      = total_samples % window_samples
```

产生：

- `full_count` 个完整稳定窗口；
- `remainder > 0` 时产生一个开放窗口；
- `is_final=true` 时，开放窗口被标记为最终稳定窗口。

4 秒窗口示例：

```text
累计 6s  -> sealed[0:4] + open[4:6]
累计 8s  -> sealed[0:4] + sealed[4:8]
累计 10s -> sealed[0:4] + sealed[4:8] + open[8:10]
```

窗口以 NumPy 视图表示；仅在 vLLM 输入要求连续内存时对单个窗口执行必要拷贝，避免每轮复制整段 PCM。

每个窗口作为独立音频项进入原生 Audio Tower。因此 Fbank 的动态范围归一化、CNN 和 AuT 双向注意力均限制在该窗口内部。开放窗口增长时允许全部重算；完整窗口封口后不再依赖未来音频。

## 8. Prompt 布局与 embedding 拼接

Adapter 将原 prompt 中唯一的：

```text
<|audio_start|><|audio_pad|><|audio_end|>
```

替换为：

```text
<|audio_start|>
<|audio_pad|>  # window 0 anchor
<|audio_pad|>  # window 1 anchor
<|audio_pad|>  # open/final window anchor
<|audio_end|>
```

原生 Qwen3-ASR Processor 对每个 anchor 按对应音频输出长度展开。若窗口输出长度为 52、52 和 26，最终输入为：

```text
<|audio_start|>
[audio_pad x 52][audio_pad x 52][audio_pad x 26]
<|audio_end|>
```

窗口之间没有额外文本、`audio_start` 或 `audio_end`。LLM 看到一个连续的音频 embedding 区间。原生 `get_mrope_input_positions` 按多模态项的 offset 排序，并从前一项最大位置继续递增，因此多个窗口的位置编码保持连续。

## 9. 窗口身份与缓存隔离

PCM 在哈希前使用规范形式：

- 16 kHz；
- mono；
- `float32`；
- little-endian；
- C-contiguous；
- 哈希后不得再执行会改变数值的归一化。

所有公开命名空间和 Adapter 状态键操作都先执行同一组 Session scope 校验：`session_id` 是精确的非空、非纯空白 `str`，`utterance_epoch` 是精确的非负 `int` 且不是 `bool`。校验必须发生在构造字典键或读取、写入、释放状态之前，避免 Python 的 `1 == True` 键别名跨逻辑 Session 访问状态。

身份计算：

```text
session_namespace = SHA256(
    session_id,
    utterance_epoch,
    model_fingerprint
)

window_id = SHA256-CBOR(
    session_namespace,
    window_index,
    window_sec,
    start_sample,
    end_sample,
    SHA256(canonical_pcm_bytes),
    feature_extractor_fingerprint,
    audio_encoder_fingerprint,
    adapter_schema_version
)
```

属性：

- 同一个稳定窗口在后续请求中得到相同 ID；
- 开放窗口内容增长时 ID 改变；
- 完全相同请求重试时 ID 不变；
- PCM、位置、窗口大小、模型或预处理配置任一变化都会失效；
- Session 和 epoch 参与命名空间，禁止跨用户或跨语句共享缓存。

`multi_modal_uuids` 使用 `window_id`。`cache_salt` 使用 `session_namespace`，作为 vLLM Prefix Cache 的额外用户隔离层。

## 10. Embedding 缓存语义

vLLM 将每个 UUID 作为独立多模态项的缓存身份。以 4 秒窗口为例：

```text
6s 请求：H0=audio[0:4], H1=audio[4:6]
8s 请求：H0=audio[0:4], H2=audio[4:8]
```

8 秒请求中：

- `H0` 命中 Encoder Cache，复用 0～4 秒 Audio Tower embedding；
- `H2` miss，重新执行 4～8 秒的 Fbank、CNN 和 AuT；
- 6 秒请求遗留的开放窗口 `H1` 可由 vLLM LRU 自然淘汰。

Adapter 不持有 NPU embedding 张量，不绕过 vLLM 的内容身份校验。

## 11. KVCache 命中语义

vLLM Prefix Cache 的块哈希包含：

- 父块哈希；
- 当前块 token IDs；
- 与当前块相交的多模态项 identifier 及其块内 offset；
- LoRA、cache salt 等额外配置。

稳定窗口 identifier 和位置不变时，其对应音频前缀 KV 可以命中。命中在第一个发生变化的开放窗口所处哈希单元停止；该单元和所有后续单元重新计算。

设开放窗口起始位置为 `dirty_token`：

```text
reusable_prefix_tokens =
    floor(dirty_token / hash_block_size) * hash_block_size
```

目标配置：

```text
physical block_size = 128
hash_block_size     = 32
```

vLLM 0.23 的 `CacheConfig.hash_block_size` 支持比物理块更细的前缀哈希粒度，前提是能够整除物理块。若 310P 集成验证不支持该组合，部署配置回退到 `hash_block_size=128`；回退只降低命中率，不改变输出。

新增音频位于 `audio_end` 和 assistant 文本之前，因此 `audio_end` 及其后的历史文本位置会移动。第一版不宣称跨轮复用这些文本 KV；文本由现有 Session API 服务按既有回滚策略重新 prefill。

## 12. Session 元数据与并发

Adapter 只保存小型 CPU 元数据：

```text
(session_id, utterance_epoch)
  -> window_sec
  -> model_fingerprint
  -> last_sample_count
  -> finished
```

它不保存累计音频、embedding 或 KVCache。

规则：

- 同一 Session/epoch 最多一个 `AsyncLLM` 请求执行；该串行约束继续由现有 Session API 服务保证；
- `inference_seq` 和请求 ID 继续由现有 Session API 服务生成，推荐请求 ID 格式为 `session_id:utterance_epoch:inference_seq`；Adapter 不生成请求 ID，也不负责任务结果排序；
- 幂等重试允许相同累计长度和内容；
- 旧结果只有在 `inference_seq` 仍为最新时才由现有 Session API 服务接受；
- `is_final=true` 后状态标记为已结束；完全相同的 final 请求允许幂等重试，任何音频追加都被拒绝；
- 现有服务收到最终结果后调用 `release_session` 删除 Adapter CPU 元数据；vLLM 缓存条目留在 LRU 中自然淘汰；
- vLLM 缓存淘汰只能导致重算，不能导致错误命中。

## 13. 错误处理

Adapter 定义：

- `InvalidSessionId`：Session ID 不是精确的非空、非纯空白 `str`；
- `InvalidUtteranceEpoch`：epoch 不是精确的非负 `int`，包括 `bool`；
- `InvalidAudioFormat`：非 mono、非 `float32` 或数组不合法；
- `InvalidSampleRate`：采样率不是 16 kHz；
- `InvalidWindowSize`：窗口不是 2、4、8 秒之一；
- `AudioTooLong`：累计音频超过 10 秒；
- `AudioLengthRegressed`：同一 epoch 累计长度缩短；
- `WindowConfigChanged`：同一 epoch 修改窗口或模型指纹；
- `InvalidPromptPlaceholder`：音频占位区间缺失或不唯一；
- `SessionAlreadyFinished`：已结束语句继续追加；
- `UnsupportedRuntimeVersion`：vLLM 或 vLLM-Ascend 版本不匹配；
- `TooManyAudioWindows`：窗口项数超过 Engine 配置上限。

所有缓存异常的安全退化路径都是重新计算。实现不提供“忽略内容哈希、按 Session 强制命中”的配置。

## 14. Engine 配置

现有 Session API 服务创建 `AsyncLLM` 时配置：

```text
enable_prefix_caching = true
block_size = 128
hash_block_size = 32
limit_mm_per_prompt.audio = 5
```

5 个音频项覆盖最大 10 秒音频和最小 2 秒窗口。模型 architecture 保持原生 `Qwen3ASRForConditionalGeneration`，不修改 checkpoint 的 `config.json`。

vLLM 0.23.0 的 `CacheConfig` 已包含 `hash_block_size`，但 `AsyncEngineArgs` 没有暴露对应字段，也不会在 `create_engine_config()` 时传入该值。独立包因此提供一个 Engine 配置辅助函数：先要求并调用可调用的 `AsyncEngineArgs.create_engine_config()` 创建 `VllmConfig`，再在 Engine 启动前设置 `vllm_config.cache_config.hash_block_size`，最后调用原生 `AsyncLLM.from_vllm_config(vllm_config)`。这只使用公开配置对象和公开构造入口，不修改 vLLM 或 vLLM-Ascend 源码。

目标环境的推荐版本矩阵遵循 vLLM-Ascend 0.23.0 官方兼容说明。benchmark 只接受 `--hash-block-size 32` 或 `128`，默认为 32，并把所选值写入每请求结果、汇总、等价性键和复现命令。只有 32 初始化或完整正确性失败时才使用 128，且必须以 128 重跑完整 token 等价性与性能矩阵；除该值外 cache-off/reuse 条件保持相同。

## 15. 可观测性

每轮记录：

- `audio_duration_seconds`；
- `window_seconds`；
- `sealed_window_count`；
- `open_window_duration_seconds`；
- `expected_reusable_audio_tokens`；
- `processor_cache_queries/hits/misses`；
- `actual_encoder_cache_hits/misses`（当前固定为 `null`）；
- `prefix_cache_hit_tokens`；
- `prefill_computed_tokens`；
- `request_latency_ms`；
- `inference_seq`（由现有 Session API 服务附加）。

vLLM 0.23 的 `vllm:mm_cache_queries/hits` 来自 renderer 的 MultiModal Processor cache 统计，不是 EngineCore Encoder-output cache。它们只能派生为 `processor_cache_queries/hits/misses`；在没有已证明的真实 Encoder-output 指标源前，`actual_encoder_cache_hits/misses` 必须保持 `null` 并附带 provenance warning，不能据此认证 Audio Tower/Encoder Cache 命中。Prometheus before/after 是进程全局快照；当 `concurrency>1` 时，单请求 counter delta 及 processor-cache 派生值必须为 `null` 并警告请求区间重叠，不能把其他请求增量归给当前请求。`RequestOutput.num_cached_tokens` 仍是每请求 Prefix Cache 观测值。

## 16. 项目结构

```text
qwen3-asr-window-cache/
├── pyproject.toml
├── src/qwen3_asr_window_cache/
│   ├── __init__.py
│   ├── config.py
│   ├── windowing.py
│   ├── identity.py
│   ├── prompt_builder.py
│   ├── request_adapter.py
│   ├── compatibility.py
│   ├── engine_config.py
│   ├── metrics.py
│   └── errors.py
├── tests/
│   ├── unit/
│   ├── contract/
│   └── integration/
├── benchmarks/
│   └── benchmark_310p.py
├── scripts/
│   ├── fetch_upstream.sh
│   └── verify_upstream_clean.sh
└── docs/
    ├── integration.md
    └── 310p-validation.md
```

上游源码由脚本拉取到被版本控制忽略的目录：

```text
.upstream/vllm          -> tag v0.23.0
.upstream/vllm-ascend   -> tag v0.23.0
```

验证脚本要求两个目录 `git diff` 为空。

## 17. 测试策略

### 17.1 本地单元测试

覆盖 6、8、10 秒累计音频和 2、4、8 秒窗口：

- 窗口无重叠、无遗漏；
- 完整、开放和最终窗口状态正确；
- ID 稳定性和失效条件正确；
- prompt 恰好一个 `audio_start` 和一个 `audio_end`；
- anchor 数等于窗口数；
- 异常输入得到确定错误。

### 17.2 vLLM 0.23 契约测试

检查：

- 多个 audio item 与多个 prompt anchor 一一对应；
- `audio_pad` 总数等于各窗口输出 token 数之和；
- 多模态 offset 连续；
- MRoPE position 连续；
- 多模态 UUID 进入 Encoder Cache 和 Prefix Cache 身份；
- `hash_block_size=32` 与 `block_size=128` 的哈希边界行为；
- 相同完整哈希块、但从不完整尾部开始改变多模态 UUID 时，完整块仍命中且输出等价由后续重算保护；
- renderer/MM processor cache 是 `vllm:mm_cache_*` Prometheus counter 的来源，不能标记为 Encoder Cache；
- vLLM 和 vLLM-Ascend 上游目录零 diff。

### 17.3 模拟缓存集成测试

使用确定性的 Fake Audio Tower 和 Fake LLM，按 4 秒窗口依次提交 6、8、10 秒：

```text
6s：miss [0:4]、miss [4:6]
8s：hit  [0:4]、miss [4:8]
10s：hit [0:4]、hit [4:8]、miss [8:10]
```

缓存开启与强制全量重算必须生成相同 token 序列。测试还覆盖 LRU 淘汰、幂等重试、多 Session 隔离和历史 PCM 变化。

### 17.4 310P 正确性验收

比较两种模式：

```text
baseline：相同固定窗口；关闭 Prefix Cache；使用一次性 UUID 强制 Encoder miss
reuse：稳定窗口 UUID；启用 Encoder Cache 和 Prefix Cache
```

固定相同模型、prompt、窗口、量化、dtype 和 `temperature=0`。要求：

- greedy token ID 逐个一致；
- 最终语言和转写文本一致；
- 多语言 CER/WER 不发生变化；
- 重试、缓存淘汰和 Session 重建不改变结果；
- Prefix Cache reset 必须返回成功，并在释放 Adapter 元数据后以同一个 Session/epoch/namespace 重放，首个重放请求必须观测到零 cached prefix tokens；
- LRU 压力前先对同一个目标 namespace 做 exact-final warm retry，压力 Session 使用不同 cache salt 并释放 CPU 元数据，之后重放同一目标 namespace；只有 cached token 相对 warm retry 降低时才认证 `after_lru_pressure`，否则该次运行仅为诊断并要求增加压力后重跑；
- embedding 和 logits 输出 dtype 相关的数值误差报告，不要求不同 NPU kernel 路径逐 bit 一致。

### 17.5 310P 性能报告

矩阵：

- 音频 6、8、10 秒；
- 窗口 2、4、8 秒；
- 并发 1 和业务典型并发；
- 缓存关闭和开启。

输出 MM Processor Cache 查询/命中/未命中、KV 命中 token、重算 token、Audio Tower 时间、LLM prefill 时间、TTFT、尾字完成时延、P50/P95 和峰值 NPU 内存。真实 Encoder Cache 命中率在没有受支持的指标源前保持未知，不能由 processor counter 替代。第一版不承诺固定提升百分比，由目标环境实测选择默认窗口。

## 18. 非侵入式验收

交付完成时必须满足：

1. vLLM 和 vLLM-Ascend 源码目录 `git diff` 为空；
2. 不使用 monkey patch；
3. 不覆盖原生 Qwen3-ASR architecture；
4. 不修改模型 checkpoint；
5. 现有 Session API 服务只增加 Adapter 构造请求、Session 释放调用和 Engine 缓存配置；
6. Adapter 删除后可恢复原有累计音频请求路径。

## 19. 参考实现依据

- [vLLM 0.23 外置模型注册](https://docs.vllm.ai/en/v0.23.0/contributing/model/registration/)
- [vLLM 0.23 Qwen3-ASR 实现](https://github.com/vllm-project/vllm/blob/v0.23.0/vllm/model_executor/models/qwen3_asr.py)
- [vLLM 0.23 Prefix Cache 块哈希](https://github.com/vllm-project/vllm/blob/v0.23.0/vllm/v1/core/kv_cache_utils.py)
- [vLLM 0.23 CacheConfig](https://github.com/vllm-project/vllm/blob/v0.23.0/vllm/config/cache.py)
- [vLLM 0.23 多模态输入与 UUID](https://docs.vllm.ai/en/v0.23.0/api/vllm/inputs/llm/)
- [vLLM-Ascend 0.23 安装说明](https://docs.vllm.ai/projects/ascend/en/v0.23.0/installation.html)
