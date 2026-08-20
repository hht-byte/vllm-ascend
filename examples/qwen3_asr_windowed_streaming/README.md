# Qwen3-ASR 窗口重算与跨请求缓存复用

本目录提供一个面向 vLLM/vllm-ascend 0.23.0 的非侵入式实现。业务仍然维护
一个长期存活的 `AsyncLLMEngine`，每收到 2 秒音频就发起一个新的异步请求；不使用
vLLM Realtime 接口，也不修改 Qwen3-ASR、调度器、KV block manager 或 NPU model
runner。

核心规则如下：

- 最近 8 秒作为可变窗口，每次重新计算 Audio Encoder 和 KV；
- 窗口外音频按 4 秒冻结成不可变 item，后续请求复用相同 UUID；
- 活跃窗口的 UUID 由内容生成，音频增长后一定变化，避免误命中旧 embedding；
- 冻结 item、活跃 item 仍处于同一个
  `<|audio_start|>...<|audio_end|>` 音频区间，中间不插入文本 token；
- 通过 `multi_modal_uuids` 复用 Audio Encoder 输出，通过 automatic prefix
  caching 复用活跃窗口之前的完整 KV block。

默认 4 秒冻结单元来自 Qwen3-ASR-1.7B 的 Audio Encoder 配置：2 秒卷积 chunk
组合成约 4 秒的 encoder attention window。按这个边界拆分比任意切点更接近累计
编码；如果使用了不同模型 revision，必须根据实际 `audio_config.n_window`、
`n_window_infer` 和特征帧率重新核对，不能直接沿用 4 秒。

## 1. 创建长期存活的引擎

0.23.0 中 `AsyncLLMEngine` 是 `AsyncLLM` 的兼容别名，可以继续使用业务已有的
`AsyncLLMEngine.from_engine_args` 初始化方式：

```python
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

MODEL_PATH = "/models/Qwen3-ASR-1.7B"

# 每个冻结 item 为 4 秒。64 个 item 大约可覆盖 4 分钟音频，另留一个活跃窗口。
MAX_AUDIO_ITEMS = 65

engine_args = AsyncEngineArgs(
    model=MODEL_PATH,
    enable_prefix_caching=True,
    limit_mm_per_prompt={"audio": MAX_AUDIO_ITEMS},
    # tensor_parallel_size=2,  # 按实际部署配置
)
engine = AsyncLLMEngine.from_engine_args(engine_args)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=32,
)
```

所有会话共享这个 engine，但每个会话必须各自创建
`WindowedAudioState` 和 `WindowedAsyncLLMEngineAdapter`。多实例部署时应对同一会话
做粘性路由，因为 encoder cache 和 KV cache 都只存在于当前 engine 实例内。

## 2. 从官方 processor 构建原始 prompt

沿用官方 Qwen-ASR 服务已经加载的 processor：

```python
messages = [
    {"role": "system", "content": context or ""},
    {"role": "user", "content": [{"type": "audio", "audio": ""}]},
]
prompt_raw = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False,
)

audio_item_placeholder = processor.audio_token  # <|audio_pad|>
assert prompt_raw.count(audio_item_placeholder) == 1
```

这里传给 builder 的必须是内部 item token `<|audio_pad|>`。不要传
`Qwen3ASRForConditionalGeneration.get_placeholder_str(...)` 返回的完整
`<|audio_start|><|audio_pad|><|audio_end|>`；否则每个冻结分段之间都会新增
start/end 文本 token，改变官方累计音频的序列布局并降低 KV 前缀一致性。

## 3. 为每个会话创建窗口状态

```python
from examples.qwen3_asr_windowed_streaming import (
    Qwen3ASRPromptBuilder,
    Qwen3ASRRollbackState,
    WindowedAsyncLLMEngineAdapter,
    WindowedAudioState,
)

audio_state = WindowedAudioState(
    sample_rate=16_000,
    step_seconds=2.0,
    freeze_unit_seconds=4.0,
    recompute_window_seconds=8.0,
    # 必须包含会影响 embedding 的模型版本和特征配置，升级后自然失效旧缓存。
    cache_namespace="qwen3-asr-1.7b@model-revision|feature-config-v1",
)

prompt_builder = Qwen3ASRPromptBuilder(
    prompt_template=prompt_raw,
    audio_item_placeholder=audio_item_placeholder,
)

rollback_state = Qwen3ASRRollbackState(
    tokenizer=processor.tokenizer,
    output_to_generated_text=lambda output: output.outputs[0].text,
    unfixed_chunk_num=2,
    unfixed_token_num=5,
)

stream = WindowedAsyncLLMEngineAdapter(
    engine=engine,
    audio_state=audio_state,
    prompt_builder=prompt_builder,
    sampling_params=sampling_params,
    output_to_committed_text=rollback_state,
)
```

`Qwen3ASRRollbackState` 输入本轮最终 `RequestOutput`，返回下一轮要追加到 prompt 的
raw prefix 字符串。它复刻官方流式实现中的策略：前 `unfixed_chunk_num` 轮返回空串，
之后对累计 raw decoded 文本回退最后 `unfixed_token_num` 个 tokenizer token，再返回
稳定前缀。用于展示的语言和文本仍对 `rollback_state.raw_decoded` 调用官方
`parse_asr_output`，不要对回退后的 `rollback_state.prefix` 调用。

如果业务暂时不需要文本前缀复用，可以不传 `output_to_committed_text`；这只影响文本
prompt 策略，不影响窗口外音频 embedding 和 KV 的安全复用。

## 4. 接收音频

```python
from qwen_asr import parse_asr_output


async def on_pcm(stream, pcm16k):
    # pcm16k 可以是 mono int16 或 float32。一次 push 可以小于或大于 2 秒。
    for result in await stream.push(pcm16k):
        final_request_output = result.output
        language, text = parse_asr_output(rollback_state.raw_decoded)
        # 将 language/text 推送给客户端。


async def on_end(stream):
    tail = await stream.flush()
    if tail is not None:
        final_request_output = tail.output
        language, text = parse_asr_output(rollback_state.raw_decoded)
```

adapter 使用会话级异步锁，保证同一会话不会并发修改音频和文本状态；不同会话仍可
并发调用同一个 `AsyncLLMEngine`。每次请求都会生成新的 request ID，不能把业务
session ID 直接重复用作 request ID。

## 缓存语义

UUID 不是“让 embedding 保持不变”的开关。只有音频内容、绝对范围、采样率、模型
版本和特征配置均不再变化的冻结 item 才能复用 UUID。活跃窗口每次增长后 UUID
变化，因此会重算 embedding，KV 复用也会在该 item 之前自动停止。缓存被驱逐只会
导致重算，不应影响正确性。某个较短的活跃 item 后来以完全相同的范围和内容晋升为
冻结 item 时会保留同一 UUID，从而复用它早期已经计算过的 encoder 输出。

`limit_mm_per_prompt["audio"]` 必须覆盖最长会话的冻结 item 数加一个活跃 item。
当前实现不会丢弃冻结前缀；超长会话应在业务层切段，或结合模型上下文上限设置明确
的单 utterance 时长限制。

## 精度与上线门槛

本方案把一段连续音频拆成多个 Audio Encoder item。分段边界处的特征提取、卷积上下文
和位置处理可能与官方“每轮累计整段音频”存在差异，因此不能仅凭缓存命中率判断精度。
8 秒窗口是偏保守的初始值，不是已经证明的最优值。

上线前至少在真实 Qwen3-ASR-1.7B 权重和目标 Ascend 卡上比较：

1. 官方累计实现与窗口实现的流式 WER/CER；
2. 已提交前缀的回改率和连续回改长度；
3. 4、8、12 秒窗口在短句、长句、静音、噪声、语码混合场景的差异；
4. encoder cache、prefix cache 命中率，以及 P50/P95 每步时延；
5. 长会话下 HBM、KV block 占用和缓存驱逐情况。

建议先以 8 秒窗口灰度；只有在业务集上确认精度接近官方 realtime 后，再尝试缩短到
4 秒。若边界误差明显，优先增大窗口或调整 4 秒冻结边界，不要复用仍可能变化的 UUID。
