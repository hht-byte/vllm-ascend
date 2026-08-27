# Session API 接入稳定窗口缓存

本文只描述现有 Session API 服务需要增加的三个接入点。现有服务继续拥有 VAD、6–10 秒累计音频、连续 2 秒块合并与推理触发、文本回滚、`inference_seq`、请求排序和最终清理调度；Adapter 不重复实现这些能力。

## 1. 进程启动：构造 Adapter 与 Engine

使用部署实际的模型、特征提取器和 Audio Tower 版本或制品摘要作为三个 fingerprint。它们是缓存身份的一部分，升级任一组件都必须变化。

```python
from vllm import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

from qwen3_asr_window_cache import (
    WindowCacheConfig,
    WindowedRequestAdapter,
    prepare_vllm_config,
)

adapter = WindowedRequestAdapter(
    WindowCacheConfig(
        model_fingerprint=model_artifact_sha256,
        feature_extractor_fingerprint=feature_extractor_version,
        audio_encoder_fingerprint=audio_tower_version,
    )
)
engine_args = AsyncEngineArgs(
    model=model_path,
    block_size=128,
    enable_prefix_caching=True,
    limit_mm_per_prompt={"audio": 5},
)
vllm_config = prepare_vllm_config(engine_args, hash_block_size=32)
engine = AsyncLLM.from_vllm_config(vllm_config)
```

每个 Engine 绑定一张 310P。同一个 `(session_id, utterance_epoch)` 仍由 Session API 串行提交；不同 Session 可以使用原服务的并发调度。

## 2. 现有推理触发点：构造并原样提交请求

现有服务完成 VAD 累计、负载合并并决定触发推理后，调用一次 `build_request`。输入是当前 VAD 语句的单条累计 mono 16 kHz、C-contiguous `float32` PCM，而不是 sealed/open 音频数组。

```python
request = adapter.build_request(
    session_id=session_id,
    utterance_epoch=utterance_epoch,
    accumulated_audio=accumulated_audio,
    sample_rate=16_000,
    window_sec=configured_window_seconds,
    is_final=is_final,
    prompt=existing_qwen3_asr_prompt,
)

async for output in engine.generate(
    request,                 # dict 原样提交，不重写 UUID/cache_salt/audio
    existing_sampling_params,
    existing_request_id,     # 仍由 Session API 用 inference_seq 生成
):
    consume_with_existing_order_and_rollback_logic(output)
```

调用方在 vLLM 消费请求前不得原地修改累计 PCM；建议像现有累计逻辑一样替换缓冲区。窗口为 2/4/8 秒之一，同一 epoch 内不能变更。Adapter 只保留小型 CPU 生命周期元数据；embedding 和 KV 张量始终由 vLLM 原生缓存持有。

## 3. 现有最终清理点：释放 Adapter 元数据

最终结果已经通过现有 `inference_seq` 新鲜度检查、文本回滚和业务清理后，再释放该 VAD 语句的 Adapter 状态：

```python
adapter.release_session(session_id, utterance_epoch)
```

调用是幂等的。它不会主动删除 vLLM Encoder/Prefix Cache；缓存条目由 vLLM LRU 自然淘汰。Session API 开始新 VAD 语句时必须增加 `utterance_epoch`，避免同一个业务 Session 错用上一句的生命周期状态。

## 职责边界与回退

接入不迁移或重写 chunk 合并、推理触发、VAD、语言识别、3–6 token 文本回滚、`inference_seq`、请求 ID、结果排序或超时取消。移除上述三个接入点即可恢复原累计音频路径。

默认物理 `block_size=128`、哈希 `hash_block_size=32`。只有 310P 初始化或完整正确性验收失败时才把哈希块改为 128；必须重新运行全部 token 等价性与性能矩阵。回退造成的吞吐下降应解释为命中粒度变粗，不能关闭 PCM 内容哈希、Session namespace 或 epoch 隔离。
