# 流式音频中点预投机：服务端接入件

本目录提供两个独立工具：

- `commit_overlap.py` 用现有逐 commit JSONL/CSV 检查前一条正式 commit 的重叠；若提供服务 DEBUG 日志，还检查**同一段内流式 chunk 推理与 commit 等待**的重叠。
- `streaming_prespec.py` 管理中点 shadow 请求、低负载 admission、候选版本、正式边界取消和 shadow 候选质量评估。

这两个文件不依赖 vLLM/Ascend，可以复制到 910B 服务项目。当前仓库没有运行中的 `realtime_connection.py` 和 `qwen3_asr_async.py`，因此模型适配器需要在服务项目中实现；**没有模型适配器时不会发送任何伪请求**。

## 先验证“上一轮推理尚未完成”的解释

```bash
python commit_overlap.py --commits baseline-c20-commits.jsonl \
  --service-log concurrency_20.log --log-utc-offset +00:00 \
  --out baseline-c20-overlap.json
```

`previous_inference_overlap_count` 只比较同一 `ws_id` 前一条非跳过 commit 的 `done_at` 与本条 `received_at`；它会漏掉“当前 `seg` 的 `chunk-0` 仍在 decode”的情形。`--service-log` 从同一 `(ws_id, seg, chunk)` 的 `inference submit` 到 DEBUG `token_count` 构造推理区间，计算它与 `[received_at, received_at + queue_wait_ms]` 的交集。重点看 `same_segment_chunk_overlap_count`、`same_segment_chunk_overlap_ms` 和 `queue_wait_ms_with_chunk_overlap`。日志时间若是北京时间，传 `--log-utc-offset +08:00`；与 commit JSONL 的 UTC 时间不一致会导致错误对齐。缺少 DEBUG 结束日志时 `same_segment_chunk_unknown_count` 会增加，不能把未检出的重叠当作 0。时间重叠支持等待原因的判断，但不能单独证明所有排队都由该 chunk 造成。

服务的 `Commit received: queue_size=N` 是把 `None` 哨兵放入当前段音频队列**之后**的大小，因此 `N-1` 是尚待协程取走的音频块数；这不是 vLLM waiting 请求数。`ws-cea6c68c/seg-24` 的 `queue_size=3` 表示两个音频块加一个哨兵。commit 在 `chunk-0` 尚未结束时到达，112 ms 的等待与该 chunk 推理重叠；协程之后还要消费剩余音频块，再取到哨兵。该样本不能把 118 ms 全部归因于 vLLM 排队。

## 接入现有服务

模型端为 `ShadowBackend.open_shadow(request)` 实现私有推理状态。这里应**调用服务已经实现的回滚投机验证逻辑**，把上一轮回滚 token 当草稿，在中点已到达音频上运行；不要另写一套 token 匹配或直接沿旧草稿续写。返回的 branch 必须提供：

1. `verify_and_repair_old(rollback_tokens, budget)`：在 `request.stable_prefix_tokens` 与 `request.audio_pcm_f32le` 所含的**已到达**音频条件下，复用现有 rollback verifier 验证旧回滚后缀，在首个分歧处修正，并返回 `VerifiedRollback(token_ids=修订后的后缀, continuation_state=修正后的私有 KV/解码状态)`；
2. `extend_from_verified(verified, budget)`：必须从传入的 `verified.continuation_state` 继续自回归生成新增音频对应的 token，不能从未验证的旧 KV 或原始 `rollback_tokens` 直接续写；
3. `close()`：包括取消和异常路径，都释放 request、KV 和模型侧资源。

branch 不得复用权威 request 的可变 KV、slot mapping 或输出缓冲。`VerifiedRollback.token_ids` 只含验证/修正后的旧后缀，`continuation_state` 是适配器管理的不透明私有状态；两者必须对应同一前缀与中点音频快照。`RepairExtendProducer` 按以上顺序调用，拒绝只返回 token 列表而没有修正后状态的适配器结果，并受 `max_candidate_tokens` 限制。候选 token **不是最终结果**，消费模式仍须交给已有的权威 target verifier 在完整正式音频上重新验证。

在服务进程创建一个 `PreSpecCoordinator(RepairExtendProducer(backend), policy)`。默认 `shadow_only=True`，正式路径仍使用原来的 rollback draft/target-only 逻辑。可传 `on_probe` 记录每次权威 chunk/finish 开始时的候选状态；写盘应通过 `logging.handlers.QueueHandler` 等非阻塞队列，避免测量本身拖慢正式路径：

```python
from dataclasses import asdict
import json

coordinator = PreSpecCoordinator(
    RepairExtendProducer(backend), policy,
    on_probe=lambda probe: probe_logger.info(
        "prespec_probe=%s", json.dumps(asdict(probe))))
```

探针包含 `session_id`、`audio_epoch`、`boundary`（`chunk`/`formal`）、单调时钟 `target_start_ns`、`status` 和就绪时的 `ready_lead_ms`。`ready_shadow_only` 表示及时就绪但未参与权威推理；`ready_consumed` 表示交给 verifier；`pending` 表示权威推理开始时伪请求尚未完成并被取消。`previously_used` 表示该段的候选已在更早的 chunk 被处理，不能当作新的失败。探针写入异常只增加 `coordinator.probe_error_count`，不会阻断正式请求。以 `(session_id, audio_epoch)` 对齐逐 commit 的 `(ws_id, seg)`，前提是接入时明确令 `audio_epoch=seg`；单调时钟不能直接与日志 UTC 时间戳相减。接入点：

```python
# 音频 append 后，当前正式窗口首次达到 1 s (16k samples) 时：
request = ShadowRequest(
    session_id=ws_id, audio_epoch=next_formal_epoch,
    base_prefix_version=stable_prefix_version,
    visible_audio_end_samples=total_received_samples,
    window_audio_samples=samples_since_last_formal,
    audio_pcm_f32le=audio_so_far.astype("<f4", copy=True).tobytes(),
    stable_prefix_tokens=tuple(stable_prefix_token_ids),
    rollback_tokens=tuple(previous_rollback_token_ids),
    deadline_ns=earliest_possible_next_commit_monotonic_ns,
)
decision = coordinator.maybe_start(
    request, live_load_snapshot, predicted_draft_p95_ms)

# 每一次权威流式 chunk 推理提交时，先尝试领取已就绪候选；
# 这可能发生在 commit 到达前，例如 seg-24/chunk-0。
chunk_candidate = coordinator.begin_authoritative_chunk(
    ws_id, audio_epoch=next_formal_epoch,
    prefix_version=stable_prefix_version,
    full_audio_end_samples=total_received_samples)
chunk_target_tokens = ()
try:
    chunk_draft = (chunk_candidate.token_ids if chunk_candidate
                   else previous_rollback_token_ids)
    chunk_result = await existing_stream_chunk_inference(
        ..., draft_tokens=chunk_draft)
    chunk_target_tokens = chunk_result.target_continuation_token_ids
finally:
    chunk_evaluation = coordinator.end_authoritative_chunk(
        ws_id, audio_epoch=next_formal_epoch,
        target_continuation_tokens=chunk_target_tokens)

# 正式 commit 到达、入队前立即调用；不得 await shadow：
candidate = coordinator.begin_formal(
    ws_id, next_formal_epoch, stable_prefix_version,
    total_received_samples, allow_candidate=not (short_skip or final_flush))
# shadow_only=True 时两个入口的 candidate 恒为 None，权威路径保持原样。
# 经过严格等价验证后才能用 shadow_only=False；候选只可经已有
# target verifier 验证，不能直接输出为识别结果。
draft = candidate.token_ids if candidate else previous_rollback_token_ids
result = await existing_authoritative_finish(..., draft_tokens=draft)
evaluation = coordinator.finish_formal(
    ws_id, next_formal_epoch, result.target_continuation_token_ids)

# WebSocket 退出时：
await coordinator.close_session(ws_id)
```

`audio_epoch` 是下一次正式边界的编号，`base_prefix_version` 在稳定前缀改变时递增。每个 `ShadowRequest` 自动生成独立 `request_id`，模型适配器应将其作为伪请求标识。`visible_audio_end_samples` 是会话绝对音频位置；权威 chunk 必须已经收到至少这么多音频。`deadline_ns` 要取**下一次权威 chunk 提交或正式 commit 可能最早发生**的时间，而不是固定假设还有 1 秒：短段提前 commit 时，过晚的伪请求会占用正式请求路径。`Evaluation.ready_lead_ms` 表示候选在权威 chunk/finish 开始前提前完成的时间。

记录 `chunk_evaluation`/`evaluation` 的 `continuous_accepted_tokens` 与 `ready_lead_ms`。同并发、同音频池分别做 baseline 与 `shadow_only=True` 的 probe，先核对候选是否赶上 chunk，再用 `streaming_prespec_monitor.py compare` 查看正式 `commit_latency_ms` 和段级 `service_queue_wait_ms` 的 P95/P99 差值。若 probe 的 P95/P99 已恶化，即使候选及时就绪也不能据此启用消费模式。

`LoadSnapshot` 应来自服务内及时读取的队列和设备状态：`formal_pending` 是服务层等待中的正式任务数，`waiting` 是 vLLM 等待数，`utilization_pct` 是目标 NPU 的利用率，`observed_ns` 使用同进程单调时钟且代表其中最旧读数，`formal_queue_wait_p95_ms` 是近期正式请求排队 P95。监测脚本约 2 秒一次的 `npu-smi` 数据适于分析，**不能直接冒充默认要求 200 ms 内的新鲜 admission 读数**。若取不到新鲜信号，不启动候选。

默认 `admission_mode="gated"`，每设备最多一个 shadow；正式队列非空、正式 commit 正在执行、同会话流式 chunk 正在推理、利用率超过 40%、读数过期或预测耗时加队列 P95/取消余量/安全余量赶不上 deadline 时跳过。任一会话的新权威 chunk 提交或正式 commit 到达，都会取消**尚未领取**的 shadow；若候选已在该 chunk 提交时就绪，可交给此 chunk 的 verifier，避免等到 commit 已无 finish 可做。受控实验可设 `AdmissionPolicy(admission_mode="force")` 比较“每个中点强制启动”的上界：它忽略队列、利用率和耗时预算，但仍拒绝已过 deadline、正在执行的正式 commit、同会话 chunk 和设备 shadow 名额已满。强制模式可能恶化正式请求时延，不应作为在线默认值。40%、100 ms 等默认阈值只是初始实验配置，需要用 910B 的实测不可抢占区间和 P95 成本校准。正式路径对尚未完成的 shadow 发取消并立即继续；**Python 取消信号本身不保证 Ascend kernel 已抢占**，模型适配器必须实现可确认的取消和资源释放，并在 probe 中检查正式 P95/P99 与队列等待。

## 验收顺序

1. 运行 `commit_overlap.py`，分别检查排队与同段流式 chunk、上一条正式 commit 的时间重叠。
2. 保持 `shadow_only=True`，记录 `LaunchDecision.reason` 和 `Evaluation.continuous_accepted_tokens`；核对候选是否赶上权威 chunk/finish 的开始、候选覆盖率及正式请求时延。
3. 只有候选及时就绪且同负载 P95/P99 不恶化、greedy token/KV/提交边界等价时，才打开 `shadow_only=False` 并经已有 verifier 消费候选。

本地验证：`python -m unittest discover -s research -p 'test_*prespec.py' -v`，以及 `python -m unittest discover -s research -p test_commit_overlap.py -v`。真机 NPU、vLLM 取消语义和模型严格等价仍需在 910B 服务代码中验证。
