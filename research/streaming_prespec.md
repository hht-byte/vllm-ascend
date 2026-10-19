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

## 服务端最小适配：一个推理包装函数 + 四组接入点

**推荐直接给 `PreSpecCoordinator` 传入 `producer(request, budget)`。** 它原本就支持普通异步函数。若现有回滚投机在一次请求里已经完成“验证草稿 → 拒绝处修正 → 继续 decode”，就复用这条完整链路；无需把 KV 暴露给协调器，也无需实现 `ShadowBackend` 的三个方法。

当前仓库实现的是调度/取消通知/候选检查/评估。它尚未连接你的服务模型调用。以下 `run_rollback_request`、`abort_if_present` 是**建议在服务里封装的函数名，不是现成的 vLLM API**；日志点来自你提供的源码位置，行号可能随修改变化。

### 1. 在 `qwen3_asr_async.py` 提取一次独立推理调用

从已有回滚推理路径提取“构造模型输入 → 提交 engine → 收集输出 token”这部分，供正式请求和 shadow 请求复用。这个函数接收音频、稳定前缀、草稿、request ID、输出上限；返回稳定前缀之后的完整生成 token。更新 session 文本/rollback 列表、推进 chunk 编号、发送 WebSocket 结果这些操作仍由正式调用方完成。

```python
# 接口示意：参数名映射到你已经实现的回滚投机入口。
async def producer(r, budget):
    try:
        tokens = await run_rollback_request(
            request_id=r.request_id,           # 独立的 prespec-... ID
            audio_pcm_f32le=r.audio_pcm_f32le,
            stable_prefix_tokens=r.stable_prefix_tokens,
            draft_tokens=r.rollback_tokens[:budget],
            max_new_tokens=budget,
        )
        return tuple(tokens)
    finally:
        # 复用服务已有的 engine abort/流关闭逻辑；已结束时必须可重复调用。
        await abort_if_present(r.request_id)
```

包装函数内部必须满足以下对应关系：

| 输入/输出 | 如何复用现有实现 |
|---|---|
| `stable_prefix_tokens=P0` | 用原 prompt 构造器设置当前稳定前缀；语言、模板、特殊 token 和音频上下文规则沿用正式路径 |
| `draft_tokens=U0` | 传给**现有 rollback verifier 的草稿入口**，不能把 U0 拼进不可修改的 prompt 前缀 |
| 中点音频 | 用它重新验证 U0；遇分歧后从修正状态继续 decode，整个过程可在同一个 engine 请求内完成 |
| 返回 `C1` | 包含实际接受的旧后缀、修正 token 和续写 token；不含 P0，不含被拒绝的旧后缀，不经过文本解码再编码 |
| 预算 | 限制 C1 的总 token 数（例如 8）；提前 EOS 正常结束，不强迫凑满 |
| request ID/资源 | 新请求有独立可变状态；不复用正式请求的可变 KV、输出缓冲或 session 对象；正常结束、取消、异常都释放资源 |

**这里的隔离通常由独立 engine 请求实现，不要求手工复制 KV。** 是否真的隔离，要检查你的回滚扩展有没有把草稿、KV 或采样配置挂到共享 session/全局对象上；仅改 request ID 不能隔离这些自定义状态。模型和 engine 可以共用，原生只读缓存按引擎规则复用。

如果当前函数只返回“验证接受了几枚 token”，它还不是上述完整推理函数：需要让原 decode 循环接着跑，并返回修正后的完整后缀。若当前回滚实现的草稿长度固定，候选预算也须服从该限制，不能直接扩大 verifier 的输入宽度。

### 2. 服务启动时创建协调器

复制 `streaming_prespec.py` 到服务的 Python 导入路径。每个共享 engine 的服务进程创建一个实例，所有连接共用；不要每条连接建一个。

```python
from streaming_prespec import AdmissionPolicy, PreSpecCoordinator, ShadowRequest

prespec = PreSpecCoordinator(
    producer,
    AdmissionPolicy(admission_mode="force", max_candidate_tokens=8,
                    max_active_shadows=1),
    shadow_only=True,
    on_probe=record_probe,   # 你的非阻塞日志函数，接收 CandidateProbe
)
```

这是**受控低并发 shadow 实验配置**：先复用已有旁路监测，不要求服务立即接入 200 ms NPU 采样。`force` 仍有设备 shadow 名额、同会话推理和正式 commit 活跃状态检查，但会跳过负载/耗时预算检查，可能影响正式延迟。多进程共享设备时这个名额不是全设备限流，首次实验应只启用一个 producer 进程。在线机会式调度后续使用默认 `gated` 模式。

### 3. 四组接入点放在哪里

| 位置 | 增加的操作 | 具体边界 |
|---|---|---|
| `qwen3_asr_async.py` 已解码音频的累计处 | `maybe_start(snapshot, ...)` | 上一轮正式推理已完成，新增音频达到 1 s，尚未达到下一正式 2 s 边界，且该段尚未 commit；调用不需要 await |
| 同文件 `inference submit` 前及这次推理的 `finally` | `begin_authoritative_chunk` / `end_authoritative_chunk` | 包住实际的正式流式 chunk 推理，提交前领取候选；结束后评估，异常也要配对退出 |
| `realtime_connection.py` 的 `Commit received` 路径及 `Segment done` 路径 | `begin_formal(..., allow_candidate=False)` / `finish_formal(..., ())` | 收到 commit、入队前标记并取消；完成/跳过/异常时配对退出。这两个位置在接收协程和消费协程中，不要写成收到 commit 后直接 await finish 推理 |
| WebSocket 清理处 | `await close_session(ws_id)` | 原有正式任务已停止、上述 finally 已执行后清理 shadow |

**先接通流式 chunk 的候选消费。commit 到达时不领取候选，finish 保持现有逻辑。** 短段提前 commit 会丢弃尚未领取的候选。你的 `seg-24` 在 commit 前已经启动 chunk-0，之后 skip finish，因此把候选只接在 finish 上无法加速这类样本。

第一版固定 `audio_epoch=seg`，**每段最多成功启动一次预投机**（当前协调器按 `(ws_id, audio_epoch)` 去重）。选择该段中第一个已有回滚草稿、且达到中点的窗口；例如 2 s 正式结果完成后，在累计到 3 s 时做预投机，供 4 s 的正式 chunk 使用。首个 1 s 若还没有 U0，就跳过；不要为凑草稿跨 VAD 段拼接旧 token。这个版本用于先验证链路；要覆盖同段内每个 2 s 窗口，还需按 chunk/window 分配 epoch 并独立回收，不能直接把 seg 和 chunk 编号混用。

因此，若 `seg-24/chunk-0` 是新段首次推理且没有合法 U0，上述最小版不会为这个仅 2.102 s 的段启动预投机；该日志证明了等待位置，不能证明这一次等待就能被本方案缩短。

中点请求的字段映射：

| 字段 | 服务端取值 |
|---|---|
| `session_id` / `audio_epoch` | `ws_id` / 当前 `seg` |
| `base_prefix_version` | 新增单调版本号；有效稳定前缀、语言/prompt 配置或音频上下文起点变化时递增；shadow 不递增它 |
| `stable_prefix_tokens` / `rollback_tokens` | 上一次正式结果按现有回滚规则拆出的 P0 / U0 的 tuple 快照 |
| `audio_pcm_f32le` | 按正式输入规则截取**截至此刻已到达的完整音频上下文**，复制为 16 kHz mono float32 小端 bytes；不是传输层 base64/PCM16，也不一定仅有新增的 1 s |
| `visible_audio_end_samples` | 快照的会话绝对采样截止位置；跨段继续计数 |
| `window_audio_samples` | 自上次正式输入截止位置以来新增的采样数，达到 16000 才触发；不是整个累计 buffer 的长度 |
| `deadline_ns` | 同进程 `monotonic_ns` 估计的下一正式 chunk 提交时刻；只是预算依据，不能保证 VAD 不提前 commit |

在同一事件循环中一次性取得音频、P0、U0、版本和截止位置，快照过程中不 await；后续不得跟随真实 session 更新。语言等 `ShadowRequest` 未携带的条件，也须通过请求快照/固定配置保持一致。

```python
# 已构造上述不可变 snapshot，且满足中点条件时；别在 WS 接收协程
# 调用同步模型推理，也别将伪音频/伪 None 塞入正式段级音频队列。
# 仅 force 实验可不提供负载，predicted_draft_ms=0 表示未启用耗时预算。
decision = prespec.maybe_start(snapshot, None, predicted_draft_ms=0)
record_launch(decision)
```

正式 chunk 的接法（`existing_stream_chunk_inference` 表示你的现有函数）：

```python
candidate = prespec.begin_authoritative_chunk(
    ws_id, audio_epoch=seg, prefix_version=stable_prefix_version,
    full_audio_end_samples=chunk_audio_end_samples,
)
target_tokens = ()
succeeded = False
try:
    result = await existing_stream_chunk_inference(
        # 保留现有其他参数；只替换“草稿来源”。
        draft_tokens=(candidate.token_ids if candidate is not None
                      else previous_rollback_token_ids),
    )
    # 与 snapshot 的 P0 同一起点，包含本次完整生成后缀；在业务回滚前取。
    target_tokens = result.target_continuation_token_ids
    succeeded = True
finally:
    evaluation = prespec.end_authoritative_chunk(
        ws_id, audio_epoch=seg, target_continuation_tokens=target_tokens)
    if succeeded and evaluation is not None:
        record_evaluation(evaluation)
```

`full_audio_end_samples` 必须是**实际提交给这次推理的输入截止位置**，不能用接收端已收到但还没被本次请求消费的更大值。正式输出引起的版本更新按原业务流程完成；校验候选时使用提交前的版本。`shadow_only=True` 时 candidate 恒为 None，原回滚草稿照常使用，协调器仍评估就绪候选与正式输出的连续匹配长度。

commit 的接法分在两处，每个 seg 各一次：

```python
# 接收协程：commit 到达，放入 None 之前；不要在这里 await shadow。
prespec.begin_formal(ws_id, seg, stable_prefix_version,
                     total_received_samples, allow_candidate=False)
# 接着执行原有入队逻辑；若入队失败，也要配对 finish/清理。

# 消费协程：Segment done 或该段退出的 finally，含短音频跳过路径。
# 本版 commit 仅负责忙闲标记，候选匹配已在 chunk 结束时统计。
prespec.finish_formal(ws_id, seg, ())
```

不要在 `commit consumed` 再次 `begin_formal`。`skip finish` 也不代表整个段没有推理；它之前的 chunk 照常计入。`close_session` 在原正式任务退出后调用，避免先清掉计数再执行正式任务的 finally。

### 4. 如何确认真的接通

- 启动预投机时记录 `(ws_id, seg, request_id, decision.reason)`，engine 日志中应出现独立的 `prespec-...` 请求；仅复制文件或运行监测脚本不会产生请求。
- `on_probe` 的 `ready_shadow_only` 表示在正式 chunk 开始前已有候选；`pending` 表示没赶上并已发取消；`version_mismatch` / `audio_mismatch` 表示快照或取值不一致。`previously_used` 在本版同段后续 chunk 上可能正常出现。
- `Evaluation.continuous_accepted_tokens` 是候选与独立正式输出的连续 token 匹配长度，shadow 模式下不是引擎真实接受数；异常 chunk 不计匹配率。
- 用同负载 baseline/probe 对比正式 P95/P99 和队列等待。Python task 被取消不代表 Ascend kernel 已抢占，确认模型 abort 生效与正式延迟仍须在 910B 上测。

确认及时就绪、同负载延迟可接受，并在离线回放验证候选经 verifier 消费后与原路径逐 token/提交边界等价，再在线设 `shadow_only=False`。此时上面的 chunk 代码自动把 C1 送入**同一个已有 rollback verifier**，用完整正式音频再次验证；不会直接提交 C1。初次实验始终保留原 2 s 正式触发规则。

### 可选接口与后续 admission

`RepairExtendProducer(backend)` 适用于你的模型代码本来就把 verify 和 decode 分成两个可调用阶段的情形：`verify_and_repair_old` 返回 `VerifiedRollback(token_ids, continuation_state)`，`extend_from_verified` 从同一修正后状态续写，`close` 清理私有资源。只有这条分阶段接口要求把不透明状态对象交回适配器；前面的单函数接法没有这项要求。

后续启用默认 `admission_mode="gated"` 时，`LoadSnapshot.waiting` 是 vLLM waiting 请求数，`formal_pending` 是服务等待的正式任务数，均不是包含 None 的段级音频队列大小。`utilization_pct` 来自目标 NPU；`observed_ns` 是同进程单调时钟下最旧读数的时间；`formal_queue_wait_p95_ms` 和 `predicted_draft_ms` 用实测成本。约 2 秒一次的 `npu-smi` 旁路数据不能冒充默认 200 ms 内的新鲜读数，取不到则不启动。40%、100 ms 等默认阈值需按实测校准。

本地协调器验证：`python -m unittest discover -s research -p 'test_*prespec.py' -v`。这些测试没有执行你的服务适配器或 NPU，不能替代 910B 上的请求取消、严格等价和时延验证。
