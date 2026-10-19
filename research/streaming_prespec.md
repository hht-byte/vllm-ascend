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

## 按三个服务文件接入（2026-10-19 修订）

依据：你提供的 `RealtimeConnection`、`websocket_router`、`Qwen3ASRModelAsync` 函数职责，以及本仓库的协调器实现。服务源码尚未在本仓库中，因此下文是**逐函数改动方案，不是已合入服务的补丁**。新增 helper 的名称是建议名；external draft 的具体字段、engine 输出格式沿用你的已有实现。

核心改动集中在 `qwen3_asr_async.py`：复用现有 external draft 构造和 engine 的 verify → correction → decode 链路。连接层只通知中点/commit/清理；路由层复用已经注入的 ASR 实例。仍直接使用 `PreSpecCoordinator(producer, ...)`，无需为了接入实现分阶段 `ShadowBackend`/KV 接口。

### 一、逐函数改动清单

| 文件 / 现有函数 | 具体改动 |
|---|---|
| `websocket_router.py`：`set_asr()` / `websocket_endpoint()` | 继续传递同一个 ASR 实例，通过 `asr.prespec` 访问共享协调器，无需增加路由或每连接实例 |
| 同文件 `set_streaming_params()` | 原有参数继续生效；确保中点阈值与 `chunk_size_sec` 一致。当前 2 s 配 16000 samples；不能改成其他步长后仍静默套用 1 s 中点 |
| `qwen3_asr_async.py`：`AsyncLLM()` | 在工厂构造出的 ASR 实例上按开关设置一个 `prespec`，producer 绑定该实例的新增 `_run_prespec()`；候选上限服从此处读取的 `kmax` |
| 同文件 `init_streaming_state()` | 初始化前缀版本等轻量元数据；创建 state 后由连接层绑定 `ws_id/seg_id`，不要依赖从 engine request ID 反查 |
| 同文件 `async_streaming_transcribe()` | **每个正式 chunk 内部**，在准备好正式音频/P0、尚未把 external draft 写进请求时领取候选；实际 engine 输出迭代结束后配对退出 |
| 同文件 `_yield_delta_text()` | 正式输出流程保留；若外层拿不到原始 token IDs，增加一个可选的“最终 token 回调”，用于评估；shadow 不调用此函数 |
| 同文件 `async_finish_streaming_transcribe()` | 第一版保留原 finish 路径；commit 到达已取消 shadow 并标记正式任务，finish 不重复领取候选 |
| `realtime_connection.py`：`_process_segment()` | 每次 `async for ... async_streaming_transcribe(...)` 完整结束后尝试中点；最外层 `finally` 配对结束该段的 commit 标记 |
| 同文件 `handle_event()` | append 保留解码入队；commit 在注入 None 前通知协调器，并保存一次配对标志；`session.update` 沿用原语义，若影响活跃段则使旧快照版本失效 |
| 同文件 `cleanup()` | 先取消并等待生成任务退出，再清理该连接 shadow；等待期间仍须让 chunk/segment 的 finally 执行 |

`handle_connection()`、`_run_generation()`、`_send_stream_item()`、`send()`、`async_transcribe()`、`_infer_asr_vllm_async()` 不需要为最小版增加预投机逻辑。`_finish_segment()` 的短尾跳过、done 统计及协议事件保持原流程；清理标记放外层 finally，才能覆盖到达 finish 之前的异常。

### 二、中点触发：放在 `_process_segment()` 消费完一次生成器之后

之前示例中的 `await existing_stream_chunk_inference(...)` 不能直接对应你的流式 API：`async_streaming_transcribe()` 是异步生成器，一次调用可能执行多个 chunk。连接层应这样挂中点检查（参数名仅示意）：

```python
# _process_segment() 原有“未提交、逐块处理”路径
async for item in self.asr.async_streaming_transcribe(audio, state):
    await self._send_stream_item(item)

# 新增同步 helper；内部只做检查/快照和 maybe_start，不等待 shadow。
self.asr.try_start_prespec(
    state, ws_id=ws_id, seg_id=seg_id,
    committed=segment.committed,
    audio_queue_empty=segment.audio_queue.empty(),
)
```

`try_start_prespec()` 放在模型类中，读模型的 buffer/P0/U0；它仅在以下条件满足时调用现有 `maybe_start()`：

1. segment 未 commit、当前段音频队列无积压、上一轮正式推理已结束。commit 后合并剩余音频的路径不启动 shadow。
2. 剩余 buffer 位于 `[中点, chunk_size_samples)`；对 2 s chunk，就是 `[16000, 32000)`。buffer 是尚未进入正式 chunk 的新增音频，而非整个累计音频长度。
3. 按**下一次正式请求的现有回滚规则**可以得到合法 U0。包括 `unfixed_chunk_num` 的预热条件、`unfixed_token_num` 和 `kmax`，不要另写固定“最后 5 个 token”的拆分。
4. 当前段尚未成功启动过预投机；正式队列、负载、剩余时间是否允许，再由协调器判定。

这里检查队列是防止消费协程追赶积压时制造“伪空档”：如果下一正式 chunk 所需音频已经在队列里，只是还没取出，就不应再起 shadow。段级队列非空是这一额外跳过条件，不能把它当作 `LoadSnapshot.waiting`（vLLM 等待请求数）。

预算按**尚缺的采样数**估计：`(chunk_size_samples - buffer_samples) / 16000`，结合实际接收节奏；不能在完成上一次推理后重新假定还有整整 1 s。VAD 仍可能提前 commit，取消只是尽力让路，不保证 NPU kernel 立即退出。

### 三、模型侧新增的三个小 helper

| 建议 helper | 责任 |
|---|---|
| `_build_stream_request(...)` | 从原 `async_streaming_transcribe()` 提取下一次请求的 P0/U0、prompt/音频视图和 external draft 打包逻辑；显式传入草稿和缓存策略，不改真实流式状态 |
| `try_start_prespec(...)` | 上节的中点检查、不可变快照、调用 `self.prespec.maybe_start()` |
| `_run_prespec(snapshot, budget)` | 用独立 `prespec-...` request ID 调用同一个 engine，只收集候选 token，取消/异常时 abort 并关闭流 |

helper 的职责可以复用现有函数，不要求机械增加三个类或三套输入类型。优先只提取请求构造的小段代码，正式路径的 delta 处理继续使用 `_yield_delta_text()`。

shadow 的单次 engine 请求必须做完：

```text
中点音频 + 稳定前缀 P0 + external draft U0
    → 现有 verifier 验证 U0
    → 首个分歧处修正
    → 从修正后的状态继续 decode，直到预算或 EOS
    → C1 = 接受的旧后缀 + 修正 token + 新生成 token
```

U0 进入**现有 external draft 字段**，不拼进不可修改的 prompt。C1 不含 P0、不含被拒绝的旧 token。候选预算初始取 `min(实验预算, kmax)`，`kmax<=0` 或现有回滚路径不生效时跳过；不能照搬固定 8。SamplingParams 及 external draft 容器要按请求复制后修改，不能原地改共享配置。若 engine 输出是累计 token IDs，保存最终累计值；不能每次 `extend()` 导致重复拼接。

**shadow 不调用 `async_streaming_transcribe()`、`async_finish_streaming_transcribe()` 或 `_yield_delta_text()`。** 前两者修改 buffer/audio_accum/正式状态，后者计算已发送 delta、写全局统计并释放窗口缓存 session。shadow 自己消费 engine 输出，不 yield `ChunkInferenceEvent`/delta/done、不写回真实 state；单独记录请求类别和统计。引擎自身的全局 spec decode 计数也可能包含 shadow，baseline/probe 比较时不能当成纯正式请求接受率。

#### 窗口缓存：第一版给 shadow 选现有全量 prompt 分支

按你的描述，服务已有窗口缓存/全量 prompt 两条构造路径。第一版为 shadow 显式选择全量路径，formal 保持原配置；不要临时关闭 `self` 上的全局缓存开关，否则会影响并发正式请求。两条路径的音频上下文、语言和 P0 必须在语义上对应。

这样 shadow 不操作正式窗口缓存会话，也不需要调用它的 `release_session()`。独立 engine request ID 不能自动隔离按 session 键保存的缓存元数据：本仓库 `WindowedRequestAdapter.build_request()` 会更新 `_states[session_id]`，`release_session()` 会删除该键。你的部署版本需核对；不能把 shadow 的释放指向正式 session。

后续若测得重复编码成本过高，再给 shadow 配独立 cache session/namespace，并验证自己的释放只影响自己。全量路径的额外 encoder/prefill 成本必须计入实验结论；这不是零成本复用正式 KV。

#### 快照具体包含什么

| 内容 | 从哪里取 / 规则 |
|---|---|
| `session_id / audio_epoch` | 最小版固定 `ws_id / seg_id`；见后面的每段一次限制 |
| P0 / U0 / `base_prefix_version` | 从共享的下一正式请求构造逻辑取；正式结果、有效 prompt/语言/context 或音频上下文起点变化时更新版本 |
| 音频 | 按正式音频视图，把已进入累计上下文及当前剩余 buffer 正确组合一次，再复制为 16 kHz mono float32 小端 bytes；不能只拿新增 1 s，也不能重复拼接已在 audio_accum 的部分 |
| 音频截止位置 | 快照实际包含的会话绝对采样位置；正式领取时比较实际送入该请求的位置，不能用更大的接收端计数 |
| `window_audio_samples / deadline_ns` | 尚未进入正式 chunk 的样本数 / 同进程单调时钟的预计下一正式提交时刻 |
| 模型条件 | 冻结当前段的语言、context、prompt 及请求级采样条件；processor、模型常量可共享 |

`ShadowRequest` 目前只有通用音频/token/版本字段，**没有语言/context/prompt 字段**。服务侧须用扩展快照携带实际请求构造器读取的这些条件，例如继承 `ShadowRequest` 并添加不可变字段，再由 `_run_prespec()` 读取；不能后台启动后再从活跃 session 取可变条件，也不要 deepcopy 整个 ASR state/engine。

### 四、正式 chunk：在函数内部替换草稿来源

挂点在 `async_streaming_transcribe()` 的**每次切 chunk 推理循环内部**：正式音频和 P0 已确定，external draft 尚未打包。只在外面包一次函数调用会漏掉同次调用的后续 chunk。

```python
# 内部每个正式 chunk：变量/字段名示意，不是完整服务补丁。
candidate = self.prespec.begin_authoritative_chunk(
    ws_id, audio_epoch=seg_id, prefix_version=prefix_version,
    full_audio_end_samples=this_chunk_audio_end,
)
raw_target_tokens = ()
try:
    draft = candidate.token_ids if candidate is not None else original_rollback
    # 用 draft 构造现有 external draft，然后执行原 engine.generate。
    # 这里保留原来的 async for ... _yield_delta_text(...) 与 yield item。
    # 保存这次实际生成的、P0 之后且业务回滚之前的原始 token IDs。
finally:
    evaluation = self.prespec.end_authoritative_chunk(
        ws_id, audio_epoch=seg_id,
        target_continuation_tokens=raw_target_tokens,
    )
```

仅推理成功时将返回的 Evaluation 计入匹配率。若现有局部变量拿不到 raw token，在 `_yield_delta_text()` 增加可选 `on_final_tokens` 回调，交回最终原始 IDs；不要把异步生成器改成 await 函数，也不要从 delta 文本重新 tokenize。try/finally 必须覆盖实际输出迭代；清理调用者时显式关闭/退出生成器，使 finally 及时执行，不能依赖垃圾回收。

`shadow_only=True` 时 candidate 恒为 None，正式路径仍用原 U0，但协调器会比较就绪 C1 与正式原始输出。切换消费模式后，C1 仍经过现有 verifier 在完整正式音频上重新验证。推理失败/中断的空 token 只用于释放生命周期，不计为候选拒绝率。

### 五、commit 与清理：连接层配对，不放进 delta helper

- `handle_event()` 处理真实 commit 时，在注入 None 之前调用 `begin_formal(ws_id, seg_id, version, received_samples, allow_candidate=False)`，随后置 `segment.prespec_formal_begun=True`。重复 commit 不能重复 begin；state 尚未初始化时也要标记忙状态，版本参数可用段的初始值，因为不领取候选。
- `_process_segment()` 最外层 finally 检查上述标志，调用 `finish_formal(ws_id, seg_id, ())` 并清标志。此处包住流式循环与 `_finish_segment()`，所以短尾跳过、正常完成和中途异常都会配对。
- 如果某段已 begin 但入队失败或被取消前根本未进入 `_process_segment()`，由对应异常处理或连接 cleanup 清理；不能期待一个从未进入的 finally 自动执行。
- `cleanup()` 先取消并等待生成任务及其流式生成器退出，再 `await asr.prespec.close_session(ws_id)`，最后做原有队列/统计清理。不要先删协调器计数再让模型 finally 退出。

`_finish_segment()` 的 `<=0.4s` 跳过条件保持原样。commit 已到达时的 residual finish 第一版不消费候选；这也是为什么正式 chunk 的接入必须放在 `async_streaming_transcribe()` 内，而不能只改 finish。

### 六、最小版范围与验证

当前协调器按 `(session_id, audio_epoch)` 去重，最小版继续 `audio_epoch=seg_id`，**每段最多成功启动一次**。选择段内第一个已有合法 U0 的中点，例如 2 s 正式结果之后，累计到 3 s 时为 4 s chunk 起草。没有合法 U0 的首个窗口跳过；不能跨 VAD 段借用旧 token。`seg-24/chunk-0` 若是新段首次推理，本版不能据其等待日志声称可获得预投机收益。

要覆盖每个 2 s 窗口，下一步需把 `(ws, seg, chunk/window)` 候选 epoch 与段级 commit 生命周期拆开，并在每次 chunk 结束回收对应 epoch。**仅把 `audio_epoch=seg` 改成 `chunk_id` 不够**：chunk 可能跨段重号，且现有 finish_formal 不会回收所有窗口。此改动不属于本次文档适配，当前代码仍是每段一次接口。

首次用共享的单个协调器、一个 active shadow、`shadow_only=True`、预算不超过 kmax。受控低并发实验可用 `admission_mode="force"` 和现有旁路监测，先打通链路；它不保证不干扰正式时延。在线 `gated` 模式仍需新鲜队列/利用率和实测耗时预算，约 2 s 的 `npu-smi` 读数不能冒充默认 200 ms 的 admission 读数。

验收只保留直接相关的检查：

1. 中点出现 `prespec-...` 请求；确认旧草稿有拒绝时，后续 decode 从 correction 状态继续。
2. 正式 chunk submit 的 `ready_shadow_only/pending`、候选连续匹配长度与成本分开记录；shadow 不产生客户端事件、不改正式 state、不释放正式缓存。
3. 提前 commit、短尾 skip、断连时，无泄漏请求和未配对计数；注意 Python 取消不等于 NPU kernel 已抢占。
4. 同音频/同并发 baseline 与 probe 比较正式 P95/P99、队列等待及总计算成本；在离线回放验证经 verifier 消费后的 token/提交边界等价，再启用消费模式。

服务源码到位后需确认的具体代码点只有：external draft 的打包字段及 kmax 约束、P0/U0 的实际拆分、raw output token 的含义、buffer/audio_accum 是否重叠、engine abort/生成器关闭方式、窗口缓存释放的实际键和范围。当前描述足以确定挂点，但不能替代这些逐行核对或 910B 验证。
