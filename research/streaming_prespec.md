# 流式音频中点预投机：服务端接入件

本目录提供以下工具：

- `commit_overlap.py` 用现有逐 commit JSONL/CSV 检查前一条正式 commit 的重叠；若提供服务 DEBUG 日志，还检查**同一段内流式 chunk 推理与 commit 等待**的重叠。
- `streaming_prespec.py` 管理中点 shadow 请求、低负载 admission、候选版本、正式边界取消和 shadow 候选质量评估。
- `streaming_prespec_helpers.py` 提供三个模型 helper 的实现；服务只需接入原有请求打包代码，并映射本地音频/token 字段，见第四节。

这些文件可以复制到 910B 服务项目，模块导入不依赖 vLLM/Ascend。当前仓库没有运行中的 `realtime_connection.py` 和 `qwen3_asr_async.py`，实际请求打包仍需接到服务已有的回滚投机实现；**只有协调器、没有模型接线时不会发送任何伪请求**。

## 先验证“上一轮推理尚未完成”的解释

```bash
python commit_overlap.py --commits baseline-c20-commits.jsonl \
  --service-log concurrency_20.log --log-utc-offset +00:00 \
  --out baseline-c20-overlap.json
```

`previous_inference_overlap_count` 只比较同一 `ws_id` 前一条非跳过 commit 的 `done_at` 与本条 `received_at`；它会漏掉“当前 `seg` 的 `chunk-0` 仍在 decode”的情形。`--service-log` 从同一 `(ws_id, seg, chunk)` 的 `inference submit` 到 DEBUG `token_count` 构造推理区间，计算它与 `[received_at, received_at + queue_wait_ms]` 的交集。重点看 `same_segment_chunk_overlap_count`、`same_segment_chunk_overlap_ms` 和 `queue_wait_ms_with_chunk_overlap`。日志时间若是北京时间，传 `--log-utc-offset +08:00`；与 commit JSONL 的 UTC 时间不一致会导致错误对齐。缺少 DEBUG 结束日志时 `same_segment_chunk_unknown_count` 会增加，不能把未检出的重叠当作 0。时间重叠支持等待原因的判断，但不能单独证明所有排队都由该 chunk 造成。

服务的 `Commit received: queue_size=N` 是把 `None` 哨兵放入当前段音频队列**之后**的大小，因此 `N-1` 是尚待协程取走的音频块数；这不是 vLLM waiting 请求数。`ws-cea6c68c/seg-24` 的 `queue_size=3` 表示两个音频块加一个哨兵。commit 在 `chunk-0` 尚未结束时到达，112 ms 的等待与该 chunk 推理重叠；协程之后还要消费剩余音频块，再取到哨兵。该样本不能把 118 ms 全部归因于 vLLM 排队。

## 按三个服务文件接入（2026-10-20 修订）

依据：你提供的 `RealtimeConnection`、`websocket_router`、`Qwen3ASRModelAsync` 函数职责，以及本仓库的协调器实现。服务源码尚未在本仓库中，因此下文是**逐函数改动方案，不是已合入服务的补丁**。新增 helper 的名称是建议名；external draft 的具体字段、engine 输出格式沿用你的已有实现。

核心改动集中在 `qwen3_asr_async.py`：复用现有 external draft 构造和 engine 的 verify → correction → decode 链路。连接层只通知中点/commit/清理；路由层复用已经注入的 ASR 实例。仍直接使用 `PreSpecCoordinator(producer, ...)`，无需为了接入实现分阶段 `ShadowBackend`/KV 接口。

**特性边界：预投机与窗口缓存相互解耦。** 本方案不新增窗口缓存的开关、适配器、namespace 或释放逻辑，也不以窗口缓存开启或关闭作为接入条件。

**音频输入合同：正式请求和预投机请求都提交截至各自时刻的完整累计音频。** 根据你给出的 `async_streaming_transcribe()`，每轮正式推理从 `state.buffer` 取够一个 chunk，追加到 `state.audio_accum`；中点的新增音频尚在 `state.buffer`，因此 shadow 输入必须是当时的 `state.audio_accum + state.buffer`。新增样本数只决定何时尝试；KV 复用由 vLLM-Ascend 判断。

### 一、先完成实例接线：这里给出实际新增代码

#### 步骤 1：复制模块并添加 import

把本仓库的 `research/streaming_prespec.py` 和 `research/streaming_prespec_helpers.py` 一起复制到服务项目的 `qwen_asr/inference/`。在 `qwen3_asr_async.py` 顶部新增：

```python
from qwen_asr.inference.streaming_prespec import (
    AdmissionPolicy,
    PreSpecCoordinator,
)
from qwen_asr.inference.streaming_prespec_helpers import (
    PreSpecModelHelpers, snapshot_stream_audio,
)
```

把现有类声明改为 `class Qwen3ASRModelAsync(PreSpecModelHelpers, Qwen3ASRModel):`，原有类体保留。这个 mixin 没有构造函数，提供第四节的模型方法及完整音频快照函数；若此前自行添加过同名占位方法，应替换它们，避免覆盖 mixin 的实现。

下面的代码分为两类：**新增代码**可放入指定位置；**原代码形态**只用于定位，因为没有你的函数原文。原有变量、构造参数不能用示例中的省略号替换。

#### 步骤 2：在模型类中新增 `enable_prespec()` 方法

放在 `class Qwen3ASRModelAsync` 内，与 `async_streaming_transcribe()` 同一级缩进。这是本方案新增的方法，不是现有 API：

```python
def enable_prespec(self, *, enabled, kmax, engine=None, chunk_size_sec=2.0,
                   candidate_tokens=None, on_probe=None,
                   admission_mode="force", shadow_only=True):
    # 仅在服务启动、连接尚未创建时调用一次。
    self.prespec = None
    if not enabled:
        return
    if kmax <= 0 or chunk_size_sec <= 0:
        raise ValueError("prespec requires positive kmax, budget and chunk size")
    if candidate_tokens is None:
        candidate_tokens = kmax + 1
    if candidate_tokens < 3:
        raise ValueError("candidate budget needs first token, draft and continuation")
    midpoint_samples = int(chunk_size_sec * 16000 / 2)
    if midpoint_samples <= 0:
        raise ValueError("prespec midpoint must contain at least one sample")
    if engine is None or not callable(getattr(self, "_pack_stream_request", None)):
        raise ValueError("prespec requires engine and the existing request packer")
    self.prespec_engine = engine  # 引用已创建的同一个 engine，不创建第二个

    self.prespec = PreSpecCoordinator(
        producer=self._run_prespec,
        policy=AdmissionPolicy(
            admission_mode=admission_mode,
            midpoint_samples=midpoint_samples,
            max_rollback_draft_tokens=kmax,
            max_candidate_tokens=candidate_tokens,
            max_active_shadows=1,
        ),
        shadow_only=shadow_only,
        on_probe=on_probe,
    )
```

这个方法创建协调器并保存 engine 引用。`admission_mode="force"`、`shadow_only=True` 是初次受控实验的默认值；验证手册使用启动 parser 参数显式覆盖。`kmax` 是引擎 proposer 能接收的 external draft 上限，`candidate_tokens` 是 shadow 请求的 `max_tokens`，不再用 `min(candidate_tokens, kmax)` 混成一个预算。默认候选预算为 `kmax+1`：第一个位置由 target 采样，后面最多 `kmax` 个候选 token 可供正式请求提交。旧草稿加首 token、至少一个续写 token 放不下时会跳过。显式设置更大的候选预算时，若最终可提交的候选后缀超过 `kmax`，正式边界会返回 `candidate_exceeds_kmax`，避免 proposer 再次静默截断。`self._run_prespec` 已由 mixin 提供；第四节的 `_pack_stream_request` 接到已有实现前仍使用 `enabled=False`。`on_probe` 可传现有非阻塞日志回调；未传时不会自动输出 probe 日志。

例如引擎 `kmax=8`、默认候选预算 9 时，旧回滚草稿最多 7 个 token 才能同时给首 token 和至少一个新续写留位置；长度恰为 8 时返回 `candidate_budget_insufficient`。把预算改为 10 虽可启动这条 shadow，但若它真输出 10 个 token，正式可提交后缀长 9，仍会以 `candidate_exceeds_kmax` 跳过。要让完整旧草稿和新续写都可提交，必须增大引擎的 `kmax` 或缩短原回滚草稿；不能在服务端静默截断旧草稿。

#### 步骤 3：修改 `AsyncLLM()` 的返回位置

搜索工厂方法末尾创建并返回 `Qwen3ASRModelAsync` 实例的位置。若原来直接返回构造结果：

```python
# 原代码形态：括号内是你原有的完整构造参数。
return cls(...)
```

改成下面的结构，**保留原构造参数，只把直接 return 拆成三步**：

```python
model = cls(...)  # 原有构造参数逐字保留
model.enable_prespec(
    enabled=False,       # 全部模型适配完成后，受控实验再改成 True
    kmax=kmax,           # 对应你从 speculative config 读出的现有局部变量
    engine=engine,      # 对应本函数已创建的 AsyncLLMEngine 实例变量
    chunk_size_sec=2.0,  # 必须与 set_streaming_params 的实际值一致
)
return model
```

若原代码已是 `model = ...; return model`，只在 `return model` 前插入 `model.enable_prespec(...)`。`AsyncLLM()` 是类方法，不能在这里写 `self.prespec = ...`。这里的 `kmax` 是待映射到现有局部变量的名称，不是要求再读一次配置。初始化必须在 `set_asr(model)` 之前完成；不要运行中再次调用此方法，否则连接持有的旧协调器引用不会自动更新。

上面的 `enabled=False` 是完成适配前的静态安全起点。A/B/C 验证时，按 [验证手册第 1.1 节](streaming_prespec_validation.md) 从现有启动 parser 的 `--prespec-mode`、`--prespec-admission` 传到 `AsyncLLM()`，再调用本方法；不要同时保留这个写死的 `enabled=False` 调用，也不要把两个新参数直接传给 vLLM engine 构造函数。

你当前的工厂调用是 `Qwen3ASRModelAsync.AsyncLLM(**base_config)`；验证手册给出的实际改法是在这一次调用中追加 `prespec_mode=args.prespec_mode`、`prespec_admission=args.prespec_admission`、`prespec_candidate_tokens=args.prespec_candidate_tokens`，并由 `AsyncLLM()` 的显式形参接收。`base_config` 保持原用途，不向 vLLM engine 透传这些服务侧参数。

#### 步骤 4：`websocket_router.py` 不修改

明确到函数：

- `set_asr()`：**新增 0 行**，保留原来保存 ASR 实例的代码。
- `websocket_endpoint()`：**新增 0 行**，保留原来把该实例交给 `RealtimeConnection` 的代码。
- `set_streaming_params()`：**新增 0 行**；把步骤 3 的 `chunk_size_sec` 配成与这里相同的值。两处已有共同配置变量时直接复用该变量。

原因是步骤 3 已经把 `prespec` 放在传入路由的 ASR 对象上；路由只传递已有对象即可。之前“通过 asr.prespec 访问共享协调器”的表述指的是**下一步的连接类代码**，并不是要求在路由函数中调用 `asr.prespec`。

#### 步骤 5：在 `RealtimeConnection.__init__()` 中保存一个引用

定位原来保存 ASR 实例的赋值。若原来是：

```python
self.asr = asr
```

就在它后面新增一行：

```python
self.asr = asr  # 原有赋值
self.prespec = getattr(asr, "prespec", None)  # 新增，引用同一个协调器
```

如果实际成员名是 `_asr`，保留 `_asr`，只加入第二行；右侧 `asr` 对应现有构造函数接收到的 ASR 参数。不要在连接构造函数里调用 `PreSpecCoordinator(...)` 或 `enable_prespec()`。

接线后的关系为：

```text
AsyncLLM() 构造 model，并设置 model.prespec
    → set_asr(model) 保存这个 model
    → websocket_endpoint 把这个 model 交给每个 RealtimeConnection
    → connection.prespec 引用 model.prespec
```

此时可以验证：启用时 `connection.prespec is connection.asr.prespec` 为 True；禁用时两者都为 None。这一步只证明共享关系正确，尚不证明产生过 shadow 请求。

### 二、其余函数：明确哪些要改、哪些不改

| 文件 / 现有函数 | 具体改动 |
|---|---|
| `websocket_router.py`：三个函数 | **不改代码**；具体见步骤 4 |
| `qwen3_asr_async.py`：`AsyncLLM()` | **插入一处调用**：在返回实例前调用新增 `enable_prespec()`，具体见步骤 2–3 |
| 同文件 `init_streaming_state()` | 初始化 `state.prespec_prefix_version=0`；流式函数已有第三个参数 `connection_id` 可作 `ws_id`，只需另传当前 `seg_id`，具体见第五、六节；`audio_accum` 和 `buffer` 沿用原有字段 |
| 同文件 `async_streaming_transcribe()` | **每个正式 chunk 内部**，在准备好正式音频/保留前缀、尚未把 external draft 写进请求时领取候选；实际 engine 输出迭代结束后配对退出 |
| 同文件 `_yield_delta_text()` | 正式输出流程保留；若外层拿不到原始 token IDs，增加一个可选的“最终 token 回调”，用于评估；shadow 不调用此函数 |
| 同文件 `async_finish_streaming_transcribe()` | 合并 `state.buffer` 到完整音频后、构造 `finish_...` 请求前，领取 commit 时保留的就绪候选；完整音频和当前前缀仍由正式请求验证 |
| `realtime_connection.py`：`_process_segment()` | 每次 `async for ... async_streaming_transcribe(...)` 完整结束后尝试中点；最外层 `finally` 配对结束该段的 commit 标记 |
| 同文件 `handle_event()` | append 保留解码入队；commit 在注入 None 前通知协调器，并保存一次配对标志；`session.update` 沿用原语义，若影响活跃段则使旧快照版本失效 |
| 同文件 `cleanup()` | 先取消并等待生成任务退出，再清理该连接 shadow；等待期间仍须让 chunk/segment 的 finally 执行 |

`handle_connection()`、`_run_generation()`、`_send_stream_item()`、`send()`、`async_transcribe()`、`_infer_asr_vllm_async()` 不需要为最小版增加预投机逻辑。`_finish_segment()` 的短尾跳过、done 统计及协议事件保持原流程；正常 finish 调用需把当前 `seg_id` 传入模型。清理标记放外层 finally，才能覆盖到达 finish 之前的异常。

### 三、中点触发：在本次 `async_streaming_transcribe()` 生成器退出后

之前示例中的 `await existing_stream_chunk_inference(...)` 不能直接对应你的流式 API：`async_streaming_transcribe()` 是异步生成器，一次调用可能执行多个 chunk。保留原有第三个位置参数 `self.connection_id`；它就是 `ws_id`。连接层在这次调用结束后挂中点检查（`seg_id` 表示当前段号，须对应现有 `seg-N` 字段）：

```python
# _process_segment() 原有“未提交、逐块处理”路径
item = await segment.audio_queue.get()
if item is None:
    continue
async for stream_item in self.asr.async_streaming_transcribe(
    item, segment.asr_state, self.connection_id,
    prespec_seg_id=seg_id,  # 唯一新增的流式函数实参
):
    await self._send_stream_item(stream_item)

window_id = segment.asr_state.chunk_id  # 下一次正式 chunk 的 ID
if (self.prespec is not None
        and segment.prespec_started_window != window_id):
    decision = self.asr.try_start_prespec_for_state(
        segment.asr_state, ws_id=self.connection_id, seg_id=seg_id,
        committed=segment.committed,
        audio_queue_empty=segment.audio_queue.empty(),
    )
    if decision is not None and decision.launched:
        segment.prespec_started_window = window_id
```

本次调用可能刚完成一个 2 s chunk，也可能仅向 `state.buffer` 追加了 80 ms 而没有进入 `while`。**后者正是检查 1 s 中点的机会。** 在生成器退出后，`state.audio_accum` 和 `state.buffer` 都已更新；这里先调用 `try_start_prespec_for_state(state, ...)`，由它构造完整快照、回滚拆分和采样截止位置，再调用 `try_start_prespec(...)`。两层实参分别列在 4.5。它仅在以下条件满足时调用现有 `maybe_start()`：

1. segment 未 commit、当前段音频队列无积压、上一轮正式推理已结束。commit 后合并剩余音频的路径不启动 shadow。
2. 自上次正式请求的音频截止位置以来，新增音频量 `new_audio_samples` 位于 `[中点, chunk_size_samples)`；对 2 s chunk，就是 `[16000, 32000)`。这只是触发计数，传给引擎的音频仍是完整累计音频。
3. 按**下一次正式请求的现有回滚规则**拆分前缀和回滚草稿。包括 `unfixed_chunk_num` 的预热条件、`unfixed_token_num` 和 `kmax`，不要另写固定“最后 5 个 token”的拆分。首个 `chunk-0` 没有上一轮结果，空草稿是合法输入；它在约 1 s 中点也可起草。
4. 当前窗口尚未成功启动过预投机；正式队列、负载、剩余时间是否允许，再由协调器判定。

这里检查队列是防止消费协程追赶积压时制造“伪空档”：如果下一正式 chunk 所需音频已经在队列里，只是还没取出，就不应再起 shadow。段级队列非空是这一额外跳过条件，不能把它当作 `LoadSnapshot.waiting`（vLLM 等待请求数）。

由已知字段计算计数与请求输入，不增加一份易漂移的接收端音频副本：

```python
audio_pcm_f32le, current_audio_end_samples, last_formal_audio_end_samples = (
    snapshot_stream_audio(state.audio_accum, state.buffer)
)
new_audio_samples = current_audio_end_samples - last_formal_audio_end_samples
```

两个截止位置都采用**段内采样坐标**：这里它们分别等于 `len(state.audio_accum) + len(state.buffer)` 和 `len(state.audio_accum)`。`current_audio_end_samples` 不包含仍在段级队列、尚未追加到 `state.buffer` 的客户端音频。shadow 快照包含已经正式处理的全部 chunk 加上当前 buffer；`new_audio_samples` 只用于中点判断，不能拿来裁剪引擎输入。

例如上一次正式输入是本段 `[0, 2s]`：到 3 s 时，触发计数是 1 s，预投机传 `[0, 3s]`；到 4 s 时，正式请求仍传 `[0, 4s]`。两次请求分别可能命中多少缓存，均由引擎决定。

预算按**距离下一正式提交尚缺的采样数**估计：`(chunk_size_samples - new_audio_samples) / 16000`，结合实际接收节奏；不能在完成上一次推理后重新假定还有整整 1 s。VAD 仍可能提前 commit，取消只是尽力让路，不保证 NPU kernel 立即退出。

### 四、模型侧 helper：直接复制的代码与需要搬出的原代码

#### 4.1 先用明确名字说明三个 token 序列

| 本文变量名 | 含义 | 放在哪里 |
|---|---|---|
| `prefix_text` / `stable_prefix_token_ids` | L486–518 从 `state._raw_decoded` 得到的前缀文本及对应 token IDs；固定前缀阶段 `prefix_text=""` | 实际 prompt 文本仍按 `state.prompt_raw + prefix_text` 构造；token IDs 用于快照记录，不自行拼到多模态 prompt |
| `rollback_draft_token_ids` | 上一轮输出中，本次需要重新验证的回滚后缀 | 已实现的 external draft 字段 |
| `candidate_token_ids` | shadow 请求的完整原始输出，含其首个 sampled token | 仅暂存；下一正式请求只把 `candidate_token_ids[1:]` 作为 external draft |

“保留前缀”只描述当前回滚算法的拆分，不额外声称这些文本永远不会变化。协调器原有字段 `stable_prefix_tokens`、`rollback_tokens`、`Candidate.token_ids` 分别记录前缀 IDs、回滚草稿和候选。`prefix_text` 在模型扩展快照中单独保存：正式请求原本拼的是文本，不能由 token IDs 猜测或重新解码来代替。

例如，保留前缀为 `[10, 11]`，旧草稿为 `[21, 22]`。当前音频下 shadow target 先采样 `20`，随后验证 `21`、把 `22` 修正为 `99`，然后生成 `30`：

```text
输入：完整累计音频 [0, 3s]
      prompt 中的保留前缀 [10, 11]
      external draft    [21, 22]
引擎：先采样 20 → 验证 21 → 修正 22 为 99 → 继续生成 30
返回：candidate_token_ids = (20, 21, 99, 30)
正式请求提交的 external draft = (21, 99, 30)
```

候选既不包含 `[10, 11]`，也不保留被拒绝的 `22`。**验证与续写在同一次 engine 请求中完成**；服务不拆成两次请求，不自行重建验证后的 KV。对用户给出的 proposer，target 先采样首 token，再提出 external draft；因此启动前要求 `candidate_tokens >= 1 + len(rollback_draft_token_ids) + 1`，分别为首 token、完整旧草稿和至少一个新续写位置。旧草稿超过 `kmax` 返回 `rollback_draft_exceeds_kmax`，预算不足返回 `candidate_budget_insufficient`；不再截掉草稿尾部。shadow 原始输出的首 token **只在正式提交时去掉一次**，旧回滚草稿已经按原规则去首，不能再去。这个检查是**静态上限**，EOS、修正或提前取消仍可能没有续写，实际 proposer 还可能因模型上下文不足缩短草稿，需要在 910B 核实有效长度和首 token 对齐。

#### 4.2 哪些已提供，哪些必须从你的函数搬出

三个完整方法已放在同目录 [streaming_prespec_helpers.py](streaming_prespec_helpers.py)，第一节的 mixin 导入后即可使用：

| 方法 | 已实现的代码 | 你需要修改的部分 |
|---|---|---|
| `try_start_prespec(...)` | 中点/队列检查、独立输入快照、截止时间、`maybe_start` | 调用处用 `audio_accum + buffer` 形成快照，按 L486–518 取得下次前缀和草稿，见 4.5 |
| `_build_stream_request(request, budget)` | 请求级采样参数副本、完整旧草稿、独立 request ID；提交前再次检查预算 | 提供下面的 `_pack_stream_request` |
| `_run_prespec(request, budget)` | `generate`、累计输出收集、截止时间、异常/取消时 abort 与关闭迭代器 | 初始化时传入已有 engine；若本地 fork 改了输出接口，按 4.6 核对 |

**从 `async_streaming_transcribe()` 提取两段只读逻辑。** 草稿拆分一段单独复用，请求打包一段放进 `_pack_stream_request`：

1. 将 L486–518 的 `state._raw_decoded → cur_ids → prefix / external_draft` 拆分代码提成一个**只读**函数，例如 `_split_stream_rollback(raw_decoded, chunk_id, unfixed_chunk_num, unfixed_token_num, kmax)`。正式 chunk、中点和使用上一轮转写构造稳定前缀的正式 finish 都调用它；保留固定阶段 `prefix=""`、`cur_ids[1:][:kmax]`，以及动态阶段反复扩大回滚 token 数直到解码前缀不含 `�` 的原有细节。该函数返回原来的 `prefix` 文本、前缀 IDs 和草稿 IDs。提取时不要提前改写 `state._raw_decoded` 或 `state.chunk_id`。
2. 将 L520–554 的 `prompt = state.prompt_raw + prefix`、processor 输入、`TokensPrompt` 或 adapter 的 `build_request` 分支搬入 `_pack_stream_request`。输入改为下面六个显式形参，使用传入的完整音频，不从活跃 `state` 取 `audio_accum`。保持原有 `CACHE_ID` 路径和输入格式；这里不增加 KV 命中判断。
3. 将 L559–568 向 `sampling_params.extra_args` 写入 `ascend_external_draft` 的原代码也搬入打包函数，使正式与 shadow 调用同一字段/格式。`_pack_stream_request` 只返回 generate 实参；不搬入 L557 的 `ChunkInferenceEvent`、L571–606 的打点、输出和状态写回。

请求打包接口如下；L486–518 的只读拆分函数和这个打包函数都需要按原实现填写，不能原样作为完整函数运行：

```python
def _pack_stream_request(
    self, *, full_audio_pcm_f32le, prefix_text, stable_prefix_token_ids,
    draft_token_ids, model_context, sampling_params,
):
    full_audio = np.frombuffer(full_audio_pcm_f32le, dtype="<f4").copy()
    prompt_text = model_context["prompt_raw"] + prefix_text

    # 在这里搬入 L520–554：
    # 以 prompt_text 和 full_audio 为输入，按原代码选择 adapter.build_request
    # 或 TokensPrompt(prompt_token_ids=..., multi_modal_data={"audio": ...})。
    # 原代码计算出 engine_prompt；如果还需其他 generate 参数，保存到
    # other_generate_kwargs。此处不要读取活跃 state 或修改真实段状态。

    sampling_params.extra_args = {
        **(sampling_params.extra_args or {}),
        "ascend_external_draft": list(draft_token_ids),
    }
    # 如果你现有实现对草稿还有包装结构，保留原代码的值格式。
    # 草稿只进 external draft；prompt_text 不附加这些 token。

    return dict(
        prompt=engine_prompt,
        sampling_params=sampling_params,
        **other_generate_kwargs,
    )
```

`np` 用服务已有的 `import numpy as np`。`other_generate_kwargs` 不含 `prompt/sampling_params/request_id`；若原调用没有其他参数，在该分支设为 `{}`。`model_context` 至少保存 `state.prompt_raw`，以及 L520–554 实际读取的语言/context/模板；不放整个 state、engine、processor 或设备 tensor。processor 和只读模型常量可以从 `self` 取。传进来的 `sampling_params` 已经是本请求专用副本，正式路径保持原有 `temperature=0,max_tokens=4096,output_kind=DELTA`；shadow 在 `_build_stream_request` 中设较小预算和 CUMULATIVE 输出。

`ascend_external_draft` 是你给出的实际 extra_args 键；其值若由原实现封装为别的结构，沿用原结构。缺少服务函数原文，`adapter.build_request` 的参数和 `TokensPrompt` 的具体 prompt token 构造仍须从 L520–554 搬入；测试里的 `test_draft` 只是 mock 字段，不能搬到服务。

正式路径也复用 `_pack_stream_request`，避免两个版本漂移。为此可把 L559–568 的采样参数构造提前到 L557 的 `ChunkInferenceEvent` 之前，再一起调用打包函数；`ChunkInferenceEvent` 仍先于 `engine.generate`，客户端事件顺序不变。正式请求保留原 `temperature=0,max_tokens=4096,DELTA`；不要让它调用限定 shadow 预算的 `_build_stream_request`：

```python
request_kwargs = self._pack_stream_request(
    full_audio_pcm_f32le=state.audio_accum.astype("<f4", copy=False).tobytes(),
    prefix_text=prefix,  # L486–518 原代码返回的文本
    stable_prefix_token_ids=stable_prefix_token_ids,
    draft_token_ids=draft_token_ids,  # 第五节选择就绪候选或原回滚草稿
    model_context=model_context,      # 至少包含 state.prompt_raw 的快照
    sampling_params=deepcopy(formal_sampling_params),
)
stream = engine.generate(**request_kwargs, request_id=formal_request_id)
# yield ChunkInferenceEvent 后，stream 交回原有 _yield_delta_text 迭代；
# L601–606 的状态写回仍留在原位置。
```

上述局部变量映射到原函数已有值，`deepcopy` 来自 `copy`。保留前缀/回滚草稿的拆分规则也沿用原代码：检查 `unfixed_chunk_num` 预热、`unfixed_token_num` 与 `kmax`，在中点按“下次正式请求”规则只读计算，不能为获得草稿先执行一次正式状态回滚。

#### 4.3 请求快照与三个 helper 的完整函数体

下面直接展示 `streaming_prespec_helpers.py` 的实现。使用第一节的 mixin 时不必再复制这些函数到模型类。

```python
import asyncio
import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Any
import numpy as np

logger = logging.getLogger(__name__)

if __package__:
    from .streaming_prespec import LaunchDecision, ShadowRequest
else:
    from streaming_prespec import LaunchDecision, ShadowRequest


@dataclass(frozen=True, kw_only=True)
class ModelShadowRequest(ShadowRequest):
    # Private copies of small request settings, never the live ASR state/engine.
    prefix_text: str
    model_context: dict[str, Any]
    sampling_params: Any


def snapshot_stream_audio(audio_accum, buffer) -> tuple[bytes, int, int]:
    """Snapshot all consumed full chunks plus the unconsumed audio tail.

    Returns complete 16 kHz float32 LE PCM, its sample count, and the last
    formal chunk cutoff. Both counts use segment-relative sample coordinates.
    """
    consumed = np.asarray(audio_accum, dtype=np.float32).reshape(-1)
    pending = np.asarray(buffer, dtype=np.float32).reshape(-1)
    complete = np.concatenate((consumed, pending)).astype("<f4", copy=False)
    return complete.tobytes(), int(complete.size), int(consumed.size)


class PreSpecModelHelpers:
    """Mixin for Qwen3ASRModelAsync, or copy these three methods into that class.

    Set ``prespec_engine`` to the existing shared engine before enabling.
    Supply ``_pack_stream_request`` as documented; it must not mutate live state.
    ``prespec`` is the existing PreSpecCoordinator, or None when disabled.
    """

    def try_start_prespec(
        self, *, ws_id, seg_id, window_id, prefix_version, full_audio_pcm_f32le,
        current_audio_end_samples, last_formal_audio_end_samples,
        chunk_size_samples, prefix_text, stable_prefix_token_ids,
        rollback_draft_token_ids,
        model_context, sampling_params, committed, audio_queue_empty,
        load=None, predicted_draft_ms=0.0,
    ):
        coordinator = getattr(self, "prespec", None)
        if coordinator is None:
            return LaunchDecision(False, "disabled")
        if committed:
            return LaunchDecision(False, "committed")
        if not audio_queue_empty:
            return LaunchDecision(False, "audio_backlog")
        new_samples = current_audio_end_samples - last_formal_audio_end_samples
        if chunk_size_samples <= 0 or new_samples < 0:
            raise ValueError("invalid chunk size or audio cutoffs")
        if new_samples < coordinator.policy.midpoint_samples:
            return LaunchDecision(False, "before_midpoint")
        if new_samples >= chunk_size_samples:
            return LaunchDecision(False, "formal_chunk_due")
        draft_len = len(rollback_draft_token_ids)
        if draft_len > coordinator.policy.max_rollback_draft_tokens:
            if coordinator.should_log_budget_skip(
                    ws_id, seg_id, window_id, "rollback_draft_exceeds_kmax"):
                logger.info(
                    "PRESPEC_SKIP ws_id=%s seg=%s window=%s reason=rollback_draft_exceeds_kmax "
                    "draft_tokens=%s kmax=%s",
                    ws_id, seg_id, window_id, draft_len,
                    coordinator.policy.max_rollback_draft_tokens,
                )
            return LaunchDecision(False, "rollback_draft_exceeds_kmax")
        if draft_len + 2 > coordinator.policy.max_candidate_tokens:
            if coordinator.should_log_budget_skip(
                    ws_id, seg_id, window_id, "candidate_budget_insufficient"):
                logger.info(
                    "PRESPEC_SKIP ws_id=%s seg=%s window=%s reason=candidate_budget_insufficient "
                    "draft_tokens=%s candidate_budget=%s required=%s",
                    ws_id, seg_id, window_id, draft_len,
                    coordinator.policy.max_candidate_tokens, draft_len + 2,
                )
            return LaunchDecision(False, "candidate_budget_insufficient")
        # Take the time before copying so snapshot overhead consumes the budget.
        deadline_ns = coordinator.clock() + int(
            (chunk_size_samples - new_samples) / 16000 * 1e9)
        audio_bytes = bytes(full_audio_pcm_f32le)
        if not audio_bytes or len(audio_bytes) % 4:
            raise ValueError("audio must be complete mono float32 LE PCM")
        if len(audio_bytes) // 4 != current_audio_end_samples:
            raise ValueError("audio bytes and sample cutoff disagree")
        request = ModelShadowRequest(
            session_id=ws_id, audio_epoch=seg_id, window_id=window_id,
            base_prefix_version=prefix_version,
            visible_audio_end_samples=current_audio_end_samples,
            window_audio_samples=new_samples,
            audio_pcm_f32le=audio_bytes,
            stable_prefix_tokens=tuple(stable_prefix_token_ids),
            rollback_tokens=tuple(rollback_draft_token_ids),
            deadline_ns=deadline_ns,
            prefix_text=prefix_text,
            model_context=deepcopy(model_context),
            sampling_params=deepcopy(sampling_params),
        )
        return coordinator.maybe_start(request, load, predicted_draft_ms)

    def _build_stream_request(self, request, budget):
        if len(request.rollback_tokens) > self.prespec.policy.max_rollback_draft_tokens:
            raise ValueError("rollback draft exceeds engine kmax")
        if len(request.rollback_tokens) + 2 > budget:
            raise ValueError("candidate budget cannot cover first token, old draft and extension")
        params = deepcopy(request.sampling_params)
        params.max_tokens = budget
        params.min_tokens = 0
        params.n = 1
        if getattr(params, "best_of", None) is not None:
            params.best_of = 1
        # Use the service's installed vLLM enum, without a vLLM import here.
        params.output_kind = type(params.output_kind).CUMULATIVE
        kwargs = self._pack_stream_request(
            full_audio_pcm_f32le=request.audio_pcm_f32le,
            prefix_text=request.prefix_text,
            stable_prefix_token_ids=list(request.stable_prefix_tokens),
            draft_token_ids=list(request.rollback_tokens),
            model_context=deepcopy(request.model_context),
            sampling_params=params,
        )
        if "prompt" not in kwargs or kwargs.get("sampling_params") is not params:
            raise ValueError("packer must return prompt and supplied sampling_params")
        kwargs = dict(kwargs)
        kwargs["request_id"] = request.request_id
        return kwargs

    async def _run_prespec(self, request, budget):
        kwargs = self._build_stream_request(request, budget)
        # Let an already-ready formal/commit callback cancel before submission.
        await asyncio.sleep(0)
        remaining_seconds = (request.deadline_ns - self.prespec.clock()) / 1e9
        if remaining_seconds <= 0:
            raise TimeoutError("prespec deadline passed during request preparation")
        stream = self.prespec_engine.generate(**kwargs)
        finished = False
        candidate_token_ids = ()
        try:
            async with asyncio.timeout(remaining_seconds):
                async for output in stream:
                    if output.outputs:
                        if len(output.outputs) != 1:
                            raise ValueError("prespec requires one output sequence")
                        completion = output.outputs[0]
                        # Cumulative IDs replace the previous snapshot.
                        candidate_token_ids = tuple(completion.token_ids)
                        if completion.finish_reason == "abort":
                            raise RuntimeError("prespec engine request aborted")
                    finished = output.finished
            if not finished or not candidate_token_ids:
                raise RuntimeError("prespec stream ended without a completed candidate")
            if len(candidate_token_ids) > budget:
                raise ValueError("engine output exceeds the candidate token budget")
            return candidate_token_ids
        finally:
            try:
                if not finished:
                    await self.prespec_engine.abort(request.request_id)
            finally:
                await stream.aclose()
```

这份代码明确完成了此前需要自行补写的部分：

- `try_start_prespec` 是同步函数，只调用 `maybe_start` 创建后台 task；它不会等待生成，也不会推进上次正式音频截止位置。
- `ModelShadowRequest` 在通用 `ShadowRequest` 上补了 `model_context/sampling_params`。创建时深拷贝小配置、音频转 bytes、token 转 tuple；它不持有真实 state。
- `_build_stream_request` 给每次 shadow 建立参数副本，设置累计输出、单序列及总 token 上限；已有回滚草稿完整传入，预算不足时跳过。
- `_run_prespec` 只取引擎给出的**已验证输出**，累计输出用覆盖赋值，不重复拼接；流异常结束时不发布半截候选。
- 提前 commit/正式 chunk 导致取消时，只 abort 自己的 `prespec-...` 请求。到预估下一正式提交时间仍未结束也会取消；正式路径没有 await shadow 的操作。

截止时间覆盖剩余生成时间，并扣除快照/打包已经花掉的时间；同步 processor 工作仍会占用事件循环，不能通过 `create_task` 或 timeout 自动抢占，应把这部分 CPU 成本计入实验。

#### 4.4 音频长度直接取已有 state 字段

根据 L475–481，进入正式请求前先从 `state.buffer` 取完整 chunk，并拼进 `state.audio_accum`。因此正式请求的音频截止是 `len(state.audio_accum)`；中点尚未达到下个 chunk 大小，当前 shadow 音频截止是 `len(state.audio_accum) + len(state.buffer)`。`snapshot_stream_audio` 同时给出这两个**段内**采样数和完整音频 bytes，无需另加 `prespec_last_formal_audio_end_samples` 字段，也无需用客户端已接收字节数估计。

例如 `state.audio_accum` 长 32000、`state.buffer` 长 16000，则新音频量为 16000，shadow 提交全部 48000 samples。到下个正式 chunk 切出后，`state.audio_accum` 长 64000，正式请求提交全部 64000。shadow 不更新这两个数组，缓存能复用多少由引擎决定。

#### 4.5 第三节挂点与 helper 的两层实参

第三节的挂点位于连接层 `_process_segment()` 的 `async for` 之后。这里调用的是模型层包装函数 `try_start_prespec_for_state()`，**只传以下五个实参**：

| `try_start_prespec_for_state` 形参 | 第三节传入的值 |
|---|---|
| `state` | `segment.asr_state` |
| `ws_id` | `self.connection_id`，与原流式函数第三个位置参数相同 |
| `seg_id` | 当前段号，对应现有 `seg-N` 字段 |
| `committed` | `segment.committed` |
| `audio_queue_empty` | `segment.audio_queue.empty()` |

包装函数读取 `state`，再向已有的 `try_start_prespec()` helper 传以下参数。**下表不是连接层需要逐项传入的实参**：

| `try_start_prespec` 形参 | 包装函数中的取值 |
|---|---|
| `ws_id`、`seg_id`、`committed`、`audio_queue_empty` | 原样转发连接层传入值 |
| `window_id` | `state.chunk_id`，标识下一次正式 chunk；同段每轮递增 |
| `prefix_version` | `state.prespec_prefix_version` |
| `full_audio_pcm_f32le`、`current_audio_end_samples`、`last_formal_audio_end_samples` | `snapshot_stream_audio(state.audio_accum, state.buffer)` 的三个返回值；音频包含完整历史和当前 buffer |
| `chunk_size_samples` | `state.chunk_size_samples` |
| `prefix_text`、`stable_prefix_token_ids`、`rollback_draft_token_ids` | 用 `state._raw_decoded`、`state.chunk_id` 按 L486–518 的规则只读拆分；动态阶段保留 `�` 修复循环 |
| `model_context` | 至少包含 `state.prompt_raw`，再加入原 processor 所需语言/context；不要把整个 state 交给后台 |
| `sampling_params` | 为 shadow 准备的采样参数；以 L559–568 的正式参数为基准，helper 将其改为较小预算和 CUMULATIVE 输出 |
| `load`、`predicted_draft_ms` | 初次 force 实验分别为 `None`、`0.0` |

`try_start_prespec_for_state` 是待加在模型类中的薄包装；第三节已有连接层调用示例，下面只展示包装函数实现。它不得修改 `state.buffer`、`state.audio_accum`、`state._raw_decoded` 或 `state.chunk_id`。`state.prompt_raw` 的字段名由你提供；其他真实成员名仍要与源码核对。

```python
def try_start_prespec_for_state(
    self, state, *, ws_id, seg_id, committed, audio_queue_empty,
):
    if self.prespec is None or committed or not audio_queue_empty:
        return None
    new_samples = len(state.buffer)  # L462–476：尚未凑满下个正式 chunk
    if not (self.prespec.policy.midpoint_samples <= new_samples
            < state.chunk_size_samples):
        return None  # 80 ms append 阶段不反复复制完整历史音频
    prefix_text, stable_prefix_token_ids, rollback_draft_token_ids = (
        self._split_stream_rollback(
            state._raw_decoded, state.chunk_id,
            self.unfixed_chunk_num, self.unfixed_token_num, self.kmax,
        )
    )
    audio_pcm_f32le, current_end, last_formal_end = snapshot_stream_audio(
        state.audio_accum, state.buffer,
    )
    model_context = {"prompt_raw": state.prompt_raw}  # 加上原 processor 所需语言/context
    return self.try_start_prespec(
        ws_id=ws_id, seg_id=seg_id, window_id=state.chunk_id,
        prefix_version=state.prespec_prefix_version,
        full_audio_pcm_f32le=audio_pcm_f32le,
        current_audio_end_samples=current_end,
        last_formal_audio_end_samples=last_formal_end,
        chunk_size_samples=state.chunk_size_samples,
        prefix_text=prefix_text,
        stable_prefix_token_ids=stable_prefix_token_ids,
        rollback_draft_token_ids=rollback_draft_token_ids,
        model_context=model_context,
        sampling_params=SamplingParams(
            temperature=0, max_tokens=4096,
            output_kind=RequestOutputKind.DELTA,
            # L559–568 若还有 stop/EOS 等参数，也逐项沿用。
        ),
        committed=committed,
        audio_queue_empty=audio_queue_empty,
        load=None, predicted_draft_ms=0.0,  # 初次 force 实验
    )
```

`self.unfixed_chunk_num`、`self.unfixed_token_num`、`self.kmax` 在示例中代表原函数已读取的配置，若实际成员/局部变量名不同，应原样映射。`SamplingParams/RequestOutputKind` 沿用原文件的 import 和 L559–568 的所有实际参数。中点若只有 `state.buffer` 且不足一个正式 chunk，不调用 `async_finish_streaming_transcribe()`；上述包装仅启动独立 shadow。

#### 4.6 输出接口假设与当前验证范围

helper 使用 `engine.generate(**kwargs)` 返回异步生成器、`await engine.abort(request_id)` 取消、`output.outputs[0].token_ids` 返回 prompt 之后的 token，以及 `output.finished` 表示结束的接口。对应公共接口参考 [vLLM AsyncLLM](https://docs.vllm.ai/en/stable/api/vllm/v1/engine/async_llm/)；累计输出选择见 [SamplingParams](https://docs.vllm.ai/en/stable/api/vllm/sampling_params/)。本地 fork 若改变了这些接口，需调整 `_run_prespec` 的相应取值，不能把文本重新 tokenize 代替原始输出 IDs。

普通 vLLM 输出不包含 prompt token，因此代码没有再切掉 `len(stable_prefix_token_ids)`；否则会丢失候选开头。引擎完成标记之后才发布候选。若你修改过输出为“整个序列”或只报告增量，先还原此合同。

按你提供的 external proposer，`propose(sampled_token_ids, ...)` 在 target 已经采样首 token 后运行，返回的 draft 被接在这些 `sampled_ids` 后面。故旧回滚草稿沿用 `_split_stream_rollback()` 已去首的结果；shadow `RequestOutput.token_ids` 是原始生成序列，正式请求使用 `Candidate.formal_draft_token_ids`（即 raw `[1:]`）。这一步是根据所给 proposer 代码作出的对齐推断：在 910B 用首 token ID、原始候选长度、提交草稿长度与 verifier 有效草稿长度核对；`max_model_len` 不足时 proposer 仍可能截短草稿。

shadow 不调用正式转写/finish/`_yield_delta_text`，不发送客户端事件、不修改正式 state；引擎全局统计仍可能包含 shadow，需按 request ID 单独分析。

本地 mock 用例覆盖完整音频、输入隔离、旧草稿传递、纠正后输出的收集、累计输出去重、跳过条件、取消及异常清理。**它验证适配逻辑，不证明真实 verifier 已正确执行，也不证明 910B 上的时延收益。** 实际服务接通后还需看一次旧草稿被拒绝后继续 decode 的日志，并比较 baseline/probe 正式 P95/P99。

### 五、正式 chunk：在函数内部替换草稿来源

挂点在 `async_streaming_transcribe()` 的**每次切 chunk 推理循环内部**：正式音频和保留前缀已确定，external draft 尚未打包。只在外面包一次函数调用会漏掉同次调用的后续 chunk。

`_process_segment()` 原本调用 `async_streaming_transcribe(item, segment.asr_state, self.connection_id)`；第三个位置参数 `connection_id` 已经是 `ws_id`，不新增 `prespec_ws_id`。只在模型函数参数列表末尾增加 `prespec_seg_id=None`，把当前段号传入：

```python
async for stream_item in self.asr.async_streaming_transcribe(
    item, segment.asr_state, self.connection_id,
    prespec_seg_id=seg_id,
):
    await self._send_stream_item(stream_item)
```

这是第三节同一个调用，不是再调用一次转写。若原调用还有其他参数，全部保留；未提交和 commit 后消费完整 chunk 的调用点都保留原第三参数，并传 `prespec_seg_id`。模型函数内部可令 `ws_id = connection_id`、`seg_id = prespec_seg_id`；`seg_id` 必须取当前段在日志 `seg-N` 中使用的段号，不能从连接 ID 推断。下面的 `prefix_version` 明确取 `state.prespec_prefix_version`，初始化和更新见第六节。

准确的改动位置在每轮 `while len(state.buffer) >= state.chunk_size_samples` **内部**：L475–481 已切块并更新 `state.audio_accum`；L486–518 已算好原 `prefix` 与草稿；在 L520 的 prompt 构造前领取候选。使用 `len(state.audio_accum)` 作为本轮正式完整音频截止。完成 L583–596 的 DELTA 输出迭代和 L601–606 的 `_raw_decoded = prefix + accumulated_delta`、解析及 `chunk_id += 1` 后配对退出。这些现有状态更新仍只在正式路径发生。

```python
# L486–518 已通过原有规则得到 prefix、rollback_draft_token_ids。
# 位置：L475–481 之后、L520 构造请求之前。
prespec = getattr(self, "prespec", None)
if ws_id is None or seg_id is None:
    prespec = None  # 未传标识的旧调用方不使用这条预投机接线
candidate = None
this_chunk_id = state.chunk_id  # 必须在本轮末尾 chunk_id += 1 之前保存
this_chunk_audio_end = len(state.audio_accum)
if prespec is not None:
    candidate = prespec.begin_authoritative_chunk(
        ws_id, audio_epoch=seg_id, window_id=this_chunk_id,
        prefix_version=state.prespec_prefix_version,
        full_audio_end_samples=this_chunk_audio_end,
    )
try:
    draft_token_ids = (candidate.formal_draft_token_ids if candidate is not None
                       else rollback_draft_token_ids)
    # L520–554：原 prompt_raw + prefix 和 adapter/TokensPrompt 分支。
    # 提前构造 L559–568 的正式 SamplingParams，用 draft_token_ids 写入
    # ascend_external_draft；这一步通过 4.2 的 _pack_stream_request 复用。
    # L557：照常 yield ChunkInferenceEvent。
    # 正式请求仍保持 temperature=0、max_tokens=4096、DELTA。
    # L583–596：继续 _yield_delta_text，照常 yield 客户端 delta。
    # L601–606：原样更新 _raw_decoded、解析输出和 chunk_id。
    state.prespec_prefix_version += 1  # 仅在正式状态成功写回后
finally:
    if prespec is not None:
        prespec.end_authoritative_chunk(
            ws_id, audio_epoch=seg_id, window_id=this_chunk_id)
```

上段是**插入位置示意**：注释所列的 L520–606 必须保留原函数实际代码，不能把注释当作代码替换。最小版不采集正式请求的原始 token IDs，也不修改 `_yield_delta_text()`。`end_authoritative_chunk()` 不传 `target_continuation_tokens` 时只清理候选，返回 `None`；不得把缺失的对照数据记作“接受 0 个 token”。若需要离线候选质量评估，按 [验证手册的可选采集挂点](streaming_prespec_validation.md) 从正式请求的 DELTA 输出流收集 token IDs；不能初始化 `raw_target_tokens = ()` 后不更新便传入。`try/finally` 覆盖 L557–606 以及真正的生成迭代；清理调用者时显式退出生成器，使 finally 执行。

`shadow_only=True` 时 candidate 恒为 None，正式路径仍用原回滚草稿；默认只记录候选是否赶上正式边界。若按验证手册另行采集正式输出 token，可离线计算原始输出前缀匹配和跳过首 token 后的草稿前缀匹配；它们不是 verifier 接受数。若候选只有首 token、无可提交后续 token，probe 为 `candidate_no_draft`；若后续 token 长度超过 `kmax`，probe 为 `candidate_exceeds_kmax`，两者都不进入正式请求。切换消费模式后，预投机候选仍经过现有 verifier 在完整正式音频上重新验证；实际接受数从引擎 verifier 日志获取。commit 的 residual finish 使用相同的领取与验证规则，接线见下一节。

### 六、commit 与清理：给出插入代码和字段初始化

#### 新字段由谁初始化

在 `SegmentState` 的定义中增加以下控制字段。若它是 dataclass/Pydantic 模型，使用字段声明；若是普通 class，在 `__init__()` 内初始化：

```python
prespec_formal_begun: bool = False
prespec_formal_window_id: int | None = None
prespec_started_window: int | None = None
prespec_finish_target_tokens: tuple[int, ...] | None = None
```

`prespec_started_window` 只在 `decision.launched` 时设为当时的 `state.chunk_id`，避免该窗口随后的 80 ms append 一再复制完整音频。正式 chunk 完成后 `state.chunk_id` 递增，下一窗口自然可以再次启动；协调器按 `(ws_id, seg_id, window_id)` 做最终去重。新段创建时两个窗口字段为 `None`，`prespec_formal_begun` 为 `False`。

**版本只保存在 ASRStreamingState，不在 SegmentState 再保存一份。** 在 ASRStreamingState 的字段定义中新增 `prespec_prefix_version: int = 0`；`init_streaming_state()` 创建它时初始值为 0。若该类型在另一个文件定义，字段声明也要加在那个定义处，不能假设允许动态附加属性。在每次正式 chunk 成功写回文本/回滚状态之后增加：

```python
state.prespec_prefix_version += 1
```

语言/context 更新若影响活跃段，也对该段 state 的版本递增；构造请求时若更换了音频上下文起点，应在领取候选前递增。快照和正式领取统一读取 `state.prespec_prefix_version`。`handle_event()` 的 `current_prefix_version` 取这个值，state 尚未创建时取 0（commit 入口不领取候选）。

`ws_id/seg_id/state/segment` 分别对应日志中的连接 ID、段号、该段 ASR state、SegmentState。`handle_event()` 的 append 分支继续只负责原来的解码入队；中点计数已经由 `state.audio_accum` 和 `state.buffer` 给出，不新增 `_prespec_received_samples`。

#### `handle_event()`：在现有 `put(None)` 前插入

```python
if self.prespec is not None and not segment.prespec_formal_begun:
    window_id = (segment.asr_state.chunk_id
                 if segment.asr_state is not None else 0)
    self.prespec.begin_formal(
        ws_id, seg_id, current_prefix_version,
        0,  # 此时队列尚未消费完，不能用它验证完整音频截止
        window_id=window_id,
        defer_candidate=True,  # 保留已完成候选；未完成候选立即取消
    )
    segment.prespec_formal_window_id = window_id
    segment.prespec_formal_begun = True

# 随后继续你原有的 None 入队代码，不新增第二个 None。
```

#### `_finish_segment()`：正常 finish 调用附加连接和段标识

保留现有 `remaining <= 0.4s` 的跳过分支，不在该分支领取候选。正常调用 `async_finish_streaming_transcribe(...)` 时，**原有位置实参保持不变**，在调用末尾增加 `prespec_ws_id=self.connection_id, prespec_seg_id=seg_id, on_finish_tokens=record_finish_tokens` 三个关键字。连接层在这次调用之前定义只记录本段原始 token IDs 的同步回调：

```python
segment.prespec_finish_target_tokens = None

def record_finish_tokens(token_ids: tuple[int, ...]) -> None:
    segment.prespec_finish_target_tokens = token_ids

# 原有 await self.asr.async_finish_streaming_transcribe(...) 的位置和
# 原有位置实参不变，只在末尾加上述三个关键字参数。
```

模型函数签名末尾增加这三个可选形参，默认均为 `None`，供未接入预投机的旧调用方继续使用。`connection_id` 是现有 `ws-...`，`seg_id` 是当前日志段号。这个回调不发送客户端事件，也不改变 finish 的返回类型；短尾跳过时字段保持 `None`。

#### `async_finish_streaming_transcribe()`：在 `finish_...` 正式请求提交前领取

此函数原有流程先把剩余 `state.buffer` 并入 `state.audio_accum`，再构造完整音频的正式 finish 请求。如果 finish 和正式 chunk 一样从 `state._raw_decoded` 提取稳定前缀及回滚草稿，应在合并音频后调用**同一个** `_split_stream_rollback()`；不能让 finish 另用一套不检查 `\ufffd` 的切分。按实际成员名替换示例中的配置字段：

```python
prefix_text, stable_prefix_token_ids, rollback_draft_token_ids = (
    self._split_stream_rollback(
        state._raw_decoded, state.chunk_id,
        self.unfixed_chunk_num, self.unfixed_token_num, self.kmax,
    )
)
# 后续 finish prompt 仍为 state.prompt_raw + prefix_text；音频仍为
# 已合并完成的完整 state.audio_accum，不把回滚草稿拼入 prompt。
```

动态阶段的 `\ufffd` 循环应留在这个只读 helper 中：如果当前 token 边界解码出的 `prefix_text` 含替换字符，就扩大待验证的回滚后缀、缩短稳定前缀，再解码检查；直到前缀有效或退到空前缀。候选和正式 finish 必须使用相同的前缀拆分，否则即使 `prefix_version`、`window_id` 相同，也可能把相对于另一条 prompt 构造的草稿交给 finish。若扩大的回滚后缀超过引擎 `kmax`，不能静默截短后继续使用该稳定前缀；正式 chunk 和 finish 都需要安全回退，并禁止消费不再对齐的候选。若实际 finish 从头解码、完全不使用上一轮稳定前缀，则不能机械加入这套拆分；它的候选也必须按该 finish 请求的真实 prompt 合同构造。由于本仓库没有服务端函数源码，这一分支需对照 910B 实现确认。

**在上述拆分和本次 `sampling_params` 确定后，紧邻 finish 请求打包之前**插入：

```python
prespec = getattr(self, "prespec", None)
candidate = None
if (prespec is not None and prespec_ws_id is not None
        and prespec_seg_id is not None):
    candidate = prespec.claim_formal_candidate(
        prespec_ws_id, prespec_seg_id,
        window_id=state.chunk_id,
        prefix_version=state.prespec_prefix_version,
        full_audio_end_samples=len(state.audio_accum),
    )

# 与中点及正式 chunk 采用同一拆分结果；不要重新 tokenize 文本，
# 也不要把候选首 token 拼进 prompt。
finish_draft_token_ids = (
    candidate.formal_draft_token_ids if candidate is not None
    else rollback_draft_token_ids
)
sampling_params.extra_args = {
    **(sampling_params.extra_args or {}),
    "ascend_external_draft": list(finish_draft_token_ids),
}
```

`sampling_params` 必须是本次 finish 请求专用的原有对象；保留其 temperature、max_tokens、output_kind、stop/EOS 等参数和完整音频 prompt，**只替换** `ascend_external_draft` 的值。若正式 finish 用 `_pack_stream_request(...)` 打包，传入相同的 `prefix_text`、`stable_prefix_token_ids` 与 `finish_draft_token_ids`，不要重复写一遍 `extra_args`。正式结果沿用原 finish 的输出拼接与解析位置，并核对稳定前缀恰好包含一次；`finish_...` `request_id`、`engine.generate(...)` 和状态写回位置保持原有时序。紧邻原 `engine.generate(...)` 前加一条来源日志：

```python
logger.info(
    "PRESPEC_TARGET ws_id=%s seg=%s window=%s boundary=formal "
    "formal_request_id=%s candidate_request_id=%s submitted_draft_tokens=%s",
    prespec_ws_id, prespec_seg_id, state.chunk_id,
    finish_request_id,  # 换成原 finish 请求实际使用的 request_id 局部变量
    candidate.request_id if candidate is not None else None,
    len(finish_draft_token_ids),
)
```

`claim_formal_candidate()` 不等待候选，只在其完成、版本相同、候选音频截止不晚于完整正式音频、草稿长度不超 `kmax` 且窗口号仍相同时返回候选。shadow 模式下返回 `None`，但记录 `PRESPEC_PROBE boundary=formal status=ready_shadow_only`，正式请求继续使用原 finish 草稿；consume 模式下可返回候选并记录 `ready_consumed`。consume 的候选必须由现有引擎在**完整正式音频**上验证，不能直接把 shadow 文本发给客户端。

#### `async_finish_streaming_transcribe()`：从同一次 `finish_...` 输出采集 token

在已有的 `engine.generate(...)` 输出消费循环中插入以下采集，保留原有文本拼接、解析、状态写回和返回值。使用本次 finish 请求实际的 `sampling_params.output_kind`；不要再次调用 `generate`，也不要从最终文本重新 tokenize：

```python
finish_raw_token_ids: list[int] = []
finish_output_complete = False
finish_output_valid = True

# 在原有 async for output in finish_stream 循环内、原文本处理之前：
if output.outputs:
    if len(output.outputs) != 1:
        finish_output_valid = False  # 统计不可用，不影响正式输出
    elif finish_output_valid:
        ids = output.outputs[0].token_ids
        if sampling_params.output_kind == RequestOutputKind.DELTA:
            finish_raw_token_ids.extend(ids)
        else:  # CUMULATIVE / FINAL_ONLY：保留最新完整快照
            finish_raw_token_ids[:] = ids
if output.finished:
    finish_output_complete = True

# 原输出循环和原结果解析成功结束后、原 return 之前：
if (on_finish_tokens is not None and finish_output_complete
        and finish_output_valid):
    try:
        on_finish_tokens(tuple(finish_raw_token_ids))
    except Exception:
        logger.exception("PRESPEC_FINISH_CAPTURE callback failed")
```

只有观察到 `output.finished` 才记录 token；真正完成但生成零个 token 时记录 `()`，中断/异常时仍为 `None`。这里记录的是 **prompt 之后的原始生成 token**，不拼接稳定前缀，也不改 `_yield_delta_text()` 的 `on_output`。若本地 finish 用的是 `async for output in engine.generate(...)` 而没有 `finish_stream` 局部变量，就把中间代码放入该现有循环。

#### `_process_segment()`：用最外层 try/finally 包住原函数主体

```python
try:
    # 这里是原有逐块流式处理、commit 后消费、_finish_segment 的完整主体。
    # 不要只包住 _finish_segment；也不要重复调用它。
    ...
finally:
    if self.prespec is not None and segment.prespec_formal_begun:
        try:
            self.prespec.finish_formal(
                ws_id, seg_id,
                target_continuation_tokens=segment.prespec_finish_target_tokens,
                window_id=segment.prespec_formal_window_id)
        finally:
            segment.prespec_formal_begun = False
            segment.prespec_formal_window_id = None
            segment.prespec_finish_target_tokens = None
```

#### `cleanup()`：在生成任务已经退出之后插入

```python
if self.prespec is not None:
    await self.prespec.close_session(ws_id)
```

若原 cleanup 只执行了 `generation_task.cancel()`，还需要等待它退出：复用原有等待逻辑，或在 cancel 后 `await asyncio.gather(generation_task, return_exceptions=True)`。实际任务属性名用原代码的名称；不要等待当前 cleanup 自己所在的 task。关闭顺序见下面的异常路径说明。

#### 异常路径和配对规则

- `handle_event()` 处理真实 commit 时，在注入 None 之前读取当前 `state.chunk_id`，调用 `begin_formal(ws_id, seg_id, version, 0, window_id=window_id, defer_candidate=True)`，并把 `window_id` 存入 `segment.prespec_formal_window_id`。这个 0 是尚未消费队列时的占位参数；保留已经完成的本窗口候选，取消仍在运行的 shadow。重复 commit 不能重复 begin；state 尚未初始化时版本和窗口 ID 均可取 0。
- `_process_segment()` 最外层 finally 检查上述标志，用保存的 `prespec_formal_window_id` 调用 `finish_formal(ws_id, seg_id, target_continuation_tokens=segment.prespec_finish_target_tokens, window_id=...)` 并清标志。不能在 finish 时重新读取可能已经递增的 `state.chunk_id`。此处包住流式循环与 `_finish_segment()`，所以短尾跳过、正常完成和中途异常都会配对；token IDs 为 `None` 时只清理、不做候选质量评估。正式 finish 完成且候选已选中时，协调器自动输出 `PRESPEC_MATCH_TOKENS` 和 `PRESPEC_MATCH boundary=formal`，无需在服务端再打印同一条摘要。
- 如果某段已 begin 但入队失败或被取消前根本未进入 `_process_segment()`，由对应异常处理或连接 cleanup 清理；不能期待一个从未进入的 finally 自动执行。
- `cleanup()` 先取消并等待生成任务及其流式生成器退出，再 `await asr.prespec.close_session(ws_id)`，最后做原有队列/统计清理。不要先删协调器计数再让模型 finally 退出。

`_finish_segment()` 的 `<=0.4s` 跳过条件保持原样：没有正式 finish 推理就不领取候选，外层 `finish_formal()` 丢弃它。若 commit 后队列中的音频又凑足一个完整 chunk，该 chunk 仍可用自己的候选；如果它使 `state.chunk_id` 前进，原窗口的候选不能转交给新的 residual finish，`claim_formal_candidate()` 会记录 `window_changed` 并回退到原 finish 草稿。任何异常或断连都由外层 finally 和 `close_session()` 清理，不等待 shadow。

### 七、最小版范围与验证

910B 上按“回滚投机 baseline → shadow → 消费候选”逐步验证的可执行命令、最小日志接线和判定表见 [streaming_prespec_validation.md](streaming_prespec_validation.md)。launch/probe 仍需要按文档接入日志；正式输出 token 交给协调器后，它会打印匹配 token 和 formal 摘要。只看原有服务日志无法判断预投机是否运行。

当前协调器按 `(session_id, audio_epoch, window_id)` 去重，其中 `audio_epoch=seg_id`、`window_id=state.chunk_id`。**每个尚未提交的正式 chunk 窗口可启动一次**：`chunk-0` 在约 1 s 中点用空旧草稿起草；2 s 正式结果后，3 s 中点为 4 s 的 `chunk-1` 起草；4 s 正式结果写回后，5 s 中点为 6 s 的 `chunk-2` 起草。每次正式 chunk 结束回收该窗口；提前 commit 则保留已完成的本窗口候选，在 residual finish 正式请求提交时核对并领取。后续窗口按原规则验证已有回滚草稿；不能跨 VAD 段借用旧 token，也不能把旧窗口候选交给新窗口。

首次用共享的单个协调器、一个 active shadow、`shadow_only=True`、默认候选预算为 `kmax+1`。受控低并发实验可用 `admission_mode="force"` 和现有旁路监测，先打通链路；它不保证不干扰正式时延。在线 `gated` 模式仍需新鲜队列/利用率和实测耗时预算，约 2 s 的 `npu-smi` 读数不能冒充默认 200 ms 的 admission 读数。

验收只保留直接相关的检查：

1. 中点出现 `prespec-...` 请求；确认旧草稿有拒绝时，后续 decode 从 correction 状态继续。
2. 正式 chunk submit 的 `ready_shadow_only/pending`、引擎 verifier 的实际接受长度与成本分开记录；shadow 不产生客户端事件、不改正式 state、不结束正式 session。
3. 提前 commit、短尾 skip、断连时，无泄漏请求和未配对计数；注意 Python 取消不等于 NPU kernel 已抢占。
4. 同音频/同并发 baseline 与 probe 比较正式 P95/P99、队列等待及总计算成本；在离线回放验证经 verifier 消费后的 token/提交边界等价，再启用消费模式。

服务源码到位后需确认的具体代码点只有：external draft 的打包字段及 kmax 约束、保留前缀/回滚草稿的实际拆分、raw output token 的含义、现有完整音频构造规则及采样截止计数、engine abort/生成器关闭方式、请求级清理与正式 session 结束的调用边界。KV 的可复用范围由引擎决定，不是服务端适配项。当前描述不能替代这些逐行核对或 910B 验证。
