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

**音频输入合同：正式请求和预投机请求都提交截至各自时刻的完整累计音频，沿用现有服务的请求构造规则。** 服务端用新增音频采样计数决定何时尝试预投机；KV 复用由 vLLM-Ascend 判断。

### 一、先完成实例接线：这里给出实际新增代码

#### 步骤 1：复制模块并添加 import

把本仓库的 `research/streaming_prespec.py` 和 `research/streaming_prespec_helpers.py` 一起复制到服务项目的 `qwen_asr/inference/`。在 `qwen3_asr_async.py` 顶部新增：

```python
from qwen_asr.inference.streaming_prespec import (
    AdmissionPolicy,
    PreSpecCoordinator,
)
from qwen_asr.inference.streaming_prespec_helpers import PreSpecModelHelpers
```

把现有类声明改为 `class Qwen3ASRModelAsync(PreSpecModelHelpers, Qwen3ASRModel):`，原有类体保留。这个 mixin 没有构造函数，只提供第四节的三个方法；若此前自行添加过同名占位方法，应替换它们，避免覆盖 mixin 的实现。

下面的代码分为两类：**新增代码**可放入指定位置；**原代码形态**只用于定位，因为没有你的函数原文。原有变量、构造参数不能用示例中的省略号替换。

#### 步骤 2：在模型类中新增 `enable_prespec()` 方法

放在 `class Qwen3ASRModelAsync` 内，与 `async_streaming_transcribe()` 同一级缩进。这是本方案新增的方法，不是现有 API：

```python
def enable_prespec(self, *, enabled, kmax, engine=None, chunk_size_sec=2.0,
                   candidate_tokens=8, on_probe=None):
    # 仅在服务启动、连接尚未创建时调用一次。
    self.prespec = None
    if not enabled:
        return
    if kmax <= 0 or candidate_tokens <= 0 or chunk_size_sec <= 0:
        raise ValueError("prespec requires positive kmax, budget and chunk size")
    midpoint_samples = int(chunk_size_sec * 16000 / 2)
    if midpoint_samples <= 0:
        raise ValueError("prespec midpoint must contain at least one sample")
    if engine is None or not callable(getattr(self, "_pack_stream_request", None)):
        raise ValueError("prespec requires engine and the existing request packer")
    self.prespec_engine = engine  # 引用已创建的同一个 engine，不创建第二个

    self.prespec = PreSpecCoordinator(
        producer=self._run_prespec,
        policy=AdmissionPolicy(
            admission_mode="force",  # 仅用于第一轮受控 shadow 实验
            midpoint_samples=midpoint_samples,
            max_candidate_tokens=min(candidate_tokens, kmax),
            max_active_shadows=1,
        ),
        shadow_only=True,
        on_probe=on_probe,
    )
```

这个方法创建协调器并保存 engine 引用。`self._run_prespec` 已由 mixin 提供；第四节的 `_pack_stream_request` 接到已有实现前仍使用 `enabled=False`。`on_probe` 可传现有非阻塞日志回调；未传时不会自动输出 probe 日志。

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
| 同文件 `init_streaming_state()` | 初始化 `state.prespec_prefix_version=0`；`ws_id/seg_id` 通过新增可选参数传入流式函数，具体见第五、六节 |
| 同文件 `async_streaming_transcribe()` | **每个正式 chunk 内部**，在准备好正式音频/保留前缀、尚未把 external draft 写进请求时领取候选；实际 engine 输出迭代结束后配对退出 |
| 同文件 `_yield_delta_text()` | 正式输出流程保留；若外层拿不到原始 token IDs，增加一个可选的“最终 token 回调”，用于评估；shadow 不调用此函数 |
| 同文件 `async_finish_streaming_transcribe()` | 第一版保留原 finish 路径；commit 到达已取消 shadow 并标记正式任务，finish 不重复领取候选 |
| `realtime_connection.py`：`_process_segment()` | 每次 `async for ... async_streaming_transcribe(...)` 完整结束后尝试中点；最外层 `finally` 配对结束该段的 commit 标记 |
| 同文件 `handle_event()` | append 保留解码入队；commit 在注入 None 前通知协调器，并保存一次配对标志；`session.update` 沿用原语义，若影响活跃段则使旧快照版本失效 |
| 同文件 `cleanup()` | 先取消并等待生成任务退出，再清理该连接 shadow；等待期间仍须让 chunk/segment 的 finally 执行 |

`handle_connection()`、`_run_generation()`、`_send_stream_item()`、`send()`、`async_transcribe()`、`_infer_asr_vllm_async()` 不需要为最小版增加预投机逻辑。`_finish_segment()` 的短尾跳过、done 统计及协议事件保持原流程；清理标记放外层 finally，才能覆盖到达 finish 之前的异常。

### 三、中点触发：放在 `_process_segment()` 消费完一次生成器之后

之前示例中的 `await existing_stream_chunk_inference(...)` 不能直接对应你的流式 API：`async_streaming_transcribe()` 是异步生成器，一次调用可能执行多个 chunk。连接层应这样挂中点检查（参数名仅示意）：

```python
# _process_segment() 原有“未提交、逐块处理”路径
async for item in self.asr.async_streaming_transcribe(audio, state):
    await self._send_stream_item(item)

# 在这里插入第四节 4.5 的 try_start_prespec 调用。
# 它需要明确传入完整音频和下一次回滚拆分结果，不再只传一个 state。
```

`try_start_prespec()` 由模型 mixin 提供，接收完整音频、保留前缀、回滚草稿和采样截止位置；不读取任何假定的 `buffer` 字段。具体实参见第四节 4.5。它仅在以下条件满足时调用现有 `maybe_start()`：

1. segment 未 commit、当前段音频队列无积压、上一轮正式推理已结束。commit 后合并剩余音频的路径不启动 shadow。
2. 自上次正式请求的音频截止位置以来，新增音频量 `new_audio_samples` 位于 `[中点, chunk_size_samples)`；对 2 s chunk，就是 `[16000, 32000)`。这只是触发计数，传给引擎的音频仍是完整累计音频。
3. 按**下一次正式请求的现有回滚规则**可以得到合法回滚草稿。包括 `unfixed_chunk_num` 的预热条件、`unfixed_token_num` 和 `kmax`，不要另写固定“最后 5 个 token”的拆分。
4. 当前段尚未成功启动过预投机；正式队列、负载、剩余时间是否允许，再由协调器判定。

这里检查队列是防止消费协程追赶积压时制造“伪空档”：如果下一正式 chunk 所需音频已经在队列里，只是还没取出，就不应再起 shadow。段级队列非空是这一额外跳过条件，不能把它当作 `LoadSnapshot.waiting`（vLLM 等待请求数）。

计数和请求输入明确分开（变量对应原服务的实际音频/采样位置）：

```python
# 两个截止位置使用同一采样坐标；不读取任何 KV 命中信息。
new_audio_samples = current_audio_end_samples - last_formal_audio_end_samples

# 构造 ShadowRequest 时：
window_audio_samples = new_audio_samples  # 保留已有字段名，仅供中点判断
audio_pcm_f32le = complete_audio.astype("<f4", copy=True).tobytes()
```

`last_formal_audio_end_samples` 在正式请求提交时记录该请求实际包含的音频截止位置；不是推理完成时间，也不是引擎已缓存到的位置。`current_audio_end_samples` 对应当前完整音频快照的截止位置，不能包含还未纳入快照的队列数据。`complete_audio` 直接使用原服务按完整输入合同构造的音频，不用 `new_audio_samples` 去切掉历史前缀。

例如上一次正式输入是本段 `[0, 2s]`：到 3 s 时，触发计数是 1 s，预投机传 `[0, 3s]`；到 4 s 时，正式请求仍传 `[0, 4s]`。两次请求分别可能命中多少缓存，均由引擎决定。

预算按**距离下一正式提交尚缺的采样数**估计：`(chunk_size_samples - new_audio_samples) / 16000`，结合实际接收节奏；不能在完成上一次推理后重新假定还有整整 1 s。VAD 仍可能提前 commit，取消只是尽力让路，不保证 NPU kernel 立即退出。

### 四、模型侧 helper：直接复制的代码与需要搬出的原代码

#### 4.1 先用明确名字说明三个 token 序列

| 本文变量名 | 含义 | 放在哪里 |
|---|---|---|
| `stable_prefix_token_ids` | 按**下一次正式请求的既有回滚规则**，本次要保留、不作为草稿重新验证的文本前缀 | 现有完整 prompt 的文本前缀部分；不是整个多模态 prompt |
| `rollback_draft_token_ids` | 上一轮输出中，本次需要重新验证的回滚后缀 | 已实现的 external draft 字段 |
| `candidate_token_ids` | 用当前完整音频验证、修正旧草稿后，继续生成得到的后缀 | 仅暂存；下一正式请求可把它作为 external draft 再验证 |

“保留前缀”只描述当前回滚算法的拆分，不额外声称这些文本永远不会变化。协调器原有字段 `stable_prefix_tokens`、`rollback_tokens`、`Candidate.token_ids` 分别对应上述三项，不需要改它的接口。

例如，保留前缀为 `[10, 11]`，旧草稿为 `[20, 21]`。当前音频下引擎接受 `20`、把 `21` 修正为 `99`，然后生成 `30`：

```text
输入：完整累计音频 [0, 3s]
      prompt 中的保留前缀 [10, 11]
      external draft    [20, 21]
引擎：验证 20 → 修正 21 为 99 → 在 [10, 11, 20, 99] 上继续生成 30
返回：candidate_token_ids = (20, 99, 30)
```

候选既不包含 `[10, 11]`，也不保留被拒绝的 `21`。**验证与续写在同一次 engine 请求中完成**；服务不拆成两次请求，不自行重建验证后的 KV。总预算包括“接受/修正旧草稿的 token + 新续写 token”；预算用尽或 EOS 时可能没有额外续写，这不等于跳过验证。

#### 4.2 哪些已提供，哪些必须从你的函数搬出

三个完整方法已放在同目录 [streaming_prespec_helpers.py](streaming_prespec_helpers.py)，第一节的 mixin 导入后即可使用：

| 方法 | 已实现的代码 | 你需要修改的部分 |
|---|---|---|
| `try_start_prespec(...)` | 中点/队列检查、独立输入快照、截止时间、`maybe_start` | 在调用处把实际音频/token/采样参数传进来，见 4.5 |
| `_build_stream_request(request, budget)` | 请求级采样参数副本、预算、调用原有打包逻辑、独立 request ID | 提供下面的 `_pack_stream_request` |
| `_run_prespec(request, budget)` | `generate`、累计输出收集、截止时间、异常/取消时 abort 与关闭迭代器 | 初始化时传入已有 engine；若本地 fork 改了输出接口，按 4.6 核对 |

**需要从 `async_streaming_transcribe()` 搬出的模型请求代码集中到 `_pack_stream_request` 一处。** 它不是另一个待设计的推理模块，而是把你现有代码移入一个纯打包函数：

1. 找到 `engine.generate(...)` 前构造完整音频、processor 输入、prompt token IDs、`TokensPrompt` 的代码。保留其中的语言/context、特殊 token 和音频占位符规则。
2. 找到已实现回滚投机中把草稿放入 external draft 的代码，包括 fork 所需的其他元数据；逐字保留字段结构。
3. 把这两段放入模型类的 `_pack_stream_request`。输入改为下面五个显式形参；输出改成 `generate` 使用的关键字参数字典。不搬入真实 state 写回、delta 发送、统计更新或 `generate` 调用。

接口固定如下；这是**唯一需要填入原服务代码的模板**，不能原样作为完整函数运行：

```python
def _pack_stream_request(
    self, *, full_audio_pcm_f32le, stable_prefix_token_ids,
    draft_token_ids, model_context, sampling_params,
):
    full_audio = np.frombuffer(full_audio_pcm_f32le, dtype="<f4").copy()

    # 在这里搬入上述两段原代码：
    # - 原来的完整音频输入变量 → full_audio
    # - 原来的保留文本前缀     → stable_prefix_token_ids
    # - 原来的待验证草稿       → draft_token_ids
    # - 原来从 state 读取的语言/context/prompt → model_context 中对应值
    # - 原来共享的采样参数     → 传入的 sampling_params
    # 搬入后应得到原有 prompt 和其他 generate 实参。
    #
    # 不要写 prompt_token_ids += draft_token_ids。
    # external draft 若写在 SamplingParams.extra_args 中，就修改此处副本；
    # 若写在 prompt 或 generate 额外实参中，则保留原有位置和字段名。

    return dict(
        prompt=prompt,                        # 搬入的代码构造
        sampling_params=sampling_params,      # 必须返回传入的这一对象
        **other_generate_kwargs,              # 搬入的其他实参；没有则用 {}
    )
```

`np` 用服务已有的 `import numpy as np`。`other_generate_kwargs` 不含 `prompt/sampling_params/request_id`，也不能混入未提交的旧草稿版本。`model_context` 是小字典，携带这段代码真正读取的语言、context、模板等；不放整个 state、engine、processor 或设备 tensor。后台打包只读取这个快照；processor 和只读模型常量可以从 `self` 取。

**这里没有猜测你的 external draft 字段名。** 函数职责摘要不足以还原自定义协议；本模板的注释区必须用你已经工作的代码填上。测试里的 `test_draft` 只是 mock 字段，不能搬到服务。

正式路径也复用 `_pack_stream_request`，避免两个版本漂移。它保留原预算和输出处理；不要让正式请求调用限定 shadow 预算的 `_build_stream_request`：

```python
request_kwargs = self._pack_stream_request(
    full_audio_pcm_f32le=complete_audio.astype("<f4", copy=False).tobytes(),
    stable_prefix_token_ids=stable_prefix_token_ids,
    draft_token_ids=draft_token_ids,  # 第五节选择就绪候选或原回滚草稿
    model_context=model_context,
    sampling_params=deepcopy(formal_sampling_params),
)
stream = engine.generate(**request_kwargs, request_id=formal_request_id)
# stream 交回原有 _yield_delta_text 迭代逻辑；其后状态写回保持原位置。
```

上述局部变量映射到原函数已有值，`deepcopy` 来自 `copy`。保留前缀/回滚草稿的拆分规则也沿用原代码：检查 `unfixed_chunk_num` 预热、`unfixed_token_num` 与 `kmax`，在中点按“下次正式请求”规则只读计算，不能为获得草稿先执行一次正式状态回滚。

#### 4.3 请求快照与三个 helper 的完整函数体

下面直接展示 `streaming_prespec_helpers.py` 的实现。使用第一节的 mixin 时不必再复制这些函数到模型类。

```python
import asyncio
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

if __package__:
    from .streaming_prespec import LaunchDecision, ShadowRequest
else:
    from streaming_prespec import LaunchDecision, ShadowRequest


@dataclass(frozen=True, kw_only=True)
class ModelShadowRequest(ShadowRequest):
    # Private copies of small request settings, never the live ASR state/engine.
    model_context: dict[str, Any]
    sampling_params: Any


class PreSpecModelHelpers:
    """Mixin for Qwen3ASRModelAsync, or copy these three methods into that class.

    Set ``prespec_engine`` to the existing shared engine before enabling.
    Supply ``_pack_stream_request`` as documented; it must not mutate live state.
    ``prespec`` is the existing PreSpecCoordinator, or None when disabled.
    """

    def try_start_prespec(
        self, *, ws_id, seg_id, prefix_version, full_audio_pcm_f32le,
        current_audio_end_samples, last_formal_audio_end_samples,
        chunk_size_samples, stable_prefix_token_ids, rollback_draft_token_ids,
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
        if not rollback_draft_token_ids:
            return LaunchDecision(False, "no_rollback_draft")
        # Take the time before copying so snapshot overhead consumes the budget.
        deadline_ns = coordinator.clock() + int(
            (chunk_size_samples - new_samples) / 16000 * 1e9)
        audio_bytes = bytes(full_audio_pcm_f32le)
        if not audio_bytes or len(audio_bytes) % 4:
            raise ValueError("audio must be complete mono float32 LE PCM")
        request = ModelShadowRequest(
            session_id=ws_id, audio_epoch=seg_id,
            base_prefix_version=prefix_version,
            visible_audio_end_samples=current_audio_end_samples,
            window_audio_samples=new_samples,
            audio_pcm_f32le=audio_bytes,
            stable_prefix_tokens=tuple(stable_prefix_token_ids),
            rollback_tokens=tuple(rollback_draft_token_ids),
            deadline_ns=deadline_ns,
            model_context=deepcopy(model_context),
            sampling_params=deepcopy(sampling_params),
        )
        return coordinator.maybe_start(request, load, predicted_draft_ms)

    def _build_stream_request(self, request, budget):
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
            stable_prefix_token_ids=list(request.stable_prefix_tokens),
            draft_token_ids=list(request.rollback_tokens[:budget]),
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
- `_build_stream_request` 给每次 shadow 建立参数副本，设置累计输出、单序列及总 token 上限；已有回滚草稿最多传入该上限。
- `_run_prespec` 只取引擎给出的**已验证输出**，累计输出用覆盖赋值，不重复拼接；流异常结束时不发布半截候选。
- 提前 commit/正式 chunk 导致取消时，只 abort 自己的 `prespec-...` 请求。到预估下一正式提交时间仍未结束也会取消；正式路径没有 await shadow 的操作。

截止时间覆盖剩余生成时间，并扣除快照/打包已经花掉的时间；同步 processor 工作仍会占用事件循环，不能通过 `create_task` 或 timeout 自动抢占，应把这部分 CPU 成本计入实验。

#### 4.4 两个必须明确维护的音频计数

在 `ASRStreamingState` 的字段定义中新增：

```python
prespec_last_formal_audio_end_samples: int = 0
```

此处示例统一使用**段内采样坐标**：段首为 0，`current_audio_end_samples` 是完整段音频快照的样本数。每个正式 chunk 取得它实际要发送的完整音频后、提交 engine 前记录：

```python
state.prespec_last_formal_audio_end_samples = this_chunk_audio_end
```

中点只读取这个字段。假设已有正式输入长 32000 samples，目前完整音频长 48000，则新增量是 16000；shadow 向引擎发送全部 48000。shadow 完成后该字段仍是 32000。第五节的正式领取同样使用段内 `this_chunk_audio_end`；不要一处段内、一处会话绝对坐标。第六节 commit 入口不领取候选，接收端总数不能用于这两项计数。

#### 4.5 第三节挂点具体传什么

在第三节的挂点，先通过模型现有代码**只读取得**下表的五组值；拿到后调用如下。`complete_audio` 等是局部别名，不能据此假定服务存在同名字段：

| 实参来源 | 取值方式 |
|---|---|
| `complete_audio` | 截至当前已经消费音频的完整段音频。复用现有全量构造规则，不猜 `buffer` 含义，不按缓存命中截断 |
| `stable_prefix_token_ids / rollback_draft_token_ids` | 按下次正式请求的既有回滚规则，只读计算；不生效时草稿传空列表 |
| `model_context` | 原打包代码需要的语言/context/prompt 小配置 |
| `formal_sampling_params` | 本段本次请求原本使用的采样参数；helper 会复制 |
| `chunk_size_samples` | 本段实际 chunk 大小，例如 2 s = 32000 |

若这些值目前只在模型函数内部可见，把“读取这五组值 + 以下调用”放回模型的只读包装方法，连接层仅传 state 和队列/commit 状态；不要在连接层重新实现 tokenizer 或回滚算法。五组值的实际字段名尚需按服务源码映射，这部分没有伪造可以直接访问的 state 成员。

```python
if self.prespec is not None and not segment.committed and segment.audio_queue.empty():
    decision = self.asr.try_start_prespec(
        ws_id=ws_id, seg_id=seg_id,
        prefix_version=state.prespec_prefix_version,
        full_audio_pcm_f32le=complete_audio.astype("<f4", copy=False).tobytes(),
        current_audio_end_samples=int(complete_audio.size),
        last_formal_audio_end_samples=state.prespec_last_formal_audio_end_samples,
        chunk_size_samples=chunk_size_samples,
        stable_prefix_token_ids=stable_prefix_token_ids,
        rollback_draft_token_ids=rollback_draft_token_ids,
        model_context=model_context,
        sampling_params=formal_sampling_params,
        committed=segment.committed,
        audio_queue_empty=segment.audio_queue.empty(),
        load=None,               # 第一节的 force 实验模式
        predicted_draft_ms=0.0,   # force 不使用它；gated 时必须换成实测预算
    )
    # 可把 decision.launched/reason 接入原有日志。
```

#### 4.6 输出接口假设与当前验证范围

helper 使用 `engine.generate(**kwargs)` 返回异步生成器、`await engine.abort(request_id)` 取消、`output.outputs[0].token_ids` 返回 prompt 之后的 token，以及 `output.finished` 表示结束的接口。对应公共接口参考 [vLLM AsyncLLM](https://docs.vllm.ai/en/stable/api/vllm/v1/engine/async_llm/)；累计输出选择见 [SamplingParams](https://docs.vllm.ai/en/stable/api/vllm/sampling_params/)。本地 fork 若改变了这些接口，需调整 `_run_prespec` 的相应取值，不能把文本重新 tokenize 代替原始输出 IDs。

普通 vLLM 输出不包含 prompt token，因此代码没有再切掉 `len(stable_prefix_token_ids)`；否则会丢失候选开头。引擎完成标记之后才发布候选。若你修改过输出为“整个序列”或只报告增量，先还原此合同。

shadow 不调用正式转写/finish/`_yield_delta_text`，不发送客户端事件、不修改正式 state；引擎全局统计仍可能包含 shadow，需按 request ID 单独分析。

本地 mock 用例覆盖完整音频、输入隔离、旧草稿传递、纠正后输出的收集、累计输出去重、跳过条件、取消及异常清理。**它验证适配逻辑，不证明真实 verifier 已正确执行，也不证明 910B 上的时延收益。** 实际服务接通后还需看一次旧草稿被拒绝后继续 decode 的日志，并比较 baseline/probe 正式 P95/P99。

### 五、正式 chunk：在函数内部替换草稿来源

挂点在 `async_streaming_transcribe()` 的**每次切 chunk 推理循环内部**：正式音频和保留前缀已确定，external draft 尚未打包。只在外面包一次函数调用会漏掉同次调用的后续 chunk。

**先把连接/段标识传到这个函数，不能假设局部变量已经存在。** 在原 `async_streaming_transcribe()` 参数列表末尾新增可选参数 `prespec_ws_id=None, prespec_seg_id=None`（保留原有参数）；把 `_process_segment()` 中对它的调用改为：

```python
async for item in self.asr.async_streaming_transcribe(
    audio, state, prespec_ws_id=ws_id, prespec_seg_id=seg_id,
):
    await self._send_stream_item(item)
```

这是替换第三节中的原有调用，不是再调用一次转写。如果原调用还有其他参数，全部保留，仅增加上述两个关键字参数；未提交和 commit 后消费完整 chunk 的调用点都要传。模型函数中令 `ws_id = prespec_ws_id`、`seg_id = prespec_seg_id`。下面的 `prefix_version` 明确取 `state.prespec_prefix_version`，初始化和更新见第六节。

```python
# 内部每个正式 chunk：变量/字段名示意，不是完整服务补丁。
prespec = getattr(self, "prespec", None)
if ws_id is None or seg_id is None:
    prespec = None  # 未传标识的旧调用方不使用这条预投机接线
candidate = None
if prespec is not None:
    candidate = prespec.begin_authoritative_chunk(
        ws_id, audio_epoch=seg_id, prefix_version=prefix_version,
        full_audio_end_samples=this_chunk_audio_end,
    )
raw_target_tokens = ()
try:
    draft_token_ids = (candidate.token_ids if candidate is not None
                       else rollback_draft_token_ids)
    # 用 draft_token_ids 调用 4.2 的 _pack_stream_request，再执行原 engine.generate。
    # 这里保留原来的 async for ... _yield_delta_text(...) 与 yield item。
    # 保存这次实际生成的、保留前缀之后且业务回滚之前的原始 token IDs。
finally:
    if prespec is not None:
        evaluation = prespec.end_authoritative_chunk(
            ws_id, audio_epoch=seg_id,
            target_continuation_tokens=raw_target_tokens,
        )
```

仅推理成功时将返回的 Evaluation 计入匹配率。若现有局部变量拿不到 raw token，在 `_yield_delta_text()` 增加可选 `on_final_tokens` 回调，交回最终原始 IDs；不要把异步生成器改成 await 函数，也不要从 delta 文本重新 tokenize。try/finally 必须覆盖实际输出迭代；清理调用者时显式关闭/退出生成器，使 finally 及时执行，不能依赖垃圾回收。

`shadow_only=True` 时 candidate 恒为 None，正式路径仍用原回滚草稿，但协调器会比较就绪预投机候选与正式原始输出。切换消费模式后，预投机候选仍经过现有 verifier 在完整正式音频上重新验证。推理失败/中断的空 token 只用于释放生命周期，不计为候选拒绝率。

### 六、commit 与清理：给出插入代码和字段初始化

#### 新字段由谁初始化

在 `SegmentState` 的定义中只增加以下配对字段。若它是 dataclass/Pydantic 模型，使用字段声明；若是普通 class，在 `__init__()` 内写成 `self.prespec_formal_begun = False`：

```python
prespec_formal_begun: bool = False
```

**版本只保存在 ASRStreamingState，不在 SegmentState 再保存一份。** 在 ASRStreamingState 的字段定义中新增 `prespec_prefix_version: int = 0`；`init_streaming_state()` 创建它时初始值为 0。若该类型在另一个文件定义，字段声明也要加在那个定义处，不能假设允许动态附加属性。在每次正式 chunk 成功写回文本/回滚状态之后增加：

```python
state.prespec_prefix_version += 1
```

语言/context 更新若影响活跃段，也对该段 state 的版本递增；构造请求时若更换了音频上下文起点，应在领取候选前递增。快照和正式领取统一读取 `state.prespec_prefix_version`。`handle_event()` 的 `current_prefix_version` 取这个值，state 尚未创建时取 0（commit 入口不领取候选）。

下面的 `ws_id/seg_id/state/segment/received_samples` 是原服务值的别名，不是新全局变量：分别对应日志中的连接 ID、段号、该段 ASR state、SegmentState、会话已收到的 float32 单声道样本总数。每个 append 按 `audio.size` 累加一次样本数，不能按字节数累加。若当前没有这个计数，在连接初始化时设为 0，并在 append 解码后、入队前更新。

具体插入代码如下（`audio` 对应原代码刚解码得到的 float32 数组）：

```python
# RealtimeConnection.__init__() 中新增
self._prespec_received_samples = 0

# handle_event() 的 append 分支：解码之后、原有入队之前新增
self._prespec_received_samples += int(audio.size)

# handle_event() 的 commit 分支，用它传给 begin_formal
received_samples = self._prespec_received_samples
```

这只是接收端计数；模型推理请求的 `this_chunk_audio_end` 仍须从实际已送入模型的音频范围计算，不能直接拿它代替。

#### `handle_event()`：在现有 `put(None)` 前插入

```python
if self.prespec is not None and not segment.prespec_formal_begun:
    self.prespec.begin_formal(
        ws_id, seg_id, current_prefix_version, received_samples,
        allow_candidate=False,
    )
    segment.prespec_formal_begun = True

# 随后继续你原有的 None 入队代码，不新增第二个 None。
```

#### `_process_segment()`：用最外层 try/finally 包住原函数主体

```python
try:
    # 这里是原有逐块流式处理、commit 后消费、_finish_segment 的完整主体。
    # 不要只包住 _finish_segment；也不要重复调用它。
    ...
finally:
    if self.prespec is not None and segment.prespec_formal_begun:
        try:
            self.prespec.finish_formal(ws_id, seg_id, ())
        finally:
            segment.prespec_formal_begun = False
```

#### `cleanup()`：在生成任务已经退出之后插入

```python
if self.prespec is not None:
    await self.prespec.close_session(ws_id)
```

若原 cleanup 只执行了 `generation_task.cancel()`，还需要等待它退出：复用原有等待逻辑，或在 cancel 后 `await asyncio.gather(generation_task, return_exceptions=True)`。实际任务属性名用原代码的名称；不要等待当前 cleanup 自己所在的 task。关闭顺序见下面的异常路径说明。

#### 异常路径和配对规则

- `handle_event()` 处理真实 commit 时，在注入 None 之前调用 `begin_formal(ws_id, seg_id, version, received_samples, allow_candidate=False)`，随后置 `segment.prespec_formal_begun=True`。重复 commit 不能重复 begin；state 尚未初始化时也要标记忙状态，版本参数可用段的初始值，因为不领取候选。
- `_process_segment()` 最外层 finally 检查上述标志，调用 `finish_formal(ws_id, seg_id, ())` 并清标志。此处包住流式循环与 `_finish_segment()`，所以短尾跳过、正常完成和中途异常都会配对。
- 如果某段已 begin 但入队失败或被取消前根本未进入 `_process_segment()`，由对应异常处理或连接 cleanup 清理；不能期待一个从未进入的 finally 自动执行。
- `cleanup()` 先取消并等待生成任务及其流式生成器退出，再 `await asr.prespec.close_session(ws_id)`，最后做原有队列/统计清理。不要先删协调器计数再让模型 finally 退出。

`_finish_segment()` 的 `<=0.4s` 跳过条件保持原样。commit 已到达时的 residual finish 第一版不消费候选；这也是为什么正式 chunk 的接入必须放在 `async_streaming_transcribe()` 内，而不能只改 finish。

### 七、最小版范围与验证

当前协调器按 `(session_id, audio_epoch)` 去重，最小版继续 `audio_epoch=seg_id`，**每段最多成功启动一次**。选择段内第一个已有合法回滚草稿的中点，例如 2 s 正式结果之后，累计到 3 s 时为 4 s chunk 起草。没有合法回滚草稿的首个窗口跳过；不能跨 VAD 段借用旧 token。`seg-24/chunk-0` 若是新段首次推理，本版不能据其等待日志声称可获得预投机收益。

要覆盖每个 2 s 窗口，下一步需把 `(ws, seg, chunk/window)` 候选 epoch 与段级 commit 生命周期拆开，并在每次 chunk 结束回收对应 epoch。**仅把 `audio_epoch=seg` 改成 `chunk_id` 不够**：chunk 可能跨段重号，且现有 finish_formal 不会回收所有窗口。此改动不属于本次文档适配，当前代码仍是每段一次接口。

首次用共享的单个协调器、一个 active shadow、`shadow_only=True`、预算不超过 kmax。受控低并发实验可用 `admission_mode="force"` 和现有旁路监测，先打通链路；它不保证不干扰正式时延。在线 `gated` 模式仍需新鲜队列/利用率和实测耗时预算，约 2 s 的 `npu-smi` 读数不能冒充默认 200 ms 的 admission 读数。

验收只保留直接相关的检查：

1. 中点出现 `prespec-...` 请求；确认旧草稿有拒绝时，后续 decode 从 correction 状态继续。
2. 正式 chunk submit 的 `ready_shadow_only/pending`、候选连续匹配长度与成本分开记录；shadow 不产生客户端事件、不改正式 state、不结束正式 session。
3. 提前 commit、短尾 skip、断连时，无泄漏请求和未配对计数；注意 Python 取消不等于 NPU kernel 已抢占。
4. 同音频/同并发 baseline 与 probe 比较正式 P95/P99、队列等待及总计算成本；在离线回放验证经 verifier 消费后的 token/提交边界等价，再启用消费模式。

服务源码到位后需确认的具体代码点只有：external draft 的打包字段及 kmax 约束、保留前缀/回滚草稿的实际拆分、raw output token 的含义、现有完整音频构造规则及采样截止计数、engine abort/生成器关闭方式、请求级清理与正式 session 结束的调用边界。KV 的可复用范围由引擎决定，不是服务端适配项。当前描述不能替代这些逐行核对或 910B 验证。
