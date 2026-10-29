# 910B 流式音频预投机验证手册

本手册验证当前 **每个 `(ws_id, seg_id, window_id)` 最多一次中点候选、各连接可并行生成候选** 的实现。A 组是“完整回滚草稿首次验证开启、预投机关闭”；它与更早的“目标先采样首 token”的回滚 baseline 不是同一实现。服务端三个源文件不在此仓库；下面标明应插入的函数与代码，需在 910B 服务项目完成接线。本仓库的 Python mock 测试不能代替真机验证。

## 0. 三组配置与共同前提

| 组 | `--prespec-mode` | 回滚投机 | 含义 |
|---|---|---|---|
| A | `off` | 开 | 完整回滚草稿首次验证 baseline |
| B | `shadow` | 开 | 后台生成候选，正式请求仍首次验证完整回滚草稿 |
| C | `consume` | 开 | 后台生成候选，正式请求在完整音频上经现有 verifier 验证候选 |

A/B/C 保持同一服务代码、模型、图配置、回滚参数、chunk 长度、音频和连接负载。配置变更后重启服务，不能在活跃连接期间替换协调器。**先用 `force` 完成链路和 A/B/C 初次对比**；此时仍检查中点、本连接的正式请求状态、预算、截止时间及显式配置的候选容量，但其他连接的 chunk/commit 不会阻止启动或取消已有候选。默认候选容量参数为 0，协调器不设进程级上限。完成实时负载接线后，再用 `gated` 重跑 B/C，回答“低利用率空档是否存在且可安全利用”；`gated` 仍检查共享设备的队列/利用率。若未接入新鲜 `LoadSnapshot`，`gated` 会返回 `missing_load`；旁路 `npu-smi` 的 1 秒采样不能冒充默认要求 200 ms 内的新鲜准入读数。

## 1. 服务端加入最小日志：逐处修改

以下代码使用 `logger = logging.getLogger(__name__)`；若文件已有 logger，沿用原对象。服务日志格式需带毫秒时间和时区；三个日志来源的主机时钟应同步。日志不记录音频、文本和完整 token 序列，只记录关联键、长度、状态和耗时。

### 1.1 启动 parser → `AsyncLLM()` → `enable_prespec()`

先在服务项目根目录定位已有的 argparse 入口和模型工厂调用，沿用原 parser 与原启动命令：

```bash
rg -n 'ArgumentParser|parse_args|AsyncLLM\(' qwen_asr
```

在现有 `parser.parse_args()` **之前**增加以下参数；保留其余已有参数：

```python
parser.add_argument(
    "--prespec-mode", choices=("off", "shadow", "consume"), default="off",
    help="off=full rollback first verify; shadow=observe; consume=verify candidate",
)
parser.add_argument(
    "--prespec-admission", choices=("force", "gated"), default="force",
    help="force for chain checks; gated requires fresh LoadSnapshot",
)
parser.add_argument(
    "--prespec-candidate-tokens", type=int, default=None,
    help="shadow output max_tokens; omitted means engine kmax",
)
parser.add_argument(
    "--prespec-max-active-shadows", type=int, default=0,
    help="process-wide shadow limit; 0=no coordinator limit; negative is invalid",
)
parser.add_argument(
    "--prespec-rejoin-policy", choices=("aligned", "ngram_1", "ngram_2", "ngram_3"),
    default="ngram_1", help="rejoin after later candidate mismatch",
)
parser.add_argument(
    "--prespec-launch-sec", type=float, default=1.6,
    help="audio seconds within each formal chunk before launching a shadow",
)
```

这里候选预算的 `default=None` 是“未显式指定”的标记：parser 阶段尚未取得引擎 `kmax`。启动时将 `None` 传给 `AsyncLLM()`、`enable_prespec()`；启用预投机后解析为 `kmax`。完整候选含首 token，必须满足 `candidate_tokens <= kmax`；原先 `kmax=15, candidate_budget=16` 的配置应改为预算 15 或提高引擎 `kmax`。启动时间默认 1.6 秒；在 16 kHz、2 秒 chunk 下为 25,600 样本，距正式边界仅 400 ms。实际触发是首个已处理音频达到阈值、队列为空且准入通过的 append，不保证墙钟恰好 1.600 秒。

你现在的调用是 `Qwen3ASRModelAsync.AsyncLLM(**base_config)`。保留 `base_config` 原内容，在同一调用上显式追加 parser 参数：

```python
model = Qwen3ASRModelAsync.AsyncLLM(
    **base_config,
    prespec_mode=args.prespec_mode,
    prespec_admission=args.prespec_admission,
    prespec_candidate_tokens=args.prespec_candidate_tokens,
    prespec_max_active_shadows=args.prespec_max_active_shadows,
    prespec_rejoin_policy=args.prespec_rejoin_policy,
    prespec_launch_sec=args.prespec_launch_sec,
)
```

`base_config` 不能同时含有上述 `prespec_...` 名字，否则 Python 会因重复关键字报错。若原调用的结果变量不是 `model`，保留原变量名。

在 `qwen3_asr_async.py` 的 `AsyncLLM()` 现有类方法签名增加对应显式形参；若签名有 `**kwargs`，把它们放在 `**kwargs` **之前**。不能只在调用处传参，否则可能得到 `unexpected keyword argument`，或落入 `**kwargs` 后被错误转发给 vLLM engine：

```python
prespec_mode: str = "off",
prespec_admission: str = "force",
prespec_candidate_tokens: int | None = None,
prespec_max_active_shadows: int = 0,
prespec_rejoin_policy: str = "ngram_1",
prespec_launch_sec: float = 1.6,
```

这里假定现有入口用 `argparse` 且解析结果名为 `args`；若名字不同，只替换该局部变量名。服务启动脚本不在本仓库，无法填写其余既有 CLI 选项。

先在 910B 服务项目根目录核对当前实际签名：

```bash
rg -n -A 35 'def enable_prespec' qwen_asr/inference/qwen3_asr_async.py
```

这里的 `enable_prespec()` **不是上游已有 API**，而是 [主接入文档步骤 2](streaming_prespec.md) 要在 `Qwen3ASRModelAsync` 中新增的方法。从 `qwen_asr.inference.streaming_prespec` 导入 `launch_samples_for_window`，同步 `admission_mode`、`shadow_only`、`max_active_shadows` 和 `launch_sec` 形参，并把旧的 `max_active_shadows=1` 改为传入参数，否则更换协调器后仍只允许一个候选运行。为记录 shadow 请求生命周期，再把其中的 `producer=self._run_prespec` 替换为 `producer=self._logged_prespec_producer`。完整的关键行是：

```python
def enable_prespec(self, *, enabled, kmax, engine=None, chunk_size_sec=2.0,
                   launch_sec=1.6, candidate_tokens=None, on_probe=None,
                   admission_mode="force", shadow_only=True,
                   max_active_shadows=0, rejoin_policy="ngram_1"):
    # 保留主接入文档中已有的 enabled/kmax/engine 校验。
    if candidate_tokens is None:
        candidate_tokens = kmax
    if candidate_tokens < 3:
        raise ValueError("candidate budget needs first token, draft and continuation")
    if candidate_tokens > kmax:
        raise ValueError("candidate budget must fit engine kmax")
    if rejoin_policy not in ("aligned", "ngram_1", "ngram_2", "ngram_3"):
        raise ValueError("invalid prespec rejoin policy")
    self.prespec_rejoin_policy = rejoin_policy
    midpoint_samples = launch_samples_for_window(launch_sec, chunk_size_sec)
    self.prespec = PreSpecCoordinator(
        producer=self._logged_prespec_producer,
        policy=AdmissionPolicy(
            admission_mode=admission_mode,
            midpoint_samples=midpoint_samples,
            max_rollback_draft_tokens=kmax,
            max_candidate_tokens=candidate_tokens,
            max_active_shadows=max_active_shadows,
        ),
        shadow_only=shadow_only,
        on_probe=on_probe,
    )
```

先在 `qwen3_asr_async.py` 顶部加入 `import asyncio`、`import logging`、`import time` 和模块级 `logger = logging.getLogger(__name__)`（已有同名导入或 logger 时不要重复）。然后在 `AsyncLLM()` 创建 `model` 后、返回它之前，替换原来的 `model.enable_prespec(...)` 调用。`kmax`、`engine` 和 `chunk_size_sec` 继续使用本函数现有的实际变量；下面代码中的这三个名字要按你的函数局部变量映射：

```python
def log_prespec_probe(p):
    logger.info(
        "PRESPEC_PROBE ws_id=%s seg=%s window=%s boundary=%s status=%s "
        "request_id=%s ready_lead_ms=%s",
        p.session_id, p.audio_epoch, p.window_id, p.boundary, p.status,
        p.request_id, p.ready_lead_ms,
    )

if prespec_mode not in {"off", "shadow", "consume"}:
    raise ValueError(f"invalid prespec_mode: {prespec_mode}")
if prespec_admission not in {"force", "gated"}:
    raise ValueError(f"invalid prespec_admission: {prespec_admission}")
if prespec_max_active_shadows < 0:
    raise ValueError("prespec_max_active_shadows must be >= 0")
model.enable_prespec(
    enabled=(prespec_mode != "off"), kmax=kmax, engine=engine,
    chunk_size_sec=chunk_size_sec,
    launch_sec=prespec_launch_sec,
    candidate_tokens=prespec_candidate_tokens,
    admission_mode=prespec_admission,
    max_active_shadows=prespec_max_active_shadows,
    rejoin_policy=prespec_rejoin_policy,
    shadow_only=(prespec_mode != "consume"),
    on_probe=log_prespec_probe if prespec_mode != "off" else None,
)
logger.info(
    "PRESPEC_CONFIG mode=%s admission=%s kmax=%s candidate_budget=%s "
    "max_active_shadows=%s launch_sec=%s launch_samples=%s",
    prespec_mode, prespec_admission, kmax,
    model.prespec.policy.max_candidate_tokens if model.prespec is not None else None,
    prespec_max_active_shadows, prespec_launch_sec,
    model.prespec.policy.midpoint_samples if model.prespec is not None else None,
)
```

`on_probe` 是同步回调；这里只做一条格式化日志，不做阻塞采样或文件读写。回调异常被协调器吞掉并计入 `probe_error_count`，因此每轮结束要检查该计数是否为 0。

### 1.2 `qwen3_asr_async.py`：记录 shadow 请求生命周期

在模型类中新增方法，并让 1.1 的 `producer` 指向它。原 `_run_prespec()` 实现不改，仍负责 engine `generate/abort`：

```python
async def _logged_prespec_producer(self, request, budget):
    started_ns = time.monotonic_ns()
    logger.info(
        "PRESPEC_REQUEST event=submit ws_id=%s seg=%s window=%s request_id=%s "
        "audio_end_samples=%s rollback_tokens=%s budget=%s",
        request.session_id, request.audio_epoch, request.window_id,
        request.request_id,
        request.visible_audio_end_samples, len(request.rollback_tokens), budget,
    )
    try:
        token_ids = await self._run_prespec(request, budget)
    except asyncio.CancelledError:
        logger.info(
            "PRESPEC_REQUEST event=cancel ws_id=%s seg=%s window=%s request_id=%s elapsed_ms=%.3f",
            request.session_id, request.audio_epoch, request.window_id,
            request.request_id,
            (time.monotonic_ns() - started_ns) / 1e6,
        )
        raise
    except TimeoutError:
        logger.info(
            "PRESPEC_REQUEST event=timeout ws_id=%s seg=%s window=%s request_id=%s elapsed_ms=%.3f",
            request.session_id, request.audio_epoch, request.window_id,
            request.request_id,
            (time.monotonic_ns() - started_ns) / 1e6,
        )
        raise
    except Exception:
        logger.exception(
            "PRESPEC_REQUEST event=failed ws_id=%s seg=%s window=%s request_id=%s",
            request.session_id, request.audio_epoch, request.window_id,
            request.request_id,
        )
        raise
    logger.info(
        "PRESPEC_REQUEST event=done ws_id=%s seg=%s window=%s request_id=%s "
        "tokens=%s elapsed_ms=%.3f",
        request.session_id, request.audio_epoch, request.window_id,
        request.request_id,
        len(token_ids), (time.monotonic_ns() - started_ns) / 1e6,
    )
    return token_ids
```

这里的 `done` 只表示 shadow 引擎生成结束，尚不表示正式请求采纳。`asyncio.timeout` 到期会把等待中的 `CancelledError` 转为 `TimeoutError`；这是候选截止时间到达，应记 `event=timeout`，不要归入引擎故障 `event=failed`。**本节日志包装器本身不会收集部分 token**；还必须按[主接入文档 `_run_prespec()` 代码](streaming_prespec.md)在每个 CUMULATIVE `RequestOutput` 到达时调用 `self.prespec.publish_partial(request, candidate_token_ids)`。协调器在超时或 commit 取消后可领取已发布且满足长度、版本和音频截止检查的部分候选；没有可用快照则回退到 rollback。不要在 `except CancelledError` 或 `except TimeoutError` 才去读取局部变量：正式请求不能等待 shadow 的取消处理。若取消后底层 NPU 工作仍持续，另看 vLLM 请求结束/abort 日志和资源占用。

### 1.3 `realtime_connection.py`：记录启动准入决定

在 [第三节](streaming_prespec.md) 的 `_process_segment()` 中，用下列代码替换 `try_start_prespec_for_state(...)` 返回后的简短 `if decision...` 块。该节已在调用前保存 `window_id = segment.asr_state.chunk_id`；`segment.prespec_started_window` 只赋值一次：

```python
if decision is not None:
    log = logger.info if decision.launched else logger.debug
    log(
        "PRESPEC_LAUNCH ws_id=%s seg=%s window=%s launched=%s reason=%s slack_ms=%s",
        self.connection_id, seg_id, window_id, decision.launched,
        decision.reason, decision.slack_ms,
    )
if decision is not None and decision.launched:
    segment.prespec_started_window = window_id
```

`try_start_prespec_for_state()` 在未达启动阈值、已 commit 或音频队列非空时返回 `None`，因此这些路径**不会**出现 `PRESPEC_LAUNCH`。`chunk-0` 没有旧回滚草稿，空草稿有效；在 consume 模式下，正式请求可验证它在默认 1.6 s 音频下生成的候选。真正进入 `try_start_prespec()` 后，旧草稿超出引擎 `kmax` 或输出预算放不下 `1 + 旧草稿长度 + 1` 时，分别返回 `rollback_draft_exceeds_kmax` / `candidate_budget_insufficient`，不会给引擎发缩短后的草稿。这两项准入检查在 shadow 和 consume 模式都执行；两者只在正式请求是否使用已完成候选或已输出部分快照上不同。helper 对每个 `(ws_id, seg_id, window_id, reason)` 只输出一次 `PRESPEC_SKIP` INFO，避免同一窗口的后续 80 ms chunk 重复刷屏。若预期到达启动阈值却没有 launch，请在包装函数这些 `return None` 前临时加一条 `DEBUG` 日志，包含 `ws_id/seg/window_id/len(state.buffer)/reason`；不要据“无 launch 日志”直接判断负载门控拒绝。

### 1.4 `qwen3_asr_async.py`：关联正式请求与候选

在 [第五节](streaming_prespec.md) 的 `begin_authoritative_chunk()` 前保存 `this_chunk_id = state.chunk_id`；在正式请求已构造好 `request_id`、紧邻现有 `engine.generate(...)` 前记录候选来源。下面的 `formal_request_id` 要换成正式请求当前使用的实际局部变量（如现有 `stream_...` ID），不能填 shadow 的 `prespec-...` ID：

```python
logger.info(
    "PRESPEC_TARGET ws_id=%s seg=%s chunk=%s event=submit "
    "formal_request_id=%s candidate_request_id=%s "
    "candidate_raw_tokens=%s candidate_partial=%s submitted_draft_tokens=%s",
    ws_id, seg_id, this_chunk_id, formal_request_id,
    candidate.request_id if candidate is not None else None,
    len(candidate.token_ids) if candidate is not None else None,
    candidate.is_partial if candidate is not None else None,
    len(draft_token_ids),
)
```

正式 chunk 的原 `finally` 只需与 `begin_authoritative_chunk()` 配对；不新增 `raw_target_tokens`，也不修改 `_yield_delta_text()`：

```python
if prespec is not None:
    prespec.end_authoritative_chunk(
        ws_id, audio_epoch=seg_id, window_id=this_chunk_id)
```

不传 `target_continuation_tokens` 表示**只清理候选，不计算匹配长度**，协调器返回 `None`；不能把此结果记为“接受 0 个 token”。C 组仍须在引擎 verifier 处逐轮记录 `request_id`、实际有效草稿长度与接受数。正式请求的预投机候选和完整回滚草稿都应由首次 prefill 验证首 token；只有 shadow 请求保留先采样首 token 的旧草稿合同。只在 proposer 的 `propose()` 打日志不足以证明草稿被验证。

```bash
rg -n 'ascend_external_draft|def propose\(|accepted.*token|rejected.*token' .
```

还要在 910B 的 scheduler 与 verifier 处确认首轮顺序。预投机消费时正式日志应满足 `candidate_raw_tokens = submitted_draft_tokens`；无候选且完整回滚草稿非空时，提交长度应为该完整草稿长度。两者都应看到同一正式请求的 `INITIAL_DRAFT status=scheduled draft_tokens=submitted_draft_tokens`。预算或 encoder 调度导致 `INITIAL_DRAFT status=fallback` 的请求应单列，不能算作首轮并行验证。

**0.23.0 worker 记账顺序与重接修正。** `model_runner_v1.py` 先调用 `_bookkeeping_sync()`：它把本轮 verifier 确认的 `valid_sampled_token_ids` 写入 `token_ids_cpu`，并把 `num_tokens_no_spec` 更新到正式前缀末尾；之后才调用 `AscendExternalDraftProposer.propose()`。因此首次验证后的重接必须读取 `token_ids_cpu[prompt_len:num_tokens_no_spec]`，不能再次写入 `sampled_ids`，也不能用 `num_tokens_no_spec + len(sampled_ids)` 当作已确认前缀末尾。旧的一次性 proposer 分支只供 shadow 或未迁移的正式请求使用。首次验证在第一个 token 失配后允许一次按位置续接；若该续接只得到一个替换 token（零草稿接受），则停止继续按位置提交更短的剩余草稿，以免每轮增加一次无收益的 verifier。`EXTERNAL_DRAFT_REJOIN` 的 `confirmed_tokens` 应以该正式请求的实际已确认输出为准，后续同一请求不得因重复写入而虚增或倒退。910B 需要先确认已部署 `external_draft_first_verify_v023.patch` 并重启引擎；仅更新服务端代码不能改变 worker 行为。旧负载的收益日志不可与修正后直接混算。

当前 `ASR_VERIFY_STEP` 示例在同一个请求的多次 verifier 调用中反复输出 `round_index=0`。它表示每次调用内的索引，不能直接当作请求级验证轮次；旧版 `streaming_prespec_gain.py` 以 `(request_id, round_index)` 唯一为前提，会把这种日志判为重复。重新统计前须把 verifier 打点改为请求级递增轮次，或让分析脚本按该请求日志出现顺序编号，并区分初次草稿和续接草稿。

若需要**候选与正式输出的离线匹配统计**，不能把旧示例中的 `raw_target_tokens = ()` 原样传入：元组没有采集动作，会一直表示“已观测到空输出”，导致错误的 0 匹配。正式请求仍保持 `output_kind=DELTA`，在原 `_yield_delta_text()` 消费的**同一条正式 vLLM 输出流**上采集原始 token IDs。可给 `_yield_delta_text()` 增加一个可选的 `on_output` 回调，并在它原有的 `async for output in stream` 循环内、处理文本之前调用；不要另起一次 generate，也不要对输出文本重新 tokenize：

```python
# _yield_delta_text 的原有参数保持不变，仅追加 on_output=None。
# 在原有 async for output in stream 循环内：
if on_output is not None:
    on_output(output)
# 下方保留原有的文本增量、计时、缓存释放和 yield 逻辑。
```

正式 chunk 的调用点每轮新建一个收集器，并把回调传给**原有那次** `_yield_delta_text()` 调用（其他实参照旧保留）：

```python
raw_target_tokens = []
formal_output_finished = False

def capture_formal_output(output):
    nonlocal formal_output_finished
    # 正式采样只有一个 completion，DELTA 的 token_ids 是本次新增 token。
    if output.outputs:
        raw_target_tokens.extend(output.outputs[0].token_ids)
    if output.finished:
        formal_output_finished = True

try:
    # 原有 prompt、SamplingParams、engine.generate 与客户端 delta 流程。
    async for delta in self._yield_delta_text(
        stream,  # 其余原有位置/关键字参数保持不变
        on_output=capture_formal_output,
    ):
        yield delta
    # 原有 _raw_decoded / chunk_id 状态写回。
finally:
    if prespec is not None:
        evaluation = prespec.end_authoritative_chunk(
            ws_id,
            audio_epoch=seg_id,
            window_id=this_chunk_id,
            target_continuation_tokens=(
                tuple(raw_target_tokens) if formal_output_finished else None
            ),
        )
        if evaluation is not None:
            logger.info(
                "PRESPEC_MATCH ws_id=%s seg=%s chunk=%s candidate_len=%s "
                "target_len=%s raw_matched_prefix_tokens=%s "
                "draft_matched_prefix_tokens=%s",
                ws_id, seg_id, this_chunk_id,
                len(evaluation.candidate_tokens),
                evaluation.target_token_count,
                evaluation.raw_matched_prefix_tokens,
                evaluation.draft_matched_prefix_tokens,
            )
```

这段仅展示采集与统计挂点，不能替换原函数的其他实参或原有状态写回。只有观察到 `output.finished` 才传入 token 列表；中断/取消时传 `None`，避免把部分输出或尚未采集的 `()` 算作 0 匹配。完成且确实生成 0 个 token 时，空元组才是有效观测。`raw_matched_prefix_tokens` 比较完整候选与正式输出，包含首 token；旧字段 `draft_matched_prefix_tokens` 仍比较两个序列各自的 `[1:]`，只为保持已有离线报告可比，**不代表现在实际提交的草稿**。两者都不是 verifier 实际接受数；后者从引擎逐轮打点取得。

finish 复用 `_split_stream_rollback()` 处理 `\ufffd` 时，`on_output` 本身不用改：它只采集 prompt 之后的原始生成 token，不读取或重算稳定前缀。要看 **提前 commit 的 finish 候选匹配统计**，优先按[主接入文档第六节](streaming_prespec.md)让 DELTA 输出的 finish 复用 `_yield_delta_text()` 消费同一条 `finish_...` 流，并通过适配回调收集原始 token IDs。`on_output` 接收单次 `RequestOutput`，不能直接传接收最终 token 元组的 `on_finish_tokens`。完成后的元组通过 `_finish_segment()` 存到本段，再由 `_process_segment()` 的 finally 传给 `finish_formal(..., target_continuation_tokens=...)`。仅在观察到 `output.finished` 后传 token 列表，中断时传 `None`；原 finish 若使用 CUMULATIVE/FINAL_ONLY，先核对 `_yield_delta_text()` 的输出语义，不能直接按 DELTA 累积。这个回调不会改变客户端转写文本或正式请求的输入。

910B 服务接入并重放一个在完整 chunk 到来前提前 commit、且剩余音频超过 0.4 秒的段后，查同一 `ws_id/seg/window` 的日志：

```bash
rg 'PRESPEC_(PROBE|TARGET|MATCH)' /path/to/server.log | rg 'ws_id=ws-实际ID seg=实际段号'
```

shadow 模式应看到 `PRESPEC_PROBE ... boundary=formal status=ready_shadow_only`（取消前已有可用快照则为 `partial_ready_shadow_only`）、`PRESPEC_MATCH_TOKENS` 和 `PRESPEC_MATCH ... boundary=formal`；consume 模式相应为 `ready_consumed` / `partial_ready_consumed`，并在 `PRESPEC_TARGET ... boundary=formal` 看到候选 request ID。`PRESPEC_MATCH` 同时给出 `raw_matched_prefix_tokens` 和 `draft_matched_prefix_tokens`。若 finish 被短尾规则跳过、候选没有可用输出或窗口变更，不会有该轮匹配统计；若正式输出未完成，也不能把部分输出作为 0 匹配。

协调器收到完整的 `target_continuation_tokens` 且已选中候选时，会另外打印 `PRESPEC_MATCH_TOKENS`。`candidate_token_ids` 是正式边界选中的候选原始 token IDs；`candidate_partial=True` 表示 shadow 被取消前最后一个已输出快照，`False` 表示完整结束的候选。更新后的 `target_token_ids` 是**完整**正式原始输出 token IDs，`target_len` 应等于其长度；旧版日志只打印到候选长度，不能从旧日志恢复尾部。例如 `candidate_token_ids=(19, 20, 21, 22)`、`target_token_ids=(19, 20, 99, 77, 55)`、`target_len=5`、`raw_matched_prefix_tokens=2`，表示前两项相同，第三项 `21` 与 `99` 不同，后面的正式 token 也已保留。日志的 `mode=shadow/consume` 标明正式请求是否使用候选；consume 模式下两组序列的匹配不代表无候选的 baseline 结果。若未采集正式 token、正式生成中断，或没有可用候选，就不会出现该日志。它同样不是 verifier 接受数。

日志关联以 `ws_id + seg + chunk` 为正式键，以 `prespec-...` 为 shadow 引擎请求键；不要把不同 request 的 prefill/decode 打点混在一起。`PRESPEC_PROBE status=ready_consumed` 发生在正式验证**之前**，不能代替 verifier 日志。

先把本仓库的两个模块复制到服务的 `qwen_asr/inference/`，并**在同一次部署中**按主接入文档更新连接层、模型包装函数和正式 chunk 的 `window_id` 实参。新协调器要求 `begin_authoritative_chunk`、`end_authoritative_chunk`、`begin_formal`、`finish_formal` 携带窗口 ID；只替换模块而保留旧调用会报错。监测脚本复制到测试工作目录；若两仓库在同一主机上，在服务项目根目录执行（替换检出路径）：

```bash
cp /path/to/vllm-ascend/research/streaming_prespec.py qwen_asr/inference/
cp /path/to/vllm-ascend/research/streaming_prespec_helpers.py qwen_asr/inference/
cp /path/to/vllm-ascend/research/streaming_prespec_monitor.py ./
```

完成服务文件修改及复制后，在服务项目根目录检查语法和挂点，再启动 910B 服务：

```bash
python -m py_compile qwen_asr/inference/streaming_prespec.py \
  qwen_asr/inference/streaming_prespec_helpers.py \
  qwen_asr/inference/qwen3_asr_async.py \
  qwen_asr/cli/realtime_connection.py
rg -n 'PRESPEC_CONFIG|PRESPEC_LAUNCH|PRESPEC_SKIP|PRESPEC_REQUEST|PRESPEC_PROBE|PRESPEC_TARGET' \
  qwen_asr/inference/qwen3_asr_async.py qwen_asr/inference/streaming_prespec_helpers.py \
  qwen_asr/cli/realtime_connection.py
```

## 2. 单连接链路检查：B 组先强制 shadow

在 910B 服务项目完成第 1 节的代码接线、复制与语法检查后，按 [主接入说明](streaming_prespec.md) 导入。以服务现有启动命令启动一份 B 组实例：

```bash
<现有服务启动命令> --prespec-mode shadow --prespec-admission force --prespec-launch-sec 1.6
```

使用现有 WebSocket 测试脚本发送**同一段超过 6 秒、6 秒前不 commit**的音频；这一步并发为 1，不开启自动补位。服务若写日志文件：

```bash
tail -F /path/to/server.log | rg --line-buffered 'PRESPEC_|inference submit|Commit received|Segment done'
```

若服务由 systemd 管理，改用 `journalctl -u <实际服务单元名> -f -o cat` 作为管道左侧。先用**连续超过 6 秒且 6 秒前不 commit 的单段音频**核对三个窗口：默认约 1.6、3.6、5.6 秒分别看到 `window=0/1/2` 的 `PRESPEC_LAUNCH/REQUEST`，约 2、4、6 秒分别看到对应 `window=0/1/2 boundary=chunk status=ready_shadow_only`，或候选仍在运行但已输出快照时的 `partial_ready_shadow_only`。三个窗口的 `request_id` 应不同，正式 `chunk=0/1/2` 分别只关联同号窗口；任一窗口跳过时不应误领前一窗口候选。`ready_lead_ms > 0` 表示候选早于对应正式请求提交完成；若为 `pending`，候选被取消，不应等待它。若是 `candidate_exceeds_kmax`，shadow 的输出超过正式 proposer 可接收长度，协调器不会把它交给正式请求。随后分别重放提前 commit、≤0.4 秒短尾和断连，确认无遗留 `prespec-...` 请求、重复 begin 或未结束的 session。约 5.95 秒就 commit 的段在 5.6 秒启动 `window=2`，若未凑足 6 秒正式 chunk，应在实际 `finish_...` 请求提交时看到 `PRESPEC_PROBE window=2 boundary=formal status=ready_shadow_only` 或 `partial_ready_shadow_only`；consume 模式则应为 `ready_consumed` 或 `partial_ready_consumed`，且 `PRESPEC_TARGET boundary=formal` 的 `candidate_request_id` 对应先前的 `prespec-...`。commit 到达时保留已完成候选或取消前已输出的部分快照，不立即打印 formal probe；若后续完整 chunk 先领取了它，finish 不得重复领取。短尾跳过 finish 时也不会出现 formal probe，外层清理会丢弃候选。

上段的 `pending` 只表示取消时没有可提交的部分草稿。接入 `publish_partial()` 后，运行中的候选若已输出足够长的快照，finish 的 probe 会改为 `partial_ready_shadow_only` 或 `partial_ready_consumed`；部分草稿不长于原回滚草稿时为 `partial_not_longer_than_rollback`，继续回滚。consume 模式下，服务端 `PRESPEC_TARGET candidate_partial=True` 及同一正式请求的 `ASR_VERIFY_STEP effective_draft_tokens>0` 要同时出现，才可判断该部分候选进入了实际 verifier。`PRESPEC_REQUEST event=cancel` 本身只表示后台请求被取消，不能据此断定它没有已输出 token。

如果链路没有 launch，打开 DEBUG 日志后先看 `return None` 的跳过原因和 `PRESPEC_LAUNCH reason`；`force` 仅绕过实时负载门控，不绕过中点、草稿、本连接的在途正式请求和显式开启的容量检查。`discarded_before_boundary` 表示本窗口启动过、但候选在边界前已被丢弃；它替代含义模糊的 `previously_used`，不表示正式请求已使用或接受候选。

单轮结束后可快速检查事件是否齐全、拒绝原因分布；路径替换为该轮切出的服务日志：

```bash
rg -c 'PRESPEC_LAUNCH.*launched=True|PRESPEC_REQUEST event=done|PRESPEC_PROBE.*ready_' /path/to/run.log
rg 'PRESPEC_SKIP|PRESPEC_LAUNCH' /path/to/run.log | rg -o 'reason=[^ ]+' | sort | uniq -c
```

### 2.1 多连接候选并发与生命周期隔离

同步协调器及上面的参数接线，重启服务：

```bash
<现有服务启动命令> --prespec-mode shadow --prespec-admission force \
  --prespec-max-active-shadows 0
```

用现有 `sustained_test` 运行并发 5、20 档，保留本轮完整日志；另用固定回放安排连接 A 在连接 B 的中点候选运行期间提交 commit。客户端参数按现有脚本填写。先查实际启动配置和候选生命周期：

```bash
rg 'PRESPEC_CONFIG|PRESPEC_REQUEST|PRESPEC_PROBE|Commit received|inference submit' /path/to/run.log
rg 'formal_inflight_on_device|shadow_capacity_full|previously_used' /path/to/run.log
```

第二条命令在本次 `force + max_active_shadows=0` 运行中应没有结果；若仍出现，核对实际加载的协调器以及服务中是否残留 `max_active_shadows=1`。本连接仍可能返回 `session_formal_inflight` / `session_chunk_inflight`，同窗口已启动则返回 `epoch_already_used`，这些不代表全局串行。

下面命令统计 producer 日志中同时运行的候选数量；必须输入单个服务进程的一轮完整日志，并在测试结束、连接清理后执行：

```bash
python - /path/to/run.log <<'PY'
import re
import sys

active = {}
peak = 0
peak_sessions = 0
with open(sys.argv[1], encoding="utf-8") as log:
    for line in log:
        if "PRESPEC_REQUEST " not in line:
            continue
        fields = dict(re.findall(r"(\w+)=([^\s]+)", line))
        request_id = fields.get("request_id")
        if fields.get("event") == "submit":
            active[request_id] = fields["ws_id"]
            peak = max(peak, len(active))
            peak_sessions = max(peak_sessions, len(set(active.values())))
        elif fields.get("event") in {"done", "cancel", "failed"}:
            active.pop(request_id, None)
print({"peak_shadow_requests": peak,
       "peak_shadow_sessions": peak_sessions,
       "unfinished_request_ids": sorted(active)})
PY
```

安排中点重叠后，`peak_shadow_sessions > 1` 可确认多个连接的 producer 生命周期重叠；它不代表 NPU kernel 同时执行。完整日志的 `unfinished_request_ids` 应为空。按同一 `ws_id/seg/window` 继续检查：

- B 的候选尚在运行时，A 的 chunk/commit 不应取消 B；B 自己到达正式边界或截止时间时仍可取消。
- B 已出现 `event=done` 后，A 的正式请求不能删除 B 的候选；B 的窗口/版本/音频边界保持有效时，应在自己的正式边界看到 `ready_shadow_only`，consume 重跑时为 `ready_consumed`。
- A 断连清理只影响 A；B 正常产出匹配统计。B 自己的 `pending`、窗口变化、版本变化或超时仍属于合法回退。

如需显式设协调器上限，改传 `--prespec-max-active-shadows 4`；这时 `shadow_capacity_full` 是预期容量保护。切换 `gated` 后仍会根据共享设备负载拒绝候选，不能把该拒绝解释为跨连接生命周期互斥。并发能力确认后，按第 4 节比较同负载正式 P95/P99，不能由并发日志直接推断没有资源争用。

## 3. 固定输入正确性：A、B、C 逐级比较

对同一批音频保存**每条连接的音频文件、切片顺序、发送时刻和 commit 时刻**。三组分别重启服务并使用同一回放清单：

```bash
# A：关闭预投机，正式请求首次验证完整回滚草稿
<现有服务启动命令> --prespec-mode off
# 使用现有 WebSocket 回放命令，保存客户端事件和服务日志

# B：后台产生候选供观测，正式请求仍首次验证完整回滚草稿
<现有服务启动命令> --prespec-mode shadow --prespec-admission force --prespec-launch-sec 1.6
# 同一回放清单

# C：先仅在固定输入回放中启用消费
<现有服务启动命令> --prespec-mode consume --prespec-admission force --prespec-launch-sec 1.6
# 同一回放清单
```

这里的 `<现有服务启动命令>` 和 WebSocket 回放命令是环境专属的唯一占位符：本仓库没有 910B 服务启动脚本或你的 `test_v2.py` 参数表，不能编造 CLI。A/B 应逐 chunk 对比已有的客户端 delta/done、最终文本与 commit 边界；若引擎本来就记录正式原始 token IDs，可额外对比，但不为此改造流式输出函数。C 还要确认候选的后续 token 实际写进**正式 chunk 与提前 commit 的 `finish_...` 请求**的 `ascend_external_draft` 并由引擎 verifier 验证；分别覆盖首个草稿 token 拒绝、部分接受、全拒绝、版本不匹配和候选超时。提前 commit 的 `PRESPEC_TARGET boundary=formal` 必须显示真实 `finish_...` 请求 ID、候选 ID 和提交草稿长度；短尾跳过或窗口号已改变时不得提交候选。此步不通过就不要开始性能对比。

## 4. 同负载性能对比：A→B、B→C、A→C

先用 B/C 的 `--prespec-admission force` 完成初次同负载对比；这一步只能衡量后台请求的收益与干扰，**不能证明低利用率准入策略有效**。分别跑现有 `sustained_test` 的并发档位 `1、5、20`，再扩到拐点附近；每组保持相同音频清单、到达时序、自动补位规则、持续时间和预热。若要单独评估晚启动，将 B/C 的 `--prespec-launch-sec` 在 `1.0` 与 `1.6` 间切换，其他参数固定，并报告候选 `ready_*` 覆盖率、首 token/后缀接受数、正式请求 queue wait 和 commit P95/P99。A/B/C 交错排序，至少重复 3 轮，记录实际测试起止 UTC。每轮单独设置 `run_id`、切片服务日志、导出该轮 `commits.jsonl`。导出行须含 `ws_id, seg, received_at, done_at, commit_latency_ms, queue_wait_ms, queue_size, is_empty_commit`；空 commit 不算正式时延。客户端执行命令仍用你现有 `test_v2.py` 的真实 CLI。

与负载同时启动旁路采样；下面以 A、并发 20、第一轮为例，`--duration` 应覆盖测试加启停等待：

```bash
python streaming_prespec_monitor.py sample \
  --run-id rollback-c20-r1 \
  --metrics-url http://127.0.0.1:8000/metrics \
  --npu-cmd-json '["npu-smi","info"]' \
  --duration 330 --metrics-interval 0.1 --interval 1 \
  --out rollback-c20-r1-samples.jsonl &
monitor_pid=$!

# 立即运行你的现有 test_v2.py 并发 20 档，记录其真实开始/结束 UTC。
# 测试和冷却结束后，等监测进程完成：
wait "$monitor_pid"
```

`npu-smi` 实际输出若不能被默认规则解析，按 [监测说明](streaming_prespec_monitor.md) 加 `--npu-util-regex` 并指定目标 NPU；确认 `npu_missing_fraction` 不为 1。采样、正式负载、服务日志必须属于同一轮。测试结束后按实际 UTC 窗口汇总：

```bash
python streaming_prespec_monitor.py summarize \
  --run-id rollback-c20-r1 --concurrency 20 \
  --samples rollback-c20-r1-samples.jsonl \
  --commits rollback-c20-r1-commits.jsonl \
  --test-start-utc 2026-10-21T01:00:00Z \
  --test-end-utc 2026-10-21T01:05:00Z \
  --out rollback-c20-r1-summary.json

python streaming_prespec_monitor.py compare \
  --baseline rollback-c20-r1-summary.json \
  --probe shadow-c20-r1-summary.json \
  --out shadow-vs-rollback-c20-r1.json

python streaming_prespec_monitor.py compare \
  --baseline rollback-c20-r1-summary.json \
  --probe consume-c20-r1-summary.json \
  --out consume-vs-rollback-c20-r1.json
```

上面两个 UTC 是**格式示例**，必须改成真实起止时间；B/C 也各自运行 `summarize` 后才能 `compare`。`--aligned-out` 可省略。另用同一批汇总比较 B→C，便于区分后台开销和消费收益。`compare` 只算描述性差值，不自动宣称通过。

### 4.1 正式输出 token、解码轮次与 verifier 接受数

这里按**正式 vLLM 请求**统计。服务端的 `ASR_FORMAL_STATS.request_id` 是传给 `engine.generate()` 的外部 `stream_...` / `finish_...` ID；vLLM 默认再给内部 ID 追加 `-` 和 8 位十六进制随机后缀，调度器的 `req_id` 因而可能是 `stream_..._1-8934184b`。两者须按一次请求关联，不能直接按字符串相等合并。A（rollback）、B（shadow）、C（consume）都要打开相同打点。`output_token_count` 是该请求完成时，`DELTA` 输出中 `token_ids` 的累计长度，包含模型原始输出的特殊 token；它不是最终转写文本重新分词后的长度。`decode_rounds` 沿用 `_yield_delta_text()` 现有的 `token_count` 计数器，记录解码轮次。两者不能互换；一次轮次可能产出多个 token。跳过 finish 推理的 commit 没有正式请求，仍只在 `commits.jsonl` 中计入对应 commit 时延，不能给它补一条零轮次请求。

在服务项目 `qwen_asr/inference/qwen3_asr_async.py` 中，沿用 1.4 节的 `on_output` 收集器：它必须对 A/B/C 的**每次正式请求**运行，而不以 `prespec is not None` 为条件。正式输出完成后，`output_token_count = len(raw_target_tokens)`；只对 `output.finished` 为真的请求写一行。为取到现有局部计数器，在 `_yield_delta_text()` 原来计算/打印 `token_count`、`prefill_ms`、`decode_total_ms` 的位置，把这些数写入本次调用传入的可选字典 `formal_metrics`（需给函数签名和调用点加 `formal_metrics=None`）：

```python
# _yield_delta_text() 中，在原有三个量都已计算之后：
if formal_metrics is not None:
    formal_metrics.update(
        decode_rounds=token_count,  # 现有计数器：解码轮次
        prefill_ms=prefill_ms,
        decode_total_ms=decode_total_ms,
    )
```

正式 chunk 和 finish 的原调用点各建一个 `formal_metrics = {}`，把它与已有 `on_output` 一起传给同一条 `_yield_delta_text(...)` 流。**当前接入示例尚未定义 `draft_source` 或下面统计用的 `boundary`；910B 服务端必须在两处正式请求打包完成后显式赋值。** 正式 chunk 的 `boundary = "chunk"`，正式 finish 的 `boundary = "finish"`；短尾跳过 finish 时没有正式请求，也不写本条统计日志。协调器的 `PRESPEC_PROBE boundary=formal` 表示 commit 边界，是另一套命名，不能直接填入这里只接受 `chunk/finish` 的分析脚本。

两处都先从**实际传给 `engine.generate` 的** `SamplingParams.extra_args` 读取草稿。chunk 的 `draft_token_ids` 是第五节选出的候选或回滚草稿；finish 的同义变量是 `finish_draft_token_ids`。分别在对应调用点赋值：

```python
# async_streaming_transcribe() 的正式 chunk 调用点：
boundary = "chunk"
selected_draft_token_ids = draft_token_ids
```

```python
# async_finish_streaming_transcribe() 的正式 finish 调用点：
boundary = "finish"
selected_draft_token_ids = finish_draft_token_ids
```

然后在**各自**的 `engine.generate` 之前执行同一段来源判定。这里的 `sampling_params` 指该次正式请求实际传入的对象（若已打包在 `request_kwargs`，先从中取出）：

```python
submitted_draft_ids = tuple(
    (sampling_params.extra_args or {}).get("ascend_external_draft") or ()
)
draft_stats_valid = submitted_draft_ids == tuple(selected_draft_token_ids)
if not draft_stats_valid:
    logger.warning("ASR_FORMAL_STATS draft mismatch request_id=%s", formal_request_id)
draft_source = (
    "none" if not submitted_draft_ids else
    "prespec" if candidate is not None else
    "rollback"
)
```

`candidate` 必须是本次正式请求**实际领取并选用**的候选，不能从 `PRESPEC_PROBE status=ready_*` 推断。若 `draft_stats_valid=False`，保留正式推理原流程，但不要输出带猜测来源的 `ASR_FORMAL_STATS`；先查打包函数为何改变了草稿。shadow 模式的 `candidate` 返回 `None`，于是正式请求依草稿是否为空标为 `rollback` 或 `none`。服务端提交长度取 `len(submitted_draft_ids)`；引擎运行时进一步裁剪的长度由 `ASR_VERIFY_STEP.effective_draft_tokens` 记录。

该流结束后，仅在正式输出确实完成且来源已核对时记录一条 JSON 日志。下面的 `formal_request_id` 换成原 `stream_...` 或 `finish_...` 的实际局部变量，`seg_id` 取本段编号，不能填 shadow 请求 ID：

```python
import json

if formal_output_finished and draft_stats_valid:
    logger.info("ASR_FORMAL_STATS %s", json.dumps({
        "request_id": formal_request_id,
        "ws_id": connection_id,
        "seg": seg_id,
        "boundary": boundary,  # "chunk" 或 "finish"
        "draft_source": draft_source,
        "submitted_draft_tokens": len(submitted_draft_ids),
        "output_token_count": len(raw_target_tokens),
        "decode_rounds": formal_metrics["decode_rounds"],
        "prefill_ms": formal_metrics.get("prefill_ms"),
        "decode_total_ms": formal_metrics.get("decode_total_ms"),
        "finished": True,
    }, separators=(",", ":")))
```

`PRESPEC_REQUEST event=done tokens=...` 是后台候选的输出长度，不可填到 `output_token_count`。若服务不能从 `_yield_delta_text()` 取得真实轮次，就先修通 `formal_metrics`，不要用输出 token 数或耗时反推。`on_output` 对 DELTA 的 `token_ids` 只累计一次；finish 若不是 DELTA，先改成与 chunk 一致的输出语义或按其实际输出类型去重。

**服务端正式请求的外部 `request_id` 应在整份日志中唯一。**旧的 `stream_{id(state)}_{chunk_id}` / `finish_{id(state)}` 命名会在 Python 释放并复用对象 ID 后撞名。新采样应在 `ASRStreamingState` 中增加一个初始化时生成的 UUID 字段，用它代替 `id(state)`，例如 `stream_{state.request_namespace}_{chunk_id}` / `finish_{state.request_namespace}`；正式请求和服务端 `ASR_FORMAL_STATS` 使用这个外部 ID。vLLM 的 `InputProcessor.assign_request_id()` 会保存外部 ID 并默认生成带 `-8hex` 后缀的内部 ID；调度器中的 `Request` 只保留内部 ID，不能在插桩处直接读取 `request.external_req_id`。因此 `ASR_VERIFY_STEP.request_id` 记录调度器实际 `req_id`，由分析器在确认后缀格式、外部 ID 存在和请求生命周期后关联到服务端记录；新外部 UUID 解决重名，但不会去掉 vLLM 的内部后缀。

旧日志如有外部 ID 重复，不能仅剥离 `-8hex` 后缀后按裸 ID 合并：那会把不同连接的 verifier 错配。无 verifier 的不同 `(ws_id, seg, boundary)` 正式记录可分别计数。存在 verifier 时，分析器先在**同一文件**用 `PRESPEC_TARGET` 提交行与对应 `ASR_FORMAL_STATS` 完成行围出请求生命周期；某条 verifier 恰好只落在一个区间内才自动关联。你贴出的 EngineCore 行只有 `MM-DD HH:MM:SS`，但这种行号关联不依赖其毫秒时间戳。若缺少 `PRESPEC_TARGET`，只有服务与引擎 verifier **都带完整毫秒时间戳且写入同一文件**时，才能显式使用 `--allow-legacy-time-join` 按完成时间推断；重叠、缺边界或跨文件时应报歧义。优先改用唯一外部 ID 重采样。

在 **vLLM-Ascend 完成 rejection sampling、调度器收到本轮输出之后**，对每个正式请求的每个验证轮次记录一行。不要用 `propose()` 的提议长度、服务端 `PRESPEC_MATCH` 或 Ascend sampler 的整批非占位输出数推测接受数。参考本地源码快照，实际判定在 `vllm_ascend/sample/rejection_sampler.py` 的 `rejection_sample(...)`；推荐打点放在 **vLLM 包**的 `vllm/v1/core/sched/scheduler.py::Scheduler.update_from_output()`，现有 `if scheduled_spec_token_ids and generated_token_ids:` 分支内，紧跟 `num_accepted = len(generated_token_ids) - 1`，且在 `_update_request_with_output()` 处理 stop token 之前。若 910B **实际启用** Ascend `RecomputeScheduler`，则改其覆盖的 `vllm_ascend/core/recompute_scheduler.py::RecomputeScheduler.update_from_output()` 同一分支；只修改实际被加载的调度器。源码快照的对应行分别约为 1369 和 923，部署版须按函数名定位。

`recompute_scheduler_enable` 默认是 `False`；本地版本仅在 P/D 分离的 D 节点（`kv_role="kv_consumer"`）启用该类。普通 ASR 部署通常选 vLLM `Scheduler` 或 `AsyncScheduler`。在 910B 上若修改 `recompute_scheduler.py` 后没有日志，先到服务所用 Python 环境的 `vllm/v1/engine/core.py` 中，模块顶部加 `import inspect`，在 `self.scheduler = Scheduler(...)` 后临时加 `logger.warning("ASR_SCHEDULER_ACTIVE cls=%s.%s file=%s", type(self.scheduler).__module__, type(self.scheduler).__qualname__, inspect.getfile(type(self.scheduler)))`，到 **EngineCore 进程日志**查看实际类与源码路径。若类正确但仍无 `ASR_VERIFY_STEP`，在 `update_from_output()` 入口临时记录一次 WARNING，进而检查本轮 `scheduled_spec_decode_tokens.get(req_id)` 与 `generated_token_ids` 是否都非空；该记录不要长期放在热路径。

这里已有的变量可以直接映射，不需要构造原示例中的 `formal_request_id` 等占位变量：

| 日志字段 | 当前代码中的来源 |
|---|---|
| `request_id` | 循环变量 `req_id`，即 vLLM 内部 ID；默认是在服务提交的 `stream_...` / `finish_...` 外部 ID 后追加 `-8hex`，不是服务端原始 ID |
| `effective_draft_tokens` | `len(scheduled_spec_token_ids) - (scheduler_output.num_invalid_spec_tokens or {}).get(req_id, 0)`；调度器可能截短草稿，结构化输出还可能补无效占位 |
| `accepted_draft_tokens` | 已有 `num_accepted = len(generated_token_ids) - 1`；生成列表在这里尚未被 stop 逻辑截断，最后一个是修正或 bonus token |
| `round_index` | 原代码没有；当前一次性 external proposer 的唯一验证轮为 `0` |
| `verifier_complete`、`actual_effective_draft_tokens` | 原代码没有；确认一个请求至多一次 external draft 验证后，本轮可分别设 `True`、本轮 `effective_draft_tokens` |

最小插桩如下。先在调度器模块顶部 `import json`；以下代码**直接插在现有 `num_accepted = len(generated_token_ids) - 1` 之后**，其后的原有 `num_rejected`、KV 回退与统计代码照常执行。`logger` 是调度器已有的模块 logger。以 `stream_` / `finish_` 过滤仅适用于当前 ASR 请求命名及“只使用一次性 external proposer”的配置；如果还有其他 proposer，会需要额外的请求来源标记，不能把它们的验证算作 external draft。

```python
# 此代码与上一行现有的 num_accepted 处于同一缩进层级。
if req_id.startswith(("stream_", "finish_")):
    invalid = (scheduler_output.num_invalid_spec_tokens or {}).get(req_id, 0)
    effective = num_draft_tokens - invalid
    if 0 <= num_accepted <= effective:
        event = {
            "request_id": req_id,
            "round_index": 0,  # 一次性 external proposer：该请求唯一验证轮
            "effective_draft_tokens": effective,
            "accepted_draft_tokens": num_accepted,
            "verifier_complete": True,
            "actual_effective_draft_tokens": effective,
        }
        if effective == 0:
            event["reason"] = "grammar_rejected_all"
        logger.info("ASR_VERIFY_STEP %s", json.dumps(event, separators=(",", ":")))
    else:
        # 保留原推理流程；不要把异常数据截断成貌似有效的接受数。
        logger.error(
            "ASR_VERIFY_INVARIANT request_id=%s effective=%s accepted=%s",
            req_id, effective, num_accepted,
        )
```

`scheduled_spec_token_ids` 来自当轮 `scheduler_output.scheduled_spec_decode_tokens.get(req_id)`，比服务提交的 `submitted_draft_tokens` 更接近实际 verifier 输入；`num_invalid_spec_tokens` 只在结构化输出等路径生成无效占位时非零。这里的 `num_accepted` 是 vLLM 调度器已有的**连续草稿接受数**，不包含拒绝后的修正或全接受后的 bonus token。调度器拿到的 `generated_token_ids` 已是 CPU 列表，所以此处日志不会为了测量而额外同步 NPU。`draft_source` 和 `boundary` 不在引擎里猜：分析脚本先将内部 `req_id` 有条件关联到服务端外部 `request_id`，再读取对应的 `ASR_FORMAL_STATS`。

上述 `round_index=0`、`verifier_complete=True` 依据目前 external proposer 的 `external_drafts.pop(req_id, [])` 一次性消费语义。若 910B 的 fork 能对同一正式请求再次提供 external draft，需要按 `req_id` 维护轮次和有效草稿累计数，只在最后一轮写 `verifier_complete=True` 及累计的 `actual_effective_draft_tokens`；不能把每轮都标为完成。分析脚本要求各轮有效数之和等于末轮给出的总数，否则判为不完整。

**未进入 verifier 的草稿要在阻断处另记 0/0。**在 external proposer 中，只有确认该正式 `req_id` 存在非空草稿且此轮确定不会验证时才记录 `effective_draft_tokens=0`、`accepted_draft_tokens=0`、`round_index=0`、`verifier_complete=True`、`actual_effective_draft_tokens=0` 和具体 `reason`。例如 `pop(req_id, [])` 后 `max_draft_len <= 0` 是明确阻断；`sampled_ids` 暂时为空并不代表阻断，而 `spec_decode_unsupported_reqs` 或上下文不足只有在确定请求终止或草稿被丢弃时才可记完成。服务提交了草稿却既无验证事件、也无明确阻断事件时，分析脚本标为 `missing_verifier`，不能据此推断接受数为 0。

部署两处打点后，将本仓库分析脚本复制到 910B 服务项目根目录，再用每轮完整日志（含服务和引擎日志）运行；若两者本来写在同一个文件，每组只传该文件一次：

```bash
cp /path/to/vllm-ascend/research/streaming_prespec_gain.py ./
python streaming_prespec_gain.py \
  --rollback-log rollback-server.log rollback-engine.log \
  --consume-log consume-server.log consume-engine.log \
  --out gain-c20.json
```

`gain-c20.json` 的 `runs.rollback`、`runs.consume` 分别有每请求 `records` 与 `all`、`by_boundary`、`by_draft_source` 汇总；每条 record 的 `request_id` 是服务端外部 ID，`verifier_request_ids` 保留关联到的引擎内部 ID，供核查后缀映射。`comparison.rollback_to_consume` 给出 consume 减 rollback 的均值/P95 差值。先看 `missing_verifier_count`、`incomplete_verifier_count` 和 `explicit_zero_effective_count`，再按 `boundary=chunk/finish` 与 `draft_source` 看每请求输出 token、轮次、`tokens_per_round`、实际接受数/有效草稿数及 `observed_acceptance_rate`。无草稿请求的接受数记 0，但不进入接受率分母；有草稿却缺验证/阻断事件记缺失。可加 `--shadow-log shadow-server.log shadow-engine.log` 输出 B 组及 A→B、B→C 差值。脚本只报告 `case_id_count`（带此字段的请求数），**不自动配对**；相同音频回放能提供稳定 `case_id` 时可用 `records` 另做逐条配对或按输出长度分组。不同音频或不同 finish 跳过比例下，均值只能作描述。最终仍以同负载的**全部非空 commit** P50/P95/P99 为收益判据，并同时报告段级音频队列等待与 vLLM waiting/running。

判定表：

| 顺序 | 必查指标 | 通过条件 |
|---|---|---|
| 1 正确性 | 原始 token、delta/done、最终文本、commit 边界 | B 与 A 等价；C 经 verifier 后与 A 等价，无候选直通 |
| 2 可用性 | launch/拒绝原因、`ready_*` 比例、`ready_lead_ms`、实际 verifier 接受数 | 完整候选或已输出部分快照先于正式请求可用，并经 verifier 验证；若 gated 几乎零 launch，只能报告策略下覆盖不足 |
| 3 正式时延 | 全部非空 commit 及正式 chunk 的 P50/P95/P99、首 delta、后续请求时延 | B 相比 A 的额外开销低于**测试前约定**的退化界限；C 相比 A 有稳定净收益 |
| 4 排队与成本 | `service_queue_wait_ms`、`audio_blocks_pending_at_commit`、vLLM waiting/running、NPU 利用率、shadow 请求数/计算量 | 无关键路径排队恶化；不能只用 NPU 利用率上升宣称收益 |

`queue_size` 是段级音频队列，含 commit 的 `None` 哨兵；`queue_waiting` 才是 vLLM 等待请求数。至少按轮次报告 A/B/C 的差值和波动，P99 样本太少时标注不确定，不能用候选命中子集的改善代替全部请求的结果。历史 `baseline-c20` 使用旧回滚首 token 合同，仅作参考；最终比较需要在同一代码和负载条件下重跑 A 组 baseline。

### 4.2 首次失配后的候选重接：离线初筛

`raw_matched_prefix_tokens=0` 但 `draft_matched_prefix_tokens>0` 表示首 token 不同、后缀相同；新方案在首次验证失配后一次性用按位置续接验证 `C[1:]`。这次验证若仍失配，预投机候选按 `--prespec-rejoin-policy` 的 ngram 策略寻找锚点；没有锚点时自回归，后续确认 token 可继续触发搜索，找到锚点就验证该后缀。`--no-prespec-first-mismatch-ngram` 可单独关闭这一阶段以做 A/B；回滚草稿仍用原有失配路径。以下旧日志回放仍以原来的 `candidate[1:]` 实现为基准，不能当作新方案的线上收益：

```bash
cp /path/to/vllm-ascend/research/streaming_prespec_rejoin.py ./
python streaming_prespec_rejoin.py --log consume-server.log \
  --kmax 15 --eos-token-id 151645 \
  --out rejoin-summary.json --cases-out rejoin-cases.jsonl
```

脚本先按正式请求已采样首 token、验证 `candidate[1:]` 的现有顺序定位首次失配；它只用截至当前已经确认的正式 token 选择下一份草稿。`aligned` 按原 token 位置尝试候选余段；`ngram_1/2/3` 用正式输出的末尾 1/2/3 个 token 在候选中找锚点，提出锚点后的最多 `kmax` 个 token。两者都逐轮与日志中的正式 token 比较，输出 `attempts`、`zero_accept_attempts`、`accepted_tokens`、`accepted_nonterminal_tokens`、`estimated_saved_steps`。提案选择不读取未来正式 token；未来 token 只用于离线评分。这里每次验证理想化为一个模型步，**不是实测 verifier 接受数或时延**。

旧版协调器的 `PRESPEC_MATCH_TOKENS.target_token_ids` 只打印到候选长度。若 `target_len` 更长，脚本标为 `censored_target`。只要所有候选位置都已打印且首次失配可见，`aligned` 仍可计算：候选耗尽后，剩余正式 token 只能逐步生成；`ngram_1/2/3` 保持 `null`，因为后续正式 token 可能与候选更早的位置形成新锚点。更新后的协调器打印完整正式 token，回放脚本允许 `target_token_ids` 比候选更长；新日志应有 `target_len == len(target_token_ids)`。`replayed_count` 只统计四种策略都能回放的样例，各策略的 `evaluated_request_count` 是各自的分母；`excluded_replay_reasons` 表示未进入完整四策略回放的原因，因此旧日志的 `censored_target` 中仍可能有 `aligned` 结果。报告还列出 `no_post_first_draft_mismatch`、`candidate_draft_exceeds_kmax` 等原因，并按 shadow/consume 及 `candidate_partial=true/false/unknown` 分组；旧日志没有此字段时归入 `unknown`。它的总体分母仅为出现这条匹配日志的候选，不是全部正式请求或 commit。你给出的 `ws-191306f2/seg-3/window-2` 例子中，首次失配后按位置续接的理想节省为 0 步；旧截断日志无法判断后续正式 token 是否会形成 ngram 锚点。

在已接入预投机的 910B 环境补采时，服务端协调器只需把 `tuple(target_continuation_tokens[:len(candidate.token_ids)])` 改成 `tuple(target_continuation_tokens)`，并使用更新后的回放脚本。下面的绝对路径换成 910B 上实际检出路径；服务启动命令和 `test_v2.py` 的参数以现有环境为准，本仓库没有这两个入口：

```bash
# 在 910B 上更新包含回放脚本的仓库
cd /path/to/vllm-ascend
git switch codex/streaming-prespec-monitor
git pull --ff-only origin codex/streaming-prespec-monitor

# 在 910B 服务仓库根目录修改已适配的协调器，并检查语法
cd /path/to/Qwen3-ASR-910B
python - <<'PY'
from pathlib import Path
p = Path('qwen_asr/inference/streaming_prespec.py')
s = p.read_text(encoding='utf-8')
old = 'tuple(target_continuation_tokens[:len(candidate.token_ids)])'
new = 'tuple(target_continuation_tokens)'
if old in s:
    assert s.count(old) == 1
    p.write_text(s.replace(old, new), encoding='utf-8')
else:
    assert new in s, '未找到预期日志表达式，请检查服务端协调器版本'
PY
python -m py_compile qwen_asr/inference/streaming_prespec.py qwen_asr/inference/qwen3_asr_async.py qwen_asr/cli/realtime_connection.py

# 按原有方式重启服务，并传入已有的 parser 参数
<现有服务启动命令> --prespec-mode consume --prespec-admission force

# 使用原有固定并发 c20 的 sustained_test 命令重跑；保留这一轮独立的服务日志
python /path/to/test_v2.py --help
<现有 c20 测试命令>
rg -m 1 'PRESPEC_MATCH_TOKENS' /path/to/consume-c20-new.log

# 在包含新回放脚本的 vllm-ascend 检出目录执行
cd /path/to/vllm-ascend
python research/streaming_prespec_rejoin.py --log /path/to/consume-c20-new.log \
  --kmax 15 --eos-token-id 151645 \
  --out rejoin-summary-full.json --cases-out rejoin-cases-full.jsonl
python -c 'import json; d=json.load(open("rejoin-summary-full.json")); print("match_events", d["match_events"], "censored_target_count", d["censored_target_count"], "partial_unknown", d["by_candidate_partial"]["unknown"]["match_events"])'
```

只把**这一轮**日志传给回放脚本；新日志的 `censored_target_count` 应为 0。若仍大于 0，先核对服务实际加载的 `streaming_prespec.py` 是否已更新。`partial_unknown` 只有在服务端还输出 `candidate_partial=` 时才会归零；它不影响完整正式 token 的 ngram 回放，只影响取消候选的分组分析。旧日志里被截掉的 token 无法从 `ASR_FORMAL_STATS` 的数量或 `Segment done` 文本恢复。在线实现还需让 proposer 保留候选池并在每次正式采样后重新选草稿；当前 `external_drafts.pop(req_id, [])` 只消费一次。随后用相同音频和并发负载比较实际 verifier 轮次、全部非空 commit 的 P95/P99、队列与 NPU 成本。

## 5. 门控空档验证：在 force 对比后接入实时负载

当前 [模型包装函数](streaming_prespec.md) 示例末尾写死 `load=None, predicted_draft_ms=0.0`。要运行 `gated`，先在模型上接入一份**后台更新、不在音频处理协程中同步查询**的负载快照；这份快照应覆盖同一 engine/目标 NPU。构造 `LoadSnapshot` 时至少填入 vLLM waiting、formal pending、NPU 利用率、采样的 `time.monotonic_ns()`、最近正式请求的队列等待 P95。在第 4.5 节 `try_start_prespec_for_state()` 的已有调用中，只替换最后两行：

```diff
-        load=None, predicted_draft_ms=0.0,  # 初次 force 实验
+        load=self.prespec_load_source.latest(),  # 非阻塞；LoadSnapshot 或 None
+        predicted_draft_ms=self.prespec_duration_estimate_ms,
```

`latest()` 和 `prespec_duration_estimate_ms` 由服务现有指标采集链路提供；本仓库没有该服务/设备接口，不能把旁路监测 JSONL 直接当成实时源。启动前核对 `LoadSnapshot.observed_ns` 与协调器同为单调时钟、快照年龄不超过 `AdmissionPolicy.max_load_age_ms`、`formal_pending` 覆盖尚未进入 vLLM 的正式工作，并用真机 shadow 耗时设置 `predicted_draft_ms`。`force` 测得的 P95/P99 只用于估计成本，不能作为零成本预算。

接线后重启 B/C 并执行第 4 节**相同命令和负载**，唯一变化是：

```bash
<现有服务启动命令> --prespec-mode shadow --prespec-admission gated
# C 组重启时将 shadow 换成 consume，其余服务参数相同。
```

从 DEBUG `PRESPEC_LAUNCH reason` 汇总 `missing_load/stale_load/formal_queue_nonempty/utilization_high/insufficient_slack/launched` 的次数，并在正式日志中检查每次 launch 前后及下一个请求的队列等待。若几乎不 launch，结论是“当前门控策略覆盖不足”，不能据此称候选无效；若 launch 后正式 P95/P99 或后续请求排队恶化，收紧门控或停用，不以 NPU 利用率上升替代时延证据。
