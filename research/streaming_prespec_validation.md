# 910B 流式音频预投机验证手册

本手册验证当前 **每个 `(ws_id, seg_id)` 最多一次中点候选** 的实现。baseline 是“现有回滚投机开启、预投机关闭”。服务端三个源文件不在此仓库；下面标明应插入的函数与代码，需在 910B 服务项目完成接线。本仓库的 Python mock 测试不能代替真机验证。

## 0. 三组配置与共同前提

| 组 | `--prespec-mode` | 回滚投机 | 含义 |
|---|---|---|---|
| A | `off` | 开 | rollback baseline |
| B | `shadow` | 开 | 后台生成候选，正式请求仍用原回滚草稿 |
| C | `consume` | 开 | 后台生成候选，正式请求在完整音频上经现有 verifier 验证候选 |

A/B/C 保持同一服务代码、模型、图配置、回滚参数、chunk 长度、音频和连接负载。配置变更后重启服务，不能在活跃连接期间替换协调器。**先用 `force` 完成链路和 A/B/C 初次对比**；此时仍受中点、正式请求在途和容量限制，但不按设备利用率准入。完成实时负载接线后，再用 `gated` 重跑 B/C，回答“低利用率空档是否存在且可安全利用”。若未接入新鲜 `LoadSnapshot`，`gated` 会返回 `missing_load`；旁路 `npu-smi` 的 1 秒采样不能冒充默认要求 200 ms 内的新鲜准入读数。

## 1. 服务端加入最小日志：逐处修改

以下代码使用 `logger = logging.getLogger(__name__)`；若文件已有 logger，沿用原对象。服务日志格式需带毫秒时间和时区；三个日志来源的主机时钟应同步。日志不记录音频、文本和完整 token 序列，只记录关联键、长度、状态和耗时。

### 1.1 启动 parser → `AsyncLLM()` → `enable_prespec()`

先在服务项目根目录定位已有的 argparse 入口和模型工厂调用，沿用原 parser 与原启动命令：

```bash
rg -n 'ArgumentParser|parse_args|AsyncLLM\(' qwen_asr
```

在现有 `parser.parse_args()` **之前**增加三个参数；保留其余已有参数：

```python
parser.add_argument(
    "--prespec-mode", choices=("off", "shadow", "consume"), default="off",
    help="off=rollback baseline; shadow=observe; consume=verify candidate",
)
parser.add_argument(
    "--prespec-admission", choices=("force", "gated"), default="force",
    help="force for chain checks; gated requires fresh LoadSnapshot",
)
parser.add_argument(
    "--prespec-candidate-tokens", type=int, default=None,
    help="shadow output max_tokens; omitted means auto (engine kmax + 1)",
)
```

这里 `default=None` 是“未显式指定”的标记：parser 阶段尚未取得引擎 `kmax`，所以不能直接写 `default=kmax + 1`。启动时将 `None` 依次传给 `AsyncLLM()`、`enable_prespec()`；仅在启用预投机后，由 `enable_prespec()` 将它解析为 `kmax + 1`。显式传入整数时原样使用，并受下文预算检查约束。

你现在的调用是 `Qwen3ASRModelAsync.AsyncLLM(**base_config)`。保留 `base_config` 原内容，在同一调用上显式追加 parser 参数：

```python
model = Qwen3ASRModelAsync.AsyncLLM(
    **base_config,
    prespec_mode=args.prespec_mode,
    prespec_admission=args.prespec_admission,
    prespec_candidate_tokens=args.prespec_candidate_tokens,
)
```

`base_config` 不能同时含有这三个 `prespec_...` 名字，否则 Python 会因重复关键字报错。若原调用的结果变量不是 `model`，保留原变量名。

在 `qwen3_asr_async.py` 的 `AsyncLLM()` 现有类方法签名增加三个显式形参；若签名有 `**kwargs`，把它们放在 `**kwargs` **之前**。不能只在调用处传参，否则可能得到 `unexpected keyword argument`，或落入 `**kwargs` 后被错误转发给 vLLM engine：

```python
prespec_mode: str = "off",
prespec_admission: str = "force",
prespec_candidate_tokens: int | None = None,
```

这里假定现有入口用 `argparse` 且解析结果名为 `args`；若名字不同，只替换该局部变量名。服务启动脚本不在本仓库，无法填写其余既有 CLI 选项。

先在 910B 服务项目根目录核对当前实际签名：

```bash
rg -n -A 35 'def enable_prespec' qwen_asr/inference/qwen3_asr_async.py
```

这里的 `enable_prespec()` **不是上游已有 API**，而是 [主接入文档步骤 2](streaming_prespec.md) 要在 `Qwen3ASRModelAsync` 中新增的方法。该文档现在已把 `admission_mode` 和 `shadow_only` 加入签名并传给协调器；若你之前照旧版示例复制了函数，必须先同步更新签名与协调器构造，才可传下面两个实参。为记录 shadow 请求生命周期，再把其中的 `producer=self._run_prespec` 替换为 `producer=self._logged_prespec_producer`。完整的关键行是：

```python
def enable_prespec(self, *, enabled, kmax, engine=None, chunk_size_sec=2.0,
                   candidate_tokens=None, on_probe=None,
                   admission_mode="force", shadow_only=True):
    # 保留主接入文档中已有的 enabled/kmax/engine 校验。
    if candidate_tokens is None:
        candidate_tokens = kmax + 1
    if candidate_tokens < 3:
        raise ValueError("candidate budget needs first token, draft and continuation")
    # midpoint_samples 沿用主接入文档中的计算。
    self.prespec = PreSpecCoordinator(
        producer=self._logged_prespec_producer,
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
model.enable_prespec(
    enabled=(prespec_mode != "off"), kmax=kmax, engine=engine,
    chunk_size_sec=chunk_size_sec,
    candidate_tokens=prespec_candidate_tokens,
    admission_mode=prespec_admission,
    shadow_only=(prespec_mode != "consume"),
    on_probe=log_prespec_probe if prespec_mode != "off" else None,
)
logger.info(
    "PRESPEC_CONFIG mode=%s admission=%s kmax=%s candidate_budget=%s",
    prespec_mode, prespec_admission, kmax,
    model.prespec.policy.max_candidate_tokens if model.prespec is not None else None,
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

这里的 `done` 只表示 shadow 引擎生成结束，尚不表示正式请求采纳。若 `cancel` 后底层 NPU 工作仍持续，另看 vLLM 请求结束/abort 日志和资源占用。

### 1.3 `realtime_connection.py`：记录中点准入决定

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

`try_start_prespec_for_state()` 在未达中点、已 commit 或音频队列非空时返回 `None`，因此这些路径**不会**出现 `PRESPEC_LAUNCH`。`chunk-0` 没有旧回滚草稿，空草稿有效；在 consume 模式下，正式请求可验证它在 1 s 音频下生成的候选。真正进入 `try_start_prespec()` 后，旧草稿超出引擎 `kmax` 或输出预算放不下 `1 + 旧草稿长度 + 1` 时，分别返回 `rollback_draft_exceeds_kmax` / `candidate_budget_insufficient`，不会给引擎发缩短后的草稿。这两项准入检查在 shadow 和 consume 模式都执行；两者只在正式请求是否使用已完成候选上不同。helper 对每个 `(ws_id, seg_id, window_id, reason)` 只输出一次 `PRESPEC_SKIP` INFO，避免同一窗口的后续 80 ms chunk 重复刷屏。若预期到达中点却没有 launch，请在包装函数这些 `return None` 前临时加一条 `DEBUG` 日志，包含 `ws_id/seg/window_id/len(state.buffer)/reason`；不要据“无 launch 日志”直接判断负载门控拒绝。

### 1.4 `qwen3_asr_async.py`：关联正式请求与候选

在 [第五节](streaming_prespec.md) 的 `begin_authoritative_chunk()` 前保存 `this_chunk_id = state.chunk_id`；在正式请求已构造好 `request_id`、紧邻现有 `engine.generate(...)` 前记录候选来源。下面的 `formal_request_id` 要换成正式请求当前使用的实际局部变量（如现有 `stream_...` ID），不能填 shadow 的 `prespec-...` ID：

```python
logger.info(
    "PRESPEC_TARGET ws_id=%s seg=%s chunk=%s event=submit "
    "formal_request_id=%s candidate_request_id=%s "
    "candidate_raw_tokens=%s submitted_draft_tokens=%s",
    ws_id, seg_id, this_chunk_id, formal_request_id,
    candidate.request_id if candidate is not None else None,
    len(candidate.token_ids) if candidate is not None else None,
    len(draft_token_ids),
)
```

正式 chunk 的原 `finally` 只需与 `begin_authoritative_chunk()` 配对；不新增 `raw_target_tokens`，也不修改 `_yield_delta_text()`：

```python
if prespec is not None:
    prespec.end_authoritative_chunk(
        ws_id, audio_epoch=seg_id, window_id=this_chunk_id)
```

不传 `target_continuation_tokens` 表示**只清理候选，不计算匹配长度**，协调器返回 `None`；不能把此结果记为“接受 0 个 token”。C 组仍须在引擎实际处理 `ascend_external_draft` 的验证结果处记录 `request_id`、提交的草稿长度、实际连续接受长度、**首个草稿 token** 是否拒绝、修正 token 数、目标模型前向步数。首个 target token 已在 proposer 之前采样，不属于 external draft。只在 proposer 的 `propose()` 打日志不足以证明草稿被验证；该函数只负责提出 token。先在服务/引擎仓库定位挂点：

```bash
rg -n 'ascend_external_draft|def propose\(|accepted.*token|rejected.*token' .
```

还要在 910B 的 proposer 处核对 `sampled_ids` 后接入草稿的真实顺序，确认 `requested_draft_len`、`effective_draft_len`、`num_tokens_no_spec`、`max_model_len`；静态 `kmax` 检查不能发现上下文空间不足造成的运行时缩短。正式日志应满足 `candidate_raw_tokens = submitted_draft_tokens + 1`，且 `effective_draft_len = submitted_draft_tokens` 才能把该轮记为“完整提交候选草稿”。

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

这段仅展示采集与统计挂点，不能替换原函数的其他实参或原有状态写回。只有观察到 `output.finished` 才传入 token 列表；中断/取消时传 `None`，避免把部分输出或尚未采集的 `()` 算作 0 匹配。完成且确实生成 0 个 token 时，空元组才是有效观测；此时 `target_len=0`，不能解释为候选首 token 不匹配。`raw_matched_prefix_tokens` 比较候选完整原始输出与正式完整原始输出。`draft_matched_prefix_tokens` 比较实际拟提交的候选 `token_ids[1:]` 与正式输出的 `token_ids[1:]`，跳过两个请求各自采样的首 token；例如候选 `(19, 20, 21)`、正式 `(99, 20, 22)` 时，两个值分别是 0 和 1。两者都是 shadow 离线匹配指标，**都不是引擎 verifier 的实际草稿接受数**；后者仍从 verifier 打点取得。候选和正式请求必须使用相同的输出 token 对齐起点（prompt 之后），且这里假设正式请求只有一个 completion。

finish 复用 `_split_stream_rollback()` 处理 `\ufffd` 时，`on_output` 本身不用改：它只采集 prompt 之后的原始生成 token，不读取或重算稳定前缀。要看 **提前 commit 的 finish 候选匹配统计**，优先按[主接入文档第六节](streaming_prespec.md)让 DELTA 输出的 finish 复用 `_yield_delta_text()` 消费同一条 `finish_...` 流，并通过适配回调收集原始 token IDs。`on_output` 接收单次 `RequestOutput`，不能直接传接收最终 token 元组的 `on_finish_tokens`。完成后的元组通过 `_finish_segment()` 存到本段，再由 `_process_segment()` 的 finally 传给 `finish_formal(..., target_continuation_tokens=...)`。仅在观察到 `output.finished` 后传 token 列表，中断时传 `None`；原 finish 若使用 CUMULATIVE/FINAL_ONLY，先核对 `_yield_delta_text()` 的输出语义，不能直接按 DELTA 累积。这个回调不会改变客户端转写文本或正式请求的输入。

910B 服务接入并重放一个在完整 chunk 到来前提前 commit、且剩余音频超过 0.4 秒的段后，查同一 `ws_id/seg/window` 的日志：

```bash
rg 'PRESPEC_(PROBE|TARGET|MATCH)' /path/to/server.log | rg 'ws_id=ws-实际ID seg=实际段号'
```

shadow 模式应看到 `PRESPEC_PROBE ... boundary=formal status=ready_shadow_only`、`PRESPEC_MATCH_TOKENS` 和 `PRESPEC_MATCH ... boundary=formal`；consume 模式把状态改为 `ready_consumed`，并在 `PRESPEC_TARGET ... boundary=formal` 看到候选 request ID。`PRESPEC_MATCH` 同时给出 `raw_matched_prefix_tokens` 和 `draft_matched_prefix_tokens`。若 finish 被短尾规则跳过、候选未完成或窗口变更，不会有该轮匹配统计；若正式输出未完成，也不能把部分输出作为 0 匹配。

协调器收到完整的 `target_continuation_tokens` 且已选中候选时，会另外打印 `PRESPEC_MATCH_TOKENS`。`candidate_token_ids` 是候选完整原始输出；`target_token_ids` 是从正式原始输出开头截取的**相同长度**序列，`target_len` 是正式输出的实际总长度。按位置对照两列即可看到 `raw_matched_prefix_tokens` 个相同 token 后的首个差异。例如 `candidate_token_ids=(19, 20, 21, 22)`、`target_token_ids=(19, 20, 99, 77)`、`target_len=5`、`raw_matched_prefix_tokens=2`，表示前两项相同，第三项 `21` 与 `99` 不同，正式输出还有一个未显示的 token。日志的 `mode=shadow/consume` 标明正式请求是否使用候选；consume 模式下两组序列的匹配不代表无候选的 baseline 结果。若未采集正式 token、正式生成中断，或没有就绪候选，就不会出现该日志。它同样不是 verifier 接受数。

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
<现有服务启动命令> --prespec-mode shadow --prespec-admission force
```

使用现有 WebSocket 测试脚本发送**同一段超过 6 秒、6 秒前不 commit**的音频；这一步并发为 1，不开启自动补位。服务若写日志文件：

```bash
tail -F /path/to/server.log | rg --line-buffered 'PRESPEC_|inference submit|Commit received|Segment done'
```

若服务由 systemd 管理，改用 `journalctl -u <实际服务单元名> -f -o cat` 作为管道左侧。先用**连续超过 6 秒且 6 秒前不 commit 的单段音频**核对三个窗口：约 1、3、5 秒分别看到 `window=0/1/2` 的 `PRESPEC_LAUNCH/REQUEST`，约 2、4、6 秒分别看到对应 `window=0/1/2 boundary=chunk status=ready_shadow_only`。三个窗口的 `request_id` 应不同，正式 `chunk=0/1/2` 分别只关联同号窗口；任一窗口跳过时不应误领前一窗口候选。`ready_lead_ms > 0` 表示候选早于对应正式请求提交完成；若为 `pending`，候选被取消，不应等待它。若是 `candidate_exceeds_kmax`，shadow 的输出超过正式 proposer 可接收长度，协调器不会把它交给正式请求。随后分别重放提前 commit、≤0.4 秒短尾和断连，确认无遗留 `prespec-...` 请求、重复 begin 或未结束的 session。约 5.95 秒就 commit 的段在 5 秒启动 `window=2`，若未凑足 6 秒正式 chunk，应在实际 `finish_...` 请求提交时看到 `PRESPEC_PROBE window=2 boundary=formal status=ready_shadow_only`；consume 模式则应为 `ready_consumed`，且 `PRESPEC_TARGET boundary=formal` 的 `candidate_request_id` 对应先前的 `prespec-...`。commit 到达时只保留已完成候选，不立即打印 formal probe；若后续完整 chunk 先领取了它，finish 不得重复领取。短尾跳过 finish 时也不会出现 formal probe，外层清理会丢弃候选。

如果链路没有 launch，打开 DEBUG 日志后先看 `return None` 的跳过原因和 `PRESPEC_LAUNCH reason`；`force` 仅绕过实时负载门控，不绕过中点、草稿、在途正式请求和容量检查。

单轮结束后可快速检查事件是否齐全、拒绝原因分布；路径替换为该轮切出的服务日志：

```bash
rg -c 'PRESPEC_LAUNCH.*launched=True|PRESPEC_REQUEST event=done|PRESPEC_PROBE.*ready_' /path/to/run.log
rg 'PRESPEC_SKIP|PRESPEC_LAUNCH' /path/to/run.log | rg -o 'reason=[^ ]+' | sort | uniq -c
```

## 3. 固定输入正确性：A、B、C 逐级比较

对同一批音频保存**每条连接的音频文件、切片顺序、发送时刻和 commit 时刻**。三组分别重启服务并使用同一回放清单：

```bash
# A：保持现有回滚投机配置不变
<现有服务启动命令> --prespec-mode off
# 使用现有 WebSocket 回放命令，保存客户端事件和服务日志

# B：正式请求仍用原回滚草稿
<现有服务启动命令> --prespec-mode shadow --prespec-admission force
# 同一回放清单

# C：先仅在固定输入回放中启用消费
<现有服务启动命令> --prespec-mode consume --prespec-admission force
# 同一回放清单
```

这里的 `<现有服务启动命令>` 和 WebSocket 回放命令是环境专属的唯一占位符：本仓库没有 910B 服务启动脚本或你的 `test_v2.py` 参数表，不能编造 CLI。A/B 应逐 chunk 对比已有的客户端 delta/done、最终文本与 commit 边界；若引擎本来就记录正式原始 token IDs，可额外对比，但不为此改造流式输出函数。C 还要确认候选的后续 token 实际写进**正式 chunk 与提前 commit 的 `finish_...` 请求**的 `ascend_external_draft` 并由引擎 verifier 验证；分别覆盖首个草稿 token 拒绝、部分接受、全拒绝、版本不匹配和候选超时。提前 commit 的 `PRESPEC_TARGET boundary=formal` 必须显示真实 `finish_...` 请求 ID、候选 ID 和提交草稿长度；短尾跳过或窗口号已改变时不得提交候选。此步不通过就不要开始性能对比。

## 4. 同负载性能对比：A→B、B→C、A→C

先用 B/C 的 `--prespec-admission force` 完成初次同负载对比；这一步只能衡量后台请求的收益与干扰，**不能证明低利用率准入策略有效**。分别跑现有 `sustained_test` 的并发档位 `1、5、20`，再扩到拐点附近；每组保持相同音频清单、到达时序、自动补位规则、持续时间和预热。A/B/C 交错排序，至少重复 3 轮，记录实际测试起止 UTC。每轮单独设置 `run_id`、切片服务日志、导出该轮 `commits.jsonl`。导出行须含 `ws_id, seg, received_at, done_at, commit_latency_ms, queue_wait_ms, queue_size, is_empty_commit`；空 commit 不算正式时延。客户端执行命令仍用你现有 `test_v2.py` 的真实 CLI。

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

判定表：

| 顺序 | 必查指标 | 通过条件 |
|---|---|---|
| 1 正确性 | 原始 token、delta/done、最终文本、commit 边界 | B 与 A 等价；C 经 verifier 后与 A 等价，无候选直通 |
| 2 可用性 | launch/拒绝原因、`ready_*` 比例、`ready_lead_ms`、实际 verifier 接受数 | 候选确实提前完成且被验证；若 gated 几乎零 launch，只能报告策略下覆盖不足 |
| 3 正式时延 | 全部非空 commit 及正式 chunk 的 P50/P95/P99、首 delta、后续请求时延 | B 相比 A 的额外开销低于**测试前约定**的退化界限；C 相比 A 有稳定净收益 |
| 4 排队与成本 | `service_queue_wait_ms`、`audio_blocks_pending_at_commit`、vLLM waiting/running、NPU 利用率、shadow 请求数/计算量 | 无关键路径排队恶化；不能只用 NPU 利用率上升宣称收益 |

`queue_size` 是段级音频队列，含 commit 的 `None` 哨兵；`queue_waiting` 才是 vLLM 等待请求数。至少按轮次报告 A/B/C 的差值和波动，P99 样本太少时标注不确定，不能用候选命中子集的改善代替全部请求的结果。历史 `baseline-c20` 仅作参考；最终比较需要在同一代码和负载条件下重跑 rollback baseline。

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
