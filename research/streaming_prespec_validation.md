# 910B 流式音频预投机验证手册

本手册验证当前 **每个 `(ws_id, seg_id)` 最多一次中点候选** 的实现。baseline 是“现有回滚投机开启、预投机关闭”。服务端三个源文件不在此仓库；下面标明应插入的函数与代码，需在 910B 服务项目完成接线。本仓库的 Python mock 测试不能代替真机验证。

## 0. 三组配置与共同前提

| 组 | `ASR_PRESPEC_MODE` | 回滚投机 | 含义 |
|---|---|---|---|
| A | `off` | 开 | rollback baseline |
| B | `shadow` | 开 | 后台生成候选，正式请求仍用原回滚草稿 |
| C | `consume` | 开 | 后台生成候选，正式请求在完整音频上经现有 verifier 验证候选 |

A/B/C 保持同一服务代码、模型、图配置、回滚参数、chunk 长度、音频和连接负载。配置变更后重启服务，不能在活跃连接期间替换协调器。**先用 `force` 完成链路和 A/B/C 初次对比**；此时仍受中点、正式请求在途和容量限制，但不按设备利用率准入。完成实时负载接线后，再用 `gated` 重跑 B/C，回答“低利用率空档是否存在且可安全利用”。若未接入新鲜 `LoadSnapshot`，`gated` 会返回 `missing_load`；旁路 `npu-smi` 的 1 秒采样不能冒充默认要求 200 ms 内的新鲜准入读数。

## 1. 服务端加入最小日志：逐处修改

以下代码使用 `logger = logging.getLogger(__name__)`；若文件已有 logger，沿用原对象。服务日志格式需带毫秒时间和时区；三个日志来源的主机时钟应同步。日志不记录音频、文本和完整 token 序列，只记录关联键、长度、状态和耗时。

### 1.1 `qwen3_asr_async.py`：模型初始化时接入 probe 和三组开关

先在 910B 服务项目根目录核对当前实际签名：

```bash
rg -n -A 35 'def enable_prespec' qwen_asr/inference/qwen3_asr_async.py
```

这里的 `enable_prespec()` **不是上游已有 API**，而是 [主接入文档步骤 2](streaming_prespec.md) 要在 `Qwen3ASRModelAsync` 中新增的方法。该文档现在已把 `admission_mode` 和 `shadow_only` 加入签名并传给协调器；若你之前照旧版示例复制了函数，必须先同步更新签名与协调器构造，才可传下面两个实参。为记录 shadow 请求生命周期，再把其中的 `producer=self._run_prespec` 替换为 `producer=self._logged_prespec_producer`。完整的关键行是：

```python
def enable_prespec(self, *, enabled, kmax, engine=None, chunk_size_sec=2.0,
                   candidate_tokens=8, on_probe=None,
                   admission_mode="force", shadow_only=True):
    # 保留主接入文档中已有的 enabled/kmax/engine 校验和 midpoint_samples 计算。
    self.prespec = PreSpecCoordinator(
        producer=self._logged_prespec_producer,
        policy=AdmissionPolicy(
            admission_mode=admission_mode,
            midpoint_samples=midpoint_samples,
            max_candidate_tokens=min(candidate_tokens, kmax),
            max_active_shadows=1,
        ),
        shadow_only=shadow_only,
        on_probe=on_probe,
    )
```

先在 `qwen3_asr_async.py` 顶部加入 `import asyncio`、`import logging`、`import os`、`import time` 和模块级 `logger = logging.getLogger(__name__)`（已有同名导入或 logger 时不要重复）。然后在 `AsyncLLM()` 创建 `model` 后、返回它之前，替换原来的 `model.enable_prespec(...)` 调用。`kmax`、`engine` 和 `chunk_size_sec` 继续使用本函数现有的实际变量；下面代码中的这三个名字要按你的函数局部变量映射：

```python
def log_prespec_probe(p):
    logger.info(
        "PRESPEC_PROBE ws_id=%s seg=%s boundary=%s status=%s "
        "request_id=%s ready_lead_ms=%s",
        p.session_id, p.audio_epoch, p.boundary, p.status,
        p.request_id, p.ready_lead_ms,
    )

mode = os.environ.get("ASR_PRESPEC_MODE", "off")
admission = os.environ.get("ASR_PRESPEC_ADMISSION", "gated")
if mode not in {"off", "shadow", "consume"}:
    raise ValueError(f"invalid ASR_PRESPEC_MODE: {mode}")
if admission not in {"force", "gated"}:
    raise ValueError(f"invalid ASR_PRESPEC_ADMISSION: {admission}")
model.enable_prespec(
    enabled=(mode != "off"), kmax=kmax, engine=engine,
    chunk_size_sec=chunk_size_sec,
    admission_mode=admission, shadow_only=(mode != "consume"),
    on_probe=log_prespec_probe if mode != "off" else None,
)
logger.info("PRESPEC_CONFIG mode=%s admission=%s", mode, admission)
```

`on_probe` 是同步回调；这里只做一条格式化日志，不做阻塞采样或文件读写。回调异常被协调器吞掉并计入 `probe_error_count`，因此每轮结束要检查该计数是否为 0。

### 1.2 `qwen3_asr_async.py`：记录 shadow 请求生命周期

在模型类中新增方法，并让 1.1 的 `producer` 指向它。原 `_run_prespec()` 实现不改，仍负责 engine `generate/abort`：

```python
async def _logged_prespec_producer(self, request, budget):
    started_ns = time.monotonic_ns()
    logger.info(
        "PRESPEC_REQUEST event=submit ws_id=%s seg=%s request_id=%s "
        "audio_end_samples=%s rollback_tokens=%s budget=%s",
        request.session_id, request.audio_epoch, request.request_id,
        request.visible_audio_end_samples, len(request.rollback_tokens), budget,
    )
    try:
        token_ids = await self._run_prespec(request, budget)
    except asyncio.CancelledError:
        logger.info(
            "PRESPEC_REQUEST event=cancel ws_id=%s seg=%s request_id=%s elapsed_ms=%.3f",
            request.session_id, request.audio_epoch, request.request_id,
            (time.monotonic_ns() - started_ns) / 1e6,
        )
        raise
    except Exception:
        logger.exception(
            "PRESPEC_REQUEST event=failed ws_id=%s seg=%s request_id=%s",
            request.session_id, request.audio_epoch, request.request_id,
        )
        raise
    logger.info(
        "PRESPEC_REQUEST event=done ws_id=%s seg=%s request_id=%s "
        "tokens=%s elapsed_ms=%.3f",
        request.session_id, request.audio_epoch, request.request_id,
        len(token_ids), (time.monotonic_ns() - started_ns) / 1e6,
    )
    return token_ids
```

这里的 `done` 只表示 shadow 引擎生成结束，尚不表示正式请求采纳。若 `cancel` 后底层 NPU 工作仍持续，另看 vLLM 请求结束/abort 日志和资源占用。

### 1.3 `realtime_connection.py`：记录中点准入决定

在 [第三节](streaming_prespec.md) 的 `_process_segment()` 中，紧跟 `try_start_prespec_for_state(...)` 返回之后插入。保留原有 `segment.prespec_started` 赋值：

```python
if decision is not None:
    log = logger.info if decision.launched else logger.debug
    log(
        "PRESPEC_LAUNCH ws_id=%s seg=%s launched=%s reason=%s slack_ms=%s",
        self.connection_id, seg_id, decision.launched,
        decision.reason, decision.slack_ms,
    )
if decision is not None and decision.launched:
    segment.prespec_started = True
```

`try_start_prespec_for_state()` 在未达中点、已 commit、音频队列非空或无合法回滚草稿时返回 `None`，因此这些路径**不会**出现 `PRESPEC_LAUNCH`。如果预期到达 3 秒却没有 launch，请在该包装函数这些 `return None` 前临时加一条 `DEBUG` 日志，包含 `ws_id/seg/chunk_id/len(state.buffer)/reason`；不要据“无 launch 日志”直接判断负载门控拒绝。`decision.reason` 只覆盖实际进入 `try_start_prespec()` 后的判断。

### 1.4 `qwen3_asr_async.py`：记录正式边界和候选匹配

在 [第五节](streaming_prespec.md) 的 `begin_authoritative_chunk()` 前保存 `this_chunk_id = state.chunk_id`；在正式请求已构造好 `request_id`、紧邻现有 `engine.generate(...)` 前记录候选来源。下面的 `formal_request_id` 要换成正式请求当前使用的实际局部变量（如现有 `stream_...` ID），不能填 shadow 的 `prespec-...` ID：

```python
logger.info(
    "PRESPEC_TARGET ws_id=%s seg=%s chunk=%s event=submit "
    "formal_request_id=%s candidate_request_id=%s",
    ws_id, seg_id, this_chunk_id, formal_request_id,
    candidate.request_id if candidate is not None else None,
)
```

在正式 `try` 前置 `formal_succeeded = False`，只在 L601–606 的正式 state 全部成功写回后置 `formal_succeeded = True`。在原 `finally` 中 `end_authoritative_chunk(...)` 返回后，只对**成功完成且取到原始 token IDs** 的正式请求记录匹配结果；异常或空 token 不计拒绝率：

```python
if formal_succeeded and evaluation is not None and raw_target_tokens:
    logger.info(
        "PRESPEC_EVAL ws_id=%s seg=%s chunk=%s "
        "candidate_tokens=%s matched_prefix_tokens=%s ready_lead_ms=%.3f",
        ws_id, seg_id, this_chunk_id,
        len(evaluation.candidate_tokens),
        evaluation.continuous_accepted_tokens,
        evaluation.ready_lead_ms,
    )
```

`PRESPEC_EVAL.matched_prefix_tokens` 是候选和正式输出的离线式连续前缀对照，**不是 vLLM verifier 实际接受数**。C 组还必须在引擎实际处理 `ascend_external_draft` 的验证结果处记录 `request_id`、提交的草稿长度、实际连续接受长度、首 token 是否拒绝、修正 token 数、目标模型前向步数。只在 proposer 的 `propose()` 打日志不足以证明草稿被验证；该函数只负责提出 token。先在服务/引擎仓库定位挂点：

```bash
rg -n 'ascend_external_draft|def propose\(|accepted.*token|rejected.*token' .
```

日志关联以 `ws_id + seg + chunk` 为正式键，以 `prespec-...` 为 shadow 引擎请求键；不要把不同 request 的 prefill/decode 打点混在一起。`PRESPEC_PROBE status=ready_consumed` 发生在正式验证**之前**，不能代替 verifier 日志。

先把本仓库的两个模块复制到服务的 `qwen_asr/inference/`，监测脚本复制到测试工作目录；若两仓库在同一主机上，在服务项目根目录执行（替换检出路径）：

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
rg -n 'PRESPEC_CONFIG|PRESPEC_LAUNCH|PRESPEC_REQUEST|PRESPEC_PROBE|PRESPEC_TARGET|PRESPEC_EVAL' \
  qwen_asr/inference/qwen3_asr_async.py qwen_asr/cli/realtime_connection.py
```

## 2. 单连接链路检查：B 组先强制 shadow

在 910B 服务项目完成第 1 节的代码接线、复制与语法检查后，按 [主接入说明](streaming_prespec.md) 导入。以服务现有启动命令启动一份 B 组实例：

```bash
export ASR_PRESPEC_MODE=shadow
export ASR_PRESPEC_ADMISSION=force
<现有服务启动命令>
```

使用现有 WebSocket 测试脚本发送**同一段超过 4 秒、在 2 秒正式 chunk 后仍有合法回滚草稿**的音频；这一步并发为 1，不开启自动补位。服务若写日志文件：

```bash
tail -F /path/to/server.log | rg --line-buffered 'PRESPEC_|inference submit|Commit received|Segment done'
```

若服务由 systemd 管理，改用 `journalctl -u <实际服务单元名> -f -o cat` 作为管道左侧。预期按序看到 `PRESPEC_CONFIG`、中点 `PRESPEC_LAUNCH launched=True`、`PRESPEC_REQUEST submit/done`、正式 chunk 的 `PRESPEC_PROBE boundary=chunk status=ready_shadow_only` 和 `PRESPEC_EVAL`。`ready_lead_ms > 0` 表示候选早于正式 chunk 提交完成；若为 `pending`，候选被取消，不应等待它。随后分别重放提前 commit、≤0.4 秒短尾和断连，确认无遗留 `prespec-...` 请求、重复 begin 或未结束的 session。

如果链路没有 launch，打开 DEBUG 日志后先看 `return None` 的跳过原因和 `PRESPEC_LAUNCH reason`；`force` 仅绕过实时负载门控，不绕过中点、草稿、在途正式请求和容量检查。

单轮结束后可快速检查事件是否齐全、拒绝原因分布；路径替换为该轮切出的服务日志：

```bash
rg -c 'PRESPEC_LAUNCH.*launched=True|PRESPEC_REQUEST event=done|PRESPEC_PROBE.*ready_' /path/to/run.log
rg -o 'PRESPEC_LAUNCH.*reason=[^ ]+' /path/to/run.log | sort | uniq -c
```

## 3. 固定输入正确性：A、B、C 逐级比较

对同一批音频保存**每条连接的音频文件、切片顺序、发送时刻和 commit 时刻**。三组分别重启服务并使用同一回放清单：

```bash
export ASR_PRESPEC_MODE=off      # A；保持现有回滚投机配置不变
<现有服务启动命令>
# 使用现有 WebSocket 回放命令，保存客户端事件和服务日志

export ASR_PRESPEC_MODE=shadow   # B；仍用原回滚草稿正式推理
export ASR_PRESPEC_ADMISSION=force
<现有服务启动命令>
# 同一回放清单

export ASR_PRESPEC_MODE=consume  # C；先仅在固定输入回放中启用消费
export ASR_PRESPEC_ADMISSION=force
<现有服务启动命令>
# 同一回放清单
```

这里的 `<现有服务启动命令>` 和 WebSocket 回放命令是环境专属的唯一占位符：本仓库没有 910B 服务启动脚本或你的 `test_v2.py` 参数表，不能编造 CLI。A/B 应逐 chunk 对比正式原始 token IDs、客户端 delta/done、最终文本与 commit 边界；C 还要确认候选实际写进正式 `ascend_external_draft` 并由引擎 verifier 验证。覆盖首 token 拒绝、部分接受、全拒绝、版本不匹配和候选超时。若原始 token 对齐基准不是同一个正式 prompt/前缀，`matched_prefix_tokens` 不可用于接受率计算。此步不通过就不要开始性能对比。

## 4. 同负载性能对比：A→B、B→C、A→C

先在 B/C 中设置 `export ASR_PRESPEC_ADMISSION=force`，完成初次同负载对比；这一步只能衡量后台请求的收益与干扰，**不能证明低利用率准入策略有效**。分别跑现有 `sustained_test` 的并发档位 `1、5、20`，再扩到拐点附近；每组保持相同音频清单、到达时序、自动补位规则、持续时间和预热。A/B/C 交错排序，至少重复 3 轮，记录实际测试起止 UTC。每轮单独设置 `run_id`、切片服务日志、导出该轮 `commits.jsonl`。导出行须含 `ws_id, seg, received_at, done_at, commit_latency_ms, queue_wait_ms, queue_size, is_empty_commit`；空 commit 不算正式时延。客户端执行命令仍用你现有 `test_v2.py` 的真实 CLI。

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
export ASR_PRESPEC_ADMISSION=gated
<现有服务启动命令>
```

从 DEBUG `PRESPEC_LAUNCH reason` 汇总 `missing_load/stale_load/formal_queue_nonempty/utilization_high/insufficient_slack/launched` 的次数，并在正式日志中检查每次 launch 前后及下一个请求的队列等待。若几乎不 launch，结论是“当前门控策略覆盖不足”，不能据此称候选无效；若 launch 后正式 P95/P99 或后续请求排队恶化，收紧门控或停用，不以 NPU 利用率上升替代时延证据。
