# vLLM 0.23：预投机首轮并行验证与失配重接

默认设置让正式请求把**完整预投机候选或完整回滚草稿** `C[0:]` 放入首次目标执行：scheduler 在 prefill 前写入 `request.spec_token_ids`，同次前向验证 `C[0]` 及后续 token。shadow 请求仍沿用“目标先采样一个 token，再由 proposer 提出旧回滚草稿 `C[1:]`”的合同。首次 verifier 输出修正 token 后，proposer 保留本次完整草稿，根据**已确认的正式输出**提出下一轮草稿：首次失配在位置 0 时按位置续接；失配在后续位置时使用 `aligned` 或 `ngram_1/2/3`，默认 `ngram_1`。直到请求结束才清理草稿池。

现有 `patch_external_draft.py` 的两个 monkey-patch 继续负责 proposer 注册和 `ascend_external_draft` 传输。本补丁不改这个文件，也不改 `model_runner_v1.py`：proposer 继承 `AscendNgramProposer`，直接复用 0.23.0 已有的 ngram 分派；新 `patch_external_draft_first_verify.py` 在原补丁之后叠加 first-verify 和 rejoin 元数据。**仅靠现有两个 worker 侧 patch 不能完成首次并行验证**：waiting 请求的首次 token 数与 `scheduled_spec_decode_tokens` 由 scheduler 决定，所以补丁仍需修改普通及 recompute scheduler 的相应分支。

## 1. 引擎补丁

[补丁文件](external_draft_first_verify_v023.patch) 是基于本机可读的 vLLM 0.23 / vLLM-Ascend 对应源码生成的 unified diff。它修改：

- `vllm/v1/core/sched/scheduler.py`：普通 scheduler 首次执行前排入完整草稿；
- `vllm_ascend/core/recompute_scheduler.py`：recompute scheduler 做同一处理；
- `vllm_ascend/spec_decode/external_draft_proposer.py`：复用已有 ngram 分派，保留候选池并逐轮重接；
- `vllm_ascend/patch/worker/__init__.py`：在原有 patch 导入之后导入新 patch；
- 新增 `vllm_ascend/patch/worker/patch_external_draft_first_verify.py`：只存取 first-verify 标记与策略；
- 新增 `vllm_ascend/spec_decode/external_draft_rejoin.py`：纯 token-ID 选择逻辑。

在 910B 的 **vLLM 源码根目录**执行：

```bash
git apply --check --include='vllm/*' /path/to/external_draft_first_verify_v023.patch
git apply         --include='vllm/*' /path/to/external_draft_first_verify_v023.patch
```

在 **vLLM-Ascend 源码根目录**执行：

```bash
git apply --check --include='vllm_ascend/*' /path/to/external_draft_first_verify_v023.patch
git apply         --include='vllm_ascend/*' /path/to/external_draft_first_verify_v023.patch
```

若 910B 源码与参考源码不同，先运行 `git apply --check`，按冲突函数人工移植；不要强制套用。尤其检查实际运行的是普通 scheduler 还是 recompute scheduler。你描述的原 patch 包装 `spec_decode.get_spec_decode_method`；本地参考版包装 `NPUModelRunner._get_drafter`。两者都负责注册 proposer，本补丁不改这段，保持服务器原实现。若 `patch/worker/__init__.py` 的导入行也不同，只需在原 patch 的导入**后**加一行 `import vllm_ascend.patch.worker.patch_external_draft_first_verify`，并确认每个 worker 进程均执行。引擎重启后要看到 `INITIAL_DRAFT ... status=scheduled draft_tokens=N`，否则不能声称完成首轮验证。`status=fallback` 表示本轮无法把完整 prompt 和草稿排进同次执行，转回原先“先采样首 token”的路径；不得把它算作首轮收益。

### 已应用旧版补丁的 910B 热修复

若服务启动时报 `BalanceScheduler object has no attribute speculative_config`，在服务器已修改的 `vllm/v1/core/sched/scheduler.py` 的 `_prepare_initial_external_draft()` 中，把唯一一行

```python
if not self.speculative_config or len(draft) > self.num_spec_tokens:
```

改为

```python
if self.num_spec_tokens <= 0 or len(draft) > self.num_spec_tokens:
```

然后重启 EngineCore。这里 `speculative_config` 在 0.23.0 的 `Scheduler.__init__` 中只是局部变量，实际保存到实例的是 `num_spec_tokens`；不要对已应用过的引擎源码再次执行整份 unified diff。修复后先确认请求不再触发 EngineCore fatal，再用 `INITIAL_DRAFT ... status=scheduled` 和 verifier 日志验证首轮候选确实执行。

## 2. 服务端参数与 chunk/finish 两个挂点

启动 parser 增加：

```python
parser.add_argument(
    "--prespec-rejoin-policy",
    choices=("aligned", "ngram_1", "ngram_2", "ngram_3"),
    default="ngram_1",
)
```

沿现有 `Qwen3ASRModelAsync.AsyncLLM(**base_config)` 调用显式传 `prespec_rejoin_policy=args.prespec_rejoin_policy`；类方法接收它后传入 `enable_prespec(rejoin_policy=prespec_rejoin_policy)`，后者校验并保存 `self.prespec_rejoin_policy`。详细形参位置见 [验证文档](streaming_prespec_validation.md)。不用环境变量。

把 [服务侧 helper](streaming_prespec_formal_args.py) 复制到 `qwen_asr/inference/streaming_prespec_formal_args.py`。910B 的 `_split_stream_rollback()` 在固定阶段用 `cur_ids[1:]`、动态阶段用 `cur_ids[end_idx + 1:]` 形成旧草稿；两者都缺少首次待验证的 token。保留旧 `external_draft` 给 shadow 使用，并在该函数原有 `return` 前增加完整正式草稿：

```python
from qwen_asr.inference.streaming_prespec_formal_args import (
    formal_external_draft_extra_args,
    formal_rollback_draft_tokens,
)

# 在 _split_stream_rollback() 的 while/if 分支结束之后、原 return 之前：
rollback_full_token_ids = formal_rollback_draft_tokens(
    cur_ids, stable_prefix_token_ids, kmax=kmax,
)
return (prefix, stable_prefix_token_ids, external_draft,
        raw_len, used_len, rollback_full_token_ids)
```

固定阶段完整草稿从 `cur_ids[0]` 开始；动态阶段从 `cur_ids[end_idx]` 开始。更新所有解包调用点以接收第六项，原前五项保持原义；中点 shadow 仍只用第三项 `external_draft`。在**正式 chunk** 和 **正式 finish** 已领取 candidate、已打包实际请求，但尚未调用 `engine.generate` 时，对实际 `request_kwargs["sampling_params"]` 执行：

910B 当前三个拆分调用点分别在 `qwen3_asr_async.py:558`（中点，只接收并忽略第六项）、`:673`（正式 chunk，使用第六项）和 `:968`（正式 finish，使用第六项）；两处正式 helper 调用在 `:751` 和 `:1047`。这些行号仅用于定位，修改后会移动。不要让中点 shadow 改用完整草稿。

```python
selected_draft_token_ids = (
    candidate.formal_draft_token_ids if candidate is not None
    else rollback_full_token_ids
)
# _pack_stream_request(..., draft_token_ids=selected_draft_token_ids, ...)
request_kwargs["sampling_params"].extra_args = formal_external_draft_extra_args(
    request_kwargs["sampling_params"].extra_args,
    candidate_tokens=(candidate.formal_draft_token_ids
                      if candidate is not None else None),
    rollback_tokens=external_draft,
    rollback_full_tokens=rollback_full_token_ids,
    kmax=self.kmax,
    rejoin_policy=self.prespec_rejoin_policy,
)
submitted_draft_ids = tuple(
    request_kwargs["sampling_params"].extra_args["ascend_external_draft"]
)
draft_stats_valid = submitted_draft_ids == tuple(selected_draft_token_ids)
```

`request_kwargs` 必须是本次正式请求实际传给引擎的字典；`SamplingParams` 必须为该请求专用。helper 在有候选时选择候选，无候选时选择完整回滚草稿，并在草稿非空时设置首次验证标记和重接策略。空草稿不标记，正常自回归。旧 `rollback_tokens` 仍用于核对完整回滚草稿没有错位；正式 chunk 和 finish 均必须传 `rollback_full_tokens`，遗漏即报错。`submitted_draft_ids` 和 `draft_stats_valid` 必须在 helper 覆盖 `extra_args` **之后**从实际请求重算；`PRESPEC_TARGET`、`ASR_FORMAL_STATS` 的 `submitted_draft_tokens` 应依据最终提交值。保留原有 prompt、完整音频、稳定前缀、温度和输出模式。shadow `_run_prespec()` 不设置 `ascend_external_draft_first_verify`。

### 2.1 回滚草稿首轮验证 A/B 开关

上一节的接线示例对应默认开启设置；A/B 实验应以本节的选择和调用代码替换该示例。

服务侧 helper 增加 `rollback_first_verify: bool = True`，默认保持当前完整回滚草稿首次验证。910B 启动 parser 使用 `argparse.BooleanOptionalAction` 提供两个互斥形式：

```python
parser.add_argument(
    "--rollback-first-verify",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="verify full rollback draft on first target execution",
)
```

沿现有 `Qwen3ASRModelAsync.AsyncLLM(**base_config, ...)` 参数链传入 `rollback_first_verify=args.rollback_first_verify`，在模型实例上保存为 `self.rollback_first_verify`。若 parser 所在文件尚未导入 `argparse`，先补 `import argparse`。正式 chunk 与 finish 两处均按下面选择草稿、打包请求、覆盖 `extra_args`；中点 shadow 不读取此开关：

```python
selected_draft_token_ids = (
    candidate.formal_draft_token_ids if candidate is not None
    else (rollback_full_token_ids if self.rollback_first_verify
          else rollback_draft_token_ids)
)
# _pack_stream_request(..., draft_token_ids=selected_draft_token_ids, ...)
request_kwargs["sampling_params"].extra_args = formal_external_draft_extra_args(
    request_kwargs["sampling_params"].extra_args,
    candidate_tokens=(candidate.formal_draft_token_ids
                      if candidate is not None else None),
    rollback_tokens=rollback_draft_token_ids,
    rollback_full_tokens=rollback_full_token_ids,
    kmax=self.kmax,
    rejoin_policy=self.prespec_rejoin_policy,
    rollback_first_verify=self.rollback_first_verify,
)
submitted_draft_ids = tuple(
    request_kwargs["sampling_params"].extra_args["ascend_external_draft"]
)
draft_stats_valid = submitted_draft_ids == tuple(selected_draft_token_ids)
draft_source = ("prespec" if candidate is not None else
                "rollback" if submitted_draft_ids else "none")
```

关 (`--no-rollback-first-verify`) 且无候选时，helper 只提交原有 `rollback_draft_token_ids=C[1:]`，删除首次验证及重接标记。worker 保留旧的一次性 proposer 路径：目标首次执行先产生 `T[0]`，下一轮才验证旧草稿；原旧草稿可能长达 `kmax`，不能直接用 `rollback_full_token_ids[1:]` 代替，否则会无意少提交最后一个 token。开时提交 `rollback_full_token_ids=C[0:]`，包括首 token。两种设置下，只要领取了 prespec 候选，仍提交完整候选并启用首次验证。`submitted_draft_tokens`、`draft_source` 和统计有效性必须在 helper 覆盖后按实际发送值重算，尤其要处理关时完整回滚只有一个 token、旧草稿为空的 `draft_source=none` 情况。

隔离比较时，先确认 `--prespec-mode off` 下正式 chunk 和 finish 仍调用该 helper，再用同一组音频和并发分别以 `--prespec-mode off --rollback-first-verify`、`--prespec-mode off --no-rollback-first-verify` 运行并重启服务，其他参数不变。真实 consume 负载也可固定 `--prespec-mode consume` 重复两组；同时报告候选覆盖率，避免仅按本轮 `draft_source=rollback` 过滤造成样本选择偏差。开组的 rollback 正式请求应看到 `INITIAL_DRAFT status=scheduled`，首个 `ASR_VERIFY_STEP` 的有效草稿包含 `C[0]`；关组不应有该 `INITIAL_DRAFT`，首次验证在目标已采样 `T[0]` 后进行，首个 verifier 接受数对应 `C[1:]`，不能用于计算首 token 匹配率。两组比较同音频请求的输出 token 数、正式解码轮次、实际目标执行次数、commit 延迟和队列等待；`decode_rounds` 不是设备前向调用计数。旧 patch 不需重新应用，只需替换服务侧 helper 与两处接线后重启。

这一开关比较的是“完整回滚首轮验证并允许后续重接”和“目标先采样首 token、旧草稿一次性验证”的整体效果；当前 worker 用首轮验证标记同时控制重接，关组也关闭了重接。若要单独估计首 token 是否参与首次执行的因果效应，两组还需使用相同的失配后重接行为，并扩展 worker 元数据和 proposer，不能把本 A/B 的全部差异归因于首 token。

本仓 `Candidate.formal_draft_token_ids` 在 `streaming_prespec.py` 返回完整 `self.token_ids`。若 910B 服务副本返回 `token_ids[1:]`，须先将其改为完整序列，否则有候选路径仍无法验证第 0 位。正式 chunk 和 finish 都已有 `formal_external_draft_extra_args()` 调用点；两处都应按上述逻辑接入。

候选预算现在**包含首 token**，因此 `--prespec-candidate-tokens` 默认 `kmax`，不得大于 `kmax`。例如现有引擎 `kmax=15` 时，应把旧的候选预算 16 改为 15，或先提高引擎预算。服务 helper 与 scheduler 均拒绝超过 `kmax` 的候选；不静默截断。

## 3. 运行时判定与验证

一条 `C=(c0,c1,c2,...)` 的正式请求，预期日志顺序：

1. 有候选时 `PRESPEC_PROBE status=ready_consumed`、`draft_source=prespec`；无候选且回滚草稿非空时 `draft_source=rollback`。两者提交长度均为所选完整 `C[0:]` 的长度；
2. `INITIAL_DRAFT status=scheduled draft_tokens=len(C)`，出现在目标首次执行之前；
3. `ASR_VERIFY_STEP round_index=0`，`effective_draft_tokens` 包含 `c0`；
4. 如果 `c0` 被拒绝，目标确认 `t0` 后下一轮按位置提交 `C[1:]`；如果 `ck`（`k>0`）被拒绝，proposer 只用已确认的 `T[:k+1]` 查找 ngram 锚点；
5. 请求结束/取消时，worker 的 `remove_request` 清理候选池和策略。

先运行 CPU 检查：

```bash
python -m unittest discover -s research -p 'test_external_draft_rejoin.py' -v
python -m unittest discover -s research -p 'test_streaming_prespec_formal_args.py' -v
python -m unittest discover -s research -p 'test_external_draft_first_verify_patch.py' -v
```

然后在 910B 上用现有 sustained test 做无候选与 consume 同负载对比，检查两组 `INITIAL_DRAFT`、首轮 verifier 接受数、总模型执行轮次、commit 延迟 P50/P95/P99、正式请求队列等待、候选请求开销和文本等价。旧的 rollback baseline 使用 `C[1:]`，新无候选组使用 `C[0:]`，不能把两者当作相同实现混算。`PRESPEC_MATCH_TOKENS` 的离线匹配仅是诊断，不能替代真实 verifier 接受数；NPU/多并发正确性与时延收益仍需实测。
