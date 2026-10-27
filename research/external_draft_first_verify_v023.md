# vLLM 0.23：预投机首轮并行验证与失配重接

本变更把**预投机候选** `C[0:]` 放入正式目标请求的首次执行：scheduler 在 prefill 前写入 `request.spec_token_ids`，同次目标前向验证 `C[0]` 及后续候选。旧回滚草稿和 shadow 请求仍沿用“目标先采样一个 token，再由 proposer 提出旧草稿”的合同。首次 verifier 输出修正 token 后，proposer 保留候选池，根据**已确认的正式输出**提出下一轮草稿：首次失配在位置 0 时按位置续接；失配在后续位置时使用 `aligned` 或 `ngram_1/2/3`，默认 `ngram_1`。直到请求结束才清理候选池。

## 1. 引擎补丁

[补丁文件](external_draft_first_verify_v023.patch) 是基于本机可读的 vLLM 0.23 / vLLM-Ascend 对应源码生成的 unified diff。它修改：

- `vllm/v1/core/sched/scheduler.py`：普通 scheduler 首次执行前排入完整草稿；
- `vllm_ascend/core/recompute_scheduler.py`：recompute scheduler 做同一处理；
- `vllm_ascend/worker/model_runner_v1.py`：显式分派 `AscendExternalDraftProposer`；
- `vllm_ascend/spec_decode/external_draft_proposer.py`、`patch/worker/patch_external_draft.py`：候选池和逐轮策略；
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

若 910B 源码与参考源码不同，先运行 `git apply --check`，按冲突函数人工移植；不要强制套用。尤其检查实际运行的是普通 scheduler 还是 recompute scheduler。引擎重启后要看到 `INITIAL_DRAFT ... status=scheduled draft_tokens=N`，否则不能声称完成首轮验证。`status=fallback` 表示本轮无法把完整 prompt 和草稿排进同次执行，转回原先“先采样首 token”的路径；不得把它算作首轮收益。

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

把 [服务侧 helper](streaming_prespec_formal_args.py) 复制到 `qwen_asr/inference/streaming_prespec_formal_args.py`。在**正式 chunk** 和 **正式 finish** 已领取 candidate、已构建该请求专用 `SamplingParams`，但尚未调用 `engine.generate` 时，执行同一段代码：

```python
from qwen_asr.inference.streaming_prespec_formal_args import (
    formal_external_draft_extra_args,
)

# candidate 是 begin_authoritative_chunk() 或 claim_formal_candidate() 的结果。
# rollback_draft_token_ids 仍由原有回滚拆分逻辑产生。
sampling_params.extra_args = formal_external_draft_extra_args(
    sampling_params.extra_args,
    candidate_tokens=(candidate.formal_draft_token_ids
                      if candidate is not None else None),
    rollback_tokens=rollback_draft_token_ids,
    kmax=self.kmax,
    rejoin_policy=self.prespec_rejoin_policy,
)
```

此处 `sampling_params` 必须是本次正式请求专用对象。helper 返回新 `extra_args` 字典，并在回退到 rollback 时删除旧的首次验证标记，避免污染下一个请求。`_pack_stream_request` 若已经写入 `ascend_external_draft`，就在其返回 `SamplingParams` 后、`engine.generate(...)` 前覆盖这三个字段；prompt、完整音频、稳定前缀、温度和输出模式保持原流程。shadow `_run_prespec()` 不设置 `ascend_external_draft_first_verify`。

候选预算现在**包含首 token**，因此 `--prespec-candidate-tokens` 默认 `kmax`，不得大于 `kmax`。例如现有引擎 `kmax=15` 时，应把旧的候选预算 16 改为 15，或先提高引擎预算。服务 helper 与 scheduler 均拒绝超过 `kmax` 的候选；不静默截断。

## 3. 运行时判定与验证

一条 `C=(c0,c1,c2,...)` 的正式请求，预期日志顺序：

1. `PRESPEC_PROBE status=ready_consumed`，正式请求 `draft_source=prespec` 且提交长度为 `len(C)`；
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

然后在 910B 上用现有 sustained test 做 rollback 与 consume 同负载对比，检查首轮 verifier 接受数、总模型执行轮次、commit 延迟 P50/P95/P99、正式请求队列等待、候选请求开销和文本等价。`PRESPEC_MATCH_TOKENS` 的离线匹配仅是诊断，不能替代真实 verifier 接受数。这个补丁尚未在 910B 上运行，NPU/多并发正确性与时延收益仍需实测。
