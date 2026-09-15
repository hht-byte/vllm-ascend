# 310P token-only FULL graph（实验实现）

基线为 vLLM-Ascend v0.23.0 `5cb98caaadeff42b5b62b996e34bb2aaa29d20fd`，配套 vLLM v0.23.0 `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`。

本实现按 token 桶复用模型图，统一使用 `_npu_paged_attention_splitfuse_v2`，保留每条请求的多 query 计算。没有改为逐 token PA，也没有修改配套 vLLM。

## 当前验证状态

- 用户此前的 context_lens 原地更新和同拓扑 task update 算子实验已通过；它不等价于此 splitfuse 图路径已通过。
- 本地 19 项 CPU / 调用边界测试通过；Ruff 检查和 Python 编译检查通过。
- 本会话运行环境为 Windows、CPU PyTorch 2.12.1，无法运行 torch_npu / 310P 测试。跨请求数设备精度、workspace、整模型与性能验收仍待完成。
- 功能默认关闭。若算子不支持零 query 行或 task descriptor 更新，会报错，不会偷偷按请求数捕获另一张图。

## 先运行设备探针

根据最新设备反馈，默认更新模式改为 `fixed + inplace`。task_update 保留为显式对照选项，不作为主路径。inplace 的 splitfuse capture 也已报告失败，因此该默认选择是实现方向，不代表设备验收通过。

在打补丁后的 vLLM-Ascend 根目录执行，每种组合使用独立进程：

```bash
python examples/310p/probe_token_graph.py --buckets 20 --layout fixed --update-mode task_update --output fixed-update.json
python examples/310p/probe_token_graph.py --buckets 20 --layout fixed --update-mode inplace --output fixed-inplace.json
python examples/310p/probe_token_graph.py --buckets 20 --layout active --update-mode task_update --output active-update.json
```

fixed 使用每桶固定请求容量；未用行 q_len=0，因此必须验证算子接受空行。active 不传空行，而是通过 task update 切换有效视图的 shape。active 与 inplace 的组合被拒绝。

通过所选模式的 20-token 探针后，扩大到全部桶。例如：

```bash
python examples/310p/probe_token_graph.py --buckets 20 80 192 --layout fixed --update-mode inplace --repeats 5 --output all-buckets.json
```

探针复用运行时的 metadata arena 和 task 更新代码，覆盖下列场景：

| 桶 | 场景 |
| --- | --- |
| 20 | 非均匀 8 请求、qLens 重排、10 请求、10/18 个真实 token 补齐、20 个单 token 请求、切回 8 请求 |
| 80 | `[1,1,1,1,5,70]`、20 个单 token 请求、4×20 token |
| 192 | `[70,40,70]`、单请求 192 token、20 个单 token 请求 |

每轮同时更换 query、真实 K/V、KV 历史长度和 block table，跨越 block 边界。比较对象包括独立 CPU 因果 GQA 参考以及不含空槽/dummy 的 eager splitfuse；检查真实槽之外的 KV 没有被改写。task update 后重新将输出置 NaN，再 replay，防止把 update 阶段的输出误当成 replay 结果。

报告包含 graph_id、capture 次数、metadata 地址、最大误差、含同步开销的 update/replay 时间、峰值显存。每桶只有一次 capture，所有桶的图对象和输入输出缓冲区都保留。`--repeats 2` 以上会按 `20 → 80 → 192 → 20 → 80 → 192` 的顺序切换桶，验证其他桶覆盖共享 metadata 后旧图仍可复用。探针时间是诊断数据，不是整模型性能结论。

如某个模式失败，保存完整报错与软件版本；不要删除断言、放宽精度比较或在失败后自动重捕获。仅通过 context_lens 的旧测试，不足以选择 inplace 模式。

## Capture 报 allocator / PagedAttentionOperation 错误时

后续 inplace 日志也在 capture_end 出现相同 PagedAttentionOperation / allocator 错误：task update 不是该错误的必要触发条件。退出清理期间的 `stream is captured` 出现在 capture 失败之后，不能当作独立的首因。当前优先使用 inplace：

```bash
ASCEND_LAUNCH_BLOCKING=1 python examples/310p/probe_token_graph.py --buckets 20 --layout fixed --update-mode inplace --eager-only --output eager-inplace.json
python examples/310p/probe_token_graph.py --buckets 20 --layout fixed --update-mode inplace --graph-pool private --output private-inplace.json
```

如果第二条仍在 capture 失败，保持其它参数不变，增加 `--capture-case dense`，输出到另一个文件。dense 先捕获 `[1]*20`，无零 query 行、无 dummy、无多 query 请求；如果连它也不能 capture，则这些因素都不是失败的必要条件。若 dense capture 成功，后续仍在同一图上运行原有 8/10 请求用例，不增加 capture 次数。该对照使用同一 splitfuse_v2，不等同于其它 PA 算子的成功实验。

2026-09-15 收到的 fixed/task_update 日志在 capture 内 `graph_task_group_end` 暴露异步错误，包含 `aclrtAllocatorGetByStream ... stream is not registered with any allocator`。尚未确认这是首个底层错误，也未证明与零 query 行或共享 pool 存在因果关系。`pool=((0,1),)` 是 torch_npu graph 上下文的正常参数包装日志，不要手动拆解 pool handle。

先更新本分支，再用独立进程依次执行。第一条只验证首个 case，绝不进入 capture：

```bash
ASCEND_LAUNCH_BLOCKING=1 python examples/310p/probe_token_graph.py --buckets 20 --layout fixed --update-mode task_update --eager-only --output eager.json
python examples/310p/probe_token_graph.py --buckets 20 --layout fixed --update-mode task_update --graph-pool private --output private-update.json
python examples/310p/probe_token_graph.py --buckets 20 --layout fixed --update-mode inplace --graph-pool private --output private-inplace.json
```

后两条图测试须保证没有继承 `ASCEND_LAUNCH_BLOCKING=1`。失败后新启进程，不在失效的 graph/stream 上重试。保留完整终端输出及同一时间段的 CANN/ATB 日志，包括 Python traceback 之前的首个底层错误。

- eager 失败：先定位相同输入的 splitfuse / cache 问题，不能归因于 graph task update。可用 active 布局的 eager-only 作为无零 query 行对照。
- eager 通过、private/task_update 通过，而原 shared/task_update 失败：才有证据进一步调查共享 pool 差异；private 仅用于诊断，不代表整模型共享池已验证。
- private/task_update 在 capture 失败、private/inplace 能 capture：缩小到 task group 与算子捕获组合。若 inplace 后续精度失败，不算 inplace 模式通过。
- 两种 private 模式均在 capture 失败：继续检查算子图支持和目标软件栈的 allocator/workspace 路径。

新增 `[EAGER PASS]` 是和独立 CPU 参考比较通过；`[CAPTURE PASS]`、`[UPDATE PASS]`、`[REPLAY PASS]` 是对应阶段已返回并同步，不代替最终精度断言。环境日志记录 torch/torch_npu 版本；另需完整 CANN、ATB、HDK 版本及已通过 Test A 的实际算子与 pool 配置。

## 模型侧启用

在原有模型启动命令中合并以下参数，保留原有模型和 ASR 参数：

```bash
--no-async-scheduling \
--max-num-seqs 20 \
--compilation-config '{"cudagraph_mode":"FULL","cudagraph_capture_sizes":[20,80,192],"max_cudagraph_capture_size":192}' \
--additional-config '{"token_graph_310p":{"enabled":true,"request_layout":"fixed","update_mode":"inplace"}}'
```

`request_layout` / `update_mode` 必须与已通过的设备探针对应；如已有 additional-config，将 token_graph_310p 合并进去，不覆盖其他设置。不要同时启用 enforce-eager。超出最大桶的 batch 沿用 eager，不创建新桶。

初版范围：单卡 dense causal decoder、单一普通 FullAttentionSpec KV 组、非量化 KV、无滑窗/ALiBi/sinks/KV sharing、无 LoRA/KV transfer/ENPU。投机验证初版支持 ngram；带独立 draft 模型的方法明确拒绝，避免将其 metadata 或图生命周期混入 target graph。Qwen3-ASR 的音频编码器仍使用原有路径，此改动针对 decoder。

启用 `VLLM_LOGGING_LEVEL=DEBUG` 可查看 `310P token graph` 日志：bucket、真实 token/request 数、metadata 行数、attention task 数。attention task 数是层数，不能当作模型图数量；图数量看 ACLGraphWrapper 的 capture 日志与实际 graph entry 数。

## 实现说明与未关闭的验收项

- Dispatcher 为 Ascend 插件内子类，描述符统一 `num_reqs=None, uniform=False`；初始化和运行时选择使用相同规则。
- KV 写入固定为桶 T，slot_mapping 为持久的一维 int32，padding slot=-1；调度/采样使用的真实计数保留在原始 common metadata。
- dummy 请求仅从有效 block 0 读取，block table 每列重复 0，KV 可见长度等于 dummy q_len，因此长 padding 不会索引到不存在的 block。dummy 不写 KV、不参与采样；probe 校验无非真实槽写入。此项仍需整模型验证。
- metadata 在 runner 的单一 arena 中分配，各桶使用视图；host qLens 改写前等待上轮完成。每层 attention task 在 capture 时登记，后续在 replay 前更新。初版使用保守同步，后续再基于 profiling 改成更细粒度事件依赖。
- 当前 task 记录强引用保留每层、每桶的 query/output，保证更新参数的生命周期；这会限制 graph pool 对中间激活的复用，显存可能随层数和各桶 token 数之和增长。共享 metadata 不等于所有图显存仅按最大桶分配。整模型验收必须记录逐桶捕获后的显存，再评估现有 `weak_ref_tensors` 机制是否适用，不能在未验证生命周期前直接释放引用。
- splitfuse_v2 的现有 Python 接口没有显式 workspace 参数，当前由 ATB 管理 setup / workspace。代码没有声称已证明全范围 workspace 上界。必须通过目标软件栈的长度、请求数、连续多轮和整模型压力测试，尤其关注 task update 后资源变化。
- 预热也使用同一 splitfuse 分支，避免非均匀 dummy batch 被送到单 query PA。
- 本地调用边界测试使用真实 runner 方法的 AST 和配套 dispatcher 源码，替代环境依赖；它不能证明完整引擎可导入或设备启动成功。

## CPU 检查

```bash
python tests/ut/_310p/test_token_graph_standalone.py
python tests/ut/_310p/test_token_graph_wiring.py
```

第二个测试需要配套 vLLM 位于相邻目录 `../vllm`。测试无需 pytest 或 torch_npu，但需要 CPU PyTorch。
