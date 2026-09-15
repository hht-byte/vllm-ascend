# 310P token-only FULL graph（实验实现）

基线：vLLM-Ascend v0.23.0 `5cb98caaadeff42b5b62b996e34bb2aaa29d20fd`，配套 vLLM v0.23.0 `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`。没有修改配套 vLLM。

## 当前方案：token + inplace

根据 Ascend310P3、torch 2.10.0+cpu、torch_npu 2.10.0.post4 的实测，splitfuse_v2 的 query/KV、context、block table 原地更新各通过 8 轮；但固定八请求的正数 host qLens 从 `[3,3,3,3,2,2,2,2]` 变为 `[2,2,2,2,3,3,3,3]` 时，eager 正确而图输出 71.2% 元素不满足精度要求。旧的 host qLens 原地修改方案不再作为模型主路径。

新默认 `request_layout=token, update_mode=inplace` 将每个 query token 映射为一个算子行。仍调用 `_npu_paged_attention_splitfuse_v2`，但算子收到的 host qLens 恒为 `[1]*T`，不会随调度 qLens 变化。对于真实请求的 Q 个 query、总 KV 长度 C，第 j 个 query（从 0 开始）对应：

- `context = C - Q + j + 1`，只看当前 query 及之前的 KV。
- block table 复制该真实请求的映射。
- query、K/V、输出顺序不变；K/V 写入仍按真实 slot mapping。

例如 `Q=3,C=130` 对应 context `[128,129,130]`。虽然先写完全部新 K/V，较早 query 的可见长度仍排除未来 token。对于 T-N 个 padding token，每行 qLen=1、context=1、block table 全零、slot=-1，不写 KV、不参与采样。调度和采样仍使用真实请求数与 token 数。

这样同一 T 图可表示不同请求数和 qLens，而算子行数、query shape、host qLens 均不变。注意：这是逐 token attention 计算，可能增加 prefill 的 KV 读取，不能宣称保留了原多 query 的性能。

## 先运行设备探针

先重跑此前失败的固定八请求对照，使用新布局：

```bash
python examples/310p/probe_token_graph.py --buckets 20 --layout token --update-mode inplace --graph-pool private --control qlens --output token-qlens.json
```

真实请求始终八条；日志 rows=20 是算子行数。`scheduled` 会变化，但 `operator_qlens` 必须一直是 20 个 1。随后验证 8/10 请求切换和 padding：

```bash
python examples/310p/probe_token_graph.py --buckets 20 --layout token --update-mode inplace --graph-pool private --output token-requests.json
```

最后扩大到共享 graph pool、多桶交错回放：

```bash
python examples/310p/probe_token_graph.py --buckets 20 80 192 --layout token --update-mode inplace --graph-pool shared --repeats 5 --output token-all.json
```

每个命令使用独立进程；图测试不要设置 `ASCEND_LAUNCH_BLOCKING=1`。默认 mixed 用例覆盖 `[1,1,1,1,1,5]`、`[1,1,1,1,5,70]`、`[1,1,1,1,7,7]`、`[70,40,70]` 以及 20 tokens 下 8/10 请求切换。每桶只 capture 一次，多轮按 `20→80→192→20` 顺序回放。

每轮比较独立 CPU 因果 GQA 参考、原始真实请求 eager splitfuse 和 graph；还检查真实写槽之外的 KV 未改变。精度失败保存 JSON 和 `.failure.pt`，包含三份独立输出、metadata 和逻辑 KV 参考。不要删除断言或放宽精度。update/replay 时间包含同步，是诊断数据，不是整模型性能数据。inplace 的 UPDATE PASS 仅表示无 task update 调用的分支返回。

## 模型侧启用

只有探针在目标设备上通过后，再将以下选项合并进原启动命令：

```bash
--no-async-scheduling \
--max-num-seqs 20 \
--compilation-config '{"cudagraph_mode":"FULL","cudagraph_capture_sizes":[20,80,192],"max_cudagraph_capture_size":192}' \
--additional-config '{"token_graph_310p":{"enabled":true,"request_layout":"token","update_mode":"inplace"}}'
```

已有 additional-config 时合并此项，不覆盖其它配置。功能默认关闭；启用后默认 token/inplace。旧 `fixed/inplace` 模型配置现在明确报错，避免继续运行已失败的 host qLens 更新路径。`active/inplace` 也不作为模型支持组合；旧 fixed/active 布局保留在探针用于对照。不要同时启用 enforce-eager。超出最大桶的 batch 使用 eager，不自动捕获新桶。

范围：单卡 dense causal decoder、单一普通 FullAttentionSpec KV 组、非量化 KV，无滑窗/ALiBi/sinks/KV sharing/LoRA/KV transfer/ENPU。投机验证仅 ngram，独立 draft 模型明确拒绝。Qwen3-ASR 音频编码器仍使用原路径，改动针对 decoder。

## 内存与设备验收边界

- 图键只按 token 桶，不包含真实请求数或 qLens。算子行数 T 可以大于 max_num_seqs，后者仍限制真实请求数。
- 一个共享 arena 按最大 T 分配 metadata，各桶使用固定前缀视图。block table 空间约为 `max_T * max_blocks * 4` 字节；每轮复制真实请求表到对应 token 行。
- host qLens 初始化为全 1，token 布局不再改写它。device context/block table/slot mapping 原地更新，回放前保守同步；attention 仍在每层只发起一个算子调用。
- task 记录仍保留每层每桶 query/output 强引用，图激活显存不保证只按最大桶增长。弱引用和更细同步需额外设备验证。
- splitfuse_v2 Python 接口不提供显式 workspace 管理。当前依赖 ATB，不能声称已证明长度变化下的全范围 workspace 上界。
- 新 token 布局目前只有 CPU 语义和接入边界验证；旧输入/context/block table 的设备结果是设计依据，不能替代新布局的组合、padding、多桶、整模型精度和性能验收。

## CPU 检查与历史记录

本地 24 项 CPU/调用边界测试、修改文件的 Ruff 和 Python 编译检查通过。未在本会话运行 NPU；完整 format.sh 检查依赖本地尚未安装的 pre-commit。

```bash
python tests/ut/_310p/test_token_graph_standalone.py
python tests/ut/_310p/test_token_graph_wiring.py
```

第二个测试需要相邻目录 `../vllm` 的 v0.23.0 源码。CPU 测试包含逐 token 与多 query 因果注意力等价比较、全部示例、block table 展开、padding、地址稳定，以及实际 runner 方法 AST / dispatcher 边界测试；不等于完整引擎或 NPU 测试。

此前的 allocator、dense、inputs/contexts/blocks 和八请求对照记录见 [历史诊断](TOKEN_GRAPH_DIAGNOSTICS.md)。
