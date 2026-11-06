# ASR 服务内的外部草稿 → MTP 接力（vLLM-Ascend 0.23）

本实现运行在 **vLLM worker 内**。加载用户提供的 `asr-mtp` checkpoint 中的辅助 heads，共享已加载目标模型的 embedding 和 logits 计算；不加载第二个 ASR 主干，不使用 HF `generate()`，不重新编码音频。回滚和 Prespec 的完整草稿仍先由已有 first-verify scheduler 补丁排入首次目标执行。随后，无论部分失配还是草稿耗尽，worker 从实际确认的目标上下文进入 MTP，后续草稿继续交给原有目标模型 / rejection sampler 验证。MTP 启用的请求不再进行旧 aligned/ngram 重接。

新增代码只有三个可部署模块，加 `worker.py` 中 runner 导入完成后的一处注册：[补丁](asr_mtp_v023.patch)。不重写 `model_runner_v1.py`、目标模型、scheduler 或 paged KV 管理。安装必须在现有 [first-verify 补丁](external_draft_first_verify_v023.md)之上。

## 1. 安装到引擎

先在服务停止更新代码的窗口应用文件，再按原方式重启服务。这里不会自动连接服务器或重启进程。

```bash
# 本分支仓库根目录；根据实际位置修改路径
python research/build_asr_mtp_patch.py \
  --ascend-root /vllm-workspace/vllm-ascend \
  --out /tmp/asr-mtp.patch

cd /vllm-workspace/vllm-ascend
git apply --check /tmp/asr-mtp.patch
git apply /tmp/asr-mtp.patch
```

导出器只读引擎：核对 worker 的真实函数签名、CPU token 表更新与 post-bookkeeping 提案顺序，再生成 diff。重复安装同版本会导出空 diff。还可直接审阅本仓的 `research/asr_mtp_v023.patch` 后应用；不同 worker 版本优先重新导出并检查。文件对应关系：

| 本仓文件 | 安装位置 |
|---|---|
| `asr_mtp_heads.py` | `vllm_ascend/spec_decode/asr_mtp_heads.py` |
| `asr_mtp_worker.py` | `vllm_ascend/spec_decode/asr_mtp_worker.py` |
| `patch_external_draft_mtp.py` | `vllm_ascend/patch/worker/patch_external_draft_mtp.py` |
| 注册 import | `vllm_ascend/worker/worker.py`，紧接 `NPUModelRunner` 的导入；受 0.23 版本条件保护 |

四个 monkey patch 分别负责 drafter 选择、带 hidden states 的提案分派、finished/abort 状态清理，以及 FULL 图初始化时保留原始 token buckets。重复 import 幂等。未配置 MTP 时委托原方法。

### 已安装版本出现循环导入

若启动报 `cannot import name 'NPUModelRunner' from partially initialized module`，并且调用链经过 `patch/worker/__init__.py -> patch_external_draft_mtp.py`，原因是旧版把 MTP import 放在了会被 runner 自身提前导入的注册表。这个问题发生在加载引擎配置之前，单独加 `--no-mtp` 不能消除该导入错误。

更新导出器后，针对**服务器实际已安装目录**重新导出并按上方命令 `git apply --check` / `git apply`。升级 diff 会移除注册表中的旧 MTP import，并在 `worker.py` 原来的 runner 导入后添加：

```python
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner  # 原有行

# Install optional ASR MTP hooks only after the runner is fully imported.
if vllm_version_is("0.23.0"):
    import vllm_ascend.patch.worker.patch_external_draft_mtp  # noqa
```

`patch/worker/__init__.py` 中原有 external-draft / first-verify import 保留，**仅删除 MTP import**。手动修复时也必须同时改这两处。只把 `patch_external_draft_mtp.py` 内的类 import 移进 `install()`，但仍在导入时执行 `install()`，仍然会循环。

修复需要按原方式重启 worker；这里不会自动重启服务。新增回归使用实际 Python package 导入顺序复现旧异常，再检查新注册在 0.23 上安装、其他版本跳过；CPU 测试不等于 910B 服务已通过启动验收。

## 2. 引擎配置

服务启动开关的完整接线见 [服务 MTP 开关](streaming_mtp_service.md)：`--mtp / --no-mtp` 控制是否加载 MTP 及正式请求，`--prespec-mtp / --no-prespec-mtp` 控制后台请求。默认总开关关闭；仅配置引擎字典并不能代替服务参数传递。以下是开启时的底层配置。

在服务构造 `AsyncEngineArgs` 的 `base_config` 中合并以下配置。checkpoint 是已在服务器上的训练产物路径，不是 HF 仓库 URL；运行时不会下载权重，也不需要安装 `spec_asr`。

```python
base_config["speculative_config"] = {
    "method": "custom_class",
    "model": "vllm_ascend.spec_decode.external_draft_proposer.AscendExternalDraftProposer",
    "num_speculative_tokens": 15,  # 保持现有完整外部草稿预算
}
base_config["additional_config"] = {
    **base_config.get("additional_config", {}),
    "asr_mtp": {
        "checkpoint": "/home/work/spec-asr/runs/pt-scale-100h-mtp5/best.pt",
        "num_heads": 5,
        "max_context_tokens": 512,
        "cache_mib": 512,
        "max_requests": 64,
    },
}
base_config["async_scheduling"] = False
base_config["enable_prefix_caching"] = False
```

### FULL 图与 token bucket 归并

启用上述 MTP 扩展并使用 `compilation_config.cudagraph_mode="FULL"` 时，图初始化按用户配置的总 token buckets 建图。实现参考 `310P-graph` 的 [46fa2fb8f](https://github.com/hht-byte/vllm-ascend/commit/46fa2fb8f558845cb07748a7e440e148fcd691b1)：仅在 `_check_and_update_cudagraph_mode()` 内临时使用 query length 1，初始化结束或异常时恢复真实值。`num_speculative_tokens=15`、运行时 query length 16、实际逐请求草稿长度均保留；不修改 scheduler、采样或目标 attention 实现。

原生 Ascend FULL dispatcher 使用 mixed descriptor，并动态更新各请求的 attention metadata；不同长度的草稿仍可同批验证。此前上游按 `K+1=16` 向上取整并去重，会把用户这组 31 个 capture sizes 归并为 17 个，导致 `6→16`、`2→16` 等 padding。保留小 bucket 后，这组配置的 6-token batch 可匹配 bucket 6，2-token batch 可匹配 bucket 2。FULL 的对齐规则与每条草稿实际长度是两个不同的概念。

此初始化 hook 仅对本扩展的 MTP drafter、精确的 `FULL` 模式及全部 attention 后端均报告 `ALWAYS` 图支持时生效。后端能力不足时，requested FULL 可能降级为 decode-only；这类情况继续使用真实的 `K+1` 进行原始安全检查和 bucket 归并。关闭 MTP 或使用 `FULL_DECODE_ONLY`、`FULL_AND_PIECEWISE`、`PIECEWISE`、`NONE` 时也沿用原始配置解析。导出器会检查图初始化 API，防止把 hook 安装到不支持的 runner 上。整个修改不在每次 decode 中执行，也不增加设备同步。

用户提供的 12:37:25 聚合表中，30 次目标执行合计 1152 个有效 token、1440 个 padded token，padding 占执行 token 的 20%；所列各行均为 FULL。这支持减少 padding 的方向，不能换算为同比例时延收益，也不能用该聚合表逐条判定请求的 prefill 阶段。更多 bucket 会增加启动捕获数量和图内存，部署后须对照实际 bucket、峰值内存及完整服务时延。

这只优化**目标 decoder 图的粒度**。当前辅助 MTP heads 仍在目标采样后以 eager 执行，音频 encoder 也不会因为 decoder 设置 FULL 自动入图。MTP/encoder 图捕获仍需各自实现和设备验证。

### 支持范围与缓存

初版支持 TP=PP=PCP=DCP=1、无量化 / LoRA 的 Qwen3-ASR、同步调度和贪心 `n=1`。不支持的引擎配置在权重加载时拒绝，不能自动退成另一种投机算法。禁用 prefix caching 是因为仅有命中的目标 KV 无法恢复 MTP 所需的完整目标 hidden 历史；不重新 prefill 来弥补。chunked prefill 可以保留每次已计算的 hidden 行。其他模型或 RoPE scaling 不在此版本范围。

checkpoint 按 supplied `asr-mtp` 的 `{"metadata": ..., "mtp": ...}` 格式读取，只装载所用辅助 heads，严格校验辅助头 state_dict。**默认关闭主模型的 `base_signature` 指纹校验**；Qwen3-ASR 架构与辅助头参数形状检查保留。MTP 训练冻结主模型，运行时共享服务已加载的目标模型，所有辅助草稿仍交由目标 verifier 验证。

缓存 arena 在 `load_model` 中按字节预算预分配，纳入引擎目标 KV 定容之前的显存统计；profile 阶段执行最坏辅助 batch 来统计临时内存。`cache_mib` **不包含辅助权重和临时 attention 工作区**，实际总增量显存大于该数值。实际槽数是 `min(max_requests, floor(cache_mib / 每槽字节数))`，启动日志 `ASR_MTP_CONFIG` 给出实际值。

用户提供的是葡语训练 checkpoint；其他语言的接受率和净收益需要单独测量，不能从葡语结果外推。

移除 `additional_config["asr_mtp"]` 即恢复原有外部草稿 proposer。不要改成原生 `method="mtp"`：它不会识别这份定制 checkpoint。

### 可选主模型指纹校验

训练仓库的旧签名包含 config 原始字节、safetensors 文件名、大小、mtime，**不包含权重内容哈希或目录路径**；复制时仅 mtime 变化也可能导致相同权重的签名不同。默认 `verify_base_signature=False` 完全跳过该检查，不读取指纹文件。需要严格复现训练环境时，在现有引擎配置中设置：

```python
base_config["additional_config"]["asr_mtp"]["verify_base_signature"] = True
```

省略该字段或显式设置 `False` 都表示跳过；字符串 `"false"`、数字等非法值会在启动时报错。此配置通过现有 `additional_config` 传递，不是 `AsyncLLM()` 新形参；服务的 `mtp_engine_config()` 会保留它，CLI checkpoint 覆盖也不会删除该设置。

升级 worker 后重新导出、应用补丁并按原方式重启。`ASR_MTP_CONFIG` 日志记录 `verify_base_signature=False/True`。开启时沿用旧算法，并在不匹配错误中打印训练和服务签名 / 模型路径；该模式也只在启动时执行，不增加 decode 阶段开销。checkpoint 文件保持原样。架构、辅助头 state_dict、引擎并行 / 缓存约束独立检查，关闭指纹校验不会影响这些检查。

## 3. ASR 请求接线与 A/B

引擎开启 MTP 后，已有 `ascend_external_draft_first_verify=True` 的正式请求自动接力，两种草稿来源相同。正式 chunk/finish 使用 first-verify；更新后的 Prespec helper 可另外给 shadow 设置 `ascend_mtp=True`，保留其原有回滚验证之后再接 MTP。两条路径的首轮草稿合同不同。

要让**无外部草稿**的正式请求也在首次目标采样后进入 MTP，或明确做请求级 A/B，把本仓更新后的 `streaming_prespec_formal_args.py` 复制到服务的 `qwen_asr/inference/`，在 **chunk 和 finish 两处**现有 helper 调用都传入：

```python
request_kwargs["sampling_params"].extra_args = formal_external_draft_extra_args(
    request_kwargs["sampling_params"].extra_args,
    candidate_tokens=(candidate.formal_draft_token_ids if candidate is not None else None),
    rollback_tokens=rollback_draft_token_ids,
    rollback_full_tokens=rollback_full_token_ids,
    kmax=self.kmax,
    rejoin_policy=self.prespec_rejoin_policy,
    rollback_first_verify=True,
    prespec_first_mismatch_ngram=self.prespec_first_mismatch_ngram,
    mtp_enabled=self.mtp_enabled,
)
```

`mtp_enabled=True` 注入 `ascend_mtp=True`；`False` 注入显式 opt-out；helper 自身的兼容默认值 `None` 不注入，由 first-verify 标记决定。服务必须按 [启动接线](streaming_mtp_service.md) 显式保存并传入 `self.mtp_enabled`。此参数仅在正式 helper 中传值，**不要在 `_pack_stream_request()` 全局注入，也不要把正式 first-verify helper 用于 shadow**。OFF/ON 均保留回滚首轮验证和相同初始草稿。MTP ON 需要 `rollback_first_verify=True`，防止误把旧的 `C[1:]` 合同当成完整草稿。MTP OFF 仍使用原重接策略，因此这是两套完整后续策略的对照，不是只衡量辅助头 kernel。

### 后台 Prespec 验证回滚后接 MTP

同时升级引擎补丁和服务的 `qwen_asr/inference/streaming_prespec_helpers.py`。若服务复制了 helper 方法，应同步复制更新后的 `_build_stream_request()`，不能只增加 `ascend_mtp=True`：旧版 worker 会直接跳过 legacy 草稿。新版 worker 会先返回一次旧回滚草稿，再在下一次验证返回后转入 MTP。

通过 [完整服务接线](streaming_mtp_service.md) 将 `args.prespec_mtp_enabled` 传到 `AsyncLLM()`，再传到 `enable_prespec()`，初始化 `self.prespec_mtp_enabled = mtp_enabled and prespec_mtp_enabled`。不能只依赖请求 builder 的 `getattr(..., True)`，也不能只修改实例中的一处赋值。`--mtp --no-prespec-mtp` 仅关闭后台 MTP，保留正式 MTP；`--no-mtp` 会同时关闭两者并移除引擎 MTP 配置。

- 有旧回滚：目标先采样首 token，worker 提交原来的 `C[1:]` **一次**；接受 0 个、部分或全部之后，均从已确认前缀和修正 / bonus drive 进入 MTP。不会 first-verify `C[1:]`，不会重复拼接回滚草稿。
- 没有旧回滚：首次目标采样后直接用 MTP，直到候选结束、预算用尽或被取消。
- shadow 请求从专用 SamplingParams 克隆上清除 formal first-verify/rejoin 标记；原音频、prefix、candidate budget、CUMULATIVE 部分结果、deadline、abort/aclose 流程保持原规则。部分候选仍必须满足已有的长于旧回滚范围等领取条件，正式请求继续验证候选。
- 容量不足时仍先保留原来的旧回滚验证，后续退回普通自回归；生命周期结束时清理 legacy 已提交标记与辅助缓存。shadow 与 formal 共用 NPU 和缓存池，性能需按同负载分别测试正式 ON / shadow OFF 与正式 ON / shadow ON，尤其检查 deadline 命中率和正式 P95。

## 4. 执行时序与开销

设采样完成后 CPU token 表中的完整已确认序列长 N。最后一个刚采样的修正 / bonus token 尚未被目标前向执行，目标 hidden 只保留前 N−1 行；拒绝草稿的其他行丢弃。MTP 把最后采样 token 作为 drive，只返回其后辅助 token。下一次目标执行同时处理 drive 和辅助草稿，**不增加修正 token 的单独目标前向**。

每个 head 的 shifted KV / hidden 独立缓存，深度 d 最多保留 `confirmed_input_length - d` 个锚点，避免保留依赖被拒绝 token 的状态。请求按 query 长度的 2 次幂分桶，避免新 prefill 与增量 decode 混批后的大量 padding；辅助 token 在设备上逐 head 生成，整次提案只有一次 batch `tolist()`。CPU 上已有的 token IDs 不算额外 GPU 同步。目标 KV 归引擎所有，辅助缓存不读取其物理布局。

采样块中 EOS 后即使附有 bonus，也会跳过这条请求的 MTP；同时遵守请求的 `ignore_eos`、显式 stop token 和 `min_tokens`，避免在即将停止的请求上增加辅助计算。

达到 context cap、缓存槽耗尽或请求采样设置不支持时，该请求后续返回空草稿，正常目标解码；不重算全历史、不重新排队等待缓存，也不重新执行音频编码。`ASR_MTP_FALLBACK` DEBUG 日志和 worker 计数记录原因。finished / abort 清理状态；暂时未调度、batch 重排不释放仍存活请求的缓存。重计算覆盖旧行时裁剪辅助状态；缺少 hidden 前缀则安全退回普通解码。辅助草稿受 head 数、引擎 kmax、max_tokens 剩余输出数、模型上下文余量共同约束。

## 5. 验证边界

本机检查包括：加载真实格式的小型 checkpoint、与 supplied SequentialMTP 比较批量辅助输出、初始草稿接受 0/1/全部后的真实 tiny Qwen3 目标循环、EOS/预算、缓存与清理、混合 shadow/formal 分派、真实 runner API 和补丁应用。测试不依赖独立 replay 脚本。

```bash
python -m unittest discover -s research -p 'test_asr_mtp_engine.py' -v
python -m unittest discover -s research -p 'test_patch_external_draft_mtp.py' -v
python -m unittest discover -s research -p 'test_build_asr_mtp_patch.py' -v
python -m unittest discover -s research -p 'test_streaming_prespec_formal_args.py' -v
python -m unittest discover -s research -p 'test_prespec_mtp.py' -v
python -m unittest discover -s research -p 'test_streaming_mtp_config.py' -v
```

前一组需要 Torch、transformers 4.57.6 及 `SPEC_ASR_TEST_ROOT` 对应的参考源码（当前测试默认使用 `.tmp/asr-mtp-reference`）；这些只用于 CPU 核对，不是线上模块的运行依赖。完整 910B checkpoint 的 BF16/NPU 数值和性能尚未实测。部署后先核对 `ASR_MTP_CONFIG`、正式 `INITIAL_DRAFT` 和后续 `ASR_VERIFY_STEP`，以相同音频 / 并发对照完整目标输出、decode 轮次、commit P50/P95/P99、队列等待和显存；辅助输出匹配或减少 head 计算位置不等于已经提高在线性能。单独 NPU 辅助 kernel 的数值差异可能改变接受率，所有输出仍由目标 verifier 决定。
