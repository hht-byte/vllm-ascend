# ASR 服务内的外部草稿 → MTP 接力（vLLM-Ascend 0.23）

本实现运行在 **vLLM worker 内**。加载用户提供的 `asr-mtp` checkpoint 中的辅助 heads，共享已加载目标模型的 embedding 和 logits 计算；不加载第二个 ASR 主干，不使用 HF `generate()`，不重新编码音频。回滚和 Prespec 的完整草稿仍先由已有 first-verify scheduler 补丁排入首次目标执行。随后，无论部分失配还是草稿耗尽，worker 从实际确认的目标上下文进入 MTP，后续草稿继续交给原有目标模型 / rejection sampler 验证。MTP 启用的请求不再进行旧 aligned/ngram 重接。

新增代码只有三个可部署模块，加 worker 注册表一行 import：[补丁](asr_mtp_v023.patch)。不重写 `model_runner_v1.py`、目标模型、scheduler 或 paged KV 管理。安装必须在现有 [first-verify 补丁](external_draft_first_verify_v023.md)之上。

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
| 注册 import | `vllm_ascend/patch/worker/__init__.py`，紧接 first-verify import |

三个 monkey patch 分别负责 drafter 选择、带 hidden states 的提案分派、finished/abort 状态清理。重复 import 幂等。未配置 MTP 时委托原方法。

## 2. 引擎配置

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

初版支持 TP=PP=PCP=DCP=1、无量化 / LoRA 的 Qwen3-ASR、同步调度和贪心 `n=1`。不支持的引擎配置在权重加载时拒绝，不能自动退成另一种投机算法。禁用 prefix caching 是因为仅有命中的目标 KV 无法恢复 MTP 所需的完整目标 hidden 历史；不重新 prefill 来弥补。chunked prefill 可以保留每次已计算的 hidden 行。其他模型或 RoPE scaling 不在此版本范围。

checkpoint 按 supplied `asr-mtp` 的 `{"metadata": ..., "mtp": ...}` 格式读取，只装载所用辅助 heads，严格校验 state_dict 和 backbone `base_signature`。签名沿用训练仓库算法，包含 config、safetensors 文件名、大小、mtime；换目录并修改 mtime 也可能不匹配，不能据此直接绕过 backbone 校验。缓存 arena 在 `load_model` 中按字节预算预分配，纳入引擎目标 KV 定容之前的显存统计；profile 阶段执行最坏辅助 batch 来统计临时内存。`cache_mib` **不包含辅助权重和临时 attention 工作区**，实际总增量显存大于该数值。实际槽数是 `min(max_requests, floor(cache_mib / 每槽字节数))`，启动日志 `ASR_MTP_CONFIG` 给出实际值。

用户提供的是葡语训练 checkpoint；其他语言的接受率和净收益需要单独测量，不能从葡语结果外推。

移除 `additional_config["asr_mtp"]` 即恢复原有外部草稿 proposer。不要改成原生 `method="mtp"`：它不会识别这份定制 checkpoint。

## 3. ASR 请求接线与 A/B

引擎开启 MTP 后，已有 `ascend_external_draft_first_verify=True` 的正式请求自动接力，两种草稿来源相同。现有 shadow 不带该标记，保持原有行为。因此，现有 first-verify 正式 chunk/finish 可以直接使用引擎扩展。

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
    mtp_enabled=True,
)
```

`mtp_enabled=True` 注入 `ascend_mtp=True`；`False` 注入显式 opt-out；默认 `None` 不注入，由 first-verify 标记决定。只在正式 helper 中传值，**不要在 `_pack_stream_request()` 全局注入，也不要在 `_run_prespec()` 注入**。A/B 可沿已有配置链保存 `self.mtp_enabled`，两处都传 `mtp_enabled=self.mtp_enabled`；OFF/ON 均保留回滚首轮验证和相同初始草稿。MTP ON 需要 `rollback_first_verify=True`，防止误把旧的 `C[1:]` 合同当成完整草稿。MTP OFF 仍使用原重接策略，因此这是两套完整后续策略的对照，不是只衡量辅助头 kernel。

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
```

前一组需要 Torch、transformers 4.57.6 及 `SPEC_ASR_TEST_ROOT` 对应的参考源码（当前测试默认使用 `.tmp/asr-mtp-reference`）；这些只用于 CPU 核对，不是线上模块的运行依赖。完整 910B checkpoint 的 BF16/NPU 数值和性能尚未实测。部署后先核对 `ASR_MTP_CONFIG`、正式 `INITIAL_DRAFT` 和后续 `ASR_VERIFY_STEP`，以相同音频 / 并发对照完整目标输出、decode 轮次、commit P50/P95/P99、队列等待和显存；辅助输出匹配或减少 head 计算位置不等于已经提高在线性能。单独 NPU 辅助 kernel 的数值差异可能改变接受率，所有输出仍由目标 verifier 决定。
