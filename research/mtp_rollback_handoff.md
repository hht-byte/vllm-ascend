# 回滚首次验证后接力 spec-asr MTP：参考验证

此实验在 `codex/streaming-prespec-monitor` 分支中验证用户指定的顺序：正式 chunk 若有回滚草稿，先用**当前 chunk 的完整累计音频**验证旧输出的回滚尾部；无论草稿全部接受还是中途失配，目标模型确认的输出（包括失配位置的修正 token）形成新的上下文；此后 `spec_asr.mtp.SequentialMTP` 从这个上下文提出草稿，仍由目标模型逐轮验证。前两个未固定的 chunk 没有可复用回滚草稿，直接进入 MTP。greedy 与 MTP 两条流分别维护自己的历史，逐 chunk 和完整 utterance 比较 token IDs 与 CER/WER。

实现位于 `mtp_rollback_handoff.py`、`mtp_streaming_replay.py`、`mtp_rollback_torch.py` 和 `evaluate_mtp_rollback_handoff.py`。纯 Python 接力层拒绝预算越界或返回未经验证 token 的 backend。Torch backend 在首次回滚验证失配时裁剪错误 token 的 KV，然后将目标修正 token 重新解码；MTP 只读取已确认的 hidden states 和 token IDs。`target_calls` 包含一次 prefill、每个 verifier forward，以及失配后修正 token 的额外 forward。

## 在 910B 模型环境运行

需要已有的 Qwen3-ASR-1.7B backbone、spec-asr 源码与训练好的 checkpoint，以及 spec-asr teacher manifest 对应的原始音频。脚本从本分支仓库根目录运行；`--spec-asr-root` 指向独立的 spec-asr 目录。不要复制 3.5 GB checkpoint 到此 Git 仓库。

```bash
python research/evaluate_mtp_rollback_handoff.py \
  --spec-asr-root /home/work/spec-asr \
  --model /models/Qwen3-ASR-1.7B \
  --checkpoint /home/work/spec-asr/runs/pt-scale-100h-mtp5/best.pt \
  --teacher-manifest /home/work/spec-asr/data/teacher/merged/test.jsonl \
  --dataset fleurs --device npu:0 \
  --rollback-tokens 5 --unfixed-chunks 2 --draft-heads 5 --kmax 15 \
  --max-utterances 8 --fail-on-mismatch \
  --output /secure-validation/mtp-handoff-fleurs.json
```

按需将 manifest 和模型路径改成目标机实际路径。先用少量 utterance 跑通，再去掉 `--max-utterances 8` 的限制并传入实际数量。脚本校验 checkpoint 的 `base_signature` 与 backbone，拒绝 MTP head 数超过 checkpoint。默认只对 `language=Portuguese` 的请求使用 MTP；其他语言仍验证回滚草稿，再走目标模型。若测试其他训练权重，显式改 `--target-language`，不要把葡语 checkpoint 的结果当作其他语言收益。

输出的 `rows[].chunks[]` 记录两条流的前缀、回滚草稿、输出 token、验证窗口和接受数；顶层汇总 `exact_match_chunks`、`chunk_count`、`rollback_accepted_tokens`、`mtp_accepted_draft_tokens` 和最终转写指标。`--fail-on-mismatch` 会先写报告再返回错误。首先检查是否每个 chunk 的两臂 token IDs 一致；若不一致，报告仍保留实际差异供排查数值漂移或接力错误。

## 与线上 vLLM 验证的边界

这条路径复用 spec-asr 的 Hugging Face 目标模型和 `SequentialMTP.propose()`，会复算移位历史。它是 correctness 和草稿接受率穿刺，**不是 vLLM-Ascend 内的优化 MTP proposer，也不能据其耗时声称线上加速**。回滚按 spec-asr teacher 的精确 token-ID 规则执行：前两块不固定，之后回滚五个 token（参数可改）。910B 服务当前 `_rollback_split()` 先把文本重新编码，再决定动态回滚边界；这可能产生与此实验不同的 token 序列。上线前必须用服务实际提交的 `rollback_draft_token_ids` 复核，不得把本报告当成服务的等价性证明。本分支已有的 `external_draft_first_verify_v023.patch` 可以在正式 vLLM 请求的首次执行验证完整回滚草稿，但当前 `AscendExternalDraftProposer` 在之后只做旧草稿重接，没有装载 spec-asr MTP 权重。现有 `formal_external_draft_extra_args()` 在选中 prespec 候选时仍让候选优先；本实验单独验证回滚先行接力，不改变那个线上策略。

下一步线上接入需要为这份 checkpoint 实现能读取目标模型已确认 hidden states、共享 embedding 和 LM head 的 vLLM-Ascend drafter，并在首次回滚验证之后交给它提案；不能直接设置 `method=mtp` 并假定 vLLM 会识别本 checkpoint。优先在现有 proposer 注册与 worker patch 扩展点实现，避免改 Qwen3-ASR 主干模型；如需修改 scheduler，只沿用本分支已有的首轮 external draft 补丁。上 NPU 前后都应逐条对齐 greedy token IDs、首轮实际有效回滚长度、后续 MTP 接受数、目标 forward 数和错误率。

本地 CPU 测试：

```bash
python -m unittest discover -s research -p 'test_mtp_rollback_handoff.py' -v
python -m unittest discover -s research -p 'test_mtp_streaming_replay.py' -v
```

## 910B 物理 0 卡实测（2026-11-04，服务器本地时间）

在已有 `vllm-23.0-hht` 容器中，将本分支验证脚本复制到 `/home/work/spec-asr/validation/mtp-handoff-1fdb0b9-20261103/`，使用 `ASCEND_RT_VISIBLE_DEVICES=0`、`--device npu:0`、`--dataset fleurs --max-utterances 8 --max-new-tokens 96 --draft-heads 5 --kmax 15 --fail-on-mismatch` 运行上面的命令。原始报告在同一目录的 `fleurs-8-npu0.json`；本地副本保存在 spec-asr 工作区的 `artifacts/mtp-handoff-fleurs-8-npu0.json`，未提交逐 token 报告。

| 指标 | 结果 |
|---|---:|
| Utterance / chunk 精确匹配 | 8/8；46/46 |
| 回滚草稿提交 / 接受 token | 150 / 121 |
| 回滚部分失配的 chunk | 18 |
| MTP 草稿提交 / 接受 token | 382 / 233 |
| Greedy / 接力目标 forward（含 prefill 和修正） | 530 / 252 |

运行前后 0 卡 HBM 均约为 52,643/65,536 MB，`vllm-23.0-hht` 服务容器持续运行。运行中一次采样为 57,735/65,536 MB。这里的 forward 数属于 Hugging Face 参考解码器，不能换算成 vLLM 在线加速。模型、checkpoint、teacher manifest 分别为 `/home/models/Qwen3-ASR-1.7B`、`/home/work/spec-asr/runs/pt-scale-100h-mtp5/best.pt`、`/home/work/spec-asr/data/teacher/pt-scale-100h/merged/test.jsonl`。
