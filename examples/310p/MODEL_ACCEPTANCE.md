# 310P 整模型 eager / token graph 验收

设备端执行；本脚本不会连接远程服务器。父进程不初始化 NPU，按 eager、graph 顺序启动两个独立进程，第一进程退出后才加载第二个模型。两个进程使用同一组原始输入和模型参数。

## 准备模型参数与样本

```bash
cp examples/310p/accept_model_config.example.json model-config.json
cp examples/310p/accept_cases.example.json model-cases.json
```

修改 model-config.json 中的 model 路径，并合并你已经正常运行的模型配置，例如 quantization、trust_remote_code、模型长度和 multimodal 限制。这里使用 vLLM `LLM` 的 Python 参数名（下划线），不是 CLI 参数名。模板不保证适合所有模型和设备显存；优先保留已运行成功的配置。

脚本固定单卡、同步调度、seed=310；eager 关闭图，graph 使用 FULL、token/inplace 和指定 buckets。通过专用 worker_extension_cls 的具名 RPC 读取审计，不启用不安全 callable 序列化；已有其它 worker extension 的配置会明确拒绝。其它配置在两侧一致。默认 max_num_seqs=20、max_num_batched_tokens=192；后者有助于产生图覆盖内的分块，但不能保证调度器产生指定 scheduled 数组。

示例 cases 是普通文本模型的原始 prompt。不要把它当作 Qwen3-ASR 音频验收。ASR 使用自己的原始 prompt 模板和本地单声道音频：

```json
[
  {
    "id": "audio-short",
    "prompt": "替换为你当前成功使用的完整 ASR prompt（含音频占位符）",
    "audio": "/absolute/path/to/short.wav"
  },
  {
    "id": "audio-long",
    "prompt": "替换为同一模型支持的完整 ASR prompt",
    "audio": "/absolute/path/to/long.wav"
  }
]
```

音频由 soundfile 读取，保留真实采样率，不默默混合声道或改变模板。相对音频路径以 cases JSON 所在目录为基准。也可提供 `prompt_token_ids` 替代 prompt，两者只能选一个。不下载模型或测试音频。建议准备至少 10 条不同长度的真实业务样本；少于批次大小时循环复用样本，报告保留样本清单和音频 SHA256。

## 运行

```bash
unset ASCEND_LAUNCH_BLOCKING
python examples/310p/accept_token_graph.py \
  --config model-config.json --cases model-cases.json \
  --buckets 20 80 192 --batch-sizes 1 8 10 20 \
  --repeats 3 --max-tokens 32 --out model-acceptance-01
```

输出目录必须不存在，防止混用旧报告。进度起始会显示日志位置；可在另一个终端 `tail -f model-acceptance-01/eager.log` 或 graph.log。每个进程先预热每种工作负载，再测量三轮。每轮有生成一个 token 的 prefill-oriented 测试，以及最多生成 32 tokens 的 decode-oriented 测试。两侧使用 temperature=0、单条 completion、允许正常 EOS。

性能和显存门槛需要根据业务指定，脚本不擅自确定。例如设置 `--max-slowdown 1.2` 表示每种工作负载的 graph 中位延迟不得超过 eager 的 1.2 倍；`--max-peak-gib` 限制 worker 的 peak reserved GiB。未指定门槛时只报告数据，保留待人工验收状态。

## 报告与判定

- eager.json / graph.json：完整参数、版本、样本清单、每条生成 token IDs/文本/结束原因、每批耗时、worker 显存、实际调度和图 replay 记录。
- summary.json：逐条输出比较、每桶图数、测量阶段图身份稳定性、覆盖缺口、延迟中位数和 eager/graph 比值、显存与未关闭警告。
- eager.log / graph.log：完整启动、capture、运行和退出日志。

退出码：0 表示脚本内已配置的门槛及覆盖检查全部通过；1 表示进程失败、输出差异或验收门槛失败；2 表示已有正确性检查通过，但覆盖、警告或性能/显存门槛仍待补充。不能仅凭生成文本相同宣布验收通过。

正确性比较要求每条 greedy 输出的 token IDs、文本、finish_reason 完全相同。差异会判失败并保留双方结果；它不自动证明图有 bug，近似浮点计算也可能改变接近并列的 greedy 选择，需进一步定位，脚本不会放宽门槛。

图检查从 worker 的真实 ACLGraphWrapper 记录 capture/replay，要求每个配置桶恰好一个 FULL 模型图、num_reqs=None、uniform=False，且测量阶段无新 capture/图身份变化。若输出相同但全程 eager 回退，判失败。要求观察到所有桶 replay、桶 20 的 8/10 真实请求、至少一次多 query FULL replay，以及 decode 测试期间实际 DecodeOnly replay（不是仅靠测试名称）；未覆盖项返回待补充状态，不强行伪造调度。

ASR 编码器仍可能使用 eager；多 query prefill 覆盖不足时，报告会明确指出。需要选择更合适的音频/文本长度、max_num_batched_tokens 或使用额外 dense decoder 工作负载，不能把 decode-only FULL 通过写成所有 prefill 已通过。

耗时为 `LLM.generate` 整批端到端时间，包含输入处理、调度、prefill、采样和结果处理，且安装了审计钩子。生成一 token 的耗时不是纯 prefill kernel 耗时；生成 32 tokens 的耗时也不是纯 decode 耗时。EOS 导致输出长度不同会反映在 generated_tokens 中。正式吞吐/TTFT 测试还需用业务服务压测工具复验。

显存为 worker 的 allocator allocated/reserved/peak 数据，包含模型、KV cache、图及其它内存；eager 和 graph 自动分配的 KV cache 容量可能不同，因此两者差值不能直接称为图显存。没有设置性能/显存门槛时，full_acceptance 不会为 true。

检测到已知 storage_offset 警告时保留未关闭项，即使精度通过也不自动宣布警告无害。当前脚本不修改生产 runner；审计 monkeypatch 只在这两个验收子进程内生效。

## 本地可检查项

```bash
python tests/ut/_310p/test_token_graph_acceptance.py
python examples/310p/accept_token_graph.py --help
```

CPU 检查覆盖报告门槛，不代表完整引擎或设备运行已验证。运行后提供 summary.json；若子进程失败，提供对应日志的首个错误及其上下文。
