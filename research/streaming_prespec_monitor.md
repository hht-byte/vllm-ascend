# 流式预投机快速监测

`streaming_prespec_monitor.py` 是旁路脚本，不需要修改 ASR 服务。每个并发档位启动一次 `sample`，让它和现有 `sustained_test` 同时运行；现有 `log_analyzer.py` 继续负责解析服务日志。监控可以提前启动，但 `--duration` 要覆盖启动等待、完整测试和收尾；汇总时用实际测试窗口排除测试前的空载数据。

## 1. 在 910B 服务主机采样

```bash
python streaming_prespec_monitor.py sample \
  --run-id baseline-c5 \
  --metrics-url http://127.0.0.1:8000/metrics \
  --npu-cmd-json '["npu-smi","info"]' \
  --duration 135 --interval 1 \
  --out baseline-c5-samples.jsonl
```

脚本按 `--interval` 尝试轮询 vLLM `running`/`waiting` 请求数、KV 占用以及原始 NPU 命令输出；若命令耗时超过间隔，实际采样会更稀疏。`time_utc` 是轮询开始时间，`metrics_time_utc` 和 `npu_time_utc` 分别是读取完成时间，后两者用于对齐。先查看一条 `npu.raw`，根据**该机器实际输出**增加 `--npu-util-regex`，捕获组必须命名为 `util`。例如输出为 `AICore Usage(%): 42` 时可用：

```bash
--npu-util-regex 'AICore Usage\(%\):\s*(?P<util>\d+(?:\.\d+)?)'
```

解析不到时 `npu.status=unparsed`、`utilization_pct=null`，不会记成 0。若服务未开放 `/metrics`，可省略 `--metrics-url`，但无法得到 vLLM 队列采样。一个 `/metrics` 端点若只覆盖部分 engine，需分别采集其它 engine；脚本会对同一响应中的 engine 求和。
多卡机器应把 `--npu-cmd-json` 改成只查询目标设备的命令（例如含 `-i 0`），不要用一个正则从全卡表格里抓到任意一张卡。

在 `run_single_concurrency` 内启动采样进程，结束 `sustained_test` 时停止它，再执行原有冷却及日志切片。保持 `run_id`、并发档位和日志切片文件名一一对应。可以先手工在同一档位运行脚本确认字段，再接入主入口。

## 2. 复用现有日志分析器的逐 commit 输出

让现有分析器每个正式 commit 导出一行 JSONL 或 CSV；要和监控数据对齐，必须有带日期和时区的 `received_at`。最小字段示例：

```json
{"ws_id":"ws-1234abcd","seg":3,"received_at":"2026-10-19T03:36:51.500+00:00","done_at":"2026-10-19T03:36:52.012+00:00","commit_latency_ms":512.4,"queue_wait_ms":21.0,"queue_size":1,"is_empty_commit":false}
```

也可用 `received_at`、`consumed_at`、`done_at` 三个带日期和时区的 ISO-8601 时间戳代替两个毫秒字段。服务日志的 `atime` 只有 `HH:MM:SS.mmm`，必须先由现有分析器补上日期与时区；跨午夜要正确递增日期。不要把 `inference submit` 当作 commit 后 finish 推理起点。空 commit 用 `is_empty_commit=true` 标记。若将伪请求期间及其后的正式 commit 区分为 `phase=current`/`phase=downstream`，汇总会分别计算两组 P95；未标注时只报告全部正式 commit。

从测试脚本记录实际 `sustained_test` 开始、结束的 UTC 时间。若测试脚本与服务不在同一台机器，先校验时钟同步；服务端时间标记更可靠。监控和日志必须属于同一 `run_id`/并发档位。

```bash
python streaming_prespec_monitor.py summarize \
  --run-id baseline-c5 --concurrency 5 \
  --samples baseline-c5-samples.jsonl \
  --commits baseline-c5-commits.jsonl \
  --test-start-utc 2026-10-19T03:36:50Z \
  --test-end-utc 2026-10-19T03:38:50Z \
  --aligned-out baseline-c5-aligned.jsonl \
  --out baseline-c5-summary.json
```

窗口是 `[test-start, test-end)`，按 commit 的 `received_at` 归属；窗口内收到而结束后完成的 commit 仍计入。`--aligned-out` 为每条正式 commit 找与 `received_at` 最近的 metrics/NPU 读数，默认最大时间差 3 秒，输出带符号的 `offset_ms`（正数表示读数在 commit 到达后）；超限时匹配为 `null`。它只方便检查时间邻近关系，不代表读数就是 commit 当时的设备状态。旧样本没有分项时间戳时退回使用轮询开始的 `time_utc`，对齐精度较低。`--commits` 可省略，但此时只有队列和利用率画像，无法判断正式提交时延是否被伪请求影响。`summary.json` 区分空 commit、服务队列等待、vLLM 等待请求数，并报告 NPU 缺样比例。

## 3. 同负载对照

保持相同音频池、随机种子、并发档位、连接补位策略、持续时间和服务配置，分别运行 target-only 与伪请求版本：

```bash
python streaming_prespec_monitor.py compare \
  --baseline baseline-c5-summary.json \
  --probe prespec-c5-summary.json \
  --out c5-comparison.json
```

先看正式 commit P95、服务队列等待 P95，以及 `current`/`downstream` 分组是否上升；同时看采样到的 vLLM waiting 和 NPU 利用率。低利用率或利用率上升本身都不能证明可投机。该脚本只输出描述性差值，不自动宣称“无时延影响”；真机结论还需要重复运行、实际到达负载核对及预先设定的可接受退化阈值。
