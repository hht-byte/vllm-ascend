# MTP 接受率与 head 数消融

## 开启统计

在原有三参数 `mtp_engine_config` 工厂调用之前合并，不覆盖已有 checkpoint、
图桶或缓存设置。默认关闭；开启后不添加 target forward、NPU 同步或设备计时。

```python
additional = dict(base_config.get("additional_config") or {})
settings = dict(additional.get("asr_mtp") or {})
settings.update({"num_heads": 3, "acceptance_stats": True})
additional["asr_mtp"] = settings
base_config["additional_config"] = additional
base_config = mtp_engine_config(
    base_config, mtp_enabled=mtp_enabled, mtp_checkpoint=mtp_checkpoint,
)
```

保持 `speculative_config.num_speculative_tokens=15`，避免同时改变外部草稿预算。
`ASR_MTP_CONFIG ... acceptance_stats=True` 确认接线；统计开关在启动时生效。
本工作不自动重启服务或推送。

## 日志与统计口径

INFO `ASR_MTP_ACCEPTANCE {json}` 在请求结束清理时输出一次汇总。
EOS / token 预算结束在当前 worker 调用内完成统计；取消请求在引擎通知 worker
清理时输出。纯回滚、Prespec 初始草稿不会计入 MTP。没有 MTP 提案的请求不输出。

worker 使用上一轮实际 MTP 提案、目标输入位置、scheduler 的实际草稿前缀和
现有同步 CPU sampler 输出核对来源。实际接受数遵循 vLLM 同步 verifier 的
`len(sampled_tokens)-1`，在 scheduler 的 EOS / max_tokens 截断之前计数。
拒绝之后的补算不靠文本匹配推断。无输出的重算步骤不当作零接受验证；不匹配、
未调度、未验证和异常形状单独记录，不混入接受率分母。

| 字段 | 含义 |
| --- | --- |
| `num_heads` / `request_type` | 按实际加载 head 数及 chunk / finish / shadow / other 分组 |
| `rounds` / `drafted` / `accepted` | 已归因的验证轮数、实际验证草稿数、接受草稿数 |
| `acceptance_rate` | `sum(accepted)/sum(drafted)`，不平均每轮百分比 |
| `mean_accepted_length` | `sum(accepted)/rounds`，含零接受轮次 |
| `mean_advance_length` | `1+mean_accepted_length`，含 verifier bonus / correction，不代表最终交付文字长度 |
| `accepted_histogram` | 索引 0 到 H 对应接受长度分布 |
| `position_survival` | 第 i 位接受次数 / 第 i 位实际送验证次数；没有暴露时为 null |
| `unverified_tokens` | 已生成但未验证的 token，包括调度截短尾部及取消时的提案 |
| `unverified_proposal_fraction` | 整轮未获得可归因验证结果的提案数 / 全部提案数；部分截短见 token 指标 |
| `discard_reasons` | 来源或位置不匹配、取消等诊断计数；不匹配增多时先排查覆盖率 |
| `proposed_fresh_positions` | 所有提案的真实新增辅助 head 位置总数，包含未验证提案的补算量 |
| `fresh_positions` | 已验证提案对应的新增辅助 head 位置，不含图 padding 或全部设备计算量 |
| `phases.cold/warm` | 每个请求首次提案 / 后续提案；warm 仍可能有拒绝后的重算 |

DEBUG `ASR_MTP_VERIFY {json}` 提供逐轮明细，仅在 DEBUG 开启时序列化。
正常消融先用 INFO 汇总，避免大量逐轮日志干扰时延。

只汇总已输出的请求生命周期。服务崩溃、日志截断、仍活跃请求或尚未收到 worker
清理通知的尾部请求不会出现在报告里，不能把它们视为零开销或零接受。
压测结束应允许 worker 完成正常清理，并检查汇总覆盖率和未验证数量。
同一日志重复输入会报错；没有汇总也会报错，不给出误导性的 0% 接受率。

## 服务端命令

检查日志：

```bash
rg -n -m 10 'ASR_MTP_CONFIG|ASR_MTP_ACCEPTANCE' "$LOG"
```

补丁安装后使用 Python 标准库分析，不需要 Torch 或独立模型：

```bash
python vllm_ascend/spec_decode/asr_mtp_metrics.py \
  --run h1=server-mtp-h1.log \
  --run h2=server-mtp-h2.log \
  --run h3=server-mtp-h3.log \
  --run h5=server-mtp-h5.log \
  --out mtp-acceptance-comparison.json
```

研究仓库也可运行 `python research/asr_mtp_metrics.py`，参数相同。
JSON 中每组同时包含全局计数和 cold/warm 分项；按各组实际样本数解释差异。

## 如何选 head 数

保持音频、并发、checkpoint、图模式和草稿来源开关一致，分别跑 OFF、1、2、3、5。
按 `ASR_MTP_CONFIG heads=` 确认实际加载数，不能只看请求配置。记录实际缓存槽数、
图命中和回退，因为同一缓存 MiB 下不同 head 数可能带来不同槽数。

接受统计与已有 `ASR_FORMAL_STATS`、commit 时延报表和 profiling 并列比较：
平均接受长度、位置存活率、未验证比例、cold/warm 补算量、MTP 设备耗时、目标验证
耗时、prefill/decode 总时延、commit mean/P95/P99、queue_wait。
`profile_stages=True` 已有的 HOST_STAGE 是主机阶段及等待时间，不能当作纯 MTP
设备计算耗时，也不能把等待时间与设备耗时直接相加。正式时延对比关闭该诊断。

稳态的粗略成本模型是 `(目标验证时间(H)+MTP时间(H))/(1+平均接受长度(H))`。
它不包含首次补算、排队、批处理变化和停止截断。平均接受长度最高不一定最快；
以端到端实测决定档位，不由该分析器自动宣称最优 H。

## 安装

上一轮 native-rows 版（研究 `3ef5b0191` / 引擎 `e5dd62de4`）使用：

```bash
git apply --check examples/910b/asr_mtp_acceptance_v023.patch
git apply examples/910b/asr_mtp_acceptance_v023.patch
```

完整、累计图、token、native-ops 和 native-rows 升级已包含此功能。
根据已安装版本选择一条升级路径，不叠加。现在共有 13 个可部署模块，新增
`asr_mtp_metrics.py`；它必须和更新后的 `asr_mtp_worker.py` 一起安装。
本机 CPU 测试不等于真实 910B 性能验证。
