# ASR 服务启动时开启 / 关闭 MTP

本仓没有服务器的 `qwen3_asr_async.py` 和启动脚本。下面列出服务项目必须完成的接线；只更新 worker 补丁或 helper 文件，不能代替启动参数传递。开关在服务启动时生效，变更后按原方式重启。

## 1. 同步服务模块和启动 parser

保留已有 `streaming_prespec.py`、`streaming_prespec_formal_args.py`，更新 `streaming_prespec_helpers.py`，新增复制本仓的 `streaming_mtp_config.py` 到 `qwen_asr/inference/`。引擎仍需安装 [在线 MTP 补丁](asr_mtp_engine.md)。

在现有服务 parser 的 `parse_args()` 前加入一次：

```python
from qwen_asr.inference.streaming_mtp_config import add_mtp_args

add_mtp_args(parser)
```

得到四个参数：`args.mtp_enabled` 默认 `False`；`args.prespec_mtp_enabled` 默认 `True`；`args.mtp_checkpoint` 和 `args.mtp_graph_mode` 默认 `None`。后者继承引擎字典的图设置（缺省 off），可用 `--mtp-graph-mode full/off` 显式覆盖。使用 `--no-mtp`、`--no-prespec-mtp` 关闭，不能写 `--mtp false`。如果入口已有这些参数，应替换旧定义，避免重复注册。

在现有工厂调用中追加，原有参数保留：

```python
model = Qwen3ASRModelAsync.AsyncLLM(
    **base_config,
    # 保留现有 prespec_mode、prespec_launch_sec 等关键字。
    mtp_enabled=args.mtp_enabled,
    prespec_mtp_enabled=args.prespec_mtp_enabled,
    mtp_checkpoint=args.mtp_checkpoint,
    mtp_graph_mode=args.mtp_graph_mode,
)
```

这三个关键字不能同时存在于 `base_config` 中；不要把 `vars(args)` 整体传给引擎。

## 2. AsyncLLM 显式接收参数，在创建引擎前处理配置

在 `Qwen3ASRModelAsync.AsyncLLM()` 的现有签名中、`**kwargs` 之前增加：

```python
mtp_enabled: bool = False,
prespec_mtp_enabled: bool = True,
mtp_checkpoint: str | None = None,
mtp_graph_mode: str | None = None,
```

在原来的 `AsyncEngineArgs(...)` **之前**加入：

```python
from qwen_asr.inference.streaming_mtp_config import mtp_engine_config

if mtp_enabled and not rollback_first_verify:
    raise ValueError("online MTP requires rollback_first_verify=True")
kwargs = mtp_engine_config(
    kwargs,
    mtp_enabled=mtp_enabled,
    mtp_checkpoint=mtp_checkpoint,
    mtp_graph_mode=mtp_graph_mode,
)
```

这里 `kwargs` 是原来传给 `AsyncEngineArgs(**kwargs)` 的引擎配置字典；如果实际局部变量叫 `base_config`，在这段代码中使用该变量名。如果 `speculative_config` 等原本是单独传入引擎的显式参数，应先纳入这个配置字典，不能只处理剩余的 `**kwargs`。`rollback_first_verify` 沿用现有工厂显式接收的回滚开关。不要在引擎创建后才改字典。

- 总开关 OFF：移除 `additional_config["asr_mtp"]`，保留其他 additional 配置和原有外部草稿 speculative_config；不加载辅助 heads / arena。即使字典里残留旧 checkpoint，OFF 也优先。
- 总开关 ON：保留已有 `asr_mtp` 中的 head / cache 参数，CLI checkpoint 可覆盖原路径；未提供 CLI 路径时复用字典中的 checkpoint。没有 checkpoint 或外部草稿 `custom_class` 配置会在启动时报错。
- ON 时缺省补 `async_scheduling=False`、`enable_prefix_caching=False`；如果显式配置了不兼容值，则报错，避免静默改变实验配置。其余 TP / PP / 量化限制仍由 worker 加载校验。

主模型的旧式文件指纹默认不校验；如需严格校验，在 `additional_config["asr_mtp"]` 中设置 [verify_base_signature=True](asr_mtp_engine.md#可选主模型指纹校验)。`mtp_engine_config()` 保留该字段；它不作为新的 `AsyncLLM()` 关键字传入。默认关闭时，架构与辅助头参数形状检查仍保留。

引擎的外部草稿配置保持为：

```python
base_config["speculative_config"] = {
    "method": "custom_class",
    "model": "vllm_ascend.spec_decode.external_draft_proposer.AscendExternalDraftProposer",
    "num_speculative_tokens": 15,
}
```

`mtp_graph_mode` 只在这条启动链路传给 `mtp_engine_config()`，由它写入 `additional_config["asr_mtp"]["graph_mode"]`；不要继续作为 `AsyncEngineArgs`、请求 SamplingParams 或 `enable_prespec()` 的关键字传入。若不便修改服务器工厂签名，可仅在现有引擎字典中设置 `graph_mode="full"`。只增加 parser 参数而未完成上述工厂转发，不会开启辅助图。

## 3. 实例初始化：包括 prespec-mode off

更新后的 `PreSpecModelHelpers` 已提供可运行的 `enable_prespec()`。如果 `Qwen3ASRModelAsync` 自己还定义了旧版同名方法，它会覆盖 mixin，必须更新该方法。最小改法是将旧方法改为转调，不再保留两份初始化实现：

```python
def enable_prespec(self, **options):
    return PreSpecModelHelpers.enable_prespec(self, **options)
```

然后在 `AsyncLLM()` 创建 `model` 后、返回前的 **原有那一次** `model.enable_prespec(...)` 调用中，保留所有已有参数，追加：

```python
    mtp_enabled=mtp_enabled,
    prespec_mtp_enabled=prespec_mtp_enabled,
    producer=model._logged_prespec_producer,
```

没有日志 wrapper 时省略 `producer`，默认使用 `_run_prespec`。`enabled=(prespec_mode != "off")` 保持原值；**这次初始化调用不要包在 `if prespec_mode != "off"` 内**，正式请求在预投机关闭时仍需要 MTP 开关。

初始化会校验两个布尔值，并在任何 `enabled=False` 返回之前写入：

```python
self.mtp_enabled = mtp_enabled
self.prespec_mtp_enabled = mtp_enabled and prespec_mtp_enabled
```

日志 `PRESPEC_MTP_CONFIG` 记录保存的实际值。它表示服务开关；开启时还必须看到 worker 的 `ASR_MTP_CONFIG`，后者才说明辅助头已加载。不要在方法末尾或其他构造路径中再次写死 `self.prespec_mtp_enabled=True`。

## 4. 正式 chunk 和 finish 都传总开关

两处现有 `formal_external_draft_extra_args(...)` 调用中，保留其他关键字，追加 / 替换：

```python
    rollback_first_verify=self.rollback_first_verify,
    mtp_enabled=self.mtp_enabled,
```

不能只改 chunk。`False` 会显式写入 `ascend_mtp=False`，避免 worker 按 first-verify 标记自动启用；`True` 也覆盖没有初始草稿的正式请求。不要在 `_pack_stream_request()` 全局加这个字段：后台 `_build_stream_request()` 使用的是 `self.prespec_mtp_enabled`，必须能够独立关闭。

## 5. 启动对照

沿用原启动命令、负载和配置，分别追加：

```bash
# 所有请求关闭 MTP；回滚 / prespec 原有投机仍保留。
--no-mtp

# 正式 chunk / finish 开启，后台 prespec 关闭。
--mtp --mtp-checkpoint /path/to/best.pt --no-prespec-mtp

# 正式请求与后台 prespec 都开启。
--mtp --mtp-checkpoint /path/to/best.pt --prespec-mtp
```

ON 组须同时保持现有 `--rollback-first-verify`。`--prespec-mode off/shadow/consume` 仍单独控制候选生成 / 领取，不是 MTP 总开关。`--no-mtp --prespec-mtp` 的有效结果仍是全部 MTP 关闭。OFF 组不需要 checkpoint；从加载了 MTP 的服务切到 OFF 须重启，不能靠修改实例属性回收已经加载的权重。

建议启动时先核对：

```bash
rg -n 'PRESPEC_MTP_CONFIG|ASR_MTP_CONFIG' "$LOG"
rg -n 'mtp_enabled|prespec_mtp_enabled|mtp_engine_config' qwen_asr/inference/qwen3_asr_async.py
```

总开关 OFF 应记录两项 `False` 且本次启动不出现 `ASR_MTP_CONFIG`；ON + 后台 OFF 为 `True/False`；两者 ON 为 `True/True`。本仓测试验证配置、请求标记和 CPU worker 行为，不能证明服务器已接线或 910B 性能收益。
