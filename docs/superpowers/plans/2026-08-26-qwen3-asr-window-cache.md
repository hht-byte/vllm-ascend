# Qwen3-ASR 稳定窗口缓存 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不修改 vLLM 0.23.0、vLLM-Ascend 0.23.0 和 Qwen3-ASR checkpoint 的前提下，为现有 Session API 服务交付可配置 2/4/8 秒稳定窗口的音频 embedding 与 LLM KVCache 复用 Adapter，并提供本地等价性测试和 Ascend 310P 验收工具。

**Architecture:** 独立 Python 包接收现有服务保存的单条累计 PCM，将其切成 sealed 窗口与 open/final 尾窗，为每个窗口生成内容安全的稳定 UUID，并把一个原生 Qwen3-ASR 音频占位区间扩成多个连续 anchor。vLLM 原生 MultiModal Processor、Encoder Cache 和 Prefix Cache 完成张量缓存；Adapter 只保存 Session 的小型 CPU 生命周期元数据。Engine 配置辅助函数从原生 `AsyncEngineArgs` 创建 `VllmConfig`，设置 0.23.0 未由 EngineArgs 暴露的 `cache_config.hash_block_size`，然后仍用原生 `AsyncLLM.from_vllm_config` 启动。

**Tech Stack:** Python 3.12、NumPy、cbor2、pytest、ruff、mypy、vLLM 0.23.0、vLLM-Ascend 0.23.0、Ascend 310P。

**Runtime ruling (2026-08-28):** 生产实际运行 Python 3.12，因此用户裁决以 `>=3.12,<3.13` 和 py312 静态门禁覆盖原计划的 Python 3.11 假设；不再声明或机械验证 Python 3.11 兼容性。

**Identity ruling (2026-09-01):** 现有 Session API 保证每个 `session_id` 同时只有一条串行推理链，并在下一条 VAD 语句前调用 `release_session(session_id)`；因此公共 API 不再要求 `utterance_epoch`。Engine 生命周期固定模型、Processor 和音频塔，升级时重启 Engine 并清空缓存，所以窗口身份仅由 Session namespace、窗口序号和 PCM SHA-256 构成。

**Spec:** `docs/superpowers/specs/2026-08-26-qwen3-asr-window-cache-design.md`

## Global Constraints

- 不改 `.upstream/vllm`、`.upstream/vllm-ascend` 或模型 checkpoint；不使用 monkey patch、私有模型注册或自定义 Qwen3-ASR Processor。
- 不实现 Session API、VAD、2 秒块合并、推理触发、文本回滚、请求排序；这些继续由现有 Session API 服务负责。
- `accumulated_audio` 是当前 VAD 语句的一条累计完整音频，mono、16 kHz、`float32`、最长 10 秒。
- 精度基线是“相同固定窗口策略、缓存关闭时的全量重算”；greedy token ID 必须逐个一致。
- Adapter 不持有 NPU embedding 或 KV 张量；缓存淘汰和 miss 只能触发重算。
- 当前无 310P，因此本地完成单元、静态契约和 Fake Pipeline 集成测试；标为 `npu` 的测试与性能矩阵在目标环境执行。
- 严格 TDD：每个行为先写测试并确认因预期原因失败，再做最小实现并确认通过。
- 每个任务独立提交；只暂存该任务列出的文件，保留工作区中任何无关改动。

## File Map

```text
pyproject.toml                                      # 包元数据、依赖、pytest/ruff/mypy 配置
.gitignore                                          # 忽略 .upstream、构建物和验收输出
src/qwen3_asr_window_cache/__init__.py               # 稳定公共 API
src/qwen3_asr_window_cache/config.py                 # 不可变 Adapter/Engine 配置
src/qwen3_asr_window_cache/errors.py                 # 领域异常
src/qwen3_asr_window_cache/windowing.py              # PCM 校验和窗口切分
src/qwen3_asr_window_cache/identity.py               # 规范 PCM、namespace、window UUID
src/qwen3_asr_window_cache/prompt_builder.py         # 单占位区间到 N 个 anchor
src/qwen3_asr_window_cache/request_adapter.py        # Session 状态、PromptType 构造、释放
src/qwen3_asr_window_cache/compatibility.py          # 0.23.0 版本门禁
src/qwen3_asr_window_cache/engine_config.py          # VllmConfig 非侵入式配置
src/qwen3_asr_window_cache/metrics.py                # 预期复用与观测快照
tests/unit/                                          # 不依赖 vLLM/NPU 的快速测试
tests/contract/                                      # 对两个 0.23.0 上游源码树的契约测试
tests/integration/                                   # Fake cache 等价性和目标 NPU 测试
tests/fakes.py                                       # 确定性 Fake Encoder/Prefix Cache
benchmarks/benchmark_310p.py                         # cache-off/reuse 正确性与时延矩阵
benchmarks/prometheus_delta.py                       # vLLM Cache counter 差值采集
scripts/fetch_upstream.sh                            # 拉取精确 tag 到 .upstream
scripts/verify_upstream_clean.sh                     # 上游树零修改检查
docs/integration.md                                  # Session API 最小接入说明
docs/310p-validation.md                              # 310P 正确性、性能、回退验收手册
```

---

### Task 1: 建立包边界、输入校验与稳定窗口领域模型

**Files:**

- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/qwen3_asr_window_cache/__init__.py`
- Create: `src/qwen3_asr_window_cache/config.py`
- Create: `src/qwen3_asr_window_cache/errors.py`
- Create: `src/qwen3_asr_window_cache/windowing.py`
- Create: `tests/unit/test_config.py`
- Create: `tests/unit/test_windowing.py`

**Interfaces:**

```python
@dataclass(frozen=True, slots=True)
class WindowCacheConfig:
    supported_window_seconds: tuple[int, ...] = (2, 4, 8)
    sample_rate: int = 16_000
    max_audio_seconds: int = 10
    max_audio_windows: int = 5

@dataclass(frozen=True, slots=True)
class AudioWindow:
    index: int
    start_sample: int
    end_sample: int
    samples: np.ndarray
    sealed: bool

def split_audio_windows(
    audio: np.ndarray,
    *,
    window_sec: int,
    sample_rate: int,
    is_final: bool,
) -> tuple[AudioWindow, ...]: ...
```

- [ ] 创建 `pyproject.toml`，声明运行依赖 `numpy>=1.26,<3`、`cbor2>=5.6,<6`，开发依赖 `pytest>=8,<9`、`pytest-asyncio>=0.24,<1`、`ruff>=0.9,<1`、`mypy>=1.14,<2`、`build>=1.2,<2`；配置 `src` layout、`strict = true` 的 mypy，以及默认排除 `contract`、`npu` marker。

  ```toml
  [tool.pytest.ini_options]
  addopts = "-q -m 'not contract and not npu'"
  testpaths = ["tests"]
  markers = [
    "contract: requires .upstream v0.23.0 source trees",
    "npu: requires Qwen3-ASR model and Ascend 310P",
  ]
  ```

- [ ] 创建 `.gitignore`，精确包含 `.upstream/`、`.pytest_cache/`、`.mypy_cache/`、`.ruff_cache/`、`__pycache__/`、`*.egg-info/`、`dist/`、`build/`、`benchmark-results/`。

- [ ] 写 `tests/unit/test_config.py` 与 `tests/unit/test_windowing.py`，覆盖 6/8/10 秒、2/4/8 秒、open/final 状态和所有输入边界。核心断言必须包含：

  ```python
  def pcm(seconds: int) -> np.ndarray:
      return np.arange(seconds * 16_000, dtype=np.float32)

  def test_four_second_windows_for_six_seconds() -> None:
      pcm_buffer = pcm(6)
      windows = split_audio_windows(
          pcm_buffer, window_sec=4, sample_rate=16_000, is_final=False
      )
      assert [(w.start_sample, w.end_sample, w.sealed) for w in windows] == [
          (0, 64_000, True),
          (64_000, 96_000, False),
      ]
      assert np.shares_memory(windows[0].samples, pcm_buffer)

  def test_final_tail_is_sealed() -> None:
      windows = split_audio_windows(
          pcm(10), window_sec=4, sample_rate=16_000, is_final=True
      )
      assert [w.sealed for w in windows] == [True, True, True]
  ```

  另测：窗口无重叠/无遗漏、整除时无空尾窗、二维/`float64`/非连续数组、非 16 kHz、非法窗口、超过 10 秒、超过 5 个窗口分别抛出设计规格中的确定异常。

- [ ] 运行 `python -m pytest tests/unit/test_config.py tests/unit/test_windowing.py`；预期失败为 `ModuleNotFoundError: No module named 'qwen3_asr_window_cache'`，不能是测试语法错误。

- [ ] 在 `errors.py` 定义规格中的异常层次；在 `config.py` 的 `__post_init__` 校验采样率、支持窗口和最大窗口覆盖关系；在 `windowing.py` 先完整校验、再用连续 slice 产生 `AudioWindow`，不复制 PCM。

  ```python
  full_count, remainder = divmod(audio.size, window_samples)
  count = full_count + int(remainder > 0)
  for index in range(count):
      start = index * window_samples
      end = min(start + window_samples, audio.size)
      sealed = end - start == window_samples or is_final
      windows.append(AudioWindow(index, start, end, audio[start:end], sealed))
  ```

- [ ] 在 `__init__.py` 只导出当前稳定公共类型；运行相同 pytest，预期全部通过。

- [ ] 运行 `python -m ruff check .` 和 `python -m mypy src`；修复所有错误，不增加全局 ignore。

- [ ] 提交：

  ```bash
  git add pyproject.toml .gitignore src/qwen3_asr_window_cache tests/unit/test_config.py tests/unit/test_windowing.py
  git commit -m "feat: add stable audio window domain"
  ```

### Task 2: 实现 canonical PCM、Session namespace 与窗口 UUID

**Files:**

- Create: `src/qwen3_asr_window_cache/identity.py`
- Create: `tests/unit/test_identity.py`
- Modify: `src/qwen3_asr_window_cache/__init__.py`

**Interfaces:**

```python
def canonical_pcm_digest(samples: np.ndarray) -> bytes: ...
def build_session_namespace(*, session_id: str) -> str: ...
def build_window_id(
    *, namespace: str, window: AudioWindow,
) -> str: ...
```

- [ ] 写 `tests/unit/test_identity.py`，先固定以下安全性质：相同输入重试 ID 相同；open 尾窗增长后 ID 改变；sealed 前窗 ID 保持；Session、窗口序号或 PCM 任一变化都会改变 ID。

  ```python
  first = ids_for(audio_6s, session_id="u1", window_sec=4)
  second = ids_for(audio_8s, session_id="u1", window_sec=4)
  assert first[0] == second[0]
  assert first[1] != second[1]
  assert ids_for(audio_8s_changed_at_sample_3, "u1", 4)[0] != second[0]
  assert ids_for(audio_8s, "u2", 4)[0] != second[0]
  ```

- [ ] 增加 canonical PCM 测试：hash 对 C-contiguous little-endian `float32` 的 byte representation 确定；函数拒绝 dtype/维度/连续性不合规输入；`-0.0` 与 `+0.0` 不被擅自归一化；NaN payload 按原始字节参与身份。

- [ ] 运行 `python -m pytest tests/unit/test_identity.py`；预期失败为缺少 `identity` 模块或待实现符号。

- [ ] 用 `cbor2.dumps(payload, canonical=True)` 和 SHA-256 实现身份，payload 使用固定 tuple 而非无序 dict，digest 输出 64 位小写 hex；PCM digest 直接消费只读 `memoryview(samples).cast("B")`，不构造 Python float 列表。

  ```python
  def _sha256_cbor(payload: object) -> str:
      return hashlib.sha256(cbor2.dumps(payload, canonical=True)).hexdigest()

  namespace = _sha256_cbor(("qwen3-asr-session-v2", session_id))
  window_id = _sha256_cbor((
      "qwen3-asr-window-v2", namespace, window.index, pcm_digest,
  ))
  ```

- [ ] 运行 identity 与 Task 1 全部单测，预期通过；运行 ruff/mypy。

- [ ] 提交：

  ```bash
  git add src/qwen3_asr_window_cache/identity.py src/qwen3_asr_window_cache/__init__.py tests/unit/test_identity.py
  git commit -m "feat: add content-safe audio window identities"
  ```

### Task 3: 构造原生 Qwen3-ASR 多窗口 Prompt

**Files:**

- Create: `src/qwen3_asr_window_cache/prompt_builder.py`
- Create: `tests/unit/test_prompt_builder.py`
- Modify: `src/qwen3_asr_window_cache/__init__.py`

**Interfaces:**

```python
AUDIO_START = "<|audio_start|>"
AUDIO_PAD = "<|audio_pad|>"
AUDIO_END = "<|audio_end|>"
AUDIO_PLACEHOLDER = AUDIO_START + AUDIO_PAD + AUDIO_END

def build_windowed_prompt(prompt: str, *, window_count: int) -> str: ...
```

- [ ] 写测试固定“一个外层音频区间、每窗一个 anchor、其他文本逐字不变”：

  ```python
  def test_replaces_one_placeholder_with_three_anchors() -> None:
      original = "<|im_start|>user\n" + AUDIO_PLACEHOLDER + "<|im_end|>"
      result = build_windowed_prompt(original, window_count=3)
      assert result.count(AUDIO_START) == 1
      assert result.count(AUDIO_END) == 1
      assert result.count(AUDIO_PAD) == 3
      assert result == original.replace(
          AUDIO_PLACEHOLDER, AUDIO_START + AUDIO_PAD * 3 + AUDIO_END
      )
  ```

  另测：零/负窗口、占位区间缺失、重复两次、拆散/嵌套 token 均抛 `InvalidPromptPlaceholder`；不能改写 assistant 文本或现有回滚内容。

- [ ] 运行 `python -m pytest tests/unit/test_prompt_builder.py`，确认因模块/实现缺失失败。

- [ ] 最小实现：先要求 `prompt.count(AUDIO_PLACEHOLDER) == 1`，再额外要求三个 token 的全局 count 均为 1，最后执行一次精确 `replace`；不使用正则宽松匹配。

- [ ] 运行 prompt 与既有单测、ruff、mypy，预期全部通过。

- [ ] 提交：

  ```bash
  git add src/qwen3_asr_window_cache/prompt_builder.py src/qwen3_asr_window_cache/__init__.py tests/unit/test_prompt_builder.py
  git commit -m "feat: build native multi-window ASR prompts"
  ```

### Task 4: 实现有状态 Request Adapter 与幂等生命周期

**Files:**

- Create: `src/qwen3_asr_window_cache/request_adapter.py`
- Create: `tests/unit/test_request_adapter.py`
- Modify: `src/qwen3_asr_window_cache/__init__.py`

**Interfaces:**

```python
class WindowedRequestAdapter:
    def __init__(self, config: WindowCacheConfig) -> None: ...

    def build_request(
        self, *, session_id: str,
        accumulated_audio: np.ndarray, sample_rate: int,
        window_sec: int, is_final: bool, prompt: str,
    ) -> dict[str, object]: ...

    def release_session(self, session_id: str) -> None: ...
```

- [ ] 写 happy-path 测试，以同一 Session 的 6s→8s→10s、4 秒窗口连续调用；断言返回 dict 可直接作为 vLLM PromptType、数组与 UUID 一一对应、cache salt 稳定、0–4 秒 UUID 三轮相同、4–8 秒 UUID 在 8s/10s 相同、open 尾窗增长时 UUID 改变。

  ```python
  def build(samples: np.ndarray, *, is_final: bool) -> dict[str, object]:
      return adapter.build_request(
          session_id="session-a",
          accumulated_audio=samples,
          sample_rate=16_000,
          window_sec=4,
          is_final=is_final,
          prompt="<|audio_start|><|audio_pad|><|audio_end|>",
      )

  six = build(audio[:96_000], is_final=False)
  eight = build(audio[:128_000], is_final=False)
  ten = build(audio, is_final=True)
  assert six["multi_modal_uuids"]["audio"][0] == eight["multi_modal_uuids"]["audio"][0]
  assert eight["multi_modal_uuids"]["audio"][:2] == ten["multi_modal_uuids"]["audio"][:2]
  assert all(
      np.shares_memory(item, audio)
      for item in ten["multi_modal_data"]["audio"]
  )
  ```

- [ ] 写状态机测试：长度回退；同一 Session 改窗口；final 后追加；完全相同 final 幂等重试；`release_session` 幂等；release 后同 `session_id` 可作为下一条 VAD 语句进入；不同 Session 完全隔离。

- [ ] 写历史 PCM 修改测试：长度增长时若旧 sealed 区域内容改变，Adapter 不错误拒绝，但受影响窗口 UUID 必须变化，未受影响窗口保持；相同长度但不同内容的非 final 请求也必须安全失效旧 UUID。

- [ ] 运行 `python -m pytest tests/unit/test_request_adapter.py`；确认失败来自缺少 Adapter。

- [ ] 实现私有 `_SessionState(window_sec, last_sample_count, finished, final_request_digest)`；用 `session_id` 为 key。所有输入、prompt、窗口和身份先计算成功，再一次性提交状态，保证异常调用不污染状态。

- [ ] 返回准确结构；固定采样率时每个 audio item 直接传 NumPy 一维数组，不包装重采样 tuple：

  ```python
  return {
      "prompt": build_windowed_prompt(prompt, window_count=len(windows)),
      "multi_modal_data": {"audio": [window.samples for window in windows]},
      "multi_modal_uuids": {"audio": window_ids},
      "cache_salt": namespace,
  }
  ```

- [ ] 用 `threading.RLock` 保护状态表，但不生成 `inference_seq`、request ID，也不排序 AsyncLLM 输出；在 docstring 明确累计数组在 vLLM 完成输入消费前不得被调用方原地修改，现有服务应替换累计 buffer 而非 mutate 已提交 buffer。

- [ ] 跑全部 unit tests、ruff、mypy；预期全部通过，并检查 Adapter 实例 `__dict__` 中没有音频 ndarray、torch tensor 或 NPU handle。

- [ ] 提交：

  ```bash
  git add src/qwen3_asr_window_cache/request_adapter.py src/qwen3_asr_window_cache/__init__.py tests/unit/test_request_adapter.py
  git commit -m "feat: assemble session-safe vllm audio requests"
  ```

### Task 5: 增加版本门禁与非侵入式 Engine 配置入口

**Files:**

- Create: `src/qwen3_asr_window_cache/compatibility.py`
- Create: `src/qwen3_asr_window_cache/engine_config.py`
- Create: `tests/unit/test_compatibility.py`
- Create: `tests/unit/test_engine_config.py`
- Modify: `src/qwen3_asr_window_cache/errors.py`
- Modify: `src/qwen3_asr_window_cache/__init__.py`

**Interfaces:**

```python
SUPPORTED_VLLM_VERSION = "0.23.0"
SUPPORTED_VLLM_ASCEND_VERSION = "0.23.0"

def validate_runtime_versions(
    *,
    version_getter: Callable[[str], str] = importlib.metadata.version,
) -> None: ...

def prepare_vllm_config(
    engine_args: object,
    *,
    hash_block_size: int = 32,
    required_audio_items: int = 5,
    validate_versions: bool = True,
) -> object: ...
```

- [ ] 写版本测试，注入 fake `version_getter`；仅 `vllm==0.23.0` 且 `vllm-ascend==0.23.0` 通过，缺包、local/dev suffix、任一错版都抛 `UnsupportedRuntimeVersion`，错误消息包含 distribution、expected、actual。

- [ ] 写不导入 vLLM 的 duck-typed Engine 测试：fake args 的 `create_engine_config()` 返回含 `cache_config` 与 `multimodal_config` 的 namespace；断言 helper 只调用一次、设置 `hash_block_size=32`、保留模型/Ascend/其他参数，并返回同一 config 对象。

  ```python
  fake_config = SimpleNamespace(
      cache_config=SimpleNamespace(
          block_size=128, enable_prefix_caching=True, hash_block_size=None
      ),
      multimodal_config=SimpleNamespace(limit_per_prompt={"audio": 5}),
  )
  args = FakeEngineArgs(fake_config)
  result = prepare_vllm_config(args, validate_versions=False)
  assert result is fake_config
  assert result.cache_config.hash_block_size == 32
  assert args.create_count == 1
  ```

- [ ] 增加失败用例：prefix cache 未开启；物理 block 不是 128；hash block 非正数或不能整除物理 block；audio limit 少于 5；必要属性缺失。分别抛 `InvalidEngineConfiguration`，且在失败前不得改写 config。

- [ ] 运行 `python -m pytest tests/unit/test_compatibility.py tests/unit/test_engine_config.py`；确认因模块/符号缺失失败。

- [ ] 实现版本门禁和 Engine helper；`compatibility.py` 之外不得在包 import 时加载 vLLM 或 torch，使 Windows 本地单测无需安装 NPU 栈。

  ```python
  config = engine_args.create_engine_config()
  cache = config.cache_config
  if cache.block_size != 128 or not cache.enable_prefix_caching:
      raise InvalidEngineConfiguration(
          "block_size must be 128 and prefix caching must be enabled"
      )
  if cache.block_size % hash_block_size:
      raise InvalidEngineConfiguration(
          "hash_block_size must divide the physical block size"
      )
  if config.multimodal_config.limit_per_prompt.get("audio", 0) < required_audio_items:
      raise InvalidEngineConfiguration("audio item limit is smaller than required")
  cache.hash_block_size = hash_block_size
  return config
  ```

- [ ] 在测试中验证调用顺序：版本检查与全部配置检查通过后才写 `hash_block_size`；失败时仍为 `None`。运行全部 unit、ruff、mypy。

- [ ] 在 `__init__.py` 导出 `prepare_vllm_config` 和 `validate_runtime_versions`；不提供包装 `AsyncLLM.generate` 的类。

- [ ] 提交：

  ```bash
  git add src/qwen3_asr_window_cache tests/unit/test_compatibility.py tests/unit/test_engine_config.py
  git commit -m "feat: configure vllm 023 caches without source edits"
  ```

### Task 6: 固化 vLLM/vLLM-Ascend 0.23.0 源码契约与零修改证明

**Files:**

- Create: `scripts/fetch_upstream.sh`
- Create: `scripts/verify_upstream_clean.sh`
- Create: `tests/contract/conftest.py`
- Create: `tests/contract/test_vllm_023_contract.py`
- Create: `tests/contract/test_upstream_clean.py`
- Modify: `.gitignore`

**Interfaces:**

```bash
bash scripts/fetch_upstream.sh
bash scripts/verify_upstream_clean.sh
python -m pytest -m contract tests/contract
```

- [ ] 写 `tests/contract/conftest.py` 的 session fixtures：要求 `.upstream/vllm` 与 `.upstream/vllm-ascend` 存在，否则用 `pytest.fail` 给出精确拉取命令；运行 `python -m pytest -m contract tests/contract`，确认首先因源码树缺失而失败。

- [ ] 创建幂等 `scripts/fetch_upstream.sh`：仅在目录不存在时 shallow clone；存在时 fetch 精确 tag；两个仓库均 detached checkout `v0.23.0`；任何 dirty tree 立即拒绝 checkout，绝不 reset 用户改动。

  ```bash
  clone_tag() {
    repo_url="$1"
    target="$2"
    if [ ! -d "$target/.git" ]; then
      git clone --depth 1 --branch v0.23.0 "$repo_url" "$target"
    fi
    test -z "$(git -C "$target" status --porcelain --untracked-files=all)"
    git -C "$target" fetch --depth 1 origin refs/tags/v0.23.0
    git -C "$target" checkout --detach FETCH_HEAD
  }
  ```

- [ ] 运行 `bash scripts/fetch_upstream.sh`；预期两个 `git describe --tags --exact-match` 均输出 `v0.23.0`。若网络受限，按执行环境流程请求一次仅限两个 GitHub 仓库的网络批准，不改用未知镜像。

- [ ] 先写 `test_vllm_023_contract.py`，用 AST/源码文本固定本方案依赖的公开行为，而非导入 CUDA/NPU 包：

  - `qwen3_asr.py` 的支持上限允许多个 audio item，`PromptReplacement` 按 `item_idx` 替换每个 `<|audio_pad|>`；
  - model 的 `embed_multimodal` 按顺序返回多个 audio embedding，`get_mrope_input_positions` 按 offset 排序并连续递增；
  - `multimodal/inputs.py` 的 `AudioItem` 接受 ndarray，固定采样率无需 tuple 重采样；
  - Prompt 输入类型包含 `multi_modal_uuids` 和 `cache_salt`；
  - `kv_cache_utils.py` 的块哈希包含多模态 identifier/offset、parent hash 和首块 cache salt；
  - `CacheConfig` 有 `hash_block_size`，并允许物理块为其整数倍；
  - `AsyncEngineArgs.create_engine_config()` 没有把该字段传给 `CacheConfig`，同时 `AsyncLLM.from_vllm_config` 存在。

- [ ] 运行 contract 测试，先观察具体失败；只在契约断言与 0.23.0 实际命名有差异时收紧 AST helper，不能放宽成“文件包含 audio/cache 字样”。预期最终全部通过。

- [ ] 实现 `scripts/verify_upstream_clean.sh` 和 `test_upstream_clean.py`：同时检查 tracked diff、staged diff 和 untracked 文件；输出每个仓库的 HEAD。测试不得清理或修改源码树。

  ```bash
  git -C .upstream/vllm diff --exit-code
  git -C .upstream/vllm diff --cached --exit-code
  test -z "$(git -C .upstream/vllm ls-files --others --exclude-standard)"
  ```

- [ ] 运行 `bash scripts/verify_upstream_clean.sh` 和 `python -m pytest -m contract tests/contract`，预期通过；再运行默认 pytest，确认 contract 被正确排除。

- [ ] 提交脚本和契约测试，不提交 `.upstream`：

  ```bash
  git add .gitignore scripts tests/contract
  git commit -m "test: lock vllm ascend 023 cache contracts"
  ```

### Task 7: 用 Fake Encoder/Prefix Cache 证明复用轨迹与输出等价

**Files:**

- Create: `src/qwen3_asr_window_cache/metrics.py`
- Create: `tests/unit/test_metrics.py`
- Create: `tests/fakes.py`
- Create: `tests/integration/test_fake_cache_equivalence.py`
- Modify: `src/qwen3_asr_window_cache/__init__.py`

**Interfaces:**

```python
@dataclass(frozen=True, slots=True)
class ReuseExpectation:
    sealed_window_count: int
    open_window_duration_seconds: float
    reusable_prefix_tokens: int

def floor_reusable_prefix_tokens(*, dirty_token: int, hash_block_size: int) -> int: ...
```

- [ ] 写 metrics 单测固定边界：`dirty_token` 为 0/31/32/127/128/133，hash block 为 32/128；负 token、非正 block 抛 `ValueError`。

  ```python
  @pytest.mark.parametrize(
      ("dirty", "block", "expected"),
      [(0, 32, 0), (31, 32, 0), (32, 32, 32),
       (127, 32, 96), (133, 32, 128), (133, 128, 128)],
  )
  def test_floor_reusable_prefix_tokens(dirty: int, block: int, expected: int) -> None:
      assert floor_reusable_prefix_tokens(
          dirty_token=dirty, hash_block_size=block
      ) == expected
  ```

- [ ] 运行该测试确认缺实现失败；实现最小纯函数与 `ReuseExpectation`，不依赖 vLLM metrics 私有对象；运行 unit tests 通过。

- [ ] 写 `tests/fakes.py`：Fake Audio Tower 的 embedding 是 `(window_id, pcm_digest)` 的确定值；Fake Encoder Cache 按 UUID 计 hit/miss；Fake Prefix Cache 按父 hash、token IDs、窗口 UUID/offset、cache salt 和 hash block 建链；Fake LLM token 输出由最终 embedding 序列确定。

- [ ] 写 4 秒窗口的 6s→8s→10s 集成测试，严格断言 Encoder 轨迹：

  ```python
  assert traces[0].encoder == ["miss", "miss"]
  assert traces[1].encoder == ["hit", "miss"]
  assert traces[2].encoder == ["hit", "hit", "miss"]
  ```

  对每轮分别运行 cache-on 与清空 cache 的 full recompute，断言 embedding 列表和 greedy token ID 完全相同。

- [ ] 增加矩阵测试：窗口 2/4/8 秒；音频 6/8/10 秒；LRU 容量 1 导致淘汰；相同请求重试；两个 Session 同音频不共享；新 epoch 不共享；历史 sealed PCM 修改只使受影响窗口及之后 KV 前缀失效；`cache_salt` 不同禁止跨 Session KV hit。

- [ ] 运行 `python -m pytest tests/integration/test_fake_cache_equivalence.py -vv`，预期全部通过；若 token 不一致，先修正 Fake Cache 的身份/offset 组合，不能降低断言为文本相等。

- [ ] 运行全部默认测试、ruff、mypy。

- [ ] 提交：

  ```bash
  git add src/qwen3_asr_window_cache/metrics.py src/qwen3_asr_window_cache/__init__.py tests/fakes.py tests/unit/test_metrics.py tests/integration/test_fake_cache_equivalence.py
  git commit -m "test: prove window cache reuse is output-equivalent"
  ```

### Task 8: 交付 Session API 接入文档与 Ascend 310P 验收工具

**Files:**

- Create: `benchmarks/prometheus_delta.py`
- Create: `benchmarks/benchmark_310p.py`
- Create: `tests/unit/test_prometheus_delta.py`
- Create: `tests/unit/test_benchmark_manifest.py`
- Create: `tests/integration/test_310p_equivalence.py`
- Create: `docs/integration.md`
- Create: `docs/310p-validation.md`
- Modify: `pyproject.toml`

**Interfaces:**

```bash
python benchmarks/benchmark_310p.py \
  --model /models/Qwen3-ASR-1.7B \
  --manifest /data/asr-validation.jsonl \
  --window-seconds 2 4 8 \
  --concurrency 1 4 8 \
  --iterations 20 \
  --output benchmark-results/310p.jsonl
```

Manifest 每行固定 schema：

```json
{"id":"zh-001","audio_npy":"/data/zh-001.float32.npy","sample_rate":16000,"checkpoints_seconds":[6,8,10],"language":"zh","reference":"测试文本"}
```

- [ ] 先写 `test_prometheus_delta.py`：给 before/after 的 Prometheus 文本，断言准确提取 `vllm:mm_cache_queries`、`vllm:mm_cache_hits`、`vllm:prefix_cache_queries`、`vllm:prefix_cache_hits` 的 counter 差值；缺失指标输出 `null` 和明确 warning，不伪造 0。

- [ ] 先写 `test_benchmark_manifest.py`：校验 JSONL、`.npy` 是一维 C-contiguous `float32`、`sample_rate==16000`、总时长 6–10 秒、`checkpoints_seconds` 严格递增且每项位于 6 秒至总时长、ID 唯一、language/reference 非空；坏记录报告行号。

- [ ] 运行两个测试，确认因 benchmark 模块缺失失败；实现 parser 与 Prometheus delta helper。`pyproject.toml` 增加 benchmark extra：`prometheus-client>=0.21,<1`。

- [ ] 实现 `benchmark_310p.py` 的 cache-off/reuse 双模式：每种模式启动独立 Engine；对一条 manifest 音频按 `checkpoints_seconds` 产生同一 Session 的累计 slice。cache-off 设置 `enable_prefix_caching=False` 且每轮把每个 `multi_modal_uuid` 替换为含 request ID 的一次性 SHA-256；reuse 使用 Adapter 稳定 UUID 与 `prepare_vllm_config(..., hash_block_size=32)`。两种模式都保持同一固定窗口、prompt、dtype、量化和 `temperature=0`。

  ```python
  sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens)
  started = time.perf_counter_ns()
  first_token_ns: int | None = None
  async for output in engine.generate(request, sampling, request_id):
      if first_token_ns is None and output.outputs[0].token_ids:
          first_token_ns = time.perf_counter_ns()
      final = output
  record.num_cached_tokens = final.num_cached_tokens
  record.token_ids = list(final.outputs[0].token_ids)
  ```

- [ ] 每条结果记录 mode、audio/window/concurrency、request ID、sealed/open 窗口数、token IDs、文本、`num_cached_tokens`、TTFT、final latency、Prometheus counter delta、峰值 NPU 内存采集字段；用 `try/finally` shutdown Engine。脚本计算 P50/P95，但不写固定提升阈值。

- [ ] 写 `test_310p_equivalence.py` 并标 `@pytest.mark.npu`：从环境变量读取 model/manifest；对 2/4/8 秒窗口逐条比较 cache-off/reuse greedy token ID、语言和文本；缓存 reset、LRU 压力、Session release/recreate 后再次比较。无环境变量时 `pytest.skip`，设置后任何不一致必须失败并保存最小复现 JSON。

- [ ] 写 `docs/integration.md`，给出现有服务唯一需要增加的三处代码：进程启动时构造 Adapter/Engine config；每次现有触发点调用 `build_request` 后把返回 dict 原样交给 `AsyncLLM.generate`；最终结果清理时调用 `release_session`。明确不迁移 chunk 合并、VAD、回滚、`inference_seq`。

  ```python
  engine_args = AsyncEngineArgs(
      model=model_path,
      block_size=128,
      enable_prefix_caching=True,
      limit_mm_per_prompt={"audio": 5},
  )
  vllm_config = prepare_vllm_config(engine_args, hash_block_size=32)
  engine = AsyncLLM.from_vllm_config(vllm_config)
  ```

- [ ] 写 `docs/310p-validation.md`：环境版本检查、数据矩阵、多语言 CER/WER 外部评估入口、预热/测量轮数、并发 1/业务典型并发、NPU 内存采集、vLLM metrics 与 msprof 的 Audio Tower/prefill 分段方法、成功标准、故障定位顺序。

- [ ] 文档明确 32→128 回退：只有 `hash_block_size=32` 在 vLLM-Ascend/310P 初始化或正确性验收失败时才改 128；重新跑完整 token 等价性与性能矩阵；输出下降解释为命中粒度变粗，不能关闭 PCM/Session 身份校验。

- [ ] 本地运行：

  ```bash
  python -m pytest
  python -m ruff check .
  python -m mypy src benchmarks
  python -m build
  bash scripts/verify_upstream_clean.sh
  git status --short
  ```

  预期：默认测试全部通过；ruff/mypy 无错误；生成 sdist/wheel；两个上游树 clean；`git status` 只显示本任务预期文件和被忽略的构建输出。

- [ ] 在有 310P 的目标机运行：

  ```bash
  python -m pytest -m npu tests/integration/test_310p_equivalence.py -vv
  python benchmarks/benchmark_310p.py --model /models/Qwen3-ASR-1.7B --manifest /data/asr-validation.jsonl --window-seconds 2 4 8 --concurrency 1 4 8 --iterations 20 --output benchmark-results/310p.jsonl
  ```

  预期：cache-off/reuse token ID 全部一致；报告包含所有矩阵项和 cache 指标；若当前开发环境无 310P，只在交付说明中标记“待目标机执行”，不得声称已通过。

- [ ] 提交：

  ```bash
  git add pyproject.toml benchmarks tests/unit/test_prometheus_delta.py tests/unit/test_benchmark_manifest.py tests/integration/test_310p_equivalence.py docs/integration.md docs/310p-validation.md
  git commit -m "feat: add 310p validation and integration tooling"
  ```

### Task 9: 最终规格追踪、非侵入式审计与发布候选

**Files:**

- Create: `docs/spec-traceability.md`
- Modify only if verification finds a defect: files owned by Tasks 1–8

- [ ] 建立 `docs/spec-traceability.md`，逐条映射设计规格第 6–18 节到实现符号、测试名和验收命令；每行必须是“已实现/目标机待验收”之一，不使用 TBD/TODO。

- [ ] 搜索侵入式或越界实现：

  ```bash
  rg -n "monkey.?patch|register_model|register_processor|sys\.path|inference_seq|text.?rollback|VAD|chunk.?merge" src benchmarks tests docs/integration.md
  ```

  预期：生产 `src` 中没有模型/Processor 注册、路径注入、Session 调度或文本回滚实现；文档命中只用于说明职责边界。

- [ ] 搜索占位与宽松跳过：

  ```bash
  rg -n "TODO|TBD|pass$|NotImplemented|xfail|skip\(" src tests benchmarks docs
  ```

  预期：生产代码没有占位；仅 NPU 测试因显式环境变量缺失进行有理由的 skip，contract 测试缺源码时 fail 而不是 skip。

- [ ] 重新执行完整本地门禁：默认 pytest、显式 contract pytest、ruff、mypy、build、上游 clean；记录命令、时间和摘要到 traceability 文档。不得粘贴敏感路径、音频内容或 Session ID。

- [ ] 检查公共 API 类型一致性：`build_request` 返回字段名与 vLLM 0.23 PromptType 契约一致；UUID 个数与 audio item/anchor 数一致；所有异常都从包根可导入；包根 import 不加载 vLLM/torch。

- [ ] 检查 Git diff 仅包含独立包、测试、脚本和文档；两个 `.upstream` 树及 checkpoint 均无 diff；不提交 benchmark 结果或真实音频。

- [ ] 提交审计文档和任何验证修复：

  ```bash
  git add docs/spec-traceability.md
  git commit -m "docs: audit stable window cache delivery"
  ```

- [ ] 输出最终交付摘要，明确区分“本地已验证”和“310P 待执行/已执行”；若 310P 尚未执行，下一动作是按 `docs/310p-validation.md` 跑目标机矩阵，而不是推断性能收益。
