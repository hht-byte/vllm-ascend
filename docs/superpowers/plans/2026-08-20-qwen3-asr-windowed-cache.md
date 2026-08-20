# Qwen3-ASR Windowed Cache Reuse Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone helper for vLLM/vllm-ascend 0.23.0 that turns cumulative Qwen3-ASR audio updates into cache-stable independent `AsyncLLMEngine` requests.

**Architecture:** Buffer PCM on a sample timeline, promote encoder-aligned immutable audio segments, keep a bounded mutable suffix, and construct one logical Qwen3-ASR audio span with stable multimodal UUIDs. An injected async engine adapter executes each ready snapshot without importing or modifying vLLM internals.

**Tech Stack:** Python 3.10+, NumPy, asyncio, pytest

**Spec:** `docs/superpowers/specs/2026-08-20-qwen3-asr-windowed-cache-design.md`

## Global Constraints

- Do not modify vLLM, the Qwen3-ASR model, the scheduler, or the KV block manager.
- Do not add dependencies or environment variables.
- Each generate call uses a unique request ID and the same long-lived engine instance.
- A multimodal UUID must identify immutable audio content; mutable content gets a new UUID.
- Preserve one audio BOS/EOS pair by expanding the inner `<|audio_pad|>` item placeholder into adjacent item placeholders.
- Real-weight accuracy validation is required before production sign-off.

---

### Task 1: PCM Timeline and Window Promotion

**Files:**

- Create: `examples/qwen3_asr_windowed_streaming/__init__.py`
- Create: `examples/qwen3_asr_windowed_streaming/window.py`
- Create: `tests/ut/examples/test_qwen3_asr_window.py`

**Interfaces:**

- Produces: `AudioSegment`, `WindowedAudioSnapshot`, and `WindowedAudioState`.
- `WindowedAudioState.push(audio: np.ndarray) -> list[WindowedAudioSnapshot]`
- `WindowedAudioState.flush() -> WindowedAudioSnapshot | None`

- [x] **Step 1: Write failing tests for int16 normalization, two-second emissions, eight-second active windows, four-second promotion, and content-stable IDs.**

- [x] **Step 2: Run `pytest -q tests/ut/examples/test_qwen3_asr_window.py` and confirm failure because the module does not exist.**

- [x] **Step 3: Implement the chunk buffer, immutable segment dataclasses, validation, promotion, snapshot emission, and tail flush.**

- [x] **Step 4: Run `pytest -q tests/ut/examples/test_qwen3_asr_window.py` and confirm all cases pass.**

### Task 2: Qwen3-ASR Prompt Construction

**Files:**

- Create: `examples/qwen3_asr_windowed_streaming/prompt.py`
- Create: `tests/ut/examples/test_qwen3_asr_prompt.py`
- Modify: `examples/qwen3_asr_windowed_streaming/__init__.py`

**Interfaces:**

- Consumes: `WindowedAudioSnapshot` from Task 1.
- Produces: `Qwen3ASRPromptBuilder.build(snapshot, committed_text="") -> dict[str, object]`.

- [x] **Step 1: Write failing tests proving one placeholder expands to adjacent items, stable items precede the active item, UUIDs match data order, and invalid templates fail.**

- [x] **Step 2: Run `pytest -q tests/ut/examples/test_qwen3_asr_prompt.py` and confirm failure because the builder does not exist.**

- [x] **Step 3: Implement prompt validation and construction using plain dictionaries accepted by vLLM prompt preprocessing.**

- [x] **Step 4: Run `pytest -q tests/ut/examples/test_qwen3_asr_prompt.py` and confirm all cases pass.**

### Task 3: AsyncLLMEngine Adapter

**Files:**

- Create: `examples/qwen3_asr_windowed_streaming/async_engine.py`
- Create: `tests/ut/examples/test_qwen3_asr_async_engine.py`
- Modify: `examples/qwen3_asr_windowed_streaming/__init__.py`

**Interfaces:**

- Consumes: `WindowedAudioState`, `Qwen3ASRPromptBuilder`, an engine exposing `generate(prompt, sampling_params, request_id)`, and an optional output-to-text callback.
- Produces: `WindowedAsyncLLMEngineAdapter.push(audio) -> list[WindowedInferenceResult]` and `flush() -> WindowedInferenceResult | None`.

- [x] **Step 1: Write failing async tests for unique request IDs, final-output selection, committed-text propagation, and per-stream serialization.**

- [x] **Step 2: Run `pytest -q tests/ut/examples/test_qwen3_asr_async_engine.py` and confirm failure because the adapter does not exist.**

- [x] **Step 3: Implement the injected engine protocol, result dataclass, lock, request execution, text callback, push, and flush.**

- [x] **Step 4: Run `pytest -q tests/ut/examples/test_qwen3_asr_async_engine.py` and confirm all cases pass.**

### Task 4: Usage Documentation and Verification

**Files:**

- Create: `examples/qwen3_asr_windowed_streaming/README.md`
- Create: `examples/qwen3_asr_windowed_streaming/rollback.py`
- Create: `tests/ut/examples/test_qwen3_asr_rollback.py`
- Modify: `docs/superpowers/plans/2026-08-20-qwen3-asr-windowed-cache.md`

**Interfaces:**

- `Qwen3ASRRollbackState` reproduces official warmup and unstable-token rollback without depending on the synchronous Qwen-ASR wrapper.
- Documents how to instantiate `WindowedAudioState`, build the official raw prompt, enable prefix caching in `AsyncEngineArgs`, integrate output parsing, and perform real-weight accuracy qualification.

- [x] **Step 0: Add and test the tokenizer-token rollback callback used by the official streaming prefix policy.**

- [x] **Step 1: Write the Chinese usage guide with an `AsyncLLMEngine` integration example and explicit cache/accuracy limitations.**

- [x] **Step 2: Run all four focused unit-test files together.**

- [x] **Step 3: Run Ruff on the new Python files and fix any findings.**

- [x] **Step 4: Run the repository formatting check for all changed file types.**

- [x] **Step 5: Inspect `git diff --check`, `git status --short`, and the complete diff.**

- [x] **Step 6: Create one signed Conventional Commit containing the implementation, tests, design, plan, and documentation.**
