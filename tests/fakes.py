"""Deterministic CPU-only cache fakes used by integration tests."""

import hashlib
from collections import OrderedDict
from dataclasses import dataclass

import cbor2
import numpy as np

from qwen3_asr_window_cache import AUDIO_END, AUDIO_PAD, AUDIO_START

_HAND_PROMPT_PREFIX = "transcribe:"
_PREFIX_TOKEN_IDS = (11, 12, 13, 14)
_WINDOW_TOKEN_IDS = (21, 22, 23, 24)
_SUFFIX_TOKEN_IDS = (31,)


@dataclass(frozen=True, slots=True)
class FakeEmbedding:
    """Numeric audio-tower output derived only from one PCM window."""

    digest_words: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FakeRunTrace:
    """Observable cache decisions and deterministic outputs for one request."""

    encoder: list[str]
    prefix: list[str]
    embeddings: list[FakeEmbedding]
    token_ids: list[int]


class FakeAudioTower:
    """Compute stable numeric embeddings without observing cache identity."""

    def encode(self, samples: np.ndarray) -> FakeEmbedding:
        if (
            not isinstance(samples, np.ndarray)
            or samples.ndim != 1
            or samples.dtype != np.float32
            or not samples.flags.c_contiguous
            or samples.size == 0
        ):
            raise ValueError(
                "audio windows must be non-empty contiguous float32 vectors"
            )
        digest = hashlib.sha256(memoryview(samples).cast("B").toreadonly()).digest()
        return FakeEmbedding(
            digest_words=tuple(
                int.from_bytes(digest[offset : offset + 4], byteorder="big")
                for offset in range(0, len(digest), 4)
            )
        )


class FakeEncoderCache:
    """UUID-keyed LRU for deterministic fake audio-tower embeddings."""

    def __init__(self, *, capacity: int | None = None) -> None:
        if capacity is not None and (type(capacity) is not int or capacity <= 0):
            raise ValueError("capacity must be a positive integer or None")
        self._capacity = capacity
        self._entries: OrderedDict[str, FakeEmbedding] = OrderedDict()
        self._tower = FakeAudioTower()

    def encode(
        self,
        *,
        window_uuid: str,
        samples: np.ndarray,
        force_full_recompute: bool,
    ) -> tuple[FakeEmbedding, str]:
        if not isinstance(window_uuid, str) or not window_uuid:
            raise ValueError("window_uuid must be a non-empty string")
        if not force_full_recompute and window_uuid in self._entries:
            embedding = self._entries.pop(window_uuid)
            self._entries[window_uuid] = embedding
            return embedding, "hit"

        embedding = self._tower.encode(samples)
        if not force_full_recompute:
            self._entries[window_uuid] = embedding
            if self._capacity is not None and len(self._entries) > self._capacity:
                self._entries.popitem(last=False)
        return embedding, "miss"


class FakePrefixCache:
    """Parent-chained block hashes over tokens, multimodal identity, and salt."""

    def __init__(self, *, hash_block_size: int) -> None:
        if type(hash_block_size) is not int or hash_block_size <= 0:
            raise ValueError("hash_block_size must be a positive integer")
        self._hash_block_size = hash_block_size
        self._entries: set[str] = set()

    def lookup(
        self,
        request: dict[str, object],
        *,
        force_full_recompute: bool,
    ) -> list[str]:
        prompt, window_uuids, cache_salt = _prefix_inputs(request)
        token_ids, multimodal_offsets = _hand_tokenize(prompt, window_uuids)
        complete_token_count = (
            len(token_ids) // self._hash_block_size * self._hash_block_size
        )
        parent_hash: str | None = None
        trace: list[str] = []

        for start in range(0, complete_token_count, self._hash_block_size):
            stop = start + self._hash_block_size
            block_tokens = tuple(token_ids[start:stop])
            block_multimodal = tuple(
                (window_uuid, offset)
                for window_uuid, offset in multimodal_offsets
                if start <= offset < stop
            )
            block_hash = hashlib.sha256(
                cbor2.dumps(
                    (
                        "fake-prefix-block-v1",
                        parent_hash,
                        block_tokens,
                        block_multimodal,
                        cache_salt,
                    ),
                    canonical=True,
                )
            ).hexdigest()
            hit = not force_full_recompute and block_hash in self._entries
            trace.append("hit" if hit else "miss")
            if not force_full_recompute:
                self._entries.add(block_hash)
            parent_hash = block_hash

        return trace


class FakeLLM:
    """Compose fake encoder and prefix caches around real adapter requests."""

    def __init__(
        self,
        *,
        encoder_capacity: int | None = None,
        hash_block_size: int = 4,
    ) -> None:
        self._encoder_cache = FakeEncoderCache(capacity=encoder_capacity)
        self._prefix_cache = FakePrefixCache(hash_block_size=hash_block_size)

    def run(
        self,
        request: dict[str, object],
        *,
        force_full_recompute: bool = False,
    ) -> FakeRunTrace:
        audio_windows, window_uuids = _encoder_inputs(request)
        embeddings: list[FakeEmbedding] = []
        encoder_trace: list[str] = []

        for samples, window_uuid in zip(
            audio_windows,
            window_uuids,
            strict=True,
        ):
            embedding, status = self._encoder_cache.encode(
                window_uuid=window_uuid,
                samples=samples,
                force_full_recompute=force_full_recompute,
            )
            embeddings.append(embedding)
            encoder_trace.append(status)

        prefix_trace = self._prefix_cache.lookup(
            request,
            force_full_recompute=force_full_recompute,
        )
        token_ids = [
            word
            for embedding in embeddings
            for word in embedding.digest_words
        ]
        return FakeRunTrace(
            encoder=encoder_trace,
            prefix=prefix_trace,
            embeddings=embeddings,
            token_ids=token_ids,
        )


def _encoder_inputs(
    request: dict[str, object],
) -> tuple[list[np.ndarray], list[str]]:
    multimodal_data = request.get("multi_modal_data")
    multimodal_uuids = request.get("multi_modal_uuids")
    if not isinstance(multimodal_data, dict) or not isinstance(
        multimodal_uuids, dict
    ):
        raise TypeError("request must contain multimodal data and UUID mappings")
    audio_windows = multimodal_data.get("audio")
    window_uuids = multimodal_uuids.get("audio")
    if (
        not isinstance(audio_windows, list)
        or not isinstance(window_uuids, list)
        or len(audio_windows) != len(window_uuids)
        or not audio_windows
        or not all(isinstance(item, np.ndarray) for item in audio_windows)
        or not all(isinstance(item, str) and item for item in window_uuids)
    ):
        raise ValueError("audio windows and UUIDs must be non-empty aligned lists")
    return audio_windows, window_uuids


def _prefix_inputs(
    request: dict[str, object],
) -> tuple[str, list[str], str]:
    prompt = request.get("prompt")
    cache_salt = request.get("cache_salt")
    _, window_uuids = _encoder_inputs(request)
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    if not isinstance(cache_salt, str) or not cache_salt:
        raise ValueError("cache_salt must be a non-empty string")
    return prompt, window_uuids, cache_salt


def _hand_tokenize(
    prompt: str,
    window_uuids: list[str],
) -> tuple[list[int], list[tuple[str, int]]]:
    prefix = _HAND_PROMPT_PREFIX + AUDIO_START
    if not prompt.startswith(prefix) or not prompt.endswith(AUDIO_END):
        raise ValueError("fake tokenizer accepts only the hand-defined ASR prompt")
    audio_pads = prompt[len(prefix) : -len(AUDIO_END)]
    if audio_pads != AUDIO_PAD * len(window_uuids):
        raise ValueError("prompt audio anchors must align with UUIDs")

    token_ids = list(_PREFIX_TOKEN_IDS)
    multimodal_offsets: list[tuple[str, int]] = []
    for window_uuid in window_uuids:
        multimodal_offsets.append((window_uuid, len(token_ids)))
        token_ids.extend(_WINDOW_TOKEN_IDS)
    token_ids.extend(_SUFFIX_TOKEN_IDS)
    return token_ids, multimodal_offsets
