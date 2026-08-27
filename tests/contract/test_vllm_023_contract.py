import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

type FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef

QWEN3_ASR_MODEL = Path("vllm/model_executor/models/qwen3_asr.py")
MULTIMODAL_INPUTS = Path("vllm/multimodal/inputs.py")
PROMPT_INPUTS = Path("vllm/inputs/llm.py")
KV_CACHE_UTILS = Path("vllm/v1/core/kv_cache_utils.py")
CACHE_CONFIG = Path("vllm/config/cache.py")
ENGINE_ARGS = Path("vllm/engine/arg_utils.py")
ASYNC_LLM = Path("vllm/v1/engine/async_llm.py")
INPUT_PROCESSOR = Path("vllm/v1/engine/input_processor.py")
ENCODER_CACHE_MANAGER = Path("vllm/v1/core/encoder_cache_manager.py")


def _parse(source_root: Path, relative_path: Path) -> ast.Module:
    path = source_root / relative_path
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _class(module: ast.Module, name: str) -> ast.ClassDef:
    for statement in module.body:
        if isinstance(statement, ast.ClassDef) and statement.name == name:
            return statement
    raise AssertionError(f"class {name!r} was not found")


def _direct_function(container: ast.Module | ast.ClassDef, name: str) -> FunctionNode:
    for statement in container.body:
        if (
            isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
            and statement.name == name
        ):
            return statement
    raise AssertionError(f"function {name!r} was not found")


def _nested_function(container: ast.AST, name: str) -> FunctionNode:
    for node in ast.walk(container):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node is not container
            and node.name == name
        ):
            return node
    raise AssertionError(f"nested function {name!r} was not found")


def _annotated_assignment(
    container: ast.Module | ast.ClassDef, name: str
) -> ast.AnnAssign:
    for statement in container.body:
        if (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == name
        ):
            return statement
    raise AssertionError(f"annotated assignment {name!r} was not found")


def _dotted_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _calls(container: ast.AST, dotted_name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(container)
        if isinstance(node, ast.Call) and _dotted_name(node.func) == dotted_name
    ]


def _keyword(call: ast.Call, name: str) -> ast.expr:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    raise AssertionError(f"keyword {name!r} was not found")


def _assignment_values(container: ast.AST, target_name: str) -> list[ast.expr]:
    values: list[ast.expr] = []
    for node in ast.walk(container):
        if isinstance(node, ast.Assign) and node.value is not None:
            if any(
                isinstance(target, ast.Name) and target.id == target_name
                for target in node.targets
            ):
                values.append(node.value)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == target_name
            and node.value is not None
        ):
            values.append(node.value)
    return values


def _union_members(annotation: ast.expr) -> list[ast.expr]:
    assert isinstance(annotation, ast.Subscript)
    assert _dotted_name(annotation.value) == "Union"
    if isinstance(annotation.slice, ast.Tuple):
        return list(annotation.slice.elts)
    return [annotation.slice]


def _names(container: ast.AST) -> set[str]:
    return {node.id for node in ast.walk(container) if isinstance(node, ast.Name)}


def test_vllm_ascend_source_tree_is_available(
    vllm_ascend_source_root: Path,
) -> None:
    assert (vllm_ascend_source_root / "vllm_ascend").is_dir()


def test_qwen3_asr_supports_multiple_audio_items_and_per_item_replacement(
    vllm_source_root: Path,
) -> None:
    module = _parse(vllm_source_root, QWEN3_ASR_MODEL)

    processing_info = _class(module, "Qwen3ASRProcessingInfo")
    supported_limits = _direct_function(processing_info, "get_supported_mm_limits")
    returned_limits = [
        node.value
        for node in ast.walk(supported_limits)
        if isinstance(node, ast.Return) and node.value is not None
    ]
    assert len(returned_limits) == 1
    assert ast.literal_eval(returned_limits[0]) == {"audio": None}

    fallback_processor = _direct_function(processing_info, "get_hf_processor")
    fallback_audio_tokens = [
        node.value.value
        for node in ast.walk(fallback_processor)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute) and target.attr == "audio_token"
            for target in node.targets
        )
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    ]
    assert fallback_audio_tokens == ["<|audio_pad|>"]

    processor = _class(module, "Qwen3ASRMultiModalProcessor")
    prompt_updates = _direct_function(processor, "_get_prompt_updates")
    replacement = _nested_function(prompt_updates, "get_replacement_qwen2_audio")
    assert [argument.arg for argument in replacement.args.args] == ["item_idx"]
    assert any(
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "audio_output_lengths"
        and isinstance(node.slice, ast.Name)
        and node.slice.id == "item_idx"
        for node in ast.walk(replacement)
    )

    prompt_replacements = _calls(prompt_updates, "PromptReplacement")
    assert len(prompt_replacements) == 1
    prompt_replacement = prompt_replacements[0]
    assert ast.literal_eval(_keyword(prompt_replacement, "modality")) == "audio"
    assert _dotted_name(_keyword(prompt_replacement, "target")) == "audio_token"
    assert _dotted_name(_keyword(prompt_replacement, "replacement")) == replacement.name


def test_qwen3_asr_audio_token_length_formula_remains_pinned(
    vllm_source_root: Path,
) -> None:
    module = _parse(vllm_source_root, QWEN3_ASR_MODEL)
    output_lengths = _direct_function(module, "_get_feat_extract_output_lengths")
    assignments = {
        node.targets[0].id: ast.unparse(node.value)
        for node in output_lengths.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    assert assignments["input_lengths_leave"] == "input_lengths % 100"
    assert assignments["feat_lengths"] == "(input_lengths_leave - 1) // 2 + 1"
    assert assignments["output_lengths"] == (
        "((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + input_lengths // 100 * 13"
    )


def test_qwen3_asr_preserves_audio_embedding_order_and_mrope_continuity(
    vllm_source_root: Path,
) -> None:
    module = _parse(vllm_source_root, QWEN3_ASR_MODEL)
    model = _class(module, "Qwen3ASRForConditionalGeneration")

    process_audio = _direct_function(model, "_process_audio_input")
    split_calls = _calls(process_audio, "audio_features.split")
    assert len(split_calls) == 1
    assert len(split_calls[0].args) == 1
    split_arg = split_calls[0].args[0]
    assert isinstance(split_arg, ast.Call)
    assert _dotted_name(split_arg.func) == "audio_output_lengths.tolist"

    embed_multimodal = _direct_function(model, "embed_multimodal")
    modality_loops = [
        node
        for node in ast.walk(embed_multimodal)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "modality"
        and isinstance(node.iter, ast.Name)
        and node.iter.id == "mm_input_by_modality"
    ]
    assert len(modality_loops) == 1
    modality_loop = modality_loops[0]
    assert len(_calls(modality_loop, "self._process_audio_input")) == 1
    assert any(
        isinstance(node, ast.AugAssign)
        and isinstance(node.op, ast.Add)
        and isinstance(node.target, ast.Name)
        and node.target.id == "multimodal_embeddings"
        and isinstance(node.value, ast.Call)
        and _dotted_name(node.value.func) == "tuple"
        and len(node.value.args) == 1
        and isinstance(node.value.args[0], ast.Name)
        and node.value.args[0].id == "audio_embeddings"
        for node in ast.walk(modality_loop)
    )
    assert any(
        isinstance(node, ast.Return)
        and isinstance(node.value, ast.Name)
        and node.value.id == "multimodal_embeddings"
        for node in ast.walk(embed_multimodal)
    )

    mrope_positions = _direct_function(model, "get_mrope_input_positions")
    sorted_calls = _calls(mrope_positions, "sorted")
    assert len(sorted_calls) == 1
    assert len(sorted_calls[0].args) == 1
    assert _dotted_name(sorted_calls[0].args[0]) == "mm_features"
    sort_key = _keyword(sorted_calls[0], "key")
    assert isinstance(sort_key, ast.Lambda)
    assert ast.unparse(sort_key.body) == "f.mm_position.offset"

    st_idx_values = _assignment_values(mrope_positions, "st_idx")
    assert any(
        isinstance(value, ast.IfExp)
        and ast.unparse(value.body) == "llm_pos_ids_list[-1].max() + 1"
        and ast.literal_eval(value.orelse) == 0
        for value in st_idx_values
    )
    assert any(
        isinstance(value, ast.BinOp)
        and isinstance(value.op, ast.Add)
        and _names(value) == {"st_idx", "text_len"}
        for value in st_idx_values
    )
    assert any(
        isinstance(value, ast.BinOp)
        and isinstance(value.op, ast.Add)
        and _names(value) == {"offset", "audio_len"}
        for value in _assignment_values(mrope_positions, "st")
    )
    position_arange_lengths = {
        call.args[0].id
        for call in _calls(mrope_positions, "torch.arange")
        if call.args and isinstance(call.args[0], ast.Name)
    }
    assert {"text_len", "audio_len"} <= position_arange_lengths


def test_audio_item_accepts_bare_ndarray_for_fixed_sample_rate(
    vllm_source_root: Path,
) -> None:
    module = _parse(vllm_source_root, MULTIMODAL_INPUTS)

    hf_audio_item = _annotated_assignment(module, "HfAudioItem")
    assert hf_audio_item.value is not None
    hf_members = {
        _dotted_name(member) for member in _union_members(hf_audio_item.value)
    }
    assert "np.ndarray" in hf_members

    audio_item = _annotated_assignment(module, "AudioItem")
    assert audio_item.value is not None
    audio_members = _union_members(audio_item.value)
    assert any(_dotted_name(member) == "HfAudioItem" for member in audio_members)
    resampling_tuples = [
        member
        for member in audio_members
        if isinstance(member, ast.Subscript) and _dotted_name(member.value) == "tuple"
    ]
    assert len(resampling_tuples) == 1
    tuple_slice = resampling_tuples[0].slice
    assert isinstance(tuple_slice, ast.Tuple)
    assert [_dotted_name(member) for member in tuple_slice.elts] == [
        "np.ndarray",
        "float",
    ]


def test_prompt_type_carries_multimodal_uuids_and_cache_salt(
    vllm_source_root: Path,
) -> None:
    module = _parse(vllm_source_root, PROMPT_INPUTS)
    prompt_options = _class(module, "_PromptOptions")

    uuid_option = _annotated_assignment(prompt_options, "multi_modal_uuids")
    salt_option = _annotated_assignment(prompt_options, "cache_salt")
    assert ast.unparse(uuid_option.annotation) == "NotRequired[MultiModalUUIDDict]"
    assert ast.unparse(salt_option.annotation) == "NotRequired[str]"

    for class_name in ("TextPrompt", "TokensPrompt", "EmbedsPrompt"):
        prompt_class = _class(module, class_name)
        assert [_dotted_name(base) for base in prompt_class.bases] == ["_PromptOptions"]

    decoder_only = _annotated_assignment(module, "DecoderOnlyPrompt")
    assert decoder_only.value is not None
    prompt_type = _annotated_assignment(module, "PromptType")
    assert prompt_type.value is not None
    assert {"TextPrompt", "TokensPrompt", "EmbedsPrompt"} <= _names(decoder_only.value)
    assert {"DecoderOnlyPrompt", "EncoderDecoderPrompt"} == _names(prompt_type.value)


def test_multimodal_uuid_becomes_encoder_cache_identifier_for_lookup_and_store(
    vllm_source_root: Path,
) -> None:
    input_module = _parse(vllm_source_root, INPUT_PROCESSOR)
    processor = _class(input_module, "InputProcessor")
    process_inputs = _direct_function(processor, "process_inputs")

    base_hash_values = _assignment_values(process_inputs, "base_mm_hash")
    assert len(base_hash_values) == 1
    assert ast.unparse(base_hash_values[0]) == "decoder_mm_hashes[modality][idx]"
    specs = _calls(process_inputs, "MultiModalFeatureSpec")
    assert len(specs) == 1
    identifier = _keyword(specs[0], "identifier")
    assert isinstance(identifier, ast.Call)
    assert _dotted_name(identifier.func) == "self._get_mm_identifier"
    assert ast.unparse(identifier.args[0]) == "base_mm_hash"

    cache_module = _parse(vllm_source_root, ENCODER_CACHE_MANAGER)
    manager = _class(cache_module, "EncoderCacheManager")
    for method_name in ("check_and_update_cache", "allocate"):
        method = _direct_function(manager, method_name)
        hash_values = _assignment_values(method, "mm_hash")
        assert [ast.unparse(value) for value in hash_values] == [
            "request.mm_features[input_id].identifier"
        ]
        cached_accesses = [
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Subscript)
            and _dotted_name(node.value) == "self.cached"
            and isinstance(node.slice, ast.Name)
            and node.slice.id == "mm_hash"
        ]
        assert cached_accesses


def test_block_hash_uses_multimodal_position_parent_and_first_block_salt(
    vllm_source_root: Path,
) -> None:
    module = _parse(vllm_source_root, KV_CACHE_UTILS)

    mm_keys = _direct_function(module, "_gen_mm_extra_hash_keys")
    appended_keys = _calls(mm_keys, "extra_keys.append")
    assert any(
        len(call.args) == 1
        and isinstance(call.args[0], ast.Tuple)
        and len(call.args[0].elts) == 2
        and ast.unparse(call.args[0].elts[0]) == "mm_feature.identifier"
        and ast.unparse(call.args[0].elts[1]) == "offset - start_token_idx"
        for call in appended_keys
    )

    extra_keys = _direct_function(module, "generate_block_hash_extra_keys")
    salt_values = _assignment_values(extra_keys, "cache_salt_keys")
    assert len(salt_values) == 1
    salt_value = salt_values[0]
    assert isinstance(salt_value, ast.IfExp)
    assert ast.unparse(salt_value.test) == (
        "start_token_idx == 0 and request.cache_salt"
    )
    assert ast.unparse(salt_value.body) == "[request.cache_salt]"
    combined_extra_keys = _assignment_values(extra_keys, "extra_keys")
    assert len(combined_extra_keys) == 1
    assert {"mm_extra_keys", "cache_salt_keys"} <= _names(combined_extra_keys[0])

    hash_tokens = _direct_function(module, "hash_block_tokens")
    hash_calls = _calls(hash_tokens, "hash_function")
    assert len(hash_calls) == 1
    assert len(hash_calls[0].args) == 1
    assert isinstance(hash_calls[0].args[0], ast.Tuple)
    assert [ast.unparse(item) for item in hash_calls[0].args[0].elts] == [
        "parent_block_hash",
        "curr_block_token_ids_tuple",
        "extra_keys",
    ]


def test_cache_config_allows_finer_hash_granularity_as_an_integer_multiple(
    vllm_source_root: Path,
) -> None:
    cache_module = _parse(vllm_source_root, CACHE_CONFIG)
    cache_config = _class(cache_module, "CacheConfig")
    hash_block_size = _annotated_assignment(cache_config, "hash_block_size")
    assert ast.unparse(hash_block_size.annotation) == "int | None"

    kv_module = _parse(vllm_source_root, KV_CACHE_UTILS)
    resolve_sizes = _direct_function(kv_module, "resolve_kv_cache_block_sizes")
    requested_values = _assignment_values(resolve_sizes, "requested")
    assert [ast.unparse(value) for value in requested_values] == [
        "cache_config.hash_block_size"
    ]
    assert any(
        isinstance(node, ast.Compare)
        and isinstance(node.left, ast.BinOp)
        and isinstance(node.left.op, ast.Mod)
        and ast.unparse(node.left) == "bs % hash_block_size"
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.NotEq)
        and len(node.comparators) == 1
        and ast.literal_eval(node.comparators[0]) == 0
        for node in ast.walk(resolve_sizes)
    )
    assert any(
        isinstance(node, ast.Return)
        and isinstance(node.value, ast.Tuple)
        and [ast.unparse(item) for item in node.value.elts]
        == ["scheduler_block_size", "hash_block_size"]
        for node in ast.walk(resolve_sizes)
    )


def test_async_engine_config_omits_hash_granularity_and_async_factory_exists(
    vllm_source_root: Path,
) -> None:
    args_module = _parse(vllm_source_root, ENGINE_ARGS)
    async_engine_args = _class(args_module, "AsyncEngineArgs")
    assert [_dotted_name(base) for base in async_engine_args.bases] == ["EngineArgs"]
    assert not any(
        isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
        and statement.name == "create_engine_config"
        for statement in async_engine_args.body
    )

    engine_args = _class(args_module, "EngineArgs")
    create_engine_config = _direct_function(engine_args, "create_engine_config")
    cache_config_calls = _calls(create_engine_config, "CacheConfig")
    assert len(cache_config_calls) == 1
    cache_keywords = {
        keyword.arg for keyword in cache_config_calls[0].keywords if keyword.arg
    }
    assert "hash_block_size" not in cache_keywords

    async_module = _parse(vllm_source_root, ASYNC_LLM)
    async_llm = _class(async_module, "AsyncLLM")
    factory = _direct_function(async_llm, "from_vllm_config")
    assert any(
        isinstance(decorator, ast.Name) and decorator.id == "classmethod"
        for decorator in factory.decorator_list
    )
    assert [argument.arg for argument in factory.args.args[:2]] == [
        "cls",
        "vllm_config",
    ]
    cls_calls = _calls(factory, "cls")
    assert len(cls_calls) == 1
    assert _dotted_name(_keyword(cls_calls[0], "vllm_config")) == "vllm_config"
