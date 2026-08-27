import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

type FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef

QWEN3_ASR_MODEL = Path("vllm/model_executor/models/qwen3_asr.py")
QWEN3_OMNI_THINKER = Path("vllm/model_executor/models/qwen3_omni_moe_thinker.py")
MULTIMODAL_INPUTS = Path("vllm/multimodal/inputs.py")
PROMPT_INPUTS = Path("vllm/inputs/llm.py")
KV_CACHE_UTILS = Path("vllm/v1/core/kv_cache_utils.py")
CACHE_CONFIG = Path("vllm/config/cache.py")
ENGINE_ARGS = Path("vllm/engine/arg_utils.py")
ASYNC_LLM = Path("vllm/v1/engine/async_llm.py")
INPUT_PROCESSOR = Path("vllm/v1/engine/input_processor.py")
ENCODER_CACHE_MANAGER = Path("vllm/v1/core/encoder_cache_manager.py")
MM_PARSE = Path("vllm/multimodal/parse.py")
PROCESSOR_INPUTS = Path("vllm/multimodal/processing/inputs.py")
RENDERER_BASE = Path("vllm/renderers/base.py")
RENDERER_PREPROCESS = Path("vllm/renderers/inputs/preprocess.py")
INPUTS_PREPROCESS = Path("vllm/inputs/preprocess.py")
INPUTS_ENGINE = Path("vllm/inputs/engine.py")
PROCESSOR = Path("vllm/multimodal/processing/processor.py")


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


def _assignments_to_name(container: ast.AST, target_name: str) -> list[ast.Assign]:
    return [
        node
        for node in ast.walk(container)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == target_name
            for target in node.targets
        )
    ]


def _union_members(annotation: ast.expr) -> list[ast.expr]:
    assert isinstance(annotation, ast.Subscript)
    assert _dotted_name(annotation.value) == "Union"
    if isinstance(annotation.slice, ast.Tuple):
        return list(annotation.slice.elts)
    return [annotation.slice]


def _names(container: ast.AST) -> set[str]:
    return {node.id for node in ast.walk(container) if isinstance(node, ast.Name)}


def _assert_qwen3_omni_audio_padding_dataflow(module: ast.Module) -> None:
    processor = _class(module, "Qwen3OmniMoeThinkerMultiModalProcessor")
    call_hf = _direct_function(processor, "_call_hf_processor")
    pad = _nested_function(call_hf, "pad_to_hop_length")
    assert any(
        isinstance(node, ast.Call)
        and _dotted_name(node.func) == "np.pad"
        and any(
            keyword.arg == "mode"
            and ast.literal_eval(keyword.value) == "constant"
            for keyword in node.keywords
        )
        for node in ast.walk(pad)
        if isinstance(node, ast.Call)
    )
    assert any(
        isinstance(node, ast.Subscript)
        and _dotted_name(node.value) == "x.shape"
        and ast.unparse(node.slice) == "-1"
        for node in ast.walk(pad)
    )

    audio_assignment = next(
        (
            node
            for node in ast.walk(call_hf)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Subscript)
            and _dotted_name(node.targets[0].value) == "mm_data"
            and isinstance(node.targets[0].slice, ast.Constant)
            and node.targets[0].slice.value == "audio"
        ),
        None,
    )
    assert audio_assignment is not None
    assert isinstance(audio_assignment.value, ast.ListComp)
    audio_list = audio_assignment.value
    assert len(audio_list.generators) == 1
    audio_generator = audio_list.generators[0]
    assert isinstance(audio_generator.target, ast.Name)
    assert audio_generator.target.id == "audio"
    assert isinstance(audio_generator.iter, ast.Name)
    assert audio_generator.iter.id == "audios"
    pad_calls = [
        node
        for node in ast.walk(audio_list)
        if isinstance(node, ast.Call)
        and _dotted_name(node.func) == "pad_to_hop_length"
    ]
    assert pad_calls
    assert all(
        len(call.args) == 2
        and ast.unparse(call.args[0]) in {"audio", "audio[0]"}
        and ast.unparse(call.args[1]) == "hop_length"
        for call in pad_calls
    )

    final_if = next(
        node
        for node in ast.walk(call_hf)
        if isinstance(node, ast.If)
        and any(
            isinstance(named, ast.NamedExpr)
            and isinstance(named.target, ast.Name)
            and named.target.id == "audios"
            and isinstance(named.value, ast.Call)
            and _dotted_name(named.value.func) == "mm_data.get"
            and len(named.value.args) == 2
            and ast.literal_eval(named.value.args[0]) == "audio"
            for named in ast.walk(node.test)
        )
    )
    assert audio_assignment.lineno < final_if.lineno
    frame_loops = [
        node
        for node in final_if.body
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Call)
        and _dotted_name(node.iter.func) == "enumerate"
        and len(node.iter.args) == 1
        and ast.unparse(node.iter.args[0]) == "audios"
    ]
    assert len(frame_loops) == 1
    frame_loop = frame_loops[0]
    audio_length = next(
        node
        for node in frame_loop.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "audio_length"
    )
    assert isinstance(audio_length.value, ast.IfExp)
    length_calls = [
        call
        for call in ast.walk(audio_length.value)
        if isinstance(call, ast.Call) and _dotted_name(call.func) == "len"
    ]
    assert {ast.unparse(call.args[0]) for call in length_calls} == {
        "audio",
        "audio[0]",
    }
    num_frame = next(
        node
        for node in frame_loop.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "num_frame"
    )
    assert {"audio_length", "hop_length"} <= _names(num_frame.value)
    append_statements = [
        node
        for node in frame_loop.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and _dotted_name(node.value.func) == "audio_num_frames.append"
    ]
    assert len(append_statements) == 1
    append_statement = append_statements[0]
    append = append_statement.value
    assert len(append.args) == 1
    assert isinstance(append.args[0], ast.Name)
    assert append.args[0].id == "num_frame"
    assert (
        frame_loop.body.index(audio_length)
        < frame_loop.body.index(num_frame)
        < frame_loop.body.index(append_statement)
    )
    feature_lengths = [
        node
        for node in ast.walk(final_if)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Subscript)
        and _dotted_name(node.targets[0].value) == "hf_inputs"
        and isinstance(node.targets[0].slice, ast.Constant)
        and node.targets[0].slice.value == "audio_feature_lengths"
    ]
    assert len(feature_lengths) == 1
    feature_length = feature_lengths[0]
    assert feature_length in final_if.body
    assert final_if.body.index(frame_loop) < final_if.body.index(feature_length)
    assert isinstance(feature_length.value, ast.Call)
    assert _dotted_name(feature_length.value.func) == "torch.tensor"
    assert len(feature_length.value.args) == 1
    assert ast.unparse(feature_length.value.args[0]) == "audio_num_frames"


def _assert_supplied_uuid_dataflow(
    *,
    preprocess: ast.Module,
    renderer: ast.Module,
    parse: ast.Module,
    processor_inputs: ast.Module,
    processor: ast.Module,
    input_processor: ast.Module,
    inputs_engine: ast.Module,
    cache: ast.Module,
) -> None:
    process_tokens = _direct_function(
        _class(preprocess, "InputPreprocessor"), "_process_tokens"
    )
    process_calls = _calls(process_tokens, "self._process_multimodal")
    assert len(process_calls) == 1
    uuid_keyword = next(
        keyword for keyword in process_calls[0].keywords if keyword.arg == "mm_uuids"
    )
    assert ast.unparse(uuid_keyword.value) == "parsed_content.get('multi_modal_uuids')"

    process_mm = _direct_function(_class(renderer, "BaseRenderer"), "_process_multimodal")
    parse_call = _calls(process_mm, "parse_mm_uuids")
    assert len(parse_call) == 1
    assert ast.unparse(parse_call[0].args[0]) == "mm_uuids"
    uuid_assignment = next(
        node
        for node in ast.walk(process_mm)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "mm_uuid_items"
            for target in node.targets
        )
        and isinstance(node.value, ast.Call)
        and _dotted_name(node.value.func) == "parse_mm_uuids"
    )
    assert ast.unparse(uuid_assignment.value.args[0]) == "mm_uuids"
    process_uuid_call = _calls(process_mm, "self._process_mm_uuids")
    assert len(process_uuid_call) == 1
    assert ast.unparse(process_uuid_call[0].args[2]) == "mm_uuid_items"
    processor_input_call = _calls(process_mm, "MMProcessorInputs")
    assert len(processor_input_call) == 1
    assert ast.unparse(processor_input_call[0].args[2]) == "mm_uuid_items"
    processor_input_assignments = _assignments_to_name(
        process_mm, "mm_processor_inputs"
    )
    assert len(processor_input_assignments) == 1
    assert processor_input_assignments[0].value is processor_input_call[0]

    parse_uuids = _direct_function(parse, "parse_mm_uuids")
    assert any(
        isinstance(node, ast.IfExp)
        and ast.unparse(node.test) == "isinstance(uuids, str)"
        and ast.unparse(node.body) == "[uuids]"
        and ast.unparse(node.orelse) == "uuids"
        for node in ast.walk(parse_uuids)
    )

    get_hashes = _direct_function(
        _class(processor_inputs, "ProcessorInputs"), "get_mm_hashes"
    )
    assert any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "mm_uuid_items"
            for target in node.targets
        )
        and ast.unparse(node.value) == "self.mm_uuid_items or {}"
        for node in ast.walk(get_hashes)
    )
    uuid_branch = next(
        node
        for node in ast.walk(get_hashes)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "uuid_item is None or hf_processor_mm_kwargs"
    )
    uuid_append = next(
        node
        for statement in uuid_branch.orelse
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
        and _dotted_name(node.func) == "hashes.append"
    )
    assert len(uuid_append.args) == 1
    assert ast.unparse(uuid_append.args[0]) == "uuid_item"
    assert any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "uuid_item"
            for target in node.targets
        )
        and ast.unparse(node.value) == "uuid_items[i]"
        for node in ast.walk(get_hashes)
    )

    apply_hf = _direct_function(
        _class(processor, "BaseMultiModalProcessor"), "_apply_hf_processor"
    )
    cached_apply_hf = _direct_function(
        _class(processor, "BaseMultiModalProcessor"), "_cached_apply_hf_processor"
    )
    for apply_method in (apply_hf, cached_apply_hf):
        mm_hash_assignments = _assignments_to_name(apply_method, "mm_hashes")
        assert len(mm_hash_assignments) == 1
        mm_hash_call = mm_hash_assignments[0].value
        assert isinstance(mm_hash_call, ast.Call)
        assert _dotted_name(mm_hash_call.func) == "inputs.get_mm_hashes"
        assert len(mm_hash_call.args) == 1
        assert ast.unparse(mm_hash_call.args[0]) == "self.info.model_id"

        mm_info_assignments = _assignments_to_name(apply_method, "mm_info")
        assert len(mm_info_assignments) == 1
        processing_info = mm_info_assignments[0].value
        assert isinstance(processing_info, ast.Call)
        assert _dotted_name(processing_info.func) == "MultiModalProcessingInfo"
        assert ast.unparse(_keyword(processing_info, "hashes")) == "mm_hashes"
        mm_info_returns = [
            node
            for node in ast.walk(apply_method)
            if isinstance(node, ast.Return)
            and isinstance(node.value, ast.Tuple)
            and [ast.unparse(item) for item in node.value.elts]
            == ["prompt_ids", "mm_info", "is_update_applied"]
        ]
        assert len(mm_info_returns) == 1

    apply = _direct_function(_class(processor, "BaseMultiModalProcessor"), "apply")
    cached_result_assignments = [
        node
        for node in ast.walk(apply)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Tuple)
        and [ast.unparse(item) for item in node.targets[0].elts]
        == ["prompt_ids", "mm_info", "is_update_applied"]
        and isinstance(node.value, ast.Call)
        and _dotted_name(node.value.func) == "self._cached_apply_hf_processor"
        and [ast.unparse(arg) for arg in node.value.args] == ["inputs", "timing_ctx"]
    ]
    assert len(cached_result_assignments) == 1
    mm_input_returns = [
        node.value
        for node in ast.walk(apply)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Call)
        and _dotted_name(node.value.func) == "mm_input"
    ]
    assert len(mm_input_returns) == 1
    assert ast.unparse(_keyword(mm_input_returns[0], "mm_hashes")) == "mm_info.hashes"

    mm_inputs_assignments = _assignments_to_name(process_mm, "mm_inputs")
    assert len(mm_inputs_assignments) == 1
    processor_apply = mm_inputs_assignments[0].value
    assert isinstance(processor_apply, ast.Call)
    assert _dotted_name(processor_apply.func) == "mm_processor.apply"
    assert [ast.unparse(arg) for arg in processor_apply.args] == [
        "mm_processor_inputs",
        "mm_timing_ctx",
    ]
    renderer_returns = [
        node
        for node in ast.walk(process_mm)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Name)
        and node.value.id == "mm_inputs"
    ]
    assert len(renderer_returns) == 1

    renderer_process_tokens = _direct_function(
        _class(renderer, "BaseRenderer"), "_process_tokens"
    )
    multimodal_branch = next(
        node
        for node in renderer_process_tokens.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.NamedExpr)
        and isinstance(node.test.target, ast.Name)
        and node.test.target.id == "multi_modal_data"
    )
    renderer_outputs = [
        node
        for node in multimodal_branch.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "engine_input"
            for target in node.targets
        )
    ]
    assert len(renderer_outputs) == 1
    assert isinstance(renderer_outputs[0].value, ast.Call)
    assert _dotted_name(renderer_outputs[0].value.func) == "self._process_multimodal"
    assert any(
        isinstance(node, ast.Return)
        and isinstance(node.value, ast.Name)
        and node.value.id == "engine_input"
        for node in renderer_process_tokens.body
    )

    process_inputs = _direct_function(
        _class(input_processor, "InputProcessor"), "process_inputs"
    )
    processed_inputs = _assignment_values(process_inputs, "processed_inputs")
    assert "prompt" in [ast.unparse(value) for value in processed_inputs]
    decoder_split = [
        node
        for node in ast.walk(process_inputs)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Tuple)
        and [ast.unparse(item) for item in node.targets[0].elts]
        == ["encoder_inputs", "decoder_inputs"]
        and isinstance(node.value, ast.Call)
        and _dotted_name(node.value.func) == "split_enc_dec_input"
        and [ast.unparse(arg) for arg in node.value.args] == ["processed_inputs"]
    ]
    assert len(decoder_split) == 1
    split_input = _direct_function(inputs_engine, "split_enc_dec_input")
    decoder_only_returns = [
        node
        for node in split_input.body
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Tuple)
        and [ast.unparse(item) for item in node.value.elts] == ["None", "inputs"]
    ]
    assert len(decoder_only_returns) == 1
    decoder_hashes = _assignment_values(process_inputs, "decoder_mm_hashes")
    assert [ast.unparse(value) for value in decoder_hashes] == [
        "decoder_inputs['mm_hashes']"
    ]
    base_hash_values = _assignment_values(process_inputs, "base_mm_hash")
    assert [ast.unparse(value) for value in base_hash_values] == [
        "decoder_mm_hashes[modality][idx]"
    ]
    specs = _calls(process_inputs, "MultiModalFeatureSpec")
    assert len(specs) == 1
    identifier = _keyword(specs[0], "identifier")
    assert isinstance(identifier, ast.Call)
    assert _dotted_name(identifier.func) == "self._get_mm_identifier"
    assert ast.unparse(identifier.args[0]) == "base_mm_hash"
    assert ast.unparse(_keyword(specs[0], "mm_hash")) == "base_mm_hash"

    manager = _class(cache, "EncoderCacheManager")
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


def test_qwen3_omni_pads_pcm_before_deriving_audio_feature_lengths(
    vllm_source_root: Path,
) -> None:
    module = _parse(vllm_source_root, QWEN3_OMNI_THINKER)
    _assert_qwen3_omni_audio_padding_dataflow(module)


def test_qwen3_omni_padding_dataflow_rejects_disconnected_audio_assignment(
    vllm_source_root: Path,
) -> None:
    module = _parse(vllm_source_root, QWEN3_OMNI_THINKER)
    for node in ast.walk(module):
        if not isinstance(node, ast.Assign) or not isinstance(node.targets[0], ast.Subscript):
            continue
        target = node.targets[0]
        if (
            isinstance(target.value, ast.Name)
            and target.value.id == "mm_data"
            and isinstance(target.slice, ast.Constant)
            and target.slice.value == "audio"
        ):
            target.slice = ast.Constant(value="disconnected")
            break
    else:
        raise AssertionError("audio assignment mutation target was not found")
    with pytest.raises(AssertionError):
        _assert_qwen3_omni_audio_padding_dataflow(module)


def test_qwen3_omni_padding_dataflow_rejects_disconnected_frame_append(
    vllm_source_root: Path,
) -> None:
    module = _parse(vllm_source_root, QWEN3_OMNI_THINKER)
    for node in ast.walk(module):
        if not isinstance(node, ast.Call) or _dotted_name(node.func) != "audio_num_frames.append":
            continue
        node.args = [ast.Constant(value=0)]
        break
    else:
        raise AssertionError("frame append mutation target was not found")
    with pytest.raises(AssertionError):
        _assert_qwen3_omni_audio_padding_dataflow(module)


def test_qwen3_omni_padding_dataflow_rejects_feature_lengths_before_append(
    vllm_source_root: Path,
) -> None:
    module = _parse(vllm_source_root, QWEN3_OMNI_THINKER)
    processor = _class(module, "Qwen3OmniMoeThinkerMultiModalProcessor")
    call_hf = _direct_function(processor, "_call_hf_processor")
    final_if = next(
        node
        for node in ast.walk(call_hf)
        if isinstance(node, ast.If)
        and any(
            isinstance(named, ast.NamedExpr)
            and isinstance(named.target, ast.Name)
            and named.target.id == "audios"
            for named in ast.walk(node.test)
        )
    )
    frame_loop = next(node for node in final_if.body if isinstance(node, ast.For))
    feature_length_index = next(
        index
        for index, node in enumerate(final_if.body)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Subscript)
        and ast.unparse(node.targets[0]) == "hf_inputs['audio_feature_lengths']"
    )
    feature_length = final_if.body.pop(feature_length_index)
    append_index = next(
        index
        for index, node in enumerate(frame_loop.body)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and _dotted_name(node.value.func) == "audio_num_frames.append"
    )
    frame_loop.body.insert(append_index, feature_length)

    with pytest.raises(AssertionError):
        _assert_qwen3_omni_audio_padding_dataflow(module)


def test_qwen3_omni_pads_pcm_before_deriving_audio_feature_lengths_legacy_fragments(
    vllm_source_root: Path,
) -> None:
    module = _parse(vllm_source_root, QWEN3_OMNI_THINKER)
    processor = _class(module, "Qwen3OmniMoeThinkerMultiModalProcessor")
    call_hf = _direct_function(processor, "_call_hf_processor")
    pad = _nested_function(call_hf, "pad_to_hop_length")
    assert any(
        isinstance(node, ast.Call)
        and _dotted_name(node.func) == "np.pad"
        and any(
            keyword.arg == "mode"
            and ast.literal_eval(keyword.value) == "constant"
            for keyword in node.keywords
        )
        for node in ast.walk(pad)
        if isinstance(node, ast.Call)
    )
    assert any(
        isinstance(node, ast.Compare)
        and ast.unparse(node.left) == "length % hop_length"
        for node in ast.walk(pad)
    )
    audio_list_comprehensions = [
        node
        for node in ast.walk(call_hf)
        if isinstance(node, ast.ListComp)
        and any(
            isinstance(call, ast.Call)
            and _dotted_name(call.func) == "pad_to_hop_length"
            for call in ast.walk(node)
        )
    ]
    assert len(audio_list_comprehensions) == 1
    frame_assignments = _assignment_values(call_hf, "num_frame")
    assert any(
        isinstance(value, ast.IfExp)
        and ast.unparse(value.body) == "audio_length // hop_length"
        and ast.unparse(value.orelse) == "audio_length // hop_length - 1"
        for value in frame_assignments
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


def test_supplied_multimodal_uuid_flows_from_prompt_to_processor_hash(
    vllm_source_root: Path,
) -> None:
    prompt_module = _parse(vllm_source_root, PROMPT_INPUTS)
    prompt_options = _class(prompt_module, "_PromptOptions")
    assert _annotated_assignment(prompt_options, "multi_modal_uuids")

    _assert_supplied_uuid_dataflow(
        preprocess=_parse(vllm_source_root, INPUTS_PREPROCESS),
        renderer=_parse(vllm_source_root, RENDERER_BASE),
        parse=_parse(vllm_source_root, MM_PARSE),
        processor_inputs=_parse(vllm_source_root, PROCESSOR_INPUTS),
        processor=_parse(vllm_source_root, PROCESSOR),
        input_processor=_parse(vllm_source_root, INPUT_PROCESSOR),
        inputs_engine=_parse(vllm_source_root, INPUTS_ENGINE),
        cache=_parse(vllm_source_root, ENCODER_CACHE_MANAGER),
    )


def test_supplied_uuid_contract_rejects_disconnected_renderer_argument(
    vllm_source_root: Path,
) -> None:
    preprocess = _parse(vllm_source_root, INPUTS_PREPROCESS)
    process_tokens = _direct_function(
        _class(preprocess, "InputPreprocessor"), "_process_tokens"
    )
    process_calls = _calls(process_tokens, "self._process_multimodal")
    assert len(process_calls) == 1
    uuid_keyword = next(keyword for keyword in process_calls[0].keywords if keyword.arg == "mm_uuids")
    uuid_keyword.value = ast.Constant(value=None)
    with pytest.raises(AssertionError):
        _assert_supplied_uuid_dataflow(
            preprocess=preprocess,
            renderer=_parse(vllm_source_root, RENDERER_BASE),
            parse=_parse(vllm_source_root, MM_PARSE),
            processor_inputs=_parse(vllm_source_root, PROCESSOR_INPUTS),
            processor=_parse(vllm_source_root, PROCESSOR),
            input_processor=_parse(vllm_source_root, INPUT_PROCESSOR),
            inputs_engine=_parse(vllm_source_root, INPUTS_ENGINE),
            cache=_parse(vllm_source_root, ENCODER_CACHE_MANAGER),
        )


def test_supplied_uuid_contract_rejects_constant_hash_replacement(
    vllm_source_root: Path,
) -> None:
    processor_inputs = _parse(vllm_source_root, PROCESSOR_INPUTS)
    get_hashes = _direct_function(_class(processor_inputs, "ProcessorInputs"), "get_mm_hashes")
    uuid_append = next(
        node
        for node in ast.walk(get_hashes)
        if isinstance(node, ast.Call)
        and _dotted_name(node.func) == "hashes.append"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "uuid_item"
    )
    uuid_append.args[0] = ast.Constant(value="constant")
    with pytest.raises(AssertionError):
        _assert_supplied_uuid_dataflow(
            preprocess=_parse(vllm_source_root, INPUTS_PREPROCESS),
            renderer=_parse(vllm_source_root, RENDERER_BASE),
            parse=_parse(vllm_source_root, MM_PARSE),
            processor_inputs=processor_inputs,
            processor=_parse(vllm_source_root, PROCESSOR),
            input_processor=_parse(vllm_source_root, INPUT_PROCESSOR),
            inputs_engine=_parse(vllm_source_root, INPUTS_ENGINE),
            cache=_parse(vllm_source_root, ENCODER_CACHE_MANAGER),
        )


def test_supplied_uuid_contract_rejects_constant_processor_hashes(
    vllm_source_root: Path,
) -> None:
    processor = _parse(vllm_source_root, PROCESSOR)
    apply_hf = _direct_function(
        _class(processor, "BaseMultiModalProcessor"), "_apply_hf_processor"
    )
    mm_hashes = next(
        node
        for node in ast.walk(apply_hf)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "mm_hashes"
            for target in node.targets
        )
    )
    mm_hashes.value = ast.Constant(value="constant")

    with pytest.raises(AssertionError):
        _assert_supplied_uuid_dataflow(
            preprocess=_parse(vllm_source_root, INPUTS_PREPROCESS),
            renderer=_parse(vllm_source_root, RENDERER_BASE),
            parse=_parse(vllm_source_root, MM_PARSE),
            processor_inputs=_parse(vllm_source_root, PROCESSOR_INPUTS),
            processor=processor,
            input_processor=_parse(vllm_source_root, INPUT_PROCESSOR),
            inputs_engine=_parse(vllm_source_root, INPUTS_ENGINE),
            cache=_parse(vllm_source_root, ENCODER_CACHE_MANAGER),
        )


def test_supplied_uuid_contract_rejects_disconnected_processor_output(
    vllm_source_root: Path,
) -> None:
    renderer = _parse(vllm_source_root, RENDERER_BASE)
    process_mm = _direct_function(
        _class(renderer, "BaseRenderer"), "_process_multimodal"
    )
    mm_inputs = next(
        node
        for node in ast.walk(process_mm)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "mm_inputs"
            for target in node.targets
        )
        and isinstance(node.value, ast.Call)
        and _dotted_name(node.value.func) == "mm_processor.apply"
    )
    mm_inputs.targets = [ast.Name(id="disconnected_mm_inputs", ctx=ast.Store())]

    with pytest.raises(AssertionError):
        _assert_supplied_uuid_dataflow(
            preprocess=_parse(vllm_source_root, INPUTS_PREPROCESS),
            renderer=renderer,
            parse=_parse(vllm_source_root, MM_PARSE),
            processor_inputs=_parse(vllm_source_root, PROCESSOR_INPUTS),
            processor=_parse(vllm_source_root, PROCESSOR),
            input_processor=_parse(vllm_source_root, INPUT_PROCESSOR),
            inputs_engine=_parse(vllm_source_root, INPUTS_ENGINE),
            cache=_parse(vllm_source_root, ENCODER_CACHE_MANAGER),
        )


def test_supplied_uuid_contract_rejects_disconnected_processor_inputs(
    vllm_source_root: Path,
) -> None:
    renderer = _parse(vllm_source_root, RENDERER_BASE)
    process_mm = _direct_function(
        _class(renderer, "BaseRenderer"), "_process_multimodal"
    )
    processor_inputs = next(
        node
        for node in ast.walk(process_mm)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "mm_processor_inputs"
            for target in node.targets
        )
        and isinstance(node.value, ast.Call)
        and _dotted_name(node.value.func) == "MMProcessorInputs"
    )
    processor_inputs.targets = [
        ast.Name(id="disconnected_processor_inputs", ctx=ast.Store())
    ]

    with pytest.raises(AssertionError):
        _assert_supplied_uuid_dataflow(
            preprocess=_parse(vllm_source_root, INPUTS_PREPROCESS),
            renderer=renderer,
            parse=_parse(vllm_source_root, MM_PARSE),
            processor_inputs=_parse(vllm_source_root, PROCESSOR_INPUTS),
            processor=_parse(vllm_source_root, PROCESSOR),
            input_processor=_parse(vllm_source_root, INPUT_PROCESSOR),
            inputs_engine=_parse(vllm_source_root, INPUTS_ENGINE),
            cache=_parse(vllm_source_root, ENCODER_CACHE_MANAGER),
        )


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


def test_decoder_hash_is_the_supplied_uuid_hash_used_for_feature_identifier(
    vllm_source_root: Path,
) -> None:
    input_module = _parse(vllm_source_root, INPUT_PROCESSOR)
    processor = _class(input_module, "InputProcessor")
    process_inputs = _direct_function(processor, "process_inputs")
    decoder_hashes = _assignment_values(process_inputs, "decoder_mm_hashes")
    assert decoder_hashes
    base_hash_values = _assignment_values(process_inputs, "base_mm_hash")
    assert [ast.unparse(value) for value in base_hash_values] == [
        "decoder_mm_hashes[modality][idx]"
    ]
    specs = _calls(process_inputs, "MultiModalFeatureSpec")
    assert len(specs) == 1
    assert ast.unparse(_keyword(specs[0], "mm_hash")) == "base_mm_hash"


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
