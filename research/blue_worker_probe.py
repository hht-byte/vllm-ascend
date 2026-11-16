"""Bounded control-plane observations; no request text, IDs or tensors leave here."""
import hashlib
from functools import wraps
import importlib
import os
from pathlib import Path
import sys

_WEIGHT_HASHES={}
_BATCH_BRIDGE = 'vllm_ascend.patch.worker.blue_batch_invariant'


def physical_device_id():
    """Read the trusted launcher's physical assignment, never logical device 0."""
    try:
        value = int(os.environ.get('LLM_OPT_PHYSICAL_NPU_ID', ''))
    except (TypeError, ValueError):
        raise RuntimeError('trusted physical NPU assignment is unavailable') from None
    if not 0 <= value < 64:
        raise RuntimeError('trusted physical NPU assignment is out of range')
    return value


def batch_invariant_status():
    """Inspect loaded registration objects without importing operator packages."""
    envs = sys.modules.get('vllm.envs')
    requested = (getattr(envs, 'VLLM_BATCH_INVARIANT', False) if envs is not None
                 else os.environ.get('VLLM_BATCH_INVARIANT', '0') == '1')
    module = sys.modules.get('vllm_ascend.batch_invariant')
    bridge = sys.modules.get(_BATCH_BRIDGE)
    check = getattr(bridge, 'fia_bridge_installed', None)
    torch_npu = sys.modules.get('torch_npu')
    registered = getattr(module, '_batch_invariant_LIB', None) is not None
    return {'requested': requested is True,
            'registered': registered,
            'has_ascendc': getattr(module, 'HAS_ASCENDC_BATCH_INVARIANT', False) is True,
            'has_triton': getattr(module, 'HAS_TRITON', False) is True,
            'bridge_registered': bool(check(torch_npu)) if callable(check) and torch_npu is not None else False,
            'matmul_backend': getattr(module, 'BLUE_BATCH_INVARIANT_MATMUL_BACKEND', None) if registered else None}


def install_batch_initializer(cls):
    module = sys.modules.get(cls.__module__)
    original = getattr(module, 'init_batch_invariance', None)
    if not callable(original) or getattr(original, '_blue_batch_invariant_initializer', False):
        return
    @wraps(original)
    def initialize(*args, **kwargs):
        result = original(*args, **kwargs)
        status = batch_invariant_status()
        if status['requested']:
            if not all(status[key] for key in ('registered', 'has_ascendc', 'has_triton')):
                raise RuntimeError('full batch-invariant operator registration is unavailable')
            expected = os.environ.get('LLM_OPT_BATCH_INVARIANT_MATMUL_BACKEND', 'ascendc')
            if expected not in ('ascendc', 'triton') or status['matmul_backend'] != expected:
                raise RuntimeError('batch-invariant matmul registration does not match the requested backend')
            torch = sys.modules.get('torch')
            torch_npu = sys.modules.get('torch_npu')
            ops = getattr(getattr(torch, 'ops', None), 'batch_invariant_ops', None)
            if torch_npu is None or ops is None:
                raise RuntimeError('batch-invariant operator namespace is unavailable')
            importlib.import_module(_BATCH_BRIDGE).install_fia_bridge(torch_npu, ops)
            if not batch_invariant_status()['bridge_registered']:
                raise RuntimeError('batch-invariant FIA compatibility bridge is unavailable')
        return result
    initialize._blue_batch_invariant_initializer = True
    module.init_batch_invariance = initialize


def probe(self,reset=False):
    if type(reset) is not bool:raise TypeError('reset must be boolean')
    physical_device = physical_device_id()
    import torch
    if reset:
        torch.npu.synchronize()
        torch.npu.reset_peak_memory_stats()
    runner=self.model_runner;drafter=getattr(runner,'drafter',None)
    online=getattr(drafter,'online',None);graphs=getattr(online,'graphs',None)
    free,total=torch.npu.mem_get_info()
    memory={key:int(value) for key,value in {
        'allocated_bytes':torch.npu.memory_allocated(),
        'reserved_bytes':torch.npu.memory_reserved(),
        'peak_allocated_bytes':torch.npu.max_memory_allocated(),
        'peak_reserved_bytes':torch.npu.max_memory_reserved(),
        'device_used_bytes':total-free,'device_total_bytes':total}.items()}
    weight_sha=None
    if online is not None:
        path=Path(drafter.settings['checkpoint']);stat=path.stat()
        key=(str(path),stat.st_size,stat.st_mtime_ns)
        if key not in _WEIGHT_HASHES:
            digest=hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(4*1024**2),b''):digest.update(block)
            _WEIGHT_HASHES[key]=digest.hexdigest()
        weight_sha=_WEIGHT_HASHES[key]
    files={}
    for name in ('vllm','vllm.v1.core.sched.scheduler','vllm_ascend',
                 'vllm_ascend.batch_invariant',
                 _BATCH_BRIDGE,
                 'vllm_ascend.worker.model_runner_v1','vllm_ascend.worker.worker',
                 'vllm_ascend.spec_decode.external_draft_proposer',
                 'vllm_ascend.patch.worker.patch_external_draft',
                 'vllm_ascend.patch.worker.patch_external_draft_first_verify',
                 'vllm_ascend.patch.worker.patch_external_draft_mtp',
                 'vllm_ascend.spec_decode.asr_mtp_heads',
                 'vllm_ascend.spec_decode.asr_mtp_fused',
                 'vllm_ascend.spec_decode.asr_mtp_qkv_rope',
                 'vllm_ascend.spec_decode.asr_mtp_row_copy',
                 'vllm_ascend.spec_decode.asr_mtp_row_copy_kernel',
                 'vllm_ascend.spec_decode.asr_mtp_worker',
                 'vllm_ascend.spec_decode.asr_mtp_graph',
                 'vllm_ascend.spec_decode.asr_mtp_staging',
                 'vllm_ascend.spec_decode.asr_mtp_token_graph',
                 'vllm_ascend.spec_decode.asr_mtp_token_attention',
                 'vllm_ascend.spec_decode.asr_mtp_paged_cache',
                 'vllm_ascend.spec_decode.asr_mtp_paged_attention',
                 'vllm_ascend.spec_decode.asr_mtp_metrics',
                 'vllm_ascend.spec_decode.asr_mtp_device_rejection'):
        module=sys.modules.get(name);origin=getattr(module,'__file__',None)
        if origin:
            path=Path(origin).resolve()
            if not path.is_relative_to('/overlay'):raise RuntimeError('worker import escaped registered overlay')
            files[name]={'origin':str(path),
                'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    stats=getattr(online,'stats',{})
    handoff=getattr(online,'device_handoff',None)
    return {'hardware_class':torch.npu.get_device_name(0).replace('Ascend',''),
        'physical_device':physical_device,
        'memory':memory,'module_files':files,'batch_invariant':batch_invariant_status(),'mtp':{
            'loaded':online is not None,'heads':getattr(getattr(online,'heads',None),'num_heads',0),
            'weights_sha256':weight_sha,'head_batches':int(stats.get('head_batches',0)),
            'graph_replays':int(getattr(graphs,'stats',{}).get('replays',0)),
            'device_handoff_batches':int(getattr(handoff,'stats',{}).get('batches',0)),
            'handoff_fallbacks':{k:int(v) for k,v in getattr(handoff,'stats',{}).items() if k.startswith('fallback_')},
            'graph_ready':bool(getattr(graphs,'ready',False))}}


def install(cls):
    if not hasattr(cls,'blue_mtp_probe'):cls.blue_mtp_probe=probe
    install_batch_initializer(cls)
