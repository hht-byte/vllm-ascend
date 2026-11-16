"""Bounded control-plane observations; no request text, IDs or tensors leave here."""
import hashlib
from pathlib import Path
import sys

_WEIGHT_HASHES={}


def probe(self,reset=False):
    import torch
    if type(reset) is not bool:raise TypeError('reset must be boolean')
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
        'memory':memory,'module_files':files,'mtp':{
            'loaded':online is not None,'heads':getattr(getattr(online,'heads',None),'num_heads',0),
            'weights_sha256':weight_sha,'head_batches':int(stats.get('head_batches',0)),
            'graph_replays':int(getattr(graphs,'stats',{}).get('replays',0)),
            'device_handoff_batches':int(getattr(handoff,'stats',{}).get('batches',0)),
            'handoff_fallbacks':{k:int(v) for k,v in getattr(handoff,'stats',{}).items() if k.startswith('fallback_')},
            'graph_ready':bool(getattr(graphs,'ready',False))}}


def install(cls):
    if not hasattr(cls,'blue_mtp_probe'):cls.blue_mtp_probe=probe
