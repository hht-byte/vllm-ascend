"""Real native-vLLM Portuguese ASR service on an experiment-local Unix socket."""
import argparse
import asyncio
from contextlib import asynccontextmanager
import hashlib
import io
import json
from pathlib import Path
import sys

EXTERNAL_CLASS='vllm_ascend.spec_decode.external_draft_proposer.AscendExternalDraftProposer'


class RequestUnavailable(RuntimeError):
    def __init__(self, status_code, message):
        self.status_code = status_code
        super().__init__(message)


class CleanupUncertainError(RuntimeError):
    pass


def reserve_request(state, request_id):
    """Atomically claim an ID after validation, immediately before generation."""
    if state.get('cleanup_uncertain') or state.get('quarantined'):
        raise RequestUnavailable(503, 'service quarantined after unconfirmed cleanup')
    if request_id in state['active']:
        raise RequestUnavailable(409, 'request ID is already active')
    state['active'].add(request_id)


async def finalize_request(state, request_id, *, finished):
    """Always release bookkeeping; quarantine if engine abort is unconfirmed."""
    try:
        if not finished:
            try:
                await state['engine'].abort(request_id)
            except BaseException as error:
                state['cleanup_uncertain'] = True
                state['quarantined'] = True
                if isinstance(error, Exception):
                    raise CleanupUncertainError('engine request cleanup could not be confirmed') from None
                raise
    finally:
        state['active'].discard(request_id)


def engine_options(config):
    mode=config.get('mode','mtp');graph=config.get('graph_mode','full')
    if mode not in ('ar','mtp') or graph not in ('off','full'):raise ValueError('unsupported engine mode')
    options=dict(model=config.get('model','/model'),dtype='bfloat16',trust_remote_code=False,
        tensor_parallel_size=1,seed=0,async_scheduling=False,enable_prefix_caching=False,
        enable_chunked_prefill=True,enforce_eager=graph=='off',block_size=128,
        max_num_seqs=config.get('max_num_seqs',20),max_model_len=config.get('max_model_len',1024),
        max_num_batched_tokens=config.get('max_num_batched_tokens',2048),
        kv_cache_memory_bytes=config.get('kv_cache_memory_bytes',2*1024**3),
        gpu_memory_utilization=0.5,mm_processor_cache_gb=0,
        limit_mm_per_prompt={'audio':{'count':1,'length':96000}},disable_log_stats=True)
    if graph=='full':
        options['compilation_config']={'cudagraph_mode':'FULL',
            'cudagraph_capture_sizes':[1,2,4,8,16,32,64,128,256]}
    if mode=='mtp':
        options['speculative_config']={'method':'custom_class','model':EXTERNAL_CLASS,'num_speculative_tokens':5}
        settings=dict(checkpoint=config.get('checkpoint','/weights/mtp.pt'),num_heads=5,
            cold_heads=config.get('cold_heads',5),warm_heads=config.get('warm_heads',3),
            max_context_tokens=config.get('max_context_tokens',1024),
            cache_mib=config.get('cache_mib',2048),max_requests=config.get('max_num_seqs',20),
            graph_mode=graph,native_fusion=config.get('native_fusion',True),
            profile_stages=False,acceptance_stats=True)
        if graph=='full':
            settings.update(kv_cache_backend='paged',kv_block_size=128,
                graph_token_sizes=config.get('graph_token_sizes',[64,256,1024,2048]),
                graph_max_batch_size=config.get('graph_max_batch_size',20),
                graph_context_tokens=config.get('max_context_tokens',1024),
                device_rejection=config.get('device_rejection',True))
        else:settings['device_rejection']=False
        options['additional_config']={'asr_mtp':settings}
    return options


def request_extra(mode,initial_draft,diagnostics_allowed):
    if mode not in ('ar','mtp'):raise ValueError('unsupported request mode')
    if initial_draft is not None:
        if (not diagnostics_allowed or mode!='mtp' or not isinstance(initial_draft,list)
                or not 0<len(initial_draft)<=5 or any(type(t) is not int or t<0 for t in initial_draft)):
            raise ValueError('initial drafts are restricted to correctness diagnostics')
    if mode=='ar':return None
    extra={'ascend_mtp':True}
    if initial_draft is not None:
        extra.update(ascend_external_draft=initial_draft,ascend_external_draft_first_verify=True,
                     ascend_external_draft_rejoin='ngram_1')
    return extra


def application(config):
    import numpy as np
    import soundfile as sf
    from fastapi import FastAPI,HTTPException,Request
    from vllm import AsyncEngineArgs,SamplingParams
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.transformers_utils.processors.qwen3_asr import Qwen3ASRProcessor
    assets=Path(config.get('assets','/assets')).resolve()
    manifest=json.loads((assets/'manifest.json').read_text())
    cases={row['case_id']:row for row in manifest['cases']}
    state={'engine':None,'active':set(),'served':0,
           'cleanup_uncertain':False,'quarantined':False}

    @asynccontextmanager
    async def lifespan(app):
        processor=Qwen3ASRProcessor.from_pretrained(config.get('model','/model'),
            local_files_only=True,fix_mistral_regex=True)
        prompt=processor.apply_chat_template([{'role':'system','content':''},
            {'role':'user','content':[{'type':'audio','audio':''}]}],
            add_generation_prompt=True,tokenize=False)+f"language {config.get('language','Portuguese')}<asr_text>"
        state.update(processor=processor,prompt=prompt)
        state['engine']=AsyncLLM.from_engine_args(AsyncEngineArgs(**engine_options(config)))
        try:yield
        finally:state['engine'].shutdown(timeout=30)

    app=FastAPI(lifespan=lifespan)

    async def observe(reset=False):
        workers=await state['engine'].collective_rpc('blue_mtp_probe',timeout=30,kwargs={'reset':reset})
        if len(workers)!=1:raise RuntimeError('one NPU worker required')
        result=workers[0]
        result.update(identity=config.get('identity',{}),active_requests=len(state['active']),served=state['served'],
                      cleanup_uncertain=state['cleanup_uncertain'],quarantined=state['quarantined'])
        return result

    @app.get('/status')
    async def status():return await observe()

    @app.post('/reset')
    async def reset():return await observe(True)

    @app.post('/shutdown')
    async def shutdown():
        if state['active']:raise HTTPException(409,'active requests remain')
        async def stop():
            await asyncio.sleep(0.05)
            app.state.server.should_exit=True
        asyncio.create_task(stop())
        return {'status':'stopping'}

    @app.post('/transcribe')
    async def transcribe(request:Request):
        case_id=request.query_params.get('case_id');request_id=request.query_params.get('request_id')
        if case_id not in cases or not request_id or len(request_id)>128 or request_id in state['active']:
            raise HTTPException(400,'invalid registered request')
        raw=await request.body()
        if len(raw)>300000 or hashlib.sha256(raw).hexdigest()!=cases[case_id]['sha256']:
            raise HTTPException(400,'audio differs from frozen case')
        audio,rate=sf.read(io.BytesIO(raw),dtype='float32')
        if rate!=16000 or audio.ndim!=1 or not np.isfinite(audio).all():raise HTTPException(400,'unsupported audio')
        try:
            draft=json.loads(request.headers['x-initial-draft']) if 'x-initial-draft' in request.headers else None
            extra=request_extra(config.get('mode','mtp'),draft,config.get('allow_initial_draft',False))
        except (ValueError,TypeError,json.JSONDecodeError):raise HTTPException(400,'invalid draft diagnostic')
        params=SamplingParams(temperature=0.0,seed=0,max_tokens=config.get('max_new_tokens',128),extra_args=extra)
        payload={'prompt':state['prompt'],'multi_modal_data':{'audio':[(audio,rate)]},
                 'multi_modal_uuids':{'audio':[request_id]}}
        # Body reads above can overlap for duplicate IDs. Recheck and reserve
        # synchronously here, with no await before entering generation.
        try:reserve_request(state,request_id)
        except RequestUnavailable as error:raise HTTPException(error.status_code,str(error)) from None
        finished=False
        try:
            final=None
            async for output in state['engine'].generate(payload,params,request_id):final=output
            if final is None or not final.finished:raise RuntimeError('incomplete engine output')
            answer=final.outputs[0];finished=True;state['served']+=1
            return {'case_id':case_id,'text':answer.text,'token_ids':list(answer.token_ids),
                    'finish_reason':answer.finish_reason}
        finally:
            await finalize_request(state,request_id,finished=finished)
    return app


def main():
    import uvicorn
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();config=json.loads(args.config.read_text())
    app=application(config)
    server=uvicorn.Server(uvicorn.Config(app,uds='/results/service.sock',log_level='warning',access_log=False))
    app.state.server=server;server.run()


if __name__=='__main__':
    try:main()
    except Exception as exc:
        import traceback
        try:
            with Path('/results/server-failure.log').open('x') as f:traceback.print_exc(file=f)
        except OSError:pass
        print(json.dumps({'status':'server_failed','error_type':type(exc).__name__}),file=sys.stderr)
        raise SystemExit(1)
