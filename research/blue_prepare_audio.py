"""Freeze four public Portuguese crops and Silero-assisted 2–6s VAD chunks."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def boundaries(total,speech,sample_rate=16000):
    """Preserve every input sample; prefer VAD ends, enforce duration bounds.

    Silence is retained between cuts. Continuous speech can force a 6s cut;
    this policy does not label a hard-cap split as a detected silence.
    """
    if type(total) is not int or type(sample_rate) is not int or total<2*sample_rate:
        raise ValueError('audio too short')
    if not speech or any(type(x['start']) is not int or type(x['end']) is not int or
        not 0<=x['start']<x['end']<=total for x in speech):
        raise ValueError('valid speech timestamps required')
    ends=sorted({x['end'] for x in speech})
    spans=[];start=0
    while total-start>6*sample_rate:
        options=[end for end in ends if start+2*sample_rate<=end<=start+6*sample_rate
                 and total-end>=2*sample_rate]
        stop=min(options,key=lambda x:(abs(x-start-4*sample_rate),x)) if options else min(
            start+6*sample_rate,total-2*sample_rate)
        spans.append((start,stop));start=stop
    spans.append((start,total))
    if any(not 2*sample_rate<=b-a<=6*sample_rate for a,b in spans):
        raise ValueError('VAD chunk bounds cannot be met')
    return spans


def prepare(archive,vad_dir,output):
    import numpy as np
    import soundfile as sf
    import torch
    archive,vad_dir,output=map(Path,(archive,vad_dir,output))
    if output.exists():raise ValueError('new preparation output required')
    if digest(archive)!='968d1b68a90132ca1f044f664866418b0db0351a93601a671d5637016da4f49b':
        raise ValueError('dataset archive differs from pinned source')
    if digest(vad_dir/'silero_vad.jit')!='e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720':
        raise ValueError('VAD model differs from pinned source')
    if digest(vad_dir/'utils_vad.py')!='dbdcfc5cef2c14abf8cc29b44afb79038509a9bc34ae8ba2e7482b48813949ee':
        raise ValueError('VAD analysis code differs from pinned source')
    sys.path.insert(0,str(vad_dir.resolve()))
    from utils_vad import get_speech_timestamps
    torch.set_num_threads(1)
    model=torch.jit.load(str(vad_dir/'silero_vad.jit'),map_location='cpu').eval()
    settings=dict(threshold=0.5,sampling_rate=16000,min_speech_duration_ms=250,
        max_speech_duration_s=6.0,min_silence_duration_ms=200,speech_pad_ms=30)
    output.mkdir(parents=True);(output/'clips').mkdir();(output/'cases').mkdir()
    clips=[];cases=[]
    with tarfile.open(archive,'r:gz') as tar:
        for member in sorted(tar.getmembers(),key=lambda x:x.name):
            if not member.isfile() or not member.name.endswith('.wav') or not 0<member.size<20000000:
                continue
            with tar.extractfile(member) as stream:raw=stream.read()
            data,rate=sf.read(io.BytesIO(raw),dtype='float32')
            if rate!=16000 or data.ndim!=1 or len(data)<15*rate or not np.isfinite(data).all():
                continue
            crop=data[:15*rate].copy();clip_id=f'pt-{len(clips):02d}'
            filename='clips/'+clip_id+'.wav';sf.write(output/filename,crop,rate,subtype='PCM_16')
            crop,_=sf.read(output/filename,dtype='float32')
            with torch.inference_mode():speech=get_speech_timestamps(torch.from_numpy(crop),model,**settings)
            spans=boundaries(len(crop),speech,rate)
            clips.append({'clip_id':clip_id,'source_member':member.name,
                'source_sha256':hashlib.sha256(raw).hexdigest(),'source_samples':len(data),
                'crop_start_sample':0,'crop_samples':len(crop),'wav':filename,
                'sha256':digest(output/filename),'vad_speech':speech})
            for i,(start,end) in enumerate(spans):
                case_id=clip_id+f'-vad-{i:02d}';name='cases/'+case_id+'.wav'
                sf.write(output/name,crop[start:end],rate,subtype='PCM_16')
                cases.append({'case_id':case_id,'clip_id':clip_id,'start_sample':start,
                    'end_sample':end,'duration_s':(end-start)/rate,'wav':name,
                    'sha256':digest(output/name),'boundary_kind':'vad_end' if end in {
                        x['end'] for x in speech} else 'duration_cap_or_recording_end'})
            if len(clips)==4:break
    if len(clips)!=4:raise ValueError('four qualifying recordings not found')
    result={'schema_version':1,'dataset':'google/fleurs','subset':'pt_br','split':'test',
        'dataset_revision':'70bb2e84b976b7e960aa89f1c648e09c59f894dd',
        'license':'cc-by-4.0','archive_sha256':digest(archive),'sample_rate':16000,
        'selection':'first four sorted archive members with >=15s mono16k audio; first15s crop',
        'vad_repository':'snakers4/silero-vad','vad_commit':'7e30209a3e901f9842f81b225f3e93d8199902b1',
        'vad_parameters':settings,'segmentation':'prefer VAD ends nearest4s; retain all samples; hard2–6s bounds',
        'clips':clips,'cases':cases}
    (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--vad-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try:
        result=prepare(args.archive,args.vad_dir,args.output)
        print(json.dumps({'status':'prepared','clips':len(result['clips']),'cases':len(result['cases'])}))
    except Exception as exc:
        print(json.dumps({'status':'failed','error_type':type(exc).__name__}),file=sys.stderr)
        raise SystemExit(1)
