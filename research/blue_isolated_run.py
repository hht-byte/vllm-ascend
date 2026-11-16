"""Linux-only trusted launcher: private namespaces, chroot and dropped capabilities.

Run from a container whose administrator explicitly permits namespace creation.
Only the output directory, temporary files and shared memory remain writable.
The model/overlay/tool directories are read-only and network has loopback only.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

_diagnostic_dir = None


def child_environment(parent, device):
    keep = ('PATH','LD_LIBRARY_PATH','ASCEND_HOME_PATH','ASCEND_TOOLKIT_HOME',
            'ASCEND_AICPU_PATH','ASCEND_OPP_PATH','ASCEND_CUSTOM_OPP_PATH',
            'CANN_TOOLKIT_HOME','ATB_HOME_PATH')
    result = {key:parent[key] for key in keep if key in parent}
    result.update(HOME='/tmp/home', TMPDIR='/tmp', PYTHONPATH='/overlay:/tools',
        PYTHONDONTWRITEBYTECODE='1', ASCEND_RT_VISIBLE_DEVICES='0',
        LLM_OPT_PHYSICAL_NPU_ID=str(device),
        VLLM_HOST_IP='127.0.0.1', MASTER_ADDR='127.0.0.1',
        GLOO_SOCKET_IFNAME='lo', HCCL_SOCKET_IFNAME='lo',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
        VLLM_CACHE_ROOT='/results/cache/vllm', TRITON_CACHE_DIR='/results/cache/triton',
        TORCHINDUCTOR_CACHE_DIR='/results/cache/inductor',
        TORCH_EXTENSIONS_DIR='/results/cache/extensions',
        ASCEND_PROCESS_LOG_PATH='/results/npu-log', OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', VLLM_WORKER_MULTIPROC_METHOD='spawn')
    vendor = [p for p in parent.get('PYTHONPATH','').split(':') if p.startswith('/usr/local/Ascend/')]
    if vendor: result['PYTHONPATH'] += ':'+':'.join(dict.fromkeys(vendor))
    return result


def validate(config):
    root = Path(config['sandbox_root']).resolve()
    sources = [Path(config[key]).resolve() for key in ('overlay','tools','model','assets','results')]
    if root.exists() or any(not path.is_dir() for path in sources):
        raise ValueError('fresh sandbox and existing registered mount directories required')
    if any(root.is_relative_to(path) for path in sources[:4]):
        raise ValueError('sandbox must be outside read-only inputs')
    readonly = sources[:4]
    if config.get('checkpoint') is not None:
        readonly.append(Path(config['checkpoint']).resolve())
    results = sources[4]
    if any(results.is_relative_to(path) or path.is_relative_to(results) for path in readonly):
        raise ValueError('results must be disjoint from read-only inputs')
    device = config['physical_device']
    if type(device) is not int or not 0 <= device < 64:
        raise ValueError('invalid physical device')
    command = config['command']
    if (not isinstance(command,list) or not command or
            any(not isinstance(x,str) or not x or '\x00' in x for x in command)):
        raise ValueError('registered argument array required')


def _inside(config):
    global _diagnostic_dir
    import fcntl
    import resource
    import socket
    import struct
    def phase(name):
        with (_diagnostic_dir/'isolation-phases.jsonl').open('a') as stream:
            stream.write(json.dumps({'phase':name})+'\n')
    phase('namespace_checks')
    # No mount mutation is possible unless both namespace transitions occurred.
    for kind in ('mnt','net','pid','ipc'):
        if str(os.stat('/proc/self/ns/'+kind).st_ino) == os.environ['BLUE_PARENT_'+kind.upper()]:
            raise RuntimeError('namespace isolation was not established')
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    data = fcntl.ioctl(sock.fileno(),0x8913,struct.pack('16sH14s',b'lo',0,b''))
    flags = struct.unpack('16sH14s',data)[1]
    fcntl.ioctl(sock.fileno(),0x8914,struct.pack('16sH14s',b'lo',flags|1,b''))
    sock.close()
    def command(*args):
        subprocess.run(args,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    command('mount','--make-rprivate','/')
    phase('mounts_private')
    root = Path(config['sandbox_root']).resolve()
    root.mkdir(parents=True)
    command('mount','--bind',str(root),str(root))
    def mount(source,destination,readonly=True,recursive=False):
        dest = root/destination.lstrip('/')
        if Path(source).is_dir(): dest.mkdir(parents=True,exist_ok=True)
        else:
            dest.parent.mkdir(parents=True,exist_ok=True);dest.touch()
        command('mount','--rbind' if recursive else '--bind',str(source),str(dest))
        if readonly:
            # A recursive /usr bind can contain driver submounts. Remount each
            # private copy read-only; never alter the original mount namespace.
            targets = [dest]
            if recursive:
                for line in Path('/proc/self/mountinfo').read_text().splitlines():
                    value = Path(line.split()[4])
                    if value != dest and value.is_relative_to(dest): targets.append(value)
            for target in sorted(set(targets),key=lambda p:len(str(p)),reverse=True):
                command('mount','-o','remount,bind,ro',str(target))
    mount('/usr','/usr',recursive=True)
    phase('usr_readonly')
    for name in ('bin','sbin','lib','lib64'):
        source = Path('/'+name)
        if source.is_symlink(): (root/name).symlink_to(os.readlink(source))
        elif source.exists(): mount(source,'/'+name,recursive=True)
    (root/'etc').mkdir()
    phase('runtime_libraries_ready')
    for name in ('ld.so.cache','ld.so.conf','nsswitch.conf','hosts','hostname',
                 'passwd','group','localtime','ascend_install.info','vnpu.cfg'):
        path = Path('/etc')/name
        if path.is_file(): shutil.copyfile(path,root/'etc'/name)
    for name in ('ld.so.conf.d','alternatives'):
        if (Path('/etc')/name).is_dir(): mount(Path('/etc')/name,'/etc/'+name)
    phase('etc_ready')
    for key,destination in (('overlay','/overlay'),('tools','/tools'),
                            ('model','/model'),('assets','/assets')):
        mount(config[key],destination)
    checkpoint = config.get('checkpoint')
    if checkpoint is not None: mount(checkpoint,'/weights/mtp.pt')
    mount(config['results'],'/results',readonly=False)
    phase('registered_inputs_ready')
    for dest,size in (('/tmp','8g'),('/run','64m'),('/dev/shm','16g')):
        target = root/dest.lstrip('/');target.mkdir(parents=True,exist_ok=True)
        command('mount','-t','tmpfs','-o','mode=1777,size='+size,'tmpfs',str(target))
    for name in ('null','zero','random','urandom',f'davinci{config["physical_device"]}',
                 'davinci8-manager','davinci_manager','dvmm_svm','devmm_svm','hisi_hdc'):
        source = Path('/dev')/name
        if source.exists(): mount(source,'/dev/'+name,readonly=False)
    mount('/sys','/sys',recursive=True)
    phase('devices_ready')
    (root/'proc').mkdir();command('mount','-t','proc','proc',str(root/'proc'))
    command('mount','-o','remount,bind,ro',str(root))
    phase('root_readonly')
    try: resource.setrlimit(resource.RLIMIT_MEMLOCK,(resource.RLIM_INFINITY,resource.RLIM_INFINITY))
    except (ValueError,OSError): pass
    os.chroot(root);os.chdir('/results')
    _diagnostic_dir = Path('/results')
    phase('chroot_ready')
    Path('/tmp/home').mkdir()
    env = child_environment(os.environ,config['physical_device'])
    os.execve('/usr/bin/setpriv',['setpriv','--bounding-set=-all','--inh-caps=-all',
        '--ambient-caps=-all','--no-new-privs',*config['command']],env)


def namespace_command(command):
    """Tie namespace init (and its descendants) to the launcher lifetime."""
    return ['/usr/bin/unshare','--mount','--net','--pid','--ipc',
            '--fork','--kill-child=SIGKILL','--mount-proc',*command]


def main():
    global _diagnostic_dir
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--inside',action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    _diagnostic_dir = Path(config['results'])
    if args.inside:
        _inside(config)
    else:
        validate(config)
        env = dict(os.environ)
        for kind in ('mnt','net','pid','ipc'):
            env['BLUE_PARENT_'+kind.upper()] = str(os.stat('/proc/self/ns/'+kind).st_ino)
        os.execve('/usr/bin/unshare',namespace_command([
            sys.executable,str(Path(__file__).resolve()),
            '--config',str(args.config.resolve()),'--inside']),env)


if __name__ == '__main__':
    try: main()
    except Exception as exc:
        if _diagnostic_dir is not None:
            import traceback
            try:
                with (_diagnostic_dir/f'isolation-failure-{os.getpid()}.log').open('x') as stream:
                    traceback.print_exc(file=stream)
                    if isinstance(exc,subprocess.CalledProcessError):
                        stream.write(repr(exc.stderr))
            except OSError: pass
        print(json.dumps({'status':'isolation_failed','error_type':type(exc).__name__}),file=sys.stderr)
        raise SystemExit(1)
