"""Service startup options for the optional online ASR MTP extension.

No vLLM/Torch import: call before constructing AsyncEngineArgs, and pass the
two service booleans separately to model.enable_prespec after engine creation.
"""
from collections.abc import Mapping
from os import fspath

EXTERNAL_DRAFT_CLASS = (
    'vllm_ascend.spec_decode.external_draft_proposer.AscendExternalDraftProposer'
)


def add_mtp_args(parser):
    """Register on the existing service parser before parse_args()."""
    from argparse import BooleanOptionalAction

    parser.add_argument(
        '--mtp', dest='mtp_enabled', action=BooleanOptionalAction, default=False,
        help='load online ASR MTP and enable it for formal requests (default: off)',
    )
    parser.add_argument(
        '--prespec-mtp', dest='prespec_mtp_enabled',
        action=BooleanOptionalAction, default=True,
        help='also enable MTP after shadow rollback verification; requires --mtp',
    )
    parser.add_argument(
        '--mtp-checkpoint', default=None,
        help='local checkpoint path; overrides additional_config.asr_mtp.checkpoint',
    )
    parser.add_argument(
        '--mtp-graph-mode', choices=('off', 'full'), default=None,
        help='auxiliary MTP graph mode; default inherits asr_mtp.graph_mode (off)',
    )


def mtp_engine_config(base_config, *, mtp_enabled=False, mtp_checkpoint=None,
                      mtp_graph_mode=None):
    """Return engine kwargs with MTP loaded only when the master switch is on.

    Preserve the external-draft speculative backend and unrelated extensions.
    ON requires that backend; unsupported scheduling/cache settings fail early.
    This is a startup switch, not hot reconfiguration of a running engine.
    """
    if type(mtp_enabled) is not bool:
        raise TypeError('mtp_enabled must be a bool')
    if mtp_graph_mode is not None and mtp_graph_mode not in ('off', 'full'):
        raise ValueError('mtp_graph_mode must be off or full')
    result = dict(base_config)
    additional = result.get('additional_config')
    if additional is not None and not isinstance(additional, Mapping):
        raise TypeError('additional_config must be a mapping or None')
    additional = dict(additional or {})
    if not mtp_enabled:
        additional.pop('asr_mtp', None)
        if 'additional_config' in result:
            result['additional_config'] = additional
        return result

    spec = result.get('speculative_config')
    get_spec = spec.get if isinstance(spec, Mapping) else lambda key: getattr(spec, key, None)
    if get_spec('method') != 'custom_class' or get_spec('model') != EXTERNAL_DRAFT_CLASS:
        raise ValueError('MTP requires the existing external_draft custom_class backend')
    settings = additional.get('asr_mtp')
    if settings is not None and not isinstance(settings, Mapping):
        raise TypeError('additional_config.asr_mtp must be a mapping')
    settings = dict(settings or {})
    if mtp_graph_mode is not None:
        settings['graph_mode'] = mtp_graph_mode
    if mtp_checkpoint is not None:
        settings['checkpoint'] = mtp_checkpoint
    checkpoint = settings.get('checkpoint')
    if checkpoint is None:
        raise ValueError('--mtp requires --mtp-checkpoint or additional_config.asr_mtp.checkpoint')
    checkpoint = fspath(checkpoint)
    if not isinstance(checkpoint, str) or not checkpoint.strip():
        raise ValueError('MTP checkpoint must be a nonempty local path')
    settings['checkpoint'] = checkpoint
    for key in ('async_scheduling', 'enable_prefix_caching'):
        if result.get(key) is not None and result[key] is not False:
            raise ValueError(f'online ASR MTP requires {key}=False')
        result[key] = False
    additional['asr_mtp'] = settings
    result['additional_config'] = additional
    return result
