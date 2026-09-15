#!/usr/bin/env python3
"""Generate reviewed Hub/CLI contracts; only --capture imports the Python oracle."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMMIT = '0123456789abcdef0123456789abcdef01234567'
OTHER_COMMIT = '89abcdef0123456789abcdef0123456789abcdef'
REPO = 'contract/tiny-llama'
SIDECARS = ['generation_config.json', 'tokenizer_config.json']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def serialized(value):
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n'


def jsonl(records):
    return ''.join(json.dumps(record, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n'
                   for record in records)


def sources():
    names = ['generate_contracts.py', 'test_contracts.py', 'manifest.json', 'requirements.lock',
             'fixtures/llama-base/inputs.json', 'fixtures/llama-base/expectations.json']
    for fixture in ['llama-base', 'llama-sharded']:
        names.extend(str(p.relative_to(ROOT)) for p in (ROOT / 'fixtures' / fixture).iterdir()
                     if p.name in ['config.json', 'tokenizer.json', *SIDECARS,
                                   'model.safetensors.index.json'] or p.suffix == '.safetensors')
    return {name: digest(ROOT / name) for name in sorted(set(names))}


def provenance(kind):
    return {'kind': kind, 'design': 't4/position-astra.md sections 2, 7.4; DECISIONS A-M',
            'base_revision': '2ef763838fe0f1a4c5b52f3c9688249092a2d57b', 'sources': sources()}


def hub_contract():
    cases = []

    def case(name, offline=False, sharded=False, absent=False, revision='main'):
        fixture = 'llama-sharded' if sharded else 'llama-base'
        directory = ROOT / 'fixtures' / fixture
        weights = (sorted(set(read(directory / 'model.safetensors.index.json')['weight_map'].values()))
                   if sharded else ['model.safetensors'])
        selected = sorted(['config.json', 'tokenizer.json', *weights]
                          + ([] if absent else SIDECARS)
                          + (['model.safetensors.index.json'] if sharded else []))
        absences = (SIDECARS[:] if absent else []) + ([] if sharded else ['model.safetensors.index.json'])
        siblings = sorted(selected + ['README.md', 'custom.py', 'pytorch_model.bin', 'chat_template.jinja'])
        order = (['model.safetensors.index.json'] if sharded else [])
        order += [f for f in selected if f not in order]
        calls = [{'operation': 'info', 'repo': REPO, 'revision': revision}]
        calls += [{'operation': 'download', 'repo': REPO, 'commit': COMMIT, 'filename': f} for f in order]
        receipt = {'schema_version': 1, 'repo': REPO, 'commit': COMMIT,
                   'weight_mode': 'indexed' if sharded else 'single',
                   'selected_files': [{'path': f, 'size': (directory / f).stat().st_size,
                                       'sha256': digest(directory / f)} for f in selected],
                   'recorded_absences': sorted(absences)}
        result = {'name': name, 'request': {'repo': REPO, 'revision': revision, 'offline': offline},
                  'fixture': fixture, 'returned_commit': COMMIT, 'siblings': siblings,
                  'selected_files': selected, 'recorded_absences': sorted(absences),
                  'setup': {'receipt': receipt if offline else None,
                            'refs': {} if revision == COMMIT else {revision: COMMIT},
                            'file_layout': 'same_repository_blob_links', 'mutations': []},
                  'expected': {'transport_calls': [] if offline else calls,
                               'client_constructions': 0 if offline else 1,
                               'selected_files': selected, 'recorded_absences': sorted(absences),
                               'receipt': receipt, 'error': None,
                               'provenance': {'repo': REPO, 'requested_revision': revision,
                                              'resolved_revision': COMMIT}}}
        cases.append(result)
        return result

    def reject(c, variant, **fields):
        c['expected']['error'] = {'variant': variant, 'fields': fields}
        c['expected']['provenance'] = None
        if not c['request']['offline']:
            c['expected']['receipt'] = copy.deepcopy(c['setup']['receipt'])

    case('online_single_snapshot')
    case('online_indexed_snapshot', sharded=True)
    c = case('moving_ref_pinned_after_info')
    c['setup']['mutations'] = [{'at': 'after_info', 'operation': 'move_ref', 'commit': OTHER_COMMIT}]
    c = case('server_returns_different_commit')
    c['setup']['mutations'] = [{'at': 'download:config.json', 'operation': 'return_commit', 'commit': OTHER_COMMIT}]
    c['expected']['transport_calls'] = c['expected']['transport_calls'][:2]
    reject(c, 'HubError::RevisionMismatch', expected=COMMIT, actual=OTHER_COMMIT)
    c = case('requested_sha_not_honored', revision=COMMIT)
    c['returned_commit'] = OTHER_COMMIT
    c['expected']['transport_calls'] = c['expected']['transport_calls'][:1]
    reject(c, 'HubError::RevisionMismatch', expected=COMMIT, actual=OTHER_COMMIT)
    for name, operation, variant in [
        ('wrong_repository_blob', 'link_other_repository_blob', 'HubError::UnsafePath'),
        ('snapshot_directory_escape', 'link_snapshot_directory_outside_cache', 'HubError::UnsafePath'),
        ('wrong_snapshot_file', 'link_other_snapshot_file', 'HubError::UnsafePath'),
        ('missing_shard_link', 'remove_file', 'HubError::MissingFile'),
        ('broken_shard_link', 'break_blob_link', 'HubError::SnapshotIntegrity'),
    ]:
        c = case(name, offline=True, sharded=True)
        filename = 'model-00002-of-00002.safetensors'
        c['setup']['mutations'] = [{'at': 'before_validation', 'operation': operation, 'path': filename}]
        fields = ({'repo': REPO, 'revision': COMMIT, 'filename': filename} if variant.endswith('MissingFile')
                  else {'path': 'snapshot/' + filename})
        reject(c, variant, **fields)
    for name, index, variant, fields in [
        ('duplicate_index_entry', '{"weight_map":{"x":"a.safetensors","x":"b.safetensors"}}',
         'HubError::Load::Weights::ConflictingIndex', {'key': 'x', 'shard': 'b.safetensors'}),
        ('traversing_index_entry', '{"weight_map":{"x":"../escape.safetensors"}}',
         'HubError::Load::Weights::ConflictingIndex', {'key': 'x', 'shard': '../escape.safetensors'}),
    ]:
        c = case(name, sharded=True)
        c['setup']['mutations'] = [{'at': 'download:model.safetensors.index.json', 'operation': 'replace_bytes', 'utf8': index}]
        c['expected']['transport_calls'] = c['expected']['transport_calls'][:2]
        c['selected_files'] = c['expected']['selected_files'] = ['model.safetensors.index.json']
        reject(c, variant, **fields)
    for sidecar in SIDECARS + ['model.safetensors.index.json']:
        c = case('recorded_absence_appears_' + sidecar, offline=True, absent=True)
        c['setup']['mutations'] = [{'at': 'before_validation', 'operation': 'copy_fixture_file',
                                   'path': sidecar, 'fixture': 'llama-sharded' if 'index' in sidecar else 'llama-base'}]
        reject(c, 'HubError::SnapshotIntegrity', path='snapshot/' + sidecar)
    c = case('hash_changed', offline=True)
    c['setup']['mutations'] = [{'at': 'before_validation', 'operation': 'flip_byte', 'path': 'model.safetensors', 'offset': 128}]
    reject(c, 'HubError::SnapshotIntegrity', path='snapshot/model.safetensors')
    c = case('unreceipted_cache', offline=True)
    c['setup']['receipt'] = None
    reject(c, 'HubError::OfflineCacheMiss', repo=REPO, revision='main')
    case('public_offline_sha_without_refs_entry', offline=True, revision=COMMIT)
    case('public_offline_named_ref', offline=True)
    c = case('offline_missing_ref', offline=True)
    c['setup']['refs'] = {}
    reject(c, 'HubError::OfflineCacheMiss', repo=REPO, revision='main')
    for phase in ['info', 'download:config.json']:
        c = case('online_failure_no_fallback_' + phase.replace(':', '_'))
        c['setup']['receipt'] = copy.deepcopy(c['expected']['receipt'])
        c['setup']['mutations'] = [{'at': phase, 'operation': 'transport_error'}]
        c['expected']['transport_calls'] = c['expected']['transport_calls'][:1 if phase == 'info' else 2]
        reject(c, 'HubError::Api')
    c = case('public_offline_local_load_error', offline=True)
    c['setup']['mutations'] = [{'at': 'before_receipt', 'operation': 'set_json', 'path': 'config.json',
                               'key': 'model_type', 'value': 'unsupported-contract-model'}]
    c['setup']['receipt'] = receipt_after_config_mutation(c)
    c['expected']['receipt'] = c['setup']['receipt']
    reject(c, 'HubError::Load::Config::UnsupportedArchitecture', value='unsupported-contract-model')
    return {'schema_version': 1, 'cohort': 'hub_contract_v1',
            'provenance': provenance('reviewed_rust_product_contract'),
            'path_symbols': {'snapshot': 'cache/models--contract--tiny-llama/snapshots/' + COMMIT,
                             'blobs': 'cache/models--contract--tiny-llama/blobs'},
            'client_contract': {'anonymous': True, 'authorization_header': None,
                                'offline_client_constructions': 0,
                                'token_file_discovery_claim': False}, 'cases': cases}


def receipt_after_config_mutation(case):
    receipt = copy.deepcopy(case['setup']['receipt'])
    config = read(ROOT / 'fixtures' / case['fixture'] / 'config.json')
    config['model_type'] = 'unsupported-contract-model'
    raw = serialized(config).encode('utf-8')
    for entry in receipt['selected_files']:
        if entry['path'] == 'config.json':
            entry.update(size=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    return receipt


def progress(total, ceiling):
    pairs = [[0, total]]
    done = 0
    while done < total - 1:
        done += min(ceiling, total - 1 - done)
        pairs.append([done, total])
    return pairs + [[total, total]]


def prefill_records(pairs):
    return [{'version': 1, 'event': 'prefill', 'processed': p, 'total': t} for p, t in pairs]


def info_record():
    directory = ROOT / 'fixtures/llama-base'
    config = read(directory / 'config.json')
    resolved = read(directory / 'expectations.json')['config']['resolved']
    dims = {name: resolved[name] for name in ['hidden_size', 'layer_count', 'intermediate_size',
            'attention_heads', 'kv_heads', 'head_dim', 'vocabulary_size', 'rms_norm_epsilon']}
    dims['max_positions'] = config['max_position_embeddings']
    return {'version': 1, 'config': {'model_type': 'llama', 'dimensions': dims,
            'rope': {**resolved['rope'], 'scaling': {'kind': 'none'}},
            'attention': [{'kind': k} for k in resolved['attention_kinds']],
            'tie_word_embeddings': resolved['tie_word_embeddings'], 'attention_bias': False,
            'mlp_bias': False, 'quantization': None},
            'tokenizer': {'bos_token': '<|begin_of_text|>', 'eos_tokens': [2, 3]}, 'hub_provenance': None}


def output(fmt, records=None, text='', info=None, exit_code=0, stderr='empty'):
    result = {'format': fmt, 'exit_code': exit_code, 'stderr_rule': stderr}
    if fmt == 'jsonl':
        result.update(records=records, stdout_utf8=jsonl(records))
    elif fmt == 'json':
        result.update(info=info, stdout_utf8=json.dumps(info, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n')
    else:
        result['stdout_utf8'] = text
    return result


def cli_contract(capture=None):
    fixture = ROOT / 'fixtures/llama-base'
    inputs = read(fixture / 'inputs.json')
    expectations = read(fixture / 'expectations.json')
    for entry in expectations['prefill']['progress'].values():
        if progress(len(entry['token_ids']), entry['ceiling']) != entry['pairs']:
            raise ValueError('committed progress golden disagrees with schedule')
    prompt = inputs['prompts']['canonical']
    ids = expectations['tokenizer']['prompt_encoding']['canonical']
    prefill = prefill_records(progress(len(ids), 2048))
    events = [*prefill,
              {'version': 1, 'event': 'token', 'token_id': 12, 'text': '', 'finish_reason': None},
              {'version': 1, 'event': 'token', 'token_id': 13, 'text': 'hello', 'finish_reason': None},
              {'version': 1, 'event': 'token', 'token_id': 2, 'text': '', 'finish_reason': 'stop'}]
    argv = ['generate', '--model', '{fixture}', '--prompt', prompt, '--max-tokens', '8']
    failures = []
    def syntax(name, args, feature='any'):
        failures.append({'name': name, 'argv': args, 'feature': feature, 'before_load': True,
                         'expected': output('text', exit_code=2, stderr='nonempty')})
    for flag in ['--max-tokens', '--top-k', '--min-tokens-to-keep']:
        syntax('zero_' + flag[2:], argv[:5] + [flag, '0'] + (['--min-p', '0.1'] if flag == '--min-tokens-to-keep' else []))
    for name, tail in [('negative_temperature', ['--temperature', '-1']),
                       ('nan_temperature', ['--temperature', 'NaN']),
                       ('infinite_temperature', ['--temperature', 'inf']),
                       ('top_p_out_of_range', ['--top-p', '1.1']),
                       ('min_p_out_of_range', ['--min-p', '-0.1']),
                       ('minimum_without_min_p', ['--min-tokens-to-keep', '1']),
                       ('empty_stop', ['--stop', '']), ('invalid_format', ['--format', 'yaml'])]:
        syntax(name, argv + tail)
    syntax('missing_source', ['info'])
    syntax('missing_prompt', ['generate', '--model', '{fixture}'])
    syntax('mutually_exclusive_sources', argv + ['--repo', REPO], 'hf-hub')
    for flag, values in [('--repo', [REPO]), ('--revision', ['main']), ('--offline', []), ('--cache-dir', ['cache'])]:
        syntax('disabled_' + flag[2:], argv + [flag] + values, 'no-hf-hub')
        if flag != '--repo':
            syntax('local_' + flag[2:], argv + [flag] + values, 'hf-hub')
    generation = {'name': 'llama_canonical_public_text', 'fixture': 'llama-base', 'argv': argv,
                  'prompt_token_ids': ids, 'prefill_records': prefill,
                  'capture_status': 'not_run', 'capture': None, 'expected': None}
    if capture is not None:
        if capture['prompt'] != prompt or capture['prompt_token_ids'] != ids:
            raise ValueError('capture prompt/BOS mismatch')
        responses = capture['responses']
        if not responses or len(responses) > 8 or sum(r['finish_reason'] is not None for r in responses) != 1:
            raise ValueError('capture must contain one terminal response')
        if responses[-1]['finish_reason'] not in ['stop', 'length']:
            raise ValueError('capture finish reason')
        tokens = [{'version': 1, 'event': 'token', 'token_id': r['token_id'],
                   'text': r['text'], 'finish_reason': r['finish_reason']} for r in responses]
        generation.update(capture_status='captured', capture=capture,
                          expected={'text': output('text', text=''.join(r['text'] for r in responses), stderr='diagnostics_only'),
                                    'jsonl': output('jsonl', records=prefill + tokens, stderr='diagnostics_only')})
    return {'schema_version': 1, 'cohort': 'cli_contract_v1',
            'provenance': provenance('reviewed_wire_contract_with_separate_python_text_capture'),
            'generation': generation, 'info': {'fixture': 'llama-base', 'argv': ['info', '--model', '{fixture}', '--format', 'json'],
                                            'expected': output('json', info=info_record())},
            'writer_cases': [{'name': 'empty_token_and_stop', 'source': 'scripted_private_OutputEvent',
                              'events': events, 'expected': output('jsonl', records=events)},
                             {'name': 'text_has_no_added_newline', 'source': 'scripted_private_OutputEvent',
                              'events': events, 'expected': output('text', text='hello')}],
            'argument_failures': failures,
            'exit_rules': {'syntax': 2, 'load': 1, 'generation': 1, 'output': 1,
                           'unknown_event_or_finish': 1, 'broken_pipe': 0},
            'writer_faults': [
                {'name': 'short_write', 'write_script': [{'accept_bytes': 1}, {'accept_remaining': True}], 'expected': 'complete_exact_bytes'},
                {'name': 'interrupted', 'write_script': [{'error_kind': 'Interrupted'}, {'accept_remaining': True}], 'expected': 'complete_exact_bytes'},
                {'name': 'ordinary_io_failure', 'write_script': [{'error_kind': 'Other'}], 'expected_exit': 1},
                {'name': 'broken_pipe', 'write_script': [{'error_kind': 'BrokenPipe'}], 'expected_exit': 0, 'drop_generation': True}],
            'stderr_rules': {'empty': 'zero bytes', 'nonempty': 'nonempty UTF-8; source chain for runtime failures',
                             'diagnostics_only': 'optional UTF-8 progress/completion diagnostics; never stdout; bytes not frozen'}}


def compare_hub(expected, actual):
    for key, cls in [('transport_calls', 'hub_transport'), ('client_constructions', 'hub_transport'),
                     ('error', 'error_class'), ('provenance', 'hub_provenance'),
                     ('selected_files', 'hub_selection'), ('recorded_absences', 'hub_integrity'),
                     ('receipt', 'hub_integrity')]:
        if expected.get(key) != actual.get(key):
            return cls
    return None


def compare_cli(expected, actual):
    if expected is None:
        raise ValueError('Python public Text capture has not run')
    if actual.get('exit_code') != expected['exit_code']:
        return 'exit_code'
    stdout = actual.get('stdout_utf8', '')
    if expected['format'] == 'jsonl':
        try:
            records = [json.loads(line) for line in stdout.splitlines()]
        except (ValueError, TypeError):
            return 'stdout'
        if len(records) != len(expected['records']):
            return 'output_count'
        for wanted, got in zip(expected['records'], records):
            if not isinstance(got, dict):
                return 'stdout'
            if wanted.get('finish_reason') != got.get('finish_reason'):
                return 'finish_reason'
            if wanted.get('token_id') != got.get('token_id'):
                return 'sampled_id'
            if wanted.get('text') != got.get('text'):
                return 'text_delta'
            if wanted != got:
                return 'progress' if wanted['event'] == 'prefill' else 'stdout'
    elif expected['format'] == 'json':
        try:
            if json.loads(stdout) != expected['info']:
                return 'config'
        except (ValueError, TypeError):
            return 'stdout'
    if stdout != expected['stdout_utf8']:
        return 'stdout'
    stderr = actual.get('stderr_utf8', '')
    if expected['stderr_rule'] == 'empty' and stderr:
        return 'stderr'
    if expected['stderr_rule'] == 'nonempty' and not stderr:
        return 'stderr'
    return None


def qualify(hub, cli):
    killed = []
    def check(name, comparator, expected, actual, cls):
        found = comparator(expected, actual)
        if found != cls:
            raise ValueError(f'{name}: expected {cls}, got {found}')
        killed.append({'name': name, 'class': cls})
    for case in hub['cases']:
        expected = case['expected']
        if compare_hub(expected, copy.deepcopy(expected)) is not None:
            raise ValueError('hub comparator rejects identity')
        actual = copy.deepcopy(expected)
        if case['request']['offline']:
            actual['transport_calls'].append({'operation': 'construct_client'})
            cls = 'hub_transport'
        elif expected['error'] is not None:
            actual['error'] = None
            cls = 'error_class'
        else:
            actual['transport_calls'][-1]['commit'] = OTHER_COMMIT
            cls = 'hub_transport'
        check(case['name'], compare_hub, expected, actual, cls)
        if expected['error'] is not None and case['request']['offline']:
            actual = copy.deepcopy(expected)
            actual['error'] = None
            check(case['name'] + '_accepted_corruption', compare_hub, expected, actual, 'error_class')
        if case['request']['offline']:
            actual = copy.deepcopy(expected)
            actual['client_constructions'] = 1
            check(case['name'] + '_client_constructed', compare_hub, expected, actual, 'hub_transport')
    expected = cli['writer_cases'][0]['expected']
    def observed(expected):
        return {**copy.deepcopy(expected), 'stderr_utf8': 'error\n' if expected['stderr_rule'] == 'nonempty' else ''}
    for item in cli['writer_cases'] + [cli['info']] + cli['argument_failures']:
        if compare_cli(item['expected'], observed(item['expected'])) is not None:
            raise ValueError('cli comparator rejects identity')
    for name, cls in [('drop_empty_token', 'output_count'), ('change_finish_reason', 'finish_reason'),
                      ('change_token_id', 'sampled_id'), ('diagnostics_on_stdout', 'stdout')]:
        actual = observed(expected)
        records = copy.deepcopy(expected['records'])
        if name == 'drop_empty_token':
            del records[3]
        elif name == 'change_finish_reason':
            records[-1]['finish_reason'] = 'length'
        elif name == 'change_token_id':
            records[-1]['token_id'] += 1
        actual['stdout_utf8'] = jsonl(records)
        if name == 'diagnostics_on_stdout':
            actual['stdout_utf8'] = 'prefill complete\n' + actual['stdout_utf8']
        check(name, compare_cli, expected, actual, cls)
    expected = cli['writer_cases'][1]['expected']
    actual = observed(expected)
    actual['stdout_utf8'] += '\n'
    check('extra_text_newline', compare_cli, expected, actual, 'stdout')
    expected = cli['info']['expected']
    actual = observed(expected)
    info = copy.deepcopy(expected['info'])
    info['config']['dimensions']['hidden_size'] += 1
    actual['stdout_utf8'] = json.dumps(info)
    check('changed_info_dimension', compare_cli, expected, actual, 'config')
    expected = cli['argument_failures'][0]['expected']
    actual = observed(expected)
    actual['exit_code'] = 0
    check('accept_zero_positive_flag', compare_cli, expected, actual, 'exit_code')
    for key, cls in [('provenance', 'hub_provenance'), ('selected_files', 'hub_selection'),
                     ('recorded_absences', 'hub_integrity'), ('receipt', 'hub_integrity')]:
        expected = hub['cases'][0]['expected']
        actual = copy.deepcopy(expected)
        actual[key] = None
        check('changed_' + key, compare_hub, expected, actual, cls)
    return killed


def capture_text():
    from manifest import check_environment
    environment = check_environment()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    import contextlib
    import importlib
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    module = importlib.import_module('mlx_lm.generate')
    # The oracle already suppresses the device-specific memory-limit advisory this way.
    module.wired_limit = lambda model, streams=None: contextlib.nullcontext()
    mx.set_default_device(mx.cpu)
    fixture = ROOT / 'fixtures/llama-base'
    prompt = read(fixture / 'inputs.json')['prompts']['canonical']
    model, tokenizer = load(str(fixture))
    ids = tokenizer.encode(prompt, add_special_tokens=not (tokenizer.bos_token and prompt.startswith(tokenizer.bos_token)))
    responses = list(stream_generate(model, tokenizer, prompt, max_tokens=8, sampler=make_sampler(temp=0.0), prefill_step_size=2048))
    return {'entry_point': 'mlx_lm.stream_generate(model, tokenizer, prompt: str)',
            'environment': environment, 'prompt': prompt, 'prompt_token_ids': ids,
            'upstream_sources': {str(p.relative_to(Path(module.__file__).parent)): digest(p)
                                 for p in [Path(module.__file__), Path(module.__file__).with_name('tokenizer_utils.py'),
                                           Path(module.__file__).with_name('utils.py')]},
            'responses': [{'token_id': int(r.token), 'text': r.text, 'finish_reason': r.finish_reason} for r in responses]}


def freeze(generated, repeat):
    names = ['hub_cases.json', 'cli_cases.json']
    for name in names:
        if (generated / name).read_bytes() != (repeat / name).read_bytes():
            raise ValueError(name + ': repeated generation differs')
    hub, cli = (read(generated / name) for name in names)
    if serialized(hub) != serialized(hub_contract()) or serialized(cli) != serialized(cli_contract(cli['generation']['capture'])):
        raise ValueError('generated contracts/source digests differ from reviewed generator')
    kills = qualify(hub, cli)
    corpus = read(ROOT / 'corpus.json')
    for name, sha in corpus['files'].items():
        if name not in names and digest(ROOT / name) != sha:
            raise ValueError(name + ': existing corpus drift')
    for name in names:
        (ROOT / name).write_bytes((generated / name).read_bytes())
        corpus['files'][name] = digest(ROOT / name)
    corpus.setdefault('cohorts', {}).update({
        'hub_contract_v1': {'file': names[0], 'cases': len(hub['cases']), 'evidence': 'reviewed_rust_product_contract'},
        'cli_contract_v1': {'file': names[1], 'writer_cases': len(cli['writer_cases']),
                            'argument_failures': len(cli['argument_failures']), 'info_cases': 1,
                            'public_text_captures': int(cli['generation']['capture_status'] == 'captured'),
                            'evidence': 'wire_contract; public Text capture status is explicit'},
    })
    corpus['contract_comparator_qualification'] = kills
    (ROOT / 'corpus.json').write_text(serialized(corpus), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--capture', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--qualify', action='store_true')
    parser.add_argument('--require-capture', action='store_true')
    parser.add_argument('--freeze-from', type=Path)
    parser.add_argument('--repeat', type=Path)
    args = parser.parse_args()
    if args.freeze_from is not None:
        if args.repeat is None or args.capture or args.output_dir or args.check:
            parser.error('--freeze-from requires --repeat and excludes --capture, --output-dir and --check')
        if args.require_capture and read(args.freeze_from / 'cli_cases.json')['generation']['capture_status'] != 'captured':
            raise SystemExit('Python public Text capture has not run')
        freeze(args.freeze_from, args.repeat)
        return
    if args.repeat is not None:
        parser.error('--repeat requires --freeze-from')
    if args.capture and args.output_dir is None:
        parser.error('--capture requires an explicit --output-dir')
    capture = capture_text() if args.capture else None
    if not args.capture and (ROOT / 'cli_cases.json').exists():
        capture = read(ROOT / 'cli_cases.json')['generation']['capture']
    hub, cli = hub_contract(), cli_contract(capture)
    if args.qualify:
        for result in qualify(hub, cli):
            print('KILLED', result['name'], result['class'])
    if args.check:
        for name, doc in [('hub_cases.json', hub), ('cli_cases.json', cli)]:
            if (ROOT / name).read_text(encoding='utf-8') != serialized(doc):
                raise SystemExit(name + ': contract/source digest drift')
        corpus = read(ROOT / 'corpus.json')
        for name, sha in corpus['files'].items():
            if digest(ROOT / name) != sha:
                raise SystemExit(name + ': corpus digest drift')
        print('PASS contracts and corpus digests')
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, doc in [('hub_cases.json', hub), ('cli_cases.json', cli)]:
            (args.output_dir / name).write_text(serialized(doc), encoding='utf-8')
    if cli['generation']['capture_status'] != 'captured':
        print('NOT RUN llama_canonical_public_text: requires pinned Python MLX on host')
        if args.require_capture:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
