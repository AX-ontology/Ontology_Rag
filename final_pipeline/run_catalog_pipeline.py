#!/usr/bin/env python3
"""Config-driven existing-first TaskFrame pipeline; no DB writes by default."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

from combine_graphs import combine_inputs, read_jsonl, write_json, write_jsonl
from run_final_pipeline import (HERE, load_config, config_path, resolve_path,
    document_paths, taskframe_command, build_pcf_config, write_pcf_config, clean_command, run_logged)
from run_flat_pipeline import materialize, merge_catalog

VERSION = 'catalog-pipeline-3.0-direct'
PHASES = ['catalog', 'graph', 'pcf', 'clean', 'validate', 'db']


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest_paths(paths):
    """Content hashes, excluding caches/logs/bytecode; missing inputs are errors."""
    h = hashlib.sha256()
    for raw in paths:
        p = Path(raw)
        if not p.exists(): raise FileNotFoundError(p)
        files = sorted(p.rglob('*')) if p.is_dir() else [p]
        for f in files:
            rel = f.relative_to(p) if p.is_dir() else Path(f.name)
            if any(x.startswith('.') or x in {'__pycache__', 'logs'} for x in rel.parts): continue
            if not f.is_file() or f.suffix in {'.pyc', '.log'}: continue
            h.update(str(rel).encode())
            with f.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


class StageStore:
    """Changed inputs get a new artifact directory, never overwrite an old run."""
    def __init__(self, root):
        self.root = Path(root)
        self.path = self.root / 'catalog_pipeline_manifest.json'
        self.state = read_json(self.path) if self.path.exists() else {'stages': {}, 'db_loads': {}}
        self.state.update(version=VERSION, status='running')

    def save(self):
        write_json(self.path, self.state)

    def run(self, name, inputs, params, action):
        implementation = digest_paths([Path(__file__), HERE / 'run_final_pipeline.py'])
        signature = hashlib.sha256(json.dumps([VERSION, implementation, digest_paths(inputs), params], sort_keys=True).encode()).hexdigest()
        out = self.root / 'artifacts' / f'{name}-{signature[:16]}'
        marker = out / '.complete.json'
        self.state.update(current_stage=name, status='running')
        self.state.pop('error', None)
        self.save()
        if marker.exists():
            info = read_json(marker)
            if info['signature'] != signature or info['output_digest'] != digest_paths([out]):
                raise ValueError(f'Completed output changed or damaged: {out}; preserve it and use a new run name')
            print(f'[reuse] {name}: {out.name}', flush=True)
        else:
            out.mkdir(parents=True, exist_ok=True)
            print(f'[stage] {name}: {out.name}', flush=True)
            action(out)
            write_json(marker, {'signature': signature, 'output_digest': digest_paths([out])})
        self.state['stages'][name] = {'signature': signature, 'path': str(out), 'status': 'complete'}
        self.save()
        return out


def extend_registry(catalog, frames):
    result = deepcopy(catalog)
    for frame in frames:
        tid = frame['taskframe_id']
        if tid in result:
            if result[tid]['canonical_name'] != frame['canonical_name']:
                raise ValueError(f'Approved task identity changed: {tid}')
            if result[tid].get('definition') != frame.get('definition'):
                raise ValueError(f'Approved catalog definition changed: {tid}; review/re-align before reuse')
            # Keep approved definitions; add alternate expressions only.
            result[tid]['aliases'] = sorted(set(result[tid].get('aliases', [])) | set(frame.get('aliases', [])))
        else:
            result[tid] = deepcopy(frame)
    return result


def validate_approved_bundle(source, bundle, catalog):
    report = read_json(bundle / 'validation_report.json')
    if report.get('status') != 'complete': raise ValueError('Approved bundle is incomplete')
    raw = read_json(source / '05_taskframes/taskframes.json')
    mapped = read_jsonl(bundle / 'source_to_target.jsonl')
    if Counter(x['source_id'] for x in mapped) != Counter(f['taskframe_id'] for f in raw):
        raise ValueError('Approved bundle does not cover the current source')
    frames = read_json(bundle / 'taskframes.json')
    if set(x['target_id'] for x in mapped) != {f['taskframe_id'] for f in frames}:
        raise ValueError('Approved mapping targets differ from bundle frames')
    return extend_registry(catalog, frames)


def reusable_source(config, doc):
    if not doc.get('reuse_taskkg'): return None
    expected = doc.get('reuse_input_sha256')
    if not expected: raise ValueError('reuse_taskkg requires reuse_input_sha256 for input change detection')
    actual = hashlib.sha256(resolve_path(config, doc['input_xlsx']).read_bytes()).hexdigest()
    if actual != expected:
        if doc.get('catalog_role') == 'seed':
            raise ValueError('Seed input changed: explicitly review/update the baseline catalog before reuse')
        return None
    return resolve_path(config, doc['reuse_taskkg'])


def api_parameters(config):
    values = {}
    env_path = config_path(config, 'taskframe_env')
    allowed = {'TASKFRAME_API_BASE', 'TASKFRAME_MODEL', 'TASKFRAME_THINKING',
               'TASKFRAME_MAX_TOKENS', 'TASKFRAME_MAX_TOKENS_LIMIT'}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            key, _, value = line.partition('=')
            if key.strip() in allowed: values[key.strip()] = value.strip().strip('"\'')
    values.update({k: os.environ[k] for k in allowed if k in os.environ})
    return values


def merge_inputs(documents, output):
    """Union document evidence before one global PCF/cleaning pass."""
    output = Path(output)
    merge_catalog(documents, output)
    files = {
        '02_extraction/chunk_extractions.jsonl': 'chunk_id',
        '03_entity_prep/task_mentions.jsonl': 'task_mention_id',
        '03_entity_prep/open_triples.jsonl': 'triple_id',
        '07_canonicalization/canonical_facts.jsonl': 'triple_id',
    }
    for filename, key in files.items():
        result, seen = [], set()
        for _, src in documents:
            for row in read_jsonl(src / filename):
                if row[key] in seen: raise ValueError(f'Duplicate cross-document {key}: {row[key]}')
                seen.add(row[key]); result.append(row)
        write_jsonl(output / filename, result)
    summaries = [read_json(src / '02_extraction/summary.json') for _, src in documents]
    summary = {k: sum(x.get(k, 0) for x in summaries) for k in
               ['task_mention_count', 'open_triple_count', 'schema_error_count', 'exact_quote_grounded_count']}
    write_json(output / '02_extraction/summary.json', summary)
    combined = combine_inputs([(doc, src / '09_kg') for doc, src in documents], output / 'raw_graph')
    (output / '09_kg').mkdir(exist_ok=True)
    for filename in ['nodes.jsonl', 'edges.jsonl']:
        shutil.copy2(output / 'raw_graph/neo4j' / filename, output / '09_kg' / filename)
    write_json(output / '09_kg/validation_report.json', combined)


def load_and_verify(config, graph, run_id, log):
    """Explicitly requested DB action, idempotent by IDs, independently verified."""
    command = [sys.executable, str(config_path(config, 'cleaning_dir') / 'load_neo4j.py'),
        '--graph-dir', str(graph), '--env-file', str(config_path(config, 'neo4j_env')),
        '--run-id', run_id, '--pipeline-version', VERSION,
        '--batch-size', str(config.get('neo4j', {}).get('batch_size', 500))]
    database = config.get('neo4j', {}).get('database')
    if database: command += ['--database', database]
    run_logged(command, HERE, log, dry_run=False)
    sys.path.insert(0, str(config_path(config, 'cleaning_dir')))
    import load_neo4j as loader
    loader.load_env_file(config_path(config, 'neo4j_env'), override=True)
    client = loader.make_client(
        os.getenv('NEO4J_QUERY_URI') or os.getenv('NEO4J_URI') or os.getenv('AURA_URI'),
        os.getenv('NEO4J_USERNAME') or os.getenv('NEO4J_USER') or os.getenv('AURA_USER'),
        os.getenv('NEO4J_PASSWORD') or os.getenv('AURA_PASS'), database or os.getenv('NEO4J_DATABASE'))
    def rows(query):
        result = client.run(query, {'run': run_id})
        if isinstance(result, list): return result
        return [dict(zip(result['data']['fields'], row)) for row in result['data']['values']]
    try:
        actual_nodes = rows('MATCH (n:KgNode) WHERE n.lastRunId=$run RETURN n.id AS id,n.type AS label')
        actual_edges = rows('MATCH (s)-[r]->(t) WHERE r.lastRunId=$run RETURN r.id AS id,type(r) AS type,s.id AS source,t.id AS target')
        expected_nodes = read_jsonl(graph / 'nodes.jsonl'); expected_edges = read_jsonl(graph / 'edges.jsonl')
        checks = {
            'nodes_match': Counter((x['id'], x['label']) for x in actual_nodes) == Counter((x['id'], x['label']) for x in expected_nodes),
            'edges_match': Counter((x['id'], x['type'], x['source'], x['target']) for x in actual_edges) == Counter((x['id'], x['type'], x['source'], x['target']) for x in expected_edges),
        }
        if not all(checks.values()): raise ValueError(f'DB verification failed: {checks}')
        return {'status': 'verified', 'run_id': run_id, 'verified_at': datetime.now(timezone.utc).isoformat(),
                'checks': checks, 'node_count': len(actual_nodes), 'edge_count': len(actual_edges)}
    finally:
        client.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE / 'config.yaml')
    parser.add_argument('--documents', nargs='+')
    parser.add_argument('--run-name', default='v4_5_catalog_final')
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--stop-after', choices=PHASES, default='validate')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--load-neo4j', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_config(args.config)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', args.run_name): raise ValueError('Unsafe run name')
    settings = config.get('catalog_pipeline', {})
    if not settings.get('baseline_taskframes'): raise ValueError('catalog_pipeline.baseline_taskframes is required')
    baseline = resolve_path(config, settings['baseline_taskframes'])
    baseline_frames = read_json(baseline)
    catalog = {f['taskframe_id']: f for f in baseline_frames}
    if len(catalog) != len(baseline_frames): raise ValueError('Duplicate baseline IDs')
    selected = args.documents or list(config['documents'])
    if not selected or len(set(selected)) != len(selected) or any(s not in config['documents'] for s in selected):
        raise ValueError('Unknown or duplicate document selections')
    if any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', s) for s in selected): raise ValueError('Unsafe document slug')
    if len({config['documents'][s]['document_id'] for s in selected}) != len(selected):
        raise ValueError('Selected documents must have distinct document_id values')
    if args.stop_after == 'db' and not args.load_neo4j: raise ValueError('--stop-after db requires --load-neo4j')
    if args.load_neo4j and args.stop_after not in {'validate', 'db'}: raise ValueError('DB load requires complete validation')
    root = (args.output_root.resolve() if args.output_root else config_path(config, 'output_root')) / args.run_name
    # Do not mix this pipeline's stage records with old/manual production runs.
    if root.exists() and not (root / 'catalog_pipeline_manifest.json').exists() and any(root.iterdir()):
        raise ValueError(f'Use a separate run name; existing output preserved: {root}')
    plan = {'version': VERSION, 'baseline': str(baseline), 'documents': selected, 'output': str(root),
            'flow': ['extract_to_entity_prep', 'direct_mention_assignment_with_fact_links',
                     'novel_only_grouping_if_needed', 'catalog_update',
                     'graph', 'global_pcf', 'global_clean', 'validate'],
            'llm_stage_policy': {'required': ['extraction', 'direct_assignment_and_fact_links'],
                                 'conditional': ['novel_grouping'], 'catalog_review': False},
            'load_neo4j': args.load_neo4j}
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2)); return 0
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    store = StageStore(root)
    store.state['plan'] = plan
    def command(name, cmd): run_logged(cmd, HERE, root / 'logs' / f'{name}.log', dry_run=False)
    def finished(phase, pointers):
        store.state.update(status='complete' if phase == 'validate' else 'stopped', completed_phase=phase,
                           documents=selected, current=pointers)
        store.save(); write_json(root / 'current.json', pointers)
        print(json.dumps({'status': store.state['status'], **pointers}, ensure_ascii=False, indent=2))
    source_code = [config_path(config, 'taskframe_dir') / 'taskkg', config_path(config, 'taskframe_dir') / 'run_pipeline.py',
                   config_path(config, 'taskframe_dir') / 'examples/entity_aliases.sample.json',
                   config_path(config, 'taskframe_dir') / 'examples/relation_aliases.sample.json']
    api_settings = api_parameters(config)
    prepared = []
    pointers = {}
    try:
        registry_path = baseline
        for slug in selected:
            doc = config['documents'][slug]
            reuse = reusable_source(config, doc)
            if reuse:
                source = reuse
                m = read_json(source / 'run_manifest.json')
                if m.get('status') != 'complete' or m.get('completed_stage') not in {'propagation', 'canonicalization', 'standard_mapping', 'kg'}:
                    raise ValueError(f'Incomplete source: {source}')
            else:
                if doc.get('catalog_role') == 'seed':
                    raise ValueError('Seed catalog requires an approved reusable source; review baseline before extraction')
                xlsx = resolve_path(config, doc['input_xlsx'])
                def extract(out):
                    cmd = taskframe_command(sys.executable, config, doc, document_paths(out, 'source'), 'llm', 0)
                    cmd += ['--stop-after', 'entity_prep', '--cache-dir', str(root / 'cache/extraction')]
                    command(f'{slug}_extract', cmd)
                a = store.run(f'{slug}-extract', [xlsx] + source_code, [doc, config.get('taskframe'), api_settings], extract)
                source = a / 'source/taskkg'
            evidence_docs = {e['document_id'] for e in read_jsonl(source / '01_evidence/evidence_spans.jsonl')}
            if evidence_docs != {doc['document_id']}: raise ValueError(f'Source evidence document ID mismatch: {slug}')
            approved = doc.get('approved_alignment') if reuse else None
            if approved:
                expected = doc.get('approved_source_digest')
                if not expected: raise ValueError('approved_alignment requires approved_source_digest')
                current_digest = digest_paths([source / '05_taskframes/taskframes.json', source / '06_propagation/facts_with_taskframes.jsonl'])
                if current_digest != expected:
                    print(f'[invalidate] {slug}: source frames/facts changed; re-align instead of using old approval', flush=True)
                    approved = None
            if doc.get('catalog_role') == 'seed':
                if approved: raise ValueError('Seed document cannot also specify an alignment')
                for f in read_json(source / '05_taskframes/taskframes.json'):
                    if f['taskframe_id'] not in catalog or catalog[f['taskframe_id']]['canonical_name'] != f['canonical_name']:
                        raise ValueError('Seed source must match the baseline catalog')
                bundle = None
            elif approved:
                bundle = resolve_path(config, approved)
                validate_approved_bundle(source, bundle, catalog)
            else:
                def align(out):
                    write_json(out / 'baseline.json', list(catalog.values()))
                    command(f'{slug}_assign_mentions', [sys.executable, str(HERE / 'assign_mentions.py'),
                        '--baseline', str(out / 'baseline.json'), '--source', str(source), '--out', str(out / 'taskkg'),
                        '--env', str(config_path(config, 'taskframe_env')),
                        '--batch-size', str(settings.get('assignment_batch_size', 48)),
                        '--max-facts', str(settings.get('assignment_max_facts', 160)),
                        '--context-radius', str(config.get('taskframe', {}).get('context_chunk_radius', 1))])
                a = store.run(f'{slug}-direct-assign', [source / '01_evidence', source / '02_extraction',
                    source / '03_entity_prep', source / 'run_manifest.json', registry_path,
                    HERE / 'assign_mentions.py', HERE / 'consolidate_taskframes.py'] + source_code,
                    [api_settings, settings, config.get('taskframe')], align)
                source, bundle = a / 'taskkg', None
            dependencies = [source / s for s in ['01_evidence', '02_extraction', '03_entity_prep', '04_clustering', '05_taskframes', '06_propagation']]
            if bundle: dependencies.append(bundle)
            snapshot = store.run(f'{slug}-catalog', dependencies + [HERE / 'run_flat_pipeline.py'], {},
                lambda out: materialize(source, out / 'taskkg', bundle))
            catalog = extend_registry(catalog, read_json(snapshot / 'taskkg/05_taskframes/taskframes.json'))
            registry = store.run(f'{slug}-registry', [registry_path, snapshot], {},
                lambda out: write_json(out / 'taskframes.json', list(catalog.values())))
            registry_path = registry / 'taskframes.json'
            prepared.append((slug, snapshot / 'taskkg'))
        pointers['catalog'] = str(registry_path)
        if args.stop_after == 'catalog': finished('catalog', pointers); return 0
        graphs = []
        for slug, snapshot in prepared:
            doc = config['documents'][slug]
            def build(out):
                target = out / 'source/taskkg'
                if not target.exists(): shutil.copytree(snapshot, target)
                cmd = taskframe_command(sys.executable, config, doc, document_paths(out, 'source'), 'llm', 0)
                if '--resume' not in cmd: cmd.append('--resume')
                if read_json(snapshot / '05_taskframes/validation_report.json').get('method') == 'direct_mentions_existing_first':
                    cmd.append('--catalog-assigned')
                command(f'{slug}_graph', cmd)
            a = store.run(f'{slug}-graph', [snapshot] + source_code, doc, build)
            graphs.append((doc['document_id'], a / 'source/taskkg'))
        merged = store.run('merge-evidence', [p for _, p in graphs] + [HERE / 'combine_graphs.py', Path(__file__)],
                           [d for d, _ in graphs], lambda out: merge_inputs(graphs, out / 'taskkg'))
        pointers['taskkg'] = str(merged / 'taskkg')
        if args.stop_after == 'graph': finished('graph', pointers); return 0
        def pcf(out):
            paths = document_paths(out, 'global'); paths['taskkg'] = merged / 'taskkg'
            cfg = build_pcf_config(config, paths)
            cfg['paths']['semantic_cache_dir'] = str(root / 'cache/embedding')
            write_pcf_config(out / 'pcf_config.yaml', cfg)
            command('global_pcf', [sys.executable, str(config_path(config, 'layer_connect_dir') / 'run_pipeline.py'), '--config', str(out / 'pcf_config.yaml')])
            if not read_json(paths['pcf'] / 'validation_report.json')['valid']: raise ValueError('PCF validation failed')
        pcf_code = list(config_path(config, 'layer_connect_dir').glob('*.py'))
        pcf_out = store.run('pcf', [merged, config_path(config, 'pcf'), resolve_path(config, config['pcf']['base_config'])] + pcf_code, config['pcf'], pcf)
        pointers['pcf'] = str(pcf_out / 'global/pcf')
        if args.stop_after == 'pcf': finished('pcf', pointers); return 0
        def clean(out):
            paths = document_paths(out, 'global'); paths.update(taskkg=merged / 'taskkg', pcf=pcf_out / 'global/pcf')
            command('global_clean', clean_command(sys.executable, config, paths, False))
        cleaned = store.run('clean', [merged, pcf_out, config_path(config, 'cleaning_dir') / 'clean_and_export.py',
            resolve_path(config, config['cleaning']['entity_aliases']), resolve_path(config, config['cleaning']['relation_aliases'])], config['cleaning'], clean)
        pointers['cleaned'] = str(cleaned / 'global/cleaned')
        if args.stop_after == 'clean': finished('clean', pointers); return 0
        def validate(out):
            clean_root = cleaned / 'global/cleaned'
            report = read_json(clean_root / 'validation_report.json')
            if not report['valid']: raise ValueError('Cleaned graph invalid')
            nodes = read_jsonl(clean_root / 'neo4j/nodes.jsonl')
            expectations = {
                'Fact': {r['triple_id'] for r in read_jsonl(merged / 'taskkg/03_entity_prep/open_triples.jsonl')},
                'TaskMention': {r['task_mention_id'] for r in read_jsonl(merged / 'taskkg/03_entity_prep/task_mentions.jsonl')},
                'TaskFrame': {f['taskframe_id'] for f in read_json(merged / 'taskkg/05_taskframes/taskframes.json')},
                'Document': {d for d, _ in graphs},
            }
            for label, ids in expectations.items():
                if {n['id'] for n in nodes if n['label'] == label} != ids: raise ValueError(f'{label} preservation failed')
            edges = read_jsonl(clean_root / 'neo4j/edges.jsonl')
            if any(e['type'] == 'SUBTASK_OF' for e in edges): raise ValueError('Task hierarchy is forbidden')
            command('validate_loader', [sys.executable, str(config_path(config, 'cleaning_dir') / 'load_neo4j.py'),
                '--graph-dir', str(clean_root / 'neo4j'), '--validate-only'])
            shutil.copytree(clean_root / 'neo4j', out / 'combined/neo4j', dirs_exist_ok=True)
            shutil.copy2(merged / 'taskkg/05_taskframes/taskframes.json', out / 'combined/taskframes.json')
            write_json(out / 'combined/validation_report.json', report)
        final = store.run('validate', [merged, cleaned, config_path(config, 'cleaning_dir') / 'load_neo4j.py'], {}, validate)
        pointers['combined'] = str(final / 'combined')
        if args.load_neo4j:
            signature = store.state['stages']['validate']['signature']
            run_id = f'{args.run_name}_{signature[:12]}'
            report = load_and_verify(config, final / 'combined/neo4j', run_id, root / 'logs/neo4j_load.log')
            write_json(root / 'db_loads' / f'{run_id}.json', report)
            store.state['db_loads'][signature] = report
        finished('validate', pointers)
        return 0
    except Exception as exc:
        store.state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        store.save()
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
