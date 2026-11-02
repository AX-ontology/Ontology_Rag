#!/usr/bin/env python3
"""Apply approved flat TaskFrames to separate document snapshots and merge KGs.

Never overwrites original extraction/PCF/graph outputs and never loads Neo4j.
Resumable stages are fingerprinted. The existing jaije3 extraction can finish
in its separate tmux session; otherwise it is started by this driver.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from combine_graphs import combine_inputs, read_jsonl, write_json, write_jsonl
from run_final_pipeline import load_config, taskframe_command, document_paths, run_logged

HERE = Path(__file__).resolve().parent
STAGES = ['01_evidence', '02_extraction', '03_entity_prep', '04_clustering', '05_taskframes', '06_propagation']


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def fingerprint(paths):
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.resolve()).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def verify_materialized(source, target):
    original = read_jsonl(source / '06_propagation/facts_with_taskframes.jsonl')
    current = read_jsonl(target / '06_propagation/facts_with_taskframes.jsonl')
    strip = lambda rows: [{k: v for k, v in r.items() if k != 'taskframe_links'} for r in rows]
    if strip(original) != strip(current):
        raise ValueError('Non-link fact content changed')
    source_frames = read_json(source / '05_taskframes/taskframes.json')
    frames = read_json(target / '05_taskframes/taskframes.json')
    old_mentions = Counter(m for f in source_frames for m in f['member_task_mentions'])
    new_mentions = Counter(m for f in frames for m in f['member_task_mentions'])
    if old_mentions != new_mentions or any(n != 1 for n in new_mentions.values()):
        raise ValueError('TaskMention partition changed')
    mapping = dict(read_json(target / 'flat_source.json')['source_to_target'])
    if set(mapping) != {f['taskframe_id'] for f in source_frames}:
        raise ValueError('Source mapping is incomplete')
    for old, new in zip(original, current):
        if {mapping[x['taskframe_id']] for x in old['taskframe_links']} != {x['taskframe_id'] for x in new['taskframe_links']}:
            raise ValueError('Fact assignment lost or changed')
    ids = {f['taskframe_id'] for f in frames}
    if any(x['taskframe_id'] not in ids for f in current for x in f['taskframe_links']):
        raise ValueError('Dangling fact link')
    return {'fact_count': len(current), 'task_mention_count': len(new_mentions), 'taskframe_count': len(ids)}


def materialize(source, target, bundle=None):
    inputs = [source / '05_taskframes/taskframes.json', source / '06_propagation/facts_with_taskframes.jsonl']
    if bundle:
        inputs += [bundle / f for f in ['taskframes.json', 'facts_with_taskframes.jsonl', 'source_to_target.jsonl']]
        if read_json(bundle / 'validation_report.json')['status'] != 'complete':
            raise ValueError('Consolidation bundle is incomplete')
    signature = fingerprint(inputs)
    marker = target / 'flat_source.json'
    if marker.exists():
        if read_json(marker)['fingerprint'] != signature:
            raise ValueError(f'Inputs changed; use a new output run: {target}')
        verify_materialized(source, target)
        return
    if target.exists():
        raise ValueError(f'Incomplete/unknown output exists; preserve and use another run: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(target.name + '.preparing')
    if staging.exists():
        raise ValueError(f'Previous interrupted staging preserved: {staging}')
    for stage in STAGES:
        shutil.copytree(source / stage, staging / stage)
    if bundle:
        frames = read_json(bundle / 'taskframes.json')
        for f in frames:
            f['schema_version'] = '4.5.1'
        write_json(staging / '05_taskframes/taskframes.json', frames)
        for name in ['facts_with_taskframes.jsonl', 'fact_taskframe_links.jsonl']:
            shutil.copy2(bundle / name, staging / '06_propagation' / name)
        mapping = [(r['source_id'], r['target_id']) for r in read_jsonl(bundle / 'source_to_target.jsonl')]
    else:
        frames = read_json(staging / '05_taskframes/taskframes.json')
        mapping = [(f['taskframe_id'], f['taskframe_id']) for f in frames]
    # No old fine-grained or hierarchical task relations in a flat catalog.
    write_jsonl(staging / '05_taskframes/task_relations.jsonl', [])
    report = read_json(staging / '05_taskframes/validation_report.json')
    report.update(taskframe_count=len(frames), mention_partition_valid=True,
                  flat_consolidated=True, total_llm2_calls=0)
    write_json(staging / '05_taskframes/validation_report.json', report)
    facts = read_jsonl(staging / '06_propagation/facts_with_taskframes.jsonl')
    summary = read_json(staging / '06_propagation/summary.json')
    summary.update(fact_count=len(facts), task_related_fact_count=sum(bool(f['taskframe_links']) for f in facts),
                   unlinked_fact_count=sum(not f['taskframe_links'] for f in facts),
                   multi_taskframe_fact_count=sum(len(f['taskframe_links']) > 1 for f in facts),
                   fact_taskframe_link_count=sum(len(f['taskframe_links']) for f in facts))
    write_json(staging / '06_propagation/summary.json', summary)
    write_json(staging / 'flat_source.json', {'fingerprint': signature, 'source': str(source.resolve()),
               'bundle': str(bundle.resolve()) if bundle else None, 'source_to_target': mapping})
    verify_materialized(source, staging)
    staging.rename(target)


def merge_catalog(documents, output):
    merged = {}
    for document_id, taskkg in documents:
        for frame in read_json(taskkg / '05_taskframes/taskframes.json'):
            tid = frame['taskframe_id']
            if tid not in merged:
                merged[tid] = deepcopy(frame)
                merged[tid]['source_document_ids'] = [document_id]
                continue
            item = merged[tid]
            if item['canonical_name'] != frame['canonical_name']:
                raise ValueError(f'Catalog name collision: {tid}')
            for key in ['member_task_mentions', 'evidence_ids', 'aliases', 'source_taskframe_ids']:
                item[key] = sorted(set(item.get(key, [])) | set(frame.get(key, [])))
            item['source_document_ids'].append(document_id)
    frames = sorted(merged.values(), key=lambda f: f['taskframe_id'])
    write_json(output / 'taskframes.json', frames)
    write_json(output / '05_taskframes/taskframes.json', frames)
    write_json(output / 'catalog_summary.json', {'documents': [d for d, _ in documents],
        'taskframe_count': len(frames), 'task_mention_count': sum(len(f['member_task_mentions']) for f in frames),
        'hierarchy_created': False})
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=HERE.parent / 'final_outputs/v4_5_final')
    parser.add_argument('--output-root', type=Path, default=HERE.parent / 'final_outputs/v4_5_flat_final')
    args = parser.parse_args()
    source_root, root = args.source_root.resolve(), args.output_root.resolve()
    if source_root == root: raise ValueError('Output must be separate from original run')
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / '.driver.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = load_config(HERE / 'config.yaml')
    state_path = root / 'flat_pipeline_manifest.json'
    state = read_json(state_path) if state_path.exists() else {'completed_stages': []}
    state.update(status='running', hierarchy_created=False, load_neo4j=False, source_root=str(source_root))

    def stage(name, work):
        if name in state['completed_stages']: return
        state.update(current_stage=name, status='running')
        state.pop('error', None)
        write_json(state_path, state)
        print(f'[flat] {name}', flush=True)
        work()
        state['completed_stages'].append(name)
        write_json(state_path, state)

    def command(name, cmd):
        run_logged(cmd, HERE, root / 'logs' / f'{name}.log', dry_run=False)

    def prepare(slug, bundle=None):
        source, target = source_root / slug / 'taskkg', root / slug / 'taskkg'
        materialize(source, target, bundle)
        cmd = taskframe_command(sys.executable, config, config['documents'][slug], document_paths(root, slug), 'llm', 0)
        if '--resume' not in cmd: cmd.append('--resume')
        command(f'{slug}_rebuild_kg', cmd)
        verify_materialized(source, target)

    def downstream(slug):
        cmd = [sys.executable, str(HERE / 'run_final_pipeline.py'), '--documents', slug,
               '--run-name', root.name, '--output-root', str(root.parent), '--start-at', 'pcf', '--stop-after', 'validate']
        # jaije1's approved catalog and facts are unchanged; reuse its existing
        # PCF candidates, but rebuild cleaning against the no-hierarchy graph.
        if slug == 'jaije_1' and (source_root / slug / 'pcf/validation_report.json').exists():
            if not (root / slug / 'pcf').exists(): shutil.copytree(source_root / slug / 'pcf', root / slug / 'pcf')
            cmd[cmd.index('--start-at') + 1] = 'clean'
        command(f'{slug}_downstream', cmd)

    def combined(slugs, name):
        documents = [(config['documents'][s]['document_id'], root / s / 'taskkg') for s in slugs]
        frames = merge_catalog(documents, root / name)
        summary = combine_inputs([(config['documents'][s]['document_id'], root / s / 'cleaned/neo4j') for s in slugs], root / name)
        if summary['nodes_by_label'].get('TaskFrame', 0) != len(frames): raise ValueError('Catalog/KG frame count mismatch')
        if summary['edges_by_type'].get('SUBTASK_OF', 0): raise ValueError('Unexpected task hierarchy')
        expected_mentions = sum(len(read_jsonl(t / '03_entity_prep/task_mentions.jsonl')) for _, t in documents)
        expected_facts = sum(len(read_jsonl(t / '06_propagation/facts_with_taskframes.jsonl')) for _, t in documents)
        if summary['nodes_by_label'].get('TaskMention') != expected_mentions: raise ValueError('Merged mentions lost')
        if summary['nodes_by_label'].get('Fact') != expected_facts: raise ValueError('Merged facts lost')
        state[name] = {'taskframes': len(frames), 'facts': expected_facts, 'mentions': expected_mentions, 'valid': True}

    try:
        cache = root / '_shared_cache'
        if not cache.exists() and (source_root / '_shared_cache').exists():
            cache.symlink_to(source_root / '_shared_cache', target_is_directory=True)
        stage('jaije_1_materialize', lambda: prepare('jaije_1'))
        stage('jaije_2_materialize', lambda: prepare('jaije_2', source_root / 'jaije_2/flat_consolidated'))
        stage('merge_12_catalog', lambda: merge_catalog([
            ('JAIJE_1', root / 'jaije_1/taskkg'), ('JAIJE_2', root / 'jaije_2/taskkg')], root / 'combined_12'))
        stage('jaije_1_downstream', lambda: downstream('jaije_1'))
        stage('jaije_2_downstream', lambda: downstream('jaije_2'))
        stage('merge_12_graph', lambda: combined(['jaije_1', 'jaije_2'], 'combined_12'))

        def extract3():
            manifest = source_root / 'jaije_3/taskkg/run_manifest.json'
            while True:
                if manifest.exists():
                    m = read_json(manifest)
                    if m.get('status') == 'complete' and m.get('completed_stage') == 'kg': return
                live = subprocess.run(['tmux', 'has-session', '-t', 'ax_jaije3_extract'], capture_output=True).returncode == 0
                if not live: break
                print('[flat] waiting for existing jaije3 extraction; originals preserved', flush=True)
                time.sleep(20)
            cmd = taskframe_command(sys.executable, config, config['documents']['jaije_3'], document_paths(source_root, 'jaije_3'), 'llm', 0)
            command('jaije_3_extract', cmd)

        stage('jaije_3_extract', extract3)
        bundle3 = root / 'jaije_3/flat_consolidated'
        stage('jaije_3_align', lambda: command('jaije_3_align', [sys.executable,
            str(HERE / 'consolidate_taskframes.py'), '--baseline', str(root / 'combined_12'),
            '--source', str(source_root / 'jaije_3/taskkg'), '--out', str(bundle3),
            '--env', str(HERE.parent / 'config/taskframe.env'), '--preserve-baseline-catalog']))
        stage('jaije_3_materialize', lambda: prepare('jaije_3', bundle3))
        stage('jaije_3_downstream', lambda: downstream('jaije_3'))
        stage('merge_123_graph', lambda: combined(['jaije_1', 'jaije_2', 'jaije_3'], 'combined'))
        state.update(status='complete', current_stage='complete')
        write_json(state_path, state)
        print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)
    except Exception as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(state_path, state)
        raise


if __name__ == '__main__':
    main()
