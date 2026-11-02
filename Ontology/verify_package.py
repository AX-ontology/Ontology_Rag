#!/usr/bin/env python3
"""Read-only package verification. Never calls an API, model, or database."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'final_pipeline'))
from run_catalog_pipeline import (read_json, digest_paths, reusable_source,
                                  validate_approved_bundle)
from run_final_pipeline import load_config, resolve_path, config_path


def contained(path, required=True):
    path = Path(path).resolve()
    if ROOT != path and ROOT not in path.parents:
        raise ValueError(f'Runtime path outside package: {path}')
    if required and not path.exists():
        raise FileNotFoundError(path)
    return path


def verify(check_hashes=True):
    cfg = load_config(ROOT / 'final_pipeline/config.yaml')
    checked = []
    for key in cfg['paths']:
        path = contained(config_path(cfg, key), required=key not in {'taskframe_env', 'neo4j_env', 'output_root'})
        checked.append(str(path.relative_to(ROOT)))
    baseline = contained(resolve_path(cfg, cfg['catalog_pipeline']['baseline_taskframes']))
    catalog = {f['taskframe_id']: f for f in read_json(baseline)}
    for slug, doc in cfg['documents'].items():
        contained(resolve_path(cfg, doc['input_xlsx']))
        if doc.get('reuse_taskkg'):
            source = contained(resolve_path(cfg, doc['reuse_taskkg']))
            if reusable_source(cfg, doc) != source:
                raise ValueError(f'Approved input changed: {slug}')
            if doc.get('approved_alignment'):
                bundle = contained(resolve_path(cfg, doc['approved_alignment']))
                actual = digest_paths([source / '05_taskframes/taskframes.json',
                                       source / '06_propagation/facts_with_taskframes.jsonl'])
                if actual != doc['approved_source_digest']:
                    raise ValueError(f'Approved extraction digest differs: {slug}')
                catalog = validate_approved_bundle(source, bundle, catalog)
    for value in [cfg['pcf']['base_config'], cfg['cleaning']['entity_aliases'], cfg['cleaning']['relation_aliases']]:
        contained(resolve_path(cfg, value))
    import yaml
    pcf_file = resolve_path(cfg, cfg['pcf']['base_config'])
    pcf = yaml.safe_load(pcf_file.read_text(encoding='utf-8'))
    for key, value in pcf['paths'].items():
        contained(pcf_file.parent / value, required=key != 'output_dir')
    if pcf['retrieval']['semantic'].get('python') or pcf['reranker'].get('python'):
        raise ValueError('Default PCF workers should use the current Python, not a fixed interpreter')
    for pointer in read_json(ROOT / 'final_outputs/catalog_refactor_check/current.json').values():
        contained(ROOT / 'final_outputs/catalog_refactor_check' / pointer)
    symlinks = [p for p in ROOT.rglob('*') if p.is_symlink() and '.venv' not in p.parts]
    for link in symlinks: contained(link)
    count, total = 0, 0
    if check_hashes:
        for item in read_json(ROOT / 'package_manifest.json')['files']:
            path = contained(ROOT / item['path'])
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
            if h.hexdigest() != item['sha256']:
                raise ValueError(f'Packaged file changed: {item["path"]}')
            count += 1; total += path.stat().st_size
    packages = {module: importlib.util.find_spec(module) is not None for module in
                ['yaml', 'pandas', 'openpyxl', 'numpy', 'sklearn', 'sentence_transformers', 'neo4j', 'requests', 'pytest']}
    return {'package_valid': True, 'root': str(ROOT), 'approved_taskframes': len(catalog),
            'documents': list(cfg['documents']), 'runtime_paths': checked,
            'verified_files': count, 'verified_bytes': total, 'python_packages': packages,
            'credentials_copied': False,
            'local_taskframe_env_present': config_path(cfg, 'taskframe_env').exists(),
            'local_neo4j_env_present': config_path(cfg, 'neo4j_env').exists(),
            'external_api_called': False, 'database_accessed': False,
            'note': 'Package integrity only; install missing dependencies. Model inference/credentials not tested.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-hashes', action='store_true', help='Path/config checks only, e.g. after intentional code edits')
    args = parser.parse_args()
    print(json.dumps(verify(not args.skip_hashes), ensure_ascii=False, indent=2))
