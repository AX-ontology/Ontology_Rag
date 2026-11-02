from collections import Counter
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch
import yaml

sys.path.insert(0, str(Path(__file__).parents[1]))
import run_catalog_pipeline as pipeline
import consolidate_taskframes as alignment
from combine_graphs import _merge_properties, validate_graph, write_json, write_jsonl


def fixture_source(root, doc, frame_id):
    for stage in ['01_evidence', '02_extraction', '03_entity_prep', '04_clustering', '05_taskframes', '06_propagation']:
        (root / stage).mkdir(parents=True)
    frame = {'taskframe_id': frame_id, 'canonical_name': 'Broad' if frame_id == 'T' else 'Detail',
        'definition': 'purpose', 'scope': 'scope', 'aliases': [], 'schema_version': '4.5.1', 'status': 'canonical',
        'member_task_mentions': ['M' + doc], 'evidence_ids': ['E' + doc]}
    fact = {'triple_id': 'F' + doc, 'evidence_quote': 'untouched original',
            'taskframe_links': [{'taskframe_id': frame_id, 'relevance': 'core', 'via_task_mentions': ['M' + doc]}]}
    mention = {'task_mention_id': 'M' + doc, 'surface_form': 'original task'}
    write_json(root / 'run_manifest.json', {'status': 'complete', 'completed_stage': 'propagation'})
    write_jsonl(root / '01_evidence/evidence_spans.jsonl', [{'document_id': doc}])
    write_json(root / '05_taskframes/taskframes.json', [frame])
    write_json(root / '05_taskframes/validation_report.json', {'mention_partition_valid': True})
    write_json(root / '06_propagation/summary.json', {})
    write_jsonl(root / '06_propagation/facts_with_taskframes.jsonl', [fact])
    write_jsonl(root / '06_propagation/fact_taskframe_links.jsonl', [])
    write_jsonl(root / '03_entity_prep/task_mentions.jsonl', [mention])
    write_jsonl(root / '03_entity_prep/open_triples.jsonl', [fact])
    write_jsonl(root / '02_extraction/task_mentions_raw.jsonl', [mention])
    write_jsonl(root / '02_extraction/chunk_extractions.jsonl', [{'chunk_id': 'CH' + doc}])
    write_json(root / '02_extraction/summary.json', {'task_mention_count': 1, 'open_triple_count': 1, 'schema_error_count': 0})
    return frame, fact


class CatalogPipelineTests(unittest.TestCase):
    def test_content_change_creates_new_artifact_and_db_history_survives(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'source.json'
            write_json(source, {'value': 1})
            store = pipeline.StageStore(root / 'run')
            store.state['db_loads']['old'] = {'status': 'verified'}
            calls = []
            def action(out):
                calls.append(out); write_json(out / 'result.json', pipeline.read_json(source))
            with redirect_stdout(io.StringIO()):
                first = store.run('test', [source], {}, action)
                self.assertEqual(first, store.run('test', [source], {}, action))
                write_json(source, {'value': 2})
                second = store.run('test', [source], {}, action)
            self.assertNotEqual(first, second)
            self.assertEqual(len(calls), 2)
            self.assertEqual(pipeline.read_json(first / 'result.json'), {'value': 1})
            self.assertEqual(pipeline.StageStore(root / 'run').state['db_loads']['old']['status'], 'verified')
            write_json(second / 'result.json', {'corrupt': True})
            with self.assertRaisesRegex(ValueError, 'changed or damaged'):
                store.run('test', [source], {}, action)

    def test_counts_add_even_when_equal(self):
        self.assertEqual(_merge_properties({'support_count': 2}, {'support_count': 2}, sum_counts=True)['support_count'], 4)

    def test_existing_match_has_no_new_task_creation(self):
        client = Mock()
        client.call.return_value = {'assignments': [
            {'source_id': 'S001', 'target_id': 'C001', 'confidence': .9, 'reason': 'same purpose'},
            {'source_id': 'S002', 'target_id': 'NOVEL', 'confidence': .9, 'reason': 'different purpose'},
            {'source_id': 'S003', 'target_id': 'REVIEW', 'confidence': .4, 'reason': 'insufficient context'}]}
        catalog = {'T': {'id': 'T', 'name': 'Broad', 'definition': 'purpose'}}
        rows = alignment.match_existing(client, 0, [{'source_id': s, 'name': s} for s in ['a', 'b', 'c']], catalog)
        self.assertEqual([r['target_id'] for r in rows], ['T', 'NOVEL', 'REVIEW'])
        client.call.return_value['new_tasks'] = [{'key': 'NEW1', 'name': 'Forbidden', 'definition': 'new'}]
        with self.assertRaisesRegex(ValueError, 'new_tasks are forbidden'):
            alignment.match_existing(client, 0, [{'source_id': 'a'}], catalog)

    def test_only_novel_rows_reach_grouping_and_review_stops_creation(self):
        for review in [False, True]:
            with self.subTest(review=review), TemporaryDirectory() as tmp:
                root = Path(tmp); source = root / 'source'; base = root / 'base'; out = root / 'out'
                frame, fact = fixture_source(source, 'X', 'a')
                write_json(source / '05_taskframes/taskframes.json', [frame, {**frame, 'taskframe_id': 'b', 'member_task_mentions': ['M2']}])
                write_jsonl(source / '02_extraction/task_mentions_raw.jsonl', [
                    {'task_mention_id': 'MX', 'surface_form': 'one'}, {'task_mention_id': 'M2', 'surface_form': 'two'}])
                write_json(base / '05_taskframes/taskframes.json', [{'taskframe_id': 'T', 'canonical_name': 'Broad', 'definition': 'purpose'}])
                client = Mock(); client.has_cached_call.return_value = False
                stages = []
                def response(stage, key, system, prompt):
                    stages.append(stage); data = json.loads(prompt)
                    if stage == 'catalog_existing_match_v1':
                        return {'assignments': [{'source_id': 'S001', 'target_id': 'C001', 'confidence': .9, 'reason': 'same'},
                            {'source_id': 'S002', 'target_id': 'REVIEW' if review else 'NOVEL', 'confidence': .9, 'reason': 'outside scope'}]}
                    if stage == 'flat_catalog_alignment_v2':
                        self.assertEqual(len(data['source_frames']), 1)
                        return {'assignments': [{'source_id': 'S001', 'target_id': 'NEW1', 'confidence': .9, 'reason': 'new purpose'}],
                            'new_tasks': [{'key': 'NEW1', 'name': 'New broad', 'definition': 'different purpose'}]}
                    return {'groups': [{'ids': [c['id']], 'representative_id': c['id'], 'reason': 'distinct'} for c in data['catalog']]}
                client.call.side_effect = response
                args = alignment.argparse.Namespace(source=source, baseline=base, out=out, env=root / 'noenv', batch_size=48,
                    strategy='existing-first', preserve_baseline_catalog=True)
                config = alignment.LLMConfig('fake', 'https://api.deepseek.com', 'fake', 1, 1, 0)
                with patch.object(alignment.LLMConfig, 'from_env', return_value=config), patch.object(alignment, 'ChatJSONClient', return_value=client), redirect_stdout(io.StringIO()):
                    if review:
                        with self.assertRaisesRegex(ValueError, 'require review'): alignment.run(args)
                        self.assertEqual(stages, ['catalog_existing_match_v1'])
                        self.assertFalse((out / 'taskframes.json').exists())
                    else:
                        alignment.run(args)
                        self.assertEqual(len(pipeline.read_json(out / 'taskframes.json')), 2)
                        self.assertEqual(stages[0], 'catalog_existing_match_v1')
                        self.assertEqual(stages[1], 'flat_catalog_alignment_v2')

    def test_generic_full_flow_resume_and_no_db_by_default(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = pipeline.load_config(pipeline.HERE / 'config.yaml')
            for key, value in config['paths'].items(): config['paths'][key] = str(pipeline.resolve_path(config, value))
            config['pcf']['base_config'] = str(pipeline.resolve_path(config, config['pcf']['base_config']))
            for key in ['entity_aliases', 'relation_aliases']:
                config['cleaning'][key] = str(pipeline.resolve_path(config, config['cleaning'][key]))
            config['paths']['output_root'] = str(root / 'outputs')
            config['documents'] = {}
            for slug, doc, frame_id in [('alpha', 'D1', 'T'), ('beta', 'D2', 'detail')]:
                src = root / slug
                frame, fact = fixture_source(src, doc, frame_id)
                input_path = root / (slug + '.xlsx'); input_path.write_bytes(b'fixture-input')
                cfg = {'document_id': doc, 'document_title': doc, 'input_xlsx': str(input_path), 'reuse_taskkg': str(src),
                       'reuse_input_sha256': hashlib.sha256(b'fixture-input').hexdigest()}
                if slug == 'alpha': cfg['catalog_role'] = 'seed'
                else:
                    bundle = root / 'approved'
                    write_json(bundle / 'validation_report.json', {'status': 'complete'})
                    write_json(bundle / 'taskframes.json', [{**frame, 'taskframe_id': 'T', 'canonical_name': 'Broad'}])
                    write_jsonl(bundle / 'source_to_target.jsonl', [{'source_id': frame_id, 'target_id': 'T'}])
                    write_jsonl(bundle / 'facts_with_taskframes.jsonl', [{**fact, 'taskframe_links': [{**fact['taskframe_links'][0], 'taskframe_id': 'T'}]}])
                    write_jsonl(bundle / 'fact_taskframe_links.jsonl', [])
                    cfg['approved_alignment'] = str(bundle)
                    cfg['approved_source_digest'] = pipeline.digest_paths([src / '05_taskframes/taskframes.json', src / '06_propagation/facts_with_taskframes.jsonl'])
                config['documents'][slug] = cfg
            config['catalog_pipeline']['baseline_taskframes'] = str(root / 'alpha/05_taskframes/taskframes.json')
            config.pop('_config_path'); config.pop('_config_dir')
            cfg_path = root / 'config.yaml'; cfg_path.write_text(yaml.safe_dump(config))
            calls = []
            def run(cmd, cwd, log_path, dry_run):
                calls.append(cmd)
                def opt(key): return Path(cmd[cmd.index(key) + 1])
                if 'taskframe_v4/run_pipeline.py' in cmd[1]:
                    target = opt('--out-dir'); doc = cmd[cmd.index('--document-id') + 1]
                    if '--stop-after' in cmd:
                        self.assertEqual(cmd[cmd.index('--stop-after') + 1], 'entity_prep')
                        fixture_source(target, doc, 'fresh-detail')
                        return
                    self.assertIn('--resume', cmd)
                    frames = pipeline.read_json(target / '05_taskframes/taskframes.json')
                    facts = pipeline.read_jsonl(target / '06_propagation/facts_with_taskframes.jsonl')
                    write_jsonl(target / '07_canonicalization/canonical_facts.jsonl', facts)
                    nodes = [{'id': i, 'label': label, 'name': i, 'properties': {}} for i, label in
                        [(doc, 'Document'), ('M' + doc, 'TaskMention'), ('F' + doc, 'Fact')]]
                    nodes += [{'id': f['taskframe_id'], 'label': 'TaskFrame', 'name': f['canonical_name'], 'properties': {}} for f in frames]
                    write_jsonl(target / '09_kg/nodes.jsonl', nodes); write_jsonl(target / '09_kg/edges.jsonl', [])
                elif 'layer_connect/run_pipeline.py' in cmd[1]:
                    cfg = yaml.safe_load(opt('--config').read_text()); out = Path(cfg['paths']['output_dir'])
                    combined_facts = pipeline.read_jsonl(Path(cfg['paths']['canonical_facts']))
                    self.assertEqual(len(combined_facts), 2)
                    write_json(out / 'validation_report.json', {'valid': True})
                elif 'assign_mentions.py' in cmd[1]:
                    self.assertEqual(pipeline.read_json(opt('--baseline'))[0]['taskframe_id'], 'T')
                    fixture_source(opt('--out'), 'D2', 'T')
                    write_json(opt('--out') / '05_taskframes/validation_report.json', {
                        'status': 'complete', 'mention_partition_valid': True, 'method': 'direct_mentions_existing_first'})
                elif 'clean_and_export.py' in cmd[1]:
                    graph, out = opt('--taskkg-dir'), opt('--output-dir')
                    shutil.copytree(graph, out / 'neo4j')
                    write_json(out / 'validation_report.json', validate_graph(pipeline.read_jsonl(graph / 'nodes.jsonl'), []))
                else: self.assertIn('--validate-only', cmd)
            args = ['--config', str(cfg_path), '--run-name', 'test']
            with patch.object(pipeline, 'run_logged', side_effect=run), patch.object(pipeline, 'load_and_verify') as db, redirect_stdout(io.StringIO()):
                self.assertEqual(pipeline.main(args), 0)
                first_count = len(calls)
                self.assertEqual(pipeline.main(args), 0)
                self.assertEqual(len(calls), first_count)
                db.assert_not_called()
            self.assertEqual(sum('layer_connect/run_pipeline.py' in c[1] for c in calls), 1)
            current = pipeline.read_json(root / 'outputs/test/current.json')
            report = pipeline.read_json(Path(current['combined']) / 'validation_report.json')
            self.assertEqual(report['nodes_by_label']['TaskFrame'], 1)
            self.assertEqual(report['nodes_by_label']['Fact'], 2)
            # A changed input cannot keep using the old approved bundle.
            input_file = Path(config['documents']['beta']['input_xlsx']); input_file.write_bytes(b'changed')
            self.assertIsNone(pipeline.reusable_source(pipeline.load_config(cfg_path), config['documents']['beta']))
            with patch.object(pipeline, 'run_logged', side_effect=run), patch.object(pipeline, 'load_and_verify') as db, redirect_stdout(io.StringIO()):
                self.assertEqual(pipeline.main(args), 0)
                db.assert_not_called()
            self.assertTrue(any('assign_mentions.py' in c[1] for c in calls))
            self.assertFalse(any('consolidate_taskframes.py' in c[1] for c in calls))
            self.assertTrue(any('--stop-after' in c and 'entity_prep' in c for c in calls))
            self.assertTrue(any('--catalog-assigned' in c for c in calls))
            with patch.object(pipeline, 'run_logged', side_effect=run), patch.object(pipeline, 'load_and_verify',
                    side_effect=lambda cfg, graph, run_id, log: {'status': 'verified', 'run_id': run_id}) as db, redirect_stdout(io.StringIO()):
                self.assertEqual(pipeline.main(args + ['--load-neo4j']), 0)
                db.assert_called_once()
                loaded = pipeline.read_json(root / 'outputs/test/catalog_pipeline_manifest.json')['db_loads']
                self.assertTrue(loaded)
                db.reset_mock()
                self.assertEqual(pipeline.main(args), 0)
                db.assert_not_called()
                self.assertEqual(pipeline.read_json(root / 'outputs/test/catalog_pipeline_manifest.json')['db_loads'], loaded)


if __name__ == '__main__': unittest.main()
