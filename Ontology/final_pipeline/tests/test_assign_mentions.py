import importlib.util
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parents[1]))
import assign_mentions as direct
from run_flat_pipeline import materialize
from run_catalog_pipeline import digest_paths
from taskkg.extraction import coerce_extraction, flatten
from taskkg.normalization import conservative_prepare


def fixture(root):
    source, baseline = root / 'source', root / 'baseline.json'
    evidence = {'evidence_id': 'E', 'document_id': 'DOC', 'section_id': 'SEC',
        'section_path': '제안', 'paragraph_id': 'PAR', 'sentence_id': 'E',
        'source_file': 'fixture.xlsx', 'page': '1', 'chapter_no': '1', 'section_no': '1',
        'evidence_type': 'sentence', 'source_text': '요구사항 분석표는 제안서 작성에 활용된다.',
        'position': {'page': '1'}}
    chunk = {'chunk_id': 'CH', 'document_id': 'DOC', 'section_id': 'SEC',
        'section_path': '제안', 'evidence_ids': ['E'], 'sentences': [evidence]}
    payload = {'task_mentions': [{'task_mention_id': 'M1', 'surface_form': '제안서 작성',
                'action_raw': '작성', 'focal_object_raw': '제안서', 'evidence_ids': ['E'], 'confidence': .9},
                {'task_mention_id': 'M2', 'surface_form': '요구사항 분석', 'action_raw': '분석',
                 'focal_object_raw': '요구사항', 'evidence_ids': ['E'], 'confidence': .9}],
        'triples': [{'triple_id': 'F1', 'subject_raw': '요구사항 분석표', 'predicate_raw': '활용된다',
            'object_raw': '제안서 작성', 'evidence_id': 'E', 'evidence_quote': evidence['source_text'],
            'task_links': [{'task_mention_id': 'M1', 'relevance': 'core'},
                           {'task_mention_id': 'M2', 'relevance': 'supporting'}]}], 'sentence_audits': []}
    extracted = coerce_extraction(payload, chunk)
    mentions, facts, _ = flatten([extracted])
    mentions, facts = conservative_prepare(mentions, facts, {})
    for path, rows in [('01_evidence/evidence_spans.jsonl', [evidence]), ('01_evidence/chunks.jsonl', [chunk]),
                       ('02_extraction/chunk_extractions.jsonl', [extracted]),
                       ('02_extraction/task_mentions_raw.jsonl', mentions),
                       ('03_entity_prep/task_mentions.jsonl', mentions), ('03_entity_prep/open_triples.jsonl', facts)]:
        direct.write_jsonl(source / path, rows)
    direct.write_json(source / 'run_manifest.json', {'pipeline_version': '4.5.1', 'status': 'complete', 'completed_stage': 'entity_prep'})
    direct.write_json(baseline, [{'taskframe_id': 'T', 'canonical_name': '제안서 작성 및 관리', 'definition': '승인된 범위',
        'scope': '', 'aliases': [], 'status': 'canonical', 'schema_version': '4.5.1',
        'member_task_mentions': ['old'], 'evidence_ids': ['old']}])
    return source, baseline, mentions, facts


def response(stage, key, system, prompt):
    data = json.loads(prompt)
    if stage != direct.STAGE: raise AssertionError('Unexpected extra LLM stage: '+stage)
    return {'assignments': [{'source_id': m['source_id'], 'target_id': 'C001', 'confidence': .9, 'reason': '같은 목적'}
                            for m in data['source_mentions']],
        'fact_decisions': [{'triple_id': f['triple_id'], 'status': 'LINK', 'reason': '문장 근거',
            'task_links': [{'task_mention_id': mid, 'relevance': 'core'} for mid in f['allowed_mentions']]}
            for f in data['ambiguous_facts']]}


class DirectAssignmentTests(unittest.TestCase):
    def test_existing_only_one_stage_preserves_raw_and_identity(self):
        with TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp); source, baseline, mentions, facts = fixture(root)
            before = digest_paths([source, baseline]); client = Mock(); client.call.side_effect = response
            report = direct.assign(source, baseline, root / 'out', client)
            self.assertEqual(client.call.call_count, 1)
            self.assertEqual(report['catalog_review_calls'], 0)
            self.assertEqual(report['novel_grouping_batches'], 0)
            self.assertEqual(report['local_induction_calls'], 0)
            self.assertEqual(digest_paths([source, baseline]), before)
            frames = direct.read_json(root / 'out/05_taskframes/taskframes.json')
            self.assertEqual(frames[0]['definition'], '승인된 범위')
            self.assertEqual(set(frames[0]['member_task_mentions']), {m['task_mention_id'] for m in mentions})
            result = direct.read_jsonl(root / 'out/06_propagation/facts_with_taskframes.jsonl')[0]
            self.assertEqual(result['evidence_quote'], facts[0]['evidence_quote'])
            self.assertEqual(len(result['taskframe_links']), 1)
            self.assertEqual(len(result['taskframe_links'][0]['via_task_mentions']), 2)
            self.assertFalse(result['exhaustive_fallback_used'])

    def test_cross_batch_fact_mentions_not_dropped(self):
        with TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp); source, baseline, _, _ = fixture(root)
            client = Mock(); client.call.side_effect = response
            direct.assign(source, baseline, root / 'out', client, batch_size=1)
            self.assertEqual(client.call.call_count, 2)
            prompt = json.loads(client.call.call_args_list[0].args[3])
            self.assertEqual(len(prompt['source_mentions']), 1)
            self.assertEqual(len(prompt['ambiguous_facts'][0]['allowed_mentions']), 2)

    def test_fact_repair_keeps_good_assignments(self):
        with TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp); source, baseline, _, _ = fixture(root)
            calls = []
            def repair(*args):
                data = json.loads(args[3]); calls.append(data)
                result = response(*args)
                if len(calls) == 1:
                    result['fact_decisions'][0]['task_links'][0]['task_mention_id'] = 'S001'
                return result
            client = Mock(); client.call.side_effect = repair
            direct.assign(source, baseline, root / 'out', client)
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[1]['source_mentions'], [])
            self.assertTrue(calls[1]['previous_validation_errors'])

    def test_large_fact_batch_splits_without_reassigning_mentions(self):
        with TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp); source, baseline, _, facts = fixture(root)
            direct.write_jsonl(source / '03_entity_prep/open_triples.jsonl',
                              facts + [{**facts[0], 'triple_id': 'second-fact'}])
            client = Mock(); client.call.side_effect = response
            report = direct.assign(source, baseline, root / 'out', client, max_facts=1)
            self.assertEqual(report['direct_assignment_batches'], 2)
            self.assertEqual(json.loads(client.call.call_args_list[1].args[3])['source_mentions'], [])
            self.assertEqual(report['fact_count'], 2)

    def test_missing_fact_context_stops_before_api(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); source, baseline, _, facts = fixture(root)
            direct.write_jsonl(source / '03_entity_prep/task_mentions.jsonl', [])
            direct.write_jsonl(source / '03_entity_prep/open_triples.jsonl', [{**facts[0], 'task_links': []}])
            client = Mock()
            with self.assertRaisesRegex(ValueError, 'review required before API'):
                direct.assign(source, baseline, root / 'out', client)
            client.call.assert_not_called()
            self.assertEqual(direct.read_jsonl(root / 'out/review_required.jsonl')[0]['kind'], 'fact')

    def test_review_stops_before_novel_creation(self):
        for kind in ['mention', 'fact']:
            with self.subTest(kind=kind), TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
                root = Path(tmp); source, baseline, _, _ = fixture(root)
                def review(*args):
                    result = response(*args)
                    result['assignments'][0]['target_id'] = 'NOVEL'
                    if kind == 'mention': result['assignments'][1]['target_id'] = 'REVIEW'
                    else: result['fact_decisions'][0].update(status='REVIEW', task_links=[])
                    return result
                client = Mock(); client.call.side_effect = review
                with self.assertRaisesRegex(ValueError, 'require review'):
                    direct.assign(source, baseline, root / 'out', client)
                self.assertEqual(client.call.call_count, 1)
                self.assertTrue(direct.read_jsonl(root / 'out/review_required.jsonl'))
                self.assertFalse((root / 'out/05_taskframes/taskframes.json').exists())

    def test_only_novel_grouped_no_final_review(self):
        with TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp); source, baseline, _, _ = fixture(root)
            stages = []
            def novel(stage, key, system, prompt):
                stages.append(stage)
                if stage == direct.STAGE:
                    result = response(stage, key, system, prompt)
                    result['assignments'][1]['target_id'] = 'NOVEL'
                    return result
                self.assertEqual(stage, 'flat_catalog_alignment_v2')
                data = json.loads(prompt)
                self.assertEqual(len(data['source_frames']), 1)
                return {'assignments': [{'source_id': 'S001', 'target_id': 'NEW1', 'confidence': .9, 'reason': '범위 밖'}],
                        'new_tasks': [{'key': 'NEW1', 'name': '별도 업무', 'definition': '다른 목적'}]}
            client = Mock(); client.call.side_effect = novel; client.has_cached_call.return_value = False
            report = direct.assign(source, baseline, root / 'out', client)
            self.assertEqual(stages, [direct.STAGE, 'flat_catalog_alignment_v2'])
            self.assertEqual(report['new_taskframe_count'], 1)
            self.assertEqual(report['taskframe_count'], 2)

    def test_graph_resume_cannot_call_llm_or_recreate_induction(self):
        with TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp); source, baseline, _, _ = fixture(root)
            client = Mock(); client.call.side_effect = response
            direct.assign(source, baseline, root / 'assigned', client)
            materialize(root / 'assigned', root / 'graph')
            spec = importlib.util.spec_from_file_location('direct_test_runner',
                Path(__file__).parents[2] / 'taskframe_v4/run_pipeline.py')
            runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
            argv = ['run_pipeline.py', '--out-dir', str(root / 'graph'), '--input-xlsx', str(root / 'none.xlsx'),
                    '--env-file', str(root / 'no.env'), '--document-id', 'DOC', '--resume', '--catalog-assigned']
            with patch.object(sys, 'argv', argv), patch.object(runner, '_client', side_effect=AssertionError('No API')), \
                 patch.object(runner, 'generate_hierarchical_llm', side_effect=AssertionError('No induction')):
                self.assertEqual(runner.main(), 0)
            manifest = direct.read_json(root / 'graph/run_manifest.json')
            self.assertEqual(manifest['llm_call_plan']['projected_total_llm_calls'], 0)
            self.assertTrue(direct.read_json(root / 'graph/09_kg/validation_report.json')['valid'])
            (root / 'graph/05_taskframes/taskframes.json').unlink()
            with patch.object(sys, 'argv', argv), patch.object(runner, '_client') as forbidden:
                with self.assertRaisesRegex(ValueError, 'refusing LLM fallback'): runner.main()
                forbidden.assert_not_called()


if __name__ == '__main__': unittest.main()
