import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch, Mock
import json
from types import SimpleNamespace
from tempfile import TemporaryDirectory

spec = importlib.util.spec_from_file_location('consolidate', Path(__file__).parents[1] / 'consolidate_taskframes.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ConsolidationTests(unittest.TestCase):
    def test_bundle_preserves_facts_mentions_and_source(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline, source, out = root / 'base', root / 'source', root / 'out'
            module.write_json(baseline / '05_taskframes/taskframes.json', [
                {'taskframe_id': 'T', 'canonical_name': 'large task', 'definition': 'large purpose'}])
            module.write_json(source / '05_taskframes/taskframes.json', [
                {'taskframe_id': i, 'canonical_name': i, 'definition': i,
                 'member_task_mentions': [m], 'evidence_ids': [e]}
                for i, m, e in [('a', 'm1', 'e1'), ('b', 'm2', 'e2')]])
            module.write_jsonl(source / '02_extraction/task_mentions_raw.jsonl', [
                {'task_mention_id': m, 'surface_form': m} for m in ['m1', 'm2']])
            fact = {'triple_id': 'F', 'evidence_quote': 'unaltered evidence', 'taskframe_links': [
                {'taskframe_id': 'a', 'via_task_mentions': ['m1'], 'relevance': 'core'},
                {'taskframe_id': 'b', 'via_task_mentions': ['m2'], 'relevance': 'context'}]}
            module.write_jsonl(source / '06_propagation/facts_with_taskframes.jsonl', [fact])
            config = SimpleNamespace(api_base='https://api.deepseek.com', model='fake')
            with patch.object(module.LLMConfig, 'from_env', return_value=config), patch.object(module, 'ChatJSONClient') as client:
                client.return_value.has_cached_call.return_value = False
                client.return_value.call.side_effect = [
                    {'assignments': [{'source_id': i, 'target_id': 'C001', 'confidence': .9, 'reason': 'same purpose'} for i in ['S001', 'S002']]},
                    {'groups': [{'ids': ['T'], 'representative_id': 'T', 'reason': 'keep'}]}]
                module.run(SimpleNamespace(baseline=baseline, source=source, out=out, env=root / 'noenv', batch_size=48))
            report = module.read_json(out / 'validation_report.json')
            self.assertEqual(report['final_frame_count'], 1)
            self.assertEqual(report['task_mention_count'], 2)
            self.assertEqual(report['fact_link_count'], 1)
            result = module.read_jsonl(out / 'facts_with_taskframes.jsonl')[0]
            self.assertEqual(result['evidence_quote'], fact['evidence_quote'])
            self.assertEqual(module.read_jsonl(source / '06_propagation/facts_with_taskframes.jsonl'), [fact])

    def batch_fixture(self):
        client = Mock()
        client.has_cached_call.return_value = False
        cards = [{'source_id': 'TF_source_a', 'name': 'detail A'},
                 {'source_id': 'TF_source_b', 'name': 'detail B'}]
        catalog = {'TF_target': {'id': 'TF_target', 'name': 'broad', 'definition': 'purpose'}}
        return client, cards, catalog

    def test_repair_only_invalid_target_with_explicit_feedback(self):
        client, cards, catalog = self.batch_fixture()
        client.call.side_effect = [
            {'assignments': [{'source_id': 'S001', 'target_id': 'C001', 'confidence': .9, 'reason': 'valid'},
                             {'source_id': 'S002', 'target_id': 'S001', 'confidence': .9, 'reason': 'wrong namespace'}]},
            {'assignments': [{'source_id': 'S002', 'target_id': 'C001', 'confidence': .9, 'reason': 'repaired'}]}]
        rows, new = module.align_batch(client, 48, cards, catalog, 'legacy')
        self.assertEqual([r['target_id'] for r in rows], ['TF_target', 'TF_target'])
        self.assertEqual([r['source_id'] for r in rows], ['TF_source_a', 'TF_source_b'])
        self.assertEqual(rows[0]['reason'], 'valid')
        prompt = json.loads(client.call.call_args_list[1].args[3])
        self.assertEqual([c['source_id'] for c in prompt['source_frames']], ['S002'])
        self.assertEqual(prompt['previous_validation_errors'][0]['rejected_target'], 'S001')
        self.assertNotIn('TF_source', client.call.call_args_list[0].args[3])
        self.assertEqual(new, [])

    def test_valid_legacy_cache_reused_without_api(self):
        client, cards, catalog = self.batch_fixture()
        with TemporaryDirectory() as tmp:
            cache = Path(tmp) / 'cache.json'
            raw = {'assignments': [{'source_id': c['source_id'], 'target_id': 'TF_target', 'confidence': .9, 'reason': 'old valid'} for c in cards]}
            module.write_json(cache, {'parsed': raw})
            client.has_cached_call.return_value = True
            client._cache_path.return_value = cache
            rows, _ = module.align_batch(client, 0, cards, catalog, 'legacy')
            self.assertEqual(rows, raw['assignments'])
            client.call.assert_not_called()

    def test_invalid_legacy_cache_untouched_and_v2_used(self):
        client, cards, catalog = self.batch_fixture()
        with TemporaryDirectory() as tmp:
            cache = Path(tmp) / 'cache.json'
            module.write_json(cache, {'parsed': {'assignments': [
                {'source_id': c['source_id'], 'target_id': 'TF_source_a', 'confidence': .9, 'reason': 'invalid'} for c in cards]}})
            before = cache.read_bytes()
            client.has_cached_call.return_value = True
            client._cache_path.return_value = cache
            client.call.return_value = {'assignments': [
                {'source_id': s, 'target_id': 'C001', 'confidence': .9, 'reason': 'valid'} for s in ['S001', 'S002']]}
            rows, _ = module.align_batch(client, 48, cards, catalog, 'legacy')
            self.assertEqual(len(rows), 2)
            self.assertEqual(client.call.call_args.args[0], 'flat_catalog_alignment_v2')
            self.assertEqual(cache.read_bytes(), before)

    def test_persistent_invalid_target_fails_closed(self):
        client, cards, catalog = self.batch_fixture()
        client.call.return_value = {'assignments': [
            {'source_id': s, 'target_id': 'TF_unknown', 'confidence': .9, 'reason': 'wrong'} for s in ['S001', 'S002']]}
        with self.assertRaisesRegex(ValueError, 'unresolved after repairs'):
            module.align_batch(client, 48, cards, catalog, 'legacy')
        self.assertEqual(client.call.call_count, 3)

    def test_new_task_survives_partial_repair(self):
        client, cards, catalog = self.batch_fixture()
        client.call.side_effect = [
            {'assignments': [{'source_id': 'S001', 'target_id': 'NEW1', 'confidence': .9, 'reason': 'new'},
                             {'source_id': 'S002', 'target_id': 'bad', 'confidence': .9, 'reason': 'bad'}],
             'new_tasks': [{'key': 'NEW1', 'name': 'new broad', 'definition': 'new purpose'}]},
            {'assignments': [{'source_id': 'S002', 'target_id': 'NEW1', 'confidence': .9, 'reason': 'same new'}]}]
        rows, new = module.align_batch(client, 0, cards, catalog, 'legacy')
        self.assertEqual([r['target_id'] for r in rows], ['NEW1', 'NEW1'])
        self.assertEqual(len(new), 1)

    def test_duplicate_and_missing_sources_repaired(self):
        client, cards, catalog = self.batch_fixture()
        row = {'source_id': 'S001', 'target_id': 'C001', 'confidence': .9, 'reason': 'duplicate'}
        client.call.side_effect = [{'assignments': [row, row]}, {'assignments': [
            {**row, 'source_id': s} for s in ['S001', 'S002']]}]
        rows, _ = module.align_batch(client, 0, cards, catalog, 'legacy')
        self.assertEqual(len(rows), 2)
        prompt = json.loads(client.call.call_args_list[1].args[3])
        self.assertEqual(len(prompt['previous_validation_errors']), 2)

    def test_missing_duplicate_unknown_assignments_rejected(self):
        good = {'source_id': 'a', 'target_id': 'T', 'confidence': .9, 'reason': 'same broad work'}
        module.validate_assignments({'assignments': [good]}, ['a'], ['T'])
        for rows in [[], [good, good], [{**good, 'target_id': 'unknown'}]]:
            with self.assertRaises(ValueError):
                module.validate_assignments({'assignments': rows}, ['a'], ['T'])

    def test_new_task_references(self):
        module.validate_assignments({'assignments': [
            {'source_id': 'a', 'target_id': 'NEW1', 'confidence': .8, 'reason': 'new purpose'}],
            'new_tasks': [{'key': 'NEW1', 'name': 'task', 'definition': 'purpose'}]}, ['a'], ['T'])

    def test_links_deduplicated_and_provenance_preserved(self):
        links = [{'taskframe_id': 'a', 'relevance': 'context', 'via_task_mentions': ['m1']},
                 {'taskframe_id': 'b', 'relevance': 'core', 'via_task_mentions': ['m2']}]
        actual = module.remap_links(links, {'a': 'T', 'b': 'T'})
        self.assertEqual(len(actual), 1)
        self.assertEqual(actual[0]['relevance'], 'core')
        self.assertEqual(actual[0]['source_taskframe_ids'], ['a', 'b'])
        self.assertEqual(actual[0]['via_task_mentions'], ['m1', 'm2'])
        self.assertEqual(links[0]['taskframe_id'], 'a')
        with self.assertRaises(KeyError):
            module.remap_links(links, {'a': 'T'})


if __name__ == '__main__':
    unittest.main()
