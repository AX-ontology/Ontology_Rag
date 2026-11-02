import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).parents[1]))
from run_flat_pipeline import merge_catalog, materialize, verify_materialized, STAGES
from combine_graphs import write_json, write_jsonl


class FlatPipelineTests(unittest.TestCase):
    def test_catalog_merges_shared_ids_and_preserves_document_members(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for n in [1, 2]:
                write_json(root / str(n) / '05_taskframes/taskframes.json', [
                    {'taskframe_id': 'T', 'canonical_name': 'Broad task', 'member_task_mentions': [f'M{n}'],
                     'evidence_ids': [f'E{n}'], 'aliases': [f'old{n}']}])
            frames = merge_catalog([('D1', root / '1'), ('D2', root / '2')], root / 'out')
            self.assertEqual(len(frames), 1)
            self.assertEqual(frames[0]['member_task_mentions'], ['M1', 'M2'])
            self.assertEqual(frames[0]['source_document_ids'], ['D1', 'D2'])

    def test_materialize_only_relinks_and_resumes_without_source_mutation(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target, bundle = root / 'source', root / 'target', root / 'bundle'
            for s in STAGES: (source / s).mkdir(parents=True)
            f = {'taskframe_id': 'old', 'canonical_name': 'detail', 'member_task_mentions': ['M'], 'evidence_ids': ['E']}
            fact = {'triple_id': 'F', 'subject_raw': 'untouched', 'taskframe_links': [{'taskframe_id': 'old'}]}
            write_json(source / '05_taskframes/taskframes.json', [f])
            write_json(source / '05_taskframes/validation_report.json', {'mention_partition_valid': True})
            write_json(source / '06_propagation/summary.json', {})
            write_jsonl(source / '06_propagation/facts_with_taskframes.jsonl', [fact])
            write_json(bundle / 'taskframes.json', [{**f, 'taskframe_id': 'T', 'canonical_name': 'Broad'}])
            write_json(bundle / 'validation_report.json', {'status': 'complete'})
            write_jsonl(bundle / 'source_to_target.jsonl', [{'source_id': 'old', 'target_id': 'T'}])
            write_jsonl(bundle / 'facts_with_taskframes.jsonl', [{**fact, 'taskframe_links': [{'taskframe_id': 'T'}]}])
            write_jsonl(bundle / 'fact_taskframe_links.jsonl', [{'triple_id': 'F', 'taskframe_id': 'T'}])
            original = (source / '06_propagation/facts_with_taskframes.jsonl').read_bytes()
            materialize(source, target, bundle)
            materialize(source, target, bundle)
            self.assertEqual(verify_materialized(source, target)['taskframe_count'], 1)
            self.assertEqual((source / '06_propagation/facts_with_taskframes.jsonl').read_bytes(), original)
            self.assertEqual((target / '05_taskframes/task_relations.jsonl').read_text(), '')
            write_json(bundle / 'taskframes.json', [])
            with self.assertRaises(ValueError): materialize(source, target, bundle)


if __name__ == '__main__':
    unittest.main()
