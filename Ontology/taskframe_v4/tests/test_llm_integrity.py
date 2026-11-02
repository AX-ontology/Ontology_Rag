from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from taskkg.induction import finalize_reconciliation, pack_reconciliation_cards, reconcile_bounded
from taskkg.llm import ChatJSONClient, LLMConfig, LLMResponseError, parse_json
from taskkg.utils import write_json


class Response:
    def __init__(self, content: str, finish: str = "stop"):
        self.body = json.dumps({
            "choices": [{"message": {"content": content}, "finish_reason": finish}],
            "usage": {"completion_tokens": 8},
        }).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.body


class LLMIntegrityTests(unittest.TestCase):
    def test_bounded_packing_covers_every_frame_once(self):
        cards = [{"provisional_taskframe_id": f"TF{i}", "canonical_name": f"업무 {i}"} for i in range(79)]
        for reverse in (False, True):
            groups = pack_reconciliation_cards(cards, max_frames=7, reverse=reverse)
            self.assertTrue(all(len(group) <= 7 for group in groups))
            ids = [card["provisional_taskframe_id"] for group in groups for card in group]
            self.assertEqual(len(ids), 79)
            self.assertEqual(set(ids), {card["provisional_taskframe_id"] for card in cards})

    def test_bounded_reconciliation_preserves_cross_batch_edges(self):
        frames = [{"taskframe_id": f"TF{i}", "canonical_name": f"업무 {i}", "definition": "test", "scope": "test", "aliases": [], "member_task_mentions": [f"M{i}"], "status": "canonical"} for i in range(7)]
        mentions = [{"task_mention_id": f"M{i}"} for i in range(7)]
        relations = [{"source_taskframe_id": "TF0", "target_taskframe_id": "TF6", "relation": "precedes", "confidence": 1, "reason": "test"}]
        class Client:
            def call(self, *_args):
                return {"taskframes": [], "task_relations": []}
        final, links, report = reconcile_bounded(Client(), frames, relations, mentions, [], max_frames=3)
        self.assertEqual({f["taskframe_id"] for f in final}, {f["taskframe_id"] for f in frames})
        self.assertEqual(len(links), 1)
        self.assertTrue(all(f["status"] == "canonical" for f in final))
        self.assertTrue(report["mention_partition_valid"])
        self.assertEqual(len(report["passes"]), 2)

    def client(self, directory: str, retries: int = 2):
        return ChatJSONClient(LLMConfig(
            api_key="test", api_base="https://example.invalid/v1", model="test",
            timeout=1, max_retries=retries, rate_limit_seconds=0,
            max_tokens=8, max_tokens_limit=32,
        ), Path(directory))

    def test_truncated_outer_object_cannot_salvage_nested_array(self):
        with self.assertRaises(ValueError):
            parse_json('{"taskframes": [{"frame_key": "F1"}], "task_relations":')
        self.assertEqual(parse_json('```json\n{"ok": true}\n```'), {"ok": True})

    def test_length_finish_is_retried_with_larger_output_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            client = self.client(directory)
            with patch("urllib.request.urlopen", side_effect=[
                Response('{"ok": true}', "length"), Response('{"ok": true}'),
            ]) as urlopen, patch("taskkg.llm.time.sleep"):
                self.assertEqual(client.call("test", "key", "system", "user"), {"ok": True})
            budgets = [json.loads(call.args[0].data)["max_tokens"] for call in urlopen.call_args_list]
            self.assertEqual(budgets, [8, 16])
            errors = list((Path(directory) / "_errors/test").glob("*.json"))
            self.assertEqual(len(errors), 1)
            self.assertEqual(json.loads(errors[0].read_text())["finish_reason"], "length")

    def test_legacy_inner_array_cache_is_preserved_and_refetched(self):
        stage = "llm_2_local_taskframe_induction"
        complete = {"taskframes": [], "task_relations": [], "triple_link_decisions": []}
        with tempfile.TemporaryDirectory() as directory:
            client = self.client(directory)
            cache_path = client._cache_path(stage, "key", "system", "user")
            write_json(cache_path, {"parsed": [{"frame_key": "F1"}]})
            with patch("urllib.request.urlopen", return_value=Response(json.dumps(complete))) as urlopen:
                self.assertEqual(client.call(stage, "key", "system", "user"), complete)
            self.assertEqual(urlopen.call_count, 1)
            backups = list((Path(directory) / "_legacy_incomplete" / stage).glob("*.json"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(json.loads(backups[0].read_text())["parsed"], [{"frame_key": "F1"}])
            with patch("urllib.request.urlopen", side_effect=AssertionError("cache must be reused")):
                self.assertEqual(client.call(stage, "key", "system", "user"), complete)

    def test_incomplete_local_schema_is_not_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            client = self.client(directory, retries=1)
            with patch("urllib.request.urlopen", return_value=Response('{"taskframes": []}')):
                with self.assertRaises(LLMResponseError):
                    client.call("llm_2_local_taskframe_induction", "key", "system", "user")
            self.assertFalse((Path(directory) / "llm_2_local_taskframe_induction").exists())

    def test_sparse_reconciliation_preserves_unchanged_status_and_relations(self):
        frames = [
            {"taskframe_id": "TF1", "member_task_mentions": ["M1"], "status": "canonical"},
            {"taskframe_id": "TF2", "member_task_mentions": ["M2"], "status": "provisional"},
        ]
        relations = [{"source_taskframe_id": "TF1", "target_taskframe_id": "TF2", "relation": "precedes", "confidence": 0.9, "reason": "test"}]
        result, links, report = finalize_reconciliation(
            {"taskframes": [], "task_relations": [], "_unchanged_are_preserved": True},
            frames, relations, [{"task_mention_id": "M1"}, {"task_mention_id": "M2"}],
        )
        self.assertEqual(result, frames)
        self.assertEqual(len(links), 1)
        self.assertEqual(report["missing_provisional_frames_recovered"], [])
        self.assertTrue(report["mention_partition_valid"])


if __name__ == "__main__":
    unittest.main()
