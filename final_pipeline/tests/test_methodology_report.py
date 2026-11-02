from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parents[1]))
from build_methodology_report import script_json, snapshot


class MethodologyReportTests(unittest.TestCase):
    def test_script_escape_roundtrip(self):
        value = {"quote": '</script><img onerror="alert(1)"> & \u2028\u2029'}
        encoded = script_json(value)
        self.assertNotIn("<", encoded)
        self.assertNotIn("&", encoded)
        self.assertEqual(json.loads(encoded), value)

    def fixture(self, root):
        nodes = [
            {"id": "D", "label": "Document", "name": "문서", "properties": {}},
            {"id": "T", "label": "TaskFrame", "name": "업무", "properties": {}},
            {"id": "M", "label": "TaskMention", "name": "언급", "properties": {"sourceDocumentIds": ["D"]}},
            {"id": "F", "label": "Fact", "name": "fact", "properties": {"sourceDocumentIds": ["D"], "quality_status": "quarantined"}},
            {"id": "E", "label": "EvidenceSpan", "name": "<원문>", "properties": {"document_id": "D", "page": "1"}},
            {"id": "P", "label": "PCFProcess", "name": "후보", "properties": {"hid": "1.1"}},
        ]
        edges = [{"id": str(i), "source": s, "target": t, "type": k, "properties": {"active": False}}
                 for i, (s, t, k) in enumerate([("M", "T", "MEMBER_OF"), ("T", "F", "HAS_CORE_FACT"),
                     ("T", "F", "HAS_CONTEXT_FACT"), ("F", "E", "SUPPORTED_BY"), ("T", "P", "PCF_MAPPING_CANDIDATE")])]
        (root / "neo4j").mkdir()
        for name, rows in [("nodes", nodes), ("edges", edges)]:
            (root / f"neo4j/{name}.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
        (root / "taskframes.json").write_text(json.dumps([{"taskframe_id": "T", "canonical_name": "업무", "member_task_mentions": ["M"]}]))
        report = {"valid": True, "node_count": len(nodes), "edge_count": len(edges),
                  "nodes_by_label": dict(Counter(n["label"] for n in nodes)),
                  "edges_by_type": dict(Counter(e["type"] for e in edges))}
        (root / "validation_report.json").write_text(json.dumps(report))

    def test_counts_samples_and_unconfirmed_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            result = snapshot(root, "test", "test")
            frame = result["frames"][0]
            self.assertEqual(frame["facts"], 1)  # Two relations to the same fact.
            self.assertEqual(frame["byDoc"], {"D": 1})
            self.assertEqual(frame["quarantined"], 1)
            self.assertEqual(frame["samples"][0]["evidence"][0]["quote"], "<원문>")
            self.assertFalse(frame["pcf"][0]["active"])
            self.assertIsNone(result["load"])

    def test_unresolved_endpoint_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            path = root / "neo4j/edges.jsonl"
            path.write_text(path.read_text().replace('"target": "P"', '"target": "missing"'))
            with self.assertRaisesRegex(ValueError, "Unresolved"):
                snapshot(root, "test", "test")

    def test_stale_validation_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            path = root / "validation_report.json"
            report = json.loads(path.read_text())
            report["node_count"] += 1
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "differs"):
                snapshot(root, "test", "test")


if __name__ == "__main__":
    unittest.main()
