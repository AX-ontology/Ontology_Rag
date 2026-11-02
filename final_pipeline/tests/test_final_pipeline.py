from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from combine_graphs import combine_inputs  # noqa: E402
from run_final_pipeline import (  # noqa: E402
    build_pcf_config,
    document_paths,
    load_config,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class FinalPipelineTests(unittest.TestCase):
    def test_default_config_has_all_documents_and_existing_inputs(self) -> None:
        config = load_config(ROOT / "config.yaml")
        self.assertEqual(list(config["documents"]), ["jaije_1", "jaije_2", "jaije_3"])
        self.assertIn("reuse_taskkg", config["documents"]["jaije_1"])
        for document in config["documents"].values():
            path = (Path(config["_config_dir"]) / document["input_xlsx"]).resolve()
            self.assertTrue(path.exists(), path)

    def test_pcf_config_uses_current_document_outputs(self) -> None:
        config = load_config(ROOT / "config.yaml")
        with tempfile.TemporaryDirectory() as temporary:
            paths = document_paths(Path(temporary), "jaije_2")
            payload = build_pcf_config(config, paths)
            self.assertEqual(payload["paths"]["output_dir"], str(paths["pcf"]))
            self.assertIn("jaije_2/taskkg", payload["paths"]["taskframes"])
            self.assertNotIn("zaije_1_v4_4_limit0", payload["paths"]["taskframes"])
            self.assertEqual(
                payload["paths"]["semantic_cache_dir"],
                str(paths["root"].parent / "_shared_cache/embedding"),
            )

    def test_combiner_deduplicates_shared_pcf_and_preserves_documents(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = root / "d1", root / "d2"
            shared = {
                "id": "PCF_1",
                "label": "PCFProcess",
                "name": "공통",
                "properties": {"hid": "1.0"},
            }
            write_jsonl(
                first / "nodes.jsonl",
                [shared, {"id": "D1", "label": "Document", "name": "1", "properties": {}}],
            )
            write_jsonl(
                second / "nodes.jsonl",
                [shared, {"id": "D2", "label": "Document", "name": "2", "properties": {}}],
            )
            write_jsonl(first / "edges.jsonl", [])
            write_jsonl(second / "edges.jsonl", [])
            summary = combine_inputs(
                [("D1", first), ("D2", second)], root / "combined"
            )
            self.assertTrue(summary["valid"])
            self.assertEqual(summary["node_count"], 3)
            nodes = [
                json.loads(line)
                for line in (root / "combined/neo4j/nodes.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            pcf = next(item for item in nodes if item["id"] == "PCF_1")
            self.assertEqual(pcf["properties"]["sourceDocumentIds"], ["D1", "D2"])


if __name__ == "__main__":
    unittest.main()
