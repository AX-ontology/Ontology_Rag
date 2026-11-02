from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import clean_and_export as cleaner  # noqa: E402
import load_neo4j as loader  # noqa: E402


class CleaningRuleTests(unittest.TestCase):
    def test_alias_registry_collision_exports_one_concept_with_all_support(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = cleaner.ConceptRegistry(Path(temporary) / 'entities.jsonl', 'CE')
            nodes = {
                'CE_A': {'id': 'CE_A', 'label': 'CanonicalEntity', 'name': '행정가', 'properties': {'surface_forms': ['행정가']}},
                'CE_B': {'id': 'CE_B', 'label': 'CanonicalEntity', 'name': '행정가로', 'properties': {'surface_forms': ['행정가로는']}},
            }
            edges = []
            for i in range(4):
                mid = f'M{i}'
                nodes[mid] = {'id': mid, 'label': 'EntityMention', 'name': '행정가', 'properties': {}}
                edges.append({'id': f'E{i}', 'source': mid, 'target': 'CE_A' if i < 2 else 'CE_B', 'type': 'CANONICALIZED_AS'})
            result, links, _, report = cleaner.clean_entities(nodes, edges, {}, set(), registry,
                min_support=2, max_length=40, quarantined_facts=set())
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]['properties']['support_count'], 4)
            self.assertEqual(result[0]['properties']['source_canonical_ids'], ['CE_A', 'CE_B'])
            self.assertEqual(len(links), 4)
            self.assertEqual(report['registry_coalesced_group_count'], 1)
            registry.save()
            restored = cleaner.ConceptRegistry(Path(temporary) / 'entities.jsonl', 'CE')
            again, again_links, _, _ = cleaner.clean_entities(nodes, edges, {}, set(), restored,
                min_support=2, max_length=40, quarantined_facts=set())
            self.assertEqual(result, again)
            self.assertEqual(links, again_links)

    def test_report_handles_zero_alias_categories(self) -> None:
        summary = {key: Counter() for key in ['neo4j', 'entities', 'relations', 'quality', 'pcf']}
        summary.update(status='valid', taskframe_aliases={})
        summary['entities']['quarantine_reason_counts'] = {}
        report = cleaner.build_markdown_report(summary, Counter(), [], [])
        self.assertIn('Accepted alias occurrences: `0`', report)
        self.assertIn('Reclassified as raw TaskMention: `0`', report)
        self.assertIn('Rejected sentence/structured aliases: `0`', report)

    def test_pcf_needs_review_stays_inactive_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "pcf_kg_nodes.jsonl").write_text(
                json.dumps(
                    {
                        "id": "PCF_1",
                        "label": "PCFProcess",
                        "name": "영업",
                        "properties": {"hid": "3.0"},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "pcf_kg_edges.jsonl").write_text("", encoding="utf-8")
            (root / "taskframe_pcf_links.jsonl").write_text(
                json.dumps(
                    {
                        "mapping_id": "MAP_1",
                        "taskframe_id": "TF_1",
                        "pcfid": 1,
                        "pcf_hid": "3.0",
                        "mapping_role": "process_match",
                        "is_primary": True,
                        "validation_status": "needs_review",
                        "supporting_evidence_ids": [],
                        "supporting_triples": [],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            _, edges, decisions, report = cleaner.clean_pcf(
                root,
                {"TF_1": {"id": "TF_1", "label": "TaskFrame", "properties": {}}},
                [],
            )
        self.assertEqual(edges[0]["type"], "PCF_MAPPING_CANDIDATE")
        self.assertFalse(edges[0]["properties"]["active"])
        self.assertEqual(decisions[0]["review_status"], "pending")
        self.assertEqual(report["confirmed_mapping_count"], 0)
        self.assertEqual(report["pending_candidate_mapping_count"], 1)

    def test_lexical_key_strips_real_list_markers_but_keeps_meaningful_digits(self) -> None:
        self.assertEqual(cleaner.lexical_key("•  제안서 작성"), "제안서 작성")
        self.assertEqual(cleaner.lexical_key("3. 제안서 작성"), "제안서 작성")
        self.assertEqual(cleaner.lexical_key("3C 분석"), "3c 분석")
        self.assertEqual(cleaner.lexical_key("2등 전략 개발"), "2등 전략 개발")
        self.assertEqual(cleaner.lexical_key("3차 연습"), "3차 연습")

    def test_taskframe_alias_rule_separates_sentences(self) -> None:
        accepted, reasons, _ = cleaner.is_alias_candidate("고객 니즈 분석", "고객 분석")
        self.assertTrue(accepted)
        self.assertEqual(reasons, [])
        accepted, reasons, _ = cleaner.is_alias_candidate(
            "고객이 말하지 못하지만 정말 중요한 니즈를 찾아야 한다", "고객 분석"
        )
        self.assertFalse(accepted)
        self.assertIn("sentence_like", reasons)

    def test_entity_rule_quarantines_noise_and_singletons(self) -> None:
        accepted, reasons = cleaner.classify_entity(
            ["(○)"], "(○)", 10,
            explicit_alias=False, min_support=2, max_length=40,
        )
        self.assertFalse(accepted)
        self.assertIn("placeholder", reasons)
        accepted, reasons = cleaner.classify_entity(
            ["고객"], "고객", 4,
            explicit_alias=False, min_support=2, max_length=40,
        )
        self.assertTrue(accepted)
        self.assertEqual(reasons, [])

    def test_entity_name_normalization_removes_particles_conservatively(self) -> None:
        self.assertEqual(cleaner.normalize_entity_name("쉬플리에서"), "쉬플리")
        self.assertEqual(cleaner.normalize_entity_name("경쟁력 있는 제안 역량이란"), "경쟁력 있는 제안 역량")
        self.assertEqual(cleaner.normalize_entity_name("이는"), "이")
        self.assertEqual(cleaner.normalize_entity_name("이에"), "이")
        self.assertEqual(cleaner.normalize_entity_name("업체들"), "업체")
        self.assertEqual(cleaner.normalize_entity_name("전문가"), "전문가")
        self.assertEqual(cleaner.normalize_entity_name("평가"), "평가")

    def test_relation_rule_only_accepts_controlled_actions(self) -> None:
        alias_map, controlled = cleaner.load_alias_config(ROOT / "config/relation_aliases.json")
        canonical, method = cleaner.canonicalize_relation("결정하고", alias_map, controlled)
        self.assertEqual(canonical, "결정하다")
        self.assertIn(method, {"verified_alias", "controlled_morphology"})
        canonical, method = cleaner.canonicalize_relation("HIGH | MEDIUM", alias_map, controlled)
        self.assertIsNone(canonical)
        self.assertEqual(method, "uncontrolled_relation")

    def test_structured_fact_quality_rejects_delimiters_and_multi_items(self) -> None:
        reasons = cleaner.fact_surface_quality_reasons(
            {
                "subject_raw": "• 활동 A<br>• 활동 B",
                "predicate_raw": "|",
                "object_raw": "• 결과 A<br>• 결과 B",
            },
            "table",
        )
        self.assertIn("symbol_only_predicate", reasons)
        self.assertIn("table_delimiter_predicate", reasons)
        self.assertIn("multi_item_subject", reasons)
        self.assertIn("multi_item_object", reasons)

    def test_display_name_removes_html_but_preserves_raw_name(self) -> None:
        nodes = {
            "EM1": {
                "id": "EM1", "label": "EntityMention",
                "name": "Capture Plan<br><br>초안", "properties": {},
            }
        }
        report = cleaner.sanitize_display_names(nodes)
        self.assertEqual(nodes["EM1"]["name"], "Capture Plan 초안")
        self.assertEqual(
            nodes["EM1"]["properties"]["rawName"], "Capture Plan<br><br>초안"
        )
        self.assertEqual(report["EntityMention"], 1)


class ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.nodes_path = ROOT / "outputs/neo4j/nodes.jsonl"
        cls.edges_path = ROOT / "outputs/neo4j/edges.jsonl"
        if not cls.nodes_path.exists() or not cls.edges_path.exists():
            raise unittest.SkipTest("clean_and_export.py를 먼저 실행해야 합니다.")
        cls.raw_nodes = loader.read_jsonl(cls.nodes_path)
        cls.raw_edges = loader.read_jsonl(cls.edges_path)

    def test_export_is_loader_compatible(self) -> None:
        nodes = loader.normalize_nodes(self.raw_nodes, "test-run")
        edges = loader.normalize_edges(self.raw_edges, "test-run")
        report = loader.validate_records(nodes, edges)
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["nodes_by_label"]["TaskFrame"], 12)
        self.assertEqual(report["nodes_by_label"]["PCFProcess"], 2017)
        self.assertEqual(report["edges_by_type"]["MAPS_TO_PCF"], 12)
        self.assertEqual(report["edges_by_type"]["ALSO_MAPS_TO_PCF"], 2)
        self.assertEqual(report["edges_by_type"]["HAS_PCF_CONTEXT"], 3)

    def test_taskframe_alias_property_is_fully_separated(self) -> None:
        taskframes = [item for item in self.raw_nodes if item["label"] == "TaskFrame"]
        self.assertTrue(taskframes)
        self.assertTrue(all("aliases" not in item.get("properties", {}) for item in taskframes))
        self.assertTrue(any(item["label"] == "TaskFrameAlias" for item in self.raw_nodes))
        self.assertTrue(any(item["type"] == "HAS_ALIAS" for item in self.raw_edges))

    def test_graph_report_is_valid(self) -> None:
        report = json.loads((ROOT / "outputs/validation_report.json").read_text(encoding="utf-8"))
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["dangling_edge_count"], 0)

    def test_nested_properties_are_serialized_for_neo4j(self) -> None:
        value = loader.neo4j_value([{"type": "condition", "value": "x"}])
        self.assertIsInstance(value, str)
        self.assertEqual(json.loads(value)[0]["type"], "condition")

    def test_layer1_and_layer2_are_explicit_and_colored(self) -> None:
        nodes = loader.normalize_nodes(self.raw_nodes, "test-run")
        node_layers = {item["id"]: item["properties"]["graphLayer"] for item in nodes}
        edges = loader.normalize_edges(self.raw_edges, "test-run", node_layers)

        pcf_nodes = [item for item in nodes if item["label"] == "PCFProcess"]
        task_nodes = [item for item in nodes if item["label"] == "TaskFrame"]
        self.assertTrue(pcf_nodes)
        self.assertTrue(task_nodes)
        self.assertTrue(all(item["layer_label"] == "Layer1" for item in pcf_nodes))
        self.assertTrue(all(item["properties"]["vizColor"] == "#2563EB" for item in pcf_nodes))
        self.assertTrue(all(item["layer_label"] == "Layer2" for item in task_nodes))
        self.assertTrue(all(item["properties"]["vizColor"] == "#F97316" for item in task_nodes))

        mappings = [item for item in edges if item["type"] == "MAPS_TO_PCF"]
        self.assertTrue(mappings)
        self.assertTrue(all(item["properties"]["graphLayer"] == "cross_layer" for item in mappings))
        self.assertTrue(all(item["properties"]["vizColor"] == "#7C3AED" for item in mappings))


if __name__ == "__main__":
    unittest.main()
