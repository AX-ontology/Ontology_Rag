from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from taskkg.clustering import build_candidate_blocks
from taskkg.evidence import load_evidence
from taskkg.extraction import (
    chunk_modality, coerce_extraction, extract_llm,
    extract_structured_refinement_llm,
)
from taskkg.graph import build_graph
from taskkg.induction import (
    build_ambiguous_fact_contexts, estimate_hierarchical_llm2_calls,
    finalize_induction, finalize_reconciliation, generate_hierarchical_llm,
    pack_candidate_blocks, propagate_fact_links,
)
from taskkg.llm import ChatJSONClient, LLMConfig, LLMResponseError
from taskkg.normalization import canonicalize, conservative_prepare, load_aliases
from taskkg.prompts import extraction_prompt, structured_extraction_prompt
from taskkg.utils import stable_id


def evidence(evidence_id: str, source: str) -> dict:
    return {
        "evidence_id": evidence_id,
        "document_id": "DOC",
        "document_title": "테스트",
        "section_id": "SEC",
        "section_path": "입찰 > 분석",
        "paragraph_id": "PAR",
        "sentence_id": evidence_id,
        "source_file": "/tmp/test.xlsx",
        "sheet": "clean_sentences",
        "excel_row": 2,
        "source_sub_index": 1,
        "page": "1",
        "chapter_no": "1",
        "chapter_title": "입찰",
        "section_no": "1.1",
        "section_title": "분석",
        "section_sentence_index": 1,
        "evidence_type": "sentence",
        "evidence_role": "content",
        "source_text": source,
        "position": {"page": "1", "excel_row": 2, "source_sub_index": 1},
    }


def chunk() -> dict:
    rows = [
        evidence("EV1", "제안 PM은 RFP 요구사항을 분석한다."),
        evidence("EV2", "요구사항 분석표는 제안전략 수립에 활용된다."),
        evidence("EV3", "시장 규모는 100억 원이다."),
    ]
    return {
        "chunk_id": "CH1",
        "document_id": "DOC",
        "section_id": "SEC",
        "section_path": "입찰 > 분석",
        "evidence_ids": [item["evidence_id"] for item in rows],
        "sentences": [
            {
                "evidence_id": item["evidence_id"],
                "source_text": item["source_text"],
                "evidence_type": "sentence",
                "evidence_role": "content",
            }
            for item in rows
        ],
    }


def table_chunk() -> dict:
    source = (
        "단계 3: 수주전략 수립 | 활동(Activity) | 결과물(Output) | | --- | --- | | "
        "• 수주전략 초안 개발<br>• 수주전략 업데이트 | "
        "• Capture Plan 초안<br>• Executive Summary 초안 |"
    )
    return {
        "chunk_id": "CH_TABLE",
        "document_id": "DOC",
        "section_id": "SEC",
        "section_path": "입찰 > 수주전략",
        "evidence_ids": ["EV_TABLE"],
        "sentences": [{
            "evidence_id": "EV_TABLE",
            "source_text": source,
            "evidence_type": "table",
            "evidence_role": "structural_artifact",
        }],
    }


def extraction_payload() -> dict:
    return {
        "task_mentions": [
            {
                "task_mention_id": "M1",
                "surface_form": "RFP 요구사항을 분석한다",
                "action_raw": "분석한다",
                "focal_object_raw": "RFP 요구사항",
                "intended_outcome_raw": "",
                "evidence_ids": ["EV1"],
                "confidence": 0.9,
            },
            {
                "task_mention_id": "M2",
                "surface_form": "제안전략 수립",
                "action_raw": "수립",
                "focal_object_raw": "제안전략",
                "intended_outcome_raw": "",
                "evidence_ids": ["EV2"],
                "confidence": 0.8,
            },
        ],
        "triples": [
            {
                "triple_id": "T1",
                "subject_raw": "제안 PM",
                "predicate_raw": "분석한다",
                "object_raw": "RFP 요구사항",
                "evidence_id": "EV1",
                "task_links": [{"task_mention_id": "M1", "relevance": "core"}],
            },
            {
                "triple_id": "T2",
                "subject_raw": "요구사항 분석표",
                "predicate_raw": "활용된다",
                "object_raw": "제안전략 수립",
                "evidence_id": "EV2",
                "task_links": [
                    {"task_mention_id": "M1", "relevance": "supporting"},
                    {"task_mention_id": "M2", "relevance": "core"},
                ],
            },
            {
                "triple_id": "T3",
                "subject_raw": "시장 규모",
                "predicate_raw": "는",
                "object_raw": "100억 원",
                "evidence_id": "EV3",
                "task_links": [],
            },
        ],
        "sentence_audits": [
            {"evidence_id": "EV1", "triple_ids": ["T1"], "status": "facts_extracted"},
            {"evidence_id": "EV2", "triple_ids": ["T2"], "status": "facts_extracted"},
            {"evidence_id": "EV3", "triple_ids": ["T3"], "status": "facts_extracted"},
        ],
    }


def fact_contract_payload() -> dict:
    return {
        "task_mentions": [
            {
                "task_mention_id": "M1",
                "surface_form": "RFP 요구사항을 분석한다",
                "action_raw": "분석한다",
                "focal_object_raw": "RFP 요구사항",
                "intended_outcome_raw": "",
                "evidence_ids": ["EV1"],
                "confidence": 0.9,
            }
        ],
        "facts": [
            {
                "fact_id": "F1",
                "subject_raw": "제안 PM",
                "raw_predicate": "분석한다",
                "object_raw": "RFP 요구사항",
                "knowledge_function": "procedure",
                "qualifiers": [],
                "task_links": [
                    {"task_mention_id": "M1", "relevance": "core"}
                ],
                "evidence": {
                    "source_id": "EV1",
                    "exact_quote": "제안 PM은 RFP 요구사항을 분석한다.",
                },
            },
            {
                "fact_id": "F2",
                "subject_raw": "시장 규모",
                "raw_predicate": "이다",
                "object_raw": "100억 원",
                "knowledge_function": "fact",
                "qualifiers": [
                    {"type": "condition", "text": "원문에 없는 조건"}
                ],
                "task_links": [],
                "evidence": {
                    "source_id": "EV3",
                    "exact_quote": "시장 규모는 100억 원이다.",
                },
            },
        ],
        "sentence_audits": [
            {"evidence_id": "EV1", "fact_ids": ["F1"], "status": "facts_extracted"},
            {"evidence_id": "EV2", "fact_ids": [], "status": "no_extractable_fact"},
            {"evidence_id": "EV3", "fact_ids": ["F2"], "status": "facts_extracted"},
        ],
    }


class V4PipelineTest(unittest.TestCase):
    def test_evidence_loader_accepts_text_for_llm_column(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "document.xlsx"
            pd.DataFrame(
                [
                    {
                        "page": 1,
                        "chapter_no": 1,
                        "chapter_title": "제안",
                        "section_no": "1.1",
                        "section_title": "분석",
                        "text_for_llm": "고객 요구사항을 분석한다.",
                    }
                ]
            ).to_excel(source, sheet_name="clean_sentences", index=False)
            rows, summary = load_evidence(
                source, "clean_sentences", "DOC_2", "문서 2"
            )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source_text"], "고객 요구사항을 분석한다.")
        self.assertEqual(summary["text_column"], "text_for_llm")

    def setUp(self) -> None:
        self.extraction = coerce_extraction(extraction_payload(), chunk())
        self.aliases = load_aliases({"RFP": "RFP", "제안요청서": "RFP"})
        self.mentions, self.triples = conservative_prepare(
            self.extraction["task_mentions"],
            self.extraction["triples"],
            self.aliases,
        )

    def test_llm1_keeps_unlinked_fact_and_multi_task_link(self) -> None:
        self.assertEqual(self.extraction["schema_errors"], [])
        self.assertEqual(len(self.extraction["triples"][1]["task_links"]), 2)
        self.assertEqual(self.extraction["triples"][2]["task_links"], [])
        self.assertTrue(all(
            item["source_text"] for item in self.extraction["triples"]
        ))

    def test_llm1_prompt_enforces_atomic_extractive_fact_boundaries(self) -> None:
        system, user = extraction_prompt([chunk()])

        self.assertIn("조사·어미·인용부호·괄호·약어·특수문자", system)
        self.assertIn("그 논항인 주체·목적어·수치·대상", system)
        self.assertIn("value in evidence.exact_quote", user)
        self.assertIn("목적어를 raw_predicate에 포함", user)
        self.assertIn(
            '"knowledge_function": '
            '"fact|definition|procedure|rule|decision|recommendation"',
            user,
        )
        self.assertIn('"type": "condition|temporal|exception"', user)
        self.assertIn(
            '"raw_predicate": "떨어뜨리고"',
            user,
        )
        self.assertIn('knowledge_function="definition"', user)
        self.assertIn('"knowledge_function":"procedure"', user)
        self.assertIn('"knowledge_function":"rule"', user)
        self.assertIn('"knowledge_function":"decision"', user)
        self.assertIn('"knowledge_function":"recommendation"', user)
        self.assertIn("`점심`은 일정 Fact지만", user)
        self.assertIn('"chunk_id": "CH1"', user)
        self.assertIn('"evidence_id": "EV1"', user)

    def test_structured_prompt_has_table_few_shots_and_forbids_delimiters(self) -> None:
        system, user = structured_extraction_prompt([table_chunk()])

        self.assertEqual(chunk_modality(table_chunk()), "structured")
        self.assertIn("표·도식 전용", system)
        self.assertIn('raw_predicate="|"', user)
        self.assertIn('raw_predicate":"활동(Activity)"', user)
        self.assertIn('raw_predicate":"결과물(Output)"', user)
        self.assertIn("서로 다른 행 또는 여러 글머리표", user)

    def test_symbol_predicate_and_multi_item_table_fact_fail_quality_gate(self) -> None:
        source = table_chunk()["sentences"][0]["source_text"]
        raw = {
            "task_mentions": [],
            "facts": [{
                "fact_id": "T1",
                "subject_raw": "• 수주전략 초안 개발<br>• 수주전략 업데이트",
                "raw_predicate": "|",
                "object_raw": "• Capture Plan 초안<br>• Executive Summary 초안",
                "knowledge_function": "procedure",
                "qualifiers": [],
                "task_links": [],
                "evidence": {"source_id": "EV_TABLE", "exact_quote": source},
            }],
            "sentence_audits": [{
                "evidence_id": "EV_TABLE", "fact_ids": ["T1"],
                "status": "facts_extracted",
            }],
        }
        extracted = coerce_extraction(raw, table_chunk())
        reasons = {item["reason"] for item in extracted["schema_errors"]}

        self.assertFalse(extracted["triples"][0]["surface_provenance_valid"])
        self.assertIn("symbol_only_predicate", reasons)
        self.assertIn("table_delimiter_predicate", reasons)
        self.assertIn("multi_item_subject", reasons)
        self.assertIn("multi_item_object", reasons)

    def test_structured_refinement_preserves_existing_taskmention_ids(self) -> None:
        source = table_chunk()["sentences"][0]["source_text"]
        previous_raw = {
            "task_mentions": [{
                "task_mention_id": "M1",
                "surface_form": "수주전략 초안 개발",
                "action_raw": "개발",
                "focal_object_raw": "수주전략 초안",
                "intended_outcome_raw": "",
                "evidence_ids": ["EV_TABLE"],
                "confidence": 1.0,
            }],
            "facts": [],
            "sentence_audits": [{
                "evidence_id": "EV_TABLE", "fact_ids": [],
                "status": "no_extractable_fact", "reason": "old",
            }],
        }
        previous = coerce_extraction(previous_raw, table_chunk())

        class Client:
            @staticmethod
            def call(_stage, _key, _system, _user):
                return {"chunks": [{
                    "chunk_id": "CH_TABLE",
                    "facts": [{
                        "fact_id": "T1",
                        "subject_raw": "단계 3: 수주전략 수립",
                        "raw_predicate": "결과물(Output)",
                        "object_raw": "• Capture Plan 초안",
                        "knowledge_function": "fact",
                        "qualifiers": [],
                        "task_links": [{
                            "task_mention_id": "M1", "relevance": "supporting",
                        }],
                        "evidence": {"source_id": "EV_TABLE", "exact_quote": source},
                    }],
                    "sentence_audits": [{
                        "evidence_id": "EV_TABLE", "fact_ids": ["T1"],
                        "status": "facts_extracted",
                    }],
                }]}

        refined = extract_structured_refinement_llm(
            Client(), [table_chunk()], {"CH_TABLE": previous}, batch_size=1
        )[0]

        self.assertEqual(
            refined["task_mentions"][0]["task_mention_id"],
            previous["task_mentions"][0]["task_mention_id"],
        )
        self.assertEqual(refined["triples"][0]["predicate_raw"], "결과물(Output)")
        self.assertTrue(refined["triples"][0]["surface_provenance_valid"])

    def test_structured_refinement_failure_keeps_mentions_and_drops_facts(self) -> None:
        previous = coerce_extraction({
            "task_mentions": [{
                "task_mention_id": "M1",
                "surface_form": "수주전략 초안 개발",
                "action_raw": "개발",
                "focal_object_raw": "수주전략 초안",
                "intended_outcome_raw": "",
                "evidence_ids": ["EV_TABLE"],
                "confidence": 1.0,
            }],
            "facts": [{
                "fact_id": "OLD",
                "subject_raw": "오염된 표 전체",
                "raw_predicate": "|",
                "object_raw": "오염된 셀 전체",
                "evidence": {"source_id": "EV_TABLE"},
            }],
        }, table_chunk())

        class FailingClient:
            @staticmethod
            def call(_stage, _key, _system, _user):
                raise RuntimeError("deadline")

        refined = extract_structured_refinement_llm(
            FailingClient(), [table_chunk()], {"CH_TABLE": previous}, batch_size=1
        )[0]

        self.assertEqual(
            refined["task_mentions"][0]["task_mention_id"],
            previous["task_mentions"][0]["task_mention_id"],
        )
        self.assertEqual(refined["triples"], [])
        self.assertEqual(
            refined["structured_refinement"]["status"],
            "no_facts_after_llm_failure",
        )

    def test_llm1_failed_batch_is_split_without_dropping_chunks(self) -> None:
        first = chunk()
        second = copy.deepcopy(first)
        second["chunk_id"] = "CH2"

        class SplitClient:
            calls: list[str] = []

            @classmethod
            def call(cls, _stage, key, _system, _user):
                cls.calls.append(key)
                chunk_ids = key.split("-", 1)[1].split("-")
                if len(chunk_ids) > 1:
                    raise LLMResponseError("malformed batch response")
                return {"chunks": [{"chunk_id": chunk_ids[0]}]}

        extracted = extract_llm(SplitClient(), [first, second], batch_size=2)

        self.assertEqual([item["chunk_id"] for item in extracted], ["CH1", "CH2"])
        self.assertEqual(len(SplitClient.calls), 3)

    def test_llm1_fact_contract_preserves_function_qualifier_and_quote(self) -> None:
        extracted = coerce_extraction(fact_contract_payload(), chunk())
        first, second = extracted["triples"]

        self.assertEqual(first["source_local_id"], "F1")
        self.assertEqual(first["predicate_raw"], "분석한다")
        self.assertEqual(first["knowledge_function"], "procedure")
        self.assertEqual(first["evidence_quote"], first["source_text"])
        self.assertTrue(first["evidence_grounding"]["exact_quote"])
        self.assertEqual(first["task_links"][0]["relevance"], "core")
        self.assertEqual(second["qualifiers"], [])
        self.assertTrue(any(
            error["reason"] == "non_extractive_qualifier"
            for error in extracted["schema_errors"]
        ))

    def test_llm1_fact_contract_rejects_invalid_quote_and_function(self) -> None:
        payload = fact_contract_payload()
        payload["facts"][0]["knowledge_function"] = "customer_domain_rule"
        payload["facts"][0]["evidence"]["exact_quote"] = "원문에 없는 인용"
        extracted = coerce_extraction(payload, chunk())
        reasons = {error["reason"] for error in extracted["schema_errors"]}

        self.assertIn("invalid_exact_quote", reasons)
        self.assertIn("invalid_knowledge_function", reasons)
        self.assertEqual(extracted["triples"][0]["knowledge_function"], "fact")
        self.assertFalse(extracted["triples"][0]["surface_provenance_valid"])

    def test_llm_client_retries_connection_reset(self) -> None:
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            @staticmethod
            def read() -> bytes:
                return json.dumps({
                    "choices": [{"message": {"content": '{"ok": true}'}}]
                }).encode()

        config = LLMConfig(
            api_key="test",
            api_base="https://example.invalid/v1",
            model="test-model",
            timeout=1,
            max_retries=2,
            rate_limit_seconds=0,
        )
        with tempfile.TemporaryDirectory() as directory:
            client = ChatJSONClient(config, Path(directory))
            with (
                patch(
                    "urllib.request.urlopen",
                    side_effect=[ConnectionResetError(104, "reset"), Response()],
                ) as urlopen,
                patch("taskkg.llm.time.sleep"),
            ):
                result = client.call("test-stage", "test-key", "system", "user")

        self.assertEqual(result, {"ok": True})
        self.assertEqual(urlopen.call_count, 2)

    def test_candidate_clustering_is_marked_blocking_only(self) -> None:
        pairs, blocks = build_candidate_blocks(
            self.mentions, self.triples, threshold=0.0, top_k=8
        )
        self.assertEqual(len(pairs), 1)
        self.assertTrue(all(item["blocking_only"] for item in blocks))
        self.assertEqual(
            set(blocks[0]["member_task_mention_ids"]),
            {item["task_mention_id"] for item in self.mentions},
        )
        batches = pack_candidate_blocks(blocks, self.mentions, max_mentions=1)
        self.assertEqual(len(batches), 2)
        estimate = estimate_hierarchical_llm2_calls(
            blocks, self.mentions, max_mentions=1
        )
        self.assertEqual(estimate["local_induction_calls"], 2)
        self.assertEqual(estimate["global_reconciliation_calls"], 1)
        self.assertEqual(estimate["total_llm2_calls"], 3)

    def test_llm2_partition_recovers_missing_mention(self) -> None:
        first_id = self.mentions[0]["task_mention_id"]
        raw = {
            "taskframes": [
                {
                    "frame_key": "F1",
                    "canonical_name": "RFP 요구사항 분석",
                    "definition": "RFP 요구사항을 분석하는 업무",
                    "scope": "입찰 분석",
                    "aliases": [],
                    "member_task_mentions": [first_id],
                    "status": "canonical",
                }
            ]
        }
        frames, _, _, report = finalize_induction(
            raw, self.mentions, self.triples, []
        )
        self.assertTrue(report["mention_partition_valid"])
        self.assertEqual(len(frames), 2)
        self.assertEqual(len(report["missing_mentions_recovered"]), 1)

    def test_llm2_rejects_top_level_taskframe_array(self) -> None:
        raw = [
            {
                "frame_key": "F1",
                "canonical_name": "통합 제안 업무",
                "member_task_mentions": [
                    item["task_mention_id"] for item in self.mentions
                ],
                "status": "canonical",
            }
        ]
        with self.assertRaisesRegex(ValueError, "complete JSON object"):
            finalize_induction(raw, self.mentions, self.triples, [])

    def test_unlinked_fact_uses_neighbor_chunk_as_context_candidate(self) -> None:
        contextual_triples = copy.deepcopy(self.triples)
        contextual_triples[2]["chunk_id"] = "CH2"
        source_chunk = chunk()
        chunks = [
            {
                **source_chunk,
                "chunk_id": "CH1",
                "evidence_ids": ["EV1", "EV2"],
                "sentences": source_chunk["sentences"][:2],
            },
            {
                **source_chunk,
                "chunk_id": "CH2",
                "evidence_ids": ["EV3"],
                "sentences": source_chunk["sentences"][2:],
            },
        ]
        cards, catalog = build_ambiguous_fact_contexts(
            contextual_triples, self.mentions, chunks
        )
        background = next(
            item for item in cards
            if item["triple_id"] == contextual_triples[2]["triple_id"]
        )
        self.assertEqual(background["current_task_links"], [])
        self.assertEqual(background["context_chunk_ids"], ["CH1", "CH2"])
        self.assertTrue(background["context_task_mention_ids"])
        self.assertEqual({item["chunk_id"] for item in catalog}, {"CH1", "CH2"})

    def test_fact_to_taskframe_propagation_is_exhaustive_and_many_to_many(self) -> None:
        frames, _, decisions, report = finalize_induction(
            {
                "taskframes": [
                    {
                        "frame_key": f"F{index}",
                        "canonical_name": mention["surface_form"],
                        "member_task_mentions": [mention["task_mention_id"]],
                    }
                    for index, mention in enumerate(self.mentions, 1)
                ]
            },
            self.mentions,
            self.triples,
            [],
        )
        self.assertTrue(report["mention_partition_valid"])
        propagated, links, summary = propagate_fact_links(
            self.triples, frames, decisions, self.mentions
        )
        self.assertEqual(len(propagated[1]["taskframe_links"]), 2)
        self.assertEqual(len(propagated[2]["taskframe_links"]), 1)
        self.assertTrue(propagated[2]["exhaustive_fallback_used"])
        self.assertEqual(summary["multi_taskframe_fact_count"], 1)
        self.assertEqual(summary["unlinked_fact_count"], 0)
        self.assertEqual(summary["all_fact_assignment_rate"], 1.0)
        self.assertEqual(summary["exhaustive_fallback_count"], 1)
        self.assertEqual(len(links), 4)

    def test_global_reconciliation_expands_provisional_members(self) -> None:
        provisional, local_relations, _, _ = finalize_induction(
            {
                "taskframes": [
                    {
                        "frame_key": f"LF{index}",
                        "canonical_name": mention["surface_form"],
                        "member_task_mentions": [mention["task_mention_id"]],
                    }
                    for index, mention in enumerate(self.mentions, 1)
                ]
            },
            self.mentions,
            self.triples,
            [],
        )
        final_frames, final_relations, report = finalize_reconciliation(
            {
                "taskframes": [
                    {
                        "frame_key": "GF1",
                        "provisional_taskframe_ids": [
                            item["taskframe_id"] for item in provisional
                        ],
                        "canonical_name": "제안 요구사항 기반 업무",
                        "status": "canonical",
                    }
                ],
                "task_relations": [],
            },
            provisional,
            local_relations,
            self.mentions,
        )
        self.assertTrue(report["mention_partition_valid"])
        self.assertEqual(len(final_frames), 1)
        self.assertEqual(
            set(final_frames[0]["member_task_mentions"]),
            {item["task_mention_id"] for item in self.mentions},
        )
        self.assertEqual(final_relations, [])

    def test_hierarchical_llm2_orchestration_uses_local_then_global(self) -> None:
        _, blocks = build_candidate_blocks(
            self.mentions, self.triples, threshold=0.0, top_k=8
        )

        class FakeClient:
            def __init__(self, mention_ids: list[str]):
                self.mention_ids = mention_ids
                self.local_index = 0
                self.stages: list[str] = []

            def call(
                self, stage: str, key: str, system: str, user: str
            ) -> dict:
                self.stages.append(stage)
                if stage == "llm_2_local_taskframe_induction":
                    mention_id = self.mention_ids[self.local_index]
                    self.local_index += 1
                    return {
                        "taskframes": [
                            {
                                "frame_key": f"LF{self.local_index}",
                                "canonical_name": f"로컬 업무 {self.local_index}",
                                "member_task_mentions": [mention_id],
                                "status": "provisional",
                            }
                        ],
                        "task_relations": [],
                        "triple_link_decisions": [],
                    }
                self.assert_global_stage(stage)
                return {
                    "taskframes": [
                        {
                            "frame_key": "GF1",
                            "provisional_taskframe_ids": [
                                stable_id("TF", mention_id)
                                for mention_id in self.mention_ids
                            ],
                            "canonical_name": "전역 조정 업무",
                            "status": "canonical",
                        }
                    ],
                    "task_relations": [],
                }

            @staticmethod
            def assert_global_stage(stage: str) -> None:
                if stage != "llm_2_global_taskframe_reconciliation":
                    raise AssertionError(stage)

        mention_ids = [item["task_mention_id"] for item in self.mentions]
        client = FakeClient(mention_ids)
        frames, relations, decisions, report = generate_hierarchical_llm(
            client, self.mentions, self.triples, blocks, [], [], max_mentions=1
        )
        self.assertEqual(
            client.stages,
            [
                "llm_2_local_taskframe_induction",
                "llm_2_local_taskframe_induction",
                "llm_2_global_taskframe_reconciliation",
            ],
        )
        self.assertEqual(report["total_llm2_calls"], 3)
        self.assertTrue(report["mention_partition_valid"])
        self.assertEqual(len(frames), 1)
        self.assertEqual(relations, [])
        self.assertEqual(decisions, [])

    def test_canonicalization_preserves_raw_fact_count(self) -> None:
        facts, entities, relations, promoted = canonicalize(
            self.triples,
            self.aliases,
            load_aliases({"분석한다": "분석하다", "활용된다": "활용되다"}),
            promotion_min_support=2,
        )
        self.assertEqual(len(facts), len(self.triples))
        self.assertTrue(all(item["raw_fact_preserved"] for item in facts))
        self.assertTrue(entities)
        self.assertTrue(relations)
        self.assertEqual(promoted, [])

    def test_minimal_meta_schema_graph_validates(self) -> None:
        frames, task_relations, decisions, _ = finalize_induction(
            {
                "taskframes": [
                    {
                        "frame_key": "F1",
                        "canonical_name": "요구사항 기반 제안업무",
                        "member_task_mentions": [
                            mention["task_mention_id"] for mention in self.mentions
                        ],
                    }
                ]
            },
            self.mentions,
            self.triples,
            [],
        )
        propagated, _, _ = propagate_fact_links(
            self.triples, frames, decisions, self.mentions
        )
        facts, entities, relations, promoted = canonicalize(
            propagated, self.aliases, {}, 3
        )
        rows = [
            evidence("EV1", "제안 PM은 RFP 요구사항을 분석한다."),
            evidence("EV2", "요구사항 분석표는 제안전략 수립에 활용된다."),
            evidence("EV3", "시장 규모는 100억 원이다."),
        ]
        nodes, edges, report = build_graph(
            "DOC", "테스트", rows, self.mentions, facts, frames,
            task_relations, entities, relations, promoted, [], [],
        )
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["unlinked_general_fact_count"], 0)
        labels = {item["label"] for item in nodes}
        self.assertTrue(
            {
                "EvidenceSpan", "Fact", "EntityMention", "RelationExpression",
                "TaskMention", "TaskFrame", "CanonicalEntity", "CanonicalTerm",
            } <= labels
        )
        self.assertTrue(edges)


if __name__ == "__main__":
    unittest.main()
