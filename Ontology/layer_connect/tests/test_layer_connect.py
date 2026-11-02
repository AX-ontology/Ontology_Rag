from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from build_pcf_index import build_index  # noqa: E402
from build_taskframe_packets import build_packets  # noqa: E402
from common import lca_hid, load_config  # noqa: E402
from aggregate_hierarchy import (  # noqa: E402
    _document_branch_gate,
    _promote_broad_parent,
    _select_lca,
)
from rerank_candidates import _specialized_domain_penalty  # noqa: E402
import retrieve_candidates as retrieval  # noqa: E402
from select_document_context import select_document_context  # noqa: E402


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )


def _config(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        f"""paths:
  pcf: {tmp_path / 'pcf.json'}
  taskframes: {tmp_path / 'frames.json'}
  task_mentions: {tmp_path / 'mentions.jsonl'}
  canonical_facts: {tmp_path / 'facts.jsonl'}
  output_dir: {tmp_path / 'outputs'}
packet_limits:
  mentions: 30
  core_facts: 30
  supporting_facts: 15
  context_facts: 5
""",
        encoding="utf-8",
    )
    return path


def test_lca() -> None:
    parents = {"3.0": None, "3.5": "3.0", "3.5.3": "3.5", "3.5.3.1": "3.5.3", "3.5.3.2": "3.5.3"}
    assert lca_hid(["3.5.3.1", "3.5.3.2"], parents) == "3.5.3"
    assert lca_hid(["3.5", "3.5.3.2"], parents) == "3.5"


def test_semantic_cache_reuses_pcf_corpus_across_documents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        input_path = Path(command[command.index("--input") + 1])
        output_path = Path(command[command.index("--output") + 1])
        texts = json.loads(input_path.read_text(encoding="utf-8"))
        calls.append(texts)
        vectors = np.arange(1, len(texts) * 3 + 1, dtype=np.float32).reshape(
            len(texts), 3
        )
        np.save(output_path, vectors, allow_pickle=False)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(retrieval.subprocess, "run", fake_run)
    settings = {
        "enabled": True,
        "required": True,
        "model": "test-model",
        "python": sys.executable,
        "device": "cpu",
        "batch_size": 2,
        "cache_embeddings": True,
    }
    cache = tmp_path / "cache"
    retrieval._semantic_similarities(
        ["첫 문서 질의"], ["PCF 업무"], ["PCF 경로"], settings, cache
    )
    _, _, metadata = retrieval._semantic_similarities(
        ["둘째 문서 질의"], ["PCF 업무"], ["PCF 경로"], settings, cache
    )
    assert len(calls) == 2
    assert calls[0] == ["첫 문서 질의", "PCF 업무", "PCF 경로"]
    assert calls[1] == ["둘째 문서 질의"]
    assert metadata["cache_hits"] == {"queries": False, "pcf_corpus": True}


def test_hierarchy_aggregation_selects_supported_lca() -> None:
    parents = {"3.0": None, "3.5": "3.0", "3.5.3": "3.5", "3.5.3.11": "3.5.3", "3.5.3.12": "3.5.3"}
    common = {
        "taskframe_id": "TF_1", "pcf_path_ids": ["3.0", "3.5", "3.5.3"],
        "pcf_level": 3, "score": 0.50, "process_signal": 0.50,
        "mention_unit_scores": [], "fact_unit_scores": [],
    }
    parent = {**common, "pcf_hid": "3.5.3"}
    first = {
        **common, "pcf_hid": "3.5.3.11", "pcf_level": 4, "score": 0.55,
        "pcf_path_ids": ["3.0", "3.5", "3.5.3", "3.5.3.11"],
        "mention_unit_scores": [
            {"task_mention_id": "TM_1", "score": 0.8},
            {"task_mention_id": "TM_2", "score": 0.7},
        ],
    }
    second = {
        **common, "pcf_hid": "3.5.3.12", "pcf_level": 4, "score": 0.54,
        "pcf_path_ids": ["3.0", "3.5", "3.5.3", "3.5.3.12"],
        "mention_unit_scores": [
            {"task_mention_id": "TM_3", "score": 0.8},
            {"task_mention_id": "TM_4", "score": 0.7},
        ],
    }
    selected, decision = _select_lca(
        first, [first, second, parent], parents,
        {"atomic_alignment_score": 0.2, "lca_min_level": 2, "lca_min_branches": 2, "lca_min_units_per_branch": 2, "lca_max_score_drop": 0.1},
    )
    assert selected["pcf_hid"] == "3.5.3"
    assert decision and decision["strategy"] == "lca"


def test_broad_frame_promotes_leaf_to_well_supported_parent() -> None:
    leaf = {
        "pcf_hid": "11.1.3.1", "pcf_parent": "11.1.3", "is_leaf": True,
        "score": 0.57, "support_coverage": 1.0,
    }
    parent = {
        "pcf_hid": "11.1.3", "is_leaf": False,
        "score": 0.51, "support_coverage": 0.94,
    }
    packet = {"source_counts": {"member_mentions": 8}}
    selected, decision = _promote_broad_parent(
        leaf, [leaf, parent], packet,
        {
            "parent_promotion_min_mentions": 5,
            "parent_promotion_max_score_drop": 0.08,
            "parent_promotion_max_coverage_drop": 0.10,
        },
    )
    assert selected["pcf_hid"] == "11.1.3"
    assert decision and decision["strategy"] == "broad_frame_parent_promotion"


def test_it_scope_is_penalized_only_when_task_has_no_it_context() -> None:
    node = {
        "path_text_ko": "정보기술(IT) 관리 > IT 리스크 관리",
        "nameKo": "리스크 관리 역할 수립",
        "descKo": "IT 부문의 역할을 정의한다.",
    }
    penalty, reasons = _specialized_domain_penalty("제안 과정의 리스크 관리", node, 0.25)
    assert penalty == 0.25
    assert reasons == ["information_technology_scope"]
    penalty, reasons = _specialized_domain_penalty("IT 리스크 관리", node, 0.25)
    assert penalty == 0.0
    assert reasons == []


def test_document_context_uses_taskframe_coverage(tmp_path: Path) -> None:
    config = load_config(_config(tmp_path))
    output = tmp_path / "outputs"
    _write_jsonl(
        output / "pcf_nodes_enriched.jsonl",
        [
            {"hid": "3.5", "level": 2, "nameKo": "영업 계획", "path_ids": ["3.0", "3.5"]},
            {"hid": "1.2", "level": 2, "nameKo": "사업 전략", "path_ids": ["1.0", "1.2"]},
        ],
    )
    _write_jsonl(
        output / "taskframe_evidence_packets.jsonl",
        [
            {"taskframe_id": "TF_1", "frame_summary": {"canonical_name": "제안"}},
            {"taskframe_id": "TF_2", "frame_summary": {"canonical_name": "입찰"}},
        ],
    )
    _write_jsonl(
        output / "taskframe_pcf_candidates.jsonl",
        [
            {"taskframe_id": "TF_1", "pcf_hid": "3.5", "pcf_name_ko": "영업 계획", "pcf_path_ids": ["3.0", "3.5"], "frame_score": 0.7, "path_score": 0.7},
            {"taskframe_id": "TF_2", "pcf_hid": "3.5", "pcf_name_ko": "영업 계획", "pcf_path_ids": ["3.0", "3.5"], "frame_score": 0.6, "path_score": 0.6},
            {"taskframe_id": "TF_1", "pcf_hid": "1.2", "pcf_name_ko": "사업 전략", "pcf_path_ids": ["1.0", "1.2"], "frame_score": 0.9, "path_score": 0.9},
        ],
    )
    selected, report = select_document_context(config)
    assert selected["pcf_hid"] == "3.5"
    assert report["selected_taskframe_coverage"] == 1.0


def test_document_branch_gate_requires_strong_external_margin() -> None:
    inside = {
        "pcf_hid": "3.5.3", "score": 0.6, "reranker_raw_score": 0.5,
        "support_coverage": 1.0, "fact_unit_scores": [],
    }
    outside = {
        "pcf_hid": "11.1.3", "score": 0.7, "reranker_raw_score": 0.7,
        "support_coverage": 0.8, "reranker_penalty_reasons": [],
        "fact_unit_scores": [
            {"relevance": "core"}, {"relevance": "core"},
        ],
    }
    parents = {"3.5": None, "3.5.3": "3.5", "11.1.3": None}
    selected, decision, _ = _document_branch_gate(
        [inside, outside], {"selected_pcf_hid": "3.5"}, parents,
        {"external_reranker_margin": 0.15, "external_min_core_facts": 2},
    )
    assert selected["pcf_hid"] == "11.1.3"
    assert decision["strategy"] == "external_process_exception"
    outside["reranker_raw_score"] = 0.60
    selected, decision, _ = _document_branch_gate(
        [inside, outside], {"selected_pcf_hid": "3.5"}, parents,
        {"external_reranker_margin": 0.15, "external_min_core_facts": 2},
    )
    assert selected["pcf_hid"] == "3.5.3"
    assert decision["strategy"] == "document_context_subtree_default"


def test_build_index_restores_path(tmp_path: Path) -> None:
    config_path = _config(tmp_path)
    _write_json(
        tmp_path / "pcf.json",
        [
            {"pcfid": 1, "hid": "1.0", "parent": None, "level": 1, "nameKo": "루트", "nameEn": "Root", "descKo": "", "descEn": ""},
            {"pcfid": 2, "hid": "1.1", "parent": "1.0", "level": 2, "nameKo": "자식", "nameEn": "Child", "descKo": "", "descEn": ""},
        ],
    )
    nodes, report = build_index(load_config(config_path))
    assert report["valid"]
    assert nodes[1]["path_ids"] == ["1.0", "1.1"]
    assert nodes[1]["is_leaf"] is True


def test_packet_joins_direct_and_mention_fact_paths(tmp_path: Path) -> None:
    config_path = _config(tmp_path)
    _write_json(tmp_path / "pcf.json", [])
    _write_json(
        tmp_path / "frames.json",
        [{"taskframe_id": "TF_1", "canonical_name": "분석", "definition": "", "scope": "", "member_task_mentions": ["TM_1"], "evidence_ids": ["EV_1", "EV_2"]}],
    )
    _write_jsonl(
        tmp_path / "mentions.jsonl",
        [{"task_mention_id": "TM_1", "surface_form": "고객 분석", "action_normalized": "분석하다", "focal_object_normalized": "고객", "intended_outcome_raw": "", "section_path": "영업", "source_text": "고객을 분석한다", "confidence": 1.0, "surface_grounding": {}, "evidence_ids": ["EV_1"]}],
    )
    _write_jsonl(
        tmp_path / "facts.jsonl",
        [
            {"triple_id": "TR_1", "subject_canonical": "고객", "predicate_canonical": "분석한다", "object_canonical": "요구", "taskframe_links": [{"taskframe_id": "TF_1", "relevance": "supporting", "via_task_mentions": [], "propagation_path": "Fact→TaskFrame"}], "task_links": [{"task_mention_id": "TM_1", "relevance": "core"}], "evidence_ids": ["EV_2"], "section_path": "영업", "confidence": 1.0}
        ],
    )
    packets, report = build_packets(load_config(config_path))
    assert report["valid"]
    assert len(packets) == 1
    assert packets[0]["core_facts"][0]["triple_id"] == "TR_1"
    assert set(packets[0]["core_facts"][0]["propagation_paths"]) == {"Fact→TaskFrame", "Fact→TaskMention→TaskFrame"}


def test_invalid_parent_is_rejected(tmp_path: Path) -> None:
    config_path = _config(tmp_path)
    _write_json(
        tmp_path / "pcf.json",
        [{"pcfid": 2, "hid": "1.1", "parent": "1.0", "level": 2, "nameKo": "고립", "nameEn": "", "descKo": "", "descEn": ""}],
    )
    with pytest.raises(ValueError, match="PCF 트리 검증 실패"):
        build_index(load_config(config_path))
