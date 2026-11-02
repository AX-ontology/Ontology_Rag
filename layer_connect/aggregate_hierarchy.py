from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import (
    clamp,
    is_ancestor,
    lca_hid,
    load_config,
    normalize_text,
    output_path,
    read_json,
    read_jsonl,
    stable_id,
    unique_text,
    write_json,
    write_jsonl,
)


def _root(record: dict[str, Any]) -> str:
    path = record.get("pcf_path_ids", [])
    return path[0] if path else record["pcf_hid"]


def _child_below(
    ancestor: str, descendant: str, parent_by_hid: dict[str, str | None]
) -> str | None:
    if ancestor == descendant:
        return ancestor
    current = descendant
    previous = descendant
    seen: set[str] = set()
    while current in parent_by_hid and current not in seen:
        seen.add(current)
        previous = current
        parent = parent_by_hid[current]
        if parent == ancestor:
            return previous
        if parent is None:
            return None
        current = parent
    return None


def _process_support_ids(record: dict[str, Any]) -> set[str]:
    result = set(record.get("supporting_task_mentions", []))
    result.update(
        item["triple_id"]
        for item in record.get("fact_unit_scores", [])
        if item.get("relevance") in {"core", "supporting"}
    )
    return result


def _best_atomic_alignments(
    candidates: list[dict[str, Any]], root_hid: str, threshold: float
) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        if _root(candidate) != root_hid:
            continue
        for item in candidate.get("mention_unit_scores", []):
            score = float(item.get("score", 0.0))
            unit_id = item["task_mention_id"]
            if score >= threshold and score > float(best.get(unit_id, {}).get("score", -1)):
                best[unit_id] = {
                    "unit_id": unit_id,
                    "kind": "mention",
                    "pcf_hid": candidate["pcf_hid"],
                    "score": score,
                }
        for item in candidate.get("fact_unit_scores", []):
            if item.get("relevance") == "context":
                continue
            score = float(item.get("score", 0.0))
            unit_id = item["triple_id"]
            if score >= threshold and score > float(best.get(unit_id, {}).get("score", -1)):
                best[unit_id] = {
                    "unit_id": unit_id,
                    "kind": f"{item.get('relevance', 'supporting')}_fact",
                    "pcf_hid": candidate["pcf_hid"],
                    "score": score,
                }
    return best


def _select_lca(
    top: dict[str, Any],
    candidates: list[dict[str, Any]],
    parent_by_hid: dict[str, str | None],
    settings: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    threshold = float(settings.get("atomic_alignment_score", 0.22))
    alignments = _best_atomic_alignments(candidates, _root(top), threshold)
    if len(alignments) < 2:
        return top, None
    candidate_by_hid = {item["pcf_hid"]: item for item in candidates}
    proposed = lca_hid(
        [item["pcf_hid"] for item in alignments.values()], parent_by_hid
    )
    if not proposed or proposed not in candidate_by_hid:
        return top, None
    proposed_record = candidate_by_hid[proposed]
    if int(proposed_record.get("pcf_level", 0)) < int(settings.get("lca_min_level", 2)):
        return top, None
    aligned_values = list(alignments.values())

    def branch_counts(
        ancestor: str, values: list[dict[str, Any]]
    ) -> tuple[Counter[str], dict[str, list[dict[str, Any]]]]:
        counts: Counter[str] = Counter()
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for alignment in values:
            branch = _child_below(ancestor, alignment["pcf_hid"], parent_by_hid)
            if branch:
                counts[branch] += 1
                grouped[branch].append(alignment)
        return counts, grouped

    branches, grouped_alignments = branch_counts(proposed, aligned_values)
    pruned_unit_count = 0
    if branches:
        dominant_branch, dominant_count = branches.most_common(1)[0]
        dominant_ratio = dominant_count / max(1, sum(branches.values()))
        if (
            dominant_branch != proposed
            and dominant_ratio >= float(settings.get("lca_dominant_branch_ratio", 0.72))
        ):
            dominant_values = grouped_alignments[dominant_branch]
            refined = lca_hid(
                [item["pcf_hid"] for item in dominant_values], parent_by_hid
            )
            if refined and refined in candidate_by_hid:
                pruned_unit_count = len(aligned_values) - len(dominant_values)
                proposed = refined
                proposed_record = candidate_by_hid[proposed]
                aligned_values = dominant_values
                branches, grouped_alignments = branch_counts(proposed, aligned_values)

    # 근거가 두 이웃 가지에 거의 반반 갈려도 frame-level 1위가 속한 가지가
    # 충분히 지지되면 그 가지를 semantic anchor로 삼아 외곽 근거를 제거한다.
    top_branch = _child_below(proposed, top["pcf_hid"], parent_by_hid)
    if top_branch and top_branch != proposed and top_branch in grouped_alignments:
        top_branch_values = grouped_alignments[top_branch]
        top_branch_ratio = len(top_branch_values) / max(1, sum(branches.values()))
        if (
            len(top_branch_values)
            >= int(settings.get("lca_frame_anchor_min_units", 3))
            and top_branch_ratio
            >= float(settings.get("lca_frame_anchor_min_ratio", 0.45))
        ):
            refined = lca_hid(
                [item["pcf_hid"] for item in top_branch_values], parent_by_hid
            )
            if refined and refined in candidate_by_hid:
                pruned_unit_count += len(aligned_values) - len(top_branch_values)
                proposed = refined
                proposed_record = candidate_by_hid[proposed]
                aligned_values = top_branch_values
                branches, grouped_alignments = branch_counts(proposed, aligned_values)

    branches = Counter(branches)
    direct_count = branches.get(proposed, 0)
    descendant_count = sum(
        count for branch, count in branches.items() if branch != proposed
    )
    min_units = int(settings.get("lca_min_units_per_branch", 2))
    if direct_count >= min_units and descendant_count >= min_units:
        if float(proposed_record["score"]) >= float(top["score"]) - float(
            settings.get("lca_max_score_drop", 0.10)
        ):
            return proposed_record, {
                "strategy": "lca_with_parent_support",
                "lca_hid": proposed,
                "branches": dict(branches),
                "aligned_unit_count": len(alignments),
                "retained_unit_count": len(aligned_values),
                "pruned_outlier_unit_count": pruned_unit_count,
                "original_top_hid": top["pcf_hid"],
            }
    substantial = {
        branch: count
        for branch, count in branches.items()
        if branch != proposed and count >= min_units
    }
    if len(substantial) < int(settings.get("lca_min_branches", 2)):
        return top, None
    if float(proposed_record["score"]) < float(top["score"]) - float(
        settings.get("lca_max_score_drop", 0.10)
    ):
        return top, None
    return proposed_record, {
        "strategy": "lca",
        "lca_hid": proposed,
        "branches": substantial,
        "aligned_unit_count": len(alignments),
        "retained_unit_count": len(aligned_values),
        "pruned_outlier_unit_count": pruned_unit_count,
        "original_top_hid": top["pcf_hid"],
    }


def _contextual_specialization(
    top: dict[str, Any],
    candidates: list[dict[str, Any]],
    parent_by_hid: dict[str, str | None],
    settings: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Prefer a context-supported specific process over a broad cross-root match."""
    context_parents = [
        item
        for item in candidates
        if _root(item) != _root(top)
        and int(item.get("pcf_level", 99)) in {2, 3}
        and not item.get("is_leaf", False)
        and not item.get("reranker_penalty_reasons")
        and float(item.get("context_signal", 0))
        >= float(settings.get("context_anchor_score", 0.28))
    ]
    if not context_parents:
        return top, None
    parent = max(
        context_parents,
        key=lambda item: (float(item.get("context_signal", 0)), float(item["score"])),
    )
    descendants = [
        item
        for item in candidates
        if item["pcf_hid"] != parent["pcf_hid"]
        and is_ancestor(parent["pcf_hid"], item["pcf_hid"], parent_by_hid)
        and item.get("reranker_raw_score") is not None
        and not item.get("reranker_penalty_reasons")
    ]
    if not descendants or top.get("reranker_raw_score") is None:
        return top, None
    best = max(descendants, key=lambda item: float(item["reranker_raw_score"]))
    if (
        float(best["reranker_raw_score"])
        >= float(top["reranker_raw_score"])
        + float(settings.get("contextual_specialization_reranker_margin", 0.005))
        and float(best["score"])
        >= float(settings.get("contextual_specialization_score_ratio", 0.80))
        * float(top["score"])
        and float(best.get("support_coverage", 0))
        >= float(settings.get("contextual_specialization_min_coverage", 0.20))
    ):
        return best, {
            "strategy": "contextual_specialization",
            "context_parent_hid": parent["pcf_hid"],
            "original_top_hid": top["pcf_hid"],
            "selected_hid": best["pcf_hid"],
        }
    return top, None


def _promote_broad_parent(
    selected: dict[str, Any],
    candidates: list[dict[str, Any]],
    packet: dict[str, Any],
    settings: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Promote a leaf when a broad frame is almost equally covered by its parent."""
    if not selected.get("is_leaf", False):
        return selected, None
    if int(packet.get("source_counts", {}).get("member_mentions", 0)) < int(
        settings.get("parent_promotion_min_mentions", 5)
    ):
        return selected, None
    parent_hid = normalize_text(selected.get("pcf_parent"))
    parent = next(
        (item for item in candidates if item["pcf_hid"] == parent_hid), None
    )
    if not parent:
        return selected, None
    score_drop = float(selected["score"]) - float(parent["score"])
    coverage_drop = float(selected.get("support_coverage", 0)) - float(
        parent.get("support_coverage", 0)
    )
    if (
        score_drop <= float(settings.get("parent_promotion_max_score_drop", 0.08))
        and coverage_drop
        <= float(settings.get("parent_promotion_max_coverage_drop", 0.10))
    ):
        return parent, {
            "strategy": "broad_frame_parent_promotion",
            "original_top_hid": selected["pcf_hid"],
            "selected_hid": parent["pcf_hid"],
            "score_drop": round(score_drop, 6),
            "coverage_drop": round(coverage_drop, 6),
        }
    return selected, None


def _document_branch_gate(
    candidates: list[dict[str, Any]],
    document_context: dict[str, Any],
    parent_by_hid: dict[str, str | None],
    settings: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Default to the document subtree and permit only evidence-backed exits."""
    anchor_hid = normalize_text(document_context.get("selected_pcf_hid"))
    if not anchor_hid:
        raise ValueError("document_pcf_context.json에 selected_pcf_hid가 없습니다.")
    inside = [
        item
        for item in candidates
        if is_ancestor(anchor_hid, item["pcf_hid"], parent_by_hid)
    ]
    outside = [
        item
        for item in candidates
        if not is_ancestor(anchor_hid, item["pcf_hid"], parent_by_hid)
    ]
    if not inside:
        raise RuntimeError(
            f"TaskFrame 후보에 문서 context subtree {anchor_hid}가 없습니다."
        )
    inside_top = max(inside, key=lambda item: float(item["score"]))
    eligible_outside: list[tuple[dict[str, Any], int]] = []
    for item in outside:
        core_fact_count = sum(
            unit.get("relevance") == "core"
            for unit in item.get("fact_unit_scores", [])
        )
        if (
            item.get("reranker_raw_score") is not None
            and not item.get("reranker_penalty_reasons")
            and core_fact_count >= int(settings.get("external_min_core_facts", 2))
            and float(item.get("support_coverage", 0))
            >= float(settings.get("external_min_coverage", 0.20))
        ):
            eligible_outside.append((item, core_fact_count))
    outside_top = (
        max(
            eligible_outside,
            key=lambda value: (
                float(value[0].get("reranker_raw_score", 0)),
                float(value[0]["score"]),
            ),
        )
        if eligible_outside
        else None
    )
    inside_reranker = float(inside_top.get("reranker_raw_score") or 0.0)
    outside_reranker = (
        float(outside_top[0].get("reranker_raw_score") or 0.0)
        if outside_top
        else 0.0
    )
    reranker_margin = outside_reranker - inside_reranker
    required_margin = float(settings.get("external_reranker_margin", 0.15))
    if outside_top and reranker_margin >= required_margin:
        selected = outside_top[0]
        strategy = "external_process_exception"
        outside_core_fact_count = outside_top[1]
    else:
        selected = inside_top
        strategy = "document_context_subtree_default"
        outside_core_fact_count = outside_top[1] if outside_top else 0
    return selected, {
        "strategy": strategy,
        "document_context_hid": anchor_hid,
        "inside_hid": inside_top["pcf_hid"],
        "inside_score": inside_top["score"],
        "inside_reranker_score": round(inside_reranker, 6),
        "outside_hid": outside_top[0]["pcf_hid"] if outside_top else None,
        "outside_score": outside_top[0]["score"] if outside_top else None,
        "outside_reranker_score": round(outside_reranker, 6),
        "outside_core_fact_count": outside_core_fact_count,
        "reranker_margin": round(reranker_margin, 6),
        "required_external_margin": required_margin,
    }, inside


def _evidence_ids(
    packet: dict[str, Any], mention_ids: list[str], triple_ids: list[str]
) -> list[str]:
    mention_set = set(mention_ids)
    triple_set = set(triple_ids)
    values: list[str] = []
    for mention in packet.get("mentions", []):
        if mention.get("task_mention_id") in mention_set:
            values.extend(mention.get("evidence_ids", []))
    for key in ("core_facts", "supporting_facts", "context_facts"):
        for fact in packet.get(key, []):
            if fact.get("triple_id") in triple_set:
                values.extend(fact.get("evidence_ids", []))
    return unique_text(values)


def _relation_scope(
    selected: dict[str, Any], packet: dict[str, Any], lca_info: dict[str, Any] | None
) -> str:
    if lca_info:
        return "partial_overlap"
    if packet.get("source_counts", {}).get("member_mentions", 0) <= 1:
        return "taskframe_narrower_than_pcf"
    if selected.get("is_leaf") and float(selected.get("support_coverage", 0)) < 0.50:
        return "taskframe_narrower_than_pcf"
    if (
        float(selected.get("frame_score", 0)) >= 0.45
        and float(selected.get("support_coverage", 0)) >= 0.50
    ):
        return "equivalent"
    return "partial_overlap"


def _confidence(record: dict[str, Any], margin: float) -> float:
    return clamp(
        0.45 * float(record.get("score", 0))
        + 0.25 * float(record.get("support_coverage", 0))
        + 0.15 * min(1.0, max(0.0, margin) / 0.15)
        + 0.15 * float(record.get("path_score", 0))
    )


def _mapping_record(
    packet: dict[str, Any],
    candidate: dict[str, Any],
    role: str,
    is_primary: bool,
    relation_scope: str,
    confidence: float,
    validation_status: str,
    rationale: str,
) -> dict[str, Any]:
    mentions = list(candidate.get("supporting_task_mentions", []))
    triples = list(candidate.get("supporting_triples", []))
    return {
        "mapping_id": stable_id("MAP", packet["taskframe_id"], candidate["pcf_hid"], role),
        "taskframe_id": packet["taskframe_id"],
        "pcfid": candidate.get("pcfid"),
        "pcf_hid": candidate["pcf_hid"],
        "pcf_name_ko": candidate.get("pcf_name_ko", ""),
        "pcf_path_ids": candidate.get("pcf_path_ids", []),
        "mapping_role": role,
        "is_primary": is_primary,
        "relation_scope": relation_scope,
        "coverage_score": round(float(candidate.get("support_coverage", 0)), 6),
        "confidence": round(confidence, 6),
        "method": "semantic_document_context_gate_evidence_hierarchy_rerank",
        "validation_status": validation_status,
        "supporting_task_mentions": mentions,
        "supporting_triples": triples,
        "supporting_evidence_ids": _evidence_ids(packet, mentions, triples),
        "rationale": rationale,
        "candidate_score": candidate.get("score", 0.0),
    }


def aggregate(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    candidates = read_jsonl(output_path(config, "taskframe_pcf_candidates.jsonl"))
    packets = read_jsonl(output_path(config, "taskframe_evidence_packets.jsonl"))
    nodes = read_jsonl(output_path(config, "pcf_nodes_enriched.jsonl"))
    document_context = read_json(output_path(config, "document_pcf_context.json"))
    settings = config.get("aggregation", {})
    branch_settings = config.get("document_context", {})
    by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_frame[candidate["taskframe_id"]].append(candidate)
    for values in by_frame.values():
        values.sort(key=lambda item: int(item.get("rank", 10**9)))
    parent_by_hid = {
        node["hid"]: normalize_text(node.get("parent")) or None for node in nodes
    }

    links: list[dict[str, Any]] = []
    review_queue: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    for packet in packets:
        frame_id = packet["taskframe_id"]
        frame_candidates = by_frame.get(frame_id, [])
        review_reasons: list[str] = []
        selection_info = None
        if not frame_candidates:
            review_reasons.append("no_candidates")
            links.append(
                {
                    "mapping_id": stable_id("MAP", frame_id, "unmapped"),
                    "taskframe_id": frame_id,
                    "pcfid": None,
                    "pcf_hid": None,
                    "pcf_name_ko": "",
                    "pcf_path_ids": [],
                    "mapping_role": "process_match",
                    "is_primary": False,
                    "relation_scope": "unmapped",
                    "coverage_score": 0.0,
                    "confidence": 0.0,
                    "method": "semantic_document_context_gate_evidence_hierarchy_rerank",
                    "validation_status": "needs_review",
                    "supporting_task_mentions": [],
                    "supporting_triples": [],
                    "supporting_evidence_ids": [],
                    "rationale": "PCF 후보가 생성되지 않음",
                    "candidate_score": 0.0,
                }
            )
            selected = None
            lca_info = None
            margin = 0.0
        else:
            member_count = int(packet.get("source_counts", {}).get("member_mentions", 0))
            top, branch_info, inside_candidates = _document_branch_gate(
                frame_candidates, document_context, parent_by_hid, branch_settings
            )
            selection_info = branch_info
            if member_count <= 1:
                frame_weight = float(settings.get("small_frame_frame_weight", 0.40))
                path_weight = float(settings.get("small_frame_path_weight", 0.60))
                weight_total = max(frame_weight + path_weight, 1e-9)
                semantic_top = max(
                    inside_candidates,
                    key=lambda item: (
                        frame_weight * float(item.get("frame_score", 0))
                        + path_weight * float(item.get("path_score", 0))
                    )
                    / weight_total,
                )
                if float(semantic_top.get("support_coverage", 0)) >= float(
                    settings.get("min_evidence_coverage", 0.10)
                ):
                    top = dict(semantic_top)
                    top["score"] = round(
                        (
                            frame_weight * float(top.get("frame_score", 0))
                            + path_weight * float(top.get("path_score", 0))
                        )
                        / weight_total,
                        6,
                    )
                    top["selection_strategy"] = "small_frame_semantic_context"
                    selection_info = {
                        "strategy": "small_frame_semantic_context",
                        "document_branch_decision": branch_info,
                        "original_top_hid": branch_info["inside_hid"],
                        "selected_hid": top["pcf_hid"],
                    }
            selected, lca_info = _select_lca(
                top, frame_candidates, parent_by_hid, settings
            )
            if not lca_info:
                selected, parent_info = _promote_broad_parent(
                    selected, frame_candidates, packet, settings
                )
                if parent_info:
                    parent_info["previous_selection"] = selection_info
                    selection_info = parent_info
            alternatives = [
                item for item in frame_candidates if item["pcf_hid"] != selected["pcf_hid"]
            ]
            second_score = float(alternatives[0]["score"]) if alternatives else 0.0
            margin = float(selected["score"]) - second_score
            min_score = float(settings.get("min_mapping_score", 0.16))
            min_coverage = float(settings.get("min_evidence_coverage", 0.10))
            if float(selected["score"]) < min_score:
                review_reasons.append("low_score")
            if float(selected["support_coverage"]) < min_coverage:
                review_reasons.append("low_evidence_coverage")
            if margin < float(settings.get("auto_accept_margin", 0.035)):
                review_reasons.append("low_margin")
            if member_count <= 1:
                review_reasons.append("single_mention_taskframe")
            frame_winner = max(frame_candidates, key=lambda item: float(item.get("frame_score", 0)))
            process_winner = max(
                frame_candidates, key=lambda item: float(item.get("process_signal", 0))
            )
            if _root(frame_winner) != _root(process_winner):
                review_reasons.append("frame_vs_evidence_conflict")
            competing_roots = [
                item
                for item in frame_candidates
                if _root(item) != _root(selected)
                and float(item.get("process_signal", 0))
                >= 0.90 * float(selected.get("process_signal", 0))
                and float(item.get("score", 0)) >= min_score
            ]
            if competing_roots:
                review_reasons.append("cross_root_conflict")
            if selected.get("penalty_reasons") and "context_only_support" in selected["penalty_reasons"]:
                review_reasons.append("context_only_support")
            viable = (
                float(selected["score"]) >= min_score
                and float(selected["support_coverage"]) >= min_coverage
            )
            auto_accept = viable and (
                float(selected["score"]) >= float(settings.get("auto_accept_score", 0.34))
                and margin >= float(settings.get("auto_accept_margin", 0.035))
                and float(selected["support_coverage"])
                >= float(settings.get("auto_accept_coverage", 0.30))
                and not review_reasons
            )
            validation_status = "auto_accepted" if auto_accept else "needs_review"
            status_counts[validation_status] += 1
            if not viable:
                links.append(
                    {
                        "mapping_id": stable_id("MAP", frame_id, "unmapped"),
                        "taskframe_id": frame_id,
                        "pcfid": None,
                        "pcf_hid": None,
                        "pcf_name_ko": "",
                        "pcf_path_ids": [],
                        "mapping_role": "process_match",
                        "is_primary": False,
                        "relation_scope": "unmapped",
                        "coverage_score": round(float(selected.get("support_coverage", 0)), 6),
                        "confidence": round(_confidence(selected, margin), 6),
                        "method": "semantic_document_context_gate_evidence_hierarchy_rerank",
                        "validation_status": "needs_review",
                        "supporting_task_mentions": [],
                        "supporting_triples": [],
                        "supporting_evidence_ids": [],
                        "rationale": (
                            f"1위 후보 {selected['pcf_hid']}가 최소 score/coverage 기준을 "
                            "충족하지 않아 자동 매핑하지 않음"
                        ),
                        "candidate_score": selected.get("score", 0.0),
                    }
                )
            else:
                confidence = _confidence(selected, margin)
                relation = _relation_scope(selected, packet, lca_info)
                rationale = (
                    f"대표 mention/fact 근거가 PCF {selected['pcf_hid']} "
                    f"'{selected.get('pcf_name_ko', '')}'를 지지함"
                )
                if lca_info:
                    rationale += (
                        f"; {len(lca_info['branches'])}개 하위 가지에 근거가 분산되어 "
                        "최소 공통 조상을 선택함"
                    )
                links.append(
                    _mapping_record(
                        packet,
                        selected,
                        "process_match",
                        True,
                        relation,
                        confidence,
                        validation_status,
                        rationale,
                    )
                )

                # 문맥 신호가 실제 업무 신호보다 뚜렷한 다른 root만 context anchor로 둔다.
                context_options = [
                    item
                    for item in frame_candidates
                    if _root(item) != _root(selected)
                    and int(item.get("pcf_level", 99)) <= 3
                    and not item.get("is_leaf", False)
                    and float(item.get("score", 0)) >= 0.75 * min_score
                    and float(item.get("support_coverage", 0)) >= min_coverage
                    and not item.get("reranker_penalty_reasons")
                    and float(item.get("context_signal", 0))
                    >= float(settings.get("context_anchor_score", 0.28))
                    and float(item.get("section_score", 0))
                    >= float(settings.get("context_anchor_section_score", 0.40))
                    and float(item.get("context_signal", 0))
                    >= 0.90 * float(item.get("process_signal", 0))
                ]
                if context_options:
                    context = max(
                        context_options,
                        key=lambda item: (float(item.get("context_signal", 0)), float(item["score"])),
                    )
                    links.append(
                        _mapping_record(
                            packet,
                            context,
                            "context_anchor",
                            False,
                            "related",
                            clamp(0.60 * float(context["context_signal"]) + 0.40 * float(context["path_score"])),
                            "needs_review",
                            "section 및 context Fact가 해당 PCF 경로를 수행 문맥으로 지지함",
                        )
                    )
                    review_reasons.append("context_anchor_candidate")

                # 독립 근거를 가진 다른 root의 강한 후보 한 개만 secondary로 보존한다.
                primary_support = _process_support_ids(selected)
                secondary_options: list[tuple[dict[str, Any], set[str]]] = []
                for item in frame_candidates:
                    if _root(item) == _root(selected):
                        continue
                    if float(item["score"]) < float(settings.get("secondary_score_ratio", 0.92)) * float(selected["score"]):
                        continue
                    exclusive = _process_support_ids(item) - primary_support
                    if len(exclusive) >= 2:
                        secondary_options.append((item, exclusive))
                if secondary_options:
                    secondary, exclusive = max(
                        secondary_options, key=lambda value: float(value[0]["score"])
                    )
                    links.append(
                        _mapping_record(
                            packet,
                            secondary,
                            "process_match",
                            False,
                            "partial_overlap",
                            _confidence(secondary, float(selected["score"]) - float(secondary["score"])),
                            "needs_review",
                            f"primary와 겹치지 않는 {len(exclusive)}개 원자 근거가 별도 PCF 가지를 지지함",
                        )
                    )
                    review_reasons.extend(["possible_taskframe_split", "multiple_process_branches"])

        review_reasons = unique_text(review_reasons)
        if review_reasons:
            review_queue.append(
                {
                    "taskframe_id": frame_id,
                    "canonical_name": packet.get("frame_summary", {}).get("canonical_name", ""),
                    "review_reasons": review_reasons,
                    "selected_pcf_hid": selected.get("pcf_hid") if selected else None,
                    "selected_pcf_name_ko": selected.get("pcf_name_ko", "") if selected else "",
                    "score": selected.get("score", 0.0) if selected else 0.0,
                    "margin": round(margin, 6),
                    "support_coverage": selected.get("support_coverage", 0.0) if selected else 0.0,
                    "lca_decision": lca_info,
                    "selection_decision": selection_info,
                    "top_candidates": [
                        {
                            "pcf_hid": item["pcf_hid"],
                            "pcf_name_ko": item.get("pcf_name_ko", ""),
                            "score": item["score"],
                            "process_signal": item.get("process_signal", 0.0),
                            "context_signal": item.get("context_signal", 0.0),
                            "support_coverage": item.get("support_coverage", 0.0),
                        }
                        for item in frame_candidates[:5]
                    ],
                }
            )
        decisions.append(
            {
                "taskframe_id": frame_id,
                "selected_pcf_hid": selected.get("pcf_hid") if selected else None,
                "review_reasons": review_reasons,
                "lca_decision": lca_info,
                "selection_decision": selection_info,
            }
        )

    summary = {
        "taskframe_count": len(packets),
        "mapping_count": len(links),
        "primary_mapping_count": sum(
            item.get("mapping_role") == "process_match" and item.get("is_primary") for item in links
        ),
        "secondary_mapping_count": sum(
            item.get("mapping_role") == "process_match" and not item.get("is_primary")
            and item.get("relation_scope") != "unmapped"
            for item in links
        ),
        "context_anchor_count": sum(item.get("mapping_role") == "context_anchor" for item in links),
        "unmapped_count": sum(item.get("relation_scope") == "unmapped" for item in links),
        "review_queue_count": len(review_queue),
        "status_counts": dict(status_counts),
        "decisions": decisions,
    }
    write_jsonl(output_path(config, "taskframe_pcf_links.jsonl"), links)
    write_jsonl(output_path(config, "review_queue.jsonl"), review_queue)
    write_json(output_path(config, "aggregation_summary.json"), summary)
    return links, review_queue, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCF 후보 근거를 계층적으로 집계합니다.")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    links, queue, summary = aggregate(load_config(args.config))
    print(
        f"Mappings: {len(links)}, primary={summary['primary_mapping_count']}, "
        f"review={len(queue)}"
    )


if __name__ == "__main__":
    main()
