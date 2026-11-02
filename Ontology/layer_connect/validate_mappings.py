from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import (
    config_path,
    load_config,
    normalize_text,
    output_path,
    read_json,
    read_json_records,
    read_jsonl,
    unique_text,
    write_json,
)


REQUIRED_LINK_FIELDS = {
    "mapping_id",
    "taskframe_id",
    "pcf_hid",
    "mapping_role",
    "is_primary",
    "relation_scope",
    "coverage_score",
    "confidence",
    "method",
    "validation_status",
    "supporting_task_mentions",
    "supporting_triples",
    "supporting_evidence_ids",
    "rationale",
}
VALID_ROLES = {"process_match", "context_anchor"}
VALID_SCOPES = {
    "equivalent",
    "taskframe_narrower_than_pcf",
    "taskframe_broader_than_pcf",
    "partial_overlap",
    "related",
    "unmapped",
}


def validate(config: dict[str, Any]) -> dict[str, Any]:
    links = read_jsonl(output_path(config, "taskframe_pcf_links.jsonl"))
    candidates = read_jsonl(output_path(config, "taskframe_pcf_candidates.jsonl"))
    nodes = read_jsonl(output_path(config, "pcf_nodes_enriched.jsonl"))
    packets = read_jsonl(output_path(config, "taskframe_evidence_packets.jsonl"))
    review_queue = read_jsonl(output_path(config, "review_queue.jsonl"))
    document_context = read_json(output_path(config, "document_pcf_context.json"))
    frames = read_json_records(config_path(config, "taskframes"), ("taskframes", "frames", "data"))
    facts = read_jsonl(config_path(config, "canonical_facts"))

    errors: list[str] = []
    warnings: list[str] = []
    frame_ids = {item["taskframe_id"] for item in frames}
    node_by_hid = {item["hid"]: item for item in nodes}
    document_context_hid = normalize_text(document_context.get("selected_pcf_hid"))
    if document_context_hid not in node_by_hid:
        errors.append(
            f"문서 PCF context가 존재하지 않는 노드입니다: {document_context_hid or '(empty)'}"
        )
    elif int(node_by_hid[document_context_hid].get("level", -1)) != int(
        config.get("document_context", {}).get("anchor_level", 2)
    ):
        errors.append("문서 PCF context의 level이 설정과 다릅니다.")
    if not bool(document_context.get("semantic_required", False)):
        errors.append("문서 PCF context가 필수 의미 임베딩 결과가 아닙니다.")
    packet_by_frame = {item["taskframe_id"]: item for item in packets}
    candidates_by_pair = {
        (item["taskframe_id"], item["pcf_hid"]): item for item in candidates
    }
    member_by_frame = {
        item["taskframe_id"]: set(item.get("member_task_mentions", [])) for item in frames
    }
    frame_by_mention = {
        mention_id: frame_id
        for frame_id, mention_ids in member_by_frame.items()
        for mention_id in mention_ids
    }
    facts_by_id = {item["triple_id"]: item for item in facts}

    mapping_ids = [item.get("mapping_id") for item in links]
    if len(mapping_ids) != len(set(mapping_ids)):
        errors.append("mapping_id가 중복됩니다.")
    unknown_frames = sorted({item.get("taskframe_id") for item in links} - frame_ids)
    if unknown_frames:
        errors.append(f"존재하지 않는 TaskFrame 링크: {unknown_frames}")
    missing_frames = sorted(frame_ids - {item.get("taskframe_id") for item in links})
    if missing_frames:
        errors.append(f"매핑 결과가 없는 TaskFrame: {missing_frames}")

    primary_counts: Counter[str] = Counter()
    link_issue_counts: Counter[str] = Counter()
    for index, link in enumerate(links, 1):
        label = f"link#{index}:{link.get('mapping_id', '(no id)')}"
        missing = REQUIRED_LINK_FIELDS - set(link)
        if missing:
            errors.append(f"{label}: 필수 필드 누락 {sorted(missing)}")
            continue
        frame_id = link["taskframe_id"]
        if link["mapping_role"] not in VALID_ROLES:
            errors.append(f"{label}: 잘못된 mapping_role")
        if link["relation_scope"] not in VALID_SCOPES:
            errors.append(f"{label}: 잘못된 relation_scope")
        for key in ("coverage_score", "confidence"):
            value = link[key]
            if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
                errors.append(f"{label}: {key}가 0~1 범위가 아닙니다.")
        if link["is_primary"]:
            primary_counts[frame_id] += 1
            if link["mapping_role"] != "process_match":
                errors.append(f"{label}: context_anchor는 primary일 수 없습니다.")
        hid = link.get("pcf_hid")
        if link["relation_scope"] == "unmapped":
            if hid is not None:
                errors.append(f"{label}: unmapped인데 pcf_hid가 있습니다.")
            continue
        node = node_by_hid.get(hid)
        if node is None:
            errors.append(f"{label}: 존재하지 않는 PCF hid {hid}")
            continue
        if link.get("pcfid") != node.get("pcfid"):
            errors.append(f"{label}: pcfid가 PCF 원본과 다릅니다.")
        if link.get("pcf_path_ids") != node.get("path_ids"):
            errors.append(f"{label}: pcf_path_ids가 복원된 트리 경로와 다릅니다.")
        if (frame_id, hid) not in candidates_by_pair:
            warnings.append(f"{label}: 후보 진단 파일에 동일한 후보가 없습니다.")

        member_ids = member_by_frame.get(frame_id, set())
        bad_mentions = sorted(set(link["supporting_task_mentions"]) - member_ids)
        if bad_mentions:
            errors.append(f"{label}: 다른 frame의 TaskMention 근거 {bad_mentions[:10]}")
        bad_triples = sorted(set(link["supporting_triples"]) - set(facts_by_id))
        if bad_triples:
            errors.append(f"{label}: 존재하지 않는 triple 근거 {bad_triples[:10]}")
        for triple_id in set(link["supporting_triples"]) - set(bad_triples):
            fact = facts_by_id[triple_id]
            direct = any(
                item.get("taskframe_id") == frame_id
                for item in fact.get("taskframe_links", [])
            )
            via = any(
                frame_by_mention.get(item.get("task_mention_id")) == frame_id
                for item in fact.get("task_links", [])
            )
            if not direct and not via:
                errors.append(f"{label}: frame에 연결되지 않은 triple 근거 {triple_id}")

        packet = packet_by_frame.get(frame_id, {})
        packet_evidence = set(packet.get("evidence_ids", []))
        outside = sorted(set(link["supporting_evidence_ids"]) - packet_evidence)
        if outside:
            errors.append(f"{label}: TaskFrame 범위 밖 Evidence {outside[:10]}")
        candidate = candidates_by_pair.get((frame_id, hid))
        if candidate and abs(float(link["coverage_score"]) - float(candidate["support_coverage"])) > 1e-6:
            errors.append(f"{label}: candidate와 coverage_score가 다릅니다.")

    for frame_id, count in primary_counts.items():
        if count > 1:
            errors.append(f"{frame_id}: primary process_match가 {count}개입니다.")
    for frame_id in frame_ids:
        frame_links = [item for item in links if item.get("taskframe_id") == frame_id]
        is_unmapped = any(item.get("relation_scope") == "unmapped" for item in frame_links)
        if not is_unmapped and primary_counts[frame_id] != 1:
            errors.append(f"{frame_id}: mapped frame의 primary가 정확히 1개가 아닙니다.")

    review_ids = [item.get("taskframe_id") for item in review_queue]
    if len(review_ids) != len(set(review_ids)):
        errors.append("review_queue에 같은 TaskFrame이 중복됩니다.")
    needs_review_frames = {
        item["taskframe_id"]
        for item in links
        if item.get("validation_status") == "needs_review"
    }
    missing_review = sorted(needs_review_frames - set(review_ids))
    if missing_review:
        warnings.append(f"needs_review 링크가 있으나 review_queue에 없는 frame: {missing_review}")

    candidate_hids_by_frame: dict[str, set[str]] = defaultdict(set)
    for item in candidates:
        candidate_hids_by_frame[item["taskframe_id"]].add(item["pcf_hid"])
    for frame_id, hids in candidate_hids_by_frame.items():
        if len(hids) != len([item for item in candidates if item["taskframe_id"] == frame_id]):
            errors.append(f"{frame_id}: 후보 pcf_hid가 중복됩니다.")

    report = {
        "valid": not errors,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": unique_text(errors),
        "warnings": unique_text(warnings),
        "counts": {
            "pcf_nodes": len(nodes),
            "taskframes": len(frames),
            "packets": len(packets),
            "candidate_rows": len(candidates),
            "mapping_rows": len(links),
            "primary_mappings": sum(primary_counts.values()),
            "review_queue": len(review_queue),
        },
        "mapping_role_counts": dict(Counter(item.get("mapping_role") for item in links)),
        "relation_scope_counts": dict(Counter(item.get("relation_scope") for item in links)),
        "validation_status_counts": dict(
            Counter(item.get("validation_status") for item in links)
        ),
        "link_issue_counts": dict(link_issue_counts),
        "document_context": {
            "pcf_hid": document_context_hid,
            "pcf_name_ko": document_context.get("selected_pcf_name_ko", ""),
            "score": document_context.get("selected_document_score", 0.0),
            "margin": document_context.get("score_margin", 0.0),
        },
    }
    write_json(output_path(config, "validation_report.json"), report)
    if errors:
        raise ValueError(f"Crosswalk 검증 실패: {len(errors)}개 오류")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCF–TaskFrame Crosswalk를 검증합니다.")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate(load_config(args.config))
    print(
        f"Validation: valid={report['valid']}, "
        f"warnings={report['warning_count']}, mappings={report['counts']['mapping_rows']}"
    )


if __name__ == "__main__":
    main()
