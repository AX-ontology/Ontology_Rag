from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from common import (
    RELEVANCE_ORDER,
    clamp,
    config_path,
    load_config,
    normalize_text,
    output_path,
    read_json_records,
    read_jsonl,
    strongest_relevance,
    unique_text,
    write_json,
    write_jsonl,
)


def _stratified_unique(
    items: list[dict[str, Any]],
    limit: int,
    signature: Callable[[dict[str, Any]], str],
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    best_by_signature: dict[str, dict[str, Any]] = {}
    for item in items:
        key = signature(item).casefold() or normalize_text(
            item.get("task_mention_id") or item.get("triple_id")
        )
        previous = best_by_signature.get(key)
        rank = (clamp(item.get("confidence")), len(normalize_text(item.get("source_text"))))
        previous_rank = (
            (clamp(previous.get("confidence")), len(normalize_text(previous.get("source_text"))))
            if previous
            else (-1.0, -1)
        )
        if previous is None or rank > previous_rank:
            best_by_signature[key] = item

    candidates = sorted(
        best_by_signature.values(),
        key=lambda item: (-clamp(item.get("confidence")), normalize_text(item.get("section_path"))),
    )
    by_section: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        by_section[normalize_text(item.get("section_path"))].append(item)
    selected: list[dict[str, Any]] = []
    positions: dict[str, int] = {key: 0 for key in by_section}
    sections = sorted(
        by_section,
        key=lambda key: (-clamp(by_section[key][0].get("confidence")), key),
    )
    while len(selected) < min(limit, len(candidates)):
        changed = False
        for section in sections:
            position = positions[section]
            if position < len(by_section[section]):
                selected.append(by_section[section][position])
                positions[section] += 1
                changed = True
                if len(selected) == limit:
                    break
        if not changed:
            break
    return selected


def _mention_record(item: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item.get(key)
        for key in (
            "task_mention_id",
            "surface_form",
            "action_normalized",
            "focal_object_normalized",
            "intended_outcome_raw",
            "section_path",
            "source_text",
            "confidence",
            "surface_grounding",
            "evidence_ids",
        )
    }


def _fact_record(
    item: dict[str, Any], link: dict[str, Any], relevance: str
) -> dict[str, Any]:
    subject = normalize_text(
        item.get("subject_canonical")
        or item.get("subject_normalized")
        or item.get("subject_raw")
    )
    predicate = normalize_text(
        item.get("predicate_canonical")
        or item.get("predicate_normalized")
        or item.get("predicate_raw")
    )
    obj = normalize_text(
        item.get("object_canonical")
        or item.get("object_normalized")
        or item.get("object_raw")
    )
    spo = " / ".join(value for value in (subject, predicate, obj) if value)
    naturalized = " ".join(value for value in (subject, obj, predicate) if value)
    return {
        "triple_id": item.get("triple_id"),
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "spo_text": spo,
        "naturalized_text": naturalized,
        "knowledge_function": item.get("knowledge_function", "fact"),
        "relevance": relevance,
        "via_task_mentions": sorted(link.get("via_task_mentions", [])),
        "propagation_paths": sorted(link.get("propagation_paths", [])),
        "section_path": item.get("section_path", ""),
        "evidence_quote": item.get("evidence_quote", ""),
        "source_text": item.get("source_text", ""),
        "confidence": clamp(item.get("confidence"), 0.5),
        "evidence_ids": unique_text(item.get("evidence_ids", [])),
    }


def build_packets(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frame_path = config_path(config, "taskframes")
    mention_path = config_path(config, "task_mentions")
    fact_path = config_path(config, "canonical_facts")
    frames = read_json_records(frame_path, ("taskframes", "frames", "data"))
    mentions = read_jsonl(mention_path)
    facts = read_jsonl(fact_path)

    frame_by_id = {normalize_text(item.get("taskframe_id")): item for item in frames}
    mention_by_id = {normalize_text(item.get("task_mention_id")): item for item in mentions}
    mention_to_frame: dict[str, str] = {}
    warnings: list[str] = []
    for frame_id, frame in frame_by_id.items():
        for mention_id in frame.get("member_task_mentions", []):
            mention_id = normalize_text(mention_id)
            if mention_id in mention_to_frame and mention_to_frame[mention_id] != frame_id:
                warnings.append(f"TaskMention이 여러 frame에 속함: {mention_id}")
            mention_to_frame[mention_id] = frame_id

    fact_links: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    invalid_frame_links = 0
    invalid_mention_links = 0
    for fact in facts:
        triple_id = normalize_text(fact.get("triple_id"))
        for raw_link in fact.get("taskframe_links", []):
            frame_id = normalize_text(raw_link.get("taskframe_id"))
            if frame_id not in frame_by_id:
                invalid_frame_links += 1
                continue
            relevance = normalize_text(raw_link.get("relevance"))
            relevance = relevance if relevance in RELEVANCE_ORDER else "context"
            previous = fact_links[frame_id].get(triple_id)
            if previous:
                previous["relevance"] = strongest_relevance(previous["relevance"], relevance)
                previous["via_task_mentions"].update(raw_link.get("via_task_mentions", []))
                previous["propagation_paths"].add(
                    normalize_text(raw_link.get("propagation_path")) or "Fact→TaskFrame"
                )
            else:
                fact_links[frame_id][triple_id] = {
                    "fact": fact,
                    "relevance": relevance,
                    "via_task_mentions": set(raw_link.get("via_task_mentions", [])),
                    "propagation_paths": {
                        normalize_text(raw_link.get("propagation_path")) or "Fact→TaskFrame"
                    },
                }

        for task_link in fact.get("task_links", []):
            mention_id = normalize_text(task_link.get("task_mention_id"))
            frame_id = mention_to_frame.get(mention_id)
            if not frame_id:
                invalid_mention_links += 1
                continue
            relevance = normalize_text(task_link.get("relevance"))
            relevance = relevance if relevance in RELEVANCE_ORDER else "context"
            previous = fact_links[frame_id].get(triple_id)
            if previous:
                previous["relevance"] = strongest_relevance(previous["relevance"], relevance)
                previous["via_task_mentions"].add(mention_id)
                previous["propagation_paths"].add("Fact→TaskMention→TaskFrame")
            else:
                fact_links[frame_id][triple_id] = {
                    "fact": fact,
                    "relevance": relevance,
                    "via_task_mentions": {mention_id},
                    "propagation_paths": {"Fact→TaskMention→TaskFrame"},
                }

    limits = config.get("packet_limits", {})
    packets: list[dict[str, Any]] = []
    missing_mentions = 0
    relevance_totals: Counter[str] = Counter()
    for frame_id, frame in frame_by_id.items():
        member_ids = [normalize_text(value) for value in frame.get("member_task_mentions", [])]
        frame_mentions: list[dict[str, Any]] = []
        for mention_id in member_ids:
            mention = mention_by_id.get(mention_id)
            if mention is None:
                missing_mentions += 1
                continue
            frame_mentions.append(_mention_record(mention))
        selected_mentions = _stratified_unique(
            frame_mentions,
            int(limits.get("mentions", 30)),
            lambda item: "|".join(
                unique_text(
                    [
                        item.get("action_normalized"),
                        item.get("focal_object_normalized"),
                    ]
                )
            ) or normalize_text(item.get("surface_form")),
        )

        grouped_facts: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for link in fact_links.get(frame_id, {}).values():
            relevance = link["relevance"]
            grouped_facts[relevance].append(_fact_record(link["fact"], link, relevance))
            relevance_totals[relevance] += 1
        selected_facts: dict[str, list[dict[str, Any]]] = {}
        for relevance, key in (
            ("core", "core_facts"),
            ("supporting", "supporting_facts"),
            ("context", "context_facts"),
        ):
            selected_facts[relevance] = _stratified_unique(
                grouped_facts.get(relevance, []),
                int(limits.get(key, 0)),
                lambda item: normalize_text(item.get("spo_text")),
            )

        sections = unique_text(
            [item.get("section_path") for item in frame_mentions]
            + [
                item.get("section_path")
                for group in selected_facts.values()
                for item in group
            ]
        )
        frame_evidence_ids = unique_text(frame.get("evidence_ids", []))
        packet_evidence_ids = unique_text(
            frame_evidence_ids
            + [
                evidence_id
                for item in frame_mentions
                for evidence_id in item.get("evidence_ids", [])
            ]
            + [
                evidence_id
                for group in selected_facts.values()
                for item in group
                for evidence_id in item.get("evidence_ids", [])
            ]
        )
        packet = {
            "taskframe_id": frame_id,
            "frame_summary": {
                "canonical_name": normalize_text(frame.get("canonical_name")),
                "definition": normalize_text(frame.get("definition")),
                "scope": normalize_text(frame.get("scope")),
            },
            "mentions": selected_mentions,
            "core_facts": selected_facts["core"],
            "supporting_facts": selected_facts["supporting"],
            "context_facts": selected_facts["context"],
            "section_paths": sections,
            "frame_evidence_ids": frame_evidence_ids,
            "evidence_ids": packet_evidence_ids,
            "source_counts": {
                "member_mentions": len(frame_mentions),
                "linked_core_facts": len(grouped_facts.get("core", [])),
                "linked_supporting_facts": len(grouped_facts.get("supporting", [])),
                "linked_context_facts": len(grouped_facts.get("context", [])),
            },
            "selected_counts": {
                "mentions": len(selected_mentions),
                "core_facts": len(selected_facts["core"]),
                "supporting_facts": len(selected_facts["supporting"]),
                "context_facts": len(selected_facts["context"]),
            },
        }
        packets.append(packet)

    report = {
        "valid": not missing_mentions and not invalid_frame_links,
        "taskframe_count": len(frames),
        "task_mention_count": len(mentions),
        "fact_count": len(facts),
        "packet_count": len(packets),
        "assigned_mention_count": len(mention_to_frame),
        "missing_member_mentions": missing_mentions,
        "invalid_fact_frame_links": invalid_frame_links,
        "invalid_fact_mention_links": invalid_mention_links,
        "linked_fact_counts": dict(relevance_totals),
        "warnings": warnings,
        "sources": {
            "taskframes": str(frame_path),
            "task_mentions": str(mention_path),
            "canonical_facts": str(fact_path),
        },
    }
    write_jsonl(output_path(config, "taskframe_evidence_packets.jsonl"), packets)
    write_json(output_path(config, "packet_validation.json"), report)
    if not report["valid"]:
        raise ValueError("Evidence Packet 입력 연결 검증에 실패했습니다.")
    return packets, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TaskFrame Evidence Packet을 생성합니다.")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    packets, report = build_packets(load_config(args.config))
    print(
        f"Evidence packets: {len(packets)} frames, "
        f"{sum(report['linked_fact_counts'].values())} linked facts"
    )


if __name__ == "__main__":
    main()
