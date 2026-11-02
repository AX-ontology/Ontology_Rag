from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import load_config, output_path, read_jsonl, write_json


def select_document_context(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select the PCF subtree that best explains the document as a whole.

    A document is represented by the complete set of its TaskFrames.  Each
    candidate subtree receives the best frame/path semantic match per
    TaskFrame, followed by a document-wide mean and coverage aggregation.
    """
    candidates = read_jsonl(output_path(config, "taskframe_pcf_candidates.jsonl"))
    packets = read_jsonl(output_path(config, "taskframe_evidence_packets.jsonl"))
    nodes = read_jsonl(output_path(config, "pcf_nodes_enriched.jsonl"))
    if not candidates or not packets or not nodes:
        raise ValueError("candidate, packet, PCF index를 먼저 생성해야 합니다.")

    settings = config.get("document_context", {})
    anchor_level = int(settings.get("anchor_level", 2))
    frame_weight = float(settings.get("frame_weight", 0.65))
    path_weight = float(settings.get("path_weight", 0.35))
    semantic_total = frame_weight + path_weight
    if semantic_total <= 0:
        raise ValueError("document_context frame/path weight 합은 0보다 커야 합니다.")
    frame_weight /= semantic_total
    path_weight /= semantic_total
    mean_weight = float(settings.get("mean_weight", 0.85))
    coverage_weight = float(settings.get("coverage_weight", 0.15))
    document_total = mean_weight + coverage_weight
    if document_total <= 0:
        raise ValueError("document_context mean/coverage weight 합은 0보다 커야 합니다.")
    mean_weight /= document_total
    coverage_weight /= document_total

    node_by_hid = {node["hid"]: node for node in nodes}
    frame_ids = [packet["taskframe_id"] for packet in packets]
    frame_name = {
        packet["taskframe_id"]: packet.get("frame_summary", {}).get("canonical_name", "")
        for packet in packets
    }
    best_by_branch: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for candidate in candidates:
        path = candidate.get("pcf_path_ids", [])
        branch_hid = next(
            (
                hid
                for hid in path
                if int(node_by_hid.get(hid, {}).get("level", -1)) == anchor_level
            ),
            None,
        )
        if not branch_hid:
            continue
        semantic_score = (
            frame_weight * float(candidate.get("frame_score", 0))
            + path_weight * float(candidate.get("path_score", 0))
        )
        frame_id = candidate["taskframe_id"]
        previous = best_by_branch[branch_hid].get(frame_id)
        if previous is None or semantic_score > float(previous["semantic_score"]):
            best_by_branch[branch_hid][frame_id] = {
                "taskframe_id": frame_id,
                "taskframe_name": frame_name.get(frame_id, ""),
                "candidate_hid": candidate["pcf_hid"],
                "candidate_name_ko": candidate.get("pcf_name_ko", ""),
                "semantic_score": round(semantic_score, 6),
            }

    rankings: list[dict[str, Any]] = []
    for branch_hid, by_frame in best_by_branch.items():
        values = [float(by_frame.get(frame_id, {}).get("semantic_score", 0)) for frame_id in frame_ids]
        coverage = sum(value > 0 for value in values) / max(1, len(frame_ids))
        mean_score = sum(values) / max(1, len(frame_ids))
        document_score = mean_weight * mean_score + coverage_weight * coverage
        node = node_by_hid[branch_hid]
        rankings.append(
            {
                "pcf_hid": branch_hid,
                "pcf_name_ko": node.get("nameKo", ""),
                "pcf_path_ids": node.get("path_ids", []),
                "anchor_level": anchor_level,
                "document_score": round(document_score, 6),
                "mean_semantic_score": round(mean_score, 6),
                "taskframe_coverage": round(coverage, 6),
                "matched_taskframe_count": sum(value > 0 for value in values),
                "taskframe_matches": [
                    by_frame[frame_id] for frame_id in frame_ids if frame_id in by_frame
                ],
            }
        )
    if not rankings:
        raise RuntimeError(f"PCF level {anchor_level} 문서 문맥 후보를 만들지 못했습니다.")
    rankings.sort(
        key=lambda item: (
            float(item["document_score"]),
            float(item["taskframe_coverage"]),
            float(item["mean_semantic_score"]),
        ),
        reverse=True,
    )
    for rank, item in enumerate(rankings, 1):
        item["rank"] = rank
    selected = rankings[0]
    report = {
        "method": "document_taskframe_set_to_pcf_subtree_semantic_coverage",
        "semantic_required": bool(
            config.get("retrieval", {}).get("semantic", {}).get("required", False)
        ),
        "taskframe_count": len(frame_ids),
        "anchor_level": anchor_level,
        "selected_pcf_hid": selected["pcf_hid"],
        "selected_pcf_name_ko": selected["pcf_name_ko"],
        "selected_document_score": selected["document_score"],
        "selected_taskframe_coverage": selected["taskframe_coverage"],
        "score_margin": round(
            float(selected["document_score"])
            - (float(rankings[1]["document_score"]) if len(rankings) > 1 else 0.0),
            6,
        ),
        "weights": {
            "frame": frame_weight,
            "path": path_weight,
            "mean": mean_weight,
            "coverage": coverage_weight,
        },
        "rankings": rankings,
    }
    write_json(output_path(config, "document_pcf_context.json"), report)
    return selected, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="문서 전체의 PCF context subtree를 선택합니다.")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected, report = select_document_context(load_config(args.config))
    print(
        f"Document PCF context: {selected['pcf_hid']} {selected['pcf_name_ko']} "
        f"(score={report['selected_document_score']:.3f}, "
        f"margin={report['score_margin']:.3f})"
    )


if __name__ == "__main__":
    main()
