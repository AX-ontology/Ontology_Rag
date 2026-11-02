from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from common import clamp, load_config, normalize_text, output_path, read_jsonl, write_json, write_jsonl


def _specialized_domain_penalty(
    query: str, node: dict[str, Any], amount: float
) -> tuple[float, list[str]]:
    """Penalize taxonomy branches whose explicit domain is absent from the task."""
    if amount <= 0:
        return 0.0, []
    query_upper = normalize_text(query).upper()
    candidate_upper = normalize_text(
        " ".join(
            [
                str(node.get("path_text_ko", "")),
                str(node.get("nameKo", "")),
                str(node.get("descKo", "")),
            ]
        )
    ).upper()
    constraints = (
        (
            "information_technology_scope",
            ("정보기술", "IT ", "IT·", "IT/", "사이버보안", "소프트웨어"),
            ("정보기술", "IT ", "IT·", "IT/", "사이버보안", "소프트웨어"),
        ),
    )
    reasons = [
        name
        for name, candidate_terms, query_terms in constraints
        if any(term in candidate_upper for term in candidate_terms)
        and not any(term in query_upper for term in query_terms)
    ]
    return (amount if reasons else 0.0), reasons


def _frame_query(packet: dict[str, Any]) -> str:
    summary = packet.get("frame_summary", {})
    sections = packet.get("section_paths", [])[:2]
    parts = [
        f"업무명: {summary.get('canonical_name', '')}",
        f"정의: {summary.get('definition', '')}",
        f"범위: {summary.get('scope', '')}",
        "문서 문맥: " + " | ".join(sections),
    ]
    return normalize_text(" ".join(parts))[:700]


def _pcf_passage(node: dict[str, Any]) -> str:
    return normalize_text(
        " ".join(
            [
                f"PCF 경로: {node.get('path_text_ko', '')}",
                f"프로세스: {node.get('nameKo', '')}",
                f"정의: {node.get('descKo', '')}",
                f"영문명: {node.get('nameEn', '')}",
            ]
        )
    )[:1200]


def _run_reranker(
    pairs: list[list[str]], settings: dict[str, Any], cache_dir: Path
) -> tuple[np.ndarray, dict[str, Any]]:
    enabled = bool(settings.get("enabled", False))
    required = bool(settings.get("required", True))
    if not enabled:
        if required:
            raise RuntimeError("reranker.required=true인데 reranker.enabled=false입니다.")
        return np.ones(len(pairs), dtype=np.float32), {"enabled": False}
    model_name = normalize_text(settings.get("model"))
    if not model_name:
        raise ValueError("reranker.model이 필요합니다.")
    worker_python = Path(str(settings.get("python") or sys.executable)).expanduser().resolve()
    if not worker_python.exists():
        raise FileNotFoundError(f"reranker Python이 없습니다: {worker_python}")
    digest = hashlib.sha256(model_name.encode("utf-8"))
    for query, passage in pairs:
        digest.update(b"\0")
        digest.update(query.encode("utf-8"))
        digest.update(b"\1")
        digest.update(passage.encode("utf-8"))
    cache_key = digest.hexdigest()[:24]
    cache_file = cache_dir / f"{cache_key}.npy"
    cache_hit = bool(settings.get("cache_scores", True) and cache_file.exists())
    if cache_hit:
        scores = np.load(cache_file, allow_pickle=False)
    else:
        worker = Path(__file__).with_name("semantic_rerank.py")
        with tempfile.TemporaryDirectory(prefix="layer_connect_rerank_") as temporary:
            temp = Path(temporary)
            input_file = temp / "pairs.json"
            output_file = temp / "scores.npy"
            input_file.write_text(json.dumps(pairs, ensure_ascii=False), encoding="utf-8")
            command = [
                str(worker_python), str(worker), "--input", str(input_file),
                "--output", str(output_file), "--model", model_name,
                "--device", str(settings.get("device", "cpu")),
                "--batch-size", str(max(1, int(settings.get("batch_size", 4)))),
                "--max-length", str(max(32, int(settings.get("max_length", 512)))),
            ]
            if settings.get("offline", False):
                command.append("--offline")
            environment = os.environ.copy()
            environment.setdefault("TOKENIZERS_PARALLELISM", "false")
            environment.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
            result = subprocess.run(
                command, check=False, capture_output=True, text=True, env=environment
            )
            if result.returncode != 0:
                detail = (result.stderr or result.stdout).strip()[-4000:]
                raise RuntimeError(f"semantic reranker 실패(model={model_name}):\n{detail}")
            scores = np.load(output_file, allow_pickle=False)
        if settings.get("cache_scores", True):
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            temporary_cache = cache_file.with_suffix(".npy.tmp")
            with temporary_cache.open("wb") as handle:
                np.save(handle, scores, allow_pickle=False)
            temporary_cache.replace(cache_file)
    if scores.shape != (len(pairs),) or not np.isfinite(scores).all():
        raise RuntimeError(f"reranker score shape/value 오류: {scores.shape}")
    return scores, {
        "enabled": True,
        "required": required,
        "model": model_name,
        "python": str(worker_python),
        "device": str(settings.get("device", "cpu")),
        "pair_count": len(pairs),
        "cache_key": cache_key,
        "cache_hit": cache_hit,
        "score_min": float(scores.min(initial=0.0)),
        "score_max": float(scores.max(initial=0.0)),
        "score_mean": float(scores.mean()) if len(scores) else 0.0,
    }


def rerank(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_path = output_path(config, "taskframe_pcf_candidates.jsonl")
    candidates = read_jsonl(candidate_path)
    packets = read_jsonl(output_path(config, "taskframe_evidence_packets.jsonl"))
    nodes = read_jsonl(output_path(config, "pcf_nodes_enriched.jsonl"))
    packet_by_frame = {item["taskframe_id"]: item for item in packets}
    node_by_hid = {item["hid"]: item for item in nodes}
    settings = config.get("reranker", {})
    top_n = max(1, int(settings.get("top_n", 80)))
    weight = clamp(settings.get("weight"), 0.55)
    by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_frame[candidate["taskframe_id"]].append(candidate)
    selected: list[dict[str, Any]] = []
    selected_penalties: list[tuple[float, list[str]]] = []
    pairs: list[list[str]] = []
    domain_penalty = clamp(settings.get("specialized_domain_penalty"), 0.25)
    for frame_id, values in by_frame.items():
        values.sort(key=lambda item: int(item.get("rank", 10**9)))
        query = _frame_query(packet_by_frame[frame_id])
        for candidate in values[:top_n]:
            candidate["pre_rerank_score"] = candidate["score"]
            selected.append(candidate)
            selected_penalties.append(
                _specialized_domain_penalty(
                    query, node_by_hid[candidate["pcf_hid"]], domain_penalty
                )
            )
            pairs.append([query, _pcf_passage(node_by_hid[candidate["pcf_hid"]])])
        for candidate in values[top_n:]:
            candidate["pre_rerank_score"] = candidate["score"]
            scope_penalty, penalty_reasons = _specialized_domain_penalty(
                query, node_by_hid[candidate["pcf_hid"]], domain_penalty
            )
            candidate["reranker_raw_score"] = None
            candidate["reranker_score"] = None
            candidate["reranker_scope_penalty"] = round(scope_penalty, 6)
            candidate["reranker_penalty_reasons"] = penalty_reasons
            candidate["score"] = round((1.0 - weight) * float(candidate["score"]), 6)

    scores, metadata = _run_reranker(
        pairs, settings, output_path(config, "reranker_cache")
    )
    for candidate, score, (scope_penalty, penalty_reasons) in zip(
        selected, scores, selected_penalties
    ):
        candidate["reranker_raw_score"] = round(float(score), 6)
        candidate["reranker_scope_penalty"] = round(scope_penalty, 6)
        candidate["reranker_penalty_reasons"] = penalty_reasons
        adjusted_score = clamp(float(score) - scope_penalty)
        candidate["reranker_score"] = round(adjusted_score, 6)
        candidate["score"] = round(
            clamp(
                (1.0 - weight) * float(candidate["pre_rerank_score"])
                + weight * adjusted_score
            ),
            6,
        )

    output_records: list[dict[str, Any]] = []
    top_changes: list[dict[str, Any]] = []
    for frame_id, values in by_frame.items():
        before = min(values, key=lambda item: int(item.get("rank", 10**9)))
        values.sort(
            key=lambda item: (
                float(item["score"]),
                float(item.get("process_signal", 0)),
                int(item.get("pcf_level", 0)),
            ),
            reverse=True,
        )
        for rank, item in enumerate(values, 1):
            item["rank"] = rank
            item["retrieval_backend"] += "_cross_encoder_reranked"
            output_records.append(item)
        after = values[0]
        top_changes.append(
            {
                "taskframe_id": frame_id,
                "before_hid": before["pcf_hid"],
                "after_hid": after["pcf_hid"],
                "after_reranker_score": after.get("reranker_score"),
            }
        )
    report = {
        "candidate_count": len(candidates),
        "reranked_count": len(selected),
        "top_n_per_frame": top_n,
        "blend_weight": weight,
        "reranker": metadata,
        "top_changes": top_changes,
    }
    write_jsonl(candidate_path, output_records)
    write_json(output_path(config, "reranker_summary.json"), report)
    return output_records, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embedding 후보를 semantic cross-encoder로 재정렬합니다.")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, report = rerank(load_config(args.config))
    print(
        f"Reranked: {report['reranked_count']}/{len(records)} candidates "
        f"with {report['reranker']['model']}"
    )


if __name__ == "__main__":
    main()
