from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common import (
    clamp,
    config_path,
    load_config,
    normalize_text,
    output_path,
    read_jsonl,
    unique_text,
    write_json,
    write_jsonl,
)


FACT_KIND_WEIGHT = {"core_fact": 1.0, "supporting_fact": 0.6, "context_fact": 0.2}


def _query_records(packet: dict[str, Any]) -> list[dict[str, Any]]:
    frame = packet["frame_summary"]
    atomic_context = " ".join(
        unique_text(
            [
                f"상위 업무 {frame.get('canonical_name', '')}",
                f"업무 정의 {frame.get('definition', '')}",
                f"업무 범위 {frame.get('scope', '')}",
            ]
        )
    )
    records: list[dict[str, Any]] = [
        {
            "query_id": f"{packet['taskframe_id']}:frame",
            "kind": "frame",
            "source_id": packet["taskframe_id"],
            "section_path": "",
            "text": " ".join(
                unique_text(
                    [frame.get("canonical_name"), frame.get("definition"), frame.get("scope")]
                )
            ),
        }
    ]
    section_context = packet.get("section_paths", [])[:12]
    if section_context:
        records.append(
            {
                "query_id": f"{packet['taskframe_id']}:frame_context",
                "kind": "frame_context",
                "source_id": packet["taskframe_id"],
                "section_path": " | ".join(section_context),
                "text": " ".join(
                    unique_text(
                        [
                            frame.get("canonical_name"),
                            frame.get("definition"),
                            frame.get("scope"),
                            "업무 문맥 " + " > ".join(section_context),
                        ]
                    )
                ),
            }
        )
    for mention in packet.get("mentions", []):
        records.append(
            {
                "query_id": f"{packet['taskframe_id']}:{mention['task_mention_id']}",
                "kind": "mention",
                "source_id": mention["task_mention_id"],
                "section_path": normalize_text(mention.get("section_path")),
                "text": " ".join(
                    unique_text(
                        [
                            mention.get("surface_form"),
                            mention.get("action_normalized"),
                            mention.get("focal_object_normalized"),
                            mention.get("intended_outcome_raw"),
                            atomic_context,
                        ]
                    )
                ),
            }
        )
    for packet_key, kind in (
        ("core_facts", "core_fact"),
        ("supporting_facts", "supporting_fact"),
        ("context_facts", "context_fact"),
    ):
        for fact in packet.get(packet_key, []):
            records.append(
                {
                    "query_id": f"{packet['taskframe_id']}:{fact['triple_id']}",
                    "kind": kind,
                    "source_id": fact["triple_id"],
                    "section_path": normalize_text(fact.get("section_path")),
                    "text": " ".join(
                        unique_text(
                            [fact.get("spo_text"), fact.get("naturalized_text")]
                            + [atomic_context]
                        )
                    ),
                }
            )
    for index, section in enumerate(packet.get("section_paths", [])):
        records.append(
            {
                "query_id": f"{packet['taskframe_id']}:section:{index}",
                "kind": "section",
                "source_id": f"section:{index}",
                "section_path": normalize_text(section),
                "text": normalize_text(section),
            }
        )
    return [record for record in records if record["text"]]


def _tfidf_similarity(
    queries: list[str], documents: list[str], analyzer: str
) -> np.ndarray:
    if analyzer == "char_wb":
        vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), min_df=1, sublinear_tf=True
        )
    else:
        vectorizer = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b",
        )
    matrix = vectorizer.fit_transform(documents + queries)
    return cosine_similarity(matrix[len(documents) :], matrix[: len(documents)])


def _semantic_similarities(
    queries: list[str],
    documents: list[str],
    paths: list[str],
    settings: dict[str, Any],
    cache_dir: Path | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    enabled = bool(settings.get("enabled", False))
    required = bool(settings.get("required", True))
    if not enabled:
        if required:
            raise RuntimeError(
                "semantic.required=true인데 semantic.enabled=false입니다. "
                "의미 임베딩 없이는 후보 검색을 실행하지 않습니다."
            )
        return None, None, {"enabled": False, "required": False}

    model_name = normalize_text(settings.get("model"))
    if not model_name:
        raise ValueError("retrieval.semantic.model이 필요합니다.")
    worker_python = Path(
        str(settings.get("python") or sys.executable)
    ).expanduser().resolve()
    if not worker_python.exists():
        raise FileNotFoundError(f"semantic worker Python이 없습니다: {worker_python}")
    worker = Path(__file__).with_name("semantic_encode.py")
    corpus = documents + paths
    use_cache = bool(cache_dir is not None and settings.get("cache_embeddings", True))

    def cache_key(namespace: str, values: list[str]) -> str:
        digest = hashlib.sha256()
        digest.update(model_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(namespace.encode("utf-8"))
        for item in values:
            digest.update(b"\0")
            digest.update(item.encode("utf-8"))
        return digest.hexdigest()[:24]

    query_key = cache_key("queries", queries)
    corpus_key = cache_key("pcf_corpus", corpus)
    query_cache = cache_dir / f"queries-{query_key}.npy" if use_cache else None
    corpus_cache = cache_dir / f"pcf-corpus-{corpus_key}.npy" if use_cache else None
    query_cache_hit = bool(query_cache and query_cache.exists())
    corpus_cache_hit = bool(corpus_cache and corpus_cache.exists())
    query_vectors = (
        np.load(query_cache, allow_pickle=False) if query_cache_hit else None
    )
    corpus_vectors = (
        np.load(corpus_cache, allow_pickle=False) if corpus_cache_hit else None
    )

    missing_groups: list[tuple[str, list[str]]] = []
    if query_vectors is None:
        missing_groups.append(("queries", queries))
    if corpus_vectors is None:
        missing_groups.append(("corpus", corpus))
    missing_texts = [text for _, values in missing_groups for text in values]
    if missing_texts:
        print(
            "[semantic] encoding "
            + ", ".join(f"{name}={len(values)}" for name, values in missing_groups),
            flush=True,
        )
        with tempfile.TemporaryDirectory(prefix="layer_connect_embedding_") as temporary:
            temporary_path = Path(temporary)
            input_path = temporary_path / "texts.json"
            output_file = temporary_path / "embeddings.npy"
            input_path.write_text(
                json.dumps(missing_texts, ensure_ascii=False), encoding="utf-8"
            )
            command = [
                str(worker_python),
                str(worker),
                "--input",
                str(input_path),
                "--output",
                str(output_file),
                "--model",
                model_name,
                "--device",
                str(settings.get("device", "cpu")),
                "--batch-size",
                str(max(1, int(settings.get("batch_size", 64)))),
            ]
            if settings.get("offline", False):
                command.append("--offline")
            environment = os.environ.copy()
            environment.setdefault("TOKENIZERS_PARALLELISM", "false")
            environment.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            if result.returncode != 0:
                detail = (result.stderr or result.stdout).strip()[-4000:]
                raise RuntimeError(
                    f"의미 임베딩 생성 실패(model={model_name}, python={worker_python}):\n{detail}"
                )
            if not output_file.exists():
                raise RuntimeError("semantic worker가 embedding 파일을 생성하지 않았습니다.")
            encoded = np.load(output_file, allow_pickle=False)
        if encoded.ndim != 2 or encoded.shape[0] != len(missing_texts):
            raise RuntimeError(
                "semantic embedding shape 불일치: "
                f"expected ({len(missing_texts)}, d), got {encoded.shape}"
            )
        offset = 0
        for name, values in missing_groups:
            group_vectors = encoded[offset : offset + len(values)]
            offset += len(values)
            if name == "queries":
                query_vectors = group_vectors
                target_cache = query_cache
            else:
                corpus_vectors = group_vectors
                target_cache = corpus_cache
            if target_cache is not None:
                target_cache.parent.mkdir(parents=True, exist_ok=True)
                temporary_cache = target_cache.with_suffix(".npy.tmp")
                with temporary_cache.open("wb") as handle:
                    np.save(handle, group_vectors, allow_pickle=False)
                temporary_cache.replace(target_cache)

    if query_vectors is None or corpus_vectors is None:
        raise RuntimeError("semantic embedding 그룹 생성 실패")
    if query_vectors.ndim != 2 or query_vectors.shape[0] != len(queries):
        raise RuntimeError(
            f"query embedding shape 불일치: {query_vectors.shape}"
        )
    if corpus_vectors.ndim != 2 or corpus_vectors.shape[0] != len(corpus):
        raise RuntimeError(
            f"corpus embedding shape 불일치: {corpus_vectors.shape}"
        )
    if query_vectors.shape[1] != corpus_vectors.shape[1]:
        raise RuntimeError("query/corpus embedding dimension 불일치")
    if not np.isfinite(query_vectors).all() or not np.isfinite(corpus_vectors).all():
        raise RuntimeError("semantic embedding에 NaN/Inf가 포함되어 있습니다.")
    document_vectors = corpus_vectors[: len(documents)]
    path_vectors = corpus_vectors[len(documents) :]
    direct = query_vectors @ document_vectors.T
    path = query_vectors @ path_vectors.T
    metadata = {
        "enabled": True,
        "required": required,
        "model": model_name,
        "python": str(worker_python),
        "device": str(settings.get("device", "cpu")),
        "dimension": int(query_vectors.shape[1]),
        "encoded_text_count": len(queries) + len(corpus),
        "cache_key": {"queries": query_key, "pcf_corpus": corpus_key},
        "cache_hit": query_cache_hit and corpus_cache_hit,
        "cache_hits": {
            "queries": query_cache_hit,
            "pcf_corpus": corpus_cache_hit,
        },
    }
    return direct, path, metadata


def _aggregate_support(
    entries: list[tuple[dict[str, Any], float]], threshold: float,
    relevance_weighted: bool = False,
) -> tuple[float, float, float]:
    if not entries:
        return 0.0, 0.0, 0.0
    weighted: list[tuple[float, float, str, float]] = []
    for query, score in entries:
        weight = FACT_KIND_WEIGHT.get(query["kind"], 1.0) if relevance_weighted else 1.0
        weighted.append((score, weight, query.get("section_path", ""), float(query.get("_support_threshold", threshold))))
    ranked = sorted(weighted, key=lambda value: value[0] * value[1], reverse=True)
    top_count = max(1, min(5, math.ceil(len(ranked) * 0.30)))
    top = ranked[:top_count]
    top_weight = sum(weight for _, weight, _, _ in top)
    top_support = sum(score * weight for score, weight, _, _ in top) / max(top_weight, 1e-9)
    total_weight = sum(weight for _, weight, _, _ in ranked)
    supported_weight = sum(weight for score, weight, _, unit_threshold in ranked if score >= unit_threshold)
    coverage = supported_weight / max(total_weight, 1e-9)
    all_sections = {section for _, _, section, _ in ranked if section}
    supported_sections = {
        section for score, _, section, unit_threshold in ranked if section and score >= unit_threshold
    }
    diversity = (
        len(supported_sections) / len(all_sections) if all_sections else coverage
    )
    score = 0.60 * top_support + 0.25 * coverage + 0.15 * diversity
    return clamp(score), clamp(coverage), clamp(diversity)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def retrieve(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    nodes = read_jsonl(output_path(config, "pcf_nodes_enriched.jsonl"))
    packets = read_jsonl(output_path(config, "taskframe_evidence_packets.jsonl"))
    if not nodes or not packets:
        raise ValueError("PCF index와 Evidence Packet을 먼저 생성해야 합니다.")

    settings = config.get("retrieval", {})
    query_top_k = max(1, int(settings.get("query_top_k", 20)))
    pool_size = max(query_top_k, int(settings.get("candidate_pool_size", 100)))
    output_top_k = max(1, int(settings.get("output_top_k", 30)))
    support_threshold = float(settings.get("atomic_support_threshold", 0.18))
    char_weight = float(settings.get("char_weight", 0.55))
    word_weight = float(settings.get("word_weight", 0.45))
    lexical_weight_total = char_weight + word_weight
    if lexical_weight_total <= 0:
        raise ValueError("char_weight + word_weight는 0보다 커야 합니다.")
    char_weight /= lexical_weight_total
    word_weight /= lexical_weight_total

    # 현재 source query가 한국어이므로 한국어 표현을 독립 검색한다. 한·영을 한 벡터에
    # 합치면 영문 토큰이 문서 벡터 norm을 키워 한국어 유사도를 인위적으로 낮춘다.
    node_texts = [
        " ".join(unique_text([node.get("nameKo"), node.get("descKo")]))
        or " ".join(unique_text([node.get("nameEn"), node.get("descEn")]))
        for node in nodes
    ]
    path_texts = [
        normalize_text(node.get("path_text_ko"))
        or normalize_text(node.get("path_text_en"))
        for node in nodes
    ]
    all_queries: list[dict[str, Any]] = []
    packet_query_ranges: dict[str, tuple[int, int]] = {}
    for packet in packets:
        start = len(all_queries)
        all_queries.extend(_query_records(packet))
        packet_query_ranges[packet["taskframe_id"]] = (start, len(all_queries))
    query_texts = [item["text"] for item in all_queries]

    semantic_settings = settings.get("semantic", {})
    semantic_weight = clamp(semantic_settings.get("weight"), 0.35)
    direct_lexical: np.ndarray | None = None
    path_lexical: np.ndarray | None = None
    if not bool(semantic_settings.get("enabled", False)) or semantic_weight < 1.0:
        direct_char = _tfidf_similarity(query_texts, node_texts, "char_wb")
        direct_word = _tfidf_similarity(query_texts, node_texts, "word")
        direct_lexical = char_weight * direct_char + word_weight * direct_word
        path_char = _tfidf_similarity(query_texts, path_texts, "char_wb")
        path_word = _tfidf_similarity(query_texts, path_texts, "word")
        path_lexical = char_weight * path_char + word_weight * path_word
    semantic_cache_dir = (
        config_path(config, "semantic_cache_dir")
        if config.get("paths", {}).get("semantic_cache_dir")
        else output_path(config, "embedding_cache")
    )
    semantic, semantic_path, semantic_metadata = _semantic_similarities(
        query_texts,
        node_texts,
        path_texts,
        semantic_settings,
        semantic_cache_dir,
    )
    if semantic is not None and semantic_path is not None:
        if semantic_weight >= 1.0:
            direct_scores = semantic
            path_scores = semantic_path
            backend = "semantic_required_sentence_transformer"
        else:
            if direct_lexical is None or path_lexical is None:
                raise RuntimeError("semantic lexical blend 행렬이 생성되지 않았습니다.")
            direct_scores = (
                (1.0 - semantic_weight) * direct_lexical
                + semantic_weight * semantic
            )
            path_scores = (
                (1.0 - semantic_weight) * path_lexical
                + semantic_weight * semantic_path
            )
            backend = "semantic_required_sentence_transformer_hybrid"
    else:
        if direct_lexical is None or path_lexical is None:
            raise RuntimeError("lexical fallback 행렬이 생성되지 않았습니다.")
        direct_scores = direct_lexical
        path_scores = path_lexical
        backend = "hybrid_char_word_tfidf"
    relative_threshold = clamp(settings.get("atomic_relative_threshold"), 0.55)
    query_support_thresholds = np.maximum(
        support_threshold, relative_threshold * np.max(direct_scores, axis=1)
    )
    for query, threshold in zip(all_queries, query_support_thresholds):
        query["_support_threshold"] = float(threshold)

    index_by_hid = {node["hid"]: index for index, node in enumerate(nodes)}
    parent_by_hid = {node["hid"]: normalize_text(node.get("parent")) or None for node in nodes}
    children_by_hid = {node["hid"]: list(node.get("children_hids", [])) for node in nodes}
    component_weights = settings.get("weights", {})
    weights = {
        key: float(component_weights.get(key, default))
        for key, default in (
            ("frame", 0.20),
            ("mention", 0.30),
            ("fact", 0.25),
            ("path", 0.15),
            ("top_atomic", 0.10),
        )
    }
    weight_total = sum(weights.values())
    if weight_total <= 0:
        raise ValueError("retrieval.weights 합은 0보다 커야 합니다.")
    weights = {key: value / weight_total for key, value in weights.items()}

    records: list[dict[str, Any]] = []
    candidate_counts: dict[str, int] = {}
    for packet in packets:
        frame_id = packet["taskframe_id"]
        start, end = packet_query_ranges[frame_id]
        queries = all_queries[start:end]
        frame_direct = direct_scores[start:end]
        frame_path = path_scores[start:end]
        seeded: set[int] = set()
        seed_priority = np.zeros(len(nodes), dtype=float)
        kind_counts: dict[str, int] = defaultdict(int)
        for query in queries:
            kind_counts[query["kind"]] += 1
        base_query_weights = {
            "frame": 1.5,
            "frame_context": 1.3,
            "mention": 1.0,
            "core_fact": 1.0,
            "supporting_fact": 0.6,
            "context_fact": 0.2,
            "section": 0.3,
        }
        required_frame_indices: set[int] = set()
        retrieval_matrix = np.empty_like(frame_direct)
        for query_index, query in enumerate(queries):
            path_weight = 0.55 if query["kind"] in {"frame_context", "section"} else 0.15
            retrieval_matrix[query_index] = (
                (1.0 - path_weight) * frame_direct[query_index]
                + path_weight * frame_path[query_index]
            )
        for query_index in range(len(queries)):
            row = retrieval_matrix[query_index]
            take = min(query_top_k, len(nodes))
            if take == len(nodes):
                top_indices = np.arange(len(nodes))
            else:
                top_indices = np.argpartition(row, -take)[-take:]
            ranked_indices = sorted(
                (int(value) for value in top_indices),
                key=lambda idx: float(row[idx]),
                reverse=True,
            )
            seeded.update(ranked_indices)
            kind = queries[query_index]["kind"]
            query_weight = base_query_weights.get(kind, 0.5) / math.sqrt(
                max(1, kind_counts[kind])
            )
            for rank, node_index in enumerate(ranked_indices, 1):
                seed_priority[node_index] += query_weight * (
                    0.70 / (10.0 + rank) + 0.30 * float(row[node_index])
                )
            if kind in {"frame", "frame_context"}:
                required_frame_indices.update(ranked_indices)
        global_best = np.max(retrieval_matrix, axis=0)
        ranked_seeded = sorted(
            seeded,
            key=lambda idx: (float(seed_priority[idx]), float(global_best[idx])),
            reverse=True,
        )
        base_candidates = ranked_seeded[:pool_size]
        base_candidate_set = set(base_candidates) | required_frame_indices
        expanded = set(base_candidate_set)
        for node_index in base_candidates[: min(20, len(base_candidates))]:
            node = nodes[node_index]
            expanded.update(index_by_hid[hid] for hid in node.get("path_ids", []) if hid in index_by_hid)
            expanded.update(
                index_by_hid[hid]
                for hid in children_by_hid.get(node["hid"], [])
                if hid in index_by_hid
            )
            parent = parent_by_hid[node["hid"]]
            if parent:
                expanded.update(
                    index_by_hid[hid]
                    for hid in children_by_hid.get(parent, [])
                    if hid in index_by_hid
                )
        extra_candidates = sorted(
            expanded - base_candidate_set,
            key=lambda idx: float(global_best[idx]),
            reverse=True,
        )[:pool_size]
        candidate_indices = list(base_candidate_set) + extra_candidates
        required_pool_hids = {
            hid
            for node_index in sorted(
                base_candidate_set, key=lambda idx: float(seed_priority[idx]), reverse=True
            )[: min(10, len(base_candidate_set))]
            for hid in nodes[node_index].get("path_ids", [])
        }
        candidate_index_set = set(candidate_indices)
        candidate_indices.extend(
            index_by_hid[hid]
            for hid in required_pool_hids
            if hid in index_by_hid and index_by_hid[hid] not in candidate_index_set
        )
        candidate_counts[frame_id] = len(candidate_indices)

        candidate_records: list[dict[str, Any]] = []
        for node_index in candidate_indices:
            node = nodes[node_index]
            scores = [float(value) for value in frame_direct[:, node_index]]
            paths = [float(value) for value in frame_path[:, node_index]]
            entries = list(zip(queries, scores))
            frame_entries = [(query, score) for query, score in entries if query["kind"] == "frame"]
            frame_context_entries = [
                (query, score) for query, score in entries if query["kind"] == "frame_context"
            ]
            mention_entries = [(query, score) for query, score in entries if query["kind"] == "mention"]
            fact_entries = [(query, score) for query, score in entries if query["kind"] in FACT_KIND_WEIGHT]
            core_entries = [(query, score) for query, score in entries if query["kind"] == "core_fact"]
            supporting_entries = [
                (query, score) for query, score in entries if query["kind"] == "supporting_fact"
            ]
            context_entries = [
                (query, score) for query, score in entries if query["kind"] == "context_fact"
            ]
            section_entries = [(query, score) for query, score in entries if query["kind"] == "section"]

            frame_summary_score = max((score for _, score in frame_entries), default=0.0)
            frame_context_score = max(
                (score for _, score in frame_context_entries), default=frame_summary_score
            )
            frame_score = 0.70 * frame_summary_score + 0.30 * frame_context_score
            mention_score, mention_coverage, mention_diversity = _aggregate_support(
                mention_entries, support_threshold
            )
            fact_score, fact_coverage, fact_diversity = _aggregate_support(
                fact_entries, support_threshold, relevance_weighted=True
            )
            core_score, core_coverage, _ = _aggregate_support(core_entries, support_threshold)
            supporting_score, _, _ = _aggregate_support(supporting_entries, support_threshold)
            context_score, _, _ = _aggregate_support(context_entries, support_threshold)
            section_score, _, _ = _aggregate_support(section_entries, support_threshold)
            frame_path_values = [
                score for query, score in zip(queries, paths) if query["kind"] == "frame"
            ]
            context_path_values = [
                score
                for query, score in zip(queries, paths)
                if query["kind"] == "frame_context"
            ]
            section_path_values = [
                score for query, score in zip(queries, paths) if query["kind"] == "section"
            ]
            path_parts = [
                (0.55, max(frame_path_values, default=0.0)),
                (0.25, max(context_path_values, default=0.0)),
                (0.20, _mean(sorted(section_path_values, reverse=True)[:5])),
            ]
            available_path_parts = [(weight, value) for weight, value in path_parts if value]
            path_score = (
                sum(weight * value for weight, value in available_path_parts)
                / max(sum(weight for weight, _ in available_path_parts), 1e-9)
            )
            atomic_scores = [score for query, score in entries if query["kind"] != "section"]
            top_atomic_score = _mean(sorted(atomic_scores, reverse=True)[:5])
            support_coverage = 0.55 * mention_coverage + 0.45 * fact_coverage
            process_parts = [mention_score]
            if core_entries:
                process_parts.append(core_score)
            if supporting_entries:
                process_parts.append(0.6 * supporting_score)
            process_signal = _mean(process_parts)
            context_signal = _mean(
                [value for value, present in ((section_score, section_entries), (context_score, context_entries)) if present]
            )
            penalty = 0.0
            penalty_reasons: list[str] = []
            if frame_score >= 0.30 and process_signal < 0.12:
                penalty += 0.05
                penalty_reasons.append("frame_name_only")
            if context_signal >= 0.25 and process_signal < 0.12:
                penalty += 0.05
                penalty_reasons.append("context_only_support")
            total = (
                weights["frame"] * frame_score
                + weights["mention"] * mention_score
                + weights["fact"] * fact_score
                + weights["path"] * path_score
                + weights["top_atomic"] * top_atomic_score
                - penalty
            )

            mention_units = [
                {
                    "task_mention_id": query["source_id"],
                    "score": round(score, 6),
                    "section_path": query.get("section_path", ""),
                }
                for query, score in mention_entries
                if score >= float(query.get("_support_threshold", support_threshold))
            ]
            fact_units = [
                {
                    "triple_id": query["source_id"],
                    "relevance": query["kind"].removesuffix("_fact"),
                    "score": round(score, 6),
                    "weighted_score": round(score * FACT_KIND_WEIGHT[query["kind"]], 6),
                    "section_path": query.get("section_path", ""),
                }
                for query, score in fact_entries
                if score >= float(query.get("_support_threshold", support_threshold))
            ]
            candidate_records.append(
                {
                    "taskframe_id": frame_id,
                    "pcfid": node.get("pcfid"),
                    "pcf_hid": node["hid"],
                    "pcf_name_ko": node.get("nameKo", ""),
                    "pcf_name_en": node.get("nameEn", ""),
                    "pcf_level": node.get("level"),
                    "pcf_parent": node.get("parent"),
                    "pcf_path_ids": node.get("path_ids", []),
                    "pcf_path_ko": node.get("path_text_ko", ""),
                    "is_leaf": node.get("is_leaf", False),
                    "frame_score": round(frame_score, 6),
                    "frame_summary_score": round(frame_summary_score, 6),
                    "frame_context_score": round(frame_context_score, 6),
                    "mention_score": round(mention_score, 6),
                    "fact_score": round(fact_score, 6),
                    "core_fact_score": round(core_score, 6),
                    "supporting_fact_score": round(supporting_score, 6),
                    "context_fact_score": round(context_score, 6),
                    "section_score": round(section_score, 6),
                    "path_score": round(path_score, 6),
                    "top_atomic_score": round(top_atomic_score, 6),
                    "process_signal": round(process_signal, 6),
                    "context_signal": round(context_signal, 6),
                    "support_coverage": round(support_coverage, 6),
                    "mention_coverage": round(mention_coverage, 6),
                    "fact_coverage": round(fact_coverage, 6),
                    "core_fact_coverage": round(core_coverage, 6),
                    "support_diversity": round(
                        0.55 * mention_diversity + 0.45 * fact_diversity, 6
                    ),
                    "penalty": round(penalty, 6),
                    "penalty_reasons": penalty_reasons,
                    "score": round(clamp(total), 6),
                    "supporting_task_mentions": [
                        item["task_mention_id"]
                        for item in sorted(mention_units, key=lambda item: item["score"], reverse=True)
                    ],
                    "supporting_triples": [
                        item["triple_id"]
                        for item in sorted(fact_units, key=lambda item: item["weighted_score"], reverse=True)
                    ],
                    "mention_unit_scores": mention_units,
                    "fact_unit_scores": fact_units,
                    "retrieval_backend": backend,
                }
            )
        candidate_records.sort(
            key=lambda item: (item["score"], item["process_signal"], item["pcf_level"]),
            reverse=True,
        )
        selected_records = candidate_records[:output_top_k]
        selected_hids = {record["pcf_hid"] for record in selected_records}
        # 집계 단계가 LCA를 점수와 함께 검토할 수 있도록 상위 후보의 조상은
        # 진단 Top-K 밖이어도 보존한다.
        required_frame_hids = {nodes[index]["hid"] for index in required_frame_indices}
        required_ancestors = {
            hid
            for record in candidate_records[: min(5, len(candidate_records))]
            for hid in record.get("pcf_path_ids", [])
        }
        required_ancestors.update(
            hid
            for record in candidate_records
            if record["pcf_hid"] in required_frame_hids
            for hid in record.get("pcf_path_ids", [])
        )
        required_hierarchy_hids = set(required_frame_hids) | required_ancestors
        # Frame query가 특정 하위업무를 찾았으면 같은 부모의 sibling도 보존해
        # leaf 강제선택과 LCA 결정을 후속 단계에서 비교할 수 있게 한다.
        for hid in list(required_frame_hids):
            parent = parent_by_hid.get(hid)
            if parent and int(nodes[index_by_hid[parent]].get("level", 0)) >= 2:
                required_hierarchy_hids.update(children_by_hid.get(parent, []))
        selected_records.extend(
            record
            for record in candidate_records
            if record["pcf_hid"] in required_hierarchy_hids
            and record["pcf_hid"] not in selected_hids
        )
        selected_records.sort(
            key=lambda item: (item["score"], item["process_signal"], item["pcf_level"]),
            reverse=True,
        )
        for rank, record in enumerate(selected_records, 1):
            record["rank"] = rank
            records.append(record)

    report = {
        "taskframe_count": len(packets),
        "pcf_node_count": len(nodes),
        "query_count": len(all_queries),
        "candidate_record_count": len(records),
        "candidate_pool_counts": candidate_counts,
        "retrieval_backend": backend,
        "semantic_enabled": semantic is not None,
        "semantic": semantic_metadata,
        "query_top_k": query_top_k,
        "candidate_pool_size": pool_size,
        "output_top_k": output_top_k,
        "atomic_support_threshold": support_threshold,
        "atomic_relative_threshold": relative_threshold,
        "weights": weights,
    }
    write_jsonl(output_path(config, "taskframe_pcf_candidates.jsonl"), records)
    write_json(output_path(config, "retrieval_summary.json"), report)
    return records, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TaskFrame별 PCF 후보를 hybrid 검색합니다.")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, report = retrieve(load_config(args.config))
    print(
        f"Candidates: {len(records)} rows for {report['taskframe_count']} frames "
        f"({report['retrieval_backend']})"
    )


if __name__ == "__main__":
    main()
