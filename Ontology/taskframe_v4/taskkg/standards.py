from __future__ import annotations

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .utils import stable_id, text


def load_standard_tasks(payload: Any) -> list[dict[str, Any]]:
    items = payload.get("standard_tasks", []) if isinstance(payload, dict) else payload
    items = items if isinstance(items, list) else []
    result: list[dict[str, Any]] = []
    for index, item in enumerate(items, 1):
        if not isinstance(item, dict):
            continue
        name = text(item.get("name") or item.get("task_name"))
        if not name:
            continue
        scheme = text(item.get("scheme")) or "CUSTOM"
        source_id = text(item.get("id") or item.get("code")) or str(index)
        result.append(
            {
                "standard_task_id": stable_id("STD", scheme, source_id, name),
                "source_standard_id": source_id,
                "scheme": scheme,
                "name": name,
                "definition": text(item.get("definition")),
                "parent": text(item.get("parent")),
            }
        )
    return result


def map_standard_tasks(
    frames: list[dict[str, Any]],
    standards: list[dict[str, Any]],
    top_k: int = 3,
    min_score: float = 0.28,
) -> list[dict[str, Any]]:
    if not frames or not standards:
        return []
    frame_text = [
        " ".join(
            [
                frame["canonical_name"], frame.get("definition", ""),
                frame.get("scope", ""), " ".join(frame.get("aliases", [])),
            ]
        )
        for frame in frames
    ]
    standard_text = [
        " ".join(
            [
                item["name"], item.get("definition", ""), item.get("parent", ""),
                item.get("scheme", ""),
            ]
        )
        for item in standards
    ]
    matrix = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), min_df=1, sublinear_tf=True
    ).fit_transform(frame_text + standard_text)
    similarities = cosine_similarity(
        matrix[: len(frames)], matrix[len(frames) :]
    )
    mappings: list[dict[str, Any]] = []
    for frame_index, frame in enumerate(frames):
        ranked = sorted(
            enumerate(similarities[frame_index]),
            key=lambda item: float(item[1]),
            reverse=True,
        )[:top_k]
        for standard_index, score_value in ranked:
            score = float(score_value)
            if score < min_score:
                continue
            relation = (
                "exactMatch" if score >= 0.90
                else "closeMatch" if score >= 0.62
                else "relatedMatch"
            )
            standard = standards[standard_index]
            mappings.append(
                {
                    "standard_mapping_id": stable_id(
                        "SMAP", frame["taskframe_id"], standard["standard_task_id"]
                    ),
                    "taskframe_id": frame["taskframe_id"],
                    "standard_task_id": standard["standard_task_id"],
                    "relation": relation,
                    "score": round(score, 6),
                    "method": "character_tfidf_structural_candidate_search",
                    "validation_status": "candidate",
                }
            )
    return mappings

