from __future__ import annotations

from collections import defaultdict
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .utils import stable_id, text, unique


def _grams(value: str) -> set[str]:
    compact = "".join(text(value).casefold().split())
    return {
        compact[index : index + 2] for index in range(max(0, len(compact) - 1))
    } or ({compact} if compact else set())


def _jaccard(left: str, right: str) -> float:
    a, b = _grams(left), _grams(right)
    return len(a & b) / len(a | b) if a and b else 0.0


def build_candidate_blocks(
    mentions: list[dict[str, Any]],
    triples: list[dict[str, Any]],
    threshold: float = 0.28,
    top_k: int = 8,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not mentions:
        return [], []
    facts_by_mention: dict[str, list[str]] = defaultdict(list)
    entities_by_mention: dict[str, set[str]] = defaultdict(set)
    for fact in triples:
        fact_text = " ".join(
            [
                fact.get("subject_normalized", ""),
                fact.get("predicate_normalized", ""),
                fact.get("object_normalized", ""),
            ]
        )
        for link in fact.get("task_links", []):
            mention_id = link["task_mention_id"]
            facts_by_mention[mention_id].append(fact_text)
            entities_by_mention[mention_id].update(
                filter(None, [fact.get("subject_normalized"), fact.get("object_normalized")])
            )
    feature_texts = []
    for mention in mentions:
        mention_id = mention["task_mention_id"]
        feature_texts.append(
            " ".join(
                [
                    mention.get("surface_form", ""),
                    mention.get("action_normalized", ""),
                    mention.get("focal_object_normalized", ""),
                    mention.get("intended_outcome_raw", ""),
                    mention.get("section_path", ""),
                    " ".join(facts_by_mention[mention_id]),
                ]
            )
        )
    if len(mentions) == 1:
        matrix_similarity = [[1.0]]
    else:
        matrix = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), min_df=1, sublinear_tf=True
        ).fit_transform(feature_texts)
        matrix_similarity = cosine_similarity(matrix)
    pairs: list[dict[str, Any]] = []
    selected_edges: set[tuple[str, str]] = set()
    for left_index, left in enumerate(mentions):
        for right_index in range(left_index + 1, len(mentions)):
            right = mentions[right_index]
            action = _jaccard(
                left.get("action_normalized", ""), right.get("action_normalized", "")
            )
            focal = _jaccard(
                left.get("focal_object_normalized", ""),
                right.get("focal_object_normalized", ""),
            )
            outcome = _jaccard(
                left.get("intended_outcome_raw", ""), right.get("intended_outcome_raw", "")
            )
            shared_entities = entities_by_mention[left["task_mention_id"]] & entities_by_mention[
                right["task_mention_id"]
            ]
            entity_score = min(1.0, len(shared_entities) / 2)
            same_section = float(left["section_id"] == right["section_id"])
            context = float(matrix_similarity[left_index][right_index])
            score = (
                0.22 * action + 0.25 * focal + 0.16 * outcome
                + 0.15 * entity_score + 0.16 * context + 0.06 * same_section
            )
            record = {
                "pair_id": stable_id(
                    "PAIR", left["task_mention_id"], right["task_mention_id"]
                ),
                "left_task_mention_id": left["task_mention_id"],
                "right_task_mention_id": right["task_mention_id"],
                "candidate_score": round(score, 6),
                "features": {
                    "action": round(action, 6),
                    "focal_object": round(focal, 6),
                    "intended_outcome": round(outcome, 6),
                    "shared_entities": sorted(shared_entities),
                    "context": round(context, 6),
                    "same_section": same_section,
                },
            }
            pairs.append(record)
    # 각 Mention 관점의 top-k를 합집합으로 사용한다. 한쪽 인덱스 방향에서만
    # 이웃을 고르면 뒤쪽 Mention의 유효 후보가 누락될 수 있어 recall이 낮아진다.
    ranked_by_mention: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in pairs:
        ranked_by_mention[record["left_task_mention_id"]].append(record)
        ranked_by_mention[record["right_task_mention_id"]].append(record)
    for records in ranked_by_mention.values():
        for record in sorted(
            records,
            key=lambda item: (item["candidate_score"], item["pair_id"]),
            reverse=True,
        )[:top_k]:
            if record["candidate_score"] >= threshold:
                selected_edges.add(
                    (
                        record["left_task_mention_id"],
                        record["right_task_mention_id"],
                    )
                )
    adjacency: dict[str, set[str]] = {
        mention["task_mention_id"]: set() for mention in mentions
    }
    for left, right in selected_edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    blocks: list[dict[str, Any]] = []
    unseen = set(adjacency)
    mention_lookup = {item["task_mention_id"]: item for item in mentions}
    while unseen:
        seed = min(unseen)
        stack, members = [seed], set()
        while stack:
            current = stack.pop()
            if current in members:
                continue
            members.add(current)
            stack.extend(adjacency[current] - members)
        unseen -= members
        member_list = sorted(members)
        blocks.append(
            {
                "candidate_block_id": stable_id("BLOCK", *member_list),
                "member_task_mention_ids": member_list,
                "actions": unique(
                    mention_lookup[mid].get("action_normalized") for mid in member_list
                ),
                "focal_objects": unique(
                    mention_lookup[mid].get("focal_object_normalized")
                    for mid in member_list
                ),
                "blocking_only": True,
            }
        )
    return pairs, blocks
