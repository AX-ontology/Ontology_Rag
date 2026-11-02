from __future__ import annotations

from collections import defaultdict
import json
import re
from typing import Any

from .extraction import RELEVANCE
from .llm import ChatJSONClient, LLMResponseError
from .prompts import induction_prompt, reconciliation_prompt
from .utils import as_list, confidence, stable_id, text, unique


RELATIONS = {
    "sameTask", "subTaskOf", "relatedTask", "precedes", "follows",
    "supports", "distinctTask",
}
ROLE_RANK = {"context": 1, "supporting": 2, "core": 3}
AMBIGUOUS_PREDICATES = {"관련", "사용", "활용", "지원", "포함", "기반", "필요"}


def mention_cards(
    mentions: list[dict[str, Any]], triples: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    facts_by_mention: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fact in triples:
        compact = {
            "triple_id": fact["triple_id"],
            "subject": fact.get("subject_raw", ""),
            "predicate": fact.get("predicate_raw", ""),
            "object": fact.get("object_raw", ""),
            "knowledge_function": fact.get("knowledge_function", "fact"),
            "qualifiers": fact.get("qualifiers", []),
            "evidence_quote": fact.get(
                "evidence_quote", fact.get("source_text", "")
            ),
            "relevance": "",
            "source_text": fact.get("source_text", ""),
        }
        for link in fact.get("task_links", []):
            facts_by_mention[link["task_mention_id"]].append(
                {**compact, "relevance": link["relevance"]}
            )
    return [
        {
            "task_mention_id": mention["task_mention_id"],
            "surface_form": mention["surface_form"],
            "action": mention.get("action_normalized", ""),
            "focal_object": mention.get("focal_object_normalized", ""),
            "intended_outcome": mention.get("intended_outcome_raw", ""),
            "section_path": mention.get("section_path", ""),
            "source_text": mention.get("source_text", ""),
            "evidence_ids": mention.get("evidence_ids", []),
            "linked_facts": facts_by_mention[mention["task_mention_id"]],
        }
        for mention in mentions
    ]


def build_ambiguous_fact_contexts(
    triples: list[dict[str, Any]],
    mentions: list[dict[str, Any]],
    chunks: list[dict[str, Any]] | None = None,
    radius: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    mentions_by_evidence: dict[str, list[str]] = defaultdict(list)
    mentions_by_chunk: dict[str, list[str]] = defaultdict(list)
    for mention in mentions:
        mentions_by_chunk[mention.get("chunk_id", "")].append(
            mention["task_mention_id"]
        )
        for evidence_id in mention.get("evidence_ids", []):
            mentions_by_evidence[evidence_id].append(mention["task_mention_id"])
    chunks = chunks or []
    chunk_index = {item["chunk_id"]: index for index, item in enumerate(chunks)}
    chunk_lookup = {item["chunk_id"]: item for item in chunks}

    def context_chunk_ids(chunk_id: str) -> list[str]:
        if chunk_id not in chunk_index:
            return [chunk_id] if chunk_id else []
        index = chunk_index[chunk_id]
        safe_radius = max(0, radius)
        start = max(0, index - safe_radius)
        end = min(len(chunks), index + safe_radius + 1)
        return [chunks[position]["chunk_id"] for position in range(start, end)]

    def nearest_task_chunk_ids(chunk_id: str) -> list[str]:
        if chunk_id not in chunk_index:
            return []
        source_index = chunk_index[chunk_id]
        source_section = chunk_lookup[chunk_id].get("section_id", "")
        candidates = [
            item["chunk_id"] for item in chunks
            if mentions_by_chunk.get(item["chunk_id"])
        ]
        if not candidates:
            return []
        ranked = sorted(
            candidates,
            key=lambda candidate_id: (
                0
                if chunk_lookup[candidate_id].get("section_id", "")
                == source_section
                else 1,
                abs(chunk_index[candidate_id] - source_index),
                chunk_index[candidate_id],
            ),
        )
        best = ranked[0]
        best_key = (
            0
            if chunk_lookup[best].get("section_id", "") == source_section
            else 1,
            abs(chunk_index[best] - source_index),
        )
        return [
            candidate_id for candidate_id in ranked
            if (
                0
                if chunk_lookup[candidate_id].get("section_id", "")
                == source_section
                else 1,
                abs(chunk_index[candidate_id] - source_index),
            )
            == best_key
        ]

    result: list[dict[str, Any]] = []
    referenced_chunk_ids: set[str] = set()
    for fact in triples:
        links = fact.get("task_links", [])
        predicate = text(fact.get("predicate_normalized")).removesuffix("하다").removesuffix("되다")
        context_ids = context_chunk_ids(fact.get("chunk_id", ""))
        contextual_mentions = unique(
            mention_id
            for context_id in context_ids
            for mention_id in mentions_by_chunk.get(context_id, [])
        )
        if not contextual_mentions:
            nearest_ids = nearest_task_chunk_ids(fact.get("chunk_id", ""))
            context_ids = unique(context_ids + nearest_ids)
            contextual_mentions = unique(
                mention_id for context_id in nearest_ids
                for mention_id in mentions_by_chunk.get(context_id, [])
            )
        is_ambiguous = (
            not links
            or len(links) > 1
            or (
                bool(mentions_by_evidence.get(fact["evidence_id"]))
                and len(links) != 1
            )
            or any(term in predicate for term in AMBIGUOUS_PREDICATES)
        )
        if is_ambiguous:
            referenced_chunk_ids.update(context_ids)
            result.append(
                {
                    "triple_id": fact["triple_id"],
                    "subject": fact.get("subject_raw", ""),
                    "predicate": fact.get("predicate_raw", ""),
                    "object": fact.get("object_raw", ""),
                    "knowledge_function": fact.get("knowledge_function", "fact"),
                    "qualifiers": fact.get("qualifiers", []),
                    "evidence_quote": fact.get(
                        "evidence_quote", fact.get("source_text", "")
                    ),
                    "source_text": fact.get("source_text", ""),
                    "section_path": fact.get("section_path", ""),
                    "current_task_links": links,
                    "coverage_required": not bool(links),
                    "same_evidence_task_mentions": mentions_by_evidence.get(
                        fact["evidence_id"], []
                    ),
                    "context_chunk_ids": context_ids,
                    "context_task_mention_ids": contextual_mentions,
                    "context_priority": [
                        "current_evidence", "current_chunk",
                        "previous_or_next_chunk", "section_path",
                    ],
                }
            )
    context_catalog: list[dict[str, Any]] = []
    for chunk_id in sorted(
        referenced_chunk_ids, key=lambda value: chunk_index.get(value, 10**9)
    ):
        chunk = chunk_lookup.get(chunk_id)
        if not chunk:
            continue
        index = chunk_index[chunk_id]
        context_catalog.append(
            {
                "chunk_id": chunk_id,
                "relative_neighbors": {
                    "previous_chunk_id": chunks[index - 1]["chunk_id"] if index > 0 else "",
                    "next_chunk_id": (
                        chunks[index + 1]["chunk_id"]
                        if index + 1 < len(chunks) else ""
                    ),
                },
                "section_path": chunk.get("section_path", ""),
                "task_mention_ids": unique(mentions_by_chunk.get(chunk_id, [])),
                "sentences": [
                    {
                        "evidence_id": sentence["evidence_id"],
                        "source_text": sentence["source_text"],
                    }
                    for sentence in chunk.get("sentences", [])
                ],
            }
        )
    return result, context_catalog


def generate_llm(
    client: ChatJSONClient,
    mentions: list[dict[str, Any]],
    triples: list[dict[str, Any]],
    blocks: list[dict[str, Any]],
    ambiguous: list[dict[str, Any]] | None = None,
    chunk_contexts: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    cards = mention_cards(mentions, triples)
    ambiguous = ambiguous if ambiguous is not None else build_ambiguous_fact_contexts(
        triples, mentions
    )[0]
    system, user = induction_prompt(cards, blocks, ambiguous, chunk_contexts or [])
    raw = client.call(
        "llm_2_global_taskframe_induction",
        stable_id("GLOBAL", *[item["task_mention_id"] for item in mentions]),
        system,
        user,
    )
    return finalize_induction(raw, mentions, triples, ambiguous)


def pack_candidate_blocks(
    blocks: list[dict[str, Any]],
    mentions: list[dict[str, Any]],
    max_mentions: int,
) -> list[dict[str, Any]]:
    """Pack whole blocking components into bounded LLM 2 local batches."""
    if max_mentions < 1:
        raise ValueError("llm2 max mentions는 1 이상이어야 합니다.")
    all_mentions = {item["task_mention_id"] for item in mentions}
    packed: list[dict[str, Any]] = []
    current_ids: list[str] = []
    current_blocks: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal current_ids, current_blocks
        if not current_ids:
            return
        packed.append(
            {
                "batch_id": stable_id("L2B", *current_ids),
                "member_task_mention_ids": unique(current_ids),
                "candidate_blocks": current_blocks,
            }
        )
        current_ids, current_blocks = [], []

    covered: set[str] = set()
    for block in blocks:
        members = [
            mention_id for mention_id in block.get("member_task_mention_ids", [])
            if mention_id in all_mentions and mention_id not in covered
        ]
        if not members:
            continue
        # 현재 blocking threshold에서 큰 component가 생겨도 입력 상한을 지킨다.
        parts = [
            members[start : start + max_mentions]
            for start in range(0, len(members), max_mentions)
        ]
        for part_index, part in enumerate(parts, 1):
            if current_ids and len(current_ids) + len(part) > max_mentions:
                flush()
            local_block = {
                **block,
                "candidate_block_id": (
                    block["candidate_block_id"]
                    if len(parts) == 1
                    else f"{block['candidate_block_id']}__part{part_index}"
                ),
                "member_task_mention_ids": part,
                "split_for_context_limit": len(parts) > 1,
            }
            current_ids.extend(part)
            current_blocks.append(local_block)
            covered.update(part)
    for mention_id in sorted(all_mentions - covered):
        if current_ids and len(current_ids) + 1 > max_mentions:
            flush()
        current_ids.append(mention_id)
        current_blocks.append(
            {
                "candidate_block_id": stable_id("BLOCK", mention_id),
                "member_task_mention_ids": [mention_id],
                "blocking_only": True,
                "recovered_singleton": True,
            }
        )
    flush()
    return packed


def estimate_hierarchical_llm2_calls(
    blocks: list[dict[str, Any]],
    mentions: list[dict[str, Any]],
    max_mentions: int,
) -> dict[str, int]:
    batches = pack_candidate_blocks(blocks, mentions, max_mentions)
    local_calls = len(batches)
    reconciliation_calls = 1 if local_calls > 1 else 0
    return {
        "local_induction_calls": local_calls,
        "global_reconciliation_calls": reconciliation_calls,
        "ambiguous_fact_extra_calls": 0,
        "total_llm2_calls": local_calls + reconciliation_calls,
    }


def generate_hierarchical_llm(
    client: ChatJSONClient,
    mentions: list[dict[str, Any]],
    triples: list[dict[str, Any]],
    blocks: list[dict[str, Any]],
    ambiguous: list[dict[str, Any]],
    chunk_contexts: list[dict[str, Any]],
    max_mentions: int,
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]
]:
    """Run bounded local induction calls and one compact global reconciliation."""
    batches = pack_candidate_blocks(blocks, mentions, max_mentions)
    if not batches:
        return [], [], [], {
            "input_mention_count": 0,
            "taskframe_count": 0,
            "mention_partition_valid": True,
            "hierarchical": True,
            "local_induction_calls": 0,
            "global_reconciliation_calls": 0,
            "total_llm2_calls": 0,
        }
    mention_lookup = {item["task_mention_id"]: item for item in mentions}
    batch_sets = [
        set(batch["member_task_mention_ids"]) for batch in batches
    ]
    ambiguous_by_batch: list[list[dict[str, Any]]] = [[] for _ in batches]
    skipped_ambiguous = 0
    for fact in ambiguous:
        candidates = {
            link["task_mention_id"] for link in fact.get("current_task_links", [])
        } | set(fact.get("context_task_mention_ids", []))
        ranked = sorted(
            (
                (len(candidates & member_ids), -index, index)
                for index, member_ids in enumerate(batch_sets)
            ),
            reverse=True,
        )
        if not ranked or ranked[0][0] == 0:
            skipped_ambiguous += 1
            continue
        batch_index = ranked[0][2]
        allowed = batch_sets[batch_index]
        ambiguous_by_batch[batch_index].append(
            {
                **fact,
                "current_task_links": [
                    link for link in fact.get("current_task_links", [])
                    if link["task_mention_id"] in allowed
                ],
                "same_evidence_task_mentions": [
                    mention_id for mention_id in fact.get(
                        "same_evidence_task_mentions", []
                    ) if mention_id in allowed
                ],
                "context_task_mention_ids": [
                    mention_id for mention_id in fact.get(
                        "context_task_mention_ids", []
                    ) if mention_id in allowed
                ],
            }
        )

    context_lookup = {
        item["chunk_id"]: item for item in chunk_contexts
    }
    local_frames: list[dict[str, Any]] = []
    local_relations: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    local_reports: list[dict[str, Any]] = []
    for index, batch in enumerate(batches):
        member_ids = batch_sets[index]
        local_mentions = [
            mention_lookup[mention_id]
            for mention_id in batch["member_task_mention_ids"]
        ]
        local_ambiguous = ambiguous_by_batch[index]
        referenced_context_ids = unique(
            chunk_id for fact in local_ambiguous
            for chunk_id in fact.get("context_chunk_ids", [])
        )
        local_contexts = []
        for chunk_id in referenced_context_ids:
            context = context_lookup.get(chunk_id)
            if not context:
                continue
            local_contexts.append(
                {
                    **context,
                    "task_mention_ids": [
                        mention_id for mention_id in context.get(
                            "task_mention_ids", []
                        ) if mention_id in member_ids
                    ],
                }
            )
        system, user = induction_prompt(
            mention_cards(local_mentions, triples),
            batch["candidate_blocks"],
            local_ambiguous,
            local_contexts,
        )
        raw = client.call(
            "llm_2_local_taskframe_induction",
            f"{index + 1}-{batch['batch_id']}",
            system,
            user,
        )
        frames, relations, local_decisions, report = finalize_induction(
            raw, local_mentions, triples, local_ambiguous
        )
        report.update(
            {
                "batch_id": batch["batch_id"],
                "batch_index": index + 1,
                "batch_mention_count": len(local_mentions),
                "batch_ambiguous_fact_count": len(local_ambiguous),
                "batch_context_chunk_count": len(local_contexts),
            }
        )
        local_frames.extend(frames)
        local_relations.extend(relations)
        decisions.extend(local_decisions)
        local_reports.append(report)
        print(
            f"[step5/llm2-local] batch={index + 1}/{len(batches)} "
            f"mentions={len(local_mentions)} frames={len(frames)}"
        )

    if len(batches) == 1:
        final_frames, final_relations = local_frames, local_relations
        reconciliation_report = {
            "performed": False,
            "reason": "single local batch is already global",
        }
        reconciliation_calls = 0
    else:
        final_frames, final_relations, reconciliation_report = reconcile_bounded(
            client, local_frames, local_relations, mentions, triples
        )
        reconciliation_calls = reconciliation_report["call_count"]
        print(
            f"[step5/llm2-global] provisional={len(local_frames)} "
            f"calls={reconciliation_calls} final={len(final_frames)}"
        )
    report = {
        "input_mention_count": len(mentions),
        "taskframe_count": len(final_frames),
        "mention_partition_valid": _partition_valid(final_frames, mentions),
        "hierarchical": True,
        "max_mentions_per_local_call": max_mentions,
        "local_induction_calls": len(batches),
        "global_reconciliation_calls": reconciliation_calls,
        "ambiguous_fact_extra_calls": 0,
        "total_llm2_calls": len(batches) + reconciliation_calls,
        "ambiguous_fact_count": len(ambiguous),
        "ambiguous_fact_assigned_to_local_batch_count": sum(
            len(items) for items in ambiguous_by_batch
        ),
        "ambiguous_fact_without_task_candidate_count": skipped_ambiguous,
        "rejudged_fact_count": len(decisions),
        "local_batch_reports": local_reports,
        "reconciliation_report": reconciliation_report,
    }
    return final_frames, final_relations, _deduplicate_decisions(decisions), report


def _partition_valid(
    frames: list[dict[str, Any]], mentions: list[dict[str, Any]]
) -> bool:
    members = [
        mention_id for frame in frames
        for mention_id in frame["member_task_mentions"]
    ]
    expected = {item["task_mention_id"] for item in mentions}
    return len(members) == len(expected) and set(members) == expected


def _provisional_frame_cards(
    frames: list[dict[str, Any]],
    mentions: list[dict[str, Any]],
    triples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    mention_lookup = {item["task_mention_id"]: item for item in mentions}
    facts_by_mention: dict[str, list[str]] = defaultdict(list)
    for fact in triples:
        display = " → ".join(
            [
                fact.get("subject_raw", ""), fact.get("predicate_raw", ""),
                fact.get("object_raw", ""),
            ]
        )
        for link in fact.get("task_links", []):
            facts_by_mention[link["task_mention_id"]].append(display)
    result: list[dict[str, Any]] = []
    for frame in frames:
        members = [
            mention_lookup[mention_id]
            for mention_id in frame["member_task_mentions"]
            if mention_id in mention_lookup
        ]
        result.append(
            {
                "provisional_taskframe_id": frame["taskframe_id"],
                "canonical_name": frame["canonical_name"],
                "definition": frame["definition"],
                "scope": frame["scope"],
                "aliases": frame["aliases"][:8],
                "actions": unique(
                    item.get("action_normalized") for item in members
                ),
                "focal_objects": unique(
                    item.get("focal_object_normalized") for item in members
                ),
                "intended_outcomes": unique(
                    item.get("intended_outcome_raw") for item in members
                ),
                "member_count": len(members),
                "linked_fact_samples": unique(
                    fact for item in members
                    for fact in facts_by_mention[item["task_mention_id"]]
                )[:6],
            }
        )
    return result


def pack_reconciliation_cards(
    cards: list[dict[str, Any]], max_frames: int = 32,
    max_chars: int = 48000, reverse: bool = False,
) -> list[list[dict[str, Any]]]:
    """Partition all frames into bounded neighborhoods; similarity never merges."""
    if not cards:
        return []
    if max_frames < 2 or max_chars < 1:
        raise ValueError("Reconciliation bounds must be positive (frames >= 2)")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    descriptions = [
        " ".join(text(card.get(key)) for key in (
            "canonical_name", "definition", "scope", "actions", "focal_objects",
        )) or card["provisional_taskframe_id"] for card in cards
    ]
    matrix = TfidfVectorizer(analyzer="char", ngram_range=(1, 3)).fit_transform(descriptions)
    similarities = cosine_similarity(matrix)
    sizes = [len(json.dumps(card, ensure_ascii=False, indent=2)) + 4 for card in cards]
    order = sorted(range(len(cards)), key=lambda i: (
        text(cards[i].get("canonical_name")), cards[i]["provisional_taskframe_id"],
    ), reverse=reverse)
    remaining = set(order)
    groups = []
    for seed in order:
        if seed not in remaining:
            continue
        selected = [seed]
        remaining.remove(seed)
        size = sizes[seed]
        neighbors = sorted(remaining, key=lambda i: (
            -float(similarities[seed, i]), cards[i]["provisional_taskframe_id"],
        ))
        for index in neighbors:
            if len(selected) >= max_frames:
                break
            if size + sizes[index] <= max_chars:
                selected.append(index)
                remaining.remove(index)
                size += sizes[index]
        groups.append([cards[index] for index in selected])
    return groups


def reconcile_bounded(
    client: ChatJSONClient, frames: list[dict[str, Any]],
    relations: list[dict[str, Any]], mentions: list[dict[str, Any]],
    triples: list[dict[str, Any]], max_frames: int = 32,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Two neighborhood passes, preserving all mentions and cross-batch edges.

    The reverse-seed second pass compares a different set of boundary neighbors.
    This is bounded candidate reconciliation, not an exhaustive all-pairs review.
    """
    original_count = len(frames)
    call_count = 0
    reports = []
    pass_count = 2 if len(frames) > max_frames else 1
    for pass_no in range(1, pass_count + 1):
        cards = _provisional_frame_cards(frames, mentions, triples)
        groups = pack_reconciliation_cards(cards, max_frames, reverse=pass_no == 2)
        merged_raw: dict[str, Any] = {
            "taskframes": [], "task_relations": [], "_unchanged_are_preserved": True,
        }

        def run_group(group: list[dict[str, Any]], key: str) -> None:
            nonlocal call_count
            if len(group) < 2:
                return
            system, user = reconciliation_prompt(group)
            call_count += 1
            print(f"[step5/llm2-global-batch] pass={pass_no} batch={key} frames={len(group)}", flush=True)
            try:
                raw = client.call(
                    "llm_2_global_taskframe_reconciliation",
                    stable_id("RECON_BOUNDED_V1", pass_no, key, *[
                        card["provisional_taskframe_id"] for card in group
                    ]), system, user,
                )
                if not isinstance(raw, dict) or not all(
                    isinstance(raw.get(field), list) for field in ("taskframes", "task_relations")
                ):
                    raise LLMResponseError("Invalid reconciliation object")
                allowed = {card["provisional_taskframe_id"] for card in group}
                seen: set[str] = set()
                keys: set[str] = set()
                for frame in raw["taskframes"]:
                    if not isinstance(frame, dict):
                        raise LLMResponseError("Invalid reconciliation frame")
                    ids = as_list(frame.get("provisional_taskframe_ids"))
                    frame_key = text(frame.get("frame_key"))
                    if not ids or set(ids) - allowed or seen.intersection(ids) or not frame_key or frame_key in keys:
                        raise LLMResponseError("Invalid, duplicate or out-of-batch reconciliation IDs")
                    seen.update(ids)
                    keys.add(frame_key)
            except LLMResponseError:
                if len(group) <= 2:
                    raise
                middle = len(group) // 2
                print(f"[step5/llm2-global-split] pass={pass_no} batch={key}", flush=True)
                run_group(group[:middle], key + "a")
                run_group(group[middle:], key + "b")
                return
            prefix = f"P{pass_no}B{key}_"
            merged_raw["taskframes"].extend(
                {**frame, "frame_key": prefix + text(frame["frame_key"])}
                for frame in raw["taskframes"]
            )
            merged_raw["task_relations"].extend(
                {**relation,
                 "source_frame_key": prefix + text(relation.get("source_frame_key")),
                 "target_frame_key": prefix + text(relation.get("target_frame_key"))}
                for relation in raw["task_relations"] if isinstance(relation, dict)
            )

        for index, group in enumerate(groups, 1):
            run_group(group, str(index))
        frames, relations, report = finalize_reconciliation(
            merged_raw, frames, relations, mentions
        )
        if not report["mention_partition_valid"]:
            raise RuntimeError("Bounded reconciliation lost/duplicated TaskMentions")
        reports.append({**report, "pass": pass_no, "batch_count": len(groups)})
        print(f"[step5/llm2-global-pass] pass={pass_no}/{pass_count} frames={len(frames)}", flush=True)
    return frames, relations, {
        "performed": True, "method": "bounded_neighborhood_two_pass",
        "exhaustive_pairwise_review": False, "max_frames_per_call": max_frames,
        "input_provisional_frame_count": original_count,
        "output_final_frame_count": len(frames), "call_count": call_count,
        "passes": reports, "mention_partition_valid": _partition_valid(frames, mentions),
    }


def finalize_reconciliation(
    raw: Any,
    provisional_frames: list[dict[str, Any]],
    provisional_relations: list[dict[str, Any]],
    mentions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    raw = raw if isinstance(raw, dict) else {}
    provisional_lookup = {
        item["taskframe_id"]: item for item in provisional_frames
    }
    mention_lookup = {item["task_mention_id"]: item for item in mentions}
    seen: set[str] = set()
    duplicates: list[str] = []
    unknown: list[str] = []
    final_frames: list[dict[str, Any]] = []
    frame_key_map: dict[str, str] = {}
    provisional_to_final: dict[str, str] = {}
    raw_frames = raw.get("taskframes")
    raw_frames = raw_frames if isinstance(raw_frames, list) else []

    def add_frame(item: dict[str, Any], provisional_ids: list[str], recovered: bool) -> None:
        members = unique(
            mention_id for provisional_id in provisional_ids
            for mention_id in provisional_lookup[provisional_id][
                "member_task_mentions"
            ]
        )
        frame_id = stable_id("TF", *sorted(members))
        source_frames = [provisional_lookup[value] for value in provisional_ids]
        aliases = unique(
            as_list(item.get("aliases"))
            + [
                alias for source in source_frames
                for alias in source.get("aliases", [])
            ]
        )
        evidence_ids = unique(
            evidence_id for mention_id in members
            for evidence_id in mention_lookup[mention_id].get("evidence_ids", [])
        )
        frame_key = text(item.get("frame_key")) or f"GF{len(final_frames) + 1}"
        frame_key_map[frame_key] = frame_id
        for provisional_id in provisional_ids:
            provisional_to_final[provisional_id] = frame_id
        final_frames.append(
            {
                "taskframe_id": frame_id,
                "source_frame_key": frame_key,
                "canonical_name": text(item.get("canonical_name"))
                or source_frames[0]["canonical_name"],
                "definition": text(item.get("definition"))
                or source_frames[0]["definition"],
                "scope": text(item.get("scope")) or source_frames[0]["scope"],
                "aliases": aliases,
                "member_task_mentions": members,
                "evidence_ids": evidence_ids,
                "status": (
                    "canonical"
                    if text(item.get("status")).casefold() == "canonical"
                    else "provisional"
                ),
                "schema_version": "4.5.1",
                "provisional_taskframe_ids": provisional_ids,
                "reconciliation_recovered": recovered,
                "merge_reason": text(item.get("merge_reason")),
            }
        )

    for item in raw_frames:
        if not isinstance(item, dict):
            continue
        valid_ids: list[str] = []
        for provisional_id in as_list(item.get("provisional_taskframe_ids")):
            if provisional_id not in provisional_lookup:
                unknown.append(provisional_id)
            elif provisional_id in seen:
                duplicates.append(provisional_id)
            else:
                seen.add(provisional_id)
                valid_ids.append(provisional_id)
        if valid_ids:
            add_frame(item, valid_ids, False)
    missing = sorted(set(provisional_lookup) - seen)
    preserve_unchanged = raw.get("_unchanged_are_preserved") is True
    for provisional_id in missing:
        source = provisional_lookup[provisional_id]
        if preserve_unchanged:
            final_frames.append(dict(source))
            provisional_to_final[provisional_id] = provisional_id
            continue
        add_frame(
            {
                "frame_key": f"RECOVERED_GF{len(final_frames) + 1}",
                "canonical_name": source["canonical_name"],
                "definition": source["definition"],
                "scope": source["scope"],
                "aliases": source["aliases"],
                "status": "provisional",
                "merge_reason": "global reconciliation 누락 보수적 복구",
            },
            [provisional_id],
            True,
        )

    relations: list[dict[str, Any]] = []
    relation_seen: set[tuple[str, str, str]] = set()

    def add_relation(
        source: str, relation: str, target: str, conf: Any, reason: Any
    ) -> None:
        key = (source, relation, target)
        if (
            source and target and source != target and relation in RELATIONS
            and key not in relation_seen
        ):
            relation_seen.add(key)
            relations.append(
                {
                    "task_relation_id": stable_id("TREL", *key),
                    "source_taskframe_id": source,
                    "relation": relation,
                    "target_taskframe_id": target,
                    "confidence": confidence(conf),
                    "reason": text(reason),
                }
            )

    for item in provisional_relations:
        add_relation(
            provisional_to_final.get(item["source_taskframe_id"], ""),
            item["relation"],
            provisional_to_final.get(item["target_taskframe_id"], ""),
            item.get("confidence"),
            item.get("reason"),
        )
    raw_relations = raw.get("task_relations")
    for item in raw_relations if isinstance(raw_relations, list) else []:
        if isinstance(item, dict):
            add_relation(
                frame_key_map.get(text(item.get("source_frame_key")), ""),
                text(item.get("relation")),
                frame_key_map.get(text(item.get("target_frame_key")), ""),
                item.get("confidence"),
                item.get("reason"),
            )
    return final_frames, relations, {
        "performed": True,
        "input_provisional_frame_count": len(provisional_frames),
        "output_final_frame_count": len(final_frames),
        "missing_provisional_frames_recovered": [] if preserve_unchanged else missing,
        "unchanged_provisional_frames_preserved": missing if preserve_unchanged else [],
        "duplicate_provisional_frames_ignored": sorted(set(duplicates)),
        "unknown_provisional_frame_ids": sorted(set(unknown)),
        "mention_partition_valid": _partition_valid(final_frames, mentions),
    }


def _deduplicate_decisions(
    decisions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in decisions:
        result[item["triple_id"]] = item
    return list(result.values())


def generate_heuristic(
    mentions: list[dict[str, Any]],
    triples: list[dict[str, Any]],
    blocks: list[dict[str, Any]],
    ambiguous: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    mention_lookup = {item["task_mention_id"]: item for item in mentions}
    raw_frames: list[dict[str, Any]] = []
    for index, block in enumerate(blocks, 1):
        members = [
            mention_lookup[mid] for mid in block["member_task_mention_ids"]
            if mid in mention_lookup
        ]
        if not members:
            continue
        representative = max(members, key=lambda item: item.get("confidence", 0))
        action = representative.get("action_normalized", "").removesuffix("하다")
        focal = representative.get("focal_object_normalized", "")
        name = text(f"{focal} {action}") or representative["surface_form"][:80]
        raw_frames.append(
            {
                "frame_key": f"F{index}",
                "canonical_name": name,
                "definition": f"{name}에 관한 문서 기반 업무",
                "scope": representative.get("section_path", ""),
                "aliases": [item["surface_form"] for item in members],
                "member_task_mentions": [item["task_mention_id"] for item in members],
                "status": "provisional",
            }
        )
    raw = {
        "taskframes": raw_frames,
        "task_relations": [],
        "triple_link_decisions": [],
    }
    frames, relations, decisions, report = finalize_induction(
        raw,
        mentions,
        triples,
        ambiguous if ambiguous is not None else build_ambiguous_fact_contexts(
            triples, mentions
        )[0],
    )
    report["backend"] = "heuristic_smoke"
    return frames, relations, decisions, report


def finalize_induction(
    raw: Any,
    mentions: list[dict[str, Any]],
    triples: list[dict[str, Any]],
    ambiguous: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not isinstance(raw, dict):
        raise ValueError("LLM 2 requires a complete JSON object; bare arrays may be truncated responses")
    raw = raw if isinstance(raw, dict) else {}
    mention_lookup = {item["task_mention_id"]: item for item in mentions}
    all_mentions = set(mention_lookup)
    errors: list[str] = []
    duplicate_mentions: list[str] = []
    seen_mentions: set[str] = set()
    frames: list[dict[str, Any]] = []
    frame_key_map: dict[str, str] = {}
    raw_frames = raw.get("taskframes")
    raw_frames = raw_frames if isinstance(raw_frames, list) else []
    for index, item in enumerate(raw_frames, 1):
        if not isinstance(item, dict):
            continue
        members: list[str] = []
        for mention_id in as_list(item.get("member_task_mentions")):
            if mention_id not in all_mentions:
                errors.append(f"unknown mention: {mention_id}")
            elif mention_id in seen_mentions:
                duplicate_mentions.append(mention_id)
            else:
                seen_mentions.add(mention_id)
                members.append(mention_id)
        if not members:
            continue
        frame_key = text(item.get("frame_key")) or f"F{index}"
        frame_id = stable_id("TF", *sorted(members))
        frame_key_map[frame_key] = frame_id
        aliases = unique(
            as_list(item.get("aliases"))
            + [mention_lookup[mid]["surface_form"] for mid in members]
        )
        evidence_ids = unique(
            evidence_id for mid in members
            for evidence_id in mention_lookup[mid].get("evidence_ids", [])
        )
        frames.append(
            {
                "taskframe_id": frame_id,
                "source_frame_key": frame_key,
                "canonical_name": text(item.get("canonical_name")) or aliases[0][:80],
                "definition": text(item.get("definition")),
                "scope": text(item.get("scope")),
                "aliases": aliases,
                "member_task_mentions": members,
                "evidence_ids": evidence_ids,
                "status": (
                    "canonical" if text(item.get("status")).casefold() == "canonical"
                    else "provisional"
                ),
                "schema_version": "4.5.1",
            }
        )
    missing_mentions = sorted(all_mentions - seen_mentions)
    for mention_id in missing_mentions:
        mention = mention_lookup[mention_id]
        frame_id = stable_id("TF", mention_id)
        frame_key = f"RECOVERED_{len(frames) + 1}"
        frame_key_map[frame_key] = frame_id
        action = mention.get("action_normalized", "").removesuffix("하다")
        name = text(f"{mention.get('focal_object_normalized', '')} {action}")
        frames.append(
            {
                "taskframe_id": frame_id,
                "source_frame_key": frame_key,
                "canonical_name": name or mention["surface_form"][:80],
                "definition": "LLM 2 누락으로 보수적으로 보존한 단일 TaskMention 업무",
                "scope": mention.get("section_path", ""),
                "aliases": [mention["surface_form"]],
                "member_task_mentions": [mention_id],
                "evidence_ids": mention["evidence_ids"],
                "status": "provisional",
                "schema_version": "4.5.1",
            }
        )
    frame_ids = {item["taskframe_id"] for item in frames}
    relations: list[dict[str, Any]] = []
    relation_seen: set[tuple[str, str, str]] = set()
    raw_relations = raw.get("task_relations")
    for item in raw_relations if isinstance(raw_relations, list) else []:
        if not isinstance(item, dict):
            continue
        source = frame_key_map.get(text(item.get("source_frame_key")), "")
        target = frame_key_map.get(text(item.get("target_frame_key")), "")
        relation = text(item.get("relation"))
        key = (source, relation, target)
        if source in frame_ids and target in frame_ids and source != target and relation in RELATIONS:
            if key not in relation_seen:
                relation_seen.add(key)
                relations.append(
                    {
                        "task_relation_id": stable_id("TREL", *key),
                        "source_taskframe_id": source,
                        "relation": relation,
                        "target_taskframe_id": target,
                        "confidence": confidence(item.get("confidence")),
                        "reason": text(item.get("reason")),
                    }
                )
    allowed_ambiguous = {item["triple_id"] for item in ambiguous}
    triple_ids = {item["triple_id"] for item in triples}
    decisions: list[dict[str, Any]] = []
    raw_decisions = raw.get("triple_link_decisions")
    for item in raw_decisions if isinstance(raw_decisions, list) else []:
        if not isinstance(item, dict):
            continue
        triple_id = text(item.get("triple_id"))
        if triple_id not in allowed_ambiguous or triple_id not in triple_ids:
            continue
        links: list[dict[str, str]] = []
        raw_links = item.get("task_links")
        for link in raw_links if isinstance(raw_links, list) else []:
            if not isinstance(link, dict):
                continue
            mention_id = text(link.get("task_mention_id"))
            relevance = text(link.get("relevance")).casefold()
            if mention_id in all_mentions:
                links.append(
                    {
                        "task_mention_id": mention_id,
                        "relevance": relevance if relevance in RELEVANCE else "context",
                        "source": "llm_2_rejudgment",
                    }
                )
        decisions.append(
            {
                "triple_id": triple_id,
                "task_links": links,
                "reason": text(item.get("reason")),
            }
        )
    report = {
        "input_mention_count": len(mentions),
        "taskframe_count": len(frames),
        "missing_mentions_recovered": missing_mentions,
        "duplicate_mentions_ignored": sorted(set(duplicate_mentions)),
        "unknown_or_schema_errors": errors,
        "ambiguous_fact_count": len(ambiguous),
        "rejudged_fact_count": len(decisions),
        "mention_partition_valid": (
            sum(len(frame["member_task_mentions"]) for frame in frames) == len(mentions)
            and len(
                {
                    mention_id for frame in frames
                    for mention_id in frame["member_task_mentions"]
                }
            ) == len(mentions)
        ),
    }
    return frames, relations, decisions, report


def propagate_fact_links(
    triples: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    mentions: list[dict[str, Any]] | None = None,
    exhaustive: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    mention_to_frame = {
        mention_id: frame["taskframe_id"]
        for frame in frames for mention_id in frame["member_task_mentions"]
    }
    decision_lookup = {item["triple_id"]: item for item in decisions}
    mention_lookup = {
        item["task_mention_id"]: item for item in (mentions or [])
    }

    def grams(value: str) -> set[str]:
        compact = "".join(text(value).casefold().split())
        return {
            compact[index : index + 2]
            for index in range(max(0, len(compact) - 1))
        } or ({compact} if compact else set())

    mention_grams = {
        mention_id: grams(
            " ".join(
                [
                    item.get("surface_form", ""),
                    item.get("action_normalized", ""),
                    item.get("focal_object_normalized", ""),
                    item.get("intended_outcome_raw", ""),
                    item.get("source_text", ""),
                ]
            )
        )
        for mention_id, item in mention_lookup.items()
        if mention_id in mention_to_frame
    }

    def fallback_mention(fact: dict[str, Any]) -> str:
        fact_grams = grams(
            " ".join(
                [
                    fact.get("subject_normalized", fact.get("subject_raw", "")),
                    fact.get("predicate_normalized", fact.get("predicate_raw", "")),
                    fact.get("object_normalized", fact.get("object_raw", "")),
                    fact.get("source_text", ""),
                ]
            )
        )
        ranked: list[tuple[float, str]] = []
        for mention_id, item in mention_lookup.items():
            if mention_id not in mention_to_frame:
                continue
            candidate_grams = mention_grams.get(mention_id, set())
            similarity = (
                len(fact_grams & candidate_grams) / len(fact_grams | candidate_grams)
                if fact_grams and candidate_grams else 0.0
            )
            score = similarity
            if set(fact.get("evidence_ids", [])) & set(item.get("evidence_ids", [])):
                score += 1.0
            if fact.get("chunk_id") == item.get("chunk_id"):
                score += 0.75
            if fact.get("section_id") == item.get("section_id"):
                score += 0.45
            ranked.append((round(score, 8), mention_id))
        return max(ranked, default=(0.0, ""))[1]

    propagated: list[dict[str, Any]] = []
    fact_frame_links: list[dict[str, Any]] = []
    fallback_count = 0
    for fact in triples:
        decision = decision_lookup.get(fact["triple_id"])
        links = (
            list(decision["task_links"])
            if decision is not None else list(fact.get("task_links", []))
        )
        fallback_used = False
        if exhaustive and not links and frames:
            mention_id = fallback_mention(fact)
            if not mention_id:
                mention_id = next(iter(mention_to_frame), "")
            if mention_id:
                links = [
                    {
                        "task_mention_id": mention_id,
                        "relevance": "context",
                        "source": "algorithm_exhaustive_fallback",
                    }
                ]
                fallback_used = True
                fallback_count += 1
        frame_roles: dict[str, str] = {}
        frame_mentions: dict[str, list[str]] = defaultdict(list)
        for link in links:
            mention_id = link["task_mention_id"]
            frame_id = mention_to_frame.get(mention_id)
            if not frame_id:
                continue
            role = link["relevance"]
            if frame_id not in frame_roles or ROLE_RANK[role] > ROLE_RANK[frame_roles[frame_id]]:
                frame_roles[frame_id] = role
            frame_mentions[frame_id].append(mention_id)
        taskframe_links = [
            {
                "taskframe_id": frame_id,
                "relevance": role,
                "via_task_mentions": unique(frame_mentions[frame_id]),
                "propagation_path": "Fact→TaskMention→TaskFrame",
            }
            for frame_id, role in sorted(frame_roles.items())
        ]
        propagated.append(
            {
                **fact,
                "task_links": links,
                "taskframe_links": taskframe_links,
                "task_related": bool(taskframe_links),
                "exhaustive_fallback_used": fallback_used,
                "link_rejudged_by_llm_2": decision is not None,
                "link_rejudgment_reason": decision.get("reason", "") if decision else "",
            }
        )
        for link in taskframe_links:
            fact_frame_links.append(
                {
                    "fact_taskframe_link_id": stable_id(
                        "FTL", fact["triple_id"], link["taskframe_id"], link["relevance"]
                    ),
                    "triple_id": fact["triple_id"],
                    **link,
                }
            )
    return propagated, fact_frame_links, {
        "fact_count": len(propagated),
        "task_related_fact_count": sum(item["task_related"] for item in propagated),
        "unlinked_fact_count": sum(not item["task_related"] for item in propagated),
        "multi_taskframe_fact_count": sum(
            len(item["taskframe_links"]) > 1 for item in propagated
        ),
        "assigned_fact_count": sum(
            bool(item["taskframe_links"]) for item in propagated
        ),
        "all_fact_assignment_rate": (
            sum(bool(item["taskframe_links"]) for item in propagated)
            / len(propagated) if propagated else 1.0
        ),
        "exhaustive_fallback_count": fallback_count,
        "fact_taskframe_link_count": len(fact_frame_links),
        "rejudged_fact_count": sum(item["link_rejudged_by_llm_2"] for item in propagated),
    }
