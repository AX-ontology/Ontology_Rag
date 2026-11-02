from __future__ import annotations

import re
from typing import Any

from .llm import ChatJSONClient, LLMResponseError
from .prompts import extraction_prompt, structured_extraction_prompt
from .utils import as_list, batched, confidence, stable_id, text, unique


RELEVANCE = {"core", "supporting", "context"}
KNOWLEDGE_FUNCTIONS = {
    "fact", "definition", "procedure", "rule", "decision", "recommendation",
}
QUALIFIER_TYPES = {"condition", "temporal", "exception"}
ACTION_TERMS = [
    "분석", "평가", "작성", "검토", "결정", "수집", "파악", "수립", "개발",
    "선정", "관리", "기획", "설계", "검증", "승인", "제출", "발표", "협상",
    "구성", "정의", "도출", "식별", "대응", "비교", "조사", "실행", "준비",
    "제공", "기록", "확인", "설명", "공유", "통합", "분류", "측정", "추적",
    "개선", "반영", "참여", "설득", "교육", "활용", "사용", "운영", "지원",
]
PREDICATE_PATTERNS = [
    r"[가-힣A-Za-z]+(?:하여야 한다|해야 한다|할 수 있다|한다|하라|하였다|했다|된다|되었다|이다|있다|없다|의미한다|포함한다|초래한다|활용된다)",
    r"(?:필요하다|중요하다|성공한다|실패한다)",
]
SYMBOL_ONLY_RE = re.compile(r"[^0-9A-Za-z가-힣]+\Z")
HTML_TAG_RE = re.compile(r"<[^>]+>", re.IGNORECASE)
LIST_MARKER_RE = re.compile(r"(?:^|<\s*br\s*/?\s*>|\n)\s*[•·▪◦●○■□◆◇▶▷►▸*-]+", re.IGNORECASE)


def _grounded(value: str, source: str) -> bool:
    return not value or value in source


def chunk_modality(chunk: dict[str, Any]) -> str:
    return (
        "structured"
        if any(
            sentence.get("evidence_type") in {"table", "figure"}
            for sentence in chunk.get("sentences", [])
        )
        else "text"
    )


def predicate_quality_reasons(predicate: str) -> list[str]:
    reasons: list[str] = []
    value = predicate.strip()
    if not value:
        reasons.append("empty_predicate")
    elif SYMBOL_ONLY_RE.fullmatch(value):
        reasons.append("symbol_only_predicate")
    if "|" in value:
        reasons.append("table_delimiter_predicate")
    if HTML_TAG_RE.search(value):
        reasons.append("html_predicate")
    return reasons


def structured_surface_reasons(subject: str, obj: str) -> list[str]:
    reasons: list[str] = []
    for field, value in (("subject", subject), ("object", obj)):
        if "|" in value:
            reasons.append(f"multi_cell_{field}")
        if len(LIST_MARKER_RE.findall(value)) > 1:
            reasons.append(f"multi_item_{field}")
    return reasons


def coerce_extraction(raw: Any, chunk: dict[str, Any]) -> dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {}
    sentence_lookup = {
        sentence["evidence_id"]: sentence for sentence in chunk["sentences"]
    }
    allowed_evidence = set(sentence_lookup)
    errors: list[dict[str, Any]] = []

    raw_mentions = raw.get("task_mentions")
    raw_mentions = raw_mentions if isinstance(raw_mentions, list) else []
    mentions: list[dict[str, Any]] = []
    local_mention_ids: dict[str, str] = {}
    for index, item in enumerate(raw_mentions, 1):
        if not isinstance(item, dict):
            continue
        evidence_ids = [
            evidence_id for evidence_id in as_list(item.get("evidence_ids"))
            if evidence_id in allowed_evidence
        ]
        if not evidence_ids:
            errors.append({"kind": "TaskMention", "index": index, "reason": "invalid_evidence"})
            continue
        primary = evidence_ids[0]
        source = sentence_lookup[primary]["source_text"]
        surface = text(item.get("surface_form"))
        action = text(item.get("action_raw"))
        focal = text(item.get("focal_object_raw"))
        outcome = text(item.get("intended_outcome_raw"))
        grounding = {
            "surface_form": _grounded(surface, source),
            "action_raw": _grounded(action, source),
            "focal_object_raw": _grounded(focal, source),
            "intended_outcome_raw": any(
                _grounded(outcome, sentence_lookup[evidence_id]["source_text"])
                for evidence_id in evidence_ids
            ),
        }
        if not all(grounding.values()) and sentence_lookup[primary]["evidence_type"] not in {
            "table", "figure",
        }:
            errors.append(
                {
                    "kind": "TaskMention",
                    "index": index,
                    "reason": "non_extractive_surface",
                    "fields": [key for key, valid in grounding.items() if not valid],
                }
            )
        local_id = text(item.get("task_mention_id")) or f"M{index}"
        if local_id in local_mention_ids:
            local_id = f"{local_id}__{index}"
            errors.append(
                {"kind": "TaskMention", "index": index, "reason": "duplicate_local_id"}
            )
        mention_id = stable_id(
            "TM", chunk["chunk_id"], primary, index, surface, action, focal
        )
        local_mention_ids[local_id] = mention_id
        mentions.append(
            {
                "task_mention_id": mention_id,
                "source_local_id": local_id,
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "section_id": chunk["section_id"],
                "section_path": chunk["section_path"],
                "surface_form": surface or source,
                "action_raw": action,
                "focal_object_raw": focal,
                "intended_outcome_raw": outcome,
                "evidence_ids": evidence_ids,
                "source_text": source,
                "confidence": confidence(item.get("confidence")),
                "surface_grounding": grounding,
            }
        )

    # v4.4 LLM output uses `facts` with nested evidence. Keep accepting the
    # v4.3 `triples` contract so cached fixtures and heuristic smoke tests remain
    # readable, but normalize both forms to the same internal Fact model.
    uses_fact_contract = isinstance(raw.get("facts"), list)
    raw_triples = raw.get("facts") if uses_fact_contract else raw.get("triples")
    raw_triples = raw_triples if isinstance(raw_triples, list) else []
    triples: list[dict[str, Any]] = []
    local_triple_ids: dict[str, str] = {}
    triples_by_evidence: dict[str, list[str]] = {}
    for index, item in enumerate(raw_triples, 1):
        if not isinstance(item, dict):
            continue
        raw_evidence = item.get("evidence")
        raw_evidence = raw_evidence if isinstance(raw_evidence, dict) else {}
        evidence_id = text(
            raw_evidence.get("source_id") if uses_fact_contract
            else item.get("evidence_id")
        )
        if evidence_id not in allowed_evidence:
            errors.append({"kind": "Fact", "index": index, "reason": "invalid_evidence"})
            continue
        source = sentence_lookup[evidence_id]["source_text"]
        subject = text(item.get("subject_raw", item.get("subject")))
        predicate = text(item.get("raw_predicate", item.get("predicate_raw")))
        obj = text(item.get("object_raw", item.get("object")))
        structured = sentence_lookup[evidence_id]["evidence_type"] in {"table", "figure"}
        quote = text(raw_evidence.get("exact_quote")) if uses_fact_contract else source
        # Structured prompts intentionally return only source_id to avoid
        # repeating a long flattened table once per Fact. Restore the full
        # Evidence text locally; no provenance is lost.
        if uses_fact_contract and structured and not quote:
            quote = source
        quote_valid = bool(quote) and quote in source
        grounding_source = quote if quote_valid else source
        grounding = {
            "subject_raw": _grounded(subject, grounding_source),
            "predicate_raw": _grounded(predicate, grounding_source),
            "object_raw": _grounded(obj, grounding_source),
        }
        predicate_reasons = predicate_quality_reasons(predicate)
        structure_reasons = structured_surface_reasons(subject, obj) if structured else []
        if uses_fact_contract and not quote_valid:
            errors.append(
                {
                    "kind": "Fact", "index": index,
                    "reason": "invalid_exact_quote",
                }
            )
        if not all(grounding.values()) and not (structured and not uses_fact_contract):
            errors.append(
                {
                    "kind": "Fact", "index": index, "reason": "non_extractive_surface",
                    "fields": [key for key, valid in grounding.items() if not valid],
                }
            )
        for reason in [*predicate_reasons, *structure_reasons]:
            errors.append({"kind": "Fact", "index": index, "reason": reason})
        knowledge_function = text(item.get("knowledge_function")).casefold() or "fact"
        if knowledge_function not in KNOWLEDGE_FUNCTIONS:
            errors.append(
                {
                    "kind": "Fact", "index": index,
                    "reason": "invalid_knowledge_function",
                    "value": knowledge_function,
                }
            )
            knowledge_function = "fact"
        qualifiers: list[dict[str, str]] = []
        raw_qualifiers = item.get("qualifiers")
        raw_qualifiers = raw_qualifiers if isinstance(raw_qualifiers, list) else []
        for qualifier_index, qualifier in enumerate(raw_qualifiers, 1):
            if not isinstance(qualifier, dict):
                continue
            qualifier_type = text(qualifier.get("type")).casefold()
            qualifier_text = text(qualifier.get("text"))
            if qualifier_type not in QUALIFIER_TYPES:
                errors.append(
                    {
                        "kind": "Fact", "index": index,
                        "reason": "invalid_qualifier_type",
                        "qualifier_index": qualifier_index,
                        "value": qualifier_type,
                    }
                )
                continue
            if not qualifier_text or qualifier_text not in grounding_source:
                errors.append(
                    {
                        "kind": "Fact", "index": index,
                        "reason": "non_extractive_qualifier",
                        "qualifier_index": qualifier_index,
                    }
                )
                continue
            qualifiers.append({"type": qualifier_type, "text": qualifier_text})
        local_id = text(item.get("fact_id", item.get("triple_id"))) or f"T{index}"
        if local_id in local_triple_ids:
            local_id = f"{local_id}__{index}"
            errors.append({"kind": "Fact", "index": index, "reason": "duplicate_local_id"})
        triple_id = stable_id(
            "TR", chunk["chunk_id"], evidence_id, index, subject, predicate, obj
        )
        local_triple_ids[local_id] = triple_id
        triples_by_evidence.setdefault(evidence_id, []).append(local_id)
        links: list[dict[str, str]] = []
        raw_links = item.get("task_links")
        raw_links = raw_links if isinstance(raw_links, list) else []
        for link in raw_links:
            if not isinstance(link, dict):
                continue
            mention_id = local_mention_ids.get(text(link.get("task_mention_id")))
            if not mention_id:
                errors.append(
                    {"kind": "Fact", "index": index, "reason": "invalid_task_link"}
                )
                continue
            relevance = text(link.get("relevance")).casefold()
            links.append(
                {
                    "task_mention_id": mention_id,
                    "relevance": relevance if relevance in RELEVANCE else "context",
                    "source": "llm_1",
                }
            )
        triples.append(
            {
                "triple_id": triple_id,
                "source_local_id": local_id,
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "section_id": chunk["section_id"],
                "section_path": chunk["section_path"],
                "subject_raw": subject,
                "predicate_raw": predicate,
                "object_raw": obj,
                "knowledge_function": knowledge_function,
                "qualifiers": qualifiers,
                "evidence_id": evidence_id,
                "evidence_ids": [evidence_id],
                "evidence_quote": quote if quote_valid else source,
                "source_text": source,
                "task_links": _dedup_links(links),
                "confidence": confidence(item.get("confidence")),
                "surface_grounding": grounding,
                "evidence_grounding": {
                    "source_id": True,
                    "exact_quote": quote_valid if uses_fact_contract else True,
                },
                "surface_provenance_valid": (
                    all(grounding.values())
                    and (quote_valid if uses_fact_contract else True)
                    and not predicate_reasons
                    and not structure_reasons
                ),
                "quality_checks": {
                    "predicate_surface_valid": not predicate_reasons,
                    "structured_atomicity_valid": not structure_reasons,
                    "reasons": [*predicate_reasons, *structure_reasons],
                },
            }
        )

    audit_lookup: dict[str, dict[str, Any]] = {}
    audits = raw.get("sentence_audits")
    for item in audits if isinstance(audits, list) else []:
        if isinstance(item, dict) and text(item.get("evidence_id")) in allowed_evidence:
            audit_lookup[text(item["evidence_id"])] = item
    sentence_audits: list[dict[str, Any]] = []
    for evidence_id in allowed_evidence:
        actual_local = triples_by_evidence.get(evidence_id, [])
        audit = audit_lookup.get(evidence_id, {})
        reported = as_list(audit.get("fact_ids", audit.get("triple_ids")))
        if not audit:
            errors.append(
                {"kind": "SentenceAudit", "evidence_id": evidence_id, "reason": "missing"}
            )
        elif set(reported) != set(actual_local):
            errors.append(
                {
                    "kind": "SentenceAudit", "evidence_id": evidence_id,
                    "reason": "triple_id_mismatch",
                }
            )
        sentence_audits.append(
            {
                "evidence_id": evidence_id,
                "triple_ids": [
                    local_triple_ids[local_id] for local_id in actual_local
                ],
                "status": "facts_extracted" if actual_local else "no_extractable_fact",
                "reason": text(audit.get("reason")) if not actual_local else "",
            }
        )
    return {
        "chunk_id": chunk["chunk_id"],
        "task_mentions": mentions,
        "triples": triples,
        "sentence_audits": sentence_audits,
        "schema_errors": errors,
    }


def _dedup_links(links: list[dict[str, str]]) -> list[dict[str, str]]:
    rank = {"context": 1, "supporting": 2, "core": 3}
    result: dict[str, dict[str, str]] = {}
    for link in links:
        mention_id = link["task_mention_id"]
        if mention_id not in result or rank[link["relevance"]] > rank[result[mention_id]["relevance"]]:
            result[mention_id] = link
    return list(result.values())


def extract_llm(
    client: ChatJSONClient, chunks: list[dict[str, Any]], batch_size: int
) -> list[dict[str, Any]]:
    by_chunk_id: dict[str, dict[str, Any]] = {}

    def extract_batch(
        modality: str, batch: list[dict[str, Any]], key: str
    ) -> list[dict[str, Any]]:
        stage = f"llm_1_{modality}_fact_task_extraction_v1"
        system, user = (
            structured_extraction_prompt(batch)
            if modality == "structured"
            else extraction_prompt(batch)
        )
        cache_key = f"{key}-{'-'.join(item['chunk_id'] for item in batch)}"
        try:
            payload = client.call(stage, cache_key, system, user)
            raw_chunks = payload.get("chunks", []) if isinstance(payload, dict) else []
            raw_by_id = {
                text(item.get("chunk_id")): item
                for item in raw_chunks if isinstance(item, dict)
            }
            missing = [item["chunk_id"] for item in batch if item["chunk_id"] not in raw_by_id]
            if missing:
                raise LLMResponseError(f"LLM extraction omitted chunks: {missing}")
        except LLMResponseError as exc:
            if len(batch) == 1:
                raise
            middle = len(batch) // 2
            print(
                f"[step2/{modality}-split] key={key} size={len(batch)} "
                f"error={type(exc).__name__}"
            )
            return [
                *extract_batch(modality, batch[:middle], f"{key}a"),
                *extract_batch(modality, batch[middle:], f"{key}b"),
            ]
        return [
            coerce_extraction(raw_by_id.get(chunk["chunk_id"], {}), chunk)
            for chunk in batch
        ]

    for modality in ("text", "structured"):
        modality_chunks = [item for item in chunks if chunk_modality(item) == modality]
        for batch_no, batch in enumerate(batched(modality_chunks, batch_size), 1):
            for item in extract_batch(modality, batch, str(batch_no)):
                by_chunk_id[item["chunk_id"]] = item
            done = min(batch_no * batch_size, len(modality_chunks))
            print(f"[step2/{modality}] chunks={done}/{len(modality_chunks)}")
    return [by_chunk_id[item["chunk_id"]] for item in chunks]


def _raw_existing_mentions(extraction: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "task_mention_id": item.get("source_local_id") or f"M{index}",
            "surface_form": item.get("surface_form", ""),
            "action_raw": item.get("action_raw", ""),
            "focal_object_raw": item.get("focal_object_raw", ""),
            "intended_outcome_raw": item.get("intended_outcome_raw", ""),
            "evidence_ids": item.get("evidence_ids", []),
            "confidence": item.get("confidence", 0.0),
        }
        for index, item in enumerate(extraction.get("task_mentions", []), 1)
    ]


def extract_structured_refinement_llm(
    client: ChatJSONClient,
    chunks: list[dict[str, Any]],
    previous_extractions: dict[str, dict[str, Any]],
    batch_size: int,
) -> list[dict[str, Any]]:
    """Re-extract only table/figure Facts while preserving TaskMention IDs."""
    structured_chunks = [item for item in chunks if chunk_modality(item) == "structured"]
    results: list[dict[str, Any]] = []

    def refine_batch(batch: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
        stage = "llm_1_structured_fact_refinement_v1"
        existing_mentions = {
            chunk["chunk_id"]: _raw_existing_mentions(
                previous_extractions[chunk["chunk_id"]]
            )
            for chunk in batch
        }
        system, user = structured_extraction_prompt(batch, existing_mentions)
        cache_key = f"{key}-{'-'.join(item['chunk_id'] for item in batch)}"
        if (
            len(batch) > 1
            and len(user) > 9000
            and not getattr(
                client, "has_cached_call", lambda *_args: False
            )(stage, cache_key, system, user)
        ):
            middle = len(batch) // 2
            print(
                f"[step2/structured-refine-presplit] key={key} size={len(batch)} "
                f"prompt_chars={len(user)}"
            )
            return [
                *refine_batch(batch[:middle], f"{key}a"),
                *refine_batch(batch[middle:], f"{key}b"),
            ]
        try:
            payload = client.call(
                stage,
                cache_key,
                system,
                user,
            )
        except RuntimeError as exc:
            if len(batch) == 1:
                chunk = batch[0]
                fallback = coerce_extraction(
                    {
                        "chunk_id": chunk["chunk_id"],
                        "task_mentions": existing_mentions[chunk["chunk_id"]],
                        "open_triples": [],
                        "sentence_audits": [],
                    },
                    chunk,
                )
                fallback["structured_refinement"] = {
                    "status": "no_facts_after_llm_failure",
                    "reason": str(exc)[:500],
                }
                print(
                    "[step2/structured-refine-fallback] "
                    f"chunk_id={chunk['chunk_id']} facts=0"
                )
                return [fallback]
            middle = len(batch) // 2
            print(
                f"[step2/structured-refine-split] key={key} size={len(batch)} "
                f"error={type(exc).__name__}"
            )
            return [
                *refine_batch(batch[:middle], f"{key}a"),
                *refine_batch(batch[middle:], f"{key}b"),
            ]
        raw_chunks = payload.get("chunks", []) if isinstance(payload, dict) else []
        raw_by_id = {
            text(item.get("chunk_id")): item
            for item in raw_chunks if isinstance(item, dict)
        }
        batch_results: list[dict[str, Any]] = []
        for chunk in batch:
            chunk_id = chunk["chunk_id"]
            raw = dict(raw_by_id.get(chunk_id, {}))
            raw["task_mentions"] = existing_mentions[chunk_id]
            refined = coerce_extraction(raw, chunk)
            old_ids = [
                item["task_mention_id"]
                for item in previous_extractions[chunk_id].get("task_mentions", [])
            ]
            new_ids = [item["task_mention_id"] for item in refined["task_mentions"]]
            if old_ids != new_ids:
                raise ValueError(f"structured refinement changed TaskMention IDs: {chunk_id}")
            batch_results.append(refined)
        return batch_results

    for batch_no, batch in enumerate(batched(structured_chunks, batch_size), 1):
        results.extend(refine_batch(batch, str(batch_no)))
        done = min(batch_no * batch_size, len(structured_chunks))
        print(f"[step2/structured-refine] chunks={done}/{len(structured_chunks)}")
    return results


def extract_heuristic(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for chunk in chunks:
        raw_mentions: list[dict[str, Any]] = []
        raw_triples: list[dict[str, Any]] = []
        audits: list[dict[str, Any]] = []
        for sentence in chunk["sentences"]:
            evidence_id, source = sentence["evidence_id"], sentence["source_text"]
            triple_keys: list[str] = []
            predicate_matches = [
                match for pattern in PREDICATE_PATTERNS
                for match in re.finditer(pattern, source)
            ]
            predicate_matches = sorted(
                {
                    (match.start(), match.end(), match.group(0)): match
                    for match in predicate_matches
                }.values(),
                key=lambda match: (match.start(), match.end()),
            )
            for match in predicate_matches[:3]:
                predicate = match.group(0)
                left = source[: match.start()].strip(" ,.;:")
                right = source[match.end() :].strip(" ,.;:")
                subject_match = re.search(r"([^,.]{1,80}?)(?:은|는|이|가)\s*$", left)
                subject = subject_match.group(1).strip() if subject_match else left[-60:]
                object_match = re.search(r"([^,.]{1,80}?)(?:을|를)\s*$", left)
                obj = object_match.group(1).strip() if object_match else right[:80]
                triple_key = f"T{len(raw_triples) + 1}"
                action = next(
                    (term for term in ACTION_TERMS if term in predicate or term in left), ""
                )
                mention_key = ""
                if action and sentence["evidence_role"] == "content":
                    mention_key = f"M{len(raw_mentions) + 1}"
                    raw_mentions.append(
                        {
                            "task_mention_id": mention_key,
                            "surface_form": source,
                            "action_raw": action,
                            "focal_object_raw": obj,
                            "intended_outcome_raw": "",
                            "evidence_ids": [evidence_id],
                            "confidence": 0.55,
                        }
                    )
                raw_triples.append(
                    {
                        "triple_id": triple_key,
                        "subject_raw": subject if subject in source else "",
                        "predicate_raw": predicate,
                        "object_raw": obj if obj in source else "",
                        "evidence_id": evidence_id,
                        "task_links": (
                            [{"task_mention_id": mention_key, "relevance": "core"}]
                            if mention_key else []
                        ),
                        "confidence": 0.5,
                    }
                )
                triple_keys.append(triple_key)
            audits.append(
                {
                    "evidence_id": evidence_id,
                    "triple_ids": triple_keys,
                    "status": "facts_extracted" if triple_keys else "no_extractable_fact",
                    "reason": "" if triple_keys else "규칙 기반 smoke parser에서 명시 술어를 찾지 못함",
                }
            )
        results.append(
            coerce_extraction(
                {
                    "task_mentions": raw_mentions,
                    "triples": raw_triples,
                    "sentence_audits": audits,
                },
                chunk,
            )
        )
    return results


def flatten(
    extractions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    return (
        [item for result in extractions for item in result["task_mentions"]],
        [item for result in extractions for item in result["triples"]],
        [item for result in extractions for item in result["sentence_audits"]],
    )
