from __future__ import annotations

from collections import Counter, defaultdict
import copy
import re
from typing import Any

from .utils import stable_id, text, unique


PARTICLES = (
    "으로", "에서", "에게", "까지", "부터", "처럼", "보다",
    "은", "는", "이", "가", "을", "를", "로",
)
EXCEPTIONS = {"전문가", "평가", "원가", "국가", "대가", "효과", "추가", "결과"}
ACTION_MAP = {
    "분석한다": "분석하다", "분석하라": "분석하다",
    "검토한다": "검토하다", "검토하라": "검토하다",
    "작성한다": "작성하다", "작성하라": "작성하다",
    "수립한다": "수립하다", "수립하라": "수립하다",
    "결정한다": "결정하다", "확인한다": "확인하다",
    "포함한다": "포함하다", "의미한다": "의미하다",
    "활용된다": "활용되다", "사용한다": "사용하다",
}


def load_aliases(payload: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    if isinstance(payload, dict):
        iterator = payload.items()
    elif isinstance(payload, list):
        iterator = (
            (alias, item.get("canonical", ""))
            for item in payload if isinstance(item, dict)
            for alias in [item.get("canonical", "")] + (
                item.get("aliases", []) if isinstance(item.get("aliases"), list)
                else [item.get("aliases", "")]
            )
        )
    else:
        iterator = []
    for alias, canonical in iterator:
        alias_value, canonical_value = text(alias), text(canonical)
        if alias_value and canonical_value:
            result[alias_value.casefold()] = canonical_value
            result[canonical_value.casefold()] = canonical_value
    return result


def normalize_entity(value: Any, aliases: dict[str, str] | None = None) -> str:
    normalized = text(value).strip(" \t\n.,;:\"'“”‘’")
    if normalized not in EXCEPTIONS:
        for particle in PARTICLES:
            if normalized.endswith(particle):
                stem = normalized[: -len(particle)].rstrip()
                if len(stem) >= 2 and stem not in EXCEPTIONS:
                    normalized = stem
                    break
    return aliases.get(normalized.casefold(), normalized) if aliases else normalized


def normalize_action(value: Any) -> str:
    normalized = text(value).strip(" \t\n.,;:")
    if normalized in ACTION_MAP:
        return ACTION_MAP[normalized]
    match = re.fullmatch(
        r"(.+?)(?:하여야 한다|해야 한다|하도록 한다|하였다|했다|한다|하라|하여|하고|함)",
        normalized,
    )
    if match:
        return f"{match.group(1).strip()}하다"
    match = re.fullmatch(r"(.+?)(?:되었다|된다|되다|됨)", normalized)
    if match:
        return f"{match.group(1).strip()}되다"
    return normalized


def conservative_prepare(
    mentions: list[dict[str, Any]],
    triples: list[dict[str, Any]],
    aliases: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prepared_mentions: list[dict[str, Any]] = []
    for mention in mentions:
        item = copy.deepcopy(mention)
        item["action_normalized"] = normalize_action(item.get("action_raw"))
        item["focal_object_normalized"] = normalize_entity(
            item.get("focal_object_raw"), aliases
        )
        item["normalization_method"] = {
            "action": "morphological_rule",
            "focal_object": (
                "verified_alias"
                if text(item.get("focal_object_raw")).casefold() in aliases
                else "surface_cleanup"
            ),
        }
        prepared_mentions.append(item)
    prepared_triples: list[dict[str, Any]] = []
    for triple in triples:
        item = copy.deepcopy(triple)
        item["subject_normalized"] = normalize_entity(item.get("subject_raw"), aliases)
        item["predicate_normalized"] = normalize_action(item.get("predicate_raw"))
        item["object_normalized"] = normalize_entity(item.get("object_raw"), aliases)
        item["raw_fact_preserved"] = True
        item["canonicalization_status"] = (
            "review_required"
            if not item.get("surface_provenance_valid", False)
            or not item["predicate_normalized"]
            else "normalized"
        )
        prepared_triples.append(item)
    return prepared_mentions, prepared_triples


def canonicalize(
    triples: list[dict[str, Any]],
    aliases: dict[str, str],
    relation_aliases: dict[str, str],
    promotion_min_support: int,
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
]:
    canonical_facts: list[dict[str, Any]] = []
    entity_forms: dict[str, set[str]] = defaultdict(set)
    relation_forms: dict[str, set[str]] = defaultdict(set)
    entity_sections: dict[str, set[str]] = defaultdict(set)
    relation_sections: dict[str, set[str]] = defaultdict(set)
    for triple in triples:
        item = copy.deepcopy(triple)
        subject = normalize_entity(item.get("subject_raw"), aliases)
        obj = normalize_entity(item.get("object_raw"), aliases)
        predicate_raw = text(item.get("predicate_raw"))
        predicate_key = normalize_action(predicate_raw)
        canonical_relation = relation_aliases.get(
            predicate_raw.casefold(),
            relation_aliases.get(predicate_key.casefold(), predicate_key),
        )
        item.update(
            {
                "subject_canonical": subject,
                "predicate_canonical": canonical_relation,
                "object_canonical": obj,
                "canonical_fact_id": stable_id(
                    "CF", subject, canonical_relation, obj
                ) if canonical_relation else "",
                "relation_mapping_status": (
                    "verified_alias"
                    if predicate_raw.casefold() in relation_aliases
                    else "surface_normalized"
                ),
            }
        )
        canonical_facts.append(item)
        for raw, canonical in (
            (item.get("subject_raw"), subject), (item.get("object_raw"), obj)
        ):
            if canonical:
                entity_forms[canonical].add(text(raw))
                entity_sections[canonical].add(item["section_id"])
        if canonical_relation:
            relation_forms[canonical_relation].add(predicate_raw)
            relation_sections[canonical_relation].add(item["section_id"])
    entities = [
        {
            "canonical_entity_id": stable_id("CE", canonical),
            "canonical_name": canonical,
            "surface_forms": sorted(forms),
            "support_count": sum(
                (item["subject_canonical"] == canonical)
                + (item["object_canonical"] == canonical)
                for item in canonical_facts
            ),
            "section_count": len(entity_sections[canonical]),
            "mapping_status": "stable" if len(forms) > 1 else "observed",
        }
        for canonical, forms in sorted(entity_forms.items())
    ]
    relations = [
        {
            "canonical_relation_id": stable_id("CR", canonical),
            "canonical_name": canonical,
            "surface_forms": sorted(forms),
            "support_count": sum(
                item["predicate_canonical"] == canonical for item in canonical_facts
            ),
            "section_count": len(relation_sections[canonical]),
            "mapping_status": (
                "verified_alias"
                if any(form.casefold() in relation_aliases for form in forms)
                else "unresolved"
            ),
        }
        for canonical, forms in sorted(relation_forms.items())
    ]
    promoted: list[dict[str, Any]] = []
    subject_types: dict[str, Counter[str]] = defaultdict(Counter)
    object_types: dict[str, Counter[str]] = defaultdict(Counter)
    for item in canonical_facts:
        relation = item["predicate_canonical"]
        if relation:
            subject_types[relation][_type_hint(item["subject_canonical"])] += 1
            object_types[relation][_type_hint(item["object_canonical"])] += 1
    for relation in relations:
        if (
            relation["support_count"] >= promotion_min_support
            and relation["section_count"] >= 2
            and relation["mapping_status"] != "unresolved"
        ):
            name = relation["canonical_name"]
            promoted.append(
                {
                    "schema_term_id": stable_id("ONT", name),
                    "kind": "relation",
                    "canonical_name": name,
                    "domain": subject_types[name].most_common(1)[0][0],
                    "range": object_types[name].most_common(1)[0][0],
                    "support_count": relation["support_count"],
                    "source_sections": sorted(relation_sections[name]),
                    "schema_version": "4.5.1",
                    "validation_status": "candidate",
                }
            )
    return canonical_facts, entities, relations, promoted


def _type_hint(value: str) -> str:
    if not value:
        return "Unknown"
    if re.search(r"(문서|제안서|RFP|보고서|표|계획서|계약서)$", value, re.I):
        return "DocumentArtifact"
    if re.search(r"(담당자|대표|PM|위원|고객|조직|팀)$", value, re.I):
        return "ActorOrOrganization"
    if re.search(r"(분석|검토|수립|작성|관리|결정)$", value):
        return "TaskConcept"
    return "DomainEntity"
