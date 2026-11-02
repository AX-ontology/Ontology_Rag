from __future__ import annotations

from collections import Counter
import json
from typing import Any

from .utils import stable_id, text


FRAME_FACT_RELATION = {
    "core": "HAS_CORE_FACT",
    "supporting": "HAS_SUPPORTING_FACT",
    "context": "HAS_CONTEXT_FACT",
}
TASK_RELATION = {
    "sameTask": "SAME_TASK",
    "subTaskOf": "SUBTASK_OF",
    "relatedTask": "RELATED_TASK",
    "precedes": "PRECEDES",
    "follows": "FOLLOWS",
    "supports": "SUPPORTS",
    "distinctTask": "DISTINCT_TASK",
}


def build_graph(
    document_id: str,
    document_title: str,
    evidence: list[dict[str, Any]],
    mentions: list[dict[str, Any]],
    facts: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    task_relations: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    promoted_terms: list[dict[str, Any]],
    standards: list[dict[str, Any]],
    standard_mappings: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[str, dict[str, Any]] = {}

    def node(node_id: str, label: str, name: str, **properties: Any) -> None:
        previous = nodes.get(node_id)
        if previous and (previous["label"], previous["name"]) != (label, name):
            raise ValueError(f"node id collision: {node_id}")
        nodes.setdefault(
            node_id,
            {"id": node_id, "label": label, "name": name, "properties": properties},
        )

    def edge(
        source: str, target: str, relation: str, layer: str,
        evidence_ids: list[str] | None = None, identity: str = "", **properties: Any,
    ) -> None:
        edge_id = stable_id("EDGE", source, relation, target, layer, identity)
        edges[edge_id] = {
            "id": edge_id,
            "source": source,
            "target": target,
            "type": relation,
            "layer": layer,
            "evidence_ids": evidence_ids or [],
            "properties": properties,
        }

    node(document_id, "Document", document_title, schema_version="4.5.1")
    seen_sections: set[str] = set()
    for item in evidence:
        section_id = item["section_id"]
        if section_id not in seen_sections:
            seen_sections.add(section_id)
            node(
                section_id, "Section", item["section_path"],
                section_no=item["section_no"], chapter_no=item["chapter_no"],
            )
            edge(document_id, section_id, "HAS_SECTION", "source")
        node(
            item["evidence_id"], "EvidenceSpan", item["source_text"],
            document_id=item["document_id"],
            section_path=item["section_path"],
            paragraph_id=item["paragraph_id"],
            sentence_id=item["sentence_id"],
            page=item["page"],
            position=item["position"],
            evidence_type=item["evidence_type"],
            source_file=item["source_file"],
        )
        edge(section_id, item["evidence_id"], "HAS_EVIDENCE", "source")

    canonical_entity_lookup = {
        item["canonical_name"]: item["canonical_entity_id"] for item in entities
    }
    for item in entities:
        node(
            item["canonical_entity_id"], "CanonicalEntity", item["canonical_name"],
            surface_forms=item["surface_forms"], support_count=item["support_count"],
            mapping_status=item["mapping_status"],
        )
    canonical_relation_lookup = {
        item["canonical_name"]: item["canonical_relation_id"] for item in relations
    }
    for item in relations:
        node(
            item["canonical_relation_id"], "CanonicalTerm", item["canonical_name"],
            term_kind="relation", surface_forms=item["surface_forms"],
            support_count=item["support_count"], mapping_status=item["mapping_status"],
        )
    for item in promoted_terms:
        node(
            item["schema_term_id"], "OntologyTerm", item["canonical_name"], **{
                key: value for key, value in item.items() if key != "schema_term_id"
            }
        )
        canonical_id = canonical_relation_lookup.get(item["canonical_name"])
        if canonical_id:
            edge(canonical_id, item["schema_term_id"], "PROMOTED_AS", "ontology")

    mention_lookup = {item["task_mention_id"]: item for item in mentions}
    for mention in mentions:
        node(
            mention["task_mention_id"], "TaskMention", mention["surface_form"],
            action_raw=mention.get("action_raw", ""),
            focal_object_raw=mention.get("focal_object_raw", ""),
            intended_outcome_raw=mention.get("intended_outcome_raw", ""),
            section_path=mention.get("section_path", ""),
            confidence=mention.get("confidence", 0),
        )
        for evidence_id in mention["evidence_ids"]:
            edge(
                mention["task_mention_id"], evidence_id, "SUPPORTED_BY",
                "task_mention", [evidence_id], identity=evidence_id,
            )
    frame_lookup = {item["taskframe_id"]: item for item in frames}
    for frame in frames:
        node(
            frame["taskframe_id"], "TaskFrame", frame["canonical_name"],
            definition=frame["definition"], scope=frame["scope"],
            aliases=frame["aliases"], status=frame["status"],
            schema_version=frame["schema_version"],
        )
        for mention_id in frame["member_task_mentions"]:
            edge(mention_id, frame["taskframe_id"], "MEMBER_OF", "taskframe")
    for item in task_relations:
        edge(
            item["source_taskframe_id"], item["target_taskframe_id"],
            TASK_RELATION[item["relation"]], "task_relation",
            identity=item["task_relation_id"], confidence=item["confidence"],
            reason=item["reason"],
        )

    for fact in facts:
        display = " → ".join(
            [
                fact.get("subject_raw") or "(생략 주체)",
                fact.get("predicate_raw", ""),
                fact.get("object_raw") or "(생략 목적어)",
            ]
        )
        node(
            fact["triple_id"], "Fact", display,
            subject_raw=fact.get("subject_raw", ""),
            predicate_raw=fact.get("predicate_raw", ""),
            object_raw=fact.get("object_raw", ""),
            knowledge_function=fact.get("knowledge_function", "fact"),
            qualifiers=fact.get("qualifiers", []),
            evidence_quote=fact.get("evidence_quote", fact.get("source_text", "")),
            subject_canonical=fact.get("subject_canonical", ""),
            predicate_canonical=fact.get("predicate_canonical", ""),
            object_canonical=fact.get("object_canonical", ""),
            canonical_fact_id=fact.get("canonical_fact_id", ""),
            raw_fact_preserved=fact.get("raw_fact_preserved", False),
            task_related=fact.get("task_related", False),
            link_rejudged_by_llm_2=fact.get("link_rejudged_by_llm_2", False),
        )
        for evidence_id in fact["evidence_ids"]:
            edge(
                fact["triple_id"], evidence_id, "SUPPORTED_BY", "fact",
                [evidence_id], identity=evidence_id,
            )
        for role, raw_key, canonical_key, relation_name in (
            ("subject", "subject_raw", "subject_canonical", "HAS_SUBJECT"),
            ("object", "object_raw", "object_canonical", "HAS_OBJECT"),
        ):
            raw_value = text(fact.get(raw_key))
            if not raw_value:
                continue
            entity_mention_id = stable_id("EM", fact["triple_id"], role, raw_value)
            node(
                entity_mention_id, "EntityMention", raw_value,
                mention_role=role, evidence_id=fact["evidence_id"],
            )
            edge(fact["triple_id"], entity_mention_id, relation_name, "fact")
            canonical_id = canonical_entity_lookup.get(fact.get(canonical_key, ""))
            if canonical_id:
                edge(
                    entity_mention_id, canonical_id, "CANONICALIZED_AS",
                    "canonicalization",
                )
        predicate_raw = text(fact.get("predicate_raw"))
        if predicate_raw:
            expression_id = stable_id("RE", fact["triple_id"], predicate_raw)
            node(
                expression_id, "RelationExpression", predicate_raw,
                evidence_id=fact["evidence_id"],
            )
            edge(fact["triple_id"], expression_id, "HAS_PREDICATE", "fact")
            canonical_id = canonical_relation_lookup.get(
                fact.get("predicate_canonical", "")
            )
            if canonical_id:
                edge(
                    expression_id, canonical_id, "CANONICALIZED_AS",
                    "canonicalization",
                )
        for link in fact.get("task_links", []):
            mention_id = link["task_mention_id"]
            if mention_id in mention_lookup:
                edge(
                    fact["triple_id"], mention_id, "RELATED_TO_TASK_MENTION",
                    "fact_task", fact["evidence_ids"], identity=mention_id,
                    relevance=link["relevance"],
                    link_source=link.get("source", ""),
                )
        for link in fact.get("taskframe_links", []):
            frame_id = link["taskframe_id"]
            if frame_id in frame_lookup:
                edge(
                    frame_id, fact["triple_id"],
                    FRAME_FACT_RELATION[link["relevance"]], "task_subgraph",
                    fact["evidence_ids"], identity=frame_id,
                    via_task_mentions=link["via_task_mentions"],
                    propagation_path=link["propagation_path"],
                )

    standard_lookup = {item["standard_task_id"]: item for item in standards}
    for item in standards:
        node(
            item["standard_task_id"], "StandardTask", item["name"],
            scheme=item["scheme"], source_standard_id=item["source_standard_id"],
            definition=item["definition"], parent=item["parent"],
        )
    for mapping in standard_mappings:
        if (
            mapping["taskframe_id"] in frame_lookup
            and mapping["standard_task_id"] in standard_lookup
        ):
            edge(
                mapping["taskframe_id"], mapping["standard_task_id"],
                "MAPS_TO_STANDARD_TASK", "standard_mapping",
                identity=mapping["standard_mapping_id"],
                relation=mapping["relation"], score=mapping["score"],
                validation_status=mapping["validation_status"],
            )
    node_records, edge_records = list(nodes.values()), list(edges.values())
    return node_records, edge_records, validate_graph(
        node_records, edge_records, evidence, mentions, facts, frames
    )


def validate_graph(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    mentions: list[dict[str, Any]],
    facts: list[dict[str, Any]],
    frames: list[dict[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    node_ids = [item["id"] for item in nodes]
    node_set = set(node_ids)
    if len(node_ids) != len(node_set):
        errors.append("duplicate node id")
    if len(edges) != len({item["id"] for item in edges}):
        errors.append("duplicate edge id")
    for item in edges:
        if item["source"] not in node_set or item["target"] not in node_set:
            errors.append(f"dangling edge: {item['id']}")
    evidence_ids = {item["evidence_id"] for item in evidence}
    mention_ids = {item["task_mention_id"] for item in mentions}
    member_counts = Counter(
        mention_id for frame in frames for mention_id in frame["member_task_mentions"]
    )
    if set(member_counts) != mention_ids or any(value != 1 for value in member_counts.values()):
        errors.append("TaskMention partition invalid")
    for item in facts:
        if not set(item["evidence_ids"]) <= evidence_ids:
            errors.append(f"Fact evidence invalid: {item['triple_id']}")
        if not item.get("raw_fact_preserved"):
            errors.append(f"raw Fact not preserved: {item['triple_id']}")
    for item in mentions:
        if not set(item["evidence_ids"]) <= evidence_ids:
            errors.append(f"TaskMention evidence invalid: {item['task_mention_id']}")
    unlinked = sum(not item.get("taskframe_links") for item in facts)
    if unlinked:
        errors.append(
            f"TaskFrame에 연결되지 않은 Fact가 {unlinked}개 존재함"
        )
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes_by_label": dict(sorted(Counter(item["label"] for item in nodes).items())),
        "edges_by_type": dict(sorted(Counter(item["type"] for item in edges).items())),
        "unlinked_general_fact_count": unlinked,
        "fact_evidence_grounding_rate": (
            sum(
                bool(item["evidence_ids"])
                and set(item["evidence_ids"]) <= evidence_ids
                for item in facts
            ) / len(facts) if facts else 1.0
        ),
        "taskmention_evidence_grounding_rate": (
            sum(
                bool(item["evidence_ids"])
                and set(item["evidence_ids"]) <= evidence_ids
                for item in mentions
            ) / len(mentions) if mentions else 1.0
        ),
    }


def csv_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": item["id"], "label": item["label"], "name": item["name"],
            "properties_json": json.dumps(item["properties"], ensure_ascii=False),
        }
        for item in nodes
    ]


def csv_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": item["id"], "source": item["source"], "target": item["target"],
            "type": item["type"], "layer": item["layer"],
            "evidence_ids": "|".join(item["evidence_ids"]),
            "properties_json": json.dumps(item["properties"], ensure_ascii=False),
        }
        for item in edges
    ]
