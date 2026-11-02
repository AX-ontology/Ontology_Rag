from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import (
    load_config,
    output_path,
    read_jsonl,
    stable_id,
    write_json,
    write_jsonl,
)


RELATION_TO_TASKKG = {
    "equivalent": "exactMatch",
    "taskframe_narrower_than_pcf": "closeMatch",
    "taskframe_broader_than_pcf": "closeMatch",
    "partial_overlap": "relatedMatch",
    "related": "relatedMatch",
}


def export(config: dict[str, Any]) -> dict[str, Any]:
    nodes = read_jsonl(output_path(config, "pcf_nodes_enriched.jsonl"))
    links = read_jsonl(output_path(config, "taskframe_pcf_links.jsonl"))
    node_id_by_hid = {node["hid"]: f"PCF_{node['pcfid']}" for node in nodes}
    standard_id_by_hid = {
        node["hid"]: stable_id("STD", "PCF", node["hid"], node.get("nameKo", ""))
        for node in nodes
    }

    standards = [
        {
            "id": node["hid"],
            "code": node["hid"],
            "scheme": "PCF",
            "name": node.get("nameKo") or node.get("nameEn", ""),
            "name_en": node.get("nameEn", ""),
            "definition": node.get("descKo") or node.get("descEn", ""),
            "definition_en": node.get("descEn", ""),
            "parent": node.get("parent") or "",
            "level": node.get("level"),
            "pcfid": node.get("pcfid"),
            "path_ids": node.get("path_ids", []),
        }
        for node in nodes
    ]
    standard_mappings: list[dict[str, Any]] = []
    for link in links:
        hid = link.get("pcf_hid")
        if not hid or link.get("mapping_role") != "process_match":
            continue
        standard_task_id = standard_id_by_hid[hid]
        standard_mappings.append(
            {
                "standard_mapping_id": stable_id(
                    "SMAP", link["taskframe_id"], standard_task_id
                ),
                "taskframe_id": link["taskframe_id"],
                "standard_task_id": standard_task_id,
                "source_standard_id": hid,
                "scheme": "PCF",
                "relation": RELATION_TO_TASKKG[link["relation_scope"]],
                "score": link["confidence"],
                "method": link["method"],
                "validation_status": link["validation_status"],
                "is_primary": link["is_primary"],
                "supporting_task_mentions": link["supporting_task_mentions"],
                "supporting_triples": link["supporting_triples"],
            }
        )

    graph_nodes = [
        {
            "id": node_id_by_hid[node["hid"]],
            "label": "PCFProcess",
            "name": node.get("nameKo") or node.get("nameEn", ""),
            "properties": {
                "pcfid": node.get("pcfid"),
                "hid": node["hid"],
                "level": node.get("level"),
                "name_en": node.get("nameEn", ""),
                "description_ko": node.get("descKo", ""),
                "description_en": node.get("descEn", ""),
                "path_ids": node.get("path_ids", []),
            },
        }
        for node in nodes
    ]
    graph_edges: list[dict[str, Any]] = []
    for node in nodes:
        parent = node.get("parent")
        if parent:
            graph_edges.append(
                {
                    "id": stable_id("EDGE", node_id_by_hid[node["hid"]], "CHILD_OF", node_id_by_hid[parent]),
                    "source": node_id_by_hid[node["hid"]],
                    "target": node_id_by_hid[parent],
                    "type": "CHILD_OF",
                    "layer": "pcf",
                    "evidence_ids": [],
                    "properties": {},
                }
            )
    for link in links:
        hid = link.get("pcf_hid")
        if not hid:
            continue
        graph_edges.append(
            {
                "id": stable_id("EDGE", link["taskframe_id"], "MAPS_TO_PCF", node_id_by_hid[hid], link["mapping_role"]),
                "source": link["taskframe_id"],
                "target": node_id_by_hid[hid],
                "type": "MAPS_TO_PCF",
                "layer": "layer_connect",
                "evidence_ids": link["supporting_evidence_ids"],
                "properties": {
                    "role": link["mapping_role"],
                    "is_primary": link["is_primary"],
                    "relation_scope": link["relation_scope"],
                    "confidence": link["confidence"],
                    "validation_status": link["validation_status"],
                    "supporting_task_mentions": link["supporting_task_mentions"],
                    "supporting_triples": link["supporting_triples"],
                    "rationale": link["rationale"],
                },
            }
        )

    write_json(output_path(config, "standard_tasks_pcf.json"), {"standard_tasks": standards})
    write_jsonl(output_path(config, "standard_mappings_pcf.jsonl"), standard_mappings)
    write_jsonl(output_path(config, "pcf_kg_nodes.jsonl"), graph_nodes)
    write_jsonl(output_path(config, "pcf_kg_edges.jsonl"), graph_edges)
    report = {
        "standard_task_count": len(standards),
        "standard_mapping_count": len(standard_mappings),
        "pcf_kg_node_count": len(graph_nodes),
        "pcf_child_edge_count": sum(item["type"] == "CHILD_OF" for item in graph_edges),
        "taskframe_mapping_edge_count": sum(item["type"] == "MAPS_TO_PCF" for item in graph_edges),
        "outputs": {
            "standard_tasks": str(output_path(config, "standard_tasks_pcf.json")),
            "standard_mappings": str(output_path(config, "standard_mappings_pcf.jsonl")),
            "kg_nodes": str(output_path(config, "pcf_kg_nodes.jsonl")),
            "kg_edges": str(output_path(config, "pcf_kg_edges.jsonl")),
        },
    }
    write_json(output_path(config, "export_summary.json"), report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crosswalk를 기존 TaskKG/그래프 형식으로 내보냅니다.")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = export(load_config(args.config))
    print(
        f"Export: {report['standard_task_count']} PCF nodes, "
        f"{report['taskframe_mapping_edge_count']} mapping edges"
    )


if __name__ == "__main__":
    main()
