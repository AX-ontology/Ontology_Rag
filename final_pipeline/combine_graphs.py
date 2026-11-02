#!/usr/bin/env python3
"""Combine per-document cleaned graph packages into one validated graph."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable


COUNT_PROPERTIES = {"support_count", "section_count"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_no}: JSON object가 필요합니다.")
            records.append(item)
    return records


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    temporary.replace(path)
    return count


def _unique(values: Iterable[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()
    for value in values:
        key = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _merge_properties(
    target: dict[str, Any], incoming: dict[str, Any], *, sum_counts: bool
) -> dict[str, Any]:
    result = deepcopy(target)
    for key, value in incoming.items():
        if key not in result or result[key] in (None, "", []):
            result[key] = deepcopy(value)
            continue
        previous = result[key]
        if sum_counts and key in COUNT_PROPERTIES and type(previous) in (int, float) and type(value) in (int, float):
            result[key] = previous + value
            continue
        if previous == value:
            continue
        if isinstance(previous, list) and isinstance(value, list):
            result[key] = _unique([*previous, *value])
        elif isinstance(previous, bool) and isinstance(value, bool):
            result[key] = previous or value
        else:
            values_key = f"{key}Values"
            current_values = result.get(values_key, [previous])
            result[values_key] = _unique([*current_values, value])
    return result


def validate_graph(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
) -> dict[str, Any]:
    node_ids = [str(item.get("id") or "") for item in nodes]
    edge_ids = [str(item.get("id") or "") for item in edges]
    node_set = set(node_ids)
    errors: list[str] = []
    if "" in node_set:
        errors.append("node id 누락")
    if "" in set(edge_ids):
        errors.append("edge id 누락")
    if len(node_ids) != len(node_set):
        errors.append("duplicate node id")
    if len(edge_ids) != len(set(edge_ids)):
        errors.append("duplicate edge id")
    dangling = [
        item.get("id")
        for item in edges
        if item.get("source") not in node_set or item.get("target") not in node_set
    ]
    if dangling:
        errors.append(f"dangling edge count: {len(dangling)}")
    return {
        "valid": not errors,
        "errors": errors,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes_by_label": dict(
            sorted(Counter(item.get("label", "Unknown") for item in nodes).items())
        ),
        "edges_by_type": dict(
            sorted(Counter(item.get("type", "UNKNOWN") for item in edges).items())
        ),
        "dangling_edge_count": len(dangling),
    }


def combine_inputs(
    inputs: list[tuple[str, Path]], output_dir: Path
) -> dict[str, Any]:
    if not inputs:
        raise ValueError("통합할 문서 그래프가 없습니다.")
    nodes_by_id: dict[str, dict[str, Any]] = {}
    edges_by_id: dict[str, dict[str, Any]] = {}
    duplicate_nodes = 0
    duplicate_edges = 0

    for document_id, graph_dir in inputs:
        graph_dir = graph_dir.expanduser().resolve()
        node_path = graph_dir / "nodes.jsonl"
        edge_path = graph_dir / "edges.jsonl"
        if not node_path.exists() or not edge_path.exists():
            raise FileNotFoundError(
                f"{document_id}: 그래프 입력 누락: {node_path}, {edge_path}"
            )
        for raw in read_jsonl(node_path):
            item = deepcopy(raw)
            node_id = str(item.get("id") or "")
            if not node_id:
                raise ValueError(f"{document_id}: node id 누락: {item}")
            props = item.setdefault("properties", {})
            props["sourceDocumentIds"] = _unique(
                [*props.get("sourceDocumentIds", []), document_id]
            )
            previous = nodes_by_id.get(node_id)
            if previous is None:
                nodes_by_id[node_id] = item
                continue
            duplicate_nodes += 1
            if (previous.get("label"), previous.get("name")) != (
                item.get("label"), item.get("name")
            ):
                raise ValueError(
                    f"node id 충돌 {node_id}: "
                    f"{previous.get('label')}/{previous.get('name')} != "
                    f"{item.get('label')}/{item.get('name')}"
                )
            previous["properties"] = _merge_properties(
                previous.get("properties", {}), props, sum_counts=True
            )

        for raw in read_jsonl(edge_path):
            item = deepcopy(raw)
            edge_id = str(item.get("id") or "")
            if not edge_id:
                raise ValueError(f"{document_id}: edge id 누락: {item}")
            props = item.setdefault("properties", {})
            props["sourceDocumentIds"] = _unique(
                [*props.get("sourceDocumentIds", []), document_id]
            )
            previous = edges_by_id.get(edge_id)
            if previous is None:
                edges_by_id[edge_id] = item
                continue
            duplicate_edges += 1
            identity = ("source", "target", "type", "layer")
            if any(previous.get(key) != item.get(key) for key in identity):
                raise ValueError(f"edge id 충돌 {edge_id}")
            previous["properties"] = _merge_properties(
                previous.get("properties", {}), props, sum_counts=False
            )
            previous["evidence_ids"] = _unique(
                [
                    *previous.get("evidence_ids", []),
                    *item.get("evidence_ids", []),
                ]
            )

    nodes = sorted(nodes_by_id.values(), key=lambda item: item["id"])
    edges = sorted(edges_by_id.values(), key=lambda item: item["id"])
    validation = validate_graph(nodes, edges)
    if not validation["valid"]:
        raise RuntimeError("통합 그래프 검증 실패: " + "; ".join(validation["errors"]))

    output_dir = output_dir.expanduser().resolve()
    write_jsonl(output_dir / "neo4j/nodes.jsonl", nodes)
    write_jsonl(output_dir / "neo4j/edges.jsonl", edges)
    summary = {
        "status": "valid",
        "documents": [document_id for document_id, _ in inputs],
        "inputs": [
            {"document_id": document_id, "graph_dir": str(path.resolve())}
            for document_id, path in inputs
        ],
        "deduplicated_node_occurrences": duplicate_nodes,
        "deduplicated_edge_occurrences": duplicate_edges,
        **validation,
        "nodes_path": str(output_dir / "neo4j/nodes.jsonl"),
        "edges_path": str(output_dir / "neo4j/edges.jsonl"),
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "validation_report.json", validation)
    return summary


def parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--input은 DOCUMENT_ID=GRAPH_DIR 형식이어야 합니다.")
    document_id, raw_path = value.split("=", 1)
    if not document_id.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("--input은 DOCUMENT_ID=GRAPH_DIR 형식이어야 합니다.")
    return document_id.strip(), Path(raw_path.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description="문서별 Neo4j 그래프를 하나로 통합합니다.")
    parser.add_argument("--input", action="append", required=True, type=parse_input)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = combine_inputs(args.input, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
