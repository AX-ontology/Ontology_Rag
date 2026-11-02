from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import (
    config_path,
    load_config,
    normalize_text,
    output_path,
    read_json_records,
    unique_text,
    write_json,
    write_jsonl,
)


def _validate_tree(nodes: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    hids = [normalize_text(node.get("hid")) for node in nodes]
    counts = Counter(hids)
    duplicates = sorted(hid for hid, count in counts.items() if not hid or count > 1)
    if duplicates:
        errors.append(f"중복 또는 빈 hid: {duplicates[:20]}")

    by_hid = {normalize_text(node.get("hid")): node for node in nodes if node.get("hid")}
    roots: list[str] = []
    missing_parents: list[dict[str, str]] = []
    level_mismatches: list[dict[str, Any]] = []
    cycles: list[list[str]] = []
    isolated: list[str] = []

    for hid, node in by_hid.items():
        parent = normalize_text(node.get("parent")) or None
        try:
            level = int(node.get("level"))
        except (TypeError, ValueError):
            level = -1
            errors.append(f"{hid}: level이 정수가 아닙니다.")
        segments = hid.split(".")
        hid_depth_valid = (
            level == 1 and len(segments) == 2 and segments[-1] == "0"
        ) or (level > 1 and len(segments) == level)
        if not hid_depth_valid:
            level_mismatches.append(
                {
                    "hid": hid,
                    "level": level,
                    "expected": "root는 N.0, 그 외 segment 수와 level 일치",
                    "reason": "hid_depth",
                }
            )
        if parent is None:
            roots.append(hid)
            if level != 1:
                level_mismatches.append(
                    {"hid": hid, "level": level, "expected": 1, "reason": "root"}
                )
        elif parent not in by_hid:
            missing_parents.append({"hid": hid, "parent": parent})
        else:
            try:
                parent_level = int(by_hid[parent].get("level"))
            except (TypeError, ValueError):
                parent_level = -1
            if level != parent_level + 1:
                level_mismatches.append(
                    {
                        "hid": hid,
                        "level": level,
                        "expected": parent_level + 1,
                        "reason": "parent_level",
                    }
                )

    cycle_keys: set[tuple[str, ...]] = set()
    for start in by_hid:
        trail: list[str] = []
        positions: dict[str, int] = {}
        current: str | None = start
        while current is not None and current in by_hid:
            if current in positions:
                cycle = trail[positions[current] :] + [current]
                key = tuple(sorted(set(cycle)))
                if key not in cycle_keys:
                    cycle_keys.add(key)
                    cycles.append(cycle)
                break
            positions[current] = len(trail)
            trail.append(current)
            current = normalize_text(by_hid[current].get("parent")) or None
        if current is not None and current not in by_hid:
            isolated.append(start)

    if missing_parents:
        errors.append(f"존재하지 않는 parent: {len(missing_parents)}개")
    if level_mismatches:
        errors.append(f"level/parent 깊이 불일치: {len(level_mismatches)}개")
    if cycles:
        errors.append(f"순환 참조: {len(cycles)}개")
    if isolated:
        errors.append(f"root에 도달하지 못하는 노드: {len(set(isolated))}개")
    if not roots:
        errors.append("root 노드가 없습니다.")

    report = {
        "valid": not errors,
        "node_count": len(nodes),
        "root_count": len(roots),
        "roots": sorted(roots),
        "level_counts": dict(sorted(Counter(node.get("level") for node in nodes).items())),
        "duplicate_hids": duplicates,
        "missing_parents": missing_parents,
        "level_mismatches": level_mismatches,
        "cycles": cycles,
        "isolated_hids": sorted(set(isolated)),
        "errors": errors,
        "warnings": warnings,
    }
    return report, errors


def build_index(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = config_path(config, "pcf")
    nodes = read_json_records(source, ("pcf", "nodes", "data"))
    report, errors = _validate_tree(nodes)
    if errors:
        write_json(output_path(config, "pcf_validation.json"), report)
        raise ValueError("PCF 트리 검증 실패: " + "; ".join(errors))

    by_hid = {normalize_text(node["hid"]): node for node in nodes}
    children: dict[str, list[str]] = defaultdict(list)
    for hid, node in by_hid.items():
        parent = normalize_text(node.get("parent"))
        if parent:
            children[parent].append(hid)
    for values in children.values():
        values.sort(key=lambda hid: (int(by_hid[hid].get("level", 0)), hid))

    path_cache: dict[str, list[str]] = {}

    def path_for(hid: str) -> list[str]:
        if hid in path_cache:
            return path_cache[hid]
        parent = normalize_text(by_hid[hid].get("parent"))
        result = (path_for(parent) if parent else []) + [hid]
        path_cache[hid] = result
        return result

    enriched: list[dict[str, Any]] = []
    for hid, original in sorted(
        by_hid.items(), key=lambda item: (int(item[1].get("level", 0)), item[0])
    ):
        path_ids = path_for(hid)
        path_names_ko = [normalize_text(by_hid[item].get("nameKo")) for item in path_ids]
        path_names_en = [normalize_text(by_hid[item].get("nameEn")) for item in path_ids]
        parent = normalize_text(original.get("parent"))
        names = unique_text([original.get("nameKo"), original.get("nameEn")])
        definitions = unique_text([original.get("descKo"), original.get("descEn")])
        parent_definitions = (
            unique_text([by_hid[parent].get("descKo"), by_hid[parent].get("descEn")])
            if parent
            else []
        )
        record = dict(original)
        record.update(
            {
                "path_ids": path_ids,
                "path_names_ko": path_names_ko,
                "path_names_en": path_names_en,
                "path_text_ko": " > ".join(filter(None, path_names_ko)),
                "path_text_en": " > ".join(filter(None, path_names_en)),
                "is_leaf": not children.get(hid),
                "child_count": len(children.get(hid, [])),
                "children_hids": children.get(hid, []),
                "node_name_text": " | ".join(names),
                "node_definition_text": " ".join(definitions),
                "path_name_text": " | ".join(
                    unique_text(path_names_ko + path_names_en)
                ),
                "parent_definition_text": " ".join(parent_definitions),
                "action_object_signature": " ".join(names + definitions),
            }
        )
        enriched.append(record)

    report["output_count"] = len(enriched)
    report["leaf_count"] = sum(bool(node["is_leaf"]) for node in enriched)
    report["source"] = str(source)
    write_jsonl(output_path(config, "pcf_nodes_enriched.jsonl"), enriched)
    write_json(output_path(config, "pcf_validation.json"), report)
    return enriched, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCF 평면 트리를 검증하고 검색 인덱스를 생성합니다.")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nodes, report = build_index(load_config(args.config))
    print(f"PCF index: {len(nodes)} nodes, {report['leaf_count']} leaves")


if __name__ == "__main__":
    main()
