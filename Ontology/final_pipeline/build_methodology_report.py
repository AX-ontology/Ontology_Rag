#!/usr/bin/env python3
"""Build offline methodology/result reports from validated local graph snapshots."""
import argparse
import base64
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
AX = HERE.parent
FACT_LINKS = {"HAS_CORE_FACT", "HAS_SUPPORTING_FACT", "HAS_CONTEXT_FACT"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def script_json(value):
    result = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    for char, escaped in [("<", "\\u003c"), (">", "\\u003e"), ("&", "\\u0026"),
                          ("\u2028", "\\u2028"), ("\u2029", "\\u2029")]:
        result = result.replace(char, escaped)
    return result


def snapshot(combined, title, key, load_report=None):
    combined = Path(combined).resolve()
    validation = read_json(combined / "validation_report.json")
    if validation.get("valid") is not True:
        raise ValueError(f"Unvalidated graph: {combined}")
    rows = read_jsonl(combined / "neo4j/nodes.jsonl")
    nodes = {row["id"]: row for row in rows}
    edges = read_jsonl(combined / "neo4j/edges.jsonl")
    if len(nodes) != len(rows) or len({e["id"] for e in edges}) != len(edges):
        raise ValueError("Duplicate graph IDs")
    if any(e["source"] not in nodes or e["target"] not in nodes for e in edges):
        raise ValueError("Unresolved graph endpoint")
    labels = Counter(n["label"] for n in rows)
    types = Counter(e["type"] for e in edges)
    if (len(rows) != validation["node_count"] or len(edges) != validation["edge_count"]
            or dict(labels) != validation["nodes_by_label"]
            or dict(types) != validation["edges_by_type"]):
        raise ValueError("Graph differs from validation report")
    documents = sorted([{"id": n["id"], "name": n["name"]}
                        for n in rows if n["label"] == "Document"], key=lambda n: n["id"])
    members, facts, evidence, mappings = (defaultdict(list) for _ in range(4))
    for edge in edges:
        if edge["type"] == "MEMBER_OF":
            members[edge["target"]].append(edge["source"])
        elif edge["type"] in FACT_LINKS:
            facts[edge["source"]].append((edge["target"], edge["type"]))
        elif edge["type"] == "SUPPORTED_BY":
            evidence[edge["source"]].append(edge["target"])
        elif (nodes[edge["source"]]["label"] == "TaskFrame"
              and nodes[edge["target"]]["label"] == "PCFProcess"):
            target = nodes[edge["target"]]
            props = edge.get("properties", {})
            mappings[edge["source"]].append({
                "id": target["id"], "name": target["name"],
                "code": target["properties"].get("hid", ""), "type": edge["type"],
                **{k: props.get(k) for k in ["confidence", "coverage_score", "rationale",
                    "validation_status", "review_status", "active", "mapping_role", "is_primary"]}})
    frames = []
    for frame in read_json(combined / "taskframes.json"):
        fid = frame["taskframe_id"]
        if fid not in nodes or set(members[fid]) != set(frame["member_task_mentions"]):
            raise ValueError(f"Membership mismatch: {fid}")
        by_doc = Counter()
        for mid in members[fid]:
            by_doc.update(nodes[mid]["properties"].get("sourceDocumentIds", []))
        fact_ids = sorted({item[0] for item in facts[fid]})
        # Stratify a bounded evidence sample across documents; never imply exhaustiveness.
        groups = defaultdict(list)
        for fact_id in fact_ids:
            props = nodes[fact_id]["properties"]
            groups[tuple(props.get("sourceDocumentIds", []))].append(fact_id)
        chosen = []
        for offset in range(4):
            for doc in sorted(groups):
                if offset < len(groups[doc]):
                    chosen.append(groups[doc][offset])
        samples = []
        for fact_id in chosen:
            fact = nodes[fact_id]
            props = fact["properties"]
            spans = [nodes[eid] for eid in evidence[fact_id] if nodes[eid]["label"] == "EvidenceSpan"]
            samples.append({"id": fact_id, "subject": props.get("subject_raw", ""),
                "predicate": props.get("predicate_raw", ""), "object": props.get("object_raw", ""),
                "quote": props.get("evidence_quote", ""), "documents": props.get("sourceDocumentIds", []),
                "quality": props.get("quality_status", "unknown"),
                "issues": props.get("quality_issues", []),
                "links": sorted({kind for tid, kind in facts[fid] if tid == fact_id}),
                "evidence": [{"id": s["id"], "quote": s["name"],
                    **{k: s["properties"].get(k) for k in ["document_id", "page", "section_path"]}}
                    for s in spans[:3]]})
        frames.append({"id": fid, "name": frame["canonical_name"], "definition": frame.get("definition", ""),
            "mentions": len(members[fid]), "byDoc": dict(by_doc), "facts": len(fact_ids),
            "quarantined": sum(nodes[i]["properties"].get("quality_status") == "quarantined" for i in fact_ids),
            "pcf": sorted(mappings[fid], key=lambda p: (not p["is_primary"], -(p["confidence"] or 0))),
            "samples": samples})
    if len(frames) != labels["TaskFrame"] or sum(f["mentions"] for f in frames) != labels["TaskMention"]:
        raise ValueError("TaskFrame/mention totals do not reconcile")
    loads = read_json(load_report) if load_report and Path(load_report).exists() else None
    fingerprints = {}
    for filename in ["taskframes.json", "validation_report.json", "neo4j/nodes.jsonl", "neo4j/edges.jsonl"]:
        fingerprints[filename] = hashlib.sha256((combined / filename).read_bytes()).hexdigest()
    return {"key": key, "title": title, "source": str(combined), "documents": documents,
        "frames": sorted(frames, key=lambda f: -f["mentions"]), "validation": validation,
        "nodeCount": len(rows), "edgeCount": len(edges), "labels": dict(labels), "types": dict(types),
        "quarantined": sum(n["label"] == "Fact" and n["properties"].get("quality_status") == "quarantined" for n in rows),
        "load": loads, "fingerprints": fingerprints}


def build(latest, previous, output):
    latest, previous, output = Path(latest), Path(previous), Path(output)
    paths = read_json(latest / "current.json")
    # Portable packaged snapshots resolve pointers relative to their manifest.
    paths = {k: str(Path(v) if Path(v).is_absolute() else latest / v) for k, v in paths.items()}
    data = {"generatedAt": datetime.now(timezone.utc).isoformat(), "snapshots": [
        snapshot(paths["combined"], "최신 · 전체 근거 통합 PCF", "latest"),
        snapshot(previous / "combined", "이전 · DB 적재 검증본", "previous",
                 previous / "combined/neo4j_load_report.json")],
        "quality": read_json(Path(paths["cleaned"]) / "summary.json")["quality"],
        "alignment": [read_json(AX / "final_outputs/v4_5_final/jaije_2/flat_consolidated/validation_report.json"),
                      read_json(previous / "jaije_3/flat_consolidated/validation_report.json")]}
    template = (HERE / "report_templates/report.html").read_text(encoding="utf-8")
    font = base64.b64encode((HERE / "report_templates/assets/NotoSansKR-report.woff2").read_bytes()).decode("ascii")
    license_text = (HERE / "report_templates/assets/OFL.txt").read_text(encoding="utf-8")
    template = template.replace("__FONT_DATA__", font).replace("__FONT_LICENSE__", license_text)
    output.mkdir(parents=True, exist_ok=True)
    for page in ["methodology", "results"]:
        rendered = template.replace("__PAGE__", page).replace("__REPORT_DATA__", script_json(data))
        (output / f"{page}.html").write_text(rendered, encoding="utf-8")
    (output / "report_manifest.json").write_text(json.dumps({
        "generated_at": data["generatedAt"], "snapshots": [
            {k: s[k] for k in ["key", "source", "fingerprints", "nodeCount", "edgeCount"]}
            for s in data["snapshots"]]}, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latest", type=Path, default=AX / "final_outputs/catalog_refactor_check")
    parser.add_argument("--previous", type=Path, default=AX / "final_outputs/v4_5_flat_final")
    parser.add_argument("--output", type=Path, default=AX / "final_outputs/methodology_report")
    args = parser.parse_args()
    print(build(args.latest, args.previous, args.output))
