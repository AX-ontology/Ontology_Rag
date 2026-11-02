#!/usr/bin/env python3
"""Load the cleaned TaskFrame v4.5.1 + PCF graph into Neo4j.

The loader understands both the cleaned JSONL schema and the original v4.4 CSV
column names (id/label/name/properties_json and
id/source/target/type/layer/evidence_ids/properties_json).  It is idempotent by
stable node/edge ID and supports Neo4j Driver URIs and Aura Query API URIs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import socket
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

try:
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover
    GraphDatabase = None


HERE = Path(__file__).resolve().parent
DEFAULT_GRAPH_DIR = HERE / "outputs/neo4j"
DEFAULT_ENV_FILE = HERE.parent / "config/neo4j.env"
DEFAULT_PIPELINE_VERSION = "taskframe-v4.5.1-cleaned"
SAFE_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")

# PCF is Layer 1.  The TaskFrame graph and all of its provenance/concept nodes
# form Layer 2.  Stable labels make the split directly styleable in Neo4j
# Browser/Bloom, while the properties are useful to other graph clients.
LAYER1_NODE_LABELS = frozenset({"PCFProcess"})
LAYER_LABELS = {"layer1": "Layer1", "layer2": "Layer2"}
LAYER_NAMES = {"layer1": "Layer 1 · PCF", "layer2": "Layer 2 · TaskFrame"}
LAYER_COLORS = {"layer1": "#2563EB", "layer2": "#F97316", "cross_layer": "#7C3AED"}


def load_env_file(path: Path, *, override: bool = False) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and (override or key not in os.environ):
            os.environ[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="정제된 v4.5 TaskFrame/PCF 그래프를 Neo4j에 적재합니다.")
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--nodes", type=Path, default=None)
    parser.add_argument("--edges", type=Path, default=None)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--uri", default=None)
    parser.add_argument("--user", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--database", default=None)
    parser.add_argument("--run-id", default="jaije_v4_5_1_cleaned")
    parser.add_argument("--pipeline-version", default=DEFAULT_PIPELINE_VERSION)
    parser.add_argument(
        "--supersede-run-id",
        help="새 run에 없는 이전 run 노드/관계를 삭제하지 않고 active=false로 전환",
    )
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--skip-schema", action="store_true")
    parser.add_argument("--skip-search-indexes", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-connection", action="store_true")
    parser.add_argument("--print-queries", action="store_true")
    args = parser.parse_args()
    if args.env_file:
        load_env_file(args.env_file.expanduser().resolve(), override=True)
    args.uri = args.uri or os.getenv("NEO4J_QUERY_URI") or os.getenv("NEO4J_URI") or os.getenv("AURA_URI")
    args.user = (
        args.user
        or os.getenv("NEO4J_USERNAME")
        or os.getenv("NEO4J_USER")
        or os.getenv("AURA_USER")
    )
    args.password = args.password or os.getenv("NEO4J_PASSWORD") or os.getenv("AURA_PASS")
    args.database = args.database or os.getenv("NEO4J_DATABASE")
    return args


def safe_token(value: Any) -> str:
    token = str(value or "").strip()
    if not SAFE_TOKEN_RE.fullmatch(token):
        raise ValueError(f"unsafe Neo4j token: {value!r}")
    return token


def quote_token(value: str) -> str:
    return f"`{safe_token(value)}`"


def flatten_statement(statement: str) -> str:
    return " ".join(statement.split())


def parse_json(value: Any, default: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    text = str(value or "").strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_no}: JSON object가 필요합니다.")
            result.append(item)
    return result


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_records(path: Path) -> list[dict[str, Any]]:
    return read_csv(path) if path.suffix.casefold() == ".csv" else read_jsonl(path)


def locate_graph_files(args: argparse.Namespace) -> tuple[Path, Path]:
    graph_dir = args.graph_dir.expanduser().resolve()
    nodes_path = args.nodes.expanduser().resolve() if args.nodes else graph_dir / "nodes.jsonl"
    edges_path = args.edges.expanduser().resolve() if args.edges else graph_dir / "edges.jsonl"
    if not nodes_path.exists() and not args.nodes:
        csv_path = graph_dir / "nodes.csv"
        if csv_path.exists():
            nodes_path = csv_path
    if not edges_path.exists() and not args.edges:
        csv_path = graph_dir / "edges.csv"
        if csv_path.exists():
            edges_path = csv_path
    return nodes_path, edges_path


def neo4j_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        primitive = [item for item in value if item is not None]
        if all(isinstance(item, (str, int, float, bool)) for item in primitive):
            primitive_types = {type(item) for item in primitive}
            numeric_types = primitive_types <= {int, float}
            if len(primitive_types) <= 1 or numeric_types:
                return primitive
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return str(value)


def neo4j_properties(properties: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in properties.items():
        converted = neo4j_value(value)
        if converted is not None:
            result[str(key)] = converted
    return result


def node_graph_layer(label: str) -> str:
    return "layer1" if label in LAYER1_NODE_LABELS else "layer2"


def edge_graph_layer(source_layer: str, target_layer: str) -> str:
    if source_layer == target_layer and source_layer in LAYER_LABELS:
        return source_layer
    return "cross_layer"


def normalize_nodes(
    raw_nodes: list[dict[str, Any]],
    run_id: str,
    pipeline_version: str = DEFAULT_PIPELINE_VERSION,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in raw_nodes:
        node_id = str(row.get("id") or row.get("node_id") or "").strip()
        label = safe_token(row.get("label") or row.get("type") or "Unknown")
        if not node_id:
            raise ValueError(f"node id 누락: {row}")
        properties = parse_json(row.get("properties", row.get("properties_json")), {})
        if not isinstance(properties, dict):
            properties = {"raw_properties": properties}
        name = str(row.get("name") or row.get("canonical_name") or "").strip()
        graph_layer = node_graph_layer(label)
        merged = {
            **properties,
            "id": node_id,
            "name": name,
            "type": label,
            "graphLayer": graph_layer,
            "graphLayerName": LAYER_NAMES[graph_layer],
            "vizColor": LAYER_COLORS[graph_layer],
            "lastRunId": run_id,
            "pipelineVersion": pipeline_version,
        }
        result.append(
            {
                "id": node_id,
                "label": label,
                "layer_label": LAYER_LABELS[graph_layer],
                "properties": neo4j_properties(merged),
                "run_id": run_id,
            }
        )
    return result


def normalize_edges(
    raw_edges: list[dict[str, Any]],
    run_id: str,
    node_layers: dict[str, str] | None = None,
    pipeline_version: str = DEFAULT_PIPELINE_VERSION,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in raw_edges:
        edge_id = str(row.get("id") or row.get("edge_id") or "").strip()
        source = str(row.get("source") or row.get("source_node_id") or "").strip()
        target = str(row.get("target") or row.get("target_node_id") or "").strip()
        relation = safe_token(row.get("type") or row.get("edge_type") or "RELATED_TO")
        if not edge_id or not source or not target:
            raise ValueError(f"edge 식별자/endpoint 누락: {row}")
        properties = parse_json(row.get("properties", row.get("properties_json")), {})
        if not isinstance(properties, dict):
            properties = {"raw_properties": properties}
        evidence_ids = row.get("evidence_ids", [])
        if isinstance(evidence_ids, str):
            parsed = parse_json(evidence_ids, None)
            evidence_ids = parsed if isinstance(parsed, list) else [item for item in evidence_ids.split("|") if item]
        layers = node_layers or {}
        source_layer = layers.get(source, "layer1" if source.startswith("PCF_") else "layer2")
        target_layer = layers.get(target, "layer1" if target.startswith("PCF_") else "layer2")
        graph_layer = edge_graph_layer(source_layer, target_layer)
        merged = {
            **properties,
            "id": edge_id,
            "edgeType": relation,
            # Keep the source schema's fine-grained layer and add a separate
            # Layer 1/2 visualization grouping.
            "layer": row.get("layer") or "",
            "graphLayer": graph_layer,
            "vizColor": LAYER_COLORS[graph_layer],
            "evidenceIds": evidence_ids if isinstance(evidence_ids, list) else [],
            "lastRunId": run_id,
            "pipelineVersion": pipeline_version,
        }
        result.append(
            {
                "id": edge_id,
                "type": relation,
                "source": source,
                "target": target,
                "properties": neo4j_properties(merged),
                "run_id": run_id,
            }
        )
    return result


def validate_records(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, Any]:
    node_ids = [item["id"] for item in nodes]
    edge_ids = [item["id"] for item in edges]
    node_set = set(node_ids)
    errors: list[str] = []
    if len(node_ids) != len(node_set):
        errors.append("duplicate node id")
    if len(edge_ids) != len(set(edge_ids)):
        errors.append("duplicate edge id")
    dangling = [item["id"] for item in edges if item["source"] not in node_set or item["target"] not in node_set]
    if dangling:
        errors.append(f"dangling edge count: {len(dangling)}")
    return {
        "valid": not errors,
        "errors": errors,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes_by_label": dict(sorted(Counter(item["label"] for item in nodes).items())),
        "nodes_by_graph_layer": dict(
            sorted(Counter(item["properties"]["graphLayer"] for item in nodes).items())
        ),
        "edges_by_type": dict(sorted(Counter(item["type"] for item in edges).items())),
        "edges_by_graph_layer": dict(
            sorted(Counter(item["properties"]["graphLayer"] for item in edges).items())
        ),
        "dangling_edge_count": len(dangling),
    }


def batches(items: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class Neo4jClient:
    def run(self, statement: str, params: dict[str, Any] | None = None) -> Any:
        raise NotImplementedError

    def close(self) -> None:
        return None


class QueryApiClient(Neo4jClient):
    def __init__(self, uri: str, user: str, password: str) -> None:
        if requests is None:
            raise SystemExit("requests 패키지가 필요합니다: pip install requests")
        self.uri = uri
        self.auth = (user, password)

    def run(self, statement: str, params: dict[str, Any] | None = None) -> Any:
        response = requests.post(
            self.uri,
            auth=self.auth,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            json={"statement": flatten_statement(statement), "parameters": params or {}},
            timeout=120,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"Neo4j Query API HTTP {response.status_code}: {response.text[:1500]}")
        data = response.json()
        if data.get("errors"):
            raise RuntimeError(json.dumps(data["errors"], ensure_ascii=False))
        return data


class DriverClient(Neo4jClient):
    def __init__(self, uri: str, user: str, password: str, database: str | None) -> None:
        if GraphDatabase is None:
            raise SystemExit("neo4j 패키지가 필요합니다: pip install neo4j")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def run(self, statement: str, params: dict[str, Any] | None = None) -> Any:
        with self.driver.session(database=self.database) as session:
            return session.run(flatten_statement(statement), params or {}).data()

    def close(self) -> None:
        self.driver.close()


def validate_hostname(uri: str) -> None:
    parsed = urlparse(uri)
    if not parsed.hostname:
        raise SystemExit(f"Neo4j URI host를 확인하지 못했습니다: {uri}")
    try:
        socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 7687))
    except OSError as exc:
        raise SystemExit(f"Neo4j host DNS 확인 실패: {parsed.hostname}\n원인: {exc}") from exc


def make_client(uri: str, user: str, password: str, database: str | None) -> Neo4jClient:
    scheme = urlparse(uri).scheme
    if scheme == "https":
        return QueryApiClient(uri, user, password)
    if scheme in {"neo4j", "neo4j+s", "bolt", "bolt+s"}:
        return DriverClient(uri, user, password, database)
    raise SystemExit(f"지원하지 않는 Neo4j URI scheme: {scheme}")


SCHEMA_STATEMENTS = [
    "CREATE CONSTRAINT cleaned_kg_node_id IF NOT EXISTS FOR (n:KgNode) REQUIRE n.id IS UNIQUE",
    "CREATE INDEX cleaned_kg_last_run IF NOT EXISTS FOR (n:KgNode) ON (n.lastRunId)",
    "CREATE INDEX cleaned_taskframe_name IF NOT EXISTS FOR (n:TaskFrame) ON (n.name)",
    "CREATE INDEX cleaned_taskframe_alias_key IF NOT EXISTS FOR (n:TaskFrameAlias) ON (n.normalized_key)",
    "CREATE INDEX cleaned_entity_name IF NOT EXISTS FOR (n:CanonicalEntity) ON (n.name)",
    "CREATE INDEX cleaned_relation_name IF NOT EXISTS FOR (n:CanonicalRelation) ON (n.name)",
    "CREATE INDEX cleaned_pcf_hid IF NOT EXISTS FOR (n:PCFProcess) ON (n.hid)",
]

SEARCH_INDEX_STATEMENTS = [
    """
    CREATE FULLTEXT INDEX cleaned_concept_search IF NOT EXISTS
    FOR (n:TaskFrame|TaskFrameAlias|CanonicalEntity|CanonicalRelation|PCFProcess)
    ON EACH [n.name, n.aliases, n.name_en, n.code]
    """,
]


def apply_schema(client: Neo4jClient, include_search_indexes: bool) -> None:
    for statement in SCHEMA_STATEMENTS:
        client.run(statement)
    if include_search_indexes:
        for statement in SEARCH_INDEX_STATEMENTS:
            client.run(statement)


def load_nodes(client: Neo4jClient, nodes: list[dict[str, Any]], batch_size: int) -> int:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in nodes:
        grouped[item["label"]].append(item)
    loaded = 0
    for label, items in sorted(grouped.items()):
        layer_label = items[0]["layer_label"]
        other_layer_label = "Layer2" if layer_label == "Layer1" else "Layer1"
        query = f"""
        UNWIND $rows AS row
        MERGE (n:KgNode {{id: row.id}})
        SET n += row.properties
        SET n:{quote_token(label)}
        SET n:{quote_token(layer_label)}
        REMOVE n:{quote_token(other_layer_label)}
        SET n.runIds = CASE
          WHEN n.runIds IS NULL THEN [row.run_id]
          WHEN row.run_id IN n.runIds THEN n.runIds
          ELSE n.runIds + row.run_id END
        """
        for batch in batches(items, batch_size):
            rows = [
                {"id": item["id"], "properties": item["properties"], "run_id": item["run_id"]}
                for item in batch
            ]
            client.run(query, {"rows": rows})
            loaded += len(rows)
            print(f"[nodes:{label}] loaded={loaded}/{len(nodes)}")
    return loaded


def load_edges(client: Neo4jClient, edges: list[dict[str, Any]], batch_size: int) -> int:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in edges:
        grouped[item["type"]].append(item)
    loaded = 0
    for relation, items in sorted(grouped.items()):
        query = f"""
        UNWIND $rows AS row
        MATCH (source:KgNode {{id: row.source}})
        MATCH (target:KgNode {{id: row.target}})
        MERGE (source)-[r:{quote_token(relation)} {{id: row.id}}]->(target)
        SET r += row.properties
        SET r.runIds = CASE
          WHEN r.runIds IS NULL THEN [row.run_id]
          WHEN row.run_id IN r.runIds THEN r.runIds
          ELSE r.runIds + row.run_id END
        """
        for batch in batches(items, batch_size):
            rows = [
                {
                    "id": item["id"],
                    "source": item["source"],
                    "target": item["target"],
                    "properties": item["properties"],
                    "run_id": item["run_id"],
                }
                for item in batch
            ]
            client.run(query, {"rows": rows})
            loaded += len(rows)
            print(f"[edges:{relation}] loaded={loaded}/{len(edges)}")
    return loaded


def supersede_previous_run(client: Neo4jClient, previous_run_id: str, run_id: str) -> None:
    if not previous_run_id or previous_run_id == run_id:
        return
    node_result = client.run(
        """
        MATCH (n:KgNode)
        WHERE $previous IN coalesce(n.runIds, [])
          AND NOT $current IN coalesce(n.runIds, [])
        SET n.active = false,
            n.lifecycleStatus = 'superseded',
            n.supersededByRunId = $current
        RETURN count(n) AS supersededNodes
        """,
        {"previous": previous_run_id, "current": run_id},
    )
    edge_result = client.run(
        """
        MATCH ()-[r]->()
        WHERE $previous IN coalesce(r.runIds, [])
          AND NOT $current IN coalesce(r.runIds, [])
        SET r.active = false,
            r.lifecycleStatus = 'superseded',
            r.supersededByRunId = $current
        RETURN count(r) AS supersededRelationships
        """,
        {"previous": previous_run_id, "current": run_id},
    )
    print(f"[supersede] previous={previous_run_id} nodes={node_result} relationships={edge_result}")


def print_query_examples(run_id: str) -> None:
    print(
        f"""

[Neo4j Browser 확인 쿼리]

// 노드/관계 총수
MATCH (n:KgNode) WHERE n.lastRunId = "{run_id}" RETURN count(n) AS nodes;
MATCH ()-[r]->() WHERE r.lastRunId = "{run_id}" RETURN count(r) AS relationships;

// Layer 1(PCF) / Layer 2(TaskFrame 그래프) 적재 수
MATCH (n:KgNode) WHERE n.lastRunId = "{run_id}"
RETURN n.graphLayer AS layer, count(*) AS nodes ORDER BY layer;

// Layer 1 ↔ Layer 2 연결 시각화 (PCF=파랑, TaskFrame=주황)
MATCH path=(t:TaskFrame:Layer2)-[:MAPS_TO_PCF|ALSO_MAPS_TO_PCF|HAS_PCF_CONTEXT]->(p:PCFProcess:Layer1)
RETURN path;

// TaskFrame -> 확정 PCF
MATCH (t:TaskFrame)-[r:MAPS_TO_PCF]->(p:PCFProcess)
RETURN t.name, p.hid, p.name, r.confidence ORDER BY t.name;

// 사람 검토가 필요한 PCF 후보
MATCH (t:TaskFrame)-[r:PCF_MAPPING_CANDIDATE]->(p:PCFProcess)
RETURN t.name, p.hid, p.name, r.proposed_relationship_type, r.confidence
ORDER BY t.name;

// TaskFrame alias
MATCH (t:TaskFrame)-[:HAS_ALIAS]->(a:TaskFrameAlias)
RETURN t.name, collect(a.name) AS aliases ORDER BY t.name;

// 원문 TaskMention 및 Evidence
MATCH (m:TaskMention)-[:MEMBER_OF]->(t:TaskFrame), (m)-[:SUPPORTED_BY]->(e:EvidenceSpan)
RETURN t.name, m.name, e.name LIMIT 100;

// 격리 Fact 확인
MATCH (f:Fact {{quality_status: "quarantined"}})-[:SUPPORTED_BY]->(e:EvidenceSpan)
RETURN f.name, f.quality_issues, e.name LIMIT 100;
""".strip()
    )


def main() -> int:
    args = parse_args()
    nodes_path, edges_path = locate_graph_files(args)
    if not nodes_path.exists() or not edges_path.exists():
        raise SystemExit(f"입력 파일을 찾지 못했습니다: {nodes_path}, {edges_path}")
    raw_nodes = read_records(nodes_path)
    raw_edges = read_records(edges_path)
    nodes = normalize_nodes(raw_nodes, args.run_id, args.pipeline_version)
    node_layers = {item["id"]: item["properties"]["graphLayer"] for item in nodes}
    edges = normalize_edges(raw_edges, args.run_id, node_layers, args.pipeline_version)
    validation = validate_records(nodes, edges)
    print(json.dumps({"nodes": str(nodes_path), "edges": str(edges_path), **validation}, ensure_ascii=False, indent=2))
    if not validation["valid"]:
        return 2
    if args.validate_only or args.dry_run:
        print("[validate-only] Neo4j에는 쓰지 않았습니다.")
        if args.print_queries:
            print_query_examples(args.run_id)
        return 0
    if not args.uri or not args.user or not args.password:
        missing = []
        if not args.uri:
            missing.append("NEO4J_URI")
        if not args.user:
            missing.append("NEO4J_USERNAME (또는 NEO4J_USER)")
        if not args.password:
            missing.append("NEO4J_PASSWORD")
        raise SystemExit(
            f"Neo4j 접속 정보 누락 ({', '.join(missing)}): {args.env_file}. "
            "--validate-only로 로컬 검증만 할 수 있습니다."
        )
    validate_hostname(args.uri)
    client = make_client(args.uri, args.user, args.password, args.database)
    try:
        if args.check_connection:
            client.run("RETURN 1 AS ok")
            print("[check] Neo4j connection OK")
            return 0
        started = time.time()
        if not args.skip_schema:
            apply_schema(client, not args.skip_search_indexes)
        loaded_nodes = load_nodes(client, nodes, max(1, args.batch_size))
        loaded_edges = load_edges(client, edges, max(1, args.batch_size))
        if args.supersede_run_id:
            supersede_previous_run(client, args.supersede_run_id, args.run_id)
        print(f"[DONE] nodes={loaded_nodes}, edges={loaded_edges}, seconds={time.time() - started:.1f}")
        if args.print_queries:
            print_query_examples(args.run_id)
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
