#!/usr/bin/env python3
"""Clean TaskFrame v4.5 + PCF crosswalk data and build one Neo4j graph package.

The source pipeline outputs are immutable inputs.  This script writes a cleaned,
auditable projection under ``cleaning/outputs`` and keeps rejected canonical
concepts in quarantine files instead of discarding their raw mentions/facts.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import unicodedata
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
AX_ROOT = HERE.parent
DEFAULT_TASKKG = AX_ROOT / "taskframe_v4/outputs/zaije_1_v4_5_structured_refine/09_kg"
DEFAULT_EXTRACTION = AX_ROOT / "taskframe_v4/outputs/zaije_1_v4_5_structured_refine/02_extraction"
DEFAULT_ENTITY_PREP = AX_ROOT / "taskframe_v4/outputs/zaije_1_v4_5_structured_refine/03_entity_prep"
DEFAULT_PCF = AX_ROOT / "layer_connect/outputs"

SAFE_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
BR_RE = re.compile(r"<\s*br\s*/?\s*>", re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")
LEADING_LIST_RE = re.compile(
    r"^\s*(?:[•·▪◦●○■□◆◇▶▷►▸*-]+\s*|"
    r"\(?\d{1,3}\)?(?:[.)번]\s*|\s+))"
)
SPACE_RE = re.compile(r"\s+")
SENTENCE_END_RE = re.compile(
    r"(?:해야\s*한다|하여야\s*한다|하도록\s*한다|할\s*수\s*있다|"
    r"한다|하였다|했다|된다|되었다|이다|있다|없다|하라|해라|하시오|"
    r"해야\s*한다|명심하라|보여\s*준다|보여준다|바랍니다|한다[.]?)\s*$"
)
NUMBER_ONLY_RE = re.compile(r"[\d\s.,%+-]+\Z")
SYMBOL_ONLY_RE = re.compile(r"[^0-9A-Za-z가-힣]+\Z")
HTML_TAG_RE = re.compile(r"<[^>]+>", re.IGNORECASE)
STRUCTURED_LIST_MARKER_RE = re.compile(
    r"(?:^|<\s*br\s*/?\s*>|\n)\s*[•·▪◦●○■□◆◇▶▷►▸*-]+",
    re.IGNORECASE,
)

ENTITY_STOP_NAMES = {
    "나", "너", "우리", "저희", "이것", "그것", "저것", "이들", "그들",
    "그", "그는", "그녀", "필자", "독자", "여기", "거기", "이하", "상",
    "것", "때", "수", "등", "및", "또는", "이", "다음", "사람들",
}
PLACEHOLDER_RE = re.compile(r"^\(\s*[○●◯xX]?\s*\)$")
ENTITY_PARTICLE_EXCEPTIONS = {"전문가", "평가", "원가", "국가", "대가", "효과", "추가", "결과"}
ENTITY_PARTICLES = (
    "으로", "에서", "에게", "까지", "부터", "처럼", "보다", "이란",
    "은", "는", "이", "가", "을", "를", "로", "에", "들",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TaskFrame v4.5와 PCF 연결 결과를 정제하여 Neo4j 입력물을 생성합니다."
    )
    parser.add_argument("--taskkg-dir", type=Path, default=DEFAULT_TASKKG)
    parser.add_argument("--extraction-dir", type=Path, default=DEFAULT_EXTRACTION)
    parser.add_argument("--entity-prep-dir", type=Path, default=DEFAULT_ENTITY_PREP)
    parser.add_argument("--pcf-dir", type=Path, default=DEFAULT_PCF)
    parser.add_argument("--entity-aliases", type=Path, default=HERE / "config/entity_aliases.json")
    parser.add_argument("--relation-aliases", type=Path, default=HERE / "config/relation_aliases.json")
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs")
    parser.add_argument("--entity-min-support", type=int, default=2)
    parser.add_argument("--entity-max-name-length", type=int, default=40)
    parser.add_argument("--ontology-min-support", type=int, default=3)
    parser.add_argument("--ontology-min-sections", type=int, default=2)
    parser.add_argument(
        "--accept-needs-review-pcf",
        action="store_true",
        help=(
            "사람 검토 전 PCF 후보도 확정 관계로 승격합니다. 기본값은 후보 관계로 "
            "보존하는 안전 모드입니다."
        ),
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


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


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def clean_text(value: Any, *, strip_list_marker: bool = False) -> str:
    if value is None:
        return ""
    result = unicodedata.normalize("NFKC", html.unescape(str(value)))
    result = BR_RE.sub(" ", result)
    result = TAG_RE.sub(" ", result)
    if strip_list_marker:
        result = LEADING_LIST_RE.sub("", result)
    return SPACE_RE.sub(" ", result).strip(" \t\r\n\"'“”‘’")


def sanitize_display_names(nodes: dict[str, dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for item in nodes.values():
        raw_name = str(item.get("name") or "")
        display_name = clean_text(raw_name)
        props = item.setdefault("properties", {})
        props["displayName"] = display_name
        if raw_name != display_name:
            props["rawName"] = raw_name
            item["name"] = display_name
            counts[item.get("label", "Unknown")] += 1
    return dict(sorted(counts.items()))


def fact_surface_quality_reasons(
    triple: dict[str, Any], evidence_type: str = ""
) -> list[str]:
    reasons: list[str] = []
    predicate = str(triple.get("predicate_raw") or "").strip()
    subject = str(triple.get("subject_raw") or "")
    obj = str(triple.get("object_raw") or "")
    if not predicate:
        reasons.append("empty_predicate")
    elif SYMBOL_ONLY_RE.fullmatch(predicate):
        reasons.append("symbol_only_predicate")
    if "|" in predicate:
        reasons.append("table_delimiter_predicate")
    if HTML_TAG_RE.search(predicate):
        reasons.append("html_predicate")
    if evidence_type in {"table", "figure"}:
        for field, value in (("subject", subject), ("object", obj)):
            if "|" in value:
                reasons.append(f"multi_cell_{field}")
            if len(STRUCTURED_LIST_MARKER_RE.findall(value)) > 1:
                reasons.append(f"multi_item_{field}")
    checks = triple.get("quality_checks") or {}
    if isinstance(checks, dict):
        reasons.extend(str(item) for item in checks.get("reasons", []) if item)
    return sorted(set(reasons))


def lexical_key(value: Any) -> str:
    result = clean_text(value, strip_list_marker=True)
    result = re.sub(r"\s*/\s*", "/", result)
    result = re.sub(r"\s*[-–—]\s*", "-", result)
    return result.strip(" .,:;!?·•").casefold()


def normalize_entity_name(value: Any) -> str:
    result = clean_text(value, strip_list_marker=True).strip(" .,:;!?·•")
    if result in ENTITY_PARTICLE_EXCEPTIONS:
        return result
    for particle in ENTITY_PARTICLES:
        if result.endswith(particle):
            stem = result[: -len(particle)].rstrip()
            if stem and stem not in ENTITY_PARTICLE_EXCEPTIONS:
                return stem
    return result


def stable_id(prefix: str, *parts: Any, length: int = 16) -> str:
    raw = "\x1f".join(clean_text(part) for part in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def unique_text(values: Iterable[Any], *, cleaned: bool = True) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = clean_text(value, strip_list_marker=True) if cleaned else str(value).strip()
        key = lexical_key(item)
        if item and key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def load_alias_config(path: Path) -> tuple[dict[str, str], set[str]]:
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"{path}: alias object 목록이 필요합니다.")
    aliases: dict[str, str] = {}
    canonical_values: set[str] = set()
    for item in payload:
        if not isinstance(item, dict):
            continue
        canonical = clean_text(item.get("canonical"), strip_list_marker=True)
        if not canonical:
            continue
        canonical_values.add(canonical)
        forms = item.get("aliases") if isinstance(item.get("aliases"), list) else []
        for form in [canonical, *forms]:
            key = lexical_key(form)
            if key:
                previous = aliases.get(key)
                if previous and previous != canonical:
                    raise ValueError(f"alias 충돌: {form!r} -> {previous!r}/{canonical!r}")
                aliases[key] = canonical
    return aliases, canonical_values


def load_registry(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


class ConceptRegistry:
    """Small persistent registry that reuses IDs when a preferred name changes."""

    def __init__(self, path: Path, prefix: str) -> None:
        self.path = path
        self.prefix = prefix
        self.records = load_registry(path)
        self.by_id = {item["concept_id"]: item for item in self.records}
        self.by_key: dict[str, set[str]] = defaultdict(set)
        for item in self.records:
            for key in item.get("identity_keys", []):
                if key:
                    self.by_key[str(key)].add(item["concept_id"])

    def resolve(
        self,
        preferred_name: str,
        identity_keys: Iterable[str],
        preferred_source_id: str | None,
    ) -> str:
        keys = {key for key in identity_keys if key}
        candidates = {cid for key in keys for cid in self.by_key.get(key, set())}
        if len(candidates) == 1:
            concept_id = next(iter(candidates))
        elif preferred_source_id:
            concept_id = preferred_source_id
        else:
            concept_id = stable_id(self.prefix, preferred_name)
        record = self.by_id.get(concept_id)
        if record is None:
            record = {
                "concept_id": concept_id,
                "preferred_name": preferred_name,
                "identity_keys": sorted(keys),
            }
            self.records.append(record)
            self.by_id[concept_id] = record
        else:
            record["preferred_name"] = preferred_name
            record["identity_keys"] = sorted(set(record.get("identity_keys", [])) | keys)
        for key in record["identity_keys"]:
            self.by_key[key].add(concept_id)
        return concept_id

    def save(self) -> None:
        write_jsonl(self.path, sorted(self.records, key=lambda item: item["concept_id"]))


def node_record(node_id: str, label: str, name: str, properties: dict[str, Any]) -> dict[str, Any]:
    return {"id": node_id, "label": label, "name": name, "properties": properties}


def edge_record(
    source: str,
    target: str,
    relation: str,
    layer: str,
    *,
    evidence_ids: Iterable[str] = (),
    properties: dict[str, Any] | None = None,
    identity: str = "",
) -> dict[str, Any]:
    return {
        "id": stable_id("EDGE", source, relation, target, layer, identity),
        "source": source,
        "target": target,
        "type": relation,
        "layer": layer,
        "evidence_ids": unique_text(evidence_ids, cleaned=False),
        "properties": properties or {},
    }


def is_alias_candidate(raw: str, preferred_name: str) -> tuple[bool, list[str], str]:
    reasons: list[str] = []
    cleaned = clean_text(raw, strip_list_marker=True).strip(" .")
    if not cleaned:
        reasons.append("empty")
    if lexical_key(cleaned) == lexical_key(preferred_name):
        reasons.append("same_as_preferred_name")
    if BR_RE.search(str(raw)) or TAG_RE.search(str(raw)):
        reasons.append("html_or_structured_text")
    if LEADING_LIST_RE.match(str(raw)):
        reasons.append("list_item")
    if len(cleaned) > 40:
        reasons.append("too_long")
    if len(cleaned) < 2:
        reasons.append("too_short")
    if re.match(r"^\d", cleaned):
        reasons.append("numbered_instruction")
    if SENTENCE_END_RE.search(cleaned):
        reasons.append("sentence_like")
    if cleaned.endswith((".", "?")):
        reasons.append("sentence_punctuation")
    return not reasons, reasons, cleaned


def classify_entity(
    raw_names: list[str],
    preferred_name: str,
    support_count: int,
    *,
    explicit_alias: bool,
    min_support: int,
    max_length: int,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    key = lexical_key(preferred_name)
    if not key:
        reasons.append("empty")
    if any(BR_RE.search(name) or "\n" in name for name in raw_names):
        reasons.append("structured_cell")
    if any(LEADING_LIST_RE.match(name) for name in raw_names):
        reasons.append("list_item")
    if PLACEHOLDER_RE.fullmatch(preferred_name):
        reasons.append("placeholder")
    if NUMBER_ONLY_RE.fullmatch(preferred_name):
        reasons.append("number_only")
    if SYMBOL_ONLY_RE.fullmatch(preferred_name):
        reasons.append("symbol_only")
    if key in {lexical_key(item) for item in ENTITY_STOP_NAMES}:
        reasons.append("pronoun_or_stop_entity")
    if len(preferred_name) < 2:
        reasons.append("too_short")
    if re.fullmatch(
        r"(?:[A-Z](?:\s*/\s*[A-Z])?|[A-Z]\s*(?:사|업체|회사|기업))",
        preferred_name,
    ):
        reasons.append("anonymized_placeholder")
    if len(preferred_name) > max_length:
        reasons.append("clause_like_long_name")
    if support_count < min_support and not explicit_alias:
        reasons.append("singleton_unverified")
    return not reasons, reasons


def normalize_relation_action(value: Any) -> str:
    text = clean_text(value, strip_list_marker=True).strip(" .,:;")
    text = re.sub(r"\s+", " ", text)
    patterns = (
        (r"(.+?)(?:하여야 한다|해야 한다|하도록 한다|하였다|했다|한다|하라|해라|하여|하고|함)$", "하다"),
        (r"(.+?)(?:되었다|됐다|된다|되고|됨)$", "되다"),
    )
    for pattern, suffix in patterns:
        match = re.fullmatch(pattern, text)
        if match:
            stem = match.group(1).strip()
            if stem:
                return f"{stem}{suffix}"
    return text


def canonicalize_relation(
    raw: str, alias_map: dict[str, str], controlled: set[str]
) -> tuple[str | None, str]:
    original = clean_text(raw, strip_list_marker=True)
    if not original:
        return None, "empty"
    if BR_RE.search(raw) or TAG_RE.search(raw):
        return None, "structured_or_html"
    if NUMBER_ONLY_RE.fullmatch(original) or SYMBOL_ONLY_RE.fullmatch(original):
        return None, "number_or_symbol"
    direct = alias_map.get(lexical_key(original))
    if direct:
        return direct, "verified_alias"
    normalized = normalize_relation_action(original)
    mapped = alias_map.get(lexical_key(normalized), normalized)
    if mapped in controlled:
        return mapped, "controlled_morphology"
    return None, "uncontrolled_relation"


def taskframe_alias_projection(
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    member_names: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        if edge.get("type") == "MEMBER_OF":
            mention = nodes.get(edge.get("source", ""))
            if mention and mention.get("label") == "TaskMention":
                member_names[edge["target"]].append(mention.get("name", ""))

    alias_to_frames: dict[str, set[str]] = defaultdict(set)
    alias_display: dict[str, str] = {}
    decisions: list[dict[str, Any]] = []
    per_frame_alias_keys: dict[str, set[str]] = defaultdict(set)
    counts = Counter()

    for frame in sorted(
        (item for item in nodes.values() if item.get("label") == "TaskFrame"),
        key=lambda item: item["id"],
    ):
        properties = frame.setdefault("properties", {})
        raw_aliases = properties.pop("aliases", [])
        raw_aliases = raw_aliases if isinstance(raw_aliases, list) else []
        mention_keys = {lexical_key(item) for item in member_names.get(frame["id"], [])}
        for raw_alias in raw_aliases:
            key = lexical_key(raw_alias)
            decision = {
                "taskframe_id": frame["id"],
                "taskframe_name": frame["name"],
                "raw_value": raw_alias,
                "normalized_key": key,
            }
            if key in mention_keys:
                decision.update(status="raw_task_mention", reasons=["member_task_mention"])
                counts["raw_task_mention"] += 1
            else:
                accepted, reasons, cleaned = is_alias_candidate(raw_alias, frame["name"])
                if accepted:
                    decision.update(status="accepted_alias", reasons=[], cleaned_value=cleaned)
                    per_frame_alias_keys[frame["id"]].add(key)
                    alias_to_frames[key].add(frame["id"])
                    alias_display.setdefault(key, cleaned)
                    counts["accepted_alias"] += 1
                else:
                    decision.update(status="rejected_raw_like", reasons=reasons, cleaned_value=cleaned)
                    counts["rejected_raw_like"] += 1
            decisions.append(decision)
        properties["alias_count"] = len(per_frame_alias_keys[frame["id"]])
        properties["raw_task_mention_count"] = len(member_names.get(frame["id"], []))
        properties["alias_model"] = "TaskFrame-HAS_ALIAS-TaskFrameAlias"

    alias_nodes: list[dict[str, Any]] = []
    alias_edges: list[dict[str, Any]] = []
    for key in sorted(alias_to_frames):
        value = alias_display[key]
        alias_id = stable_id("TFA", key)
        latin_upper = re.sub(r"[^A-Za-z]", "", value)
        alias_type = "abbreviation" if latin_upper and latin_upper.isupper() else "synonym"
        alias_nodes.append(
            node_record(
                alias_id,
                "TaskFrameAlias",
                value,
                {
                    "normalized_key": key,
                    "alias_type": alias_type,
                    "language": "ko" if re.search(r"[가-힣]", value) else "en",
                    "validation_status": "accepted",
                    "ambiguous": len(alias_to_frames[key]) > 1,
                    "target_count": len(alias_to_frames[key]),
                },
            )
        )
        for frame_id in sorted(alias_to_frames[key]):
            alias_edges.append(
                edge_record(
                    frame_id,
                    alias_id,
                    "HAS_ALIAS",
                    "taskframe_alias",
                    properties={"validation_status": "accepted"},
                )
            )
    return alias_nodes, alias_edges, decisions, dict(counts)


def decorate_quality(
    nodes: dict[str, dict[str, Any]],
    entity_prep_dir: Path,
    extraction_dir: Path,
) -> tuple[set[str], set[str], list[dict[str, Any]], dict[str, Any]]:
    triples = {item["triple_id"]: item for item in read_jsonl(entity_prep_dir / "open_triples.jsonl")}
    mentions = {item["task_mention_id"]: item for item in read_jsonl(entity_prep_dir / "task_mentions.jsonl")}
    extraction_summary = read_json(extraction_dir / "summary.json")
    chunk_rows = read_jsonl(extraction_dir / "chunk_extractions.jsonl")
    missing_audits = {
        error.get("evidence_id")
        for chunk in chunk_rows
        for error in chunk.get("schema_errors", [])
        if error.get("kind") == "SentenceAudit" and error.get("reason") == "missing"
    }
    missing_audits.discard(None)
    evidence_types = {
        node_id: item.get("properties", {}).get("evidence_type", "")
        for node_id, item in nodes.items()
        if item.get("label") == "EvidenceSpan"
    }

    quarantined_facts: set[str] = set()
    review_mentions: set[str] = set()
    issues: list[dict[str, Any]] = []
    for triple_id, triple in triples.items():
        reasons: list[str] = []
        if not triple.get("surface_provenance_valid", False):
            reasons.append("non_extractive_surface")
        grounding = triple.get("evidence_grounding") or {}
        if not grounding.get("exact_quote", False):
            reasons.append("invalid_exact_quote")
        if triple.get("canonicalization_status") == "review_required":
            reasons.append("canonicalization_review_required")
        reasons.extend(
            fact_surface_quality_reasons(
                triple, evidence_types.get(triple.get("evidence_id", ""), "")
            )
        )
        node = nodes.get(triple_id)
        if node:
            props = node.setdefault("properties", {})
            props["quality_status"] = "quarantined" if reasons else "accepted"
            props["quality_issues"] = sorted(set(reasons))
            props["active"] = not reasons
        if reasons:
            quarantined_facts.add(triple_id)
            issues.append({"record_type": "Fact", "record_id": triple_id, "reasons": sorted(set(reasons))})

    for mention_id, mention in mentions.items():
        grounding = mention.get("surface_grounding") or {}
        false_fields = sorted(key for key, valid in grounding.items() if valid is False)
        node = nodes.get(mention_id)
        if node:
            props = node.setdefault("properties", {})
            props["quality_status"] = "review_required" if false_fields else "accepted"
            props["quality_issues"] = [f"non_extractive_{field}" for field in false_fields]
            props["canonicalization_eligible"] = not false_fields
        if false_fields:
            review_mentions.add(mention_id)
            issues.append(
                {
                    "record_type": "TaskMention",
                    "record_id": mention_id,
                    "reasons": [f"non_extractive_{field}" for field in false_fields],
                }
            )

    for evidence_id in sorted(missing_audits):
        node = nodes.get(evidence_id)
        if node:
            node.setdefault("properties", {})["sentence_audit_status"] = "missing"
        issues.append(
            {"record_type": "EvidenceSpan", "record_id": evidence_id, "reasons": ["sentence_audit_missing"]}
        )

    report = {
        "source_schema_error_count": extraction_summary.get("schema_error_count", 0),
        "quarantined_fact_count": len(quarantined_facts),
        "review_task_mention_count": len(review_mentions),
        "missing_sentence_audit_count": len(missing_audits),
        "fact_quality_reason_counts": dict(sorted(Counter(
            reason
            for issue in issues if issue.get("record_type") == "Fact"
            for reason in issue.get("reasons", [])
        ).items())),
    }
    return quarantined_facts, review_mentions, issues, report


def clean_entities(
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    alias_map: dict[str, str],
    canonical_alias_values: set[str],
    registry: ConceptRegistry,
    *,
    min_support: int,
    max_length: int,
    quarantined_facts: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source_entities = {
        node_id: item for node_id, item in nodes.items() if item.get("label") == "CanonicalEntity"
    }
    entity_mentions = {
        node_id for node_id, item in nodes.items() if item.get("label") == "EntityMention"
    }
    mention_parent_fact: dict[str, str] = {}
    for edge in edges:
        if edge.get("type") in {"HAS_SUBJECT", "HAS_OBJECT"}:
            mention_parent_fact[edge["target"]] = edge["source"]
    old_links = [
        edge for edge in edges
        if edge.get("type") == "CANONICALIZED_AS"
        and edge.get("source") in entity_mentions
        and edge.get("target") in source_entities
    ]
    for edge in old_links:
        if mention_parent_fact.get(edge["source"]) in quarantined_facts:
            mention = nodes.get(edge["source"])
            if mention:
                props = mention.setdefault("properties", {})
                props["canonicalization_status"] = "parent_fact_quarantined"
                props["canonicalization_issues"] = ["parent_fact_quarantined"]
    incoming: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in old_links:
        if mention_parent_fact.get(edge["source"]) not in quarantined_facts:
            incoming[edge["target"]].append(edge)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source_id, item in source_entities.items():
        cleaned_name = normalize_entity_name(item.get("name"))
        preferred = alias_map.get(lexical_key(cleaned_name), cleaned_name)
        groups[lexical_key(preferred)].append(
            {
                "source_id": source_id,
                "source": item,
                "preferred": preferred,
                "incoming": incoming.get(source_id, []),
            }
        )

    accepted_nodes: list[dict[str, Any]] = []
    accepted_edges: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    merged_group_count = 0

    for key, members in sorted(groups.items()):
        preferred = members[0]["preferred"]
        raw_names = [member["source"].get("name", "") for member in members]
        source_ids = [member["source_id"] for member in members]
        source_forms = [
            form
            for member in members
            for form in member["source"].get("properties", {}).get("surface_forms", [])
        ]
        group_edges = [edge for member in members for edge in member["incoming"]]
        support_count = len({edge["source"] for edge in group_edges})
        alias_keys = {lexical_key(value) for value in [preferred, *raw_names, *source_forms]}
        explicit_alias = (
            preferred in canonical_alias_values
            or any(alias_key in alias_map for alias_key in alias_keys)
        )
        accepted, reasons = classify_entity(
            raw_names,
            preferred,
            support_count,
            explicit_alias=explicit_alias,
            min_support=min_support,
            max_length=max_length,
        )
        if not accepted:
            quarantine.append(
                {
                    "source_canonical_ids": source_ids,
                    "preferred_name": preferred,
                    "raw_names": raw_names,
                    "support_count": support_count,
                    "surface_forms": unique_text(source_forms),
                    "status": "quarantined",
                    "reasons": reasons,
                }
            )
            for edge in group_edges:
                mention = nodes.get(edge["source"])
                if mention:
                    props = mention.setdefault("properties", {})
                    props["canonicalization_status"] = "quarantined"
                    props["rejected_canonical_ids"] = source_ids
                    props["canonicalization_issues"] = reasons
            continue

        preferred_source_id = next(
            (
                member["source_id"]
                for member in members
                if lexical_key(member["source"].get("name")) == lexical_key(preferred)
            ),
            sorted(source_ids)[0] if source_ids else None,
        )
        concept_id = registry.resolve(preferred, alias_keys, preferred_source_id)
        if len(members) > 1:
            merged_group_count += 1
        aliases = [
            value for value in unique_text([*raw_names, *source_forms])
            if lexical_key(value) != lexical_key(preferred) and len(value) <= max_length
        ]
        accepted_nodes.append(
            node_record(
                concept_id,
                "CanonicalEntity",
                preferred,
                {
                    "aliases": aliases,
                    "surface_forms": unique_text(source_forms),
                    "support_count": support_count,
                    "mapping_status": "accepted",
                    "validation_status": "accepted",
                    "source_canonical_ids": sorted(source_ids),
                    "explicit_alias": explicit_alias,
                },
            )
        )
        for edge in group_edges:
            accepted_edges.append(
                edge_record(
                    edge["source"],
                    concept_id,
                    "CANONICALIZED_AS",
                    "canonicalization",
                    properties={
                        "validation_status": "accepted",
                        "source_edge_id": edge.get("id", ""),
                    },
                )
            )
            mention = nodes.get(edge["source"])
            if mention:
                mention.setdefault("properties", {})["canonicalization_status"] = "accepted"

    for mention_id in entity_mentions:
        mention = nodes[mention_id]
        props = mention.setdefault("properties", {})
        props.setdefault("canonicalization_status", "unresolved")
        if props["canonicalization_status"] == "accepted":
            props.setdefault("canonicalization_issues", [])
        else:
            props.setdefault("canonicalization_issues", ["no_accepted_canonical_entity"])

    # Different normalized-name groups may resolve to the same persistent
    # concept ID via overlapping aliases. Export that concept only once,
    # without dropping any mention links or original names/source IDs.
    grouped_nodes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in accepted_nodes:
        grouped_nodes[item["id"]].append(item)
    registry_coalesced = len(accepted_nodes) - len(grouped_nodes)
    edges_by_id: dict[str, dict[str, Any]] = {}
    for edge in accepted_edges:
        previous = edges_by_id.get(edge["id"])
        if previous is None:
            edges_by_id[edge["id"]] = edge
        else:
            props = previous["properties"]
            props["source_edge_ids"] = sorted(set(props.get("source_edge_ids", [props.get("source_edge_id", "")])) |
                                              {edge["properties"].get("source_edge_id", "")})
    accepted_edges = list(edges_by_id.values())
    support_by_concept: dict[str, set[str]] = defaultdict(set)
    for edge in accepted_edges:
        support_by_concept[edge["target"]].add(edge["source"])
    accepted_nodes = []
    for concept_id, items in sorted(grouped_nodes.items()):
        result = deepcopy(items[0])
        if len(items) > 1:
            result["name"] = registry.by_id[concept_id]["preferred_name"]
            properties = result["properties"]
            for field in ("aliases", "surface_forms", "source_canonical_ids"):
                properties[field] = sorted({v for item in items for v in item["properties"].get(field, [])})
            properties["aliases"] = sorted({*properties["aliases"], *(item["name"] for item in items)} - {result["name"]})
            properties["explicit_alias"] = any(item["properties"]["explicit_alias"] for item in items)
        result["properties"]["support_count"] = len(support_by_concept[concept_id])
        accepted_nodes.append(result)

    report = {
        "source_canonical_entity_count": len(source_entities),
        "accepted_canonical_entity_count": len(accepted_nodes),
        "quarantined_canonical_entity_group_count": len(quarantine),
        "accepted_entity_mapping_edge_count": len(accepted_edges),
        "merged_entity_group_count": merged_group_count,
        "registry_coalesced_group_count": registry_coalesced,
        "quarantine_reason_counts": dict(
            Counter(reason for item in quarantine for reason in item["reasons"])
        ),
        "entity_mention_status_counts": dict(sorted(
            Counter(
                nodes[mention_id].get("properties", {}).get("canonicalization_status", "")
                for mention_id in entity_mentions
            ).items()
        )),
    }
    return accepted_nodes, accepted_edges, quarantine, report


def clean_relations(
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    alias_map: dict[str, str],
    controlled: set[str],
    registry: ConceptRegistry,
    *,
    ontology_min_support: int,
    ontology_min_sections: int,
    quarantined_facts: set[str],
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]],
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]
]:
    relation_expressions = {
        node_id: item for node_id, item in nodes.items() if item.get("label") == "RelationExpression"
    }
    source_relations = {
        node_id: item for node_id, item in nodes.items() if item.get("label") == "CanonicalTerm"
    }
    fact_by_expression = {
        edge["target"]: edge["source"]
        for edge in edges if edge.get("type") == "HAS_PREDICATE"
    }
    fact_ids_in_graph = {
        node_id for node_id, item in nodes.items() if item.get("label") == "Fact"
    }
    evidence_by_fact: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        if edge.get("type") == "SUPPORTED_BY" and edge.get("source") in fact_ids_in_graph:
            evidence_by_fact[edge["source"]].add(edge["target"])
    section_by_evidence = {
        edge["target"]: edge["source"]
        for edge in edges if edge.get("type") == "HAS_EVIDENCE"
    }
    existing_id_by_name = {
        lexical_key(item.get("name")): node_id for node_id, item in source_relations.items()
    }

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    decisions: list[dict[str, Any]] = []
    for expression_id, expression in sorted(relation_expressions.items()):
        fact_id = fact_by_expression.get(expression_id, "")
        canonical, method = canonicalize_relation(expression.get("name", ""), alias_map, controlled)
        if fact_id in quarantined_facts:
            canonical = None
            method = "parent_fact_quarantined"
        decision = {
            "relation_expression_id": expression_id,
            "fact_id": fact_id,
            "surface_form": expression.get("name", ""),
            "canonical_name": canonical,
            "status": "accepted" if canonical else "quarantined",
            "method": method,
        }
        decisions.append(decision)
        props = expression.setdefault("properties", {})
        props["canonicalization_status"] = decision["status"]
        props["canonicalization_method"] = method
        if canonical:
            grouped[canonical].append(decision)

    canonical_nodes: list[dict[str, Any]] = []
    canonical_edges: list[dict[str, Any]] = []
    ontology_nodes: list[dict[str, Any]] = []
    ontology_edges: list[dict[str, Any]] = []
    for canonical, items in sorted(grouped.items()):
        forms = unique_text(item["surface_form"] for item in items)
        fact_ids = {item["fact_id"] for item in items if item["fact_id"]}
        sections = {
            section_by_evidence[evidence_id]
            for fact_id in fact_ids
            for evidence_id in evidence_by_fact.get(fact_id, set())
            if evidence_id in section_by_evidence
        }
        keys = {lexical_key(canonical), *(lexical_key(form) for form in forms)}
        concept_id = registry.resolve(
            canonical,
            keys,
            existing_id_by_name.get(lexical_key(canonical)),
        )
        eligible = len(fact_ids) >= ontology_min_support and len(sections) >= ontology_min_sections
        canonical_nodes.append(
            node_record(
                concept_id,
                "CanonicalRelation",
                canonical,
                {
                    "surface_forms": forms,
                    "support_count": len(fact_ids),
                    "section_count": len(sections),
                    "source_sections": sorted(sections),
                    "mapping_status": "accepted",
                    "validation_status": "accepted",
                    "ontology_eligible": eligible,
                },
            )
        )
        for item in items:
            canonical_edges.append(
                edge_record(
                    item["relation_expression_id"],
                    concept_id,
                    "CANONICALIZED_AS",
                    "canonicalization",
                    properties={
                        "validation_status": "accepted",
                        "method": item["method"],
                    },
                )
            )
        if eligible:
            ontology_id = stable_id("ONT", canonical)
            ontology_nodes.append(
                node_record(
                    ontology_id,
                    "OntologyTerm",
                    canonical,
                    {
                        "kind": "relation",
                        "support_count": len(fact_ids),
                        "section_count": len(sections),
                        "source_sections": sorted(sections),
                        "schema_version": "4.5.1-cleaned",
                        "validation_status": "rule_validated",
                    },
                )
            )
            ontology_edges.append(
                edge_record(concept_id, ontology_id, "PROMOTED_AS", "ontology")
            )

    used_source_relation_names = {lexical_key(item["name"]) for item in canonical_nodes}
    quarantine = [
        {
            "source_canonical_id": source_id,
            "canonical_name": item.get("name", ""),
            "properties": item.get("properties", {}),
            "status": "quarantined",
            "reasons": ["uncontrolled_or_rebuilt_relation"],
        }
        for source_id, item in sorted(source_relations.items())
        if lexical_key(item.get("name")) not in used_source_relation_names
    ]
    report = {
        "source_canonical_relation_count": len(source_relations),
        "accepted_canonical_relation_count": len(canonical_nodes),
        "quarantined_source_relation_count": len(quarantine),
        "accepted_relation_expression_count": len(canonical_edges),
        "quarantined_relation_expression_count": sum(
            item["status"] == "quarantined" for item in decisions
        ),
        "promoted_ontology_term_count": len(ontology_nodes),
        "decision_method_counts": dict(Counter(item["method"] for item in decisions)),
    }
    return canonical_nodes, canonical_edges, ontology_nodes, ontology_edges, quarantine, decisions, report


def clean_pcf(
    pcf_dir: Path,
    taskkg_nodes: dict[str, dict[str, Any]],
    taskkg_edges: list[dict[str, Any]],
    *,
    accept_needs_review: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    pcf_nodes = read_jsonl(pcf_dir / "pcf_kg_nodes.jsonl")
    pcf_edges = read_jsonl(pcf_dir / "pcf_kg_edges.jsonl")
    links = read_jsonl(pcf_dir / "taskframe_pcf_links.jsonl")
    output_nodes: list[dict[str, Any]] = []
    for item in pcf_nodes:
        node = deepcopy(item)
        props = node.setdefault("properties", {})
        props["scheme"] = "PCF"
        props["code"] = props.get("hid", "")
        props["name_en"] = clean_text(props.get("name_en"))
        props["validation_status"] = "accepted"
        output_nodes.append(node)

    hierarchy_edges = [deepcopy(item) for item in pcf_edges if item.get("type") == "CHILD_OF"]
    active_fact_ids = {
        node_id for node_id, item in taskkg_nodes.items()
        if item.get("label") == "Fact"
        and item.get("properties", {}).get("active", True)
    }
    frame_facts: dict[str, set[str]] = defaultdict(set)
    facts_by_evidence: dict[str, set[str]] = defaultdict(set)
    for item in taskkg_edges:
        if (
            item.get("type") in {"HAS_CORE_FACT", "HAS_SUPPORTING_FACT", "HAS_CONTEXT_FACT"}
            and item.get("target") in active_fact_ids
        ):
            frame_facts[item["source"]].add(item["target"])
        elif item.get("type") == "SUPPORTED_BY" and item.get("source") in active_fact_ids:
            facts_by_evidence[item["target"]].add(item["source"])

    mapping_edges: list[dict[str, Any]] = []
    mapping_decisions: list[dict[str, Any]] = []
    remapped_mapping_count = 0
    removed_stale_support_count = 0
    added_current_support_count = 0
    confirmed_mapping_count = 0
    pending_mapping_count = 0
    unmapped_candidate_count = 0
    for link in links:
        if link.get("pcfid") is None or not link.get("pcf_hid"):
            unmapped_candidate_count += 1
            mapping_decisions.append(
                {
                    **link,
                    "source_validation_status": link.get("validation_status", ""),
                    "validation_status": "needs_review",
                    "review_status": "pending",
                    "neo4j_relationship_type": None,
                    "neo4j_edge_id": None,
                }
            )
            continue
        if link.get("mapping_role") == "context_anchor":
            proposed_relation = "HAS_PCF_CONTEXT"
        elif link.get("is_primary"):
            proposed_relation = "MAPS_TO_PCF"
        else:
            proposed_relation = "ALSO_MAPS_TO_PCF"
        source_status = str(link.get("validation_status") or "needs_review")
        confirmed = source_status in {"accepted", "auto_accepted", "confirmed"}
        if accept_needs_review and source_status == "needs_review":
            confirmed = True
        relation = proposed_relation if confirmed else "PCF_MAPPING_CANDIDATE"
        if confirmed:
            confirmed_mapping_count += 1
        else:
            pending_mapping_count += 1
        target = f"PCF_{link['pcfid']}"
        original_support = set(link.get("supporting_triples", []))
        preserved_support = original_support & active_fact_ids
        evidence_support = set().union(*(
            facts_by_evidence.get(evidence_id, set())
            for evidence_id in link.get("supporting_evidence_ids", [])
        )) if link.get("supporting_evidence_ids") else set()
        current_support = sorted(
            preserved_support
            | (frame_facts.get(link["taskframe_id"], set()) & evidence_support)
        )
        removed_stale = original_support - active_fact_ids
        added_current = set(current_support) - preserved_support
        if removed_stale or added_current:
            remapped_mapping_count += 1
            removed_stale_support_count += len(removed_stale)
            added_current_support_count += len(added_current)
        properties = {
            "mapping_id": link["mapping_id"],
            "mapping_role": link["mapping_role"],
            "is_primary": bool(link.get("is_primary")),
            "relation_scope": link.get("relation_scope", ""),
            "coverage_score": link.get("coverage_score"),
            "confidence": link.get("confidence"),
            "method": link.get("method", ""),
            "validation_status": "accepted" if confirmed else "needs_review",
            "source_validation_status": source_status,
            "review_status": "confirmed" if confirmed else "pending",
            "confirmed_by": (
                "pipeline_auto_accept"
                if source_status == "auto_accepted"
                else "explicit_accept_needs_review"
                if confirmed and source_status == "needs_review"
                else "source_review"
                if confirmed
                else ""
            ),
            "active": confirmed,
            "proposed_relationship_type": proposed_relation,
            "supporting_task_mentions": link.get("supporting_task_mentions", []),
            "supporting_triples": current_support,
            "supporting_triples_remapped": bool(removed_stale or added_current),
            "removed_stale_support_count": len(removed_stale),
            "added_current_support_count": len(added_current),
            "rationale": link.get("rationale", ""),
        }
        edge = edge_record(
            link["taskframe_id"],
            target,
            relation,
            "layer_connect",
            evidence_ids=link.get("supporting_evidence_ids", []),
            properties=properties,
            identity=link["mapping_id"],
        )
        mapping_edges.append(edge)
        mapping_decisions.append(
            {
                **link,
                "source_supporting_triples": link.get("supporting_triples", []),
                "supporting_triples": current_support,
                "supporting_triples_remapped": bool(removed_stale or added_current),
                "removed_stale_support_count": len(removed_stale),
                "added_current_support_count": len(added_current),
                "source_validation_status": link.get("validation_status", ""),
                "validation_status": "accepted" if confirmed else "needs_review",
                "review_status": "confirmed" if confirmed else "pending",
                "neo4j_relationship_type": relation,
                "neo4j_edge_id": edge["id"],
            }
        )
    report = {
        "pcf_node_count": len(output_nodes),
        "pcf_hierarchy_edge_count": len(hierarchy_edges),
        "confirmed_mapping_count": confirmed_mapping_count,
        "confirmed_primary_mapping_count": sum(edge["type"] == "MAPS_TO_PCF" for edge in mapping_edges),
        "confirmed_secondary_mapping_count": sum(edge["type"] == "ALSO_MAPS_TO_PCF" for edge in mapping_edges),
        "confirmed_context_mapping_count": sum(edge["type"] == "HAS_PCF_CONTEXT" for edge in mapping_edges),
        "pending_candidate_mapping_count": pending_mapping_count,
        "unmapped_candidate_count": unmapped_candidate_count,
        "remapped_mapping_count": remapped_mapping_count,
        "removed_stale_support_count": removed_stale_support_count,
        "added_current_support_count": added_current_support_count,
    }
    return output_nodes, [*hierarchy_edges, *mapping_edges], mapping_decisions, report


def validate_graph(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    node_ids = [item.get("id", "") for item in nodes]
    edge_ids = [item.get("id", "") for item in edges]
    node_set = set(node_ids)
    if "" in node_set:
        errors.append("empty node id")
    if len(node_ids) != len(node_set):
        errors.append("duplicate node id")
    if len(edge_ids) != len(set(edge_ids)):
        errors.append("duplicate edge id")
    for item in nodes:
        if not SAFE_TOKEN.fullmatch(str(item.get("label", ""))):
            errors.append(f"unsafe node label: {item.get('label')!r}")
    dangling: list[str] = []
    for item in edges:
        if not SAFE_TOKEN.fullmatch(str(item.get("type", ""))):
            errors.append(f"unsafe edge type: {item.get('type')!r}")
        if item.get("source") not in node_set or item.get("target") not in node_set:
            dangling.append(item.get("id", ""))
        missing_evidence = [
            evidence_id for evidence_id in item.get("evidence_ids", [])
            if evidence_id not in node_set
        ]
        if missing_evidence:
            warnings.append(
                f"edge evidence not materialized: {item.get('id')} ({len(missing_evidence)})"
            )
    if dangling:
        errors.append(f"dangling edge count: {len(dangling)}")

    label_counts = Counter(item["label"] for item in nodes)
    type_counts = Counter(item["type"] for item in edges)
    node_by_id = {item["id"]: item for item in nodes}
    frame_ids = {item["id"] for item in nodes if item["label"] == "TaskFrame"}
    membership_counts = Counter(
        item["source"] for item in edges if item["type"] == "MEMBER_OF" and item["target"] in frame_ids
    )
    task_mentions = {item["id"] for item in nodes if item["label"] == "TaskMention"}
    if set(membership_counts) != task_mentions or any(count != 1 for count in membership_counts.values()):
        errors.append("TaskMention MEMBER_OF partition invalid")
    mapping_types = {"MAPS_TO_PCF", "ALSO_MAPS_TO_PCF", "HAS_PCF_CONTEXT"}
    candidate_type = "PCF_MAPPING_CANDIDATE"
    primary_counts = Counter()
    for item in edges:
        if item["type"] not in mapping_types | {candidate_type}:
            continue
        props = item.get("properties", {})
        proposed_type = props.get("proposed_relationship_type", item["type"])
        if proposed_type == "MAPS_TO_PCF":
            primary_counts[item["source"]] += 1
        if item["type"] in mapping_types:
            if (
                props.get("validation_status") != "accepted"
                or props.get("review_status") != "confirmed"
                or not props.get("active", False)
            ):
                errors.append(f"unconfirmed active PCF mapping: {item['id']}")
        else:
            if (
                props.get("validation_status") != "needs_review"
                or props.get("review_status") != "pending"
                or props.get("active", True)
                or proposed_type not in mapping_types
            ):
                errors.append(f"invalid pending PCF candidate: {item['id']}")
        for support_key in ("supporting_task_mentions", "supporting_triples"):
            missing_support = [value for value in props.get(support_key, []) if value not in node_set]
            if missing_support:
                errors.append(
                    f"mapping support missing: {item['id']} {support_key}={len(missing_support)}"
                )
    if any(count > 1 for count in primary_counts.values()):
        errors.append("TaskFrame primary PCF mapping is not unique")
    missing_primary = sorted(frame_ids - set(primary_counts))
    if missing_primary:
        warnings.append(
            f"TaskFrame without viable primary PCF mapping: {len(missing_primary)}"
        )

    incoming_by_target = Counter(item["target"] for item in edges)
    for item in nodes:
        if item["label"] in {"CanonicalEntity", "CanonicalRelation"} and not any(
            edge["target"] == item["id"] and edge["type"] == "CANONICALIZED_AS"
            for edge in edges
        ):
            errors.append(f"canonical concept without incoming mapping: {item['id']}")
        if item["label"] == "TaskFrameAlias" and incoming_by_target[item["id"]] == 0:
            errors.append(f"TaskFrameAlias without HAS_ALIAS: {item['id']}")
        if item["label"] == "EntityMention" and not item.get("properties", {}).get("canonicalization_status"):
            errors.append(f"EntityMention canonicalization status missing: {item['id']}")

    pcf_ids = {item["id"] for item in nodes if item["label"] == "PCFProcess"}
    pcf_parent_counts = Counter(
        item["source"] for item in edges if item["type"] == "CHILD_OF"
    )
    pcf_roots = pcf_ids - set(pcf_parent_counts)
    if len(pcf_roots) != 13:
        errors.append(f"PCF root count != 13: {len(pcf_roots)}")
    expected_hierarchy_edges = len(pcf_ids) - len(pcf_roots)
    if type_counts["CHILD_OF"] != expected_hierarchy_edges:
        errors.append(
            "PCF hierarchy edge count mismatch: "
            f"expected={expected_hierarchy_edges}, actual={type_counts['CHILD_OF']}"
        )
    if any(count != 1 for count in pcf_parent_counts.values()):
        errors.append("PCF node with multiple parents")
    for item in nodes:
        if item["label"] == "TaskFrame" and "aliases" in item.get("properties", {}):
            errors.append(f"TaskFrame alias property not separated: {item['id']}")

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes_by_label": dict(sorted(label_counts.items())),
        "edges_by_type": dict(sorted(type_counts.items())),
        "dangling_edge_count": len(dangling),
        "duplicate_node_id_count": len(node_ids) - len(node_set),
        "duplicate_edge_id_count": len(edge_ids) - len(set(edge_ids)),
    }


def build_markdown_report(
    summary: dict[str, Any],
    validation: dict[str, Any],
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> str:
    node_by_id = {item["id"]: item for item in nodes}
    aliases_by_frame: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        if edge["type"] == "HAS_ALIAS":
            aliases_by_frame[edge["source"]].append(node_by_id[edge["target"]]["name"])
    frames = sorted(
        (item for item in nodes if item["label"] == "TaskFrame"),
        key=lambda item: item["name"],
    )
    relations = sorted(
        (item for item in nodes if item["label"] == "CanonicalRelation"),
        key=lambda item: (-item["properties"]["support_count"], item["name"]),
    )
    entity_reasons = summary["entities"]["quarantine_reason_counts"]
    lines = [
        "# TaskFrame v4.5 + PCF cleaning report",
        "",
        f"- Status: `{summary['status']}`",
        f"- Neo4j nodes: `{summary['neo4j']['node_count']}`",
        f"- Neo4j relationships: `{summary['neo4j']['edge_count']}`",
        f"- Dangling relationships: `{validation['dangling_edge_count']}`",
        f"- Duplicate node IDs: `{validation['duplicate_node_id_count']}`",
        f"- Duplicate relationship IDs: `{validation['duplicate_edge_id_count']}`",
        "",
        "## TaskFrame aliases",
        "",
        f"- Accepted alias occurrences: `{summary['taskframe_aliases'].get('accepted_alias', 0)}`",
        f"- Reclassified as raw TaskMention: `{summary['taskframe_aliases'].get('raw_task_mention', 0)}`",
        f"- Rejected sentence/structured aliases: `{summary['taskframe_aliases'].get('rejected_raw_like', 0)}`",
        "",
        "| TaskFrame | Curated aliases |",
        "|---|---|",
    ]
    for frame in frames:
        aliases = sorted(aliases_by_frame.get(frame["id"], []))
        lines.append(f"| {frame['name']} | {', '.join(aliases) if aliases else '-'} |")
    lines.extend(
        [
            "",
            "## Canonical entities",
            "",
            f"- Source concepts: `{summary['entities']['source_canonical_entity_count']}`",
            f"- Accepted concepts: `{summary['entities']['accepted_canonical_entity_count']}`",
            f"- Quarantined groups: `{summary['entities']['quarantined_canonical_entity_group_count']}`",
            f"- Accepted mention mappings: `{summary['entities']['accepted_entity_mapping_edge_count']}`",
            "",
            "| Quarantine reason | Count |",
            "|---|---:|",
        ]
    )
    for reason, count in sorted(entity_reasons.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{reason}` | {count} |")
    lines.extend(
        [
            "",
            "## Canonical relations",
            "",
            f"- Source concepts: `{summary['relations']['source_canonical_relation_count']}`",
            f"- Accepted controlled relations: `{summary['relations']['accepted_canonical_relation_count']}`",
            f"- Accepted relation expressions: `{summary['relations']['accepted_relation_expression_count']}`",
            f"- Quarantined relation expressions: `{summary['relations']['quarantined_relation_expression_count']}`",
            f"- Promoted ontology terms: `{summary['relations']['promoted_ontology_term_count']}`",
            "",
            "| Relation | Support | Sections | Ontology eligible |",
            "|---|---:|---:|---|",
        ]
    )
    for relation in relations:
        props = relation["properties"]
        lines.append(
            f"| {relation['name']} | {props['support_count']} | {props['section_count']} | "
            f"{'yes' if props['ontology_eligible'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Evidence quality",
            "",
            f"- Source extraction schema errors: `{summary['quality']['source_schema_error_count']}`",
            f"- Quarantined Facts: `{summary['quality']['quarantined_fact_count']}`",
            f"- Review-required TaskMentions: `{summary['quality']['review_task_mention_count']}`",
            f"- Missing sentence audits: `{summary['quality']['missing_sentence_audit_count']}`",
            "",
            "## Confirmed PCF links",
            "",
            f"- Primary `MAPS_TO_PCF`: `{summary['pcf']['confirmed_primary_mapping_count']}`",
            f"- Secondary `ALSO_MAPS_TO_PCF`: `{summary['pcf']['confirmed_secondary_mapping_count']}`",
            f"- Context `HAS_PCF_CONTEXT`: `{summary['pcf']['confirmed_context_mapping_count']}`",
            f"- Pending `PCF_MAPPING_CANDIDATE`: `{summary['pcf']['pending_candidate_mapping_count']}`",
            f"- Unmapped candidates: `{summary['pcf']['unmapped_candidate_count']}`",
            "",
            "Only reviewed or pipeline-auto-accepted PCF links use "
            "`validation_status=accepted`, `review_status=confirmed`, and `active=true`. "
            "Pending links remain `PCF_MAPPING_CANDIDATE` with `active=false`.",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    taskkg_dir = args.taskkg_dir.expanduser().resolve()
    extraction_dir = args.extraction_dir.expanduser().resolve()
    entity_prep_dir = args.entity_prep_dir.expanduser().resolve()
    pcf_dir = args.pcf_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    required = [
        taskkg_dir / "nodes.jsonl",
        taskkg_dir / "edges.jsonl",
        extraction_dir / "summary.json",
        extraction_dir / "chunk_extractions.jsonl",
        entity_prep_dir / "open_triples.jsonl",
        entity_prep_dir / "task_mentions.jsonl",
        pcf_dir / "pcf_kg_nodes.jsonl",
        pcf_dir / "pcf_kg_edges.jsonl",
        pcf_dir / "taskframe_pcf_links.jsonl",
        args.entity_aliases,
        args.relation_aliases,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("필수 입력 누락:\n" + "\n".join(missing))

    source_nodes = read_jsonl(taskkg_dir / "nodes.jsonl")
    source_edges = read_jsonl(taskkg_dir / "edges.jsonl")
    nodes = {item["id"]: deepcopy(item) for item in source_nodes}
    edges = [deepcopy(item) for item in source_edges]
    display_report = sanitize_display_names(nodes)

    entity_alias_map, entity_alias_values = load_alias_config(args.entity_aliases)
    relation_alias_map, controlled_relations = load_alias_config(args.relation_aliases)
    entity_registry = ConceptRegistry(output_dir / "registry/entity_registry.jsonl", "CE")
    relation_registry = ConceptRegistry(output_dir / "registry/relation_registry.jsonl", "CR")

    quarantined_facts, _, quality_issues, quality_report = decorate_quality(
        nodes, entity_prep_dir, extraction_dir
    )
    alias_nodes, alias_edges, alias_decisions, alias_report = taskframe_alias_projection(nodes, edges)

    entity_nodes, entity_edges, entity_quarantine, entity_report = clean_entities(
        nodes,
        edges,
        entity_alias_map,
        entity_alias_values,
        entity_registry,
        min_support=args.entity_min_support,
        max_length=args.entity_max_name_length,
        quarantined_facts=quarantined_facts,
    )
    (
        relation_nodes,
        relation_edges,
        ontology_nodes,
        ontology_edges,
        relation_quarantine,
        relation_decisions,
        relation_report,
    ) = clean_relations(
        nodes,
        edges,
        relation_alias_map,
        controlled_relations,
        relation_registry,
        ontology_min_support=args.ontology_min_support,
        ontology_min_sections=args.ontology_min_sections,
        quarantined_facts=quarantined_facts,
    )
    pcf_nodes, pcf_edges, mapping_decisions, pcf_report = clean_pcf(
        pcf_dir,
        nodes,
        edges,
        accept_needs_review=args.accept_needs_review_pcf,
    )

    excluded_labels = {"CanonicalEntity", "CanonicalTerm", "OntologyTerm"}
    base_nodes = [item for item in nodes.values() if item.get("label") not in excluded_labels]
    source_entity_ids = {item["id"] for item in source_nodes if item.get("label") == "CanonicalEntity"}
    source_relation_ids = {item["id"] for item in source_nodes if item.get("label") == "CanonicalTerm"}
    source_ontology_ids = {item["id"] for item in source_nodes if item.get("label") == "OntologyTerm"}
    excluded_targets = source_entity_ids | source_relation_ids | source_ontology_ids
    base_edges = [
        item for item in edges
        if item.get("type") not in {"PROMOTED_AS"}
        and not (
            item.get("type") == "CANONICALIZED_AS"
            and item.get("target") in excluded_targets
        )
    ]

    final_nodes = [
        *base_nodes,
        *entity_nodes,
        *relation_nodes,
        *ontology_nodes,
        *alias_nodes,
        *pcf_nodes,
    ]
    final_edges = [
        *base_edges,
        *entity_edges,
        *relation_edges,
        *ontology_edges,
        *alias_edges,
        *pcf_edges,
    ]
    final_nodes.sort(key=lambda item: item["id"])
    final_edges.sort(key=lambda item: item["id"])
    validation = validate_graph(final_nodes, final_edges)

    write_jsonl(output_dir / "neo4j/nodes.jsonl", final_nodes)
    write_jsonl(output_dir / "neo4j/edges.jsonl", final_edges)
    write_jsonl(output_dir / "audit/taskframe_alias_decisions.jsonl", alias_decisions)
    write_jsonl(output_dir / "audit/relation_expression_decisions.jsonl", relation_decisions)
    write_jsonl(output_dir / "audit/quality_issues.jsonl", quality_issues)
    write_jsonl(output_dir / "audit/pcf_mapping_decisions.jsonl", mapping_decisions)
    write_jsonl(output_dir / "quarantine/canonical_entities.jsonl", entity_quarantine)
    write_jsonl(output_dir / "quarantine/canonical_relations.jsonl", relation_quarantine)
    entity_registry.save()
    relation_registry.save()

    summary = {
        "status": "valid" if validation["valid"] else "invalid",
        "inputs": {
            "taskkg_dir": str(taskkg_dir),
            "extraction_dir": str(extraction_dir),
            "entity_prep_dir": str(entity_prep_dir),
            "pcf_dir": str(pcf_dir),
        },
        "source": {"node_count": len(source_nodes), "edge_count": len(source_edges)},
        "taskframe_aliases": alias_report,
        "entities": entity_report,
        "relations": relation_report,
        "quality": quality_report,
        "display_names": {"sanitized_by_label": display_report},
        "pcf": pcf_report,
        "neo4j": {
            "node_count": len(final_nodes),
            "edge_count": len(final_edges),
            "nodes_path": str(output_dir / "neo4j/nodes.jsonl"),
            "edges_path": str(output_dir / "neo4j/edges.jsonl"),
        },
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "validation_report.json", validation)
    write_text(
        output_dir / "cleaning_report.md",
        build_markdown_report(summary, validation, final_nodes, final_edges),
    )
    if not validation["valid"]:
        raise RuntimeError("정제 그래프 검증 실패: " + "; ".join(validation["errors"]))
    return summary


def main() -> int:
    args = parse_args()
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
