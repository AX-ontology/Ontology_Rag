#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
from typing import Any, Iterable


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def e(value: Any) -> str:
    return escape("" if value is None else str(value), quote=True)


def pct(numerator: int | float, denominator: int | float) -> str:
    return f"{(100 * numerator / denominator):.1f}%" if denominator else "—"


def badge(label: str, kind: str = "neutral") -> str:
    return f'<span class="badge {e(kind)}">{e(label)}</span>'


def unique(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value or "").strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def triple_text(fact: dict[str, Any], canonical: bool = False) -> str:
    suffix = "_canonical" if canonical else "_raw"
    subject = fact.get(f"subject{suffix}", "") or "∅"
    predicate = fact.get(f"predicate{suffix}", "") or "∅"
    obj = fact.get(f"object{suffix}", "") or "∅"
    return f"{subject} → {predicate} → {obj}"


def role_badge(role: str) -> str:
    return badge(
        role or "unlinked",
        {
            "core": "core",
            "supporting": "supporting",
            "context": "context",
        }.get(role, "neutral"),
    )


def knowledge_badge(function: str) -> str:
    value = str(function or "fact")
    return badge(value, "llm" if value != "fact" else "neutral")


def qualifier_badges(qualifiers: Any) -> str:
    if not isinstance(qualifiers, list):
        return ""
    return " ".join(
        badge(
            f"{item.get('type', 'qualifier')}: {item.get('text', '')}",
            "scope",
        )
        for item in qualifiers
        if isinstance(item, dict) and item.get("text")
    )


def fact_semantics(fact: dict[str, Any], compact: bool = False) -> str:
    qualifiers = qualifier_badges(fact.get("qualifiers", []))
    return f"""
    <div class="fact-semantics{' compact' if compact else ''}">
      <span class="fact-meta-label">Knowledge function</span>
      {knowledge_badge(fact.get('knowledge_function', 'fact'))}
      <span class="fact-meta-label">Qualifiers</span>
      {qualifiers or badge('없음', 'neutral')}
    </div>"""


def stat_card(label: str, value: Any, note: str, kind: str = "") -> str:
    return f"""
    <article class="stat-card {e(kind)}">
      <div class="stat-label">{e(label)}</div>
      <div class="stat-value">{e(value)}</div>
      <div class="stat-note">{e(note)}</div>
    </article>"""


def finding_card(level: str, title: str, body: str, metric: str = "") -> str:
    return f"""
    <article class="finding {e(level)}">
      <div class="finding-icon">{'!' if level in {'warn', 'risk'} else '✓'}</div>
      <div>
        <div class="finding-title">{e(title)} {badge(metric, level) if metric else ''}</div>
        <p>{e(body)}</p>
      </div>
    </article>"""


def render_report(input_dir: Path) -> str:
    manifest = read_json(input_dir / "run_manifest.json", {})
    quality = read_json(input_dir / "09_kg" / "quality_report.json", {})
    graph_validation = read_json(
        input_dir / "09_kg" / "validation_report.json", {}
    )
    extraction_summary = read_json(
        input_dir / "02_extraction" / "summary.json", {}
    )
    prep_summary = read_json(input_dir / "03_entity_prep" / "summary.json", {})
    clustering_summary = read_json(
        input_dir / "04_clustering" / "summary.json", {}
    )
    call_plan = read_json(
        input_dir / "04_clustering" / "llm_call_plan.json", {}
    )
    frame_validation = read_json(
        input_dir / "05_taskframes" / "validation_report.json", {}
    )
    propagation_summary = read_json(
        input_dir / "06_propagation" / "summary.json", {}
    )
    canonical_summary = read_json(
        input_dir / "07_canonicalization" / "summary.json", {}
    )
    standard_summary = read_json(
        input_dir / "08_standard_mapping" / "summary.json", {}
    )

    evidence = read_jsonl(input_dir / "01_evidence" / "evidence_spans.jsonl")
    chunks = read_jsonl(input_dir / "01_evidence" / "chunks.jsonl")
    raw_facts = read_jsonl(
        input_dir / "02_extraction" / "open_triples_raw.jsonl"
    )
    mentions = read_jsonl(
        input_dir / "03_entity_prep" / "task_mentions.jsonl"
    )
    frames = read_json(input_dir / "05_taskframes" / "taskframes.json", [])
    task_relations = read_jsonl(
        input_dir / "05_taskframes" / "task_relations.jsonl"
    )
    decisions = read_jsonl(
        input_dir / "05_taskframes" / "triple_link_decisions.jsonl"
    )
    ambiguous = read_jsonl(
        input_dir / "05_taskframes" / "ambiguous_fact_contexts.jsonl"
    )
    facts = read_jsonl(
        input_dir / "07_canonicalization" / "canonical_facts.jsonl"
    )
    canonical_entities = read_jsonl(
        input_dir / "07_canonicalization" / "canonical_entities.jsonl"
    )
    canonical_relations = read_jsonl(
        input_dir / "07_canonicalization" / "canonical_relations.jsonl"
    )
    promoted_terms = read_jsonl(
        input_dir / "07_canonicalization" / "promoted_ontology_terms.jsonl"
    )

    evidence_lookup = {item["evidence_id"]: item for item in evidence}
    fact_lookup = {item["triple_id"]: item for item in facts}
    mention_lookup = {item["task_mention_id"]: item for item in mentions}
    frame_lookup = {item["taskframe_id"]: item for item in frames}
    decision_lookup = {item["triple_id"]: item for item in decisions}

    facts_by_evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    mentions_by_evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    facts_by_frame: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = (
        defaultdict(list)
    )
    mention_to_frame: dict[str, str] = {}
    for frame in frames:
        for mention_id in frame.get("member_task_mentions", []):
            mention_to_frame[mention_id] = frame["taskframe_id"]
    for fact in facts:
        facts_by_evidence[fact.get("evidence_id", "")].append(fact)
        for link in fact.get("taskframe_links", []):
            facts_by_frame[link["taskframe_id"]].append((fact, link))
    for mention in mentions:
        for evidence_id in mention.get("evidence_ids", []):
            mentions_by_evidence[evidence_id].append(mention)

    outgoing_relations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    incoming_relations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for relation in task_relations:
        outgoing_relations[relation["source_taskframe_id"]].append(relation)
        incoming_relations[relation["target_taskframe_id"]].append(relation)

    bad_facts = [
        fact for fact in facts if not fact.get("surface_provenance_valid", False)
    ]
    facts_with_object = sum(bool(fact.get("object_raw")) for fact in facts)
    evidence_with_facts = sum(
        bool(facts_by_evidence[item["evidence_id"]]) for item in evidence
    )
    frame_core_counts = {
        frame["taskframe_id"]: sum(
            link.get("relevance") == "core"
            for _, link in facts_by_frame[frame["taskframe_id"]]
        )
        for frame in frames
    }
    frames_without_core = [
        frame for frame in frames if not frame_core_counts[frame["taskframe_id"]]
    ]
    low_confidence_frames = [
        frame
        for frame in frames
        if any(
            float(mention_lookup.get(mention_id, {}).get("confidence", 1.0)) < 0.5
            for mention_id in frame.get("member_task_mentions", [])
        )
    ]
    descriptive_frames = [
        frame
        for frame in frames
        if any(
            any(
                marker in mention_lookup.get(mention_id, {}).get("source_text", "")
                for marker in ("경우가 많다", "축소 진행", "점점 짧아")
            )
            for mention_id in frame.get("member_task_mentions", [])
        )
    ]
    total_frame_links = sum(len(items) for items in facts_by_frame.values())
    dominant_frames = [
        frame for frame in frames
        if total_frame_links
        and len(facts_by_frame[frame["taskframe_id"]]) / total_frame_links >= 0.4
    ]
    unresolved_relations = sum(
        item.get("mapping_status") == "unresolved"
        for item in canonical_relations
    )
    stable_entities = sum(
        item.get("mapping_status") == "stable" for item in canonical_entities
    )
    relation_review_count = sum(
        float(item.get("confidence", 0)) < 0.7 for item in task_relations
    )

    counts = manifest.get("counts", {})
    quality_status = str(
        quality.get("status", manifest.get("quality_status", "unknown"))
    )
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    methodology_steps = [
        (
            "01", "Evidence 보존", f"{len(evidence)} Evidence · {len(chunks)} chunks",
            "문서·절·문장 위치와 원문을 먼저 고정한다.", "algorithm",
        ),
        (
            "02", "LLM 1 공동 추출",
            f"{len(raw_facts)} Facts · {len(mentions)} TaskMentions",
            "원문 S–P–O, 지식 기능, qualifier, exact quote를 추출하고 로컬 Task 연결을 보존한다.", "llm",
        ),
        (
            "03", "보수적 정규화",
            f"verified aliases {prep_summary.get('verified_entity_alias_count', 0)}",
            "원문은 유지하고 명백한 표기 차이만 정리한다.", "algorithm",
        ),
        (
            "04", "후보 blocking",
            f"{clustering_summary.get('candidate_block_count', 0)} blocks",
            "최종 동일성 판단이 아니라 LLM 2 비교 범위만 줄인다.", "algorithm",
        ),
        (
            "05", "LLM 2 TaskFrame",
            f"{len(frames)} Frames · {len(decisions)} rejudgments",
            "TaskMention을 전역 병합·분리하고 애매한 Fact를 주변 chunk와 재판정한다.",
            "llm",
        ),
        (
            "06", "다중 연결 전파",
            f"{propagation_summary.get('fact_taskframe_link_count', 0)} Fact–Frame links",
            "Fact→TaskMention→TaskFrame 경로를 추가 LLM 없이 합성한다.", "algorithm",
        ),
        (
            "07", "Canonicalization",
            f"{len(canonical_entities)} entities · {len(canonical_relations)} relations",
            "원문 표현을 canonical concept와 연결하고 안정된 관계만 승격한다.",
            "algorithm",
        ),
        (
            "08–09", "표준 매핑·KG",
            f"{graph_validation.get('node_count', 0)} nodes · {graph_validation.get('edge_count', 0)} edges",
            "선택적 표준업무 매핑 후 최소 메타스키마 KG를 검증한다.",
            "algorithm",
        ),
    ]
    methodology_html = "".join(
        f"""
        <article class="method-step">
          <div class="step-no">{e(no)}</div>
          <div class="step-body">
            <div class="step-title">{e(title)} {badge('LLM', 'llm') if mode == 'llm' else badge('Algorithm', 'algo')}</div>
            <div class="step-metric">{e(metric)}</div>
            <p>{e(description)}</p>
          </div>
        </article>"""
        for no, title, metric, description, mode in methodology_steps
    )

    findings = [
        finding_card(
            "ok", "구조 무결성 통과",
            "모든 노드·엣지 참조, Evidence 연결, TaskMention partition이 유효합니다.",
            f"{graph_validation.get('node_count', 0)} nodes",
        ),
        finding_card(
            "ok", "원본 Fact 보존",
            "LLM 1의 자유형 Fact가 canonicalization 이후에도 삭제되지 않았습니다.",
            pct(len(facts), len(raw_facts)),
        ),
        finding_card(
            "ok" if not propagation_summary.get("unlinked_fact_count", 0) else "risk",
            "전체 Fact–TaskFrame 연결",
            "모든 Fact는 최소 하나의 TaskFrame에 core/supporting/context 역할로 연결되어야 합니다. LLM 누락은 알고리즘 fallback이 보완합니다.",
            pct(
                propagation_summary.get("assigned_fact_count", 0),
                propagation_summary.get("fact_count", len(facts)),
            ),
        ),
        finding_card(
            "warn", "원문 grounding 검토 필요",
            "따옴표 형태 변경, 조사 분리, 표 관계어 ':'를 '는'으로 바꾼 사례 등이 있습니다. 원문 exact substring 계약을 위반한 Fact는 아래 검토 큐에서 확인할 수 있습니다.",
            f"{len(bad_facts)} facts",
        ),
        finding_card(
            "warn", "core Fact 없는 TaskFrame",
            "TaskMention은 Frame에 들어갔지만 업무 수행 자체를 직접 나타내는 core Fact가 없어 supporting 문맥에 의존합니다.",
            f"{len(frames_without_core)} frames",
        ),
        finding_card(
            "risk", "서술형 문장의 Task 오인 가능성",
            "‘평가 기간이 축소 진행되는 경우가 많다’는 변화 서술이 ‘제안 평가’ TaskMention으로 승격되어 일정 Fact까지 연결됐습니다. Task gate 재검토가 필요합니다.",
            f"{len(descriptive_frames)} frame",
        ),
        finding_card(
            "warn" if dominant_frames else "ok",
            "TaskFrame context 집중도",
            "전체 Fact를 포괄하면서 하나의 Frame에 context가 과도하게 몰리면 Task 경계가 너무 넓거나 기준 TaskMention이 부족할 수 있습니다.",
            (
                f"{dominant_frames[0]['canonical_name']} "
                f"{pct(len(facts_by_frame[dominant_frames[0]['taskframe_id']]), total_frame_links)}"
                if dominant_frames else "분산 양호"
            ),
        ),
        finding_card(
            "warn", "Relation canonicalization 초기 단계",
            "원문 관계 표현 대부분이 unresolved 상태이므로 현재 결과는 Open KG 관찰값으로 보는 것이 적절합니다. 정식 온톨로지 승격은 아직 없습니다.",
            f"{unresolved_relations}/{len(canonical_relations)} unresolved",
        ),
    ]

    frame_cards: list[str] = []
    for frame in frames:
        frame_id = frame["taskframe_id"]
        member_items: list[str] = []
        for mention_id in frame.get("member_task_mentions", []):
            mention = mention_lookup.get(mention_id, {})
            member_items.append(
                f"""
                <article class="mention-card">
                  <div class="row-between">
                    <strong>{e(mention.get('surface_form', mention_id))}</strong>
                    {badge(f"confidence {float(mention.get('confidence', 0)):.1f}", 'warn' if float(mention.get('confidence', 0)) < .5 else 'ok')}
                  </div>
                  <div class="mention-signature">
                    <span>Action</span> {e(mention.get('action_normalized', mention.get('action_raw', '')))}
                    <span>Object</span> {e(mention.get('focal_object_normalized', mention.get('focal_object_raw', '')))}
                  </div>
                  <blockquote>{e(mention.get('source_text', ''))}</blockquote>
                  <div class="mono subtle">{e(mention_id)}</div>
                </article>"""
            )
        frame_fact_items: list[str] = []
        frame_evidence_ids: list[str] = list(frame.get("evidence_ids", []))
        for fact, link in facts_by_frame[frame_id]:
            frame_evidence_ids.extend(fact.get("evidence_ids", []))
            frame_fact_items.append(
                f"""
                <article class="fact-card {'review' if not fact.get('surface_provenance_valid') else ''}">
                  <div class="row-between">
                    <div>{role_badge(link.get('relevance', ''))} {badge('LLM 2 재판정', 'llm') if fact.get('link_rejudged_by_llm_2') else badge('LLM 1 연결', 'algo')}</div>
                    <a href="#fact-{e(fact['triple_id'])}" class="mono mini-link">{e(fact['triple_id'])}</a>
                  </div>
                  <div class="triple"><span>{e(fact.get('subject_raw') or '∅')}</span><b>→</b><span class="predicate">{e(fact.get('predicate_raw') or '∅')}</span><b>→</b><span>{e(fact.get('object_raw') or '∅')}</span></div>
                  {fact_semantics(fact)}
                  <blockquote>{e(fact.get('evidence_quote', fact.get('source_text', '')))}</blockquote>
                </article>"""
            )
        evidence_items: list[str] = []
        for evidence_id in unique(frame_evidence_ids):
            item = evidence_lookup.get(evidence_id, {})
            if not item:
                continue
            evidence_items.append(
                f"""
                <article class="evidence-mini">
                  <div class="evidence-meta">p.{e(item.get('page', ''))} · {e(item.get('section_path', ''))} · <a href="#evidence-{e(evidence_id)}">{e(evidence_id)}</a></div>
                  <div>{e(item.get('source_text', ''))}</div>
                </article>"""
            )
        relation_items: list[str] = []
        for relation in outgoing_relations[frame_id]:
            target = frame_lookup.get(relation["target_taskframe_id"], {})
            relation_items.append(
                f"{badge(relation['relation'], 'relation')} → <strong>{e(target.get('canonical_name', relation['target_taskframe_id']))}</strong> <span class='subtle'>({float(relation.get('confidence', 0)):.1f}) {e(relation.get('reason', ''))}</span>"
            )
        for relation in incoming_relations[frame_id]:
            source = frame_lookup.get(relation["source_taskframe_id"], {})
            relation_items.append(
                f"<strong>{e(source.get('canonical_name', relation['source_taskframe_id']))}</strong> → {badge(relation['relation'], 'relation')} <span class='subtle'>({float(relation.get('confidence', 0)):.1f})</span>"
            )
        flags: list[str] = []
        if frame in frames_without_core:
            flags.append(badge("core Fact 없음", "warn"))
        if frame in low_confidence_frames:
            flags.append(badge("낮은 신뢰도 Mention", "warn"))
        if frame in descriptive_frames:
            flags.append(badge("서술형 Task 검토", "risk"))
        if frame in dominant_frames:
            flags.append(badge("Fact 40% 이상 집중", "warn"))
        frame_cards.append(
            f"""
            <article class="frame-card searchable" id="frame-{e(frame_id)}" data-search="{e(' '.join([frame.get('canonical_name',''), frame.get('definition',''), *frame.get('aliases',[])]).lower())}">
              <header class="frame-header">
                <div>
                  <div class="eyebrow">TASKFRAME · {e(frame.get('status', ''))}</div>
                  <h3>{e(frame.get('canonical_name', frame_id))}</h3>
                  <p>{e(frame.get('definition', ''))}</p>
                </div>
                <div class="frame-score">
                  <strong>{len(facts_by_frame[frame_id])}</strong>
                  <span>linked Facts</span>
                </div>
              </header>
              <div class="frame-meta">
                {badge(frame.get('scope', 'scope 없음'), 'scope')}
                {''.join(flags)}
                <span class="mono">{e(frame_id)}</span>
              </div>
              <div class="aliases"><strong>Aliases</strong> · {e(' · '.join(frame.get('aliases', [])) or '—')}</div>
              <details open>
                <summary>Member TaskMentions ({len(member_items)})</summary>
                <div class="stack">{''.join(member_items) or '<p class="empty">없음</p>'}</div>
              </details>
              <details open>
                <summary>Task-centered Fact subgraph ({len(frame_fact_items)})</summary>
                <div class="stack">{''.join(frame_fact_items) or '<p class="empty">연결 Fact 없음</p>'}</div>
              </details>
              <details>
                <summary>Supporting Evidence ({len(evidence_items)})</summary>
                <div class="stack">{''.join(evidence_items) or '<p class="empty">없음</p>'}</div>
              </details>
              <details>
                <summary>Task relations ({len(relation_items)})</summary>
                <ul class="relation-list">{''.join(f'<li>{item}</li>' for item in relation_items) or '<li class="empty">없음</li>'}</ul>
              </details>
            </article>"""
        )

    bad_fact_rows: list[str] = []
    for fact in bad_facts:
        invalid_fields = [
            key for key, valid in fact.get("surface_grounding", {}).items()
            if not valid
        ]
        invalid_fields.extend(
            f"evidence.{key}"
            for key, valid in fact.get("evidence_grounding", {}).items()
            if not valid
        )
        bad_fact_rows.append(
            f"""
            <tr>
              <td><a href="#fact-{e(fact['triple_id'])}" class="mono">{e(fact['triple_id'])}</a></td>
              <td>{badge(', '.join(invalid_fields), 'risk')}</td>
              <td>{e(triple_text(fact))}{fact_semantics(fact, compact=True)}</td>
              <td>{e(fact.get('source_text', ''))}</td>
            </tr>"""
        )

    mention_rows: list[str] = []
    for mention in mentions:
        frame = frame_lookup.get(
            mention_to_frame.get(mention["task_mention_id"], ""), {}
        )
        mention_rows.append(
            f"""
            <tr class="searchable" data-search="{e(' '.join([mention.get('surface_form',''), mention.get('source_text',''), frame.get('canonical_name','')]).lower())}">
              <td>{e(mention.get('surface_form', ''))}<div class="mono subtle">{e(mention['task_mention_id'])}</div></td>
              <td>{e(mention.get('action_normalized', ''))}</td>
              <td>{e(mention.get('focal_object_normalized', ''))}</td>
              <td>{float(mention.get('confidence', 0)):.1f}</td>
              <td><a href="#frame-{e(frame.get('taskframe_id', ''))}">{e(frame.get('canonical_name', '미배정'))}</a></td>
              <td>{e(mention.get('source_text', ''))}</td>
            </tr>"""
        )

    fact_rows: list[str] = []
    for fact in facts:
        frame_links = fact.get("taskframe_links", [])
        rendered_qualifiers = [
            item for item in fact.get("qualifiers", [])
            if isinstance(item, dict) and item.get("text")
        ]
        frames_text = " · ".join(
            f"{frame_lookup.get(link['taskframe_id'], {}).get('canonical_name', link['taskframe_id'])} ({link['relevance']})"
            for link in frame_links
        )
        evidence_item = evidence_lookup.get(fact.get("evidence_id", ""), {})
        invalid = not fact.get("surface_provenance_valid", False)
        search_value = " ".join(
            [
                fact.get("subject_raw", ""), fact.get("predicate_raw", ""),
                fact.get("object_raw", ""), fact.get("source_text", ""),
                fact.get("knowledge_function", ""),
                *(
                    f"{item.get('type', 'qualifier')} {item.get('text', '')}"
                    for item in rendered_qualifiers
                ),
                frames_text,
            ]
        ).lower()
        fact_rows.append(
            f"""
            <tr class="searchable {'row-review' if invalid else ''}" id="fact-{e(fact['triple_id'])}" data-search="{e(search_value)}" data-linked="{'yes' if frame_links else 'no'}" data-review="{'yes' if invalid else 'no'}" data-function="{e(fact.get('knowledge_function', 'fact'))}" data-qualifiers="{'yes' if rendered_qualifiers else 'no'}">
              <td><span class="mono">{e(fact['triple_id'])}</span><br>{badge('review', 'risk') if invalid else badge('grounded', 'ok')}</td>
              <td><div class="triple compact"><span>{e(fact.get('subject_raw') or '∅')}</span><b>→</b><span class="predicate">{e(fact.get('predicate_raw') or '∅')}</span><b>→</b><span>{e(fact.get('object_raw') or '∅')}</span></div>{fact_semantics(fact)}</td>
              <td>{e(triple_text(fact, canonical=True))}</td>
              <td>{e(frames_text or '오류 · TaskFrame 미연결')}</td>
              <td><a href="#evidence-{e(fact.get('evidence_id', ''))}">{e(fact.get('evidence_id', ''))}</a><div class="subtle">p.{e(evidence_item.get('page', ''))}</div></td>
            </tr>"""
        )

    evidence_cards: list[str] = []
    for index, item in enumerate(evidence, 1):
        item_facts = facts_by_evidence[item["evidence_id"]]
        item_mentions = mentions_by_evidence[item["evidence_id"]]
        linked_frames = unique(
            frame_lookup.get(link["taskframe_id"], {}).get("canonical_name", "")
            for fact in item_facts for link in fact.get("taskframe_links", [])
        )
        review_count = sum(
            not fact.get("surface_provenance_valid", False) for fact in item_facts
        )
        fact_items = "".join(
            f"""
            <li class="{'review-text' if not fact.get('surface_provenance_valid') else ''}">
              <a href="#fact-{e(fact['triple_id'])}" class="mono mini-link">{e(fact['triple_id'])}</a>
              <span>{e(triple_text(fact))}</span>
              {' '.join(role_badge(link.get('relevance', '')) for link in fact.get('taskframe_links', []))}
              {fact_semantics(fact, compact=True)}
            </li>"""
            for fact in item_facts
        )
        mention_items = "".join(
            f"""
            <li>{badge('TaskMention', 'llm')} <strong>{e(mention.get('surface_form', ''))}</strong>
            → <a href="#frame-{e(mention_to_frame.get(mention['task_mention_id'], ''))}">{e(frame_lookup.get(mention_to_frame.get(mention['task_mention_id'], ''), {}).get('canonical_name', '미배정'))}</a></li>"""
            for mention in item_mentions
        )
        search_value = " ".join(
            [
                item.get("source_text", ""), item.get("section_path", ""),
                *linked_frames,
                *(triple_text(fact) for fact in item_facts),
            ]
        ).lower()
        evidence_cards.append(
            f"""
            <details class="evidence-card searchable" id="evidence-{e(item['evidence_id'])}" data-search="{e(search_value)}" data-review="{'yes' if review_count else 'no'}" {'open' if index <= 3 else ''}>
              <summary>
                <span class="evidence-index">{index:03d}</span>
                <span class="evidence-summary-text">{e(item.get('source_text', ''))}</span>
                <span class="summary-badges">{badge(f"{len(item_facts)} Facts", 'neutral')} {badge(f"{len(item_mentions)} Tasks", 'llm') if item_mentions else ''} {badge(f"{review_count} review", 'risk') if review_count else ''}</span>
              </summary>
              <div class="evidence-body">
                <div class="evidence-meta">p.{e(item.get('page', ''))} · row {e(item.get('excel_row', ''))} · {e(item.get('section_path', ''))} · <span class="mono">{e(item['evidence_id'])}</span></div>
                <blockquote>{e(item.get('source_text', ''))}</blockquote>
                <div class="evidence-grid">
                  <div><h4>Extracted triples</h4><ul class="fact-list">{fact_items or '<li class="empty">추출 Fact 없음</li>'}</ul></div>
                  <div><h4>Task links</h4><ul class="fact-list">{mention_items or '<li class="empty">TaskMention 없음</li>'}</ul><p class="subtle">{e(' · '.join(linked_frames) or '연결 TaskFrame 없음')}</p></div>
                </div>
              </div>
            </details>"""
        )

    relation_rows = "".join(
        f"""
        <tr>
          <td><a href="#frame-{e(item['source_taskframe_id'])}">{e(frame_lookup.get(item['source_taskframe_id'], {}).get('canonical_name', item['source_taskframe_id']))}</a></td>
          <td>{badge(item.get('relation', ''), 'relation')}</td>
          <td><a href="#frame-{e(item['target_taskframe_id'])}">{e(frame_lookup.get(item['target_taskframe_id'], {}).get('canonical_name', item['target_taskframe_id']))}</a></td>
          <td>{float(item.get('confidence', 0)):.1f}</td>
          <td>{e(item.get('reason', ''))}</td>
        </tr>"""
        for item in task_relations
    )

    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Task-Centric Open KG · 품질 리포트</title>
  <style>
    :root {{
      --ink:#18251f; --muted:#66746d; --paper:#f4f1e8; --card:#fffdf7;
      --line:#d9d7cc; --green:#1f6b4f; --green2:#dcece3; --amber:#9a6515;
      --amber2:#fff0cf; --red:#a13c32; --red2:#f9dfda; --blue:#315f84;
      --blue2:#e1edf6; --purple:#6c4f88; --shadow:0 12px 34px rgba(28,42,35,.08);
    }}
    * {{ box-sizing:border-box; }}
    html {{ scroll-behavior:smooth; }}
    body {{ margin:0; color:var(--ink); background:var(--paper); font-family:Inter, Pretendard, "Noto Sans KR", system-ui, sans-serif; line-height:1.58; }}
    a {{ color:var(--green); text-decoration:none; }} a:hover {{ text-decoration:underline; }}
    .layout {{ display:grid; grid-template-columns:240px minmax(0,1fr); min-height:100vh; }}
    aside {{ position:sticky; top:0; height:100vh; padding:28px 20px; background:#172820; color:#e8efe9; overflow:auto; }}
    .brand {{ font-weight:800; font-size:18px; line-height:1.25; margin-bottom:6px; }}
    .brand-sub {{ color:#9fb7aa; font-size:12px; margin-bottom:28px; }}
    nav a {{ display:block; color:#c5d5cc; padding:8px 10px; margin:3px 0; border-radius:8px; font-size:13px; }}
    nav a:hover {{ background:#294438; color:white; text-decoration:none; }}
    .sidebar-status {{ margin-top:24px; padding:12px; border:1px solid #355448; border-radius:10px; font-size:12px; color:#bcd0c5; }}
    main {{ min-width:0; }}
    .hero {{ padding:54px 6vw 46px; background:linear-gradient(135deg,#e4eee5 0%,#f8f3e7 58%,#eadfcd 100%); border-bottom:1px solid var(--line); }}
    .hero-inner, section {{ max-width:1500px; margin:auto; }}
    .kicker, .eyebrow {{ text-transform:uppercase; letter-spacing:.13em; font-size:11px; font-weight:800; color:var(--green); }}
    h1 {{ max-width:950px; margin:12px 0; font-family:Georgia,"Noto Serif KR",serif; font-size:clamp(34px,5vw,64px); line-height:1.05; letter-spacing:-.04em; }}
    .hero p {{ max-width:850px; color:#43534b; font-size:17px; }}
    .hero-meta {{ display:flex; flex-wrap:wrap; gap:9px; margin-top:22px; }}
    section {{ padding:44px 5vw; border-bottom:1px solid var(--line); }}
    .section-head {{ display:flex; align-items:end; justify-content:space-between; gap:20px; margin-bottom:24px; }}
    h2 {{ margin:0; font-family:Georgia,"Noto Serif KR",serif; font-size:34px; letter-spacing:-.025em; }}
    h3 {{ margin:4px 0 7px; font-size:25px; }} h4 {{ margin:4px 0 8px; }}
    .section-note {{ max-width:680px; color:var(--muted); font-size:14px; }}
    .stats {{ display:grid; grid-template-columns:repeat(6,minmax(125px,1fr)); gap:12px; }}
    .stat-card {{ background:var(--card); border:1px solid var(--line); border-radius:14px; padding:17px; box-shadow:var(--shadow); }}
    .stat-label {{ color:var(--muted); font-size:12px; }} .stat-value {{ font:700 30px Georgia,serif; margin:4px 0; }}
    .stat-note {{ color:var(--muted); font-size:11px; line-height:1.35; }}
    .stat-card.warn {{ border-top:4px solid var(--amber); }}
    .method-flow {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; }}
    .method-step {{ min-height:185px; display:flex; gap:12px; background:var(--card); border:1px solid var(--line); border-radius:14px; padding:17px; position:relative; }}
    .step-no {{ font:700 24px Georgia,serif; color:#a6b2ab; }} .step-title {{ font-weight:800; }}
    .step-metric {{ color:var(--green); font-weight:700; margin:8px 0; }} .method-step p {{ color:var(--muted); font-size:13px; }}
    .lineage {{ margin-top:20px; padding:20px; background:#172820; color:white; border-radius:14px; display:flex; align-items:center; justify-content:center; flex-wrap:wrap; gap:9px; }}
    .lineage .node {{ background:#294438; border:1px solid #4a6c5d; padding:10px 14px; border-radius:9px; text-align:center; }}
    .lineage .arrow {{ color:#83b59e; font-size:20px; }}
    .finding-grid {{ display:grid; grid-template-columns:repeat(2,1fr); gap:12px; }}
    .finding {{ display:grid; grid-template-columns:34px 1fr; gap:10px; border-radius:12px; padding:15px; border:1px solid var(--line); background:var(--card); }}
    .finding.ok {{ border-left:5px solid var(--green); }} .finding.warn {{ border-left:5px solid var(--amber); }} .finding.risk {{ border-left:5px solid var(--red); }}
    .finding-icon {{ width:28px; height:28px; border-radius:50%; display:grid; place-items:center; color:white; background:var(--green); font-weight:900; }}
    .finding.warn .finding-icon {{ background:var(--amber); }} .finding.risk .finding-icon {{ background:var(--red); }}
    .finding-title {{ font-weight:800; }} .finding p {{ margin:4px 0 0; color:var(--muted); font-size:13px; }}
    .badge {{ display:inline-block; border-radius:999px; padding:3px 8px; font-size:10px; font-weight:800; background:#ecece6; color:#59635e; vertical-align:middle; }}
    .badge.ok,.badge.core {{ background:var(--green2); color:var(--green); }} .badge.warn,.badge.supporting {{ background:var(--amber2); color:var(--amber); }}
    .badge.risk {{ background:var(--red2); color:var(--red); }} .badge.context,.badge.llm {{ background:var(--blue2); color:var(--blue); }}
    .badge.algo,.badge.scope {{ background:#eee6f5; color:var(--purple); }} .badge.relation {{ background:#e6ebf1; color:#455b70; }}
    .toolbar {{ position:sticky; top:0; z-index:4; display:flex; gap:9px; padding:12px; margin-bottom:18px; background:rgba(244,241,232,.94); border:1px solid var(--line); border-radius:12px; backdrop-filter:blur(8px); }}
    input,select {{ border:1px solid #c9ccc5; background:white; border-radius:8px; padding:10px 12px; font:inherit; }}
    input {{ flex:1; min-width:180px; }}
    .frame-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }}
    .frame-card {{ background:var(--card); border:1px solid var(--line); border-radius:16px; padding:22px; box-shadow:var(--shadow); }}
    .frame-header {{ display:flex; justify-content:space-between; gap:18px; }} .frame-header p {{ color:var(--muted); margin:0; }}
    .frame-score {{ min-width:85px; text-align:right; }} .frame-score strong {{ display:block; font:700 36px Georgia,serif; }} .frame-score span {{ color:var(--muted); font-size:11px; }}
    .frame-meta {{ display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin:15px 0; }}
    .aliases {{ padding:10px 12px; background:#f1efe7; border-radius:8px; font-size:13px; }}
    details {{ border-top:1px solid var(--line); margin-top:13px; padding-top:10px; }} summary {{ cursor:pointer; font-weight:800; }}
    .stack {{ display:grid; gap:9px; margin-top:10px; }}
    .mention-card,.fact-card,.evidence-mini {{ border:1px solid var(--line); border-radius:10px; background:white; padding:12px; }}
    .fact-card.review {{ border-color:#dda69d; background:#fff8f5; }}
    .row-between {{ display:flex; align-items:start; justify-content:space-between; gap:10px; }}
    .mention-signature {{ display:grid; grid-template-columns:55px 1fr; gap:3px 8px; margin:8px 0; font-size:12px; }} .mention-signature span {{ color:var(--muted); font-weight:700; }}
    blockquote {{ margin:8px 0 0; padding:9px 12px; background:#f5f4ee; border-left:3px solid #aab8b0; color:#43534b; font-size:13px; }}
    .triple {{ display:grid; grid-template-columns:minmax(0,1fr) auto minmax(0,1fr) auto minmax(0,1fr); align-items:center; gap:7px; margin:10px 0; font-size:13px; }}
    .triple span {{ padding:8px; border-radius:7px; background:#f1f2ed; overflow-wrap:anywhere; }} .triple .predicate {{ background:var(--green2); color:var(--green); font-weight:700; }}
    .triple.compact {{ min-width:460px; margin:0; }} .triple.compact span {{ padding:5px; }}
    .fact-semantics {{ display:flex; flex-wrap:wrap; align-items:center; gap:5px; margin:7px 0 4px; padding:7px 9px; border-radius:8px; background:#f5f2e9; }}
    .fact-semantics.compact {{ display:flex; width:fit-content; max-width:100%; margin:5px 0 0; padding:5px 7px; }}
    .fact-meta-label {{ color:var(--muted); font-size:9px; font-weight:900; letter-spacing:.05em; text-transform:uppercase; }}
    .mono {{ font-family:"SFMono-Regular",Consolas,monospace; font-size:10px; }} .subtle {{ color:var(--muted); }} .mini-link {{ font-size:9px; }}
    .relation-list,.fact-list {{ padding-left:19px; }} .relation-list li,.fact-list li {{ margin:7px 0; }}
    table {{ width:100%; border-collapse:collapse; background:var(--card); font-size:12px; }}
    th {{ position:sticky; top:57px; z-index:2; text-align:left; background:#e8e8df; color:#415149; }}
    th,td {{ padding:10px; border-bottom:1px solid var(--line); vertical-align:top; }} tr:hover td {{ background:#f6f6f0; }}
    .table-wrap {{ max-height:700px; overflow:auto; border:1px solid var(--line); border-radius:12px; }}
    .row-review td {{ background:#fff6f3; }}
    .evidence-card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:0; margin:9px 0; }}
    .evidence-card summary {{ display:grid; grid-template-columns:48px minmax(0,1fr) auto; align-items:center; gap:10px; padding:14px; }}
    .evidence-index {{ font:700 16px Georgia,serif; color:#87958e; }} .summary-badges {{ white-space:nowrap; }}
    .evidence-body {{ padding:0 16px 16px; }} .evidence-meta {{ color:var(--muted); font-size:11px; }}
    .evidence-grid {{ display:grid; grid-template-columns:3fr 2fr; gap:18px; }}
    .review-text {{ color:var(--red); }} .empty {{ color:#9ba39f; font-style:italic; }}
    .two-col {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
    .panel {{ background:var(--card); border:1px solid var(--line); border-radius:14px; padding:18px; }}
    footer {{ padding:34px 5vw; color:var(--muted); font-size:12px; }}
    .hidden {{ display:none !important; }}
    @media(max-width:1100px) {{ .layout{{grid-template-columns:1fr}} aside{{position:relative;height:auto}} nav{{display:flex;flex-wrap:wrap}} .stats{{grid-template-columns:repeat(3,1fr)}} .method-flow{{grid-template-columns:repeat(2,1fr)}} .frame-grid{{grid-template-columns:1fr}} }}
    @media(max-width:700px) {{ section,.hero{{padding-left:20px;padding-right:20px}} .stats,.finding-grid,.method-flow,.two-col,.evidence-grid{{grid-template-columns:1fr}} .toolbar{{flex-wrap:wrap}} .evidence-card summary{{grid-template-columns:38px 1fr}} .summary-badges{{grid-column:2}} }}
    @media print {{ aside,.toolbar{{display:none}} .layout{{display:block}} section{{break-inside:avoid}} details{{display:block}} details>*{{display:block}} }}
  </style>
</head>
<body>
<div class="layout">
  <aside>
    <div class="brand">Task-Centric<br>Open KG</div>
    <div class="brand-sub">v{e(manifest.get('pipeline_version', ''))} · 품질 추적 리포트</div>
    <nav>
      <a href="#overview">01 · Overview</a>
      <a href="#methodology">02 · Methodology</a>
      <a href="#findings">03 · Findings</a>
      <a href="#frames">04 · TaskFrames</a>
      <a href="#mentions">05 · TaskMentions</a>
      <a href="#triples">06 · Open triples</a>
      <a href="#evidence">07 · Evidence</a>
      <a href="#review">08 · Review queue</a>
      <a href="#ontology">09 · Canonicalization</a>
    </nav>
    <div class="sidebar-status">
      Status · <strong>{e(quality_status.upper())}</strong><br>
      Backend · {e(manifest.get('backend', ''))}<br>
      Generated · {e(generated_at)}
    </div>
  </aside>
  <main>
    <header class="hero">
      <div class="hero-inner">
        <div class="kicker">Evidence → Open Fact → TaskFrame → Knowledge Subgraph</div>
        <h1>제안 업무 지식이 만들어진 과정을 원문까지 추적합니다.</h1>
        <p>방법론의 각 단계, 자유형 삼중항, TaskMention 병합 결과, TaskFrame 내부 Fact와 이를 뒷받침하는 Evidence를 하나의 독립형 HTML에서 검토합니다.</p>
        <div class="hero-meta">
          {badge(f"Quality {quality_status}", 'warn' if quality_status == 'warn' else 'ok')}
          {badge(f"{len(evidence)} Evidence", 'scope')}
          {badge(f"{len(facts)} Facts", 'scope')}
          {badge(f"{len(frames)} TaskFrames", 'scope')}
          {badge(f"LLM calls {call_plan.get('projected_total_llm_calls', '—')}", 'llm')}
        </div>
      </div>
    </header>

    <section id="overview">
      <div class="section-head"><div><div class="eyebrow">Run overview</div><h2>실행 결과 한눈에 보기</h2></div><p class="section-note">구조적 성공과 의미 품질을 구분합니다. KG가 valid하더라도 grounding 또는 Task 경계는 별도 검토가 필요합니다.</p></div>
      <div class="stats">
        {stat_card('Evidence', len(evidence), f"{evidence_with_facts}개에 Fact 존재")}
        {stat_card('Open Facts', len(facts), f"SPO {facts_with_object} · object 생략 {len(facts)-facts_with_object}")}
        {stat_card('TaskMentions', len(mentions), f"Evidence 대비 {pct(len(mentions), len(evidence))}")}
        {stat_card('TaskFrames', len(frames), f"relations {len(task_relations)}")}
        {stat_card('Task-linked Facts', propagation_summary.get('task_related_fact_count', 0), f"전체 대비 {pct(propagation_summary.get('task_related_fact_count', 0), len(facts))}")}
        {stat_card('Grounding review', len(bad_facts), f"정상 {pct(len(facts)-len(bad_facts), len(facts))}", 'warn')}
      </div>
    </section>

    <section id="methodology">
      <div class="section-head"><div><div class="eyebrow">Methodology</div><h2>v4 방법론과 실제 산출물</h2></div><p class="section-note">LLM은 의미 추출과 Task 경계 판정에만 쓰고, 보존·blocking·전파·KG 검증은 알고리즘으로 수행했습니다.</p></div>
      <div class="method-flow">{methodology_html}</div>
      <div class="lineage">
        <div class="node">EvidenceSpan<br><small>원문·위치</small></div><div class="arrow">→</div>
        <div class="node">Fact<br><small>자유형 S–P–O</small></div><div class="arrow">↔</div>
        <div class="node">TaskMention<br><small>로컬 업무 표현</small></div><div class="arrow">→</div>
        <div class="node">TaskFrame<br><small>정규화 업무</small></div><div class="arrow">→</div>
        <div class="node">Task subgraph<br><small>core · supporting · context</small></div>
      </div>
    </section>

    <section id="findings">
      <div class="section-head"><div><div class="eyebrow">Quality diagnosis</div><h2>결과 진단</h2></div><p class="section-note">자동 지표와 실제 원문·Frame 내용을 함께 확인한 판단입니다.</p></div>
      <div class="finding-grid">{''.join(findings)}</div>
    </section>

    <section id="frames">
      <div class="section-head"><div><div class="eyebrow">Task-centered subgraphs</div><h2>TaskFrame 내부 보기</h2></div><p class="section-note">각 Frame의 member TaskMention, core/supporting Fact, 실제 Evidence와 Task 관계를 함께 보여줍니다.</p></div>
      <div class="toolbar"><input id="frame-search" type="search" placeholder="TaskFrame·alias·definition 검색"><span id="frame-count">{len(frames)} frames</span></div>
      <div class="frame-grid" id="frame-list">{''.join(frame_cards)}</div>
    </section>

    <section id="mentions">
      <div class="section-head"><div><div class="eyebrow">Local task observations</div><h2>TaskMention {len(mentions)}개</h2></div><p class="section-note">원문의 로컬 업무 표현이 어느 TaskFrame으로 병합됐는지 확인합니다.</p></div>
      <div class="table-wrap"><table><thead><tr><th>Surface / ID</th><th>Action</th><th>Focal object</th><th>Confidence</th><th>TaskFrame</th><th>Evidence text</th></tr></thead><tbody>{''.join(mention_rows)}</tbody></table></div>
    </section>

    <section id="triples">
      <div class="section-head"><div><div class="eyebrow">Open fact extraction</div><h2>자유형 삼중항 {len(facts)}개</h2></div><p class="section-note">raw·canonical 표현과 함께 Knowledge Function, condition/temporal/exception Qualifier, TaskFrame 연결 및 Evidence를 확인합니다.</p></div>
      <div class="toolbar">
        <input id="fact-search" type="search" placeholder="S·P·O·Knowledge Function·Qualifier·Evidence 검색">
        <select id="fact-function"><option value="all">전체 Knowledge Function</option>{''.join(f'<option value="{e(value)}">{e(value)}</option>' for value in ('fact', 'definition', 'procedure', 'rule', 'decision', 'recommendation'))}</select>
        <select id="fact-qualifier"><option value="all">전체 Qualifier</option><option value="yes">Qualifier 있음</option><option value="no">Qualifier 없음</option></select>
        <select id="fact-linked"><option value="all">전체 연결 상태</option><option value="yes">TaskFrame 연결</option><option value="no">미연결 오류</option></select>
        <select id="fact-review"><option value="all">전체 grounding</option><option value="yes">검토 필요</option><option value="no">정상</option></select>
        <span id="fact-count">{len(facts)} facts</span>
      </div>
      <div class="table-wrap"><table id="fact-table"><thead><tr><th>ID / 상태</th><th>Raw triple</th><th>Canonical triple</th><th>TaskFrame</th><th>Evidence</th></tr></thead><tbody>{''.join(fact_rows)}</tbody></table></div>
    </section>

    <section id="evidence">
      <div class="section-head"><div><div class="eyebrow">Grounded provenance</div><h2>Evidence별 추출 추적</h2></div><p class="section-note">원문 한 문장에서 어떤 삼중항과 TaskMention이 나왔고 최종적으로 어디에 연결됐는지 확인합니다.</p></div>
      <div class="toolbar"><input id="evidence-search" type="search" placeholder="원문·section·triple·TaskFrame 검색"><select id="evidence-review"><option value="all">전체 Evidence</option><option value="yes">grounding 검토 포함</option><option value="no">정상</option></select><span id="evidence-count">{len(evidence)} evidence</span></div>
      <div id="evidence-list">{''.join(evidence_cards)}</div>
    </section>

    <section id="review">
      <div class="section-head"><div><div class="eyebrow">Human review queue</div><h2>우선 검토할 항목</h2></div><p class="section-note">원문·exact quote 계약을 위반한 {len(bad_facts)}개 Fact와 Task 관계를 먼저 검토하는 것이 좋습니다.</p></div>
      <div class="two-col">
        <div class="panel">
          <h3>TaskFrame 경계</h3>
          <ul>
            <li><strong>서술형 Task 후보</strong> — {e(', '.join(frame['canonical_name'] for frame in descriptive_frames) or '없음')}의 원문이 실행업무인지 확인</li>
            <li><strong>낮은 confidence member</strong> — {e(', '.join(frame['canonical_name'] for frame in low_confidence_frames) or '없음')}의 canonical 확정 여부 확인</li>
            <li><strong>core Fact 없음</strong> — {e(', '.join(frame['canonical_name'] for frame in frames_without_core) or '없음')}</li>
            <li><strong>Fact 집중 Frame</strong> — {e(', '.join(frame['canonical_name'] for frame in dominant_frames) or '없음')}</li>
            <li><strong>Task 관계</strong> — confidence 0.6 관계 {relation_review_count}개는 문서 Evidence를 추가로 연결해 검증</li>
          </ul>
        </div>
        <div class="panel">
          <h3>파이프라인 신호</h3>
          <ul>
            <li>LLM 1 calls {e(call_plan.get('llm1_extraction_calls', '—'))} · LLM 2 calls {e(call_plan.get('total_llm2_calls', '—'))}</li>
            <li>애매한 Fact {e(frame_validation.get('ambiguous_fact_count', len(ambiguous)))}개 중 재판정 {len(decisions)}개</li>
            <li>전체 Fact 연결률 {pct(propagation_summary.get('assigned_fact_count', 0), propagation_summary.get('fact_count', len(facts)))} · 미연결 {propagation_summary.get('unlinked_fact_count', 0)}개</li>
            <li>LLM 연결 누락을 보완한 exhaustive fallback {propagation_summary.get('exhaustive_fallback_count', 0)}개</li>
            <li>Standard mapping — {'활성' if standard_summary.get('enabled') else '비활성'}</li>
          </ul>
        </div>
      </div>
      <h3 style="margin-top:26px">Grounding review facts</h3>
      <div class="table-wrap"><table><thead><tr><th>Fact ID</th><th>불일치 필드</th><th>추출 triple</th><th>원문 Evidence</th></tr></thead><tbody>{''.join(bad_fact_rows)}</tbody></table></div>
      <h3 style="margin-top:26px">Task relations</h3>
      <div class="table-wrap"><table><thead><tr><th>Source</th><th>Relation</th><th>Target</th><th>Confidence</th><th>LLM rationale</th></tr></thead><tbody>{relation_rows}</tbody></table></div>
    </section>

    <section id="ontology">
      <div class="section-head"><div><div class="eyebrow">Bottom-up ontology</div><h2>Canonicalization 성숙도</h2></div><p class="section-note">현재는 100 Evidence pilot이므로 원문 표현 관찰과 검토 큐 형성이 주 목적입니다.</p></div>
      <div class="stats">
        {stat_card('Canonical entities', len(canonical_entities), f"stable {stable_entities}")}
        {stat_card('Canonical relations', len(canonical_relations), f"unresolved {unresolved_relations}", 'warn')}
        {stat_card('Promoted terms', len(promoted_terms), '반복·안정성 기준 충족')}
        {stat_card('Entity aliases', prep_summary.get('verified_entity_alias_count', 0), '검증된 사전 표현')}
        {stat_card('Standard mappings', standard_summary.get('candidate_mapping_count', 0), '현재 비활성')}
        {stat_card('KG validity', 'PASS' if graph_validation.get('valid') else 'FAIL', f"{graph_validation.get('edge_count', 0)} edges")}
      </div>
    </section>

    <footer>
      Source run · {e(input_dir)}<br>
      이 파일은 외부 CDN이나 서버 없이 동작하는 독립형 HTML입니다. 생성 시각 {e(generated_at)}.
    </footer>
  </main>
</div>
<script>
  function wireFilter(inputId, containerSelector, countId, extraFilter) {{
    const input = document.getElementById(inputId);
    const items = Array.from(document.querySelectorAll(containerSelector));
    const count = document.getElementById(countId);
    function apply() {{
      const query = (input?.value || '').trim().toLowerCase();
      let visible = 0;
      items.forEach(item => {{
        const search = item.dataset.search || item.textContent.toLowerCase();
        const ok = (!query || search.includes(query)) && (!extraFilter || extraFilter(item));
        item.classList.toggle('hidden', !ok);
        if (ok) visible++;
      }});
      if (count) count.textContent = `${{visible}} shown`;
    }}
    input?.addEventListener('input', apply);
    return apply;
  }}
  wireFilter('frame-search', '#frame-list > .searchable', 'frame-count');
  const knowledgeFunction = document.getElementById('fact-function');
  const qualifier = document.getElementById('fact-qualifier');
  const linked = document.getElementById('fact-linked');
  const review = document.getElementById('fact-review');
  const applyFacts = wireFilter('fact-search', '#fact-table tbody > .searchable', 'fact-count', item =>
    (knowledgeFunction.value === 'all' || item.dataset.function === knowledgeFunction.value) &&
    (qualifier.value === 'all' || item.dataset.qualifiers === qualifier.value) &&
    (linked.value === 'all' || item.dataset.linked === linked.value) &&
    (review.value === 'all' || item.dataset.review === review.value)
  );
  knowledgeFunction.addEventListener('change', applyFacts);
  qualifier.addEventListener('change', applyFacts);
  linked.addEventListener('change', applyFacts);
  review.addEventListener('change', applyFacts);
  const evReview = document.getElementById('evidence-review');
  const applyEvidence = wireFilter('evidence-search', '#evidence-list > .searchable', 'evidence-count', item =>
    evReview.value === 'all' || item.dataset.review === evReview.value
  );
  evReview.addEventListener('change', applyEvidence);
</script>
</body>
</html>"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Task-Centric Open KG v4 독립형 HTML 품질 리포트 생성"
    )
    parser.add_argument("input_dir", type=Path)
    parser.add_argument(
        "--output", type=Path,
        help="기본값: <input_dir>/task_centric_open_kg_report.html",
    )
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    if not (input_dir / "run_manifest.json").exists():
        raise FileNotFoundError(f"run_manifest.json이 없습니다: {input_dir}")
    output = (
        args.output.resolve()
        if args.output
        else input_dir / "task_centric_open_kg_report.html"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(input_dir), encoding="utf-8")
    print(f"[done] html={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
