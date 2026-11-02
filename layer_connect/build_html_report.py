from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import load_config, output_path, read_jsonl, write_json


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def _json_for_script(value: Any) -> str:
    # Source text에 HTML/script 조각이 있어도 inline JSON 경계를 벗어나지 않게 한다.
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def build_html_report(config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    output_dir = output_path(config, "placeholder").parent
    nodes = read_jsonl(output_dir / "pcf_nodes_enriched.jsonl")
    packets = read_jsonl(output_dir / "taskframe_evidence_packets.jsonl")
    candidates = read_jsonl(output_dir / "taskframe_pcf_candidates.jsonl")
    links = read_jsonl(output_dir / "taskframe_pcf_links.jsonl")
    review_queue = read_jsonl(output_dir / "review_queue.jsonl")
    validation = _read_optional_json(output_dir / "validation_report.json")
    retrieval = _read_optional_json(output_dir / "retrieval_summary.json")
    aggregation = _read_optional_json(output_dir / "aggregation_summary.json")
    document_context = _read_optional_json(output_dir / "document_pcf_context.json")

    if not nodes or not packets:
        raise ValueError("HTML 생성 전에 PCF index와 Evidence Packet이 필요합니다.")

    candidates_by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        candidates_by_frame[item["taskframe_id"]].append(item)
    for values in candidates_by_frame.values():
        values.sort(key=lambda item: int(item.get("rank", 10**9)))

    links_by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in links:
        links_by_frame[item["taskframe_id"]].append(item)
    review_by_frame = {item["taskframe_id"]: item for item in review_queue}

    frames: list[dict[str, Any]] = []
    for packet in packets:
        frame_id = packet["taskframe_id"]
        frame_links = links_by_frame.get(frame_id, [])
        primary = next(
            (
                item
                for item in frame_links
                if item.get("mapping_role") == "process_match" and item.get("is_primary")
            ),
            None,
        )
        unmapped = next(
            (item for item in frame_links if item.get("relation_scope") == "unmapped"),
            None,
        )
        process_links = [
            item
            for item in frame_links
            if item.get("mapping_role") == "process_match"
            and item.get("relation_scope") != "unmapped"
        ]
        context_links = [
            item for item in frame_links if item.get("mapping_role") == "context_anchor"
        ]
        top_candidates = candidates_by_frame.get(frame_id, [])[:5]
        summary = packet.get("frame_summary", {})
        mention_samples = [
            {
                "id": item.get("task_mention_id"),
                "text": item.get("surface_form", ""),
                "section": item.get("section_path", ""),
            }
            for item in packet.get("mentions", [])[:4]
        ]
        fact_samples = [
            {
                "id": item.get("triple_id"),
                "text": item.get("spo_text", ""),
                "relevance": item.get("relevance", key.removesuffix("_facts")),
            }
            for key in ("core_facts", "supporting_facts")
            for item in packet.get(key, [])[:2]
        ][:5]
        frames.append(
            {
                "taskframe_id": frame_id,
                "name": summary.get("canonical_name", ""),
                "definition": summary.get("definition", ""),
                "scope": summary.get("scope", ""),
                "source_counts": packet.get("source_counts", {}),
                "selected_counts": packet.get("selected_counts", {}),
                "primary": primary,
                "unmapped": unmapped,
                "process_links": process_links,
                "context_links": context_links,
                "candidates": top_candidates,
                "review": review_by_frame.get(frame_id),
                "mention_samples": mention_samples,
                "fact_samples": fact_samples,
            }
        )

    node_payload = [
        {
            "hid": item["hid"],
            "parent": item.get("parent"),
            "level": item.get("level"),
            "name": item.get("nameKo") or item.get("nameEn", ""),
            "path_ids": item.get("path_ids", []),
            "path": item.get("path_text_ko") or item.get("path_text_en", ""),
        }
        for item in nodes
    ]
    roots = [item for item in node_payload if not item["parent"]]
    roots.sort(key=lambda item: int(item["hid"].split(".")[0]))
    generated_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "generated_at": generated_at,
        "nodes": node_payload,
        "roots": roots,
        "frames": frames,
        "summary": {
            "pcf_nodes": len(nodes),
            "taskframes": len(frames),
            "primary_candidates": sum(frame["primary"] is not None for frame in frames),
            "unmapped": sum(frame["primary"] is None for frame in frames),
            "needs_review": sum(frame["review"] is not None for frame in frames),
            "auto_accepted": sum(
                bool(frame["primary"])
                and frame["primary"].get("validation_status") == "auto_accepted"
                for frame in frames
            ),
            "validation_errors": validation.get("error_count", 0),
            "validation_warnings": validation.get("warning_count", 0),
            "candidate_rows": retrieval.get("candidate_record_count", len(candidates)),
            "retrieval_backend": retrieval.get("retrieval_backend", ""),
            "context_anchors": aggregation.get("context_anchor_count", 0),
            "document_context_hid": document_context.get("selected_pcf_hid", ""),
            "document_context_name": document_context.get("selected_pcf_name_ko", ""),
            "document_context_score": document_context.get("selected_document_score", 0.0),
            "document_context_margin": document_context.get("score_margin", 0.0),
        },
    }

    html = HTML_TEMPLATE.replace("__CROSSWALK_DATA__", _json_for_script(payload)).replace(
        "__GENERATED_AT__", generated_at
    )
    html_path = output_dir / "pcf_taskframe_crosswalk.html"
    html_path.write_text(html, encoding="utf-8")
    report = {
        "html": str(html_path),
        "bytes": html_path.stat().st_size,
        "pcf_nodes": len(nodes),
        "taskframes": len(frames),
        "primary_candidates": payload["summary"]["primary_candidates"],
        "unmapped": payload["summary"]["unmapped"],
        "generated_at": generated_at,
    }
    write_json(output_dir / "html_report_summary.json", report)
    return html_path, report


HTML_TEMPLATE = r'''<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PCF ↔ TaskFrame Crosswalk</title>
  <style>
    :root {
      --ink: #17233b; --muted: #64748b; --line: #dbe4ef; --paper: #ffffff;
      --canvas: #f4f7fb; --pcf: #0f766e; --pcf-soft: #e6f5f2; --task: #3156a3;
      --task-soft: #edf2ff; --warn: #b45309; --warn-soft: #fff7e6; --danger: #b42318;
      --shadow: 0 14px 35px rgba(23,35,59,.08); --radius: 18px;
    }
    * { box-sizing: border-box; }
    body { margin: 0; color: var(--ink); background: var(--canvas); font-family: Inter, Pretendard, "Noto Sans KR", system-ui, sans-serif; }
    button, input, select { font: inherit; }
    .shell { max-width: 1540px; margin: 0 auto; padding: 30px 26px 70px; }
    .hero { color: white; padding: 30px 34px; border-radius: 24px; background: radial-gradient(circle at 85% 10%, #3f7f87 0, transparent 32%), linear-gradient(125deg,#17233b,#24406d 58%,#0f766e); box-shadow: var(--shadow); }
    .eyebrow { margin: 0 0 7px; color: #9ee8db; font-size: 12px; letter-spacing: .12em; font-weight: 800; }
    h1 { margin: 0; font-size: clamp(25px,3vw,40px); letter-spacing: -.035em; }
    .hero p { margin: 10px 0 0; max-width: 860px; color: #dbeafe; line-height: 1.65; }
    .hero-meta { margin-top: 18px; display: flex; flex-wrap: wrap; gap: 8px; }
    .hero-meta span { padding: 6px 10px; border: 1px solid rgba(255,255,255,.18); border-radius: 999px; background: rgba(255,255,255,.08); font-size: 12px; }
    .kpis { display: grid; grid-template-columns: repeat(6,minmax(130px,1fr)); gap: 12px; margin: 18px 0; }
    .kpi { padding: 17px 18px; border-radius: 16px; background: var(--paper); border: 1px solid var(--line); box-shadow: 0 8px 22px rgba(23,35,59,.04); }
    .kpi .label { color: var(--muted); font-size: 12px; font-weight: 700; }
    .kpi .value { margin-top: 4px; font-size: 27px; font-weight: 850; letter-spacing: -.03em; }
    .kpi.good .value { color: var(--pcf); } .kpi.warn .value { color: var(--warn); }
    .notice { display: flex; gap: 12px; align-items: flex-start; margin: 18px 0; padding: 14px 17px; background: var(--warn-soft); border: 1px solid #f5d39a; border-radius: 14px; color: #78420c; line-height: 1.55; font-size: 13px; }
    .notice strong { white-space: nowrap; }
    .panel { margin-top: 18px; padding: 22px; border: 1px solid var(--line); border-radius: var(--radius); background: var(--paper); box-shadow: var(--shadow); }
    .panel-head { display: flex; justify-content: space-between; align-items: end; gap: 18px; margin-bottom: 16px; }
    .panel-head h2 { margin: 0; font-size: 20px; letter-spacing: -.02em; }
    .panel-head p { margin: 5px 0 0; color: var(--muted); font-size: 13px; }
    .controls { display: flex; flex-wrap: wrap; gap: 9px; }
    .control { min-height: 38px; padding: 8px 11px; border: 1px solid var(--line); border-radius: 10px; color: var(--ink); background: #fff; }
    input.control { min-width: 240px; }
    .legend { display: flex; flex-wrap: wrap; gap: 12px; color: var(--muted); font-size: 12px; }
    .legend span::before { content: ""; display: inline-block; width: 9px; height: 9px; margin-right: 6px; border-radius: 50%; vertical-align: -1px; }
    .legend .candidate::before { background: var(--task); } .legend .suggestion::before { background: var(--warn); }
    .mapping-list { display: grid; gap: 12px; }
    .mapping-card { overflow: hidden; border: 1px solid var(--line); border-radius: 15px; background: #fff; }
    .mapping-card.hidden { display: none; }
    .mapping-main { display: grid; grid-template-columns: minmax(270px,1fr) 140px minmax(330px,1.15fr); align-items: stretch; min-height: 138px; }
    .layer-box { padding: 18px 19px; }
    .layer-box.task { background: linear-gradient(135deg,var(--task-soft),#fff); border-right: 1px solid var(--line); }
    .layer-box.pcf { background: linear-gradient(225deg,var(--pcf-soft),#fff); border-left: 1px solid var(--line); }
    .layer-box.empty { background: linear-gradient(225deg,var(--warn-soft),#fff); }
    .layer-label { font-size: 10px; letter-spacing: .1em; font-weight: 850; color: var(--muted); text-transform: uppercase; }
    .layer-title { margin-top: 7px; font-size: 17px; font-weight: 820; line-height: 1.35; }
    .layer-code { margin-top: 5px; color: var(--muted); font: 11px ui-monospace, SFMono-Regular, monospace; }
    .definition { margin-top: 9px; color: #49566c; font-size: 12px; line-height: 1.55; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
    .counts { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
    .counts span, .chip { padding: 4px 7px; border-radius: 7px; background: rgba(255,255,255,.82); border: 1px solid var(--line); color: var(--muted); font-size: 10px; font-weight: 700; }
    .connector { position: relative; display: grid; place-items: center; padding: 15px 8px; text-align: center; background: #fbfcfe; }
    .connector::before { content: ""; position: absolute; left: 10px; right: 10px; top: 50%; height: 2px; background: var(--line); }
    .connector::after { content: ""; position: absolute; right: 9px; top: calc(50% - 4px); border-left: 7px solid var(--line); border-top: 5px solid transparent; border-bottom: 5px solid transparent; }
    .status { position: relative; z-index: 1; padding: 6px 9px; border-radius: 999px; font-size: 10px; font-weight: 850; border: 1px solid; background: white; }
    .status.mapped { color: var(--task); border-color: #b9c8ef; } .status.unmapped { color: var(--warn); border-color: #efc77c; }
    .path { margin-top: 9px; color: #45615e; font-size: 11px; line-height: 1.5; }
    .score-row { display: flex; align-items: center; gap: 8px; margin-top: 10px; color: var(--muted); font-size: 10px; }
    .bar { flex: 1; max-width: 130px; height: 6px; overflow: hidden; border-radius: 99px; background: #e8edf3; }
    .bar i { display: block; height: 100%; background: var(--pcf); border-radius: inherit; }
    .candidate-hint { margin-top: 9px; padding: 8px 9px; border-left: 3px solid #e2a33b; background: #fffaf0; color: #70430c; font-size: 11px; line-height: 1.45; }
    details.card-detail { border-top: 1px solid var(--line); }
    details.card-detail > summary { cursor: pointer; padding: 10px 18px; color: var(--muted); font-size: 11px; font-weight: 750; list-style: none; }
    details.card-detail > summary::-webkit-details-marker { display: none; }
    details.card-detail > summary::before { content: "+"; display: inline-block; width: 18px; color: var(--task); font-size: 15px; }
    details.card-detail[open] > summary::before { content: "−"; }
    .detail-grid { display: grid; grid-template-columns: 1.35fr 1fr; gap: 18px; padding: 0 18px 18px; }
    .subhead { margin: 4px 0 8px; font-size: 11px; color: var(--muted); letter-spacing: .05em; font-weight: 850; }
    table { width: 100%; border-collapse: collapse; font-size: 11px; }
    th, td { padding: 8px 7px; border-bottom: 1px solid #edf1f5; text-align: left; vertical-align: top; }
    th { color: var(--muted); font-weight: 750; background: #f8fafc; }
    .candidate-row.selected { background: var(--task-soft); }
    .reason-list { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }
    .reason { padding: 5px 7px; border-radius: 7px; color: #8a4d0a; background: var(--warn-soft); border: 1px solid #f1d39d; font-size: 10px; }
    .evidence { margin: 0; padding-left: 17px; color: #4c5b70; font-size: 11px; line-height: 1.55; }
    .tree-toolbar { display: flex; justify-content: space-between; gap: 15px; align-items: center; margin-bottom: 12px; }
    .toggle { display: inline-flex; gap: 7px; align-items: center; color: var(--muted); font-size: 12px; }
    .tree { max-height: 760px; overflow: auto; padding: 7px 12px 15px; border: 1px solid var(--line); border-radius: 12px; background: #fbfcfe; }
    .tree details { margin-left: 18px; }
    .tree > details { margin-left: 0; }
    .tree summary { cursor: pointer; padding: 5px 4px; border-radius: 7px; list-style-position: outside; font-size: 12px; }
    .tree summary:hover { background: #eef3f8; }
    .hid { display: inline-block; min-width: 72px; color: var(--pcf); font: 10px ui-monospace, SFMono-Regular, monospace; font-weight: 750; }
    .tf-badges { display: inline-flex; flex-wrap: wrap; gap: 4px; margin-left: 7px; vertical-align: middle; }
    .tf-badge { padding: 3px 6px; border-radius: 999px; color: white; background: var(--task); font-size: 9px; font-weight: 750; }
    .tf-badge.suggestion { color: #7a4306; background: #ffe4ad; border: 1px dashed #d69628; }
    .empty-state { padding: 34px; color: var(--muted); text-align: center; border: 1px dashed var(--line); border-radius: 12px; }
    .footer { margin-top: 20px; text-align: center; color: var(--muted); font-size: 11px; }
    @media (max-width: 1050px) { .kpis { grid-template-columns: repeat(3,1fr); } .mapping-main { grid-template-columns: 1fr; } .connector { min-height: 62px; } .connector::before { top: 8px; bottom: 8px; left: 50%; right: auto; width: 2px; height: auto; } .connector::after { right: auto; left: calc(50% - 4px); top: auto; bottom: 6px; border-left: 5px solid transparent; border-right: 5px solid transparent; border-top: 7px solid var(--line); border-bottom: 0; } .layer-box.task,.layer-box.pcf { border: 0; } }
    @media (max-width: 700px) { .shell { padding: 14px 10px 45px; } .hero,.panel { padding: 18px; border-radius: 16px; } .kpis { grid-template-columns: repeat(2,1fr); } .panel-head,.tree-toolbar { align-items: stretch; flex-direction: column; } input.control { min-width: 0; width: 100%; } .detail-grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <p class="eyebrow">LAYER 1 ↔ LAYER 2</p>
      <h1>PCF–TaskFrame Crosswalk</h1>
      <p>TaskFrame이 PCF의 어느 경로에 연결 후보로 놓였는지, 아직 결정되지 않은 항목은 무엇인지 한 화면에서 확인합니다.</p>
      <div class="hero-meta"><span id="generated">생성: __GENERATED_AT__</span><span id="docContext"></span><span id="backend"></span><span id="health"></span></div>
    </section>
    <section class="kpis" id="kpis"></section>
    <div class="notice"><strong>판정 상태 안내</strong><span>문서 전체의 TaskFrame 분포로 선택한 PCF subtree를 기본 검색 범위로 사용합니다. 각 TaskFrame은 subtree 내부 최적 후보와 PCF 전체의 외부 최적 후보를 비교하며, 외부 후보는 충분한 reranker 차이와 핵심 근거가 있을 때만 예외로 선택됩니다. <code>needs_review</code>는 확정 연결이 아닙니다.</span></div>

    <section class="panel">
      <div class="panel-head">
        <div><h2>TaskFrame → PCF 연결 현황</h2><p>검색과 필터로 업무별 PCF 경로, 후보 점수, 검토 사유 및 대표 근거를 확인할 수 있습니다.</p></div>
        <div class="controls">
          <input id="search" class="control" type="search" placeholder="TaskFrame, PCF, 코드 검색">
          <select id="statusFilter" class="control"><option value="all">전체 상태</option><option value="mapped">연결 후보</option><option value="unmapped">미매핑</option><option value="review">검토 필요</option></select>
          <select id="rootFilter" class="control"><option value="all">전체 PCF Root</option></select>
        </div>
      </div>
      <div class="legend"><span class="candidate">Crosswalk 연결 후보</span><span class="suggestion">미매핑 항목의 검색 1위</span></div>
      <div class="mapping-list" id="mappingList"></div>
      <div class="empty-state" id="cardEmpty" hidden>조건에 맞는 TaskFrame이 없습니다.</div>
    </section>

    <section class="panel">
      <div class="panel-head"><div><h2>PCF 계층에서 보기</h2><p>파란 배지는 현재 연결 후보, 주황 점선 배지는 미매핑 항목의 1위 제안 위치입니다.</p></div></div>
      <div class="tree-toolbar">
        <input id="treeSearch" class="control" type="search" placeholder="PCF 이름 또는 hid 검색">
        <label class="toggle"><input id="linkedOnly" type="checkbox" checked> TaskFrame 관련 경로만 보기</label>
      </div>
      <div class="tree" id="tree"></div>
    </section>
    <p class="footer">이 HTML은 외부 라이브러리나 네트워크 없이 단독으로 열 수 있습니다.</p>
  </main>
  <script id="crosswalk-data" type="application/json">__CROSSWALK_DATA__</script>
  <script>
  (() => {
    const DATA = JSON.parse(document.getElementById('crosswalk-data').textContent);
    const nodes = new Map(DATA.nodes.map(n => [n.hid,n]));
    const children = new Map();
    DATA.nodes.forEach(n => { const key=n.parent || '__root__'; if(!children.has(key)) children.set(key,[]); children.get(key).push(n); });
    children.forEach(list => list.sort((a,b) => a.hid.localeCompare(b.hid,undefined,{numeric:true})));
    const esc = value => String(value ?? '').replace(/[&<>'"]/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[ch]));
    const pct = value => `${(Number(value || 0)*100).toFixed(1)}%`;
    const reasonKo = {
      low_score:'낮은 종합 점수', low_margin:'1·2위 점수 차이 부족', low_evidence_coverage:'근거 coverage 부족',
      frame_vs_evidence_conflict:'Frame 요약과 원자 근거 충돌', cross_root_conflict:'서로 다른 PCF root 충돌',
      single_mention_taskframe:'단일 mention 프레임', possible_taskframe_split:'TaskFrame 분리 검토',
      multiple_process_branches:'복수 process 가지', context_only_support:'context 근거만 존재',
      context_anchor_candidate:'문맥 anchor 후보', no_candidates:'후보 없음'
    };
    const rootOf = frame => {
      const hid = frame.primary?.pcf_hid || frame.candidates?.[0]?.pcf_hid;
      const node = nodes.get(hid); return node?.path_ids?.[0] || '';
    };
    const summary = DATA.summary;
    document.getElementById('docContext').textContent = `문서 문맥: ${summary.document_context_hid || '-'} ${summary.document_context_name || ''}`;
    document.getElementById('backend').textContent = `검색: ${summary.retrieval_backend || 'unknown'}`;
    document.getElementById('health').textContent = `검증: 오류 ${summary.validation_errors} · 경고 ${summary.validation_warnings}`;
    const kpis = [
      ['PCF 노드',summary.pcf_nodes,''],['TaskFrame',summary.taskframes,''],['연결 후보',summary.primary_candidates,'good'],
      ['미매핑',summary.unmapped,'warn'],['검토 필요',summary.needs_review,'warn'],['자동 승인',summary.auto_accepted,'good']
    ];
    document.getElementById('kpis').innerHTML = kpis.map(([label,value,cls]) => `<article class="kpi ${cls}"><div class="label">${esc(label)}</div><div class="value">${esc(value)}</div></article>`).join('');

    const rootSelect = document.getElementById('rootFilter');
    DATA.roots.forEach(root => rootSelect.insertAdjacentHTML('beforeend',`<option value="${esc(root.hid)}">${esc(root.hid)} ${esc(root.name)}</option>`));
    const candidateTable = (frame) => {
      if(!frame.candidates.length) return '<div class="empty-state">후보가 없습니다.</div>';
      return `<table><thead><tr><th>#</th><th>PCF 후보</th><th>종합</th><th>근거</th><th>경로</th></tr></thead><tbody>${frame.candidates.map(c => `<tr class="candidate-row ${frame.primary?.pcf_hid===c.pcf_hid?'selected':''}"><td>${esc(c.rank)}</td><td><b>${esc(c.pcf_hid)}</b><br>${esc(c.pcf_name_ko)}</td><td>${pct(c.score)}</td><td>${pct(c.support_coverage)}</td><td>${pct(c.path_score)}</td></tr>`).join('')}</tbody></table>`;
    };
    const cardHtml = frame => {
      const mapped = Boolean(frame.primary);
      const target = frame.primary;
      const suggested = frame.candidates[0];
      const counts = frame.source_counts || {};
      const reasons = frame.review?.review_reasons || [];
      const targetHtml = mapped ? `
        <div class="layer-label">Layer 1 · PCF</div><div class="layer-title">${esc(target.pcf_hid)} ${esc(target.pcf_name_ko)}</div>
        <div class="path">${esc((target.pcf_path_ids||[]).map(h => `${h} ${nodes.get(h)?.name||''}`).join(' › '))}</div>
        <div class="score-row"><span>후보 점수 ${pct(target.candidate_score)}</span><span class="bar"><i style="width:${Math.min(100,Number(target.candidate_score||0)*100)}%"></i></span><span>coverage ${pct(target.coverage_score)}</span></div>` : `
        <div class="layer-label">Layer 1 · PCF</div><div class="layer-title">아직 연결되지 않음</div>
        <div class="layer-code">relation_scope: unmapped</div>
        ${suggested ? `<div class="candidate-hint"><b>검색 1위 제안</b><br>${esc(suggested.pcf_hid)} ${esc(suggested.pcf_name_ko)} · 점수 ${pct(suggested.score)}</div>` : ''}`;
      const evidence = [...frame.mention_samples.map(x=>`<li><b>${esc(x.id)}</b> ${esc(x.text)}</li>`),...frame.fact_samples.map(x=>`<li><b>${esc(x.id)}</b> ${esc(x.text)}</li>`)].join('');
      return `<article class="mapping-card" data-id="${esc(frame.taskframe_id)}" data-status="${mapped?'mapped':'unmapped'}" data-root="${esc(rootOf(frame))}">
        <div class="mapping-main">
          <div class="layer-box task"><div class="layer-label">Layer 2 · TaskFrame</div><div class="layer-title">${esc(frame.name)}</div><div class="layer-code">${esc(frame.taskframe_id)}</div><div class="definition">${esc(frame.definition)}</div><div class="counts"><span>Mention ${esc(counts.member_mentions||0)}</span><span>Core Fact ${esc(counts.linked_core_facts||0)}</span><span>Supporting ${esc(counts.linked_supporting_facts||0)}</span></div></div>
          <div class="connector"><span class="status ${mapped?'mapped':'unmapped'}">${mapped?(target.validation_status==='auto_accepted'?'자동 승인':'연결 후보'):'미매핑'}</span></div>
          <div class="layer-box pcf ${mapped?'':'empty'}">${targetHtml}</div>
        </div>
        <details class="card-detail"><summary>후보·검토 사유·근거 상세</summary><div class="detail-grid"><div><div class="subhead">TOP 5 PCF 후보</div>${candidateTable(frame)}</div><div><div class="subhead">검토 사유</div><div class="reason-list">${reasons.length?reasons.map(r=>`<span class="reason">${esc(reasonKo[r]||r)}</span>`).join(''):'<span class="chip">없음</span>'}</div><div class="subhead">대표 Evidence Packet</div><ul class="evidence">${evidence||'<li>대표 근거 없음</li>'}</ul></div></div></details>
      </article>`;
    };
    const list = document.getElementById('mappingList');
    list.innerHTML = DATA.frames.map(cardHtml).join('');
    const cardText = new Map(DATA.frames.map(f => [f.taskframe_id,JSON.stringify(f).toLocaleLowerCase()]));
    const applyFilters = () => {
      const q = document.getElementById('search').value.trim().toLocaleLowerCase();
      const status = document.getElementById('statusFilter').value;
      const root = rootSelect.value; let visible=0;
      document.querySelectorAll('.mapping-card').forEach(card => {
        const frame = DATA.frames.find(f => f.taskframe_id===card.dataset.id);
        const ok = (!q || cardText.get(card.dataset.id).includes(q)) && (status==='all' || status===card.dataset.status || (status==='review' && frame.review)) && (root==='all' || card.dataset.root===root);
        card.classList.toggle('hidden',!ok); if(ok) visible++;
      });
      document.getElementById('cardEmpty').hidden = visible!==0;
    };
    ['search','statusFilter','rootFilter'].forEach(id => document.getElementById(id).addEventListener(id==='search'?'input':'change',applyFilters));

    const attachments = new Map();
    const addAttachment = (hid,frame,type) => { if(!hid)return; if(!attachments.has(hid))attachments.set(hid,[]); attachments.get(hid).push({frame,type}); };
    DATA.frames.forEach(frame => frame.primary ? addAttachment(frame.primary.pcf_hid,frame,'candidate') : addAttachment(frame.candidates?.[0]?.pcf_hid,frame,'suggestion'));
    const ancestorSet = () => { const set=new Set(); attachments.forEach((_,hid) => { let current=nodes.get(hid); while(current){set.add(current.hid);current=nodes.get(current.parent);} }); return set; };
    const renderTree = () => {
      const only = document.getElementById('linkedOnly').checked;
      const q = document.getElementById('treeSearch').value.trim().toLocaleLowerCase();
      let visible = only ? ancestorSet() : new Set(DATA.nodes.map(n=>n.hid));
      if(q){ const matches=DATA.nodes.filter(n=>`${n.hid} ${n.name} ${n.path}`.toLocaleLowerCase().includes(q)); const qset=new Set(); matches.forEach(n=>{let c=n;while(c){qset.add(c.hid);c=nodes.get(c.parent);}}); visible=new Set([...visible].filter(h=>qset.has(h))); }
      const makeNode = n => {
        if(!visible.has(n.hid))return '';
        const kids=(children.get(n.hid)||[]).filter(k=>visible.has(k.hid));
        const badges=(attachments.get(n.hid)||[]).map(a=>`<span class="tf-badge ${a.type==='suggestion'?'suggestion':''}" title="${a.type==='suggestion'?'미매핑 검색 1위':'Crosswalk 연결 후보'}">${esc(a.frame.name)}</span>`).join('');
        if(!kids.length) return `<div class="tree-leaf"><span class="hid">${esc(n.hid)}</span>${esc(n.name)}<span class="tf-badges">${badges}</span></div>`;
        return `<details ${n.level<=1?'open':''}><summary><span class="hid">${esc(n.hid)}</span>${esc(n.name)}<span class="tf-badges">${badges}</span></summary>${kids.map(makeNode).join('')}</details>`;
      };
      const html=DATA.roots.filter(r=>visible.has(r.hid)).map(makeNode).join('');
      document.getElementById('tree').innerHTML=html||'<div class="empty-state">조건에 맞는 PCF 경로가 없습니다.</div>';
    };
    document.getElementById('linkedOnly').addEventListener('change',renderTree);
    document.getElementById('treeSearch').addEventListener('input',renderTree);
    renderTree();
  })();
  </script>
</body>
</html>
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCF–TaskFrame 연결 현황 HTML을 생성합니다.")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path, report = build_html_report(load_config(args.config))
    print(
        f"HTML report: {path} ({report['taskframes']} TaskFrames, "
        f"{report['pcf_nodes']} PCF nodes)"
    )


if __name__ == "__main__":
    main()
