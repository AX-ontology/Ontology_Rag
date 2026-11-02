#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

from taskkg import __version__
from taskkg.clustering import build_candidate_blocks
from taskkg.evidence import build_chunks, load_evidence
from taskkg.extraction import (
    chunk_modality, extract_heuristic, extract_llm,
    extract_structured_refinement_llm, flatten,
)
from taskkg.graph import build_graph, csv_edges, csv_nodes
from taskkg.induction import (
    build_ambiguous_fact_contexts, estimate_hierarchical_llm2_calls,
    generate_heuristic, generate_hierarchical_llm, pack_candidate_blocks,
    propagate_fact_links,
)
from taskkg.llm import ChatJSONClient, LLMConfig
from taskkg.normalization import (
    canonicalize, conservative_prepare, load_aliases,
)
from taskkg.standards import (
    load_standard_tasks, map_standard_tasks,
)
from taskkg.utils import (
    load_env, read_json, read_jsonl, write_csv, write_json, write_jsonl,
)


STAGES = [
    "evidence", "extraction", "entity_prep", "clustering", "taskframes",
    "propagation", "canonicalization", "standard_mapping", "kg",
]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Task-Centric Open KG v4.5: Evidence → Open Fact/TaskMention → "
            "TaskFrame induction → multi-link propagation → canonical KG"
        )
    )
    parser.add_argument(
        "--input-xlsx", type=Path,
        default=root.parent / "data/process_data/자이제_1.xlsx",
    )
    parser.add_argument("--sheet", default="clean_sentences")
    parser.add_argument("--document-id", default="JAIJE_1")
    parser.add_argument("--document-title", default="자이제 1")
    parser.add_argument("--chapter-no")
    parser.add_argument("--limit", type=int, default=0, help="Evidence 제한(0=전체)")
    parser.add_argument(
        "--backend", choices=["llm", "heuristic"], default="llm",
        help="heuristic은 코드/KG 무결성 smoke test 전용",
    )
    parser.add_argument("--out-dir", type=Path, default=root / "outputs" / "zaije_1_v4")
    parser.add_argument("--env-file", type=Path, default=root.parent / "config/taskframe.env")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--base-output-dir", type=Path,
        help="부분 재추출 시 재사용할 기존 완료 output 디렉터리",
    )
    parser.add_argument(
        "--reextract-structured-only", action="store_true",
        help="기존 TaskMention/TaskFrame은 유지하고 table/figure Fact만 전용 prompt로 재추출",
    )
    parser.add_argument(
        "--entity-alias-json", type=Path,
        default=root / "examples" / "entity_aliases.sample.json",
    )
    parser.add_argument(
        "--relation-alias-json", type=Path,
        default=root / "examples" / "relation_aliases.sample.json",
    )
    parser.add_argument("--standards-json", type=Path)
    parser.add_argument("--max-chunk-sentences", type=int, default=5)
    parser.add_argument("--max-chunk-chars", type=int, default=1800)
    parser.add_argument("--extraction-batch-size", type=int, default=4)
    parser.add_argument("--candidate-threshold", type=float, default=0.28)
    parser.add_argument("--candidate-top-k", type=int, default=8)
    parser.add_argument(
        "--context-chunk-radius", type=int, default=1,
        help="애매한 Fact의 앞·뒤 참조 chunk 수(기본 1)",
    )
    parser.add_argument(
        "--llm2-max-mentions", type=int, default=60,
        help="LLM 2 로컬 induction 호출당 최대 TaskMention 수",
    )
    parser.add_argument("--promotion-min-support", type=int, default=3)
    parser.add_argument("--standard-top-k", type=int, default=3)
    parser.add_argument("--standard-min-score", type=float, default=0.28)
    parser.add_argument("--stop-after", choices=STAGES, default="kg")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--catalog-assigned", action="store_true",
                        help="Require completed direct catalog assignment; never invoke local/global induction")
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def _stop(args: argparse.Namespace, stage: str) -> bool:
    return STAGES.index(stage) >= STAGES.index(args.stop_after)


def _finish(
    output: Path, manifest: dict[str, Any], stage: str, status: str = "complete"
) -> int:
    manifest.update(
        {
            "status": status,
            "completed_stage": stage,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    write_json(output / "run_manifest.json", manifest)
    print(f"[done] stage={stage} status={status} out={output}")
    return 0


def _client(args: argparse.Namespace) -> ChatJSONClient:
    config = LLMConfig.from_env()
    print(f"[llm] model={config.model} base={config.api_base}")
    return ChatJSONClient(
        config,
        args.cache_dir or (args.out_dir / ".cache" / "llm"),
        use_cache=not args.no_cache,
    )


def _load_alias_file(path: Path | None) -> dict[str, str]:
    return load_aliases(read_json(path)) if path and path.exists() else {}


def main() -> int:
    args = parse_args()
    args.out_dir = args.out_dir.resolve()
    if args.cache_dir:
        args.cache_dir = args.cache_dir.resolve()
    if args.base_output_dir:
        args.base_output_dir = args.base_output_dir.resolve()
    if args.reextract_structured_only:
        if args.backend != "llm":
            raise ValueError("--reextract-structured-only는 --backend llm이 필요합니다.")
        if not args.base_output_dir:
            raise ValueError("--reextract-structured-only에는 --base-output-dir가 필요합니다.")
        required = args.base_output_dir / "02_extraction" / "chunk_extractions.jsonl"
        if not required.exists():
            raise FileNotFoundError(f"기존 extraction을 찾지 못했습니다: {required}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.catalog_assigned:
        if not args.resume or args.reextract_structured_only:
            raise ValueError('--catalog-assigned requires --resume and forbids re-extraction')
        required = ['01_evidence/evidence_spans.jsonl', '01_evidence/chunks.jsonl',
            '02_extraction/chunk_extractions.jsonl', '03_entity_prep/task_mentions.jsonl',
            '03_entity_prep/open_triples.jsonl', '04_clustering/pair_scores.jsonl',
            '04_clustering/candidate_blocks.jsonl', '05_taskframes/taskframes.json',
            '05_taskframes/task_relations.jsonl', '05_taskframes/triple_link_decisions.jsonl',
            '05_taskframes/validation_report.json', '06_propagation/facts_with_taskframes.jsonl',
            '06_propagation/fact_taskframe_links.jsonl', '06_propagation/summary.json']
        for filename in required:
            if not (args.out_dir / filename).is_file():
                raise ValueError(f'Incomplete direct assignment; refusing LLM fallback: {filename}')
        report = read_json(args.out_dir / '05_taskframes/validation_report.json')
        if report.get('method') != 'direct_mentions_existing_first' or report.get('status') != 'complete':
            raise ValueError('Expected completed direct catalog assignment')
    load_env(args.env_file)
    manifest_path = args.out_dir / "run_manifest.json"
    if args.resume and manifest_path.exists():
        previous = read_json(manifest_path)
        if previous.get("pipeline_version") != __version__:
            print(
                f"[error] resume version mismatch: output={previous.get('pipeline_version')} "
                f"code={__version__}. 새 out-dir를 사용하세요.",
                file=sys.stderr,
            )
            return 2
    planned_llm_stages = [] if args.catalog_assigned or args.stop_after == 'evidence' else [
        'llm_1_open_fact_task_extraction']
    if not args.catalog_assigned and STAGES.index(args.stop_after) >= STAGES.index('taskframes'):
        planned_llm_stages.append('llm_2_hierarchical_taskframe_induction_and_reconciliation')
    manifest: dict[str, Any] = {
        "pipeline_version": __version__,
        "method": "task-aware-open-triples-and-task-centric-knowledge-subgraphs",
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
        "llm_stage_count": len(planned_llm_stages),
        "llm_stages": planned_llm_stages,
        "llm_stage_count_note": "Configured stage types, not actual API calls; graph-only catalog resume uses zero.",
        "v4_invariants": [
            "raw Fact and Evidence are never deleted by canonicalization",
            "every Fact is connected to at least one TaskFrame",
            "one Fact may link to multiple TaskFrames",
            "TaskFrame is a task-centered Fact subgraph, not a fixed slot record",
        ],
        "input": str(args.input_xlsx.resolve()),
        "sheet": args.sheet,
        "limit": args.limit,
        "stop_after": args.stop_after,
        "thresholds": {
            "candidate": args.candidate_threshold,
            "context_chunk_radius": args.context_chunk_radius,
            "llm2_max_mentions": args.llm2_max_mentions,
            "promotion_min_support": args.promotion_min_support,
            "standard_min_score": args.standard_min_score,
        },
        "base_output_dir": str(args.base_output_dir) if args.base_output_dir else "",
        "structured_refinement_only": args.reextract_structured_only,
        "counts": {},
        "warning": "heuristic backend is smoke-test only" if args.backend == "heuristic" else "",
    }
    write_json(manifest_path, manifest)

    try:
        # Step 1: Document structure and Evidence
        evidence_path = args.out_dir / "01_evidence" / "evidence_spans.jsonl"
        chunks_path = args.out_dir / "01_evidence" / "chunks.jsonl"
        if args.resume and evidence_path.exists() and chunks_path.exists():
            evidence, chunks = read_jsonl(evidence_path), read_jsonl(chunks_path)
            print(f"[step1] resumed evidence={len(evidence)} chunks={len(chunks)}")
        elif args.reextract_structured_only:
            base_evidence = args.base_output_dir / "01_evidence" / "evidence_spans.jsonl"
            base_chunks = args.base_output_dir / "01_evidence" / "chunks.jsonl"
            evidence, chunks = read_jsonl(base_evidence), read_jsonl(base_chunks)
            write_jsonl(evidence_path, evidence)
            write_jsonl(chunks_path, chunks)
            write_json(
                args.out_dir / "01_evidence" / "summary.json",
                {
                    "source": "reused_for_structured_refinement",
                    "base_output_dir": str(args.base_output_dir),
                    "evidence_count": len(evidence),
                    "chunk_count": len(chunks),
                },
            )
            print(f"[step1] reused evidence={len(evidence)} chunks={len(chunks)}")
        else:
            evidence, summary = load_evidence(
                args.input_xlsx, args.sheet, args.document_id,
                args.document_title, args.chapter_no, args.limit,
            )
            chunks = build_chunks(
                evidence, args.max_chunk_sentences, args.max_chunk_chars
            )
            summary["chunk_count"] = len(chunks)
            write_jsonl(evidence_path, evidence)
            write_jsonl(chunks_path, chunks)
            write_json(args.out_dir / "01_evidence" / "summary.json", summary)
            print(f"[step1] evidence={len(evidence)} chunks={len(chunks)}")
        manifest["counts"].update(evidence=len(evidence), chunks=len(chunks))
        if _stop(args, "evidence"):
            return _finish(args.out_dir, manifest, "evidence")

        # Step 2: LLM 1 joint extraction
        extraction_path = args.out_dir / "02_extraction" / "chunk_extractions.jsonl"
        if args.reextract_structured_only:
            base_extractions = read_jsonl(
                args.base_output_dir / "02_extraction" / "chunk_extractions.jsonl"
            )
            base_by_id = {item["chunk_id"]: item for item in base_extractions}
            missing = [item["chunk_id"] for item in chunks if item["chunk_id"] not in base_by_id]
            if missing:
                raise RuntimeError(f"base extraction에 없는 chunk: {missing[:10]}")
            refined = extract_structured_refinement_llm(
                _client(args), chunks, base_by_id, args.extraction_batch_size
            )
            refined_by_id = {item["chunk_id"]: item for item in refined}
            extractions = [
                refined_by_id.get(chunk["chunk_id"], base_by_id[chunk["chunk_id"]])
                for chunk in chunks
            ]
            write_jsonl(extraction_path, extractions)
            print(
                f"[step2] structured-refined={len(refined)} "
                f"text-reused={len(extractions) - len(refined)}"
            )
        elif args.resume and extraction_path.exists():
            extractions = read_jsonl(extraction_path)
            print("[step2] resumed")
        else:
            extractions = (
                extract_llm(_client(args), chunks, args.extraction_batch_size)
                if args.backend == "llm" else extract_heuristic(chunks)
            )
            write_jsonl(extraction_path, extractions)
        mentions, raw_triples, audits = flatten(extractions)
        write_jsonl(
            args.out_dir / "02_extraction" / "task_mentions_raw.jsonl", mentions
        )
        write_jsonl(args.out_dir / "02_extraction" / "open_triples_raw.jsonl", raw_triples)
        schema_errors = [
            error for item in extractions for error in item.get("schema_errors", [])
        ]
        knowledge_functions = Counter(
            item.get("knowledge_function", "fact") for item in raw_triples
        )
        qualifier_types = Counter(
            qualifier.get("type", "")
            for item in raw_triples
            for qualifier in item.get("qualifiers", [])
            if isinstance(qualifier, dict) and qualifier.get("type")
        )
        extraction_summary = {
            "task_mention_count": len(mentions),
            "open_triple_count": len(raw_triples),
            "linked_triple_count": sum(bool(item["task_links"]) for item in raw_triples),
            "unlinked_general_fact_count": sum(
                not item["task_links"] for item in raw_triples
            ),
            "multi_task_mention_triple_count": sum(
                len(item["task_links"]) > 1 for item in raw_triples
            ),
            "audited_evidence_count": len(audits),
            "knowledge_function_counts": dict(sorted(knowledge_functions.items())),
            "qualifier_type_counts": dict(sorted(qualifier_types.items())),
            "exact_quote_grounded_count": sum(
                item.get("evidence_grounding", {}).get("exact_quote", False)
                for item in raw_triples
            ),
            "schema_error_count": len(schema_errors),
            "schema_errors": schema_errors,
            "extraction_mode_counts": dict(sorted(Counter(
                chunk_modality(item) for item in chunks
            ).items())),
            "structured_refinement_only": args.reextract_structured_only,
            "reused_text_chunk_count": sum(
                chunk_modality(item) == "text" for item in chunks
            ) if args.reextract_structured_only else 0,
        }
        write_json(args.out_dir / "02_extraction" / "summary.json", extraction_summary)
        manifest["counts"].update(
            task_mentions=len(mentions), raw_open_triples=len(raw_triples),
            extraction_schema_errors=len(schema_errors),
        )
        print(f"[step2] mentions={len(mentions)} facts={len(raw_triples)}")
        if _stop(args, "extraction"):
            return _finish(args.out_dir, manifest, "extraction")

        # Step 3: conservative Entity preparation
        mentions_path = args.out_dir / "03_entity_prep" / "task_mentions.jsonl"
        triples_path = args.out_dir / "03_entity_prep" / "open_triples.jsonl"
        if args.resume and mentions_path.exists() and triples_path.exists():
            mentions, prepared_triples = read_jsonl(mentions_path), read_jsonl(triples_path)
            print("[step3] resumed")
        else:
            entity_aliases = _load_alias_file(args.entity_alias_json)
            mentions, prepared_triples = conservative_prepare(
                mentions, raw_triples, entity_aliases
            )
            write_jsonl(mentions_path, mentions)
            write_jsonl(triples_path, prepared_triples)
            write_json(
                args.out_dir / "03_entity_prep" / "summary.json",
                {
                    "method": "conservative_alias_and_surface_cleanup",
                    "verified_entity_alias_count": len(entity_aliases),
                    "task_mention_count": len(mentions),
                    "fact_count": len(prepared_triples),
                    "raw_fact_preservation_rate": (
                        len(prepared_triples) / len(raw_triples)
                        if raw_triples else 1.0
                    ),
                },
            )
        print(f"[step3] mentions={len(mentions)} facts={len(prepared_triples)}")
        if _stop(args, "entity_prep"):
            return _finish(args.out_dir, manifest, "entity_prep")

        # Step 4: high-recall blocking only
        pairs_path = args.out_dir / "04_clustering" / "pair_scores.jsonl"
        blocks_path = args.out_dir / "04_clustering" / "candidate_blocks.jsonl"
        if args.resume and pairs_path.exists() and blocks_path.exists():
            pairs, blocks = read_jsonl(pairs_path), read_jsonl(blocks_path)
            print("[step4] resumed")
        else:
            pairs, blocks = build_candidate_blocks(
                mentions, prepared_triples,
                args.candidate_threshold, args.candidate_top_k,
            )
            write_jsonl(pairs_path, pairs)
            write_jsonl(blocks_path, blocks)
            write_json(
                args.out_dir / "04_clustering" / "summary.json",
                {
                    "method": "high_recall_blocking_not_final_identity",
                    "pair_count": len(pairs),
                    "candidate_block_count": len(blocks),
                    "multi_member_block_count": sum(
                        len(item["member_task_mention_ids"]) > 1 for item in blocks
                    ),
                },
            )
        manifest["counts"].update(candidate_pairs=len(pairs), candidate_blocks=len(blocks))
        llm2_call_plan = ({} if args.catalog_assigned else estimate_hierarchical_llm2_calls(
            blocks, mentions, args.llm2_max_mentions
        ))
        modality_counts = Counter(chunk_modality(item) for item in chunks)
        if args.catalog_assigned:
            llm1_input_count = llm1_call_count = 0
        elif args.reextract_structured_only:
            llm1_input_count = modality_counts.get("structured", 0)
            llm1_call_count = (
                (llm1_input_count + args.extraction_batch_size - 1)
                // args.extraction_batch_size
            )
        else:
            llm1_input_count = len(chunks)
            # text와 structured는 서로 다른 prompt/call stream으로 실행된다.
            llm1_call_count = sum(
                (count + args.extraction_batch_size - 1)
                // args.extraction_batch_size
                for count in modality_counts.values()
            )
        if args.reextract_structured_only or args.catalog_assigned:
            llm2_call_plan = {
                "local_induction_calls": 0,
                "global_reconciliation_calls": 0,
                "ambiguous_fact_extra_calls": 0,
                "total_llm2_calls": 0,
            }
        call_plan = {
            "llm1_extraction_calls": llm1_call_count,
            **llm2_call_plan,
            "projected_total_llm_calls": (
                llm1_call_count + llm2_call_plan["total_llm2_calls"]
            ),
            "cache_note": "동일 prompt cache hit는 새 API 호출을 발생시키지 않습니다.",
            "retry_note": "일시 오류 재시도 HTTP attempt는 별도일 수 있습니다.",
        }
        write_json(
            args.out_dir / "04_clustering" / "llm_call_plan.json", call_plan
        )
        planned_batches = [] if args.catalog_assigned else pack_candidate_blocks(
            blocks, mentions, args.llm2_max_mentions
        )
        write_jsonl(
            args.out_dir / "04_clustering" / "llm2_batch_plan.jsonl",
            [
                {
                    "batch_id": batch["batch_id"],
                    "task_mention_count": len(
                        batch["member_task_mention_ids"]
                    ),
                    "candidate_block_ids": [
                        item["candidate_block_id"]
                        for item in batch["candidate_blocks"]
                    ],
                }
                for batch in planned_batches
            ],
        )
        manifest["llm_call_plan"] = call_plan
        print(f"[step4] candidate_blocks={len(blocks)} (blocking only)")
        print(
            f"[call-plan] llm1={llm1_call_count} "
            f"llm2={llm2_call_plan['total_llm2_calls']} "
            f"total={call_plan['projected_total_llm_calls']}"
        )
        if _stop(args, "clustering"):
            return _finish(args.out_dir, manifest, "clustering")

        # Step 5: LLM 2 global TaskFrame induction and ambiguous link judgment
        frames_path = args.out_dir / "05_taskframes" / "taskframes.json"
        relations_path = args.out_dir / "05_taskframes" / "task_relations.jsonl"
        decisions_path = args.out_dir / "05_taskframes" / "triple_link_decisions.jsonl"
        report_path = args.out_dir / "05_taskframes" / "validation_report.json"
        ambiguous_path = (
            args.out_dir / "05_taskframes" / "ambiguous_fact_contexts.jsonl"
        )
        chunk_context_path = (
            args.out_dir / "05_taskframes" / "surrounding_chunk_contexts.jsonl"
        )
        ambiguous_facts, surrounding_chunk_contexts = build_ambiguous_fact_contexts(
            prepared_triples, mentions, chunks, args.context_chunk_radius
        )
        write_jsonl(ambiguous_path, ambiguous_facts)
        write_jsonl(chunk_context_path, surrounding_chunk_contexts)
        if args.reextract_structured_only:
            base_taskframes = args.base_output_dir / "05_taskframes"
            frames = read_json(base_taskframes / "taskframes.json")
            task_relations = read_jsonl(base_taskframes / "task_relations.jsonl")
            decisions = read_jsonl(base_taskframes / "triple_link_decisions.jsonl")
            induction_report = read_json(base_taskframes / "validation_report.json")
            current_mentions = {item["task_mention_id"] for item in mentions}
            framed_mentions = {
                mention_id
                for frame in frames
                for mention_id in frame.get("member_task_mentions", [])
            }
            if current_mentions != framed_mentions:
                raise RuntimeError(
                    "structured refinement 후 TaskMention partition이 변경되었습니다: "
                    f"missing={len(current_mentions - framed_mentions)}, "
                    f"extra={len(framed_mentions - current_mentions)}"
                )
            induction_report = {
                **induction_report,
                "reused_for_structured_refinement": True,
                "base_output_dir": str(args.base_output_dir),
                "total_llm2_calls": 0,
            }
            write_json(frames_path, frames)
            write_jsonl(relations_path, task_relations)
            write_jsonl(decisions_path, decisions)
            write_json(report_path, induction_report)
            print("[step5] reused reviewed TaskFrames (mention partition unchanged)")
        elif args.resume and all(
            path.exists() for path in (
                frames_path, relations_path, decisions_path, report_path,
                ambiguous_path, chunk_context_path,
            )
        ):
            frames = read_json(frames_path)
            task_relations = read_jsonl(relations_path)
            decisions = read_jsonl(decisions_path)
            induction_report = read_json(report_path)
            print("[step5] resumed")
        else:
            frames, task_relations, decisions, induction_report = (
                generate_hierarchical_llm(
                    _client(args), mentions, prepared_triples, blocks,
                    ambiguous_facts, surrounding_chunk_contexts,
                    args.llm2_max_mentions,
                )
                if args.backend == "llm"
                else generate_heuristic(
                    mentions, prepared_triples, blocks, ambiguous_facts
                )
            )
            induction_report["surrounding_chunk_context_count"] = len(
                surrounding_chunk_contexts
            )
            induction_report["context_candidate_unlinked_fact_count"] = sum(
                not item["current_task_links"] and bool(
                    item["context_task_mention_ids"]
                )
                for item in ambiguous_facts
            )
            write_json(frames_path, frames)
            write_jsonl(relations_path, task_relations)
            write_jsonl(decisions_path, decisions)
            write_json(report_path, induction_report)
        if not induction_report["mention_partition_valid"]:
            raise RuntimeError("TaskMention partition 검증 실패")
        manifest["counts"].update(
            taskframes=len(frames), task_relations=len(task_relations),
            ambiguous_fact_rejudgments=len(decisions),
            ambiguous_fact_contexts=len(ambiguous_facts),
            surrounding_chunk_contexts=len(surrounding_chunk_contexts),
            llm2_calls=induction_report.get("total_llm2_calls", 0),
        )
        print(
            f"[step5] taskframes={len(frames)} valid=True "
            f"ambiguous_facts={len(ambiguous_facts)} "
            f"context_chunks={len(surrounding_chunk_contexts)}"
        )
        if _stop(args, "taskframes"):
            return _finish(args.out_dir, manifest, "taskframes")

        # Step 6: algorithmic multi-link propagation
        propagated_path = args.out_dir / "06_propagation" / "facts_with_taskframes.jsonl"
        fact_links_path = args.out_dir / "06_propagation" / "fact_taskframe_links.jsonl"
        if args.resume and propagated_path.exists() and fact_links_path.exists():
            propagated = read_jsonl(propagated_path)
            fact_frame_links = read_jsonl(fact_links_path)
            propagation_summary = read_json(
                args.out_dir / "06_propagation" / "summary.json"
            )
            print("[step6] resumed")
        else:
            propagated, fact_frame_links, propagation_summary = propagate_fact_links(
                prepared_triples, frames, decisions, mentions, exhaustive=True
            )
            write_jsonl(propagated_path, propagated)
            write_jsonl(fact_links_path, fact_frame_links)
            write_json(
                args.out_dir / "06_propagation" / "summary.json",
                propagation_summary,
            )
        manifest["counts"].update(
            task_related_facts=propagation_summary["task_related_fact_count"],
            unlinked_general_facts=propagation_summary["unlinked_fact_count"],
            multi_taskframe_facts=propagation_summary["multi_taskframe_fact_count"],
        )
        print(
            f"[step6] task-related={propagation_summary['task_related_fact_count']} "
            f"general-unlinked={propagation_summary['unlinked_fact_count']} "
            f"multi-frame={propagation_summary['multi_taskframe_fact_count']}"
        )
        if _stop(args, "propagation"):
            return _finish(args.out_dir, manifest, "propagation")

        # Step 7: context-aware entity/relation canonicalization and promotion
        canonical_facts_path = (
            args.out_dir / "07_canonicalization" / "canonical_facts.jsonl"
        )
        if args.resume and canonical_facts_path.exists():
            canonical_facts = read_jsonl(canonical_facts_path)
            canonical_entities = read_jsonl(
                args.out_dir / "07_canonicalization" / "canonical_entities.jsonl"
            )
            canonical_relations = read_jsonl(
                args.out_dir / "07_canonicalization" / "canonical_relations.jsonl"
            )
            promoted_terms = read_jsonl(
                args.out_dir / "07_canonicalization" / "promoted_ontology_terms.jsonl"
            )
            print("[step7] resumed")
        else:
            entity_aliases = _load_alias_file(args.entity_alias_json)
            relation_aliases = _load_alias_file(args.relation_alias_json)
            canonical_facts, canonical_entities, canonical_relations, promoted_terms = (
                canonicalize(
                    propagated, entity_aliases, relation_aliases,
                    args.promotion_min_support,
                )
            )
            write_jsonl(canonical_facts_path, canonical_facts)
            write_jsonl(
                args.out_dir / "07_canonicalization" / "canonical_entities.jsonl",
                canonical_entities,
            )
            write_jsonl(
                args.out_dir / "07_canonicalization" / "canonical_relations.jsonl",
                canonical_relations,
            )
            write_jsonl(
                args.out_dir / "07_canonicalization" / "promoted_ontology_terms.jsonl",
                promoted_terms,
            )
            write_json(
                args.out_dir / "07_canonicalization" / "summary.json",
                {
                    "fact_count": len(canonical_facts),
                    "raw_fact_preservation_rate": (
                        sum(item["raw_fact_preserved"] for item in canonical_facts)
                        / len(canonical_facts) if canonical_facts else 1.0
                    ),
                    "canonical_entity_count": len(canonical_entities),
                    "canonical_relation_count": len(canonical_relations),
                    "promoted_ontology_term_count": len(promoted_terms),
                    "promotion_status": "candidate_requires_validation",
                },
            )
        manifest["counts"].update(
            canonical_entities=len(canonical_entities),
            canonical_relations=len(canonical_relations),
            promoted_ontology_terms=len(promoted_terms),
        )
        print(
            f"[step7] entities={len(canonical_entities)} "
            f"relations={len(canonical_relations)} promoted={len(promoted_terms)}"
        )
        if _stop(args, "canonicalization"):
            return _finish(args.out_dir, manifest, "canonicalization")

        # Step 8: optional NCS/APQC/PCF/custom standard candidate mapping
        standards: list[dict[str, Any]] = []
        standard_mappings: list[dict[str, Any]] = []
        if args.standards_json:
            if not args.standards_json.exists():
                raise FileNotFoundError(f"표준업무 JSON이 없습니다: {args.standards_json}")
            standards = load_standard_tasks(read_json(args.standards_json))
            standard_mappings = map_standard_tasks(
                frames, standards, args.standard_top_k, args.standard_min_score
            )
        write_jsonl(
            args.out_dir / "08_standard_mapping" / "standard_tasks.jsonl", standards
        )
        write_jsonl(
            args.out_dir / "08_standard_mapping" / "candidate_mappings.jsonl",
            standard_mappings,
        )
        write_json(
            args.out_dir / "08_standard_mapping" / "summary.json",
            {
                "enabled": bool(args.standards_json),
                "standard_task_count": len(standards),
                "candidate_mapping_count": len(standard_mappings),
                "note": "후보 매핑은 TaskFrame 의미를 변경하지 않으며 검증 전 candidate 상태입니다.",
            },
        )
        manifest["counts"]["standard_mapping_candidates"] = len(standard_mappings)
        print(f"[step8] standard_mapping_candidates={len(standard_mappings)}")
        if _stop(args, "standard_mapping"):
            return _finish(args.out_dir, manifest, "standard_mapping")

        # Step 9: minimal meta-schema KG
        nodes, edges, graph_report = build_graph(
            args.document_id, args.document_title, evidence, mentions,
            canonical_facts, frames, task_relations, canonical_entities,
            canonical_relations, promoted_terms, standards, standard_mappings,
        )
        kg_dir = args.out_dir / "09_kg"
        write_jsonl(kg_dir / "nodes.jsonl", nodes)
        write_jsonl(kg_dir / "edges.jsonl", edges)
        write_csv(kg_dir / "nodes.csv", csv_nodes(nodes))
        write_csv(kg_dir / "edges.csv", csv_edges(edges))
        write_json(kg_dir / "validation_report.json", graph_report)
        raw_preservation = (
            len(canonical_facts) == len(raw_triples)
            and all(item.get("raw_fact_preserved") for item in canonical_facts)
        )
        quality_status = (
            "fail"
            if (
                not graph_report["valid"]
                or not raw_preservation
                or propagation_summary["all_fact_assignment_rate"] < 1.0
            )
            else "warn" if schema_errors else "pass"
        )
        quality_report = {
            "status": quality_status,
            "graph_valid": graph_report["valid"],
            "raw_fact_preservation_valid": raw_preservation,
            "raw_fact_count": len(raw_triples),
            "final_fact_count": len(canonical_facts),
            "fact_evidence_grounding_rate": graph_report[
                "fact_evidence_grounding_rate"
            ],
            "taskmention_evidence_grounding_rate": graph_report[
                "taskmention_evidence_grounding_rate"
            ],
            "mention_partition_valid": induction_report["mention_partition_valid"],
            "extraction_schema_error_count": len(schema_errors),
            "unlinked_general_fact_count": graph_report[
                "unlinked_general_fact_count"
            ],
            "all_fact_assignment_rate": propagation_summary[
                "all_fact_assignment_rate"
            ],
            "exhaustive_fallback_count": propagation_summary[
                "exhaustive_fallback_count"
            ],
            "multi_taskframe_fact_count": propagation_summary[
                "multi_taskframe_fact_count"
            ],
            "ambiguous_fact_context_count": len(ambiguous_facts),
            "surrounding_chunk_context_count": len(surrounding_chunk_contexts),
            "notes": [
                "모든 Fact는 최소 하나 이상의 TaskFrame에 연결되어야 합니다.",
                "일반 Fact는 가장 관련 있는 TaskFrame에 context role로 연결됩니다.",
                "애매한 Fact는 현재 Evidence와 앞·뒤 chunk 문맥을 LLM 2가 함께 판정합니다.",
                "heuristic backend 결과는 연구 평가에 사용할 수 없습니다."
                if args.backend == "heuristic" else "",
            ],
        }
        write_json(kg_dir / "quality_report.json", quality_report)
        manifest["quality_status"] = quality_status
        manifest["counts"].update(nodes=len(nodes), edges=len(edges))
        print(
            f"[step9] nodes={len(nodes)} edges={len(edges)} "
            f"valid={graph_report['valid']} quality={quality_status}"
        )
        if not graph_report["valid"]:
            raise RuntimeError(f"KG validation failed: {graph_report['errors'][:5]}")
        return _finish(args.out_dir, manifest, "kg")
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(manifest_path, manifest)
        print(f"[error] {manifest['error']}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
