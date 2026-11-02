#!/usr/bin/env python3
"""One-command TaskFrame -> PCF -> cleaned/combined Neo4j pipeline."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PIPELINE_VERSION = "1.0.0"
DOCUMENT_STAGES = ("taskframe", "pcf", "clean", "validate")
ALL_STAGES = (*DOCUMENT_STAGES, "combine")
HERE = Path(__file__).resolve().parent


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_config(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("config root는 object여야 합니다.")
    if int(config.get("version", 0)) != 1:
        raise ValueError(f"지원하지 않는 config version: {config.get('version')}")
    config["_config_path"] = str(path)
    config["_config_dir"] = str(path.parent)
    return config


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(config["_config_dir"]) / path
    return path.resolve()


def config_path(config: dict[str, Any], key: str) -> Path:
    raw = config.get("paths", {}).get(key)
    if not raw:
        raise KeyError(f"config.paths.{key}가 필요합니다.")
    return resolve_path(config, raw)


def document_paths(run_root: Path, slug: str) -> dict[str, Path]:
    root = run_root / slug
    return {
        "root": root,
        "taskkg": root / "taskkg",
        "pcf": root / "pcf",
        "cleaned": root / "cleaned",
        "generated": root / "generated",
        "logs": root / "logs",
        "manifest": root / "pipeline_manifest.json",
    }


def prepare_reused_taskkg(
    config: dict[str, Any],
    document: dict[str, Any],
    paths: dict[str, Path],
    *,
    dry_run: bool,
) -> Path | None:
    raw_source = document.get("reuse_taskkg")
    if not raw_source:
        return None
    source = resolve_path(config, raw_source)
    required = [
        source / "run_manifest.json",
        source / "05_taskframes/taskframes.json",
        source / "09_kg/nodes.jsonl",
        source / "09_kg/edges.jsonl",
    ]
    require_files(required, "reused taskframe", dry_run=dry_run)
    if not dry_run:
        manifest = json.loads((source / "run_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "complete" or manifest.get("completed_stage") != "kg":
            raise ValueError(f"재사용 TaskFrame이 완료 상태가 아닙니다: {source}")
        if not str(manifest.get("pipeline_version", "")).startswith("4.5"):
            raise ValueError(
                f"재사용 TaskFrame 버전이 호환되지 않습니다: "
                f"{manifest.get('pipeline_version')}"
            )
    target = paths["taskkg"]
    if target.exists() or target.is_symlink():
        if target.is_symlink() and target.resolve() == source:
            return source
        # 실제 output이 이미 있으면 그 결과를 resume하고 재사용 source는 무시한다.
        return None
    print(f"[taskframe] reuse {source} -> {target}", flush=True)
    if not dry_run:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source, target_is_directory=True)
    return source


def build_pcf_config(
    config: dict[str, Any], paths: dict[str, Path]
) -> dict[str, Any]:
    raw_base = config.get("pcf", {}).get("base_config")
    if not raw_base:
        raise KeyError("config.pcf.base_config가 필요합니다.")
    base_path = resolve_path(config, raw_base)
    with base_path.open(encoding="utf-8") as handle:
        result = yaml.safe_load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"PCF base config root는 object여야 합니다: {base_path}")
    result["paths"] = {
        **result.get("paths", {}),
        "pcf": str(config_path(config, "pcf")),
        "taskframes": str(paths["taskkg"] / "05_taskframes/taskframes.json"),
        "task_mentions": str(paths["taskkg"] / "03_entity_prep/task_mentions.jsonl"),
        "canonical_facts": str(
            paths["taskkg"] / "07_canonicalization/canonical_facts.jsonl"
        ),
        "output_dir": str(paths["pcf"]),
        "semantic_cache_dir": str(paths["root"].parent / "_shared_cache/embedding"),
    }
    return result


def write_pcf_config(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    temporary.replace(path)


def run_logged(
    command: list[str], cwd: Path, log_path: Path, *, dry_run: bool
) -> None:
    rendered = shlex.join(command)
    print(f"\n$ (cd {cwd} && {rendered})", flush=True)
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"\n$ (cd {cwd} && {rendered})\n")
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"명령 실패({return_code}): {rendered}")


def require_files(paths: list[Path], stage: str, *, dry_run: bool) -> None:
    if dry_run:
        return
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{stage} 필수 산출물 누락:\n" + "\n".join(missing))


def taskframe_command(
    python: str,
    config: dict[str, Any],
    document: dict[str, Any],
    paths: dict[str, Path],
    backend: str,
    limit: int,
) -> list[str]:
    settings = config.get("taskframe", {})
    command = [
        python,
        str(config_path(config, "taskframe_dir") / "run_pipeline.py"),
        "--input-xlsx",
        str(resolve_path(config, document["input_xlsx"])),
        "--sheet",
        str(document.get("sheet", "clean_sentences")),
        "--document-id",
        str(document["document_id"]),
        "--document-title",
        str(document["document_title"]),
        "--backend",
        backend,
        "--env-file",
        str(config_path(config, "taskframe_env")),
        "--out-dir",
        str(paths["taskkg"]),
        "--limit",
        str(limit),
        "--extraction-batch-size",
        str(settings.get("extraction_batch_size", 4)),
        "--llm2-max-mentions",
        str(settings.get("llm2_max_mentions", 60)),
        "--context-chunk-radius",
        str(settings.get("context_chunk_radius", 1)),
        "--candidate-threshold",
        str(settings.get("candidate_threshold", 0.28)),
        "--candidate-top-k",
        str(settings.get("candidate_top_k", 8)),
        "--promotion-min-support",
        str(settings.get("promotion_min_support", 3)),
    ]
    if document.get("chapter_no") is not None:
        command.extend(["--chapter-no", str(document["chapter_no"])])
    if (paths["taskkg"] / "run_manifest.json").exists():
        command.append("--resume")
    return command


def clean_command(
    python: str,
    config: dict[str, Any],
    paths: dict[str, Path],
    accept_needs_review: bool,
) -> list[str]:
    settings = config.get("cleaning", {})
    command = [
        python,
        str(config_path(config, "cleaning_dir") / "clean_and_export.py"),
        "--taskkg-dir",
        str(paths["taskkg"] / "09_kg"),
        "--extraction-dir",
        str(paths["taskkg"] / "02_extraction"),
        "--entity-prep-dir",
        str(paths["taskkg"] / "03_entity_prep"),
        "--pcf-dir",
        str(paths["pcf"]),
        "--entity-aliases",
        str(resolve_path(config, settings["entity_aliases"])),
        "--relation-aliases",
        str(resolve_path(config, settings["relation_aliases"])),
        "--output-dir",
        str(paths["cleaned"]),
        "--entity-min-support",
        str(settings.get("entity_min_support", 2)),
        "--entity-max-name-length",
        str(settings.get("entity_max_name_length", 40)),
        "--ontology-min-support",
        str(settings.get("ontology_min_support", 3)),
        "--ontology-min-sections",
        str(settings.get("ontology_min_sections", 2)),
    ]
    if accept_needs_review:
        command.append("--accept-needs-review-pcf")
    return command


def stage_selected(stage: str, start_at: str, stop_after: str) -> bool:
    return ALL_STAGES.index(start_at) <= ALL_STAGES.index(stage) <= ALL_STAGES.index(stop_after)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="자이제 문서 TaskFrame → PCF → Neo4j 통합 파이프라인"
    )
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--documents", nargs="+", help="기본값: config의 모든 문서")
    parser.add_argument("--run-name", default="v4_5_final")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--backend", choices=["llm", "heuristic"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-at", choices=ALL_STAGES, default="taskframe")
    parser.add_argument("--stop-after", choices=ALL_STAGES, default="combine")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--accept-needs-review-pcf",
        action="store_true",
        help="검토 대기 PCF 후보를 확정 관계로 승격(명시적 승인용)",
    )
    parser.add_argument(
        "--load-neo4j",
        action="store_true",
        help="통합 패키지 검증 후 실제 Neo4j에 씁니다.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    documents = config.get("documents", {})
    if not isinstance(documents, dict) or not documents:
        raise ValueError("config.documents가 비어 있습니다.")
    selected_slugs = args.documents or list(documents)
    if len(selected_slugs) != len(set(selected_slugs)):
        raise ValueError("--documents에 같은 문서를 중복 지정할 수 없습니다.")
    unknown = [slug for slug in selected_slugs if slug not in documents]
    if unknown:
        raise ValueError(f"알 수 없는 문서: {', '.join(unknown)}")
    if ALL_STAGES.index(args.start_at) > ALL_STAGES.index(args.stop_after):
        raise ValueError("--start-at은 --stop-after보다 뒤일 수 없습니다.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_name):
        raise ValueError("--run-name은 영문자/숫자로 시작하고 ._-만 포함할 수 있습니다.")
    if args.load_neo4j and args.stop_after != "combine":
        raise ValueError("--load-neo4j는 --stop-after combine과 함께 사용해야 합니다.")

    backend = args.backend or str(config.get("taskframe", {}).get("backend", "llm"))
    limit = args.limit if args.limit is not None else int(
        config.get("taskframe", {}).get("limit", 0)
    )
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root
        else config_path(config, "output_root")
    )
    run_root = output_root / args.run_name
    accept_needs_review = bool(
        args.accept_needs_review_pcf
        or config.get("cleaning", {}).get("accept_needs_review_pcf", False)
    )
    overall_manifest: dict[str, Any] = {
        "pipeline_version": PIPELINE_VERSION,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config": str(Path(config["_config_path"])),
        "run_name": args.run_name,
        "backend": backend,
        "limit": limit,
        "documents": selected_slugs,
        "accept_needs_review_pcf": accept_needs_review,
        "load_neo4j": args.load_neo4j,
        "document_runs": {},
    }
    if not args.dry_run:
        write_json(run_root / "pipeline_manifest.json", overall_manifest)

    failures: dict[str, str] = {}
    document_slugs = [] if args.start_at == "combine" else selected_slugs
    for slug in document_slugs:
        document = documents[slug]
        paths = document_paths(run_root, slug)
        manifest = {
            "status": "running",
            "document": slug,
            "document_id": document["document_id"],
            "started_at": datetime.now(timezone.utc).isoformat(),
            "stages": [],
        }
        if not args.dry_run:
            write_json(paths["manifest"], manifest)
        try:
            print(f"\n{'=' * 72}\n[{slug}] {document['document_title']}\n{'=' * 72}")
            require_files(
                [resolve_path(config, document["input_xlsx"])], "input", dry_run=args.dry_run
            )
            reused_taskkg = prepare_reused_taskkg(
                config, document, paths, dry_run=args.dry_run
            )
            if (
                backend == "llm"
                and reused_taskkg is None
                and stage_selected("taskframe", args.start_at, args.stop_after)
            ):
                env_file = config_path(config, "taskframe_env")
                if not env_file.exists() and not os.getenv("TASKFRAME_API_KEY"):
                    raise FileNotFoundError(f"LLM 환경 파일이 없습니다: {env_file}")

            if stage_selected("taskframe", args.start_at, args.stop_after):
                if reused_taskkg is not None:
                    manifest["stages"].append("taskframe_reused")
                    manifest["reused_taskkg"] = str(reused_taskkg)
                else:
                    run_logged(
                        taskframe_command(
                            args.python, config, document, paths, backend, limit
                        ),
                        config_path(config, "taskframe_dir"),
                        paths["logs"] / "01_taskframe.log",
                        dry_run=args.dry_run,
                    )
                    manifest["stages"].append("taskframe")
            require_files(
                [
                    paths["taskkg"] / "05_taskframes/taskframes.json",
                    paths["taskkg"] / "03_entity_prep/task_mentions.jsonl",
                    paths["taskkg"] / "07_canonicalization/canonical_facts.jsonl",
                    paths["taskkg"] / "09_kg/nodes.jsonl",
                    paths["taskkg"] / "09_kg/edges.jsonl",
                ],
                "taskframe",
                dry_run=args.dry_run,
            )
            if args.stop_after == "taskframe":
                manifest["status"] = "completed"
                continue

            pcf_config_path = paths["generated"] / "pcf_config.yaml"
            pcf_payload = build_pcf_config(config, paths)
            if not args.dry_run:
                write_pcf_config(pcf_config_path, pcf_payload)
            if stage_selected("pcf", args.start_at, args.stop_after):
                run_logged(
                    [
                        args.python,
                        str(config_path(config, "layer_connect_dir") / "run_pipeline.py"),
                        "--config",
                        str(pcf_config_path),
                    ],
                    config_path(config, "layer_connect_dir"),
                    paths["logs"] / "02_pcf.log",
                    dry_run=args.dry_run,
                )
                manifest["stages"].append("pcf")
            require_files(
                [
                    paths["pcf"] / "pcf_kg_nodes.jsonl",
                    paths["pcf"] / "pcf_kg_edges.jsonl",
                    paths["pcf"] / "taskframe_pcf_links.jsonl",
                    paths["pcf"] / "validation_report.json",
                ],
                "pcf",
                dry_run=args.dry_run,
            )
            if args.stop_after == "pcf":
                manifest["status"] = "completed"
                continue

            if stage_selected("clean", args.start_at, args.stop_after):
                run_logged(
                    clean_command(
                        args.python, config, paths, accept_needs_review
                    ),
                    config_path(config, "cleaning_dir"),
                    paths["logs"] / "03_clean.log",
                    dry_run=args.dry_run,
                )
                manifest["stages"].append("clean")
            require_files(
                [
                    paths["cleaned"] / "neo4j/nodes.jsonl",
                    paths["cleaned"] / "neo4j/edges.jsonl",
                    paths["cleaned"] / "validation_report.json",
                ],
                "clean",
                dry_run=args.dry_run,
            )
            if args.stop_after == "clean":
                manifest["status"] = "completed"
                continue

            if stage_selected("validate", args.start_at, args.stop_after):
                run_logged(
                    [
                        args.python,
                        str(config_path(config, "cleaning_dir") / "load_neo4j.py"),
                        "--graph-dir",
                        str(paths["cleaned"] / "neo4j"),
                        "--run-id",
                        f"{slug}_{args.run_name}",
                        "--pipeline-version",
                        f"final-pipeline-{PIPELINE_VERSION}",
                        "--validate-only",
                    ],
                    config_path(config, "cleaning_dir"),
                    paths["logs"] / "04_validate.log",
                    dry_run=args.dry_run,
                )
                manifest["stages"].append("validate")
            manifest["status"] = "completed"
        except Exception as exc:
            manifest["status"] = "failed"
            manifest["error"] = f"{type(exc).__name__}: {exc}"
            failures[slug] = manifest["error"]
        finally:
            manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
            overall_manifest["document_runs"][slug] = manifest
            if not args.dry_run:
                write_json(paths["manifest"], manifest)
                write_json(run_root / "pipeline_manifest.json", overall_manifest)
        if failures and not args.continue_on_error:
            break

    combined_dir = run_root / "combined"
    if not failures and stage_selected("combine", args.start_at, args.stop_after):
        combine_command = [
            args.python,
            str(HERE / "combine_graphs.py"),
        ]
        for slug in selected_slugs:
            paths = document_paths(run_root, slug)
            combine_command.extend(
                [
                    "--input",
                    f"{documents[slug]['document_id']}={paths['cleaned'] / 'neo4j'}",
                ]
            )
        combine_command.extend(["--output-dir", str(combined_dir)])
        run_logged(
            combine_command,
            HERE,
            run_root / "logs/05_combine.log",
            dry_run=args.dry_run,
        )
        overall_manifest["combined_graph"] = str(combined_dir / "neo4j")

    if args.load_neo4j:
        if failures:
            raise RuntimeError("실패한 문서가 있어 Neo4j 적재를 중단합니다.")
        require_files(
            [
                combined_dir / "neo4j/nodes.jsonl",
                combined_dir / "neo4j/edges.jsonl",
            ],
            "combined",
            dry_run=args.dry_run,
        )
        neo4j_settings = config.get("neo4j", {})
        command = [
            args.python,
            str(config_path(config, "cleaning_dir") / "load_neo4j.py"),
            "--graph-dir",
            str(combined_dir / "neo4j"),
            "--env-file",
            str(config_path(config, "neo4j_env")),
            "--run-id",
            f"{'_'.join(selected_slugs)}_{args.run_name}",
            "--pipeline-version",
            f"final-pipeline-{PIPELINE_VERSION}",
            "--batch-size",
            str(neo4j_settings.get("batch_size", 500)),
            "--print-queries",
        ]
        if neo4j_settings.get("database"):
            command.extend(["--database", str(neo4j_settings["database"])])
        run_logged(
            command,
            config_path(config, "cleaning_dir"),
            run_root / "logs/06_neo4j_load.log",
            dry_run=args.dry_run,
        )
        overall_manifest["neo4j_loaded"] = True

    overall_manifest["status"] = (
        "dry_run"
        if args.dry_run
        else "partial_failure"
        if failures
        else "completed"
    )
    overall_manifest["failures"] = failures
    overall_manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    if not args.dry_run:
        write_json(run_root / "pipeline_manifest.json", overall_manifest)
    print(
        json.dumps(
            {
                "status": overall_manifest["status"],
                "run_root": str(run_root),
                "documents": selected_slugs,
                "combined_graph": overall_manifest.get("combined_graph"),
                "neo4j_loaded": overall_manifest.get("neo4j_loaded", False),
                "failures": failures,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
