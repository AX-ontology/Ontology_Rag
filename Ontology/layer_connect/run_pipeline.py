from __future__ import annotations

import argparse
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from aggregate_hierarchy import aggregate
from build_pcf_index import build_index
from build_taskframe_packets import build_packets
from build_html_report import build_html_report
from common import load_config, output_path, write_json
from export_to_taskkg import export
from retrieve_candidates import retrieve
from rerank_candidates import rerank
from select_document_context import select_document_context
from validate_mappings import validate


Stage = tuple[str, Callable[[dict[str, Any]], Any]]
STAGES: list[Stage] = [
    ("pcf", build_index),
    ("packets", build_packets),
    ("retrieve", retrieve),
    ("context", select_document_context),
    ("rerank", rerank),
    ("aggregate", aggregate),
    ("validate", validate),
    ("export", export),
    ("visualize", build_html_report),
]


def run(config: dict[str, Any], requested_stage: str = "all") -> dict[str, Any]:
    selected = STAGES if requested_stage == "all" else [
        stage for stage in STAGES if stage[0] == requested_stage
    ]
    if not selected:
        raise ValueError(f"알 수 없는 stage: {requested_stage}")
    started = datetime.now(timezone.utc)
    manifest: dict[str, Any] = {
        "status": "running",
        "started_at": started.isoformat(),
        "finished_at": None,
        "config": config["_config_path"],
        "python": platform.python_version(),
        "requested_stage": requested_stage,
        "stages": [],
    }
    manifest_path = output_path(config, "run_manifest.json")
    write_json(manifest_path, manifest)
    try:
        for name, function in selected:
            stage_started = time.monotonic()
            function(config)
            elapsed = time.monotonic() - stage_started
            manifest["stages"].append(
                {"name": name, "status": "completed", "elapsed_seconds": round(elapsed, 3)}
            )
            write_json(manifest_path, manifest)
            print(f"[{name}] completed in {elapsed:.2f}s", flush=True)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        write_json(manifest_path, manifest)
        raise
    manifest["status"] = "completed"
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["elapsed_seconds"] = round(
        (datetime.fromisoformat(manifest["finished_at"]) - started).total_seconds(), 3
    )
    write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCF–TaskFrame Layer Connect 파이프라인")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument(
        "--stage",
        choices=["all"] + [name for name, _ in STAGES],
        default="all",
        help="all은 의존 순서대로 전체 실행, 그 외 값은 해당 단계만 실행",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run(load_config(args.config), args.stage)
    print(
        f"Pipeline {manifest['status']}: {len(manifest['stages'])} stages, "
        f"{manifest.get('elapsed_seconds', 0):.2f}s"
    )


if __name__ == "__main__":
    main()
