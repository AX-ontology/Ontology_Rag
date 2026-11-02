from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
from typing import Any

import pandas as pd

from .utils import stable_id, text


REQUIRED = {
    "page", "chapter_no", "chapter_title", "section_no", "section_title",
}
META = ["page", "chapter_no", "chapter_title", "section_no", "section_title"]
TEXT_COLUMNS = ("text_for_llm_human", "text_for_llm", "사람 검수")


def _number(value: Any) -> str:
    value = text(value)
    return value[:-2] if value.endswith(".0") else value


def split_units(value: Any) -> list[dict[str, Any]]:
    raw = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw:
        return []
    if re.search(r"\|\s*:?-{3,}", raw) or re.match(r"^\s*(표|Table)\s*\d", raw, re.I):
        return [{"text": text(raw), "kind": "table", "structural": True}]
    if re.match(r"^\s*(그림|Figure)\s*\d", raw, re.I):
        return [{"text": text(raw), "kind": "figure", "structural": True}]
    prepared = re.sub(r"(?<=[가-힣)])(?=-\s*\d{1,2}\s*(~|시|:))", "\n", raw)
    prepared = re.sub(r"(하라)(?=프로세스\s*\()", r"\1\n", prepared)
    units: list[dict[str, Any]] = []
    for line in (part.strip() for part in re.split(r"\n+", prepared)):
        if not line:
            continue
        structural = bool(re.match(r"^(예시|사례|[①-⑳•●▪▶]|-\s*\d|\d+[.)])", line))
        for part in re.split(r"(?<=[.!?。])\s+", line):
            if part.strip():
                units.append(
                    {
                        "text": text(part),
                        "kind": "list_item" if structural else "sentence",
                        "structural": structural,
                    }
                )
    return units


def load_evidence(
    source: Path,
    sheet: str,
    document_id: str,
    document_title: str,
    chapter_no: str | None = None,
    limit: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not source.exists():
        raise FileNotFoundError(f"입력 파일이 없습니다: {source}")
    frame = pd.read_excel(source, sheet_name=sheet)
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"필수 열이 없습니다: {', '.join(sorted(missing))}")
    text_column = next(
        (column for column in TEXT_COLUMNS if column in frame.columns), None
    )
    if text_column is None:
        raise ValueError(
            "본문 열이 없습니다: " + ", ".join(TEXT_COLUMNS)
        )
    repaired = [int(index) + 2 for index in frame.index[frame[META].isna().any(axis=1)]]
    frame[META] = frame[META].ffill().bfill()
    frame = frame[frame[text_column].notna()].copy()
    frame["_row"] = frame.index
    if chapter_no is not None:
        wanted = text(chapter_no).removesuffix(".0")
        frame = frame[frame["chapter_no"].map(_number) == wanted]
    section_index: dict[str, int] = defaultdict(int)
    evidence: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        excel_row = int(row["_row"]) + 2
        chapter, section = _number(row["chapter_no"]), _number(row["section_no"])
        section_title = text(row["section_title"])
        section_id = stable_id("SEC", document_id, chapter, section, section_title, length=12)
        section_path = " > ".join(
            value for value in [text(row["chapter_title"]), section_title] if value
        )
        units = split_units(row[text_column])
        for sub_index, unit in enumerate(units, 1):
            section_index[section_id] += 1
            source_text = unit["text"]
            evidence_id = stable_id("EV", document_id, excel_row, sub_index, source_text)
            structural_fragment = bool(
                re.fullmatch(r"(?:\d+[.)]?|[A-Z][.)]|단계\s*\d+[.]?)", source_text, re.I)
            )
            evidence.append(
                {
                    "evidence_id": evidence_id,
                    "document_id": document_id,
                    "document_title": document_title,
                    "section_id": section_id,
                    "section_path": section_path,
                    "paragraph_id": stable_id("PAR", document_id, excel_row, length=12),
                    "sentence_id": evidence_id,
                    "source_file": str(source.resolve()),
                    "sheet": sheet,
                    "excel_row": excel_row,
                    "source_sub_index": sub_index,
                    "page": _number(row["page"]),
                    "chapter_no": chapter,
                    "chapter_title": text(row["chapter_title"]),
                    "section_no": section,
                    "section_title": section_title,
                    "section_sentence_index": section_index[section_id],
                    "evidence_type": unit["kind"],
                    "evidence_role": (
                        "structural_artifact"
                        if unit["kind"] in {"table", "figure"}
                        else "context_fragment" if structural_fragment else "content"
                    ),
                    "source_text": source_text,
                    "position": {
                        "page": _number(row["page"]),
                        "excel_row": excel_row,
                        "source_sub_index": sub_index,
                    },
                }
            )
    if limit > 0:
        evidence = evidence[:limit]
    if not evidence:
        raise ValueError("처리할 Evidence가 없습니다.")
    included_rows = {item["excel_row"] for item in evidence}
    return evidence, {
        "source_file": str(source.resolve()),
        "sheet": sheet,
        "text_column": text_column,
        "document_id": document_id,
        "evidence_count": len(evidence),
        "section_count": len({item["section_id"] for item in evidence}),
        "source_row_count": len(included_rows),
        "metadata_repaired_excel_rows": [row for row in repaired if row in included_rows],
        "limit": limit,
    }


def build_chunks(
    evidence: list[dict[str, Any]], max_sentences: int = 5, max_chars: int = 1800
) -> list[dict[str, Any]]:
    by_section: dict[str, list[dict[str, Any]]] = defaultdict(list)
    order: list[str] = []
    for item in evidence:
        if item["section_id"] not in by_section:
            order.append(item["section_id"])
        by_section[item["section_id"]].append(item)
    chunks: list[dict[str, Any]] = []
    for section_id in order:
        current: list[dict[str, Any]] = []
        size = 0
        for item in by_section[section_id]:
            standalone = item["evidence_type"] in {"table", "figure"}
            if current and (len(current) >= max_sentences or size + len(item["source_text"]) > max_chars):
                chunks.append(_chunk(current))
                current, size = [], 0
            if standalone and current:
                chunks.append(_chunk(current))
                current, size = [], 0
            current.append(item)
            size += len(item["source_text"])
            if standalone:
                chunks.append(_chunk(current))
                current, size = [], 0
        if current:
            chunks.append(_chunk(current))
    return chunks


def _chunk(items: list[dict[str, Any]]) -> dict[str, Any]:
    first = items[0]
    ids = [item["evidence_id"] for item in items]
    return {
        "chunk_id": stable_id("CH", first["document_id"], first["section_id"], *ids),
        "document_id": first["document_id"],
        "section_id": first["section_id"],
        "section_path": first["section_path"],
        "evidence_ids": ids,
        "sentences": [
            {
                "evidence_id": item["evidence_id"],
                "source_text": item["source_text"],
                "evidence_type": item["evidence_type"],
                "evidence_role": item["evidence_role"],
            }
            for item in items
        ],
    }
