#!/usr/bin/env python3
"""자이제_1.xlsx에서 근거 추적 가능한 한국어 QA 평가셋을 생성한다.

OpenAI API를 호출하는 실제 생성 모드와, API 없이 원본·청크 구성을 점검하는
--dry-run 모드를 제공한다. 출처 메타데이터와 evidence_text는 모델이 만들지
않고 source_row_ids를 이용해 Excel 원본에서 결정적으로 복원한다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal, Sequence

from openpyxl import load_workbook


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR.parent / "자이제_1.xlsx"
DEFAULT_SHEET = "clean_sentences"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"
DEFAULT_SYSTEM_PROMPT = BASE_DIR / "prompts" / "system_prompt.txt"
DEFAULT_FEW_SHOTS = BASE_DIR / "prompts" / "few_shot_examples.json"
DEFAULT_MODEL = "gpt-5.6-terra"
PROMPT_VERSION = "jaije-qa-v1"

REQUIRED_HEADERS = (
    "page",
    "chapter_no",
    "chapter_title",
    "section_no",
    "section_title",
    "text_for_llm_human",
)

QUESTION_TYPES = (
    "fact",
    "numeric",
    "definition",
    "reason",
    "method",
    "comparison",
    "procedure",
    "misconception",
)

DIFFICULTIES = ("easy", "medium", "hard")

CSV_FIELDS = (
    "qa_id",
    "question",
    "answer",
    "chapter_no",
    "chapter_title",
    "section_no",
    "section_title",
    "page",
    "page_start",
    "page_end",
    "evidence_text",
    "source_row_ids",
    "question_type",
    "difficulty",
    "answerable",
    "source_answerable",
    "ontology_answerable",
    "expected_behavior",
    "unanswerable_reason",
    "ontology_gap_reason",
    "evidence_mode",
    "gold_taskframe",
    "gold_fact_ids",
    "gold_evidence_span_ids",
    "gold_node_ids",
    "gold_relation_types",
    "gold_path",
    "min_hops",
    "split",
    "review_status",
    "reviewer",
    "review_notes",
    "source_file",
    "sheet_name",
    "chunk_id",
    "generation_model",
    "generation_response_id",
    "prompt_version",
    "generated_at",
)


@dataclass(frozen=True)
class SourceRecord:
    source_row_id: int
    page: int
    chapter_no: str
    chapter_title: str
    section_no: str
    section_title: str
    text: str

    @property
    def section_key(self) -> tuple[str, str, str, str]:
        return (
            self.chapter_no,
            self.chapter_title,
            self.section_no,
            self.section_title,
        )

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "source_row_id": self.source_row_id,
            "page": self.page,
            "chapter_no": self.chapter_no,
            "chapter_title": self.chapter_title,
            "section_no": self.section_no,
            "section_title": self.section_title,
            "text": self.text,
        }


@dataclass(frozen=True)
class SourceChunk:
    chunk_id: str
    records: tuple[SourceRecord, ...]

    @property
    def section_key(self) -> tuple[str, str, str, str]:
        return self.records[0].section_key

    @property
    def pages(self) -> list[int]:
        return sorted({record.page for record in self.records})


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_env_file(path: Path) -> None:
    """python-dotenv가 있으면 로컬 .env를 로드한다."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(path)


def normalize_identifier(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return format(value, "g")
    return str(value).strip()


def normalize_page(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"유효하지 않은 page 값: {value!r}")
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric.is_integer():
            return int(numeric)
    text = str(value).strip()
    try:
        numeric = float(text)
    except ValueError as exc:
        raise ValueError(f"숫자로 변환할 수 없는 page 값: {value!r}") from exc
    if not numeric.is_integer():
        raise ValueError(f"정수가 아닌 page 값: {value!r}")
    return int(numeric)


def read_source_records(path: Path, sheet_name: str) -> list[SourceRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Excel 파일을 찾을 수 없습니다: {path}")

    workbook = load_workbook(path, read_only=True, data_only=True)
    if sheet_name not in workbook.sheetnames:
        raise ValueError(
            f"시트 {sheet_name!r}가 없습니다. 사용 가능한 시트: {workbook.sheetnames}"
        )
    sheet = workbook[sheet_name]

    headers = tuple(
        "" if cell.value is None else str(cell.value).strip()
        for cell in next(sheet.iter_rows(min_row=1, max_row=1, max_col=6))
    )
    if headers != REQUIRED_HEADERS:
        raise ValueError(
            "Excel 헤더가 예상과 다릅니다.\n"
            f"예상: {REQUIRED_HEADERS}\n"
            f"실제: {headers}"
        )

    records: list[SourceRecord] = []
    for row_id, values in enumerate(
        sheet.iter_rows(min_row=2, max_col=6, values_only=True),
        start=2,
    ):
        page, chapter_no, chapter_title, section_no, section_title, text = values
        if text is None or not str(text).strip():
            continue
        metadata = (page, chapter_no, chapter_title, section_no, section_title)
        if any(value is None or not str(value).strip() for value in metadata):
            raise ValueError(f"원본 {row_id}행의 출처 메타데이터가 비어 있습니다.")
        records.append(
            SourceRecord(
                source_row_id=row_id,
                page=normalize_page(page),
                chapter_no=normalize_identifier(chapter_no),
                chapter_title=str(chapter_title).strip(),
                section_no=normalize_identifier(section_no),
                section_title=str(section_title).strip(),
                text=str(text).strip(),
            )
        )
    workbook.close()
    if not records:
        raise ValueError("유효한 본문 행을 찾지 못했습니다.")
    return records


def numeric_path_key(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for part in value.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            parts.append(999_999)
    return tuple(parts)


def chunk_records(
    records: Sequence[SourceRecord],
    *,
    excluded_row_ids: set[int],
    max_rows: int,
    max_chars: int,
) -> list[SourceChunk]:
    grouped: dict[tuple[str, str, str, str], list[SourceRecord]] = defaultdict(list)
    for record in records:
        if record.source_row_id not in excluded_row_ids:
            grouped[record.section_key].append(record)

    chunks_by_section: dict[
        tuple[str, str, str, str], list[tuple[SourceRecord, ...]]
    ] = {}
    for section_key, section_records in grouped.items():
        section_records.sort(key=lambda item: item.source_row_id)
        section_chunks: list[tuple[SourceRecord, ...]] = []
        current: list[SourceRecord] = []
        current_chars = 0
        for record in section_records:
            record_chars = len(record.text)
            exceeds = current and (
                len(current) >= max_rows or current_chars + record_chars > max_chars
            )
            if exceeds:
                section_chunks.append(tuple(current))
                current = []
                current_chars = 0
            current.append(record)
            current_chars += record_chars
        if current:
            section_chunks.append(tuple(current))
        chunks_by_section[section_key] = section_chunks

    sections_by_chapter: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
    for section_key in chunks_by_section:
        sections_by_chapter[section_key[0]].append(section_key)
    for chapter_no in sections_by_chapter:
        sections_by_chapter[chapter_no].sort(
            key=lambda key: numeric_path_key(key[2])
        )

    chapter_order = sorted(sections_by_chapter, key=numeric_path_key)
    interleaved_sections: list[tuple[str, str, str, str]] = []
    max_sections = max(len(value) for value in sections_by_chapter.values())
    for section_index in range(max_sections):
        for chapter_no in chapter_order:
            chapter_sections = sections_by_chapter[chapter_no]
            if section_index < len(chapter_sections):
                interleaved_sections.append(chapter_sections[section_index])

    ordered_raw_chunks: list[tuple[SourceRecord, ...]] = []
    max_rounds = max(len(value) for value in chunks_by_section.values())
    for chunk_round in range(max_rounds):
        for section_key in interleaved_sections:
            section_chunks = chunks_by_section[section_key]
            if chunk_round < len(section_chunks):
                ordered_raw_chunks.append(section_chunks[chunk_round])

    return [
        SourceChunk(chunk_id=f"CHUNK-{index:04d}", records=raw_chunk)
        for index, raw_chunk in enumerate(ordered_raw_chunks, start=1)
    ]


def load_few_shots(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list) or not payload:
        raise ValueError("few-shot 파일은 비어 있지 않은 JSON 배열이어야 합니다.")
    return payload


def collapse_whitespace(value: Any) -> str:
    return " ".join(str(value).split())


def validate_few_shots_against_source(
    examples: Sequence[dict[str, Any]],
    records: Sequence[SourceRecord],
) -> None:
    source_by_id = {record.source_row_id: record for record in records}
    seen_questions: set[str] = set()
    for example in examples:
        name = str(example.get("name") or "unnamed")
        prompt_records = example.get("source_records")
        expected = example.get("expected")
        if not isinstance(prompt_records, list) or not isinstance(expected, dict):
            raise ValueError(f"few-shot {name!r}의 구조가 올바르지 않습니다.")
        prompt_ids: set[int] = set()
        for prompt_record in prompt_records:
            row_id = int(prompt_record["source_row_id"])
            prompt_ids.add(row_id)
            actual = source_by_id.get(row_id)
            if actual is None:
                raise ValueError(
                    f"few-shot {name!r}의 원본 행 {row_id}을 Excel에서 찾지 못했습니다."
                )
            expected_metadata = (
                normalize_page(prompt_record["page"]),
                normalize_identifier(prompt_record["chapter_no"]),
                str(prompt_record["chapter_title"]).strip(),
                normalize_identifier(prompt_record["section_no"]),
                str(prompt_record["section_title"]).strip(),
            )
            actual_metadata = (
                actual.page,
                actual.chapter_no,
                actual.chapter_title,
                actual.section_no,
                actual.section_title,
            )
            if expected_metadata != actual_metadata:
                raise ValueError(
                    f"few-shot {name!r}의 {row_id}행 메타데이터가 Excel과 다릅니다."
                )
            if collapse_whitespace(prompt_record["text"]) != collapse_whitespace(
                actual.text
            ):
                raise ValueError(
                    f"few-shot {name!r}의 {row_id}행 본문이 Excel과 다릅니다."
                )

        expected_ids = {int(value) for value in expected.get("source_row_ids", [])}
        if not expected_ids or not expected_ids.issubset(prompt_ids):
            raise ValueError(
                f"few-shot {name!r}의 정답 source_row_ids가 입력 근거와 맞지 않습니다."
            )
        question = str(expected.get("question") or "").strip()
        answer = str(expected.get("answer") or "").strip()
        normalized_question = normalize_question(question)
        if not question.endswith("?") or not normalized_question:
            raise ValueError(f"few-shot {name!r}의 질문 형식이 올바르지 않습니다.")
        if normalized_question in seen_questions:
            raise ValueError(f"few-shot 질문이 중복되었습니다: {question}")
        seen_questions.add(normalized_question)
        if not 1 <= count_sentences(answer) <= 2:
            raise ValueError(f"few-shot {name!r}의 답변은 1~2문장이어야 합니다.")


def few_shot_row_ids(examples: Sequence[dict[str, Any]]) -> set[int]:
    return {
        int(record["source_row_id"])
        for example in examples
        for record in example["source_records"]
    }


def format_generation_request(
    source_records: Sequence[dict[str, Any]],
    target_count: int,
    *,
    chunk_id: str | None = None,
) -> str:
    label = f"chunk_id={chunk_id}\n" if chunk_id else ""
    context = json.dumps(source_records, ensure_ascii=False, indent=2)
    return (
        f"{label}아래 SOURCE_RECORDS에서 최대 {target_count}개의 QA를 생성하세요. "
        "충분히 근거가 있는 문항만 반환하세요.\n\n"
        f"SOURCE_RECORDS:\n{context}"
    )


def build_api_input(
    system_prompt: str,
    few_shots: Sequence[dict[str, Any]],
    chunk: SourceChunk,
    target_count: int,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt}
    ]
    for example in few_shots:
        messages.append(
            {
                "role": "user",
                "content": format_generation_request(
                    example["source_records"],
                    1,
                ),
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {"items": [example["expected"]]},
                    ensure_ascii=False,
                ),
            }
        )
    messages.append(
        {
            "role": "user",
            "content": format_generation_request(
                [record.to_prompt_dict() for record in chunk.records],
                target_count,
                chunk_id=chunk.chunk_id,
            ),
        }
    )
    return messages


def build_structured_output_model() -> tuple[Any, Any]:
    try:
        from openai import OpenAI
        from pydantic import BaseModel, ConfigDict, Field
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI 생성 모드에 필요한 패키지가 없습니다. "
            "python -m pip install -r requirements.txt 를 실행하세요."
        ) from exc

    class GeneratedQA(BaseModel):
        model_config = ConfigDict(extra="forbid")

        question: str = Field(min_length=5, max_length=250)
        answer: str = Field(min_length=5, max_length=500)
        source_row_ids: list[int] = Field(min_length=1, max_length=4)
        question_type: Literal[
            "fact",
            "numeric",
            "definition",
            "reason",
            "method",
            "comparison",
            "procedure",
            "misconception",
        ]
        difficulty: Literal["easy", "medium", "hard"]

    class QABatch(BaseModel):
        model_config = ConfigDict(extra="forbid")

        items: list[GeneratedQA] = Field(max_length=6)

    QABatch.model_rebuild(_types_namespace={"GeneratedQA": GeneratedQA})
    return OpenAI, QABatch


def call_openai(
    client: Any,
    response_model: Any,
    *,
    model: str,
    input_messages: list[dict[str, str]],
    max_retries: int,
    retry_base_seconds: float,
) -> tuple[str, list[dict[str, Any]]]:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = client.responses.parse(
                model=model,
                input=input_messages,
                text_format=response_model,
            )
            parsed = response.output_parsed
            if parsed is None:
                raise RuntimeError(
                    "Structured Output이 비어 있습니다. 모델 거절 또는 불완전 응답을 확인하세요."
                )
            return response.id, [item.model_dump() for item in parsed.items]
        except Exception as exc:  # SDK 오류 유형 전체에 동일한 bounded retry 적용
            last_error = exc
            if attempt >= max_retries:
                break
            delay = retry_base_seconds * (2**attempt)
            print(
                f"  API 요청 실패: {type(exc).__name__}; "
                f"{delay:.1f}초 후 재시도 {attempt + 1}/{max_retries}"
            )
            time.sleep(delay)
    assert last_error is not None
    raise last_error


def count_sentences(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    boundaries = re.findall(r"[.!?]+(?=\s|$)", stripped)
    return max(1, len(boundaries))


def normalize_question(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    return "".join(character for character in normalized if character.isalnum())


def is_near_duplicate(
    question: str,
    accepted: Sequence[dict[str, Any]],
    blocked_questions: set[str],
    *,
    threshold: float,
) -> bool:
    normalized = normalize_question(question)
    if not normalized or normalized in blocked_questions:
        return True
    for item in accepted:
        existing = normalize_question(str(item["question"]))
        if normalized == existing:
            return True
        if SequenceMatcher(None, normalized, existing).ratio() >= threshold:
            return True
    return False


def validate_and_enrich(
    candidate: dict[str, Any],
    chunk: SourceChunk,
    *,
    source_file: str,
    sheet_name: str,
    model: str,
    response_id: str,
    generated_at: str,
) -> tuple[dict[str, Any] | None, str | None]:
    question = str(candidate.get("question") or "").strip()
    answer = str(candidate.get("answer") or "").strip()
    if not question.endswith("?"):
        return None, "question_without_question_mark"
    if not 5 <= len(question) <= 250:
        return None, "invalid_question_length"
    if not 5 <= len(answer) <= 500:
        return None, "invalid_answer_length"
    if not 1 <= count_sentences(answer) <= 2:
        return None, "answer_not_one_or_two_sentences"

    try:
        row_ids = [int(value) for value in candidate.get("source_row_ids", [])]
    except (TypeError, ValueError):
        return None, "invalid_source_row_ids"
    row_ids = list(dict.fromkeys(row_ids))
    if not 1 <= len(row_ids) <= 4:
        return None, "invalid_source_row_count"

    chunk_by_id = {record.source_row_id: record for record in chunk.records}
    if any(row_id not in chunk_by_id for row_id in row_ids):
        return None, "source_row_outside_chunk"
    evidence_records = [chunk_by_id[row_id] for row_id in row_ids]
    if len({record.section_key for record in evidence_records}) != 1:
        return None, "cross_section_evidence"

    question_type = str(candidate.get("question_type") or "")
    difficulty = str(candidate.get("difficulty") or "")
    if question_type not in QUESTION_TYPES:
        return None, "invalid_question_type"
    if difficulty not in DIFFICULTIES:
        return None, "invalid_difficulty"

    first = evidence_records[0]
    pages = sorted({record.page for record in evidence_records})
    contains_table = any("|" in record.text for record in evidence_records)
    if contains_table:
        evidence_mode = "table"
    elif len(evidence_records) == 1:
        evidence_mode = "single_sentence"
    else:
        evidence_mode = "multi_sentence"

    return (
        {
            "qa_id": "",
            "question": question,
            "answer": answer,
            "chapter_no": first.chapter_no,
            "chapter_title": first.chapter_title,
            "section_no": first.section_no,
            "section_title": first.section_title,
            "page": pages,
            "page_start": min(pages),
            "page_end": max(pages),
            "evidence_text": [record.text for record in evidence_records],
            "source_row_ids": row_ids,
            "question_type": question_type,
            "difficulty": difficulty,
            "answerable": True,
            "source_answerable": True,
            "ontology_answerable": None,
            "expected_behavior": "answer_with_evidence",
            "unanswerable_reason": "",
            "ontology_gap_reason": "",
            "evidence_mode": evidence_mode,
            "gold_taskframe": None,
            "gold_fact_ids": [],
            "gold_evidence_span_ids": [],
            "gold_node_ids": [],
            "gold_relation_types": [],
            "gold_path": [],
            "min_hops": None,
            "split": "unassigned",
            "review_status": "pending",
            "reviewer": "",
            "review_notes": "",
            "source_file": source_file,
            "sheet_name": sheet_name,
            "chunk_id": chunk.chunk_id,
            "generation_model": model,
            "generation_response_id": response_id,
            "prompt_version": PROMPT_VERSION,
            "generated_at": generated_at,
        },
        None,
    )


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_resume_log(path: Path) -> tuple[set[str], list[dict[str, Any]]]:
    processed: set[str] = set()
    accepted: list[dict[str, Any]] = []
    if not path.exists():
        return processed, accepted
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"재개 로그 {path}의 {line_number}행이 잘못된 JSON입니다."
                ) from exc
            if payload.get("status") != "ok":
                continue
            processed.add(str(payload["chunk_id"]))
            accepted.extend(payload.get("accepted", []))
    return processed, accepted


def with_stable_ids(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {**record, "qa_id": f"JAIJE1-QA-{index:04d}"}
        for index, record in enumerate(records, start=1)
    ]


def write_jsonl_atomic(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return value


def write_csv_atomic(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({key: csv_value(record.get(key)) for key in CSV_FIELDS})
    temporary.replace(path)


def build_report(
    records: Sequence[dict[str, Any]],
    *,
    source_count: int,
    chunk_count: int,
    rejected_reasons: Counter[str],
    prompt_sha256: str,
) -> dict[str, Any]:
    return {
        "dataset_size": len(records),
        "source_row_count": source_count,
        "chunk_count": chunk_count,
        "chapters": dict(Counter(item["chapter_title"] for item in records)),
        "sections_covered": len(
            {(item["chapter_no"], item["section_no"]) for item in records}
        ),
        "question_types": dict(Counter(item["question_type"] for item in records)),
        "difficulties": dict(Counter(item["difficulty"] for item in records)),
        "evidence_modes": dict(Counter(item["evidence_mode"] for item in records)),
        "rejected_reasons": dict(rejected_reasons),
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha256,
        "generated_at": utc_now(),
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def print_dry_run(
    records: Sequence[SourceRecord],
    chunks: Sequence[SourceChunk],
    *,
    excluded_count: int,
    target_size: int,
    qas_per_chunk: int,
) -> None:
    print("[Dry run: API를 호출하지 않습니다]")
    print(f"유효 원본 행: {len(records)}")
    print(f"few-shot 전용 제외 행: {excluded_count}")
    print(f"장 수: {len({record.chapter_no for record in records})}")
    print(f"절 수: {len({record.section_key for record in records})}")
    print(f"생성 청크 수: {len(chunks)}")
    estimated = min(len(chunks), (target_size + qas_per_chunk - 1) // qas_per_chunk)
    print(f"최소 예상 API 호출 수: 약 {estimated}회(모델이 매번 최대 개수 반환 시)")
    print("첫 8개 청크:")
    for chunk in chunks[:8]:
        first = chunk.records[0]
        print(
            f"  {chunk.chunk_id}: chapter={first.chapter_no} "
            f"section={first.section_no} pages={chunk.pages} rows={len(chunk.records)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="자이제 Excel 원문에서 근거 추적 가능한 QA 평가셋을 생성합니다."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--sheet", default=DEFAULT_SHEET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--system-prompt", type=Path, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--few-shots", type=Path, default=DEFAULT_FEW_SHOTS)
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--target-size", type=int, default=150)
    parser.add_argument("--qas-per-chunk", type=int, default=3)
    parser.add_argument("--max-source-rows", type=int, default=18)
    parser.add_argument("--max-source-chars", type=int, default=5_000)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-base-seconds", type=float, default=2.0)
    parser.add_argument("--duplicate-threshold", type=float, default=0.92)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.target_size <= 0:
        raise ValueError("--target-size는 1 이상이어야 합니다.")
    if not 1 <= args.qas_per_chunk <= 6:
        raise ValueError("--qas-per-chunk는 1~6이어야 합니다.")
    if args.max_source_rows <= 0 or args.max_source_chars <= 0:
        raise ValueError("청크 크기 제한은 양수여야 합니다.")
    if args.max_retries < 0 or args.retry_base_seconds <= 0:
        raise ValueError("재시도 설정이 올바르지 않습니다.")
    if not 0.0 < args.duplicate_threshold <= 1.0:
        raise ValueError("--duplicate-threshold는 0보다 크고 1 이하여야 합니다.")


def main() -> None:
    load_env_file(BASE_DIR / ".env")
    args = parse_args()
    validate_args(args)

    records = read_source_records(args.input.resolve(), args.sheet)
    few_shots = load_few_shots(args.few_shots.resolve())
    validate_few_shots_against_source(few_shots, records)
    excluded_ids = few_shot_row_ids(few_shots)
    chunks = chunk_records(
        records,
        excluded_row_ids=excluded_ids,
        max_rows=args.max_source_rows,
        max_chars=args.max_source_chars,
    )
    if args.dry_run:
        print_dry_run(
            records,
            chunks,
            excluded_count=len(excluded_ids),
            target_size=args.target_size,
            qas_per_chunk=args.qas_per_chunk,
        )
        return

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY가 없습니다. 노출된 기존 키가 아닌 새 키를 .env에 저장하세요."
        )

    system_prompt = args.system_prompt.read_text(encoding="utf-8").strip()
    prompt_sha256 = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
    OpenAI, response_model = build_structured_output_model()
    client = OpenAI(timeout=120.0)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "generation_log.jsonl"
    if args.resume:
        processed_chunk_ids, accepted = load_resume_log(log_path)
        print(
            f"재개: 완료 청크 {len(processed_chunk_ids)}개, "
            f"복원 QA {len(accepted)}개"
        )
    else:
        processed_chunk_ids, accepted = set(), []
        if log_path.exists():
            raise FileExistsError(
                f"기존 생성 로그가 있습니다: {log_path}\n"
                "이어 하려면 --resume을 사용하거나 outputs를 별도 보관하세요."
            )

    blocked_questions = {
        normalize_question(str(example["expected"]["question"]))
        for example in few_shots
    }
    rejected_reasons: Counter[str] = Counter()

    for chunk in chunks:
        if len(accepted) >= args.target_size:
            break
        if chunk.chunk_id in processed_chunk_ids:
            continue
        remaining = args.target_size - len(accepted)
        requested = min(args.qas_per_chunk, remaining)
        print(
            f"[{chunk.chunk_id}] section={chunk.records[0].section_no} "
            f"rows={len(chunk.records)} 요청={requested}"
        )
        messages = build_api_input(system_prompt, few_shots, chunk, requested)
        started = time.perf_counter()
        try:
            response_id, candidates = call_openai(
                client,
                response_model,
                model=args.model,
                input_messages=messages,
                max_retries=args.max_retries,
                retry_base_seconds=args.retry_base_seconds,
            )
        except Exception as exc:
            append_jsonl(
                log_path,
                {
                    "status": "error",
                    "chunk_id": chunk.chunk_id,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "logged_at": utc_now(),
                },
            )
            print(f"  최종 실패: {type(exc).__name__}: {exc}")
            continue

        generated_at = utc_now()
        accepted_this_chunk: list[dict[str, Any]] = []
        for candidate in candidates[:requested]:
            enriched, rejection = validate_and_enrich(
                candidate,
                chunk,
                source_file=args.input.name,
                sheet_name=args.sheet,
                model=args.model,
                response_id=response_id,
                generated_at=generated_at,
            )
            if rejection is not None or enriched is None:
                rejected_reasons[rejection or "unknown_validation_error"] += 1
                continue
            if is_near_duplicate(
                enriched["question"],
                [*accepted, *accepted_this_chunk],
                blocked_questions,
                threshold=args.duplicate_threshold,
            ):
                rejected_reasons["duplicate_question"] += 1
                continue
            accepted_this_chunk.append(enriched)

        accepted.extend(accepted_this_chunk)
        elapsed_ms = round((time.perf_counter() - started) * 1_000, 1)
        append_jsonl(
            log_path,
            {
                "status": "ok",
                "chunk_id": chunk.chunk_id,
                "response_id": response_id,
                "candidate_count": len(candidates),
                "accepted": accepted_this_chunk,
                "elapsed_ms": elapsed_ms,
                "logged_at": utc_now(),
            },
        )
        print(
            f"  후보={len(candidates)} 채택={len(accepted_this_chunk)} "
            f"누적={len(accepted)}/{args.target_size}"
        )

    final_records = with_stable_ids(accepted[: args.target_size])
    jsonl_path = output_dir / "qa_dataset.jsonl"
    csv_path = output_dir / "qa_dataset.csv"
    report_path = output_dir / "qa_dataset_report.json"
    write_jsonl_atomic(jsonl_path, final_records)
    write_csv_atomic(csv_path, final_records)
    report = build_report(
        final_records,
        source_count=len(records),
        chunk_count=len(chunks),
        rejected_reasons=rejected_reasons,
        prompt_sha256=prompt_sha256,
    )
    write_json_atomic(report_path, report)

    print("\n[완료]")
    print(f"QA 수: {len(final_records)}")
    print(f"JSONL: {jsonl_path}")
    print(f"CSV: {csv_path}")
    print(f"리포트: {report_path}")
    if len(final_records) < args.target_size:
        print(
            "주의: 목표 개수보다 적습니다. 생성 로그와 rejection 통계를 검토한 뒤 "
            "청크 또는 목표값을 조정하세요."
        )


if __name__ == "__main__":
    main()
