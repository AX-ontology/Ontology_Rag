#!/usr/bin/env python3
"""Reuse the original jaije-qa-v1 pipeline with reviewed, multi-column workbooks.

source_row_ids always identify physical rows of the selected Excel sheet.
The original prompt, five-shot examples, chunking, schema and QA validators are
imported unchanged. Empty reviewed cells stay excluded, including deleted text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from openpyxl import load_workbook

import generate_qa_dataset as core


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_reviewed(path: Path, sheet_name: str, text_column: str):
    records, provenance, skipped = [], [], []
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = workbook[sheet_name]
        rows = sheet.iter_rows(values_only=True)
        headers = [str(v).strip() if v is not None else "" for v in next(rows)]
        required = [*core.REQUIRED_HEADERS[:-1], text_column]
        if any(headers.count(k) != 1 for k in required):
            raise ValueError(f"Required columns must occur exactly once: {required}")
        for row_id, values in enumerate(rows, 2):
            row = dict(zip(headers, values))
            value = row.get(text_column)
            if value is None or not str(value).strip():
                skipped.append(row_id)
                continue
            if any(row.get(k) is None or not str(row[k]).strip()
                   for k in core.REQUIRED_HEADERS[:-1]):
                raise ValueError(f"Missing source metadata at Excel row {row_id}")
            record = core.SourceRecord(
                source_row_id=row_id,
                page=core.normalize_page(row['page']),
                chapter_no=core.normalize_identifier(row['chapter_no']),
                chapter_title=str(row['chapter_title']).strip(),
                section_no=core.normalize_identifier(row['section_no']),
                section_title=str(row['section_title']).strip(),
                text=str(value).strip(),
            )
            records.append(record)
            provenance.append({
                **record.to_prompt_dict(), 'source_file': path.name,
                'sheet_name': sheet_name, 'text_column': text_column,
                'original_source_row': row.get('source_row'),
                'clean_id': row.get('clean_id'),
                'content_type': row.get('content_type'),
            })
    finally:
        workbook.close()
    if not records:
        raise ValueError('No reviewed source records')
    return records, provenance, skipped


def audit(items, chunks, args, few_shots):
    by_chunk = {c.chunk_id: c for c in chunks}
    blocked = {core.normalize_question(x['expected']['question']) for x in few_shots}
    checked = []
    for i, item in enumerate(items, 1):
        if item['qa_id'] != f'{args.id_prefix}-QA-{i:04d}':
            raise ValueError('QA ID mismatch')
        rebuilt, rejection = core.validate_and_enrich(
            item, by_chunk[item['chunk_id']], source_file=args.input.name,
            sheet_name=args.sheet, model=args.model,
            response_id=item['generation_response_id'], generated_at=item['generated_at'],
        )
        if rejection or rebuilt is None:
            raise ValueError(f"{item['qa_id']}: {rejection}")
        for key in core.CSV_FIELDS:
            if key != 'qa_id' and item[key] != rebuilt[key]:
                raise ValueError(f"{item['qa_id']}: field mismatch: {key}")
        if core.is_near_duplicate(item['question'], checked, blocked, threshold=.92):
            raise ValueError(f"{item['qa_id']}: duplicate question")
        checked.append(item)
    return {
        'validated_questions': len(items), 'target_questions': args.target_size,
        'target_reached': len(items) == args.target_size,
        'source_rows_text_and_metadata_match': True,
        'same_section_evidence': True, 'unique_ids_and_questions': True,
        'near_duplicate_threshold': .92, 'answer_sentences': '1-2',
        'evidence_rows_per_question': '1-4', 'human_review_status': 'pending',
        'ontology_gold_status': 'not_annotated',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--sheet', required=True)
    parser.add_argument('--text-column', default='사람 검수')
    parser.add_argument('--id-prefix', choices=['JAIJE2', 'JAIJE3'], required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model', default=core.DEFAULT_MODEL)
    parser.add_argument('--target-size', type=int, default=150)
    parser.add_argument('--env-file', type=Path, default=core.BASE_DIR / '.env')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.target_size < 1:
        parser.error('--target-size must be positive')
    args.input = args.input.resolve()
    records, provenance, skipped = read_reviewed(args.input, args.sheet, args.text_column)
    few_shots = core.load_few_shots(core.DEFAULT_FEW_SHOTS)
    original = core.read_source_records(core.DEFAULT_INPUT, core.DEFAULT_SHEET)
    core.validate_few_shots_against_source(few_shots, original)
    # Example row numbers belong to Jaije 1, not the new workbook. Exclude by
    # matching example text instead of excluding unrelated rows with the same ID.
    example_texts = {core.collapse_whitespace(r['text'])
                     for x in few_shots for r in x['source_records']}
    excluded = {r.source_row_id for r in records
                if core.collapse_whitespace(r.text) in example_texts}
    chunks = core.chunk_records(records, excluded_row_ids=excluded,
                                max_rows=18, max_chars=5000)
    if not chunks:
        raise ValueError('No generation chunks remain')
    system_prompt = core.DEFAULT_SYSTEM_PROMPT.read_text(encoding='utf-8').strip()
    settings = {
        'input_path': str(args.input), 'input_sha256': digest(args.input),
        'sheet_name': args.sheet, 'text_column': args.text_column,
        'blank_reviewed_policy': 'exclude_without_raw_text_fallback',
        'id_prefix': args.id_prefix, 'model': args.model,
        'target_size': args.target_size, 'qas_per_chunk': 3,
        'max_source_rows': 18, 'max_source_chars': 5000,
        'duplicate_threshold': .92, 'prompt_version': core.PROMPT_VERSION,
        'system_prompt_sha256': hashlib.sha256(system_prompt.encode()).hexdigest(),
        'few_shots_sha256': digest(core.DEFAULT_FEW_SHOTS),
        'few_shot_reference_source_sha256': digest(core.DEFAULT_INPUT),
        'original_generator_sha256': digest(Path(core.__file__)),
        'adapter_sha256': digest(Path(__file__)),
        'source_record_count': len(records), 'chunk_count': len(chunks),
        'blank_reviewed_rows_excluded': skipped,
        'few_shot_matching_rows_excluded': sorted(excluded),
    }
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True,
                                ensure_ascii=False).encode()).hexdigest()
    if args.dry_run:
        print(json.dumps(settings, ensure_ascii=False, indent=2))
        core.print_dry_run(records, chunks, excluded_count=len(excluded),
                           target_size=args.target_size, qas_per_chunk=3)
        return
    core.load_env_file(args.env_file)
    if not os.getenv('OPENAI_API_KEY'):
        raise RuntimeError('Set OPENAI_API_KEY or configure --env-file before generation')
    out = args.output_dir.resolve()
    manifest_path = out / 'generation_manifest.json'
    log_path = out / 'generation_log.jsonl'
    if args.resume:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('fingerprint') != fingerprint:
            raise ValueError('Resume settings or input files have changed')
        processed, accepted = core.load_resume_log(log_path)
    else:
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(f'Output directory is not empty: {out}')
        out.mkdir(parents=True, exist_ok=True)
        manifest = {'fingerprint': fingerprint, 'settings': settings,
                    'started_at': core.utc_now(), 'status': 'running'}
        core.write_json_atomic(manifest_path, manifest)
        core.write_jsonl_atomic(out / 'source_records.jsonl', provenance)
        processed, accepted = set(), []
    OpenAI, response_model = core.build_structured_output_model()
    client = OpenAI(timeout=120.0, max_retries=0)
    blocked = {core.normalize_question(x['expected']['question']) for x in few_shots}
    rejected: Counter[str] = Counter()
    if log_path.exists():
        for line in log_path.read_text(encoding='utf-8').splitlines():
            event = json.loads(line)
            if event.get('status') == 'ok':
                rejected.update(event.get('rejected_reasons', {}))

    def save():
        final = [{**r, 'qa_id': f'{args.id_prefix}-QA-{i:04d}'}
                 for i, r in enumerate(accepted[:args.target_size], 1)]
        validation = audit(final, chunks, args, few_shots)
        core.write_jsonl_atomic(out / 'qa_dataset.jsonl', final)
        core.write_csv_atomic(out / 'qa_dataset.csv', final)
        report = core.build_report(final, source_count=len(records), chunk_count=len(chunks),
            rejected_reasons=rejected, prompt_sha256=settings['system_prompt_sha256'])
        report.update({'source_file': args.input.name, 'sheet_name': args.sheet,
                       'generation_model': args.model, 'completed_chunks': len(processed)})
        core.write_json_atomic(out / 'qa_dataset_report.json', report)
        core.write_json_atomic(out / 'validation_report.json', validation)
        manifest.update({'accepted_count': len(final), 'updated_at': core.utc_now(),
            'status': 'complete' if validation['target_reached'] else 'incomplete',
            'output_sha256': {name: digest(out / name) for name in
                             ['qa_dataset.csv', 'qa_dataset.jsonl']}})
        core.write_json_atomic(manifest_path, manifest)

    try:
        for chunk in chunks:
            if len(accepted) >= args.target_size:
                break
            if chunk.chunk_id in processed:
                continue
            requested = min(3, args.target_size - len(accepted))
            messages = core.build_api_input(system_prompt, few_shots, chunk, requested)
            print(f'[{args.id_prefix}/{chunk.chunk_id}] accepted={len(accepted)}/{args.target_size}', flush=True)
            # Preserve exact request payloads for auditing and resumption.
            core.append_jsonl(out / 'request_log.jsonl', {
                'chunk_id': chunk.chunk_id, 'model': args.model,
                'input': messages, 'requested': requested, 'started_at': core.utc_now(),
            })
            response_id, candidates = core.call_openai(client, response_model,
                model=args.model, input_messages=messages, max_retries=2, retry_base_seconds=2.)
            now = core.utc_now()
            additions, chunk_rejected = [], Counter()
            for candidate in candidates[:requested]:
                item, reason = core.validate_and_enrich(candidate, chunk,
                    source_file=args.input.name, sheet_name=args.sheet, model=args.model,
                    response_id=response_id, generated_at=now)
                if reason or item is None:
                    chunk_rejected[reason or 'validation_error'] += 1
                elif core.is_near_duplicate(item['question'], [*accepted, *additions],
                                            blocked, threshold=.92):
                    chunk_rejected['duplicate_question'] += 1
                else:
                    additions.append(item)
            core.append_jsonl(log_path, {'status': 'ok', 'chunk_id': chunk.chunk_id,
                'response_id': response_id, 'candidate_count': len(candidates),
                'candidates': candidates, 'accepted': additions,
                'rejected_reasons': dict(chunk_rejected), 'logged_at': now})
            accepted.extend(additions)
            rejected.update(chunk_rejected)
            processed.add(chunk.chunk_id)
            save()
    except Exception as exc:
        core.append_jsonl(log_path, {'status': 'error', 'error_type': type(exc).__name__,
                                    'logged_at': core.utc_now()})
        save()
        raise
    finally:
        client.close()
    save()
    if len(accepted) != args.target_size:
        raise RuntimeError(f'Incomplete dataset: {len(accepted)}/{args.target_size}; inspect logs')
    print(f'Completed and validated: {out}', flush=True)


if __name__ == '__main__':
    main()
