#!/usr/bin/env python3
"""Validate and concatenate the three source-grounded 150-QA datasets; no API calls."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

BASE = Path(__file__).resolve().parent
INPUTS = [BASE, BASE / 'jaije2_3/outputs/jaije2_terra_v1',
          BASE / 'jaije2_3/outputs/jaije3_terra_v1']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def csv_value(value):
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return '' if value is None else str(value)


def normalized_question(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text).lower() if c.isalnum())


def build(output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output directory must be empty: {output}')
    combined, input_info, columns = [], [], None
    for number, folder in enumerate(INPUTS, 1):
        jsonl_path, csv_path = folder / 'qa_dataset.jsonl', folder / 'qa_dataset.csv'
        rows = [json.loads(line) for line in jsonl_path.read_text(encoding='utf-8').splitlines()]
        with csv_path.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            csv_rows, names = list(reader), reader.fieldnames
        if columns is None:
            columns = names
        if names != columns or len(rows) != 150 or len(csv_rows) != 150:
            raise ValueError(f'Unexpected schema or count: {folder}')
        for index, (row, csv_row) in enumerate(zip(rows, csv_rows), 1):
            if set(row) != set(columns):
                raise ValueError('JSONL schema differs from CSV schema')
            if {k: csv_value(row[k]) for k in columns} != csv_row:
                raise ValueError(f'CSV/JSONL mismatch: {row["qa_id"]}')
            if row['qa_id'] != f'JAIJE{number}-QA-{index:04d}':
                raise ValueError('Unexpected QA ID')
            if row['source_file'] != f'자이제_{number}.xlsx':
                raise ValueError('Unexpected source file')
            if row['review_status'] != 'pending' or row['split'] != 'unassigned':
                raise ValueError('Review/split policy changed; review before merging')
            if not 1 <= len(row['source_row_ids']) <= 4:
                raise ValueError('Invalid evidence row count')
            if len(row['evidence_text']) != len(row['source_row_ids']):
                raise ValueError('Evidence IDs and text do not correspond')
        combined.extend(rows)
        input_info.append({
            'source_file': rows[0]['source_file'], 'sheet_name': rows[0]['sheet_name'],
            'text_column': 'text_for_llm_human' if number == 1 else '사람 검수',
            'questions': len(rows), 'id_first': rows[0]['qa_id'], 'id_last': rows[-1]['qa_id'],
            'csv': csv_path.relative_to(BASE).as_posix(), 'csv_sha256': sha(csv_path),
            'jsonl': jsonl_path.relative_to(BASE).as_posix(), 'jsonl_sha256': sha(jsonl_path),
        })
    if len(combined) != 450 or len({r['qa_id'] for r in combined}) != 450:
        raise ValueError('Combined count or ID uniqueness check failed')
    norm = [normalized_question(r['question']) for r in combined]
    duplicates = []
    for i, a in enumerate(norm):
        if not a:
            raise ValueError('Empty question')
        for j in range(i + 1, len(norm)):
            b = norm[j]
            if 2 * min(len(a), len(b)) / (len(a) + len(b)) < .92:
                continue
            score = SequenceMatcher(None, a, b).ratio()
            if score >= .92:
                duplicates.append({'left': combined[i]['qa_id'], 'right': combined[j]['qa_id'],
                                   'similarity': score})
    # Keep records intact. Potential duplicate pairs are reported, never silently dropped.
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'qa_dataset.jsonl').open('w', encoding='utf-8') as handle:
        for row in combined:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    with (output / 'qa_dataset.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: csv_value(r[k]) for k in columns} for r in combined)
    notes_path = INPUTS[1] / 'spot_check_notes.json'
    notes = json.loads(notes_path.read_text(encoding='utf-8'))
    notes['source_notes_path'] = notes_path.relative_to(BASE).as_posix()
    notes['source_notes_sha256'] = sha(notes_path)
    notes['scope'] = 'Inherited Jaije 2 spot checks only; all 450 questions await human semantic review.'
    write_json(output / 'review_notes.json', notes)
    report = {
        'dataset_size': len(combined), 'columns': len(columns),
        'source_counts': dict(Counter(r['source_file'] for r in combined)),
        'chapter_counts': dict(Counter(r['chapter_title'] for r in combined)),
        'question_types': dict(Counter(r['question_type'] for r in combined)),
        'difficulties': dict(Counter(r['difficulty'] for r in combined)),
        'evidence_modes': dict(Counter(r['evidence_mode'] for r in combined)),
        'unique_chapter_sections': len({(r['chapter_no'], r['section_no']) for r in combined}),
        'unique_source_sheet_sections': len({(r['source_file'], r['sheet_name'], r['chapter_no'],
                                            r['section_no']) for r in combined}),
        'page_range': [min(r['page_start'] for r in combined), max(r['page_end'] for r in combined)],
        'review_status': dict(Counter(r['review_status'] for r in combined)),
        'unique_qa_ids': len({r['qa_id'] for r in combined}),
        'unique_normalized_questions': len(set(norm)),
        'near_duplicate_threshold': .92, 'near_duplicate_pairs': duplicates,
        'input_csv_jsonl_equivalence': 'pass', 'preserved_original_records': True,
        'inherited_review_note_count': len(notes['notes']),
        'independent_test_split': False, 'graph_gold_annotated': False,
    }
    write_json(output / 'validation_report.json', report)
    manifest = {
        'dataset_name': 'jaije_1_2_3_source_grounded_qa_450', 'version': '2026-09-30-v1',
        'built_at': datetime.now(timezone.utc).isoformat(),
        'build_method': 'Validated concatenation, Jaije 1 then 2 then 3; no generation API calls',
        'builder_sha256': sha(Path(__file__)), 'inputs': input_info,
        'questions': 450, 'synthetic_scenario_extension_questions': 0,
        'original_ids_and_fields_preserved': True, 'original_generation_model': 'gpt-5.6-terra',
        'output_sha256': {name: sha(output / name) for name in
            ['qa_dataset.csv', 'qa_dataset.jsonl', 'validation_report.json', 'review_notes.json']},
        'limitations': [
            'Final integration version, not expert-approved Gold data.',
            'All questions are source-derived; reference answers await human review.',
            'Source 1 includes questions observed in previous retrieval experiments.',
            'No independent train/dev/test split is established by this merge.',
        ],
    }
    write_json(output / 'manifest.json', manifest)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=BASE / 'final_450')
    build(parser.parse_args().output_dir.resolve())
