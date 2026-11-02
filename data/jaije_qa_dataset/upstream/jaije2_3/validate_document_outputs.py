#!/usr/bin/env python3
"""Independently check completed Jaije 2/3 outputs against the reviewed Excel cells."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from openpyxl import load_workbook

import generate_qa_dataset as core


def main():
    all_rows = []
    summaries = []
    for number, sheet_name in [(2, '종학 파트'), (3, 'clean_sentences')]:
        source = core.BASE_DIR.parent / f'자이제_{number}.xlsx'
        output = core.BASE_DIR / 'outputs' / f'jaije{number}_terra_v1'
        rows = [json.loads(line) for line in
                (output / 'qa_dataset.jsonl').read_text(encoding='utf-8').splitlines()]
        manifest = json.loads((output / 'generation_manifest.json').read_text())
        assert len(rows) == 150 and manifest['status'] == 'complete'
        assert manifest['settings']['input_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
        workbook = load_workbook(source, read_only=True, data_only=True)
        try:
            iterator = workbook[sheet_name].iter_rows(values_only=True)
            headers = next(iterator)
            excel = {i: dict(zip(headers, values)) for i, values in enumerate(iterator, 2)}
        finally:
            workbook.close()
        for i, row in enumerate(rows, 1):
            assert row['qa_id'] == f'JAIJE{number}-QA-{i:04d}'
            assert row['source_file'] == source.name and row['sheet_name'] == sheet_name
            assert row['generation_model'] == 'gpt-5.6-terra'
            assert row['review_status'] == 'pending' and row['ontology_answerable'] is None
            assert not row['gold_evidence_span_ids']
            assert 1 <= len(row['source_row_ids']) <= 4
            evidence = [excel[j] for j in row['source_row_ids']]
            assert all(str(x.get('사람 검수') or '').strip() for x in evidence)
            assert row['evidence_text'] == [str(x['사람 검수']).strip() for x in evidence]
            pages = sorted({int(x['page']) for x in evidence})
            assert row['page'] == pages and row['page_start'] == min(pages) and row['page_end'] == max(pages)
            for field in ['chapter_no', 'chapter_title', 'section_no', 'section_title']:
                assert all(core.normalize_identifier(x[field]) == row[field] for x in evidence)
            assert row['question'].endswith('?') and 5 <= len(row['question']) <= 250
            assert 5 <= len(row['answer']) <= 500 and 1 <= core.count_sentences(row['answer']) <= 2
        normalized = [core.normalize_question(r['question']) for r in rows]
        assert len(set(normalized)) == 150
        for i, left in enumerate(normalized):
            for right in normalized[i + 1:]:
                assert SequenceMatcher(None, left, right).ratio() < .92
        with (output / 'qa_dataset.csv').open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            assert reader.fieldnames == list(core.CSV_FIELDS)
            csv_rows = list(reader)
        assert len(csv_rows) == 150
        for row, csv_row in zip(rows, csv_rows):
            assert all(csv_row[k] == str(core.csv_value(row[k])) for k in core.CSV_FIELDS)
        events = [json.loads(l) for l in (output / 'generation_log.jsonl').read_text().splitlines()]
        request_count = len((output / 'request_log.jsonl').read_text().splitlines())
        summary = {
            'source_file': source.name, 'sheet_name': sheet_name, 'text_column': '사람 검수',
            'questions': 150, 'source_cell_and_metadata_check': 'pass',
            'within_dataset_duplicate_check': 'pass', 'csv_jsonl_equivalence': 'pass',
            'successful_api_chunks': sum(e['status'] == 'ok' for e in events),
            'request_log_entries': request_count,
            'chapters': dict(Counter(r['chapter_title'] for r in rows)),
            'sections_covered': len({(r['chapter_no'],r['section_no']) for r in rows}),
            'page_range': [min(r['page_start'] for r in rows), max(r['page_end'] for r in rows)],
            'answer_semantic_human_review': 'pending',
            'sha256': {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                       for name in ['qa_dataset.csv', 'qa_dataset.jsonl']},
        }
        core.write_json_atomic(output / 'independent_validation_report.json', summary)
        summaries.append(summary)
        all_rows.append(rows)
    cross_duplicates = []
    for left in all_rows[0]:
        for right in all_rows[1]:
            similarity = SequenceMatcher(None, core.normalize_question(left['question']),
                                          core.normalize_question(right['question'])).ratio()
            if similarity >= .92:
                cross_duplicates.append({'left': left['qa_id'], 'right': right['qa_id'],
                                         'similarity': similarity})
    report = {'validated_at': core.utc_now(), 'datasets': summaries,
              'cross_dataset_near_duplicates': cross_duplicates}
    core.write_json_atomic(core.BASE_DIR / 'outputs' / 'jaije2_3_validation_summary.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
