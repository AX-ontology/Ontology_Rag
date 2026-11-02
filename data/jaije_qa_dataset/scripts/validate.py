#!/usr/bin/env python3
"""Validate the packaged QA collection without models or network access."""
from pathlib import Path
from collections import Counter
import hashlib
import json

root = Path(__file__).resolve().parents[1]
read = lambda name: json.loads((root / name).read_text())
qa, corpus = read('qa.json'), read('corpus.json')
gold, split = read('evaluation/gold.json'), read('evaluation/split.json')
stats, provenance = read('statistics.json'), read('provenance.json')
qids = [q['qa_id'] for q in qa]
assert len(qids) == len(set(qids)) == 450
assert set(qids) == set(gold) == set(split)
assert len(corpus['ids']) == len(set(corpus['ids'])) == 3995
assert set(corpus['ids']) == set(corpus['evidence'])
assert Counter(split.values()) == {'dev': 150, 'test': 300}
for q in qa:
    assert q['question'] and q['answer']
    assert q['evidence_ids'] == gold[q['qa_id']]
    assert q['split'] == split[q['qa_id']]
    assert all(e in corpus['evidence'] for e in q['evidence_ids'])
assert dict(Counter(q['question_type'] for q in qa)) == stats['question_categories']
assert dict(Counter(str(len(q['evidence_ids'])) for q in qa)) == stats['evidence_rows_per_question']
assert dict(Counter(e['source_file'] for e in corpus['evidence'].values())) == stats['source_rows']
for name, expected in provenance['files_sha256'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
print('PASS: 450 questions, 3995 source rows, dev150/test300, evidence links, statistics and hashes.')
