#!/usr/bin/env python3
"""Direct mention-to-catalog assignment; no provisional frames or catalog review."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import shutil

from consolidate_taskframes import (
    SYSTEM, ChatJSONClient, LLMConfig, align_batch, load_env, read_json,
    read_jsonl, stable_id, validate_assignments, write_json, write_jsonl,
)
from taskkg.induction import build_ambiguous_fact_contexts, propagate_fact_links


STAGE = 'catalog_direct_mentions_and_fact_links_v1'


def candidate_ids(fact):
    return sorted(set(fact.get('context_task_mention_ids', [])) |
                  set(fact.get('same_evidence_task_mentions', [])) |
                  {link['task_mention_id'] for link in fact['current_task_links']})


def mention_card(mention):
    return {'source_id': mention['task_mention_id'], 'name': mention['surface_form'],
            'definition': mention.get('intended_outcome_raw', ''),
            'action': mention.get('action_normalized', mention.get('action_raw', '')),
            'object': mention.get('focal_object_normalized', mention.get('focal_object_raw', '')),
            'section_path': mention.get('section_path', ''),
            'source_text': mention.get('source_text', ''), 'evidence_ids': mention.get('evidence_ids', [])}


def direct_batch(client, key, mentions, ambiguous, all_mentions, catalog, contexts):
    """Assign mentions AND judge fact links in one request; repair only invalid rows."""
    source_refs = {f'S{i+1:03d}': m['task_mention_id'] for i, m in enumerate(mentions)}
    target_refs = {f'C{i+1:03d}': tid for i, tid in enumerate(catalog)}
    eligible = sorted({mid for fact in ambiguous for mid in candidate_ids(fact)})
    mention_refs = {f'M{i+1:03d}': mid for i, mid in enumerate(eligible)}
    reverse_mentions = {mid: ref for ref, mid in mention_refs.items()}
    fact_refs = {f'F{i+1:03d}': f for i, f in enumerate(ambiguous)}
    sources = [{**mention_card(m), 'source_id': ref} for ref, m in zip(source_refs, mentions)]
    cards = [{k: v for k, v in {**f, 'triple_id': ref,
        'allowed_mentions': [reverse_mentions[mid] for mid in candidate_ids(f)],
        'current_task_links': [{**link, 'task_mention_id': reverse_mentions[link['task_mention_id']]}
                               for link in f['current_task_links']]}.items()
              if k not in {'context_task_mention_ids', 'same_evidence_task_mentions'}}
             for ref, f in fact_refs.items()]
    accepted, decisions, errors = {}, {}, []
    system = SYSTEM + '''
입력은 임시 TaskFrame이 아니라 원문에서 추출한 업무 언급이다.
S 번호를 기존 C 업무/NOVEL/REVIEW 중 하나에 배정한다. 이 단계에서는 신규 업무를 생성하지 않는다.
동시에 ambiguous_facts의 연결을 원문·인접 문맥으로 재판정한다. F 번호마다 정확히 한 번 응답한다.
Fact 연결은 해당 F의 allowed_mentions에 있는 M 번호만 사용한다. S/C 번호와 혼동하지 않는다.
관계 중요도는 core/supporting/context. 관련 업무가 근거로 확인되면 LINK, 불확실하거나
근거 있는 연결이 없으면 REVIEW와 빈 task_links를 반환한다. 연결을 억지로 만들지 않는다.
목록 배정과 Fact 연결 외에 업무 생성·병합·원문 수정은 하지 않는다.'''
    for attempt in range(3):
        pending_sources = [s for s in sources if s['source_id'] not in accepted]
        pending_facts = [f for f in cards if f['triple_id'] not in decisions]
        needed = {cid for f in pending_facts for cid in f.get('context_chunk_ids', [])}
        prompt = json.dumps({'catalog': [{'id': ref, 'name': catalog[tid]['canonical_name'],
                'definition': catalog[tid]['definition'], 'scope': catalog[tid].get('scope', '')}
                for ref, tid in target_refs.items()],
            'source_mentions': pending_sources, 'ambiguous_facts': pending_facts,
            'link_mentions': [{**mention_card(all_mentions[mid]), 'source_id': ref}
                              for ref, mid in mention_refs.items()],
            'surrounding_contexts': [{k: v for k, v in c.items() if k != 'task_mention_ids'}
                                    for c in contexts if c['chunk_id'] in needed],
            'previous_validation_errors': errors,
            'output_schema': {'assignments': [{'source_id': 'S001', 'target_id': 'C001|NOVEL|REVIEW',
                'confidence': .9, 'reason': '업무 목적에 근거한 이유'}],
                'fact_decisions': [{'triple_id': 'F001', 'status': 'LINK|REVIEW',
                    'task_links': [{'task_mention_id': 'M001', 'relevance': 'core|supporting|context'}],
                    'reason': '원문 근거에 기반한 연결 이유'}]}}, ensure_ascii=False)
        raw = client.call(STAGE, f'{key}-{attempt}', system, prompt)
        errors = []
        try:
            if not isinstance(raw, dict) or raw.get('new_tasks'):
                raise ValueError('Direct matching cannot create tasks')
            for field, pending, id_key in [('assignments', pending_sources, 'source_id'),
                                           ('fact_decisions', pending_facts, 'triple_id')]:
                rows = raw.get(field)
                allowed_ids = {x[id_key] for x in pending}
                if not isinstance(rows, list) or any(not isinstance(r, dict) or
                    not isinstance(r.get(id_key), str) or r[id_key] not in allowed_ids for r in rows):
                    raise ValueError(f'{field} must contain only pending IDs')
            for source in pending_sources:
                ref = source['source_id']
                rows = [r for r in raw['assignments'] if r['source_id'] == ref]
                try:
                    validate_assignments({'assignments': rows}, [ref], set(target_refs) | {'NOVEL', 'REVIEW'})
                except (ValueError, TypeError, KeyError) as exc:
                    errors.append({'source_id': ref, 'error': str(exc)})
                else:
                    accepted[ref] = rows[0]
            for fact in pending_facts:
                ref = fact['triple_id']
                rows = [r for r in raw['fact_decisions'] if r['triple_id'] == ref]
                try:
                    if len(rows) != 1: raise ValueError('Return this fact exactly once')
                    row = rows[0]; links = row.get('task_links')
                    if row.get('status') not in {'LINK', 'REVIEW'} or not isinstance(row.get('reason'), str) or not row['reason'].strip():
                        raise ValueError('Fact requires status and rationale')
                    if not isinstance(links, list) or any(not isinstance(l, dict) or
                        l.get('task_mention_id') not in fact['allowed_mentions'] or
                        l.get('relevance') not in {'core', 'supporting', 'context'} for l in links):
                        raise ValueError('Fact link outside eligible mention IDs or invalid relevance')
                    if len({l['task_mention_id'] for l in links}) != len(links):
                        raise ValueError('Duplicate fact mention link')
                    if bool(links) != (row['status'] == 'LINK'):
                        raise ValueError('LINK requires links; REVIEW requires an empty list')
                except (ValueError, TypeError, KeyError) as exc:
                    errors.append({'triple_id': ref, 'error': str(exc)})
                else:
                    decisions[ref] = row
        except (ValueError, TypeError, KeyError) as exc:
            errors = [{'error': str(exc)}]
        if len(accepted) == len(sources) and len(decisions) == len(cards):
            return ([{**accepted[ref], 'source_id': mid,
                      'target_id': target_refs.get(accepted[ref]['target_id'], accepted[ref]['target_id'])}
                     for ref, mid in source_refs.items()],
                    [{**decisions[ref], 'triple_id': f['triple_id'],
                      'task_links': [{**link, 'task_mention_id': mention_refs[link['task_mention_id']]}
                                     for link in decisions[ref]['task_links']]}
                     for ref, f in fact_refs.items()])
    raise ValueError(f'Direct assignment unresolved after repairs: {errors}')


def group_novel(client, rows, mentions, catalog, batch_size):
    """Only NOVEL rows enter grouping; each later batch sees prior new tasks."""
    rows = deepcopy(rows)
    choices = {tid: {'id': tid, 'name': f['canonical_name'], 'definition': f['definition'],
                     'baseline': True} for tid, f in catalog.items()}
    new_frames = {}
    novel = [r for r in rows if r['target_id'] == 'NOVEL']
    replacements = {}
    for offset in range(0, len(novel), batch_size):
        cards = [mention_card(mentions[r['source_id']]) for r in novel[offset:offset+batch_size]]
        assignments, new = align_batch(client, offset, cards, choices,
                                       json.dumps({'direct_novel_mentions': cards}, ensure_ascii=False))
        ids = {}
        for task in new:
            # Reuse an exactly identical newly introduced definition, never rename an approved ID.
            identical = next((tid for tid, c in choices.items()
                              if (c['name'], c['definition']) == (task['name'], task['definition'])), None)
            tid = identical or stable_id('TF', 'direct_catalog_v1', task['name'], task['definition'])
            if tid in choices and not identical: raise ValueError('New task ID collision')
            ids[task['key']] = tid
            if tid not in choices:
                choices[tid] = {'id': tid, 'name': task['name'], 'definition': task['definition'], 'baseline': False}
                new_frames[tid] = {'taskframe_id': tid, 'canonical_name': task['name'],
                    'definition': task['definition'], 'scope': '', 'aliases': [],
                    'status': 'provisional', 'schema_version': '4.5.1'}
        for row in assignments:
            replacements[row['source_id']] = {**row, 'target_id': ids.get(row['target_id'], row['target_id'])}
    rows = [replacements.get(r['source_id'], r) for r in rows]
    return rows, new_frames


def assign(source, baseline, out, client, batch_size=48, context_radius=1, max_facts=160):
    if batch_size < 1: raise ValueError('batch_size must be positive')
    if max_facts < 1: raise ValueError('max_facts must be positive')
    if source.resolve() == out.resolve() or source.resolve() in out.resolve().parents:
        raise ValueError('Output must be separate from source')
    if (out / 'run_manifest.json').exists(): raise ValueError('Output already finalized; use stage reuse')
    mentions = read_jsonl(source / '03_entity_prep/task_mentions.jsonl')
    triples = read_jsonl(source / '03_entity_prep/open_triples.jsonl')
    chunks = read_jsonl(source / '01_evidence/chunks.jsonl')
    baseline_frames = read_json(baseline)
    catalog = {f['taskframe_id']: f for f in baseline_frames}
    lookup = {m['task_mention_id']: m for m in mentions}
    if len(lookup) != len(mentions) or len(catalog) != len(baseline_frames): raise ValueError('Duplicate IDs')
    if len({f['triple_id'] for f in triples}) != len(triples): raise ValueError('Duplicate fact IDs')
    if any(l['task_mention_id'] not in lookup for f in triples for l in f.get('task_links', [])):
        raise ValueError('Unknown extraction mention link')
    ambiguous, contexts = build_ambiguous_fact_contexts(triples, mentions, chunks, context_radius)
    batches = [mentions[i:i+batch_size] for i in range(0, len(mentions), batch_size)]
    batch_for = {m['task_mention_id']: i for i, b in enumerate(batches) for m in b}
    facts_by_batch, review = defaultdict(list), []
    for f in ambiguous:
        candidates = candidate_ids(f)
        if candidates:
            # Keep all cross-batch candidate mentions in the request, not just the assigned batch.
            facts_by_batch[min(batch_for[mid] for mid in candidates)].append(f)
        else:
            review.append({'kind': 'fact', 'triple_id': f['triple_id'], 'reason': 'No context task mention candidates'})
    if review:
        write_jsonl(out / 'review_required.jsonl', review)
        raise ValueError('Fact context missing; review required before API calls')
    rows, decisions, assignment_batches = [], [], 0
    for index, batch in enumerate(batches):
        # Large evidence payloads stay in the same judgment stage, but are split
        # to bound response size. A mention is assigned only on the first part.
        fact_batch = facts_by_batch[index]
        for offset in range(0, max(1, len(fact_batch)), max_facts):
            matches, links = direct_batch(client, f'{index}-{offset}', batch if offset == 0 else [],
                                          fact_batch[offset:offset+max_facts], lookup, catalog, contexts)
            rows.extend(matches); decisions.extend(links); assignment_batches += 1
        print(f'[direct-match] mentions={len(rows)}/{len(mentions)} facts={len(decisions)}/{len(ambiguous)}', flush=True)
    write_jsonl(out / 'existing_match_decisions.jsonl', rows)
    write_jsonl(out / 'fact_link_decisions.jsonl', decisions)
    review = [{'kind': 'mention', **r} for r in rows if r['target_id'] == 'REVIEW'] + [
              {'kind': 'fact', **d} for d in decisions if d['status'] == 'REVIEW']
    write_jsonl(out / 'review_required.jsonl', review)
    if review: raise ValueError(f'{len(review)} decisions require review; no new tasks created')
    novel_count = sum(r['target_id'] == 'NOVEL' for r in rows)
    rows, additions = group_novel(client, rows, lookup, catalog, batch_size)
    catalog = {**catalog, **additions}
    if Counter(r['source_id'] for r in rows) != Counter(lookup.keys()): raise ValueError('Mention partition mismatch')
    members = defaultdict(list)
    for row in rows:
        if row['target_id'] not in catalog: raise ValueError('Unknown final task ID')
        members[row['target_id']].append(row['source_id'])
    frames = []
    for tid, mids in sorted(members.items()):
        frame = deepcopy(catalog[tid])
        frame.update(member_task_mentions=sorted(mids),
                     evidence_ids=sorted({e for mid in mids for e in lookup[mid].get('evidence_ids', [])}),
                     source_document_ids=sorted({lookup[mid]['document_id'] for mid in mids}))
        # Baseline identity/definition is copied, never model-edited.
        frames.append(frame)
    propagated, links, summary = propagate_fact_links(triples, frames, decisions, mentions, exhaustive=False)
    if any(not f['taskframe_links'] for f in propagated): raise ValueError('Unlinked fact requires review')
    if [f['triple_id'] for f in propagated] != [f['triple_id'] for f in triples]: raise ValueError('Facts lost')
    for original, result in zip(triples, propagated):
        if any(result[k] != v for k, v in original.items() if k != 'task_links'):
            raise ValueError('Raw fact content changed')
    for name in ['01_evidence', '02_extraction', '03_entity_prep']:
        shutil.copytree(source / name, out / name, dirs_exist_ok=True)
    write_jsonl(out / '04_clustering/pair_scores.jsonl', [])
    write_jsonl(out / '04_clustering/candidate_blocks.jsonl', [])
    write_json(out / '04_clustering/summary.json', {'method': 'skipped_direct_catalog_assignment'})
    write_json(out / '05_taskframes/taskframes.json', frames)
    write_jsonl(out / '05_taskframes/task_relations.jsonl', [])
    write_jsonl(out / '05_taskframes/triple_link_decisions.jsonl', decisions)
    write_jsonl(out / '05_taskframes/ambiguous_fact_contexts.jsonl', ambiguous)
    write_jsonl(out / '05_taskframes/surrounding_chunk_contexts.jsonl', contexts)
    report = {'status': 'complete', 'method': 'direct_mentions_existing_first',
        'mention_partition_valid': True, 'fact_ids_preserved': True,
        'taskframe_count': len(frames), 'task_mention_count': len(mentions), 'fact_count': len(triples),
        'novel_mention_count': novel_count, 'new_taskframe_count': len(additions),
        'direct_assignment_batches': assignment_batches, 'mention_batches': len(batches),
        'novel_grouping_batches': (novel_count+batch_size-1)//batch_size,
        'local_induction_calls': 0, 'global_reconciliation_calls': 0, 'catalog_review_calls': 0,
        'call_count_note': 'Batch counts exclude retries and cache hits; not actual HTTP request counts.',
        'hierarchy_created': False, 'ambiguous_fact_count': len(ambiguous), 'rejudged_fact_count': len(decisions)}
    write_json(out / '05_taskframes/validation_report.json', report)
    write_jsonl(out / '06_propagation/facts_with_taskframes.jsonl', propagated)
    write_jsonl(out / '06_propagation/fact_taskframe_links.jsonl', links)
    write_json(out / '06_propagation/summary.json', summary)
    write_jsonl(out / 'mention_to_taskframe.jsonl', rows)
    write_jsonl(out / 'review_queue.jsonl', [r for r in rows if r['confidence'] < .7])
    write_json(out / 'run_manifest.json', {**read_json(source / 'run_manifest.json'),
        'status': 'complete', 'completed_stage': 'propagation', 'catalog_assignment': 'direct',
        'llm_stage_count': 2 + bool(novel_count),
        'llm_stages': ['extraction', 'direct_existing_assignment_and_fact_links'] + (['novel_grouping'] if novel_count else []),
        'direct_assignment_report': report})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['source', 'baseline', 'out', 'env']: parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=48)
    parser.add_argument('--context-radius', type=int, default=1)
    parser.add_argument('--max-facts', type=int, default=160)
    args = parser.parse_args()
    load_env(args.env)
    config = LLMConfig.from_env()
    if config.api_base.rstrip('/') != 'https://api.deepseek.com':
        raise ValueError('This workflow is authorized for https://api.deepseek.com only')
    client = ChatJSONClient(config, args.out / '.cache/llm')
    assign(args.source, args.baseline, args.out, client, args.batch_size, args.context_radius, args.max_facts)


if __name__ == '__main__': main()
