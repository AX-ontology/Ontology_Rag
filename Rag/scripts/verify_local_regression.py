#!/usr/bin/env python3
"""Compare portable routing/context preparation against archived local runs.

Usage: python Rag/scripts/verify_local_regression.py --study-root PATH --output PATH
Requires the original local caches (not part of the public repository).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from task_kg_rag.hierarchy import HierarchicalRetriever, merge_candidates
from task_kg_rag.cli import assets
from task_kg_rag.context import ContextBuilder
from task_kg_rag.context_retrieval import select_context_bundles

read=lambda p:json.loads(Path(p).read_text())
lines=lambda p:[json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]

def main():
    p=argparse.ArgumentParser();p.add_argument('--study-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--tokenizer',type=Path)
    args=p.parse_args();s=args.study_root/'runs/scope_context_v1_20261031';a=s/'retrieval_assets_dev_format_v2'
    n=args.study_root/'runs/aspect_hierarchy_global_node_assets_20261101_v1';pilot=args.study_root/'runs/aspect_hierarchy_local_global_pilot_20261101_v3'
    data=assets(ROOT/'assets');ids=read(a/'raw_ids.json')
    retriever=HierarchicalRetriever(ids=ids,raw_vectors=np.load(a/'raw.npy'),frames=data['graph']['frames'],paths=data['paths'],snapshot=data['snapshot'],pcf=data['pcf'],node_ids=read(n/'global_node_ids.json'),node_vectors=np.load(n/'global_node_vectors.npy'))
    vectors=dict(zip(read(a/'query_cache_keys.json'),np.load(a/'query_cache.npy')))
    queries={r['qid']:r for r in read(a/'queries.json')};count=0
    for expected in lines(pilot/'candidate_sets.jsonl'):
        q=queries[expected['qid']];qv=vectors[q['question_key']];av=[vectors[k] for k in q['requirement_keys']]
        original=retriever.rank_raw(qv)[:40];method=expected['method']
        if method=='LQ':actual=original
        elif method=='LA':actual=merge_candidates(original,[retriever.rank_raw(v) for v in av])
        else:
            use=[qv] if method=='GQ' else av
            actual=merge_candidates(original,[retriever.route(v,method=='GHA')[0] for v in use])
        assert actual==expected['candidate_evidence_ids'],(expected['qid'],method)
        count+=1
    result={'hierarchy_questions':12,'hierarchy_candidate_sets_matched':count,'candidate_ids_and_order_exact':True,'model_inference_executed':False,'task_context_cases':0}
    if args.tokenizer:
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
        core=args.study_root/'runs/core_20261031';corpus=read(core/'inputs/corpus.json')
        builder=ContextBuilder(data['graph'],data['snapshot'],corpus['evidence'],lambda t:len(tok.encode(t,add_special_tokens=False)))
        queries=dict(zip(read(core/'single_query_embeddings/questions_ids.json'),np.load(core/'single_query_embeddings/questions.npy')))
        av=dict(zip(read(core/'context_bundles/aspect_texts.json'),np.load(core/'context_bundles/aspect_vectors.npy')))
        pv=np.load(core/'fresh_path_embeddings/paths.npy');raw=np.load(core/'fresh_embeddings_full/raw.npy')
        records=read(core/'aspects/aspects.json');cases=[records[0]]+[q for q in records if len(q['aspects'])>1][:2]
        for q in cases:
            expected=read(core/'context_bundles'/(q['qid']+'.json'))
            candidate_ids=[c['evidence_id'] for c in expected['candidates']]
            actual=select_context_bundles(q['qid'],q['question'],q['aspects'],candidate_ids,evidence=corpus['evidence'],ids=corpus['ids'],paths=data['paths'],path_vectors=pv,raw_vectors=raw,query_vector=queries[q['qid']],aspect_vectors=[av[t] for t in q['aspects']],builder=builder)
            for got,want in zip(actual['candidates'],expected['candidates']):
                assert got['evidence_id']==want['evidence_id']
                assert got['raw']==want['raw']
                assert got['contexts']['P01']==want['contexts']['P01'],(q['qid'],got['evidence_id'])
                assert got['trace']['bundle']==want['trace']['bundle']
            result['task_context_cases']+=1
        result['task_context_candidates_matched']=result['task_context_cases']*40
    result['published_code_sha256']={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (ROOT/'src').rglob('*.py')}
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='published_code_sha256'}))
if __name__=='__main__':main()
