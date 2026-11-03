"""Portable commands: build-index, retrieve; all paths are explicit/configurable."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .hierarchy import HierarchicalRetriever
from .context import ContextBuilder
from .context_retrieval import select_context_bundles
from .backends import Encoder, Reranker, chat, pack_evidence
from .legacy_aspects import parse_aspects
from .extraction import RequirementExtractor

ROOT = Path(__file__).resolve().parents[2]
read = lambda path: json.loads(Path(path).read_text())
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def assets(folder):
    folder = Path(folder)
    manifest = read(folder/'manifest.json')
    for name, expected in manifest['files_sha256'].items():
        if sha(folder/name) != expected:
            raise ValueError('Graph asset hash mismatch: '+name)
    return {name:read(folder/(name+'.json')) for name in ['graph','snapshot','pcf','paths','global_nodes']}


def build_index(args):
    data = assets(args.assets)
    corpus = read(args.corpus)
    if len(corpus['ids']) < 40:
        raise ValueError('At least 40 source rows required')
    args.output.mkdir(parents=True, exist_ok=False)
    encoder = Encoder(args.encoder, args.device)
    sources = {'raw':(corpus['ids'], [corpus['evidence'][e]['name'] for e in corpus['ids']]),
               'paths':([p['id'] for p in data['paths']], [p['text'] for p in data['paths']]),
               'nodes':([n['id'] for n in data['global_nodes']], [n['text'] for n in data['global_nodes']])}
    audit = {}
    for name, (ids,texts) in sources.items():
        lengths = [len(encoder.model.tokenizer.encode(t, truncation=False)) for t in texts]
        vectors = np.stack([encoder(t, allow_truncation=True) for t in texts])
        np.save(args.output/(name+'.npy'), vectors)
        save(args.output/(name+'_ids.json'), ids)
        audit[name] = dict(count=len(ids),max_tokens=max(lengths),truncated=sum(n>1536 for n in lengths))
    save(args.output/'manifest.json', dict(corpus_sha256=sha(args.corpus),
         assets_manifest_sha256=sha(args.assets/'manifest.json'), encoder=str(Path(args.encoder).resolve()),
         device=args.device, max_length=1536, encoding='single_text_float32_renormalized',audit=audit,
         files_sha256={p.name:sha(p) for p in args.output.iterdir() if p.is_file()},
         historical_scores_reproduced=False))
    print('Index written:',args.output)


def extract(args):
    if args.aspects:
        value = read(args.aspects)
        if value['question'] != args.question:
            raise ValueError('Precomputed extraction does not match question')
        queries = value['aspects']
        limit = 4 if args.method == 'task-context' else 3
        if not isinstance(queries,list) or not 1 <= len(queries) <= limit or any(not isinstance(t,str) or not t.strip() for t in queries):
            raise ValueError('Invalid precomputed Aspect queries')
        return queries,dict(mode='precomputed_question_only',record=value)
    if args.original_question:
        return [args.question],dict(mode='explicit_no_aspect_control')
    if args.method == 'task-context':
        contract = read(ROOT/'config/legacy_aspects.json')
        messages = [dict(role='system',content=contract['prompt']),dict(role='user',content=args.question)]
        response = chat(args.extractor_url,args.extractor_model,messages,contract['schema'],600,20261028)
        if response['finish_reason'] != 'stop':
            raise ValueError('Incomplete legacy extraction')
        value = parse_aspects(args.question,json.loads(response['content'])['aspects'])
        return value['aspects'],dict(mode='legacy_live',response=response,validation=value)
    async def call(messages,schema):
        return chat(args.extractor_url,args.extractor_model,messages,schema,1536)
    extractor = RequirementExtractor((ROOT/'config/requirements_system.txt').read_text(),call)
    value = asyncio.run(extractor.extract(args.question))
    return value['output']['queries'],dict(mode='requirements_live',record=value)


def retrieve(args):
    if args.output.exists():
        raise ValueError('Output exists; use a new path')
    data = assets(args.assets)
    corpus = read(args.corpus)
    manifest = read(args.index/'manifest.json')
    if sha(args.corpus) != manifest['corpus_sha256'] or sha(args.assets/'manifest.json') != manifest['assets_manifest_sha256']:
        raise ValueError('Index/data identity mismatch')
    if str(Path(args.encoder).resolve()) != manifest['encoder'] or args.device != manifest['device']:
        raise ValueError('Use the same encoder path and device profile as index construction')
    for name, expected in manifest['files_sha256'].items():
        if sha(args.index/name) != expected:
            raise ValueError('Index hash mismatch: '+name)
    ids = read(args.index/'raw_ids.json')
    if ids != corpus['ids'] or read(args.index/'paths_ids.json') != [p['id'] for p in data['paths']]:
        raise ValueError('Vector row order mismatch')
    raw = np.load(args.index/'raw.npy',allow_pickle=False)
    paths = np.load(args.index/'paths.npy',allow_pickle=False)
    encoder = Encoder(args.encoder,args.device)
    ce = Reranker(args.reranker,args.device) if args.reranker else None
    if args.method == 'task-context' and ce is None:
        raise ValueError('Task-context requires --reranker (also used to count context tokens)')
    reader_tokenizer = None
    if args.reader_url:
        if not ce or not args.reader_tokenizer or not args.reader_model:
            raise ValueError('Reader requires --reranker, --reader-tokenizer and --reader-model')
        from transformers import AutoTokenizer
        reader_tokenizer = AutoTokenizer.from_pretrained(args.reader_tokenizer,local_files_only=True)
    started = time.perf_counter()
    stages = {}
    def timed(name, fn):
        start = time.perf_counter()
        value = fn()
        stages[name] = time.perf_counter()-start
        return value
    aspects, extraction = timed('aspect_extraction',lambda:extract(args))
    query = timed('query_embedding',lambda:encoder(args.question))
    av = timed('aspect_embedding',lambda:[encoder(a) for a in aspects])
    if args.method == 'task-context':
        def search():
            scores = raw @ query
            return [ids[i] for i in sorted(range(len(ids)),key=lambda i:(-float(scores[i]),ids[i]))[:40]]
        candidates = timed('dense_search',search)
        def contexts():
            builder = ContextBuilder(data['graph'],data['snapshot'],corpus['evidence'],
                lambda t:len(ce.tokenizer.encode(t,add_special_tokens=False)))
            return select_context_bundles('interactive',args.question,aspects,candidates,
                evidence=corpus['evidence'],ids=ids,paths=data['paths'],path_vectors=paths,
                raw_vectors=raw,query_vector=query,aspect_vectors=av,builder=builder)
        bundle = timed('kg_context',contexts)
        passages = [c['raw']+('\n[과업 문맥]\n'+c['contexts']['P01'] if c['contexts']['P01'] else '') for c in bundle['candidates']]
        result = dict(candidate_evidence_ids=candidates,context_trace=bundle)
    else:
        def route():
            retriever = HierarchicalRetriever(ids=ids,raw_vectors=raw,frames=data['graph']['frames'],
                paths=data['paths'],snapshot=data['snapshot'],pcf=data['pcf'],
                node_ids=read(args.index/'nodes_ids.json'),node_vectors=np.load(args.index/'nodes.npy',allow_pickle=False))
            return retriever.retrieve(query,av)
        result = timed('hierarchical_retrieval',route)
        candidates = result['candidate_evidence_ids']
        passages = [corpus['evidence'][e]['name'] for e in candidates]
    if ce:
        scored = timed('cross_encoder',lambda:ce(args.question,passages,128 if args.method=='th-kgr' else None))
        selected = [candidates[i] for i in sorted(range(40),key=lambda i:(-scored['scores'][i],candidates[i]))[:5]]
        result.update(selected_evidence_ids=selected,ce=scored,CE_scored=True)
    if args.reader_url:
        delivered,context = timed('evidence_packing',lambda:pack_evidence(selected,corpus['evidence'],reader_tokenizer))
        messages = [dict(role='system',content='제공된 원문 근거만으로 한국어로 답변하고 근거 ID를 인용하세요. 근거가 부족하면 부족하다고 답하세요.'),
                    dict(role='user',content=f'질문: {args.question}\n\n근거:\n{context}')]
        answer = timed('answer_generation',lambda:chat(args.reader_url,args.reader_model,messages))
        if answer['finish_reason'] != 'stop':
            raise ValueError('Incomplete answer generation')
        result.update(answer=answer,delivered_evidence_ids=delivered,QA_executed=True)
    result.update(question=args.question,method=args.method,aspects=aspects,extraction=extraction,
                  stage_seconds=stages,online_wall_seconds=time.perf_counter()-started,
                  model_loading_included=False,verification_executed=False,
                  historical_main_table_reproduction=False)
    save(args.output,result)
    print('Result written:',args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    for name in ['build-index','retrieve']:
        p = sub.add_parser(name)
        p.add_argument('--corpus',type=Path,default=ROOT.parent/'data/jaije_qa_dataset/corpus.json')
        p.add_argument('--assets',type=Path,default=ROOT/'assets')
        p.add_argument('--encoder',required=True,help='Local BGE-M3 checkpoint directory')
        p.add_argument('--device',default='cpu',choices=['cpu','cuda:0'])
        p.add_argument('--output',type=Path,required=True)
        if name == 'retrieve':
            p.add_argument('--index',type=Path,required=True)
            p.add_argument('--method',choices=['task-context','th-kgr'],required=True)
            p.add_argument('--question',required=True)
            source = p.add_mutually_exclusive_group(required=True)
            source.add_argument('--aspects',type=Path)
            source.add_argument('--original-question',action='store_true')
            source.add_argument('--extractor-url')
            p.add_argument('--extractor-model',default='Qwen3-8B')
            p.add_argument('--reranker',help='Local bge-reranker-v2-m3 checkpoint')
            p.add_argument('--reader-url')
            p.add_argument('--reader-model')
            p.add_argument('--reader-tokenizer')
    args = parser.parse_args()
    (build_index if args.command=='build-index' else retrieve)(args)

if __name__ == '__main__':
    main()
