"""Uncached online KG-context retrieval; no QA references or answer outputs."""
from collections import defaultdict
import time


def select_context_bundles(qid,question,aspects,candidates,*,evidence,ids,paths,path_vectors,raw_vectors,query_vector,aspect_vectors,builder):
    """The original per-question path selection and ablation gates, unchanged."""
    import numpy as np
    if len(candidates)!=40 or len(set(candidates))!=40:raise ValueError('Require shared distinct Dense40')
    if not aspects or len(aspect_vectors)!=len(aspects):raise ValueError('Aspect/vector population mismatch')
    ix={e:i for i,e in enumerate(ids)};support=defaultdict(list)
    for j,path in enumerate(paths):
        for eid in path['evidence_ids']:support[eid].append(j)
    a=np.stack(aspect_vectors).astype(np.float32,copy=True)
    a/=np.linalg.norm(a,axis=1,keepdims=True)+1e-8
    query=np.asarray(query_vector,dtype=np.float32).copy()
    query/=np.linalg.norm(query)+1e-8
    mean=a[0].copy() if len(a)==1 else a.mean(0)
    if len(a)>1:mean/=np.linalg.norm(mean)+1e-8
    # Use identical matrix-vector operations for Aspect and question controls.
    # Batched GEMM versus GEMV roundoff must not manufacture an Aspect effect.
    ps=np.stack([path_vectors@v for v in a]);ms=path_vectors@mean;qs=path_vectors@query
    rawsim=raw_vectors@mean;qrawsim=raw_vectors@query;rows=[]
    for eid in candidates:
        si=support[eid];selected=None;selected0=None
        if si:
            winners={si[int(np.argmax(s[si]))] for s in ps if float(np.max(s[si]))>0}
            if winners:selected=min(winners,key=lambda j:(-float(ms[j]),j))
            if float(np.max(qs[si]))>0:selected0=si[int(np.argmax(qs[si]))]
        bundle=builder.bundle(paths[selected],eid,lambda e:float(rawsim[ix[e]])) if selected is not None else None
        no_aspect=builder.bundle(paths[selected0],eid,lambda e:float(qrawsim[ix[e]])) if selected0 is not None else None
        contexts,trace=builder.contexts(qid,eid,bundle,no_aspect)
        rows.append(dict(evidence_id=eid,raw=evidence[eid]['name'],contexts=contexts,trace=trace))
    return dict(qid=qid,question=question,aspects=aspects,candidates=rows)


class OnlineContextRetrieval:
    METHODS={'P01','B11',*[f'A{i:02}' for i in range(1,8)]}

    def __init__(self,*,corpus,paths,path_vectors,raw_vectors,builder,extract_aspects,embed,rerank,pack):
        self.corpus=corpus;self.paths=paths;self.path_vectors=path_vectors;self.raw_vectors=raw_vectors
        self.builder=builder;self.extract_aspects=extract_aspects;self.embed=embed;self.rerank=rerank;self.pack=pack

    async def retrieve(self,method,qid,question):
        import numpy as np
        if method not in self.METHODS:raise ValueError('Online context method not implemented')
        started=time.perf_counter();t=started
        extraction=await self.extract_aspects(question)
        stages={'aspect_extraction':time.perf_counter()-t};aspects=extraction['aspects']
        if not isinstance(aspects,list) or not 1<=len(aspects)<=4 or any(not isinstance(a,str) or not a.strip() for a in aspects):
            raise ValueError('Require validated source-grounded Aspect extraction or explicit question fallback')
        def timed(name,fn):
            t=time.perf_counter();value=fn();stages[name]=time.perf_counter()-t;return value
        query=timed('query_embedding',lambda:self.embed(question))
        av=timed('aspect_embedding',lambda:[self.embed(a) for a in aspects])
        ids=self.corpus['ids']
        def search():
            scores=self.raw_vectors@query
            return [ids[i] for i in sorted(range(len(ids)),key=lambda i:(-float(scores[i]),ids[i]))[:40]]
        candidates=timed('dense_index_search',search)
        bundle=timed('kg_lookup_and_context',lambda:select_context_bundles(qid,question,aspects,candidates,evidence=self.corpus['evidence'],ids=ids,
            paths=self.paths,path_vectors=self.path_vectors,raw_vectors=self.raw_vectors,query_vector=query,aspect_vectors=av,builder=self.builder))
        passages=[c['raw']+('\n[과업 문맥]\n'+c['contexts'][method] if c['contexts'][method] else '') for c in bundle['candidates']]
        ce=timed('cross_encoder',lambda:self.rerank(question,passages));scores=ce['scores']
        if len(scores)!=40 or not np.isfinite(scores).all():raise ValueError('CE score population mismatch')
        selected=[candidates[i] for i in sorted(range(40),key=lambda i:(-scores[i],candidates[i]))[:5]]
        delivered,context=timed('delivery_context',lambda:self.pack(method,selected))
        if len(set(delivered))!=len(delivered) or set(delivered)-set(selected):raise ValueError('Delivery changed evidence population')
        return dict(method=method,qid=qid,candidate_evidence_ids=candidates,selected_evidence_ids=selected,delivered_evidence_ids=delivered,
            context=context,ce=ce,aspect_extraction=extraction,stage_seconds=stages,retrieval_wall_s=time.perf_counter()-started,
            augmented_n=sum(bool(c['contexts'][method]) for c in bundle['candidates']),includes_answer_generation=False,main_latency_complete=False)
