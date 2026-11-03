import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path
import json
import numpy as np
from task_kg_rag.hierarchy import merge_candidates, HierarchicalRetriever
from task_kg_rag.requirements import validate_requirements, RequirementError
from task_kg_rag.context import ContextBuilder
from task_kg_rag.legacy_aspects import parse_aspects
from task_kg_rag.backends import pack_evidence


class RetrievalTests(unittest.TestCase):
    def test_round_robin_and_fallback(self):
        original=[f'r{i:02}' for i in range(40)]
        actual=merge_candidates(original,[[original[0],'new-a'],['new-b','new-a']])
        self.assertEqual(actual[:20],original[:20])
        self.assertEqual(actual[20:22],['new-a','new-b'])
        self.assertEqual(len(set(actual)),40)
        self.assertEqual(actual[22:],original[20:38])

    def test_duplicate_dense_rejected(self):
        with self.assertRaises(ValueError):merge_candidates(['x']*40,[])

    def test_typed_hierarchy_and_unbacked_scope(self):
        ids=[f'r{i}' for i in range(40)]
        frames=[{'id':f,'name':f} for f in ['a','b','c','d','unbacked']]
        paths=[{'frame':f,'evidence_ids':ids[i*10:(i+1)*10]} for i,f in enumerate(['a','b','c','d'])]
        raw=np.tile(np.array([1.,0.],dtype=np.float32),(40,1))
        snapshot={'edges':[{'type':'SUBTASK_OF','source':'a','target':'c'},
                           {'type':'RELATED_TASK','source':'a','target':'d'},
                           {'type':'SUBTASK_OF','source':'a','target':'unbacked'}]}
        # Unbacked node has the highest score but must never become a seed.
        nodes=np.array([[1,0],[.9,.1],[.8,.2],[.7,.3],[2,0]],dtype=np.float32)
        r=HierarchicalRetriever(ids=ids,raw_vectors=raw,frames=frames,paths=paths,snapshot=snapshot,
             pcf={'pcf_child_of':[],'pcf_mappings':[]},node_ids=[f['id'] for f in frames],node_vectors=nodes)
        ranked,trace=r.route(np.array([1.,0.],dtype=np.float32))
        self.assertEqual(trace['base_frames'],['a','b'])
        self.assertEqual(trace['selected_frames'],['a','b','c'])
        self.assertNotIn('r39',ranked)
        self.assertIn('unbacked',trace['excluded_frames_without_aligned_paths'])

    def test_original_question_fallback(self):
        q='대상은 누구인가요?'
        result=parse_aspects(q,[{'requirement_span':'누구','query':'대상은 누구인가요?'}])
        self.assertEqual(result['aspects'],[q])
        self.assertTrue(result['fallback_to_question'])

    def test_invalid_requirement_span_rejected(self):
        payload={'status':'ok','reason':'','entity_spans':[{'id':'e1','text':'존재하지않음','occurrence':0}],'requirements':[]}
        with self.assertRaises(RequirementError):validate_requirements('대상은 누구인가요?',payload)

    def test_context_gate_and_original_source(self):
        graph={'facts':[],'mentions':[],'frames':[{'id':'a','name':'과업'}],'paths':[]}
        evidence={'e1':{'name':'짧은 원문','source_file':'source'},'e2':{'name':'가'*101,'source_file':'source'}}
        builder=ContextBuilder(graph,{'nodes':[]},evidence,len)
        bundle={'frame':'과업','mention':'','title':'절','direct_facts':[],'sibling_fact':None,'mention_id':None}
        contexts,trace=builder.contexts('q','e1',bundle,bundle)
        self.assertTrue(trace['gate']);self.assertIn('과업:',contexts['P01'])
        contexts,trace=builder.contexts('q','e2',bundle,bundle)
        self.assertFalse(trace['gate']);self.assertEqual(contexts['P01'],'')
        self.assertEqual(evidence['e1']['name'],'짧은 원문')

    def test_packing_never_injects_kg_context(self):
        class Tokenizer:
            def encode(self,text,**kwargs):return list(text)
        evidence={'e1':{'name':'원문','source_file':'doc'},'e2':{'name':'나'*100,'source_file':'doc'}}
        delivered,text=pack_evidence(['e1','e2'],evidence,Tokenizer(),50)
        self.assertEqual(delivered,['e1']);self.assertIn('원문',text);self.assertNotIn('나'*100,text)

if __name__=='__main__':unittest.main()
