"""Portable command integration using deterministic test doubles, not LLM scores."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from task_kg_rag import cli

class FakeTokenizer:
    def encode(self,text,**kwargs):return list(text)

class FakeEncoder:
    def __init__(self,*args):
        self.model=argparse.Namespace(tokenizer=FakeTokenizer())
    def __call__(self,text,**kwargs):return np.array([1.,0.],dtype=np.float32)

class FakeReranker:
    def __init__(self,*args):self.tokenizer=FakeTokenizer()
    def __call__(self,question,passages,question_prefix=None):
        return {'scores':[float(i) for i in range(len(passages))]}

class CommandIntegration(unittest.TestCase):
    def test_build_retrieve_and_reject_changed_data(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);assets=root/'assets';assets.mkdir()
            ids=[f'e{i:02}' for i in range(45)]
            corpus={'ids':ids,'evidence':{i:{'id':i,'name':'원문 '+i,'source_file':'source'} for i in ids}}
            corpus_path=root/'corpus.json';corpus_path.write_text(json.dumps(corpus))
            frames=[{'id':'f','name':'과업'}]
            paths=[{'id':'p','frame':'f','evidence_ids':ids,'text':'과업 경로'}]
            inputs={'graph':{'frames':frames,'facts':[],'mentions':[],'paths':paths},'paths':paths,
                    'snapshot':{'nodes':[{'p':frames[0]}],'edges':[]},
                    'pcf':{'pcf_child_of':[],'pcf_mappings':[]},
                    'global_nodes':[{'id':'f','text':'과업'}]}
            for name,data in inputs.items():(assets/(name+'.json')).write_text(json.dumps(data))
            manifest={'files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in assets.iterdir()}}
            (assets/'manifest.json').write_text(json.dumps(manifest))
            args=argparse.Namespace(assets=assets,corpus=corpus_path,output=root/'index',encoder='/test/checkpoint',device='cpu')
            with patch.object(cli,'Encoder',FakeEncoder),patch.object(cli,'Reranker',FakeReranker),contextlib.redirect_stdout(io.StringIO()):
                cli.build_index(args)
                args.index=args.output;args.question='과업의 목적은?';args.aspects=None;args.original_question=True
                args.reader_url=None;args.reranker='/test/reranker'
                for method in ['th-kgr','task-context']:
                    args.method=method;args.output=root/(method+'.json');cli.retrieve(args)
                    result=json.loads(args.output.read_text())
                    self.assertEqual(len(set(result['candidate_evidence_ids'])),40)
                    self.assertEqual(len(result['selected_evidence_ids']),5)
                    self.assertFalse(result['historical_main_table_reproduction'])
                    self.assertTrue(set(result['selected_evidence_ids'])<=set(result['candidate_evidence_ids']))
                    with self.assertRaisesRegex(ValueError,'Output exists'):cli.retrieve(args)
                args.output=root/'changed.json';corpus_path.write_text('{}')
                with self.assertRaisesRegex(ValueError,'identity mismatch'):cli.retrieve(args)

if __name__=='__main__':unittest.main()
