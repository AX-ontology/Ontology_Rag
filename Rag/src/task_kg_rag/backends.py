"""Explicit local model and OpenAI-compatible serving adapters.

No credentials, server launch, GPU selection, or model download is implicit.
"""
import json
import os
import urllib.request
import numpy as np


class Encoder:
    def __init__(self, path, device='cpu', max_length=1536):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(path, device=device, local_files_only=True)
        self.model.max_seq_length = max_length
        if device.startswith('cuda'):
            self.model.half()
        self.max_length = max_length

    def __call__(self, text, allow_truncation=False):
        length = len(self.model.tokenizer.encode(text, truncation=False))
        if length > self.max_length and not allow_truncation:
            raise ValueError(f'Embedding input exceeds {self.max_length} tokens; query truncation forbidden')
        vector = self.model.encode([text], batch_size=1, normalize_embeddings=True,
                                   show_progress_bar=False).astype(np.float32)[0]
        norm = np.linalg.norm(vector)
        if not np.isfinite(vector).all() or norm == 0:
            raise ValueError('Invalid embedding')
        return vector / norm


class Reranker:
    def __init__(self, path, device='cpu'):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.torch, self.device = torch, device
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        if not self.tokenizer.is_fast:
            raise ValueError('Fast tokenizer required for truncation audit')
        dtype = torch.float16 if device.startswith('cuda') else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(
            path, local_files_only=True, torch_dtype=dtype).to(device).eval()

    def __call__(self, question, passages, question_prefix=None):
        question_used = question
        if question_prefix is not None:
            offsets = self.tokenizer(question, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
            if len(offsets) > question_prefix:
                question_used = question[:offsets[question_prefix-1][1]]
        scores, audit = [], []
        for start in range(0, len(passages), 32):
            texts = passages[start:start+32]
            enc = self.tokenizer([question_used]*len(texts), texts, padding=True,
                                 truncation=True, max_length=512, return_tensors='pt', return_offsets_mapping=True)
            offsets = enc.pop('offset_mapping').tolist()
            for i, spans in enumerate(offsets):
                retained = [end for seq, (begin,end) in zip(enc.sequence_ids(i),spans) if seq == 1 and end > begin]
                audit.append(dict(passage_chars=len(texts[i]), passage_last_retained_char=max(retained, default=0)))
            with self.torch.inference_mode():
                scores.extend(self.model(**enc.to(self.device)).logits.view(-1).float().cpu().tolist())
        if len(scores) != len(passages) or not np.isfinite(scores).all():
            raise ValueError('Invalid cross-encoder scores')
        return dict(scores=scores, question_used=question_used, truncation=audit)


def chat(base_url, model, messages, schema=None, max_tokens=1536, seed=42):
    payload = dict(model=model, messages=messages, temperature=0, top_p=1,
                   seed=seed, max_tokens=max_tokens)
    if schema:
        payload['response_format'] = {'type':'json_schema','json_schema':{
            'name':'extraction','schema':schema,'strict':True}}
    headers = {'Content-Type':'application/json'}
    if os.getenv('RAG_API_KEY'):
        headers['Authorization'] = 'Bearer '+os.environ['RAG_API_KEY']
    request = urllib.request.Request(base_url.rstrip('/')+'/chat/completions',
        data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(request, timeout=600) as response:
        value = json.load(response)
    choice = value['choices'][0]
    return dict(content=choice['message']['content'], finish_reason=choice['finish_reason'],
                usage=value.get('usage',{}))


def pack_evidence(ids, evidence, tokenizer, budget=3500):
    selected, blocks = [], []
    for eid in ids:
        e = evidence[eid]
        block = f"[{eid}]\nsource: {e.get('source_file','')}\nsection: {e.get('section_path','')}\n{e['name']}"
        if len(tokenizer.encode('\n\n'.join(blocks+[block]), add_special_tokens=False)) <= budget:
            selected.append(eid)
            blocks.append(block)
    return selected, '\n\n'.join(blocks)
