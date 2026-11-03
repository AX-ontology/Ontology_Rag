"""Question-only requirements: strict spans, references, no generated search facts."""
import json
from .spans import exact_span,nfc,normalization_record

RELATIONS={'what','who','when','why','how','condition','exception','comparison','other'}
ANSWER_TYPES={'entity','text','time','reason','procedure','condition','comparison','other'}


class RequirementError(ValueError):pass


def keys(value,expected):
    if not isinstance(value,dict) or set(value)!=set(expected):
        raise RequirementError('Unexpected fields: expected '+','.join(expected))


def span(question,value):
    keys(value,['text','occurrence'])
    try:return exact_span(question,value['text'],value['occurrence'])
    except (ValueError,TypeError) as exc:raise RequirementError(str(exc)) from exc


def fallback(question,reason,status='invalid'):
    return dict(status=status,reason=reason,entity_spans=[],requirements=[],queries=[nfc(question)],
                fallback_to_question=True,original_question=question,extraction_question=nfc(question),
                normalization=normalization_record(question))


def validate_requirements(question,payload,max_count=3):
    """Validate structure and derive all search expressions from question spans."""
    if not isinstance(question,str) or not question.strip():raise RequirementError('Empty question')
    q=nfc(question)
    keys(payload,['status','reason','entity_spans','requirements'])
    if not isinstance(payload['status'],str) or payload['status'] not in {'ok','overflow','invalid'} or not isinstance(payload['reason'],str):
        raise RequirementError('Invalid status or reason')
    if not isinstance(payload['entity_spans'],list) or not isinstance(payload['requirements'],list):
        raise RequirementError('Entity spans and requirements must be lists')
    if payload['status']!='ok':
        if payload['requirements']:raise RequirementError('Non-ok output must not retain partial requirements')
        return fallback(question,payload['reason'],payload['status'])
    entities={}
    for entity in payload['entity_spans']:
        keys(entity,['id','text','occurrence']);eid=entity['id']
        if not isinstance(eid,str) or not eid or eid in entities:raise RequirementError('Invalid or duplicate entity ID')
        entities[eid]=dict(id=eid,**span(q,{k:entity[k] for k in ['text','occurrence']}))
    reqs={}
    for raw in payload['requirements']:
        keys(raw,['id','entity_ids','requirement_span','occurrence','requested_relation','explicit_constraint_spans','answer_type','refers_to'])
        rid=raw['id']
        if not isinstance(rid,str) or not rid or rid in reqs:raise RequirementError('Invalid or duplicate requirement ID')
        for key in ['entity_ids','explicit_constraint_spans','refers_to']:
            if not isinstance(raw[key],list):raise RequirementError('Expected list '+key)
        for key in ['entity_ids','refers_to']:
            if any(not isinstance(x,str) for x in raw[key]) or len(raw[key])!=len(set(raw[key])):
                raise RequirementError('Repeated or invalid reference '+key)
        if any(eid not in entities for eid in raw['entity_ids']):raise RequirementError('Unknown entity reference')
        if (not isinstance(raw['requested_relation'],str) or raw['requested_relation'] not in RELATIONS
                or not isinstance(raw['answer_type'],str) or raw['answer_type'] not in ANSWER_TYPES):
            raise RequirementError('Unknown requirement tag')
        value=dict(raw);value['span']=span(q,dict(text=raw['requirement_span'],occurrence=raw['occurrence']))
        value['explicit_constraint_spans']=[span(q,s) for s in raw['explicit_constraint_spans']]
        reqs[rid]=value
    if not reqs:raise RequirementError('ok output must have requirements')
    active=set();visited=set()
    def visit(rid):
        if rid not in reqs:raise RequirementError('Unknown requirement reference')
        if rid in active:raise RequirementError('Cyclic requirement references')
        if rid in visited:return
        active.add(rid)
        for reference in reqs[rid]['refers_to']:visit(reference)
        active.remove(rid);visited.add(rid)
    for rid in reqs:visit(rid)
    # Only exact duplicates are collapsed. Shared spans and differing relation
    # tags (especially when/why) remain distinct requirements.
    signatures={};aliases={};retained=[]
    for rid,r in reqs.items():
        signature=(tuple((entities[e]['text'],entities[e]['start'],entities[e]['end']) for e in r['entity_ids']),
                   r['requirement_span'],r['span']['start'],r['span']['end'],r['requested_relation'],r['answer_type'],
                   tuple((s['text'],s['start'],s['end']) for s in r['explicit_constraint_spans']),tuple(r['refers_to']))
        if signature in signatures:aliases[rid]=signatures[signature]
        else:signatures[signature]=rid;retained.append(r)
    if len(retained)>max_count:return fallback(question,'More than maximum supported requirements','overflow')
    for r in retained:
        r['refers_to']=[aliases.get(ref,ref) for ref in r['refers_to']]
        if r['id'] in r['refers_to']:raise RequirementError('Duplicate resolution creates a self reference')
        pieces=[entities[e]['text'] for e in r['entity_ids']]
        pieces += [reqs[ref]['requirement_span'] for ref in r['refers_to']]
        pieces += [r['requirement_span'],*[s['text'] for s in r['explicit_constraint_spans']]]
        r['query']=' '.join(dict.fromkeys(pieces))
    return dict(status='ok',reason=payload['reason'],entity_spans=list(entities.values()),requirements=retained,
                queries=[r['query'] for r in retained],fallback_to_question=False,
                original_question=question,extraction_question=q,deduplicated_ids=aliases,
                normalization=normalization_record(question))


def parse_requirements(question,raw,max_count=3):
    try:payload=json.loads(raw)
    except (TypeError,ValueError) as exc:raise RequirementError('Invalid JSON') from exc
    return validate_requirements(question,payload,max_count)


def extraction_messages(system_prompt,question):
    # This API accepts no document, reference, intermediate answer or retrieval.
    return [dict(role='system',content=system_prompt),dict(role='user',content='QUESTION\n'+nfc(question))]


def output_schema():
    span_schema=dict(type='object',additionalProperties=False,
        properties={'text':{'type':'string'},'occurrence':{'type':'integer','minimum':0}},required=['text','occurrence'])
    entity=dict(type='object',additionalProperties=False,properties={'id':{'type':'string'},**span_schema['properties']},
                required=['id','text','occurrence'])
    req=dict(type='object',additionalProperties=False,properties={
        'id':{'type':'string'},'entity_ids':{'type':'array','items':{'type':'string'}},
        'requirement_span':{'type':'string'},'occurrence':{'type':'integer','minimum':0},
        'requested_relation':{'type':'string','enum':sorted(RELATIONS)},
        'explicit_constraint_spans':{'type':'array','items':span_schema},
        'answer_type':{'type':'string','enum':sorted(ANSWER_TYPES)},
        'refers_to':{'type':'array','items':{'type':'string'}}},required=[
            'id','entity_ids','requirement_span','occurrence','requested_relation','explicit_constraint_spans','answer_type','refers_to'])
    return dict(type='object',additionalProperties=False,properties={
        'status':{'type':'string','enum':['ok','overflow','invalid']},'reason':{'type':'string'},
        'entity_spans':{'type':'array','items':entity},'requirements':{'type':'array','items':req}},
        required=['status','reason','entity_spans','requirements'])
