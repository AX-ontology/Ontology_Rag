"""At most one question-only format repair; every failed attempt stays in cost."""
import time
import json
from .requirements import extraction_messages,parse_requirements,output_schema,RequirementError,fallback
from .spans import nfc,occurrences


def question_span_feedback(question,content):
    """Explain invalid spans using only the question; never repair model output."""
    try:payload=json.loads(content)
    except (ValueError,TypeError):return ''
    if not isinstance(payload,dict):return ''
    spans=[]
    entities=payload.get('entity_spans',[])
    if isinstance(entities,list):
        spans.extend((f'entity_spans[{i}]',e) for i,e in enumerate(entities))
    requirements=payload.get('requirements',[])
    if isinstance(requirements,list):
        for i,r in enumerate(requirements):
            if not isinstance(r,dict):continue
            spans.append((f'requirements[{i}]',dict(text=r.get('requirement_span'),occurrence=r.get('occurrence'))))
            constraints=r.get('explicit_constraint_spans',[])
            if isinstance(constraints,list):
                spans.extend((f'requirements[{i}].explicit_constraint_spans[{j}]',s) for j,s in enumerate(constraints))
    invalid=[]
    for field,span in spans:
        if not isinstance(span,dict) or not isinstance(span.get('text'),str) or not span['text'].strip():continue
        count=len(occurrences(nfc(question),span['text']));value=span.get('occurrence')
        if type(value) is not int or not 0<=value<count:
            invalid.append(dict(field=field,text=span['text'],received_occurrence=value,
                exact_match_count=count,allowed_occurrences=list(range(count))))
    if not invalid:return ''
    return ('\n질문 원문만으로 계산한 span 검사: '+json.dumps(invalid,ensure_ascii=False)+
        '\noccurrence는 각 문자열 자체의 반복 등장 순서다. 배열 항목 번호나 문자 위치가 아니다. '
        '서로 다른 문자열이 각각 한 번 나오면 각각 occurrence=0이다. '
        'exact_match_count=0이면 질문의 연속 문자열이 아니며 코드가 수정하지 않는다. '
        '모델이 같은 질문만 보고 지정 형식으로 한 번 다시 출력하라.')


class RequirementExtractor:
    def __init__(self,system_prompt,call,max_count=3,repair_span_feedback=False):
        self.system_prompt=system_prompt;self.call=call;self.max_count=max_count
        self.repair_span_feedback=repair_span_feedback

    async def extract(self,question):
        initial=extraction_messages(self.system_prompt,question);messages=initial
        attempts=[];last=None
        for number in range(2):
            started=time.perf_counter();record=dict(attempt=number+1,format_repair=bool(number),messages=messages)
            try:
                response=await self.call(messages,output_schema());record['response']=response
            except Exception as exc:
                record.update(error_type='transport_or_server_failure',error=str(exc),elapsed_s=time.perf_counter()-started,usage_available=False)
                attempts.append(record)
                return dict(output=fallback(question,'transport_or_server_failure'),attempts=attempts,format_repair_used=bool(number))
            content=response.get('content')
            try:
                if response.get('validation_error'):raise RequirementError(response['validation_error'])
                if response.get('finish_reason')!='stop':raise RequirementError('Incomplete generation')
                value=parse_requirements(question,content,self.max_count)
                record['elapsed_s']=time.perf_counter()-started;attempts.append(record)
                return dict(output=value,attempts=attempts,format_repair_used=bool(number))
            except (RequirementError,TypeError,ValueError,KeyError) as exc:
                last=str(exc);record.update(error_type='format_or_span_validation',error=last,elapsed_s=time.perf_counter()-started)
                attempts.append(record)
                if number==0:
                    detail=question_span_feedback(question,content) if self.repair_span_feedback else ''
                    messages=[*initial,dict(role='assistant',content=content if isinstance(content,str) else ''),
                        dict(role='user',content='직전 출력이 JSON/질문 span 검사에서 실패했습니다: '+last+
                             '\n같은 질문만 사용해 지정 형식을 한 번 수정하세요. 질문에 없는 span이나 답을 만들지 마세요. '
                             '요구가 4개 이상이면 overflow와 빈 requirements로 반환하세요.'+detail)]
        return dict(output=fallback(question,'format_repair_exhausted: '+str(last)),attempts=attempts,format_repair_used=True)
