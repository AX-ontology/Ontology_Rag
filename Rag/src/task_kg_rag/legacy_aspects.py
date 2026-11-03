"""Original source-span Aspect validation shared by preparation and online paths."""
def parse_aspects(question,extracted):
    if not isinstance(extracted,list) or not 1<=len(extracted)<=4:raise ValueError('Require one through4 extracted requirements')
    valid=[];used=[];rejected=[]
    for x in extracted:
        span=x['requirement_span'].strip();start=question.find(span) if span else -1
        if start<0 or any(start<end and start+len(span)>begin for begin,end in used) or not x['query'].strip():
            rejected.append(x);continue
        valid.append(x['query'].strip());used.append((start,start+len(span)))
    unique=list(dict.fromkeys(valid))
    return dict(aspects=unique if len(unique)>1 else [question],rejected_spans=rejected,fallback_to_question=len(unique)<=1,raw_extraction=extracted)
