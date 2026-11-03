"""Exact raw source spans with an explicit, conservative NFC boundary map."""
import hashlib
import json
import unicodedata


def nfc(text):
    if not isinstance(text,str):raise ValueError('Text must be a string')
    return unicodedata.normalize('NFC',text)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def occurrences(source,text):
    if not isinstance(text,str) or not text.strip():raise ValueError('Span text must be nonempty')
    start=0;positions=[]
    while True:
        at=source.find(text,start)
        if at<0:return positions
        positions.append(at);start=at+1


def normalization_record(original):
    normalized=nfc(original)
    if normalized==original:
        boundaries=[[i] for i in range(len(original)+1)]
    else:
        boundaries=[[] for _ in range(len(normalized)+1)]
        for i in range(len(original)+1):
            prefix=nfc(original[:i])
            # A composing/reordered cluster need not have a raw contiguous
            # boundary for every normalized character. Do not invent one.
            if normalized.startswith(prefix):boundaries[len(prefix)].append(i)
    return dict(original=original,normalized=normalized,boundary_to_original=boundaries,
                original_sha256=hashlib.sha256(original.encode()).hexdigest(),
                normalized_sha256=hashlib.sha256(normalized.encode()).hexdigest())


def exact_span(source,text,occurrence=0,source_row_id=None):
    if type(occurrence) is not int or occurrence<0:raise ValueError('Occurrence must be a nonnegative integer')
    positions=occurrences(source,text)
    if occurrence>=len(positions):raise ValueError('Span occurrence does not exist')
    start=positions[occurrence];end=start+len(text)
    value=dict(text=text,start=start,end=end,occurrence=occurrence)
    if source_row_id is not None:value['source_row_id']=source_row_id
    if source[start:end]!=text:raise ValueError('Raw source span mismatch')
    return value


class SourceSpans:
    def __init__(self,evidence):
        self.evidence=evidence;self.normalized={}

    def record(self,eid):
        if eid not in self.normalized:self.normalized[eid]=normalization_record(self.evidence[eid]['name'])
        return self.normalized[eid]

    def align(self,text,source_ids):
        """Return every exact source-aligned match, retaining the raw substring.

        NFC matching may locate a canonically equivalent quotation, but the
        stored text and offsets always refer to the unchanged original row.
        Ambiguous normalized boundaries are rejected rather than guessed.
        """
        quote=nfc(text);matches=[]
        if not quote.strip():return matches
        for eid in dict.fromkeys(source_ids):
            if eid not in self.evidence:raise ValueError('Unknown source evidence '+eid)
            record=self.record(eid);raw=record['original'];normalized=record['normalized']
            for at in occurrences(normalized,quote):
                left=record['boundary_to_original'][at];right=record['boundary_to_original'][at+len(quote)]
                if len(left)!=1 or len(right)!=1:continue
                start,end=left[0],right[0];literal=raw[start:end]
                if nfc(literal)!=quote:continue
                span=exact_span(raw,literal,occurrences(raw,literal).index(start),eid)
                span['normalized_text']=quote;matches.append(span)
        return matches

    def verify(self,span):
        raw=self.evidence[span['source_row_id']]['name']
        checked=exact_span(raw,span['text'],span['occurrence'],span['source_row_id'])
        if any(checked[k]!=span[k] for k in ['text','start','end','occurrence','source_row_id']):
            raise ValueError('Source span metadata differs from the original row')
        return True
