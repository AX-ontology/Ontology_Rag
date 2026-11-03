"""Global-to-local candidate construction ported from the dev12 v3 pilot.

No reference answers or gold evidence are accepted by this module.
"""
from collections import defaultdict
import numpy as np


def merge_candidates(original, rankings):
    if len(original) != 40 or len(set(original)) != 40:
        raise ValueError('Expected 40 distinct original Dense candidates')
    selected = list(original[:20])
    cursors = [0] * len(rankings)
    while len(selected) < 40:
        advanced = False
        for i, ranked in enumerate(rankings):
            while cursors[i] < len(ranked) and ranked[cursors[i]] in selected:
                cursors[i] += 1
            if cursors[i] < len(ranked):
                selected.append(ranked[cursors[i]])
                cursors[i] += 1
                advanced = True
            if len(selected) == 40:
                break
        if not advanced:
            break
    for eid in original:
        if len(selected) == 40:
            break
        if eid not in selected:
            selected.append(eid)
    return selected


class HierarchicalRetriever:
    def __init__(self, *, ids, raw_vectors, frames, paths, snapshot, pcf,
                 node_ids, node_vectors):
        self.ids, self.raw = ids, raw_vectors
        self.index = {eid: i for i, eid in enumerate(ids)}
        self.frames = {f['id']: f for f in frames}
        self.node_ids, self.node_vectors = node_ids, node_vectors
        self.node_index = {fid: i for i, fid in enumerate(node_ids)}
        self.frame_rows = defaultdict(set)
        for path in paths:
            if path['frame'] not in self.frames or not set(path['evidence_ids']) <= set(ids):
                raise ValueError('Invalid source-aligned path')
            self.frame_rows[path['frame']].update(path['evidence_ids'])
        if not set(self.frame_rows) <= set(self.node_index):
            raise ValueError('Missing TaskFrame descriptor vector')
        self.parents, self.children = defaultdict(set), defaultdict(set)
        for edge in pcf['pcf_child_of']:
            child, parent = edge['child_id'], edge['parent_id']
            self.parents[child].add(parent)
            self.children[parent].add(child)
        self.mapping, self.reverse = defaultdict(set), defaultdict(set)
        for edge in pcf['pcf_mappings']:
            if edge['relationship_type'] != 'MAPS_TO_PCF':
                continue
            fid, pid = edge['taskframe_id'], edge['pcf_id']
            self.mapping[fid].add(pid)
            self.reverse[pid].add(fid)
        self.up, self.down = defaultdict(set), defaultdict(set)
        for edge in snapshot['edges']:
            if edge['type'] == 'SUBTASK_OF':
                self.up[edge['source']].add(edge['target'])
                self.down[edge['target']].add(edge['source'])

    def rank_raw(self, vector, allowed=None):
        scores = self.raw @ vector
        pool = self.ids if allowed is None else sorted(allowed)
        return sorted(pool, key=lambda eid: (-float(scores[self.index[eid]]), eid))

    def route(self, vector, hierarchy=True):
        values = self.node_vectors @ vector
        scores = {fid: float(values[self.node_index[fid]]) for fid in self.frame_rows}
        base = sorted(self.frame_rows, key=lambda fid: (-scores[fid], fid))[:2]
        routes = {fid: [{'kind': 'direct_global_frame', 'frame': fid}] for fid in base}
        if hierarchy:
            for fid in base:
                for parent in sorted(self.up[fid]):
                    routes.setdefault(parent, []).append(dict(kind='TaskFrame_up', source=fid, relation='SUBTASK_OF', target=parent))
                for child in sorted(self.down[fid]):
                    routes.setdefault(child, []).append(dict(kind='TaskFrame_down', source=child, relation='SUBTASK_OF', target=fid))
                for pid in sorted(self.mapping[fid]):
                    for other in sorted(self.reverse[pid]):
                        routes.setdefault(other, []).append(dict(kind='same_primary_PCF', source_frame=fid, PCF=pid, target_frame=other))
                    for parent in sorted(self.parents[pid]):
                        for near in sorted({parent} | self.children[parent]):
                            for other in sorted(self.reverse[near]):
                                routes.setdefault(other, []).append(dict(kind='PCF_up_then_down', source_frame=fid, source_PCF=pid, parent_PCF=parent, target_PCF=near, target_frame=other))
        selected = base + sorted((set(routes) & set(self.frame_rows)) - set(base), key=lambda fid: (-scores[fid], fid))[:2]
        allowed = set().union(*(self.frame_rows[fid] for fid in selected))
        return self.rank_raw(vector, allowed), dict(base_frames=base, selected_frames=selected,
            frame_names={f: self.frames[f]['name'] for f in selected}, frame_scores={f: scores[f] for f in selected},
            routes={f: routes[f] for f in selected}, eligible_source_rows=len(allowed),
            excluded_frames_without_aligned_paths=sorted(set(routes) - set(self.frame_rows)))

    def retrieve(self, query_vector, aspect_vectors, hierarchy=True):
        if not 1 <= len(aspect_vectors) <= 3:
            raise ValueError('TH-KGR expects one through three Aspect vectors')
        original = self.rank_raw(query_vector)[:40]
        rankings, traces = zip(*(self.route(v, hierarchy) for v in aspect_vectors))
        return dict(candidate_evidence_ids=merge_candidates(original, rankings),
                    original_Dense40=original, routes=list(traces), CE_scored=False, QA_executed=False)
