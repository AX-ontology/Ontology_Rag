"""Source-only KG context construction and explicit ablation controls.

No answer labels, document adjacency, or generated facts are used here.
"""
from collections import defaultdict
import hashlib
import json
import re

DEICTICS = re.compile(r"이것|그것|이러한|그러한|이처럼|이때|이 경우|이 과정|따라서|그러나|전자는|후자는|둘 다|그 결과")


class ContextBuilder:
    def __init__(self, graph, snapshot, evidence, token_count, seed=20261028):
        self.facts = {f["id"]: f for f in graph["facts"]}
        self.mentions = {m["id"]: m for m in graph["mentions"]}
        self.frames = {f["id"]: f for f in graph["frames"]}
        self.nodes = {n["p"]["id"]: n["p"] for n in snapshot["nodes"]}
        self.ev, self.token_count, self.seed = evidence, token_count, seed
        self.mf, self.fm = defaultdict(set), defaultdict(set)
        for path in graph["paths"]:
            for mid in self.mids(path):
                if path.get("fact"):
                    self.mf[mid].add(path["fact"])
                    self.fm[path["fact"]].add(mid)
        self.claims = {fid: self.claim(fid) for fid in self.facts}
        self.lengths = {fid: token_count(text) for fid, text in self.claims.items()}
        self.documents = {fid: {evidence[e]["source_file"] for e in fact["evidence_ids"]}
                          for fid, fact in self.facts.items()}

    @staticmethod
    def mids(path):
        return path.get("mentions", []) or ([path["mention"]] if path.get("mention") else [])

    def claim(self, fid):
        qualifiers = self.nodes.get(fid, {}).get("qualifiers") or []
        if isinstance(qualifiers, str):
            try:
                qualifiers = json.loads(qualifiers)
            except ValueError:
                qualifiers = []
        return self.facts[fid]["claim"] + (
            " (조건: " + "; ".join(q.get("text", "") for q in qualifiers if isinstance(q, dict)) + ")"
            if qualifiers else "")

    def bundle(self, path, eid, similarity):
        if path is None:
            return None
        assert eid in path["evidence_ids"], "Path must support the center evidence"
        mids = self.mids(path)
        mid = max(mids, key=lambda m: (len(self.nodes[m].get("name", "")), m)) if mids else None
        fid = path.get("fact")
        direct = [fid] if fid else []
        if mid:
            direct += sorted(f for f in self.mf[mid] if f != fid and eid in self.facts[f]["evidence_ids"])[:1]
        siblings = sorted(((similarity(e), e, f) for f in self.mf[mid]
                           for e in self.facts[f]["evidence_ids"] if e != eid),
                          key=lambda x: (-x[0], x[1], x[2])) if mid else []
        chosen = siblings[0] if siblings else None
        return {"path_id": path["id"], "frame_id": path["frame"], "mention_id": mid,
                "frame": self.frames[path["frame"]]["name"],
                "mention": self.nodes[mid].get("name", self.mentions[mid]["text"]) if mid else "",
                "title": self.nodes[mid].get("section_path", "") if mid else "",
                "direct_facts": direct, "sibling_fact": chosen[2] if chosen else None,
                "sibling_evidence": chosen[1] if chosen else None}

    def render(self, bundle, title=True, sibling=True, replacements=None):
        if bundle is None:
            return "", ""
        replacements = replacements or {}
        claim = lambda fid: self.claims[replacements.get(fid, fid)]
        text = "과업: " + bundle["frame"]
        if bundle["mention"]:
            text += "\n과업 실행: " + bundle["mention"]
        title_start = None
        if title and bundle["title"]:
            text += "\n과업이 설명된 절: "
            title_start = len(text)
            text += bundle["title"]
        if bundle["direct_facts"]:
            text += "\n이 원문에 연결된 주장: " + "; ".join(claim(fid) for fid in bundle["direct_facts"])
        if sibling and bundle["sibling_fact"]:
            text += " \n같은 과업 실행에 연결된 다른 주장: " + claim(bundle["sibling_fact"])
        # B11 receives only the exact title substring visible in P01's 500-char context.
        visible_title = (text[title_start:min(500, title_start + len(bundle["title"]))]
                         if title_start is not None and title_start < 500 else "")
        return text[:500], visible_title

    def donor(self, qid, eid, bundle, target_text, salt):
        original = set(bundle["direct_facts"]) | {bundle["sibling_fact"]}
        # Exclude every Mention of the selected path's original Facts as well as selected Mention.
        mids = {bundle["mention_id"]} | set().union(*(self.fm[f] for f in original if f))
        n = self.token_count(target_text)
        candidates = [f for f in sorted(self.facts)
                      if f not in original and eid not in self.facts[f]["evidence_ids"]
                      and not (self.fm[f] & mids)
                      and self.ev[eid]["source_file"] in self.documents[f]
                      and abs(self.lengths[f] - n) <= max(1, .1 * n)]
        if not candidates:
            return None
        return min(candidates, key=lambda f: hashlib.sha256(
            f"{self.seed}|{qid}|{eid}|{salt}|{f}".encode()).digest())

    def contexts(self, qid, eid, bundle, no_aspect_bundle):
        raw = self.ev[eid]["name"]
        eligible = bundle is not None and len(raw) <= 400
        gate = eligible and (len(raw) <= 100 or bool(DEICTICS.search(raw)))
        main, title = self.render(bundle)
        contexts = {m: "" for m in ["P01", "B11", *[f"A{i:02}" for i in range(1, 8)]]}
        trace = {"gate": gate, "always_gate": eligible, "bundle": bundle,
                 "no_aspect_bundle": no_aspect_bundle, "original_section": self.ev[eid].get("section_path", ""),
                 "p01_visible_title": title, "donors": {}, "control_unchanged": {}}
        if eligible:
            contexts["A07"] = main
        if gate:
            contexts.update(P01=main, B11=title, A01=self.render(no_aspect_bundle)[0],
                            A02="; ".join(self.claims[f] for f in bundle["direct_facts"])[:500],
                            A03=self.render(bundle, sibling=False)[0],
                            A04=self.render(bundle, title=False)[0])
            replacements = {}
            for f in dict.fromkeys([*bundle["direct_facts"], bundle["sibling_fact"]]):
                if f:
                    donor = self.donor(qid, eid, bundle, self.claims[f], "A05:" + f)
                    if donor:
                        replacements[f] = donor
            contexts["A05"] = self.render(bundle, replacements=replacements)[0]
            donor = self.donor(qid, eid, bundle, main, "A06")
            contexts["A06"] = self.claims[donor][:500] if donor else main
            trace["donors"] = {"A05": replacements, "A06": donor}
            trace["control_unchanged"] = {m: contexts[m] == main for m in ["A05", "A06"]}
            trace["control_token_delta"] = {m: self.token_count(contexts[m]) - self.token_count(main)
                                             for m in ["A05", "A06"]}
        return contexts, trace
