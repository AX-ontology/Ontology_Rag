#!/usr/bin/env python3
"""Export only source properties needed by the published retrievers."""
import argparse
import hashlib
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--core-run',type=Path,required=True);p.add_argument('--global-nodes',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    sources=['graph/aligned_ontology.json','inputs/ontology_snapshot.json','inputs/pcf_snapshot.json','fresh_path_embeddings/paths.json']
    graph,snapshot,pcf,paths=[read(a.core_run/n) for n in sources]
    needed={f['id'] for group in ['frames','facts','mentions'] for f in graph[group]}
    compact={'nodes':[{'p':{k:v for k,v in n['p'].items() if k in ['id','name','qualifiers','section_path','scope','definition']}} for n in snapshot['nodes'] if n['p'].get('id') in needed], 'edges':[e for e in snapshot['edges'] if e['type']=='SUBTASK_OF']}
    process={'pcf_child_of':pcf['pcf_child_of'],'pcf_mappings':[e for e in pcf['pcf_mappings'] if e['relationship_type']=='MAPS_TO_PCF']}
    a.output.mkdir(parents=True,exist_ok=False)
    for name,value in [('graph',graph),('snapshot',compact),('pcf',process),('paths',paths),('global_nodes',read(a.global_nodes))]:
        (a.output/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,separators=(',',':'))+'\n')
    manifest={'source_sha256':{name:sha(a.core_run/name) for name in sources},'global_nodes_source_sha256':sha(a.global_nodes),'files_sha256':{f.name:sha(f) for f in a.output.iterdir()}}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Exported graph assets:',a.output)
if __name__=='__main__':main()
