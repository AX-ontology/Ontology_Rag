// 1) 현재 run의 Layer 1 / Layer 2 개수
MATCH (n:KgNode)
WHERE n.lastRunId = 'zaije_1_v4_4_cleaned'
RETURN n.graphLayer AS layer, count(*) AS nodes
ORDER BY layer;

// 2) Layer 1(PCF, 파랑) ↔ Layer 2(TaskFrame, 주황)
MATCH path=(t:TaskFrame:Layer2)
  -[:MAPS_TO_PCF|ALSO_MAPS_TO_PCF|HAS_PCF_CONTEXT]->
  (p:PCFProcess:Layer1)
RETURN path;

// 3) TaskFrame별 primary PCF mapping
MATCH (t:TaskFrame:Layer2)-[r:MAPS_TO_PCF]->(p:PCFProcess:Layer1)
RETURN t.name AS taskframe,
       p.hid AS pcf_code,
       p.name AS pcf_process,
       r.confidence AS confidence
ORDER BY taskframe;

// 4) 한 TaskFrame의 Layer 2 근거와 Layer 1 PCF를 함께 펼치기
// 아래 이름을 조회할 TaskFrame 이름으로 바꿔서 실행한다.
MATCH (t:TaskFrame:Layer2 {name: '리스크 관리'})
OPTIONAL MATCH evidence_path=(t)-[:HAS_CORE_FACT|HAS_SUPPORTING_FACT|HAS_CONTEXT_FACT]->(f:Fact:Layer2)
OPTIONAL MATCH pcf_path=(t)-[:MAPS_TO_PCF|ALSO_MAPS_TO_PCF|HAS_PCF_CONTEXT]->(p:PCFProcess:Layer1)
RETURN t, evidence_path, pcf_path;
