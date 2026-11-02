# TaskFrame v4.5 + PCF cleaning

원본 `taskframe_v4`와 `layer_connect` 산출물은 수정하지 않고, Neo4j에 넣을
정제 projection을 이 디렉터리에서 재현 가능하게 생성한다.

## 실행

```bash
cd /home/hyuna/AX/cleaning
python3 clean_and_export.py
python3 -m unittest discover -s tests -v
python3 load_neo4j.py --validate-only --print-queries
```

실제 Neo4j 쓰기는 별도 확인 후 다음처럼 실행한다.

```bash
cp .env.example .env
# .env의 접속 정보를 실제 값으로 수정
python3 load_neo4j.py --print-queries
```

`.env`는 아래 두 username 변수명 중 하나를 사용할 수 있다.

```dotenv
NEO4J_URI=neo4j+s://<instance-id>.databases.neo4j.io
NEO4J_USERNAME=neo4j
# NEO4J_USER=neo4j  # 기존 이름도 지원
NEO4J_PASSWORD=<password>
```

연결만 먼저 확인하려면 `python3 load_neo4j.py --check-connection`을 실행한다.

## Layer 1 / Layer 2 색상

- `PCFProcess`에는 `:Layer1`, `graphLayer=layer1`, 파랑 `#2563EB`을 적용한다.
- TaskFrame과 근거/개념 서브그래프에는 `:Layer2`, `graphLayer=layer2`, 주황
  `#F97316`을 적용한다.
- 두 레이어를 연결하는 PCF mapping 관계는 `graphLayer=cross_layer`, 보라
  `#7C3AED`로 기록한다.

Neo4j Browser에서 `:style`을 열어 `neo4j_style.grass`를 import하면 위 색상으로
표시된다. `neo4j_view_queries.cypher` 또는 `python3 load_neo4j.py --print-queries`가
출력하는 Layer 1↔2 쿼리로 바로 확인할 수 있다.

`load_neo4j.py`는 현재 정제 JSONL뿐 아니라 TaskFrame v4 계열의
`id,label,name,properties_json` / `id,source,target,type,layer,evidence_ids,properties_json`
CSV도 읽을 수 있다. 노드와 관계는 안정적인 `id`로 `MERGE`하므로 같은 입력을 다시
적재해도 중복 생성되지 않는다.

## 모델링 결정

- 원문 업무 표현은 `TaskMention-[:MEMBER_OF]->TaskFrame`으로 보존한다.
- 짧은 검색 동의어만 `TaskFrameAlias`로 만들고
  `TaskFrame-[:HAS_ALIAS]->TaskFrameAlias`로 연결한다.
- raw `EntityMention`과 `RelationExpression`은 삭제하지 않는다.
- 정제 기준을 통과한 concept만 `CanonicalEntity`와 `CanonicalRelation`으로 만들고,
  탈락한 후보는 `outputs/quarantine`에 보존한다.
- 근거 경계 오류가 있는 Fact는 삭제하지 않고 `quality_status=quarantined`,
  `active=false`로 표시한다.
- 확정된 PCF 연결은 역할에 따라 `MAPS_TO_PCF`, `ALSO_MAPS_TO_PCF`,
  `HAS_PCF_CONTEXT`로 구분한다.
- `needs_review` PCF 연결은 확정으로 위장하지 않고
  `PCF_MAPPING_CANDIDATE`, `active=false`로 보존한다.
- 검토 전 후보를 강제로 확정하려면 `--accept-needs-review-pcf`를 명시해야 한다.

## 주요 산출물

- `outputs/neo4j/nodes.jsonl`
- `outputs/neo4j/edges.jsonl`
- `outputs/summary.json`
- `outputs/validation_report.json`
- `outputs/audit/taskframe_alias_decisions.jsonl`
- `outputs/audit/relation_expression_decisions.jsonl`
- `outputs/audit/quality_issues.jsonl`
- `outputs/audit/pcf_mapping_decisions.jsonl`
- `outputs/quarantine/canonical_entities.jsonl`
- `outputs/quarantine/canonical_relations.jsonl`
- `outputs/registry/entity_registry.jsonl`
- `outputs/registry/relation_registry.jsonl`

registry는 향후 preferred name이나 alias가 바뀌더라도 기존 concept ID를 재사용하기
위한 파일이므로 다음 실행에서도 보존한다.
