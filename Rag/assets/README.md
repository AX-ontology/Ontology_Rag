# 검색용 KG 입력

실제 자이제 실험의 KG를 검색 실행에 필요한 형태로 정리했습니다. 모델 가중치나 임베딩 캐시는 포함하지 않습니다.

| 파일 | 내용 |
|---|---|
| `graph.json` | 과업·과업 표현·주장 및 원문 ID 정렬 경로 |
| `snapshot.json` | 문맥 구성에 쓰는 노드 속성과 `SUBTASK_OF` 관계 |
| `pcf.json` | 프로세스 `CHILD_OF`와 primary `MAPS_TO_PCF` 연결 |
| `paths.json` | 원문에 정렬된 검색 경로·경로 설명 |
| `global_nodes.json` | 원본 속성으로 구성한 과업/프로세스 설명; TH-KGR의 Global 순위 대상은 과업 |
| `manifest.json` | 원본과 배포 파일의 SHA-256 |

원문 본문은 `../../data/jaije_qa_dataset/corpus.json`을 사용합니다. 원문 식별자를 새로 매기지 않습니다. `snapshot.json`은 기존 전체 Neo4j 스냅샷이 아니라 검색 코드가 읽는 속성만 남긴 실행용 자료입니다.

원래 실행 자료에서 다시 내보내려면 저장소 루트에서 실행합니다.

```bash
python Rag/scripts/export_graph_assets.py \
  --core-run /path/to/runs/core_20261031 \
  --global-nodes /path/to/global_nodes.json \
  --output /path/to/new_assets_directory
```

`Ontology/`로 새 KG를 구축하는 경우 해당 출력과 이 스키마의 노드·원문 ID·관계 방향을 맞춰야 합니다. 최신 온톨로지 구축 결과를 이 과거 실험 KG와 자동으로 동일시하지 않습니다.
