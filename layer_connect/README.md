# PCF–TaskFrame Layer Connect

`PCF_TASKFRAME_CONNECTION_METHODOLOGY.md`의 Layer 1(PCF)–Layer 2(TaskFrame)
연결 절차를 실행 가능한 파이프라인으로 구현한 디렉터리입니다. PCF 원본은 수정하지
않고 별도 crosswalk와 TaskKG adapter 산출물을 생성합니다.

## 실행

현재 Zaije v4.5.1 데이터 경로는 `config.yaml`에 설정되어 있습니다. 자이제
1·2·3을 한 번에 처리할 때는 `../final_pipeline/run_all.bash`를 사용합니다.

```bash
cd /home/hyuna/AX/layer_connect
python3 run_pipeline.py --config config.yaml
```

개별 단계를 다시 실행할 수도 있습니다. 개별 단계는 앞 단계 산출물이 이미 있어야
합니다.

```bash
python3 run_pipeline.py --config config.yaml --stage retrieve
python3 validate_mappings.py --config config.yaml
```

후보 검색은 한국어 sentence embedding을 필수로 사용합니다. 기본 설정은
`dragonkue/BGE-m3-ko`와
`/home/hyuna/.conda/envs/mmrs/bin/python`을 사용합니다. 의미 임베딩 생성에 실패하면
TF-IDF로 fallback하지 않고 실행을 중단합니다. TF-IDF는 임베딩 결과에 결합되는 보조
신호로도 사용하지 않으며, 현재 기본 설정의 후보 유사도는 100% embedding입니다.
통합 실행기는 공통 PCF corpus embedding cache를 문서 간 공유하고 문서별 query만
추가로 encoding합니다.

## 처리 단계

1. `build_pcf_index.py`: PCF 트리 무결성 검사, 전체 경로와 검색 필드 생성
2. `build_taskframe_packets.py`: TaskFrame–TaskMention–Fact provenance 결합 및 대표 근거 선별
3. `retrieve_candidates.py`: frame/mention/fact/section 다중 query 검색 및 후보별 근거 점수 출력
4. `select_document_context.py`: 전체 TaskFrame coverage로 문서 PCF context subtree 선택
5. `rerank_candidates.py`: `dragonkue/bge-reranker-v2-m3-ko`로 embedding 후보 정밀 재정렬
6. `aggregate_hierarchy.py`: 문서 subtree 내부/전체 PCF 후보 비교, 예외 이탈, LCA 판단
7. `validate_mappings.py`: 스키마, 트리, provenance, coverage 일관성 검사
8. `export_to_taskkg.py`: 기존 StandardTask adapter와 PCF KG node/edge 출력
9. `build_html_report.py`: TaskFrame 연결 행과 PCF 계층 트리를 포함한 standalone HTML 생성

## 문서 문맥과 TaskFrame 분기 규칙

문서 문맥은 긴 원문을 단일 벡터로 압축하지 않고, 문서에 속한 모든 TaskFrame의
`frame_score`와 PCF 경로 `path_score`를 PCF level 2 subtree별로 모아 평균 의미 점수와
TaskFrame coverage가 가장 높은 subtree를 선택합니다. 현재 데이터에서는 `3.5 영업 계획
개발·관리`가 선택됩니다.

그다음 각 TaskFrame마다 `3.5` 내부 최적 후보와 전체 PCF의 외부 최적 후보를 비교합니다.
기본값은 `3.5` 내부 후보이며, 외부 후보는 reranker 점수가 내부보다 `0.15` 이상 높고,
핵심 Fact가 2개 이상이며, 근거 coverage가 `0.20` 이상이고, 전문 도메인 불일치 penalty가
없을 때만 예외적으로 선택됩니다. 이 기준은 `config.yaml`의 `document_context`에서 조정할
수 있습니다.

## 핵심 산출물

- `outputs/pcf_nodes_enriched.jsonl`
- `outputs/taskframe_evidence_packets.jsonl`
- `outputs/taskframe_pcf_candidates.jsonl`
- `outputs/document_pcf_context.json`
- `outputs/taskframe_pcf_links.jsonl`
- `outputs/review_queue.jsonl`
- `outputs/validation_report.json`
- `outputs/standard_tasks_pcf.json`
- `outputs/standard_mappings_pcf.jsonl`
- `outputs/pcf_kg_nodes.jsonl`, `outputs/pcf_kg_edges.jsonl`
- `outputs/pcf_taskframe_crosswalk.html`

자동 판정 threshold는 초기값입니다. `needs_review` 결과를 사람이 검토해 12개
TaskFrame gold set을 만든 다음 `config.yaml`의 검색/집계 값을 calibration하는 것이
다음 단계입니다. `run_manifest.json`에는 단계별 실행 시간과 성공/실패 상태가 남습니다.
