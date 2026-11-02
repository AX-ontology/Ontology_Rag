# 공통 TaskFrame 기준 최종 파이프라인

이 디렉터리는 `AX_ontology/Ontology` 독립 패키지의 일부다. 실행 준비와 비밀값 설정은
상위 `README.md`를 먼저 참고한다. 형제 디렉터리는 모두 같은 패키지 안에 포함돼 있다.

기본 실행은 `run_all.bash → run_catalog_pipeline.py`이다.
기존 승인 업무를 먼저 재사용하고, 실제 신규 업무만 추가하는 **평면 업무 목록**을 만든다.

## 방법론 HTML · 결과 시각화

```bash
python build_methodology_report.py
```

`../final_outputs/methodology_report/methodology.html`과 `results.html`을 브라우저로 연다.
서버·CDN 없이 동작하는 한국어 정적 HTML이며, 생성 과정에서 API 호출이나 DB 쓰기를 하지 않는다.
공통 업무별 문서 분포, 클릭 가능한 연결 지도, PCF 후보, 원문 근거 표본을 제공한다.
근거는 문서 조합별 Fact ID 순으로 최대 4개씩 표시하며 전체 근거가 아닌 표본이다.

기본 비교 대상은 `catalog_refactor_check/current.json`이 가리키는 최신 검증 결과와
`v4_5_flat_final/combined`의 이전 DB 적재 검증본이다. 최신 결과의 DB 미적재 상태와
이전 결과의 과거 검증 시점을 구분하며 실시간 DB 상태를 주장하지 않는다.
`--latest`, `--previous`, `--output`으로 자이제 결과 경로와 보고서 출력 경로를 바꿀 수 있다.
입력 그래프의 노드·관계 수, ID 중복, 연결 대상, 업무 구성원을 검증한 뒤 생성하며
`report_manifest.json`에 입력 파일 SHA-256을 남긴다. 원본 결과는 변경하지 않는다.

## 실행

```bash
cd /path/to/AX_ontology/Ontology/final_pipeline

# 실행 계획만 확인: 파일 생성·API 호출·DB 쓰기 없음
bash run_all.bash --dry-run

# 전체 처리: 기본적으로 DB에 쓰지 않음
bash run_all.bash

# 공통 업무 목록까지만 확인
bash run_all.bash --run-name catalog_check --stop-after catalog

# 검증 완료된 이번 실행의 그래프를 DB에 적재하고 실제 ID/관계를 조회 검증
bash run_all.bash --load-neo4j
```

기본 출력은 `../final_outputs/v4_5_catalog_final`이다.
이미 DB에 적재한 `v4_5_flat_final` 및 원본 `v4_5_final`은 덮어쓰지 않는다.
현재 경로로 기존 실행을 덮어쓰려 하면 오류로 막는다.

## 처리 순서

1. 원문 → 업무 언급·삼중항 추출 및 규칙 기반 표면 정규화.
   새 문서의 기존 추출기는 `entity_prep`까지만 실행한다. 임시 TaskFrame 생성·문서 내부 재통합은 생략한다.
2. **업무 언급 직접 배정:** `assign_mentions.py`가 각 TaskMention을 승인된 공통 목록에 직접 배정한다.
   같은 요청에서 애매한 Fact의 업무 언급 연결도 원문·인접 문맥으로 재판정한다.
   배치 밖의 연결 후보도 문맥에 보존한다. 이 API 요청에서는 신규 업무를 만들 수 없다.
   모든 기존 업무의 범위 밖이라는 근거가 있는 경우만 `NOVEL`, 불확실하면 `REVIEW`로 분리한다.
3. **신규 업무 생성:** `NOVEL` 후보만 따로 모아 큰 업무 단위로 묶는다.
   생성 요청에 기존 목록과 앞 배치에서 만든 신규 업무도 넣어 중복 생성을 줄인다.
   별도의 최종 목록 중복 검토 LLM 호출은 하지 않는다. 의미적 중복이 완전히 제거된다는 보장은 없다.
   업무 배정 또는 Fact 연결에 `REVIEW`가 있으면 검토 목록을 남기고 신규 생성 전에 중단한다.
   연결 후보가 없는 Fact도 검토 대상으로 중단하며, 자동 유사도 fallback으로 억지 연결하지 않는다.
4. 기존 ID·정의는 유지하고 새 업무를 공통 목록에 추가한다. 다음 문서는 갱신된 목록을 사용한다.
   하위 업무/상하위 관계는 생성하지 않는다.
5. 통합된 업무 기준으로 정규화·문서 그래프를 생성하고, 모든 문서의 Fact·근거를 모은다.
6. **문서별이 아니라 공통 업무와 전체 근거에 대해 PCF를 한 번 계산**하고, 전체 그래프를 정제한다.
7. TaskFrame·업무 언급·Fact·문서 보존, 관계 연결 및 적재 형식을 검증한다.
8. `--load-neo4j`를 명시한 경우에만 DB 적재 후 노드 ID/라벨과 관계 ID/종류/양 끝점을 대조한다.

배정 API는 입력 `S001…`과 기준 업무 `C001…`을 분리한다.
Fact는 `F001…`, 연결 대상 언급은 `M001…`로 별도 구분한다.
응답 오류가 있으면 정상 항목을 유지하고 오류 항목만 이유·허용 선택지와 함께 재판정한다.
반복 오류를 임의 배정으로 덮지 않는다. 승인된 기존 업무끼리의 재병합은 금지한다.
정식 경로는 `catalog-pipeline-3.0-direct`이다. `consolidate_taskframes.py`의 기존 프레임 기반
전략은 과거 결과 복구용 독립 도구로만 남기고 새 문서의 기본 경로에서는 실행하지 않는다.

### LLM 사용 범위와 호출 수

- 기본 2단계: 추출, 직접 기존 업무 배정 + 애매한 Fact 연결 판단.
- 조건부 1단계: NOVEL 업무 언급이 있을 때만 신규 묶기.
- 임시 업무 생성, 임시 업무 재통합, 최종 목록 중복 검토의 별도 LLM 요청은 제거했다.
- 한 단계는 여러 API 요청일 수 있다. 기본 업무 언급 48개/배치이며 Fact는 최대 160개/요청이다.
  Fact가 많으면 같은 단계에서 나눠 판단하고, 업무 배정은 첫 요청에만 포함한다.
- 새 배정 산출물의 `05_taskframes/validation_report.json`에 배치 수를 기록한다.
  배치 수는 캐시 적중·응답 재판정·HTTP 재시도를 반영한 실제 과금 호출 수가 아니다.
- 요청 횟수·토큰·비용이 일정 비율로 줄었다고 보장하지 않는다. 실문서 재처리 전후 측정이 필요하다.
- PCF는 로컬 BGE 임베딩·reranker, 정제·검증·DB·HTML은 코드 처리로 생성형 API를 쓰지 않는다.
- 그래프 재개는 `--catalog-assigned`로 완료된 직접 배정 파일을 요구한다. 파일이 빠지면
  기존 LLM 업무 생성기로 되돌아가지 않고 중단한다.

## 현재 자이제1·2·3 설정

`config.yaml`에는 이번에 승인된 결과를 재사용하도록 설정되어 있다.

- 기준 목록: 자이제1의 12개 TaskFrame.
- 자이제1: 기준 문서(`catalog_role: seed`).
- 자이제2·3: 완료된 원문 추출과 승인된 배정 결과(`approved_alignment`) 재사용.
- 기존 자이제2 검토 후보 55개에 대한 배정도 사용자의 승인대로 유지.
- 원문 SHA-256과 원래 TaskFrame/Fact 묶음의 digest를 대조한 뒤 재사용.
- 원문이 바뀌면 기존 추출·승인 배정을 사용하지 않고 다시 처리한다.
  기준 문서가 바뀌면 자동으로 기준을 바꾸지 않고 기준 목록 재검토를 요청한다.
- 공통 업무 정의가 변경되면 과거 승인을 자동 적용하지 않는다.
- TaskFrame 배정 승인과 PCF 확정 승인은 별개다. 이 실행기는 검토 대기 PCF를 임의 승격하지 않는다.

따라서 현재 설정에서는 업무 추출·배정 API를 다시 호출할 필요가 없다.
새로운 전체 근거 기준 PCF 계산은 과거 문서별 계산과 후보·점수가 달라질 수 있다.
이미 적재한 DB 결과를 바꾸려면 새 결과를 확인한 뒤 `--load-neo4j`를 명시한다.

## 새 문서 추가

`config.yaml`의 `documents`에 문서를 추가한다. 처음 처리하는 문서에는
`reuse_taskkg`, `approved_alignment`, `catalog_role`을 넣지 않는다.

```yaml
documents:
  new_document:
    input_xlsx: ../data/process_data/new_document.xlsx
    sheet: clean_sentences
    document_id: NEW_DOCUMENT
    document_title: 새 문서
```

`catalog_pipeline.baseline_taskframes`로 기준 업무 목록 파일을 지정한다.
필요하면 이전 실행의 승인된 공통 목록을 사용할 수 있다.
기준은 **목록 파일**이며 DB의 모든 과거 TaskFrame을 자동으로 읽는 방식은 아니다.

```bash
bash run_all.bash --run-name next_catalog --documents new_document
```

이 명령은 선택한 문서만 이번 그래프에 포함한다.
기존 문서의 근거까지 함께 합치려면 `--documents`에 기존 문서도 포함하거나,
전체 문서가 들어 있는 설정에서 이 옵션을 생략한다.
문서마다 고유한 `document_id`가 필요하다.

## 결과와 재개

- `current.json`: 현재 업무 목록·통합 입력·PCF·정제 결과·최종 패키지의 실제 경로.
- `catalog_pipeline_manifest.json`: 단계 상태, 입력 signature, DB 검증 이력.
- `artifacts/<단계>-<signature>/`: 변경 불가능한 완료 단계별 결과.
- `logs/`: 단계별 실행 로그.
- `db_loads/<run-id>.json`: 실제 DB 대조 검증 보고서.

같은 명령으로 재개한다. 입력·설정·관련 코드가 같고 완료 파일도 온전하면 재사용한다.
변경된 입력은 새로운 artifact 디렉터리를 만들고 이후 단계도 새 signature로 실행한다.
기존 결과는 삭제하지 않는다. 완료 산출물 자체가 훼손되면 자동 덮어쓰기 대신 중단한다.
중단된 `.preparing` 복사본이 남아 있다면 보존하고 새 run name으로 재시도한다.

DB 적재 이력은 일반 재실행에서 초기화하지 않는다.
DB 적재는 안정 ID 기반 MERGE이며 이전 실행 전용 데이터는 삭제하지 않는다.
입력 버전별 run ID로 결과를 구분하며, 검증까지 성공해야 적재 완료로 기록한다.

## 이전 실행기

- `run_final_pipeline.py`: 문서별 PCF를 계산하던 이전 실행기 및 공통 보조 함수.
- `run_flat_pipeline.py`: 이번 자이제1·2·3 결과를 복구·통합했던 고정 경로 실행기.
- `consolidate_taskframes.py`: 독립 TaskFrame 배정/통합 도구.
- `assign_mentions.py`: 새 정식 경로의 직접 TaskMention 배정·조건부 신규 생성 도구.

두 이전 실행기는 과거 결과의 추적을 위해 남겨두지만 기본 진입점은 아니다.
이전 `--start-at`, `--backend`, `--limit` 옵션을 새 진입점에 그대로 전달하지 않는다.
새 진입점의 중간 종료는 `--stop-after catalog|graph|pcf|clean|validate`를 사용한다.
기존의 완료된 결과와 DB 적재 검증 기록은 `v4_5_flat_final`에 그대로 남아 있다.

## 테스트

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s ../cleaning/tests -v
```

단위/통합 테스트는 API와 DB를 모의 처리하며, 기본 실행의 DB 쓰기 금지,
기존 우선 배정, 신규 후보 분리, 입력 변경에 따른 재개 무효화와 원문 보존을 확인한다.
