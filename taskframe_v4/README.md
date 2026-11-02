# Task-Centric Open KG v4.5.1

`task_centric_open_kg_methodology.md`를 실행 가능한 파이프라인으로 구현한 코드다.
v3.5.2에서 검증된 Evidence 원문 보존, LLM JSON 검증, 캐시·재시도,
문장 audit, resume 버전 검사를 유지하고 v4의 Task 중심 open KG 구조로 변경했다.

입력 본문 열은 `text_for_llm_human`, `text_for_llm`, `사람 검수` 순서로 자동
선택한다. 자이제 1·2·3 전체 통합 실행은 `../final_pipeline/run_all.bash`를 사용한다.

## v3와 다른 핵심 동작

- LLM 1이 자유형 Fact, TaskMention, `core/supporting/context` 연결을 공동 추출한다.
- 각 Fact는 원문 `subject_raw/raw_predicate/object_raw`, 도메인 독립
  `knowledge_function`, condition/temporal/exception qualifier,
  `source_id + exact_quote` 근거를 가진다.
- LLM 1 프롬프트에는 여섯 knowledge function과 원문 경계 오류를 다루는
  필수 few-shot이 항상 포함된다.
- LLM 1에서는 Task와 직접 관련 없는 Fact도 미연결 상태로 보존하고,
  LLM 2와 알고리즘 fallback이 최종적으로 가장 관련 있는 TaskFrame에 context로 연결한다.
- 최종 KG에서는 모든 Fact가 최소 하나 이상의 TaskFrame에 연결된다.
- 하나의 Fact가 여러 TaskMention과 TaskFrame에 연결될 수 있다.
- TaskFrame은 고정 슬롯 레코드가 아니라 관련 Fact를 연결한 지식 서브그래프다.
- LLM 2는 candidate block을 보존한 bounded local batch에서 TaskFrame을 만든 뒤,
  compact TaskFrame만 한 번 전역 reconciliation한다.
- 경계가 애매한 Fact 연결은 해당 local batch 안에서 함께 판정하므로 별도 호출이 없다.
- 애매한 Fact는 현재 Evidence, 현재 chunk, 앞·뒤 chunk, section path 순으로
  문맥을 확인하여 가장 관련 있는 TaskFrame에 연결한다.
- Entity/Relation canonicalization, Fact–TaskFrame 전파, 온톨로지 승격 후보,
  표준업무 후보 검색은 알고리즘으로 수행한다.

## 설치

```bash
cd /home/hyuna/AX/taskframe_v4
python3 -m pip install -r requirements.txt
cp .env.example .env
```

`.env`에 OpenAI-compatible API 값을 넣는다. v4 `.env`가 없으면 `run.bash`는
기존 v3의 `.env`를 자동으로 사용한다.

## 실행

비용 없는 코드/KG smoke test:

```bash
BACKEND=heuristic LIMIT=30 OUT_DIR=outputs/smoke_30 bash run.bash
```

LLM 100 Evidence 실행:

```bash
bash run.bash
```

전체 Evidence 실행:

```bash
LIMIT=0 OUT_DIR=outputs/zaije_1_v4_full bash run.bash
```

전체 실행의 기본 LLM 2 배치는 TaskMention 60개다. 메모리보다 모델 context가
작은 환경에서는 다음처럼 줄일 수 있다.

```bash
LIMIT=0 LLM2_MAX_MENTIONS=40 OUT_DIR=outputs/zaije_1_v4_full bash run.bash
```

LLM 1 호출 수를 줄이려면 추출 batch를 늘릴 수 있다. 기본값 4가 가장
보수적이며, batch 8은 전체 책의 예상 LLM 1 호출을 104회에서 52회로 줄이지만
한 응답의 JSON이 커지므로 pilot 품질을 먼저 확인한다.

```bash
LIMIT=0 EXTRACTION_BATCH_SIZE=8 OUT_DIR=outputs/zaije_1_v4_full_b8 bash run.bash
```

중단된 동일 v4 출력 재사용:

```bash
RESUME=1 OUT_DIR=outputs/zaije_1_v4_full LIMIT=0 bash run.bash
```

NCS/APQC/PCF 형식의 표준업무 후보 매핑을 함께 수행하려면:

```bash
STANDARDS_JSON=examples/standard_tasks.sample.json bash run.bash
```

표준 파일 스키마는 `examples/standard_tasks.sample.json`을 참고한다.

## 주요 산출물

```text
outputs/<run>/
  run_manifest.json
  01_evidence/          EvidenceSpan과 의미 청크
  02_extraction/        자유형 Fact, TaskMention, 로컬 연결, sentence audit
  03_entity_prep/       보수적 사전/문자열 정규화
  04_clustering/        high-recall 비교 후보 block과 pair 진단
  05_taskframes/        TaskFrame, Task 관계, 앞·뒤 chunk 기반 애매한 Fact 재판정
  06_propagation/       다중 Fact–TaskFrame 연결
  07_canonicalization/  CanonicalEntity/Relation 및 온톨로지 승격 후보
  08_standard_mapping/  선택적 NCS/APQC/PCF 후보
  09_kg/                nodes/edges JSONL·CSV, validation/quality report
```

`09_kg/quality_report.json`에서 원본 Fact 보존, Evidence grounding,
TaskMention partition, 전체 Fact 연결률, 다중 TaskFrame 연결 수를 확인할 수 있다.
미연결 Fact가 하나라도 있으면 최종 KG validation은 실패한다.

`run.bash`는 KG 생성이 끝나면 기본적으로 같은 출력 디렉터리에
`task_centric_open_kg_report.html`을 만든다. 기존 결과의 HTML만 다시 만들려면:

```bash
python3 build_html_report.py outputs/<run>
```

자동 생성을 끄려면 `GENERATE_HTML=0 bash run.bash`를 사용한다.

`04_clustering/llm_call_plan.json`에는 실행 전 예상 호출 수가,
`04_clustering/llm2_batch_plan.jsonl`에는 각 LLM 2 local batch가 기록된다.
기본 100 Evidence는 보통 LLM 2 한 번으로 끝나며, 현재 전체 책의 heuristic 규모
(TaskMention 328개)에서는 기본값 기준 local 6회 + global reconciliation 1회가
예상된다. 애매한 Fact 연결용 추가 호출은 0회다.

## 직접 실행 옵션

```bash
python3 run_pipeline.py --help
```

대표 옵션:

- `--backend llm|heuristic`
- `--limit 0`
- `--candidate-threshold 0.28`
- `--context-chunk-radius 1`
- `--llm2-max-mentions 60`
- `--promotion-min-support 3`
- `--standards-json PATH`
- `--stop-after STAGE`
- `--resume`

## 테스트

```bash
python3 -m py_compile run_pipeline.py taskkg/*.py tests/*.py
python3 -m unittest discover -s tests -v
```
