# PCF–TaskFrame 의미 기반 계층 연결 방법론

## 1. 목적과 결론

이 문서는 Layer 1의 PCF(Process Classification Framework)와 Layer 2의
TaskFrame을 연결하는 현재 방법론과 판정 기준을 정의한다.

현재 방식의 핵심은 다음 두 단계다.

1. 문서에 속한 **전체 TaskFrame 분포**로 문서의 대표 PCF subtree를 먼저 선택한다.
2. 각 TaskFrame에서 **대표 subtree 내부 후보**와 **전체 PCF의 외부 후보**를 비교한다.

현재 문서에서는 12개 TaskFrame 전체의 의미 점수와 coverage를 집계한 결과,
`3.5 영업 계획 개발·관리`가 문서 PCF 문맥으로 선택되었다. 이후 TaskFrame별 판정에서는
3.5 내부 후보를 기본값으로 사용하되, 외부 process의 근거가 명확할 때만 3.5 밖으로
나가도록 한다.

```text
문서 전체 TaskFrame
  -> 대표 PCF subtree 선택: 3.5 영업 계획 개발·관리
  -> TaskFrame별 후보 비교
       ├─ 3.5 내부 최적 후보
       └─ 전체 PCF 중 3.5 외부 최적 후보
  -> 외부 후보의 우위와 근거가 충분하면 외부 process 선택
  -> 그렇지 않으면 3.5 내부 process 선택
```

이 방식은 생성형 LLM의 주관적 판정에 의존하지 않는다. 한국어 의미 임베딩,
cross-encoder reranker, 근거 coverage 및 PCF 계층 규칙으로 재현 가능한 결과를 만든다.

---

## 2. 기존의 전체 PCF 직접 비교가 잘 동작하지 않은 이유

초기 접근은 각 TaskFrame을 전체 2,017개 PCF 노드와 바로 비교하고, 유사도가 가장 높은
노드를 선택하는 방식이었다. 후보를 넓게 찾는 데에는 유용하지만 최종 매핑을 그대로
결정하기에는 다음 문제가 있었다.

### 2.1 업무 문맥과 원자 행위가 분리되지 않는다

제안 문서에는 `전략 수립`, `고객 분석`, `리스크 관리`, `일정 수립`처럼 다른 PCF
영역에도 흔히 등장하는 표현이 많다. 전체 PCF에서 바로 최대값을 고르면 문서의 실제
목적인 영업·제안 문맥보다 일반 전략, 고객 경험, 프로젝트 관리 등의 단어 유사성이
앞설 수 있다.

예를 들어 `고객 및 평가자 분석`은 일반 고객 경험 process와도 유사하지만, 이 문서에서
수행되는 실제 역할은 제안 전략과 제안서 개발을 지원하는 분석이다. 문서 문맥을 고려하지
않으면 일반 고객 process로 빠지기 쉽다.

### 2.2 전체 후보와 3.5 후보의 단순 최대값 비교는 비대칭이다

전체 PCF 후보 집합 안에는 3.5 후보도 포함된다. 따라서 다음 비교는 논리적으로 공정하지
않다.

```text
max(전체 PCF) vs max(3.5 내부)
```

왼쪽 집합이 오른쪽 집합을 포함하므로 전체 최대값은 항상 3.5 최대값 이상이다. 이 구조를
그대로 사용하면 미세한 점수 차이만으로도 외부 후보가 선택되어, 문서의 공통 업무 문맥이
쉽게 무너진다.

현재 방식은 전체 집합에서 3.5를 제외한 **외부 후보만 별도로 구성**한 뒤 다음을 비교한다.

```text
max(3.5 내부) vs max(3.5 외부)
```

### 2.3 유사도 최고점만으로는 ‘관련 있음’과 ‘실제 process’를 구분하기 어렵다

임베딩 점수가 높다는 것은 두 텍스트가 의미적으로 관련 있다는 뜻이지, 반드시 같은
업무 process라는 뜻은 아니다. 다음 항목을 함께 봐야 한다.

- TaskMention이 실제 수행하는 행위와 대상
- `core` Fact가 직접 지지하는 process
- 전체 핵심 근거 중 후보가 설명하는 비율
- PCF 부모 경로가 문서 문맥과 일치하는지
- 일부 일반 단어가 아니라 여러 독립 근거가 반복적으로 지지하는지

### 2.4 TaskFrame 크기와 PCF 깊이가 서로 다르다

TaskFrame마다 mention 수가 1개에서 수십 개까지 다르다. 넓은 TaskFrame을 하나의 긴
문장으로 만들면 반복 표현이 점수를 지배하고, 여러 하위 업무를 포함한 프레임이 임의의
leaf 하나로 과도하게 구체화될 수 있다. 반대로 이름만 짧은 TaskFrame은 우연히 비슷한
PCF 명칭에 붙기 쉽다.

따라서 대표 근거를 선별하고, 여러 sibling leaf가 함께 지지되면 최소 공통 조상(LCA)을
선택해야 한다.

### 2.5 실제 비교에서 확인된 패턴

전체 후보의 reranker 점수가 3.5 내부 후보보다 조금 높은 사례가 여러 개 있었지만,
대부분 차이가 작고 제안 문서 문맥을 버릴 만큼 강하지 않았다.

| TaskFrame 예시 | 외부 후보 우위 | 해석 |
|---|---:|---|
| 고객 및 평가자 분석 | `+0.101` | 관련은 있으나 3.5를 벗어날 정도로 명확하지 않음 |
| 영업 정보 관리 | `+0.075` | 일반 영업 process와 가깝지만 3.5 내부가 문서 역할에 적합 |
| 리스크 관리 | `+0.388` | 외부 리스크 process가 명확하게 우세 |

이 결과 때문에 단순 최대값 대신 **문서 subtree 기본값 + 충분히 강한 외부 예외** 구조를
채택했다.

---

## 3. 핵심 설계 원칙

### 3.1 순수 의미 검색을 사용한다

후보 검색은 `dragonkue/BGE-m3-ko` sentence embedding을 필수로 사용한다.
TF-IDF나 문자 일치 점수를 임베딩 결과에 섞지 않으며, 의미 임베딩이 실패하면 lexical
검색으로 fallback하지 않고 실행을 중단한다.

후보 정밀 비교에는 `dragonkue/bge-reranker-v2-m3-ko` cross-encoder를 사용한다.

### 3.2 문서 문맥은 prior이지 강제 잠금이 아니다

3.5가 문서 대표 subtree로 선택되더라도 모든 TaskFrame을 3.5에 강제로 넣지 않는다.
기본 검색 범위와 동점 해소 기준으로 사용하되, 명확한 외부 process에는 빠져나갈 수 있는
예외 경로를 둔다.

### 3.3 process와 context를 분리한다

`제안 과정의 리스크 관리`는 다음처럼 표현한다.

- primary `process_match`: `11.1.3 사업부·기능 리스크 관리`
- 부가 `context_anchor`: `3.5.3 영업 제안서·견적서 개발·관리`

업무가 수행되는 장소와 업무 자체의 종류를 하나의 PCF로 억지로 합치지 않는다.

### 3.4 모든 연결은 원문 근거로 추적 가능해야 한다

최종 링크에는 해당 결정을 지지한 다음 ID를 보존한다.

- `supporting_task_mentions`
- `supporting_triples`
- `supporting_evidence_ids`

### 3.5 가장 깊은 노드가 아니라 근거가 설명 가능한 깊이를 선택한다

특정 leaf를 반복적으로 지지하면 leaf를 선택하고, 근거가 여러 하위 가지에 분산되면
부모 또는 LCA로 올린다. 근거가 부족한 경우에는 `needs_review`나 `unmapped`를 허용한다.

---

## 4. 입력 데이터

### 4.1 PCF

```text
/home/hyuna/AX/data/ai-fellow-data/02_pcf/pcf_data.json
```

주요 필드는 `pcfid`, `hid`, `parent`, `level`, `nameKo`, `nameEn`, `descKo`,
`descEn`이다. 현재 PCF는 2,017개 노드로 구성된 평면형 트리이며, `parent`를 따라 전체
경로를 복원한다.

### 4.2 TaskFrame

```text
/home/hyuna/AX/taskframe_v4/outputs/zaije_1_v4_5_structured_refine/05_taskframes/taskframes.json
```

대표 이름, 정의, 범위뿐 아니라 `member_task_mentions`와 `evidence_ids`를 사용한다.

### 4.3 TaskMention

```text
/home/hyuna/AX/taskframe_v4/outputs/zaije_1_v4_5_structured_refine/03_entity_prep/task_mentions.jsonl
```

실제 문서에 나타난 원자 업무 표현이다. `surface_form`, 정규화된 행위·대상,
의도한 결과, section 경로와 원문 문장을 비교 근거로 사용한다.

### 4.4 Canonical Fact

```text
/home/hyuna/AX/taskframe_v4/outputs/zaije_1_v4_5_structured_refine/07_canonicalization/canonical_facts.jsonl
```

Fact의 `core`, `supporting`, `context` 관련도를 구분한다. 주변 설명인 `context` Fact가
직접 업무인 `core` Fact를 압도하지 않도록 별도 가중치와 상한을 적용한다.

---

## 5. 전체 파이프라인

```text
1. PCF 트리 및 검색 인덱스 생성
2. TaskFrame Evidence Packet 생성
3. 전체 PCF 의미 후보 검색
4. 문서 대표 PCF subtree 선택
5. cross-encoder 후보 재정렬
6. TaskFrame별 내부/외부 분기 및 계층 집계
7. 스키마·트리·근거 일관성 검증
8. TaskKG 형식으로 export
9. standalone HTML 시각화
```

실행 명령:

```bash
cd /home/hyuna/AX/layer_connect
python3 run_pipeline.py --config config.yaml --stage all
```

---

## 6. Stage 1: PCF 트리와 검색 표현 생성

`hid`와 `parent`를 사용하여 각 노드에 다음 파생 정보를 만든다.

```json
{
  "hid": "3.5.3.4",
  "path_ids": ["3.0", "3.5", "3.5.3", "3.5.3.4"],
  "path_text_ko": "제품·서비스 마케팅 및 판매 > 영업 계획 개발·관리 > 영업 제안서·견적서 개발·관리 > 경쟁 분석 수행",
  "is_leaf": true
}
```

검색 표현은 현재 노드 이름·정의와 전체 부모 경로를 분리해 보존한다. 모든 조상 정의를
하나의 긴 문자열로 합치면 상위 문장의 일반적인 의미가 leaf 의미를 희석할 수 있기
때문이다.

이 단계에서 중복 `hid`, 존재하지 않는 부모, level 불일치, 순환 참조와 고립 노드를
검증한다.

---

## 7. Stage 2: TaskFrame Evidence Packet 생성

TaskFrame마다 다음 정보를 하나의 packet으로 결합한다.

```json
{
  "taskframe_id": "TF_...",
  "frame_summary": {
    "canonical_name": "경쟁사 분석",
    "definition": "...",
    "scope": "..."
  },
  "mentions": [],
  "core_facts": [],
  "supporting_facts": [],
  "context_facts": [],
  "section_paths": [],
  "evidence_ids": []
}
```

TaskFrame 크기 차이와 반복 문장 편향을 줄이기 위해 현재 대표 근거 상한은 다음과 같다.

| 근거 | 최대 개수 |
|---|---:|
| TaskMention | 30 |
| core Fact | 30 |
| supporting Fact | 15 |
| context Fact | 5 |

Frame 요약, mention, Fact, section을 각각 독립 query로 사용한다. 긴 텍스트 하나로 합치지
않기 때문에 세부 업무 신호가 프레임 설명에 묻히는 것을 줄일 수 있다.

---

## 8. Stage 3: 전체 PCF 의미 후보 검색과 재정렬

각 query를 전체 PCF에 대해 의미 검색하고 후보 합집합을 만든다.

- query별 Top-K: 20
- TaskFrame별 후보 pool: 최대 100
- 출력 후보: 최대 30
- 임베딩 가중치: 1.0
- lexical/TF-IDF 혼합: 사용하지 않음

후보 점수에는 다음 의미 신호가 포함된다.

```text
CandidateScore =
    Frame 의미 유사도
  + TaskMention 지지
  + Fact 지지
  + PCF path 일관성
  + 상위 원자 근거 지지
  - 전문 도메인 불일치 penalty
```

검색된 후보는 TaskFrame 요약과 PCF 경로·정의를 함께 입력한 cross-encoder로 다시
정렬한다. 현재 최종 후보 점수에서 reranker의 가중치는 0.35다.

---

## 9. Stage 4: 문서 대표 PCF subtree 선택

### 9.1 문서 표현 방식

문서 전체를 한 개의 긴 embedding으로 만들지 않는다. 긴 문서는 일부 반복 문장과 넓은
표현에 의해 중심 의미가 희석될 수 있기 때문이다. 대신 문서에 속한 12개 TaskFrame을
문서의 의미 단위 집합으로 사용한다.

PCF level 2 subtree마다 각 TaskFrame의 가장 좋은 하위 후보를 찾는다.

```text
TaskFrameSemantic(t, p)
  = 0.65 × frame_score(t, p)
  + 0.35 × path_score(t, p)
```

각 subtree `b`의 문서 점수는 다음과 같다.

```text
MeanSemantic(b)
  = 모든 TaskFrame에서 subtree b의 최적 의미 점수 평균

Coverage(b)
  = subtree b에서 유효 후보를 가진 TaskFrame 수 / 전체 TaskFrame 수

DocumentScore(b)
  = 0.85 × MeanSemantic(b)
  + 0.15 × Coverage(b)
```

평균 점수만 보지 않고 coverage를 함께 사용하므로, TaskFrame 한두 개에만 강한 subtree가
문서 전체 대표로 선택되는 것을 방지한다.

### 9.2 현재 결과

| 순위 | PCF subtree | 문서 점수 | TaskFrame coverage |
|---:|---|---:|---:|
| 1 | `3.5 영업 계획 개발·관리` | `0.590138` | `12/12` |
| 2 | `1.2 사업 전략 수립` | `0.458544` | `10/12` |
| 3 | `13.2 포트폴리오·프로그램·프로젝트 관리` | `0.448611` | `10/12` |

3.5와 2위의 차이는 `0.131594`다. 따라서 현재 문서의 대표 업무 문맥을 3.5로 사용한다.

결과는 다음 파일에 전체 순위와 TaskFrame별 지지 후보를 함께 기록한다.

```text
outputs/document_pcf_context.json
```

---

## 10. Stage 5: TaskFrame별 3.5 내부와 전체 PCF 비교

### 10.1 후보 집합 분리

문서 subtree가 3.5로 결정되면 각 TaskFrame의 후보를 다음처럼 분리한다.

```text
Inside(t)  = PCF 3.5 또는 그 descendant 후보
Outside(t) = 전체 PCF 후보 중 3.5 descendant가 아닌 후보
```

각 집합에서 최적 후보를 구한 뒤 비교한다.

### 10.2 기본값

기본 선택은 `Inside(t)`의 최적 후보다. 이는 점수와 무관하게 3.5에 고정한다는 뜻이
아니라, 문서 전체에서 충분히 입증된 공통 문맥을 약한 국소 점수 차이보다 우선한다는
뜻이다.

### 10.3 외부 예외 조건

외부 후보는 다음 조건을 모두 만족할 때만 선택한다.

1. 외부 후보의 raw reranker 점수가 내부 후보보다 `0.15` 이상 높음
2. 외부 후보를 직접 지지하는 `core` Fact가 2개 이상
3. 외부 후보의 evidence coverage가 `0.20` 이상
4. 전문 도메인 불일치 penalty가 없음

```text
if outside_reranker - inside_reranker >= 0.15
   and outside_core_facts >= 2
   and outside_coverage >= 0.20
   and no_domain_penalty:
       select outside candidate
else:
       select inside candidate
```

`0.15`는 현재 12개 TaskFrame 결과를 진단해 설정한 초기 안전 margin이다. 향후 사람이
확정한 gold set이 생기면 calibration해야 한다.

### 10.4 리스크 관리 예시

```text
TaskFrame: 리스크 관리
3.5 내부 후보: 3.5.1.5 기회 수주 계획 수립
외부 후보: 11.1.3.1 계열 리스크 process
raw reranker 차이: +0.387683
외부 core Fact: 7개
판정: external_process_exception
```

계층 일반화 후 최종 primary는 `11.1.3 사업부·기능 리스크 관리`가 된다. 제안 문맥은
`3.5.3` context anchor로 별도 보존한다.

---

## 11. Stage 6: PCF 계층 깊이와 복수 연결 결정

내부/외부 가지가 결정된 후 다음 계층 규칙을 적용한다.

### 11.1 leaf 선택

TaskMention과 core Fact가 같은 세부 노드를 반복적으로 지지하고, 부모나 sibling보다
근거가 명확하면 leaf를 선택한다.

### 11.2 최소 공통 조상 선택

근거가 여러 sibling에 분산되면 특정 leaf 하나를 강제로 선택하지 않고 LCA로 올린다.

```text
제안서 작성 및 관리
  ├─ 내부 검토
  ├─ 내부 승인
  ├─ 고객 제출
  └─ 수정
       -> 3.5.3 영업 제안서·견적서 개발·관리
```

### 11.3 넓은 TaskFrame의 부모 승격

mention이 많은 넓은 TaskFrame은 leaf와 부모의 점수·coverage 차이가 작으면 부모를
선택한다. 일부 표현에만 맞는 leaf로 과도하게 구체화되는 것을 방지한다.

### 11.4 작은 TaskFrame

mention이 1개뿐인 TaskFrame은 3.5 내부에서 frame 의미와 path 의미를 다시 비교하되,
자동 확정하지 않고 `single_mention_taskframe` 검토 사유를 남긴다.

### 11.5 mapping role

| 역할 | 의미 |
|---|---|
| primary `process_match` | TaskFrame이 수행하는 대표 업무 |
| secondary `process_match` | 독립 근거가 있는 추가 업무 가지 |
| `context_anchor` | 업무가 수행되는 문서·사업 문맥 |
| `unmapped` | 최소 점수나 coverage를 충족하지 못함 |

---

## 12. 현재 12개 TaskFrame 결과

| TaskFrame | primary PCF | 판정 요약 |
|---|---|---|
| 제안전략 수립 | `3.5.3` 영업 제안서·견적서 개발·관리 | 3.5 내부 기본 |
| 고객 및 평가자 분석 | `3.5.3` 영업 제안서·견적서 개발·관리 | 3.5 내부, 넓은 부모로 승격 |
| 제안서 작성 및 관리 | `3.5.3` 영업 제안서·견적서 개발·관리 | 여러 하위 가지의 LCA |
| 영업 활동 및 관계 구축 | `3.5` 영업 계획 개발·관리 | 넓은 범위의 상위 process |
| 리스크 관리 | `11.1.3` 사업부·기능 리스크 관리 | 강한 외부 예외 |
| 프레젠테이션 준비 및 수행 | `3.5.1.8.2` 사전 영업 활동 수행 | 3.5 내부, leaf 검토 필요 |
| 경쟁사 분석 | `3.5.3.4` 경쟁 분석 수행 | 직접 의미 일치 |
| 입찰 의사결정 및 기회 관리 | `3.5.1.5` 기회 수주 계획 수립 | 직접 의미 일치 |
| 영업 정보 관리 | `3.5.1.5` 기회 수주 계획 수립 | 3.5 내부 후보 유지 |
| 전술 개발 및 솔루션 최적화 | `3.5.3.6` 고객 비즈니스·요구사항 이해 | 3.5 내부, leaf 검토 필요 |
| 프로그램 개발 일정 수립 | `3.5.3` 영업 제안서·견적서 개발·관리 | 3.5 내부, 세부 위치 검토 필요 |
| 고객 평가 체크 | `3.5.3.4` 경쟁 분석 수행 | 소형 TaskFrame, 검토 필요 |

primary 기준 12개 중 11개는 3.5 subtree 내부이며, 리스크 관리 1개만 외부 process로
선택되었다. 상위 구조는 문서 의미와 일관적이지만, 프레젠테이션·전술/솔루션·일정·고객
평가 체크와 같은 일부 세부 leaf는 사람이 한 번 확인하는 것이 안전하다.

현재 모든 primary는 `needs_review`로 유지한다. 이는 연결 실패가 아니라, 아직 사람이
확정한 12개 gold mapping이 없으므로 자동 확정하지 않았다는 뜻이다.

---

## 13. 관계 범위와 검토 상태

`relation_scope`는 다음 값을 사용한다.

| 값 | 의미 |
|---|---|
| `equivalent` | TaskFrame과 PCF 범위가 실질적으로 유사 |
| `taskframe_narrower_than_pcf` | TaskFrame이 PCF보다 세부적 |
| `taskframe_broader_than_pcf` | TaskFrame이 여러 PCF 하위 업무를 포괄 |
| `partial_overlap` | 일부 내용만 겹침 |
| `related` | 직접 process가 아닌 관련 문맥 |
| `unmapped` | 대응 PCF를 결정할 근거 부족 |

주요 검토 사유는 다음과 같다.

- `low_score`
- `low_margin`
- `low_evidence_coverage`
- `frame_vs_evidence_conflict`
- `cross_root_conflict`
- `single_mention_taskframe`
- `possible_taskframe_split`
- `multiple_process_branches`

절대 similarity 하나만으로 자동 승인하지 않고 score, margin, coverage, 계층 일관성과
근거 provenance를 함께 검증한다.

---

## 14. 출력과 검증

### 14.1 주요 산출물

| 파일 | 내용 |
|---|---|
| `outputs/pcf_nodes_enriched.jsonl` | 경로가 복원된 PCF 검색 인덱스 |
| `outputs/taskframe_evidence_packets.jsonl` | TaskFrame별 대표 mention·Fact·Evidence |
| `outputs/taskframe_pcf_candidates.jsonl` | 전체 PCF 의미 후보와 세부 점수 |
| `outputs/document_pcf_context.json` | 문서 subtree 순위와 3.5 선택 근거 |
| `outputs/taskframe_pcf_links.jsonl` | 최종 Crosswalk |
| `outputs/review_queue.jsonl` | 사람 검토 대상과 사유 |
| `outputs/validation_report.json` | 스키마·트리·근거 검증 결과 |
| `outputs/pcf_taskframe_crosswalk.html` | 전체 연결 시각화 |

### 14.2 검증 항목

- 모든 TaskFrame에 결과가 존재하는지
- mapped TaskFrame마다 primary가 정확히 1개인지
- PCF `hid`, `pcfid`, 전체 경로가 원본 트리와 일치하는지
- supporting mention이 해당 TaskFrame 소속인지
- supporting triple과 Evidence가 실제로 존재하고 해당 frame과 연결되는지
- 문서 대표 PCF가 설정한 level에 존재하는지
- 문서 대표 PCF가 필수 의미 임베딩 결과인지
- `needs_review` 결과가 review queue에 포함되는지

현재 실행 결과는 다음과 같다.

```text
TaskFrame: 12
Primary mapping: 12
Unmapped: 0
Validation errors: 0
Validation warnings: 0
Unit tests: 9 passed
```

---

## 15. 구현 모듈

| 모듈 | 책임 |
|---|---|
| `build_pcf_index.py` | PCF 트리 복원과 검색 표현 생성 |
| `build_taskframe_packets.py` | TaskFrame–Mention–Fact–Evidence 결합 |
| `retrieve_candidates.py` | 전체 PCF 순수 의미 후보 검색 |
| `select_document_context.py` | 전체 TaskFrame coverage로 문서 subtree 선택 |
| `rerank_candidates.py` | 한국어 cross-encoder 후보 재정렬 |
| `aggregate_hierarchy.py` | 3.5 내부/외부 분기, LCA, 부모 승격, 복수 연결 |
| `validate_mappings.py` | 스키마·트리·provenance 검증 |
| `export_to_taskkg.py` | TaskKG node/edge 및 표준 매핑 형식 export |
| `build_html_report.py` | standalone HTML 생성 |
| `run_pipeline.py` | 9개 단계를 의존 순서대로 실행 |

판정 임계값과 모델 설정은 `config.yaml`에서 관리한다. 출력 링크의 `method` 값은 현재
방식을 반영한 다음 문자열을 사용한다.

```text
semantic_document_context_gate_evidence_hierarchy_rerank
```

---

## 16. LLM 사용 여부와 향후 보정

현재 결과만 보면 생성형 LLM 판정은 필수가 아니다. 의미 임베딩과 reranker가 후보를
충분히 좁히고, 문서 문맥 gate가 대부분의 잘못된 외부 이탈을 막고 있다.

LLM을 추가한다면 전체 PCF에서 직접 하나를 고르게 해서는 안 된다. 다음 경우의 Top-K
후보 설명 및 검토 보조에만 제한적으로 사용하는 것이 적절하다.

- 내부/외부 점수 차이가 threshold 근처인 경우
- 서로 다른 PCF 가지에 core 근거가 분산된 경우
- 세부 leaf 간 의미 차이가 작아 사람이 설명을 확인해야 하는 경우
- TaskFrame split 여부를 검토하는 경우

다음 단계는 12개 TaskFrame의 사람이 확정한 gold mapping을 만들고, 이를 기준으로
`external_reranker_margin`, LCA, 부모 승격 및 자동 승인 threshold를 calibration하는
것이다. 그 전까지 `needs_review`는 정상적인 안전장치로 유지한다.
