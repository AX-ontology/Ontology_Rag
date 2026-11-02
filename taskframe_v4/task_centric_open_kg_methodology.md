# Task-Centric Open KG 구축 방법론

## 1. 연구 개요

### 1.1 연구 목표

본 연구의 목표는 수주·제안 관련 문서에서 업무 지식을 추출하고, 문서에 분산되어 있는 다양한 사실을 **업무(Task)** 중심으로 조직하여 활용 가능한 지식그래프를 구축하는 것이다.

기존의 단순 삼중항 기반 지식그래프는 문서에 포함된 개별 사실을 표현하는 데에는 적합하지만, 다음과 같은 업무 중심 질문에 답하기 어렵다.

- 특정 업무는 누가 수행하는가?
- 해당 업무는 무엇을 대상으로 하는가?
- 어떤 문서나 정보를 입력으로 사용하며 어떤 산출물을 생성하는가?
- 어떤 조건이나 의사결정과 연결되는가?
- 어떤 업무가 선행 또는 후행하는가?
- 동일한 업무가 문서마다 어떤 표현으로 나타나는가?
- 문서에서 발견된 업무가 NCS, APQC, PCF 등의 표준 업무와 어떻게 대응되는가?

이를 해결하기 위해 본 연구에서는 문서의 자유형 삼중항을 보존하면서, 관련 삼중항을 **TaskFrame 중심의 지식 서브그래프**로 구성한다.

---

### 1.2 핵심 연구 방향

본 연구는 다음 세 가지 원칙을 따른다.

1. **문서의 의미 있는 사실을 자유형 삼중항으로 최대한 보존한다.**
2. **TaskFrame을 고정된 슬롯 집합이 아니라 Task 중심 지식 서브그래프로 정의한다.**
3. **도메인 스키마를 처음부터 완전히 고정하지 않고, 자유 추출 결과를 누적·정규화하여 점진적으로 확장한다.**

전체 접근은 다음과 같이 요약할 수 있다.

> Minimal Meta-Schema  
> + Task-aware Open Triple Extraction  
> + Task-Centric Knowledge Organization  
> + Bottom-up Ontology Induction

---

## 2. 핵심 개념 정의

### 2.1 Evidence

Evidence는 추출된 지식의 근거가 되는 원문 단위이다.

Evidence에는 다음 정보가 포함된다.

- 문서 ID
- 섹션 ID
- 섹션 경로
- 문단 ID
- 문장 ID
- 원문 텍스트
- 페이지 및 위치 정보
- 표·각주·목록 등의 구조 정보

모든 삼중항과 TaskMention은 반드시 하나 이상의 Evidence와 연결되어야 한다.

---

### 2.2 자유형 삼중항

자유형 삼중항은 문서에서 명시적으로 표현된 사실을 원문의 의미와 표현을 최대한 유지하여 추출한 구조이다.

```text
Subject → Predicate → Object
```

예시는 다음과 같다.

```text
제안 PM → 분석한다 → 고객 요구사항
RFP → 포함한다 → 평가 기준
요구사항 분석 → 산출한다 → 요구사항 분석표
요구사항 분석표 → 활용된다 → 제안전략 수립
필수 요구사항 미충족 → 초래할 수 있다 → 제안 탈락
```

초기 추출 단계에서는 관계를 사전에 정의된 제한된 목록으로 강제하지 않는다.

그 이유는 다음과 같다.

- 문서에 존재하는 도메인 고유 관계를 놓치지 않기 위해서
- 초기 스키마 편향을 줄이기 위해서
- 후속 누적 데이터에 기반해 관계 체계를 발견하기 위해서
- 원문 의미를 지나치게 일반화하지 않기 위해서

---

### 2.3 TaskMention

TaskMention은 문서의 특정 위치에서 관찰된 로컬 업무 표현이다.

예시는 다음과 같다.

```text
RFP를 검토한다
고객 요구사항을 분석한다
필수 요구조건을 확인한다
제안요청서의 요구사항을 파악한다
```

TaskMention은 최종적으로 정규화된 업무 개념이 아니다.

동일한 업무가 문서의 여러 섹션에서 서로 다른 표현으로 나타날 수 있으므로, TaskMention은 후속 단계에서 TaskFrame으로 병합되거나 분리된다.

TaskMention은 후보 클러스터링을 지원하기 위해 다음 정보를 가질 수 있다.

| 필드 | 의미 |
|---|---|
| `surface_form` | 문서에 나타난 업무 표현 |
| `action_raw` | 원문 행위 표현 |
| `focal_object_raw` | 업무의 중심 대상 표현 |
| `intended_outcome_raw` | 문맥에서 추론 가능한 업무 목적 또는 기대 결과 |
| `evidence_ids` | 근거 Evidence |
| `section_path` | 업무가 등장한 문서 구조 위치 |

이 필드들은 고정된 온톨로지 슬롯이라기보다 TaskMention 비교를 돕는 **로컬 Task Signature**이다.

---

### 2.4 TaskFrame

본 연구에서 TaskFrame은 다음과 같이 정의한다.

> **TaskFrame은 정규화된 업무 개념을 중심으로, 해당 업무의 수행 주체, 대상, 입력, 산출물, 조건, 목적, 절차, 의사결정, 결과 및 관련 맥락을 설명하는 모든 삼중항과 Evidence를 연결한 Task 중심 지식 서브그래프이다.**

TaskFrame은 단순한 다음 형태의 레코드가 아니다.

```text
Task
- Actor
- Input
- Output
- Target
```

대신 원래의 삼중항 구조를 유지하면서 TaskFrame을 중심으로 관련 Fact를 연결한다.

```text
TaskFrame: RFP 요구사항 분석
│
├─ 제안 PM → 분석한다 → 고객 요구사항
├─ 영업대표 → 검토한다 → 필수 요구조건
├─ RFP → 포함한다 → 평가 기준
├─ 요구사항 분석 → 산출한다 → 요구사항 분석표
├─ 요구사항 분석표 → 활용된다 → 제안전략 수립
├─ 필수 요구사항 미충족 → 초래할 수 있다 → 제안 탈락
└─ RFP 요구사항 분석 → 지원한다 → Bid/No Bid 의사결정
```

따라서 TaskFrame은 다음 두 요소로 구성된다.

```text
TaskFrame
├─ Task Core
│  ├─ canonical task name
│  ├─ definition
│  ├─ aliases
│  ├─ scope
│  └─ member TaskMentions
│
└─ Task-related Knowledge Subgraph
   ├─ core facts
   ├─ supporting facts
   ├─ context facts
   ├─ related entities
   ├─ related tasks
   └─ supporting evidence
```

---

### 2.5 TaskFrame과 삼중항의 관계

TaskFrame은 삼중항을 내부 속성으로 변환하여 원래 관계를 제거하지 않는다.

각 삼중항은 별도의 Fact로 유지하고 TaskFrame과 연결한다.

```text
TaskFrame -[:HAS_CORE_FACT]-> Fact
TaskFrame -[:HAS_SUPPORTING_FACT]-> Fact
TaskFrame -[:HAS_CONTEXT_FACT]-> Fact
```

하나의 삼중항은 여러 TaskFrame과 관련될 수 있다.

예를 들어 다음 삼중항은 두 업무를 연결한다.

```text
요구사항 분석표 → 활용된다 → 제안전략 수립
```

이 Fact는 다음 TaskFrame에 동시에 연결될 수 있다.

```text
RFP 요구사항 분석
제안전략 수립
```

따라서 Fact를 하나의 TaskFrame에만 강제 배정하지 않고, 다중 연결을 허용한다.

---

## 3. 온톨로지 스키마 설계 원칙

### 3.1 최소 메타 스키마

본 연구에서는 도메인 관계를 처음부터 모두 고정하지 않는다.

대신 지식 추출과 관리에 필요한 최소 메타 스키마만 사전에 정의한다.

#### 핵심 클래스

| 클래스 | 의미 |
|---|---|
| `Document` | 원본 문서 |
| `EvidenceSpan` | 지식의 원문 근거 |
| `Fact` | 문서에서 추출된 자유형 삼중항 |
| `EntityMention` | 원문에 등장한 엔터티 표현 |
| `CanonicalEntity` | 정규화된 엔터티 개념 또는 인스턴스 |
| `RelationExpression` | 원문에서 추출된 관계 표현 |
| `TaskMention` | 문서에 등장한 로컬 업무 표현 |
| `TaskFrame` | 정규화된 업무 중심 지식 서브그래프 |
| `CanonicalTerm` | 정규화된 용어 |
| `StandardTask` | NCS·APQC·PCF 등에 정의된 표준 업무 |

#### 핵심 메타 관계

```text
Fact -[:HAS_SUBJECT]-> EntityMention
Fact -[:HAS_PREDICATE]-> RelationExpression
Fact -[:HAS_OBJECT]-> EntityMention
Fact -[:SUPPORTED_BY]-> EvidenceSpan

TaskMention -[:SUPPORTED_BY]-> EvidenceSpan
Fact -[:RELATED_TO_TASK_MENTION]-> TaskMention
TaskMention -[:MEMBER_OF]-> TaskFrame

TaskFrame -[:HAS_CORE_FACT]-> Fact
TaskFrame -[:HAS_SUPPORTING_FACT]-> Fact
TaskFrame -[:HAS_CONTEXT_FACT]-> Fact

EntityMention -[:CANONICALIZED_AS]-> CanonicalEntity
RelationExpression -[:CANONICALIZED_AS]-> CanonicalTerm
TaskFrame -[:MAPS_TO_STANDARD_TASK]-> StandardTask
```

---

### 3.2 엔터티의 본질적 타입과 문맥적 역할 분리

RFP와 같은 엔터티는 Input, Target, Reference 등으로 각각 다른 노드를 만들지 않는다.

동일한 실제 엔터티는 하나의 canonical node로 관리한다.

```text
RFP rdf:type DocumentArtifact
```

특정 Task에서의 역할은 관계로 표현한다.

```text
RFP 요구사항 분석 -[:USES_INPUT]-> RFP
RFP 요구사항 분석 -[:ACTS_ON]-> RFP
Bid Validation -[:VALIDATES]-> RFP
제안서 작성 -[:REFERENCES]-> RFP
```

즉 다음을 구분한다.

| 구분 | 질문 | 예시 |
|---|---|---|
| 본질적 타입 | 대상 자체가 무엇인가? | RFP는 문서 산출물이다 |
| 문맥적 역할 | 특정 업무에서 어떻게 사용되는가? | RFP는 분석 업무의 입력이다 |

`Input`, `Target`, `Output`은 엔터티의 영구 타입이 아니라 특정 Task와의 관계에서 나타나는 역할이다.

---

### 3.3 개념과 인스턴스 구분

일반적인 RFP 개념과 실제 사업별 RFP 문서는 구분해야 한다.

```text
RFP                           ← 문서 유형 또는 개념
├─ A사업_RFP_2026             ← 실제 문서 인스턴스
├─ B사업_RFP_2026
└─ C사업_RFP_2026
```

문서가 일반적인 업무 지침을 설명하는 경우에는 RFP 개념 노드를 사용할 수 있다.

실제 프로젝트 문서를 다룰 경우에는 각각 별도 인스턴스로 생성하고 상위 RFP 개념과 연결한다.

---

## 4. 전체 방법론

```text
문서 구조화
  ↓
Evidence 보존
  ↓
[LLM 1]
자유형 삼중항 + TaskMention + 로컬 연결 공동 추출
  ↓
보수적 Entity 정규화
  ↓
TaskMention 후보 클러스터링
  ↓
[LLM 2]
TaskFrame 구성 및 TaskMention 병합·분리
  ↓
Triple–TaskFrame 연결 자동 전파
  ↓
Entity·Relation canonicalization
  ↓
동의·상하·관련 관계 판별
  ↓
안정화된 개념·관계의 온톨로지 승격
  ↓
NCS·APQC·PCF 매핑
```

---

# 5. STEP 1. 문서 구조화 및 Evidence 보존

## 5.1 입력 단위

입력 문서는 문서 구조를 보존한 상태로 처리한다.

```text
Document
└─ Section
   └─ Paragraph
      └─ Sentence
```

각 단위에는 고유 ID를 부여한다.

```text
document_id
section_id
section_path
paragraph_id
sentence_id
```

표, 목록, 각주, 캡션 등은 가능한 경우 별도의 구조 단위로 유지한다.

---

## 5.2 LLM 입력 청크

LLM 입력은 문장 하나가 아니라 **섹션 또는 의미 단락 단위 청크**로 구성한다.

권장 입력 구성은 다음과 같다.

```text
문서 제목
섹션 경로
해당 섹션 또는 의미 단락의 원문
문장별 Evidence ID
```

이전 문장과 다음 문장을 기계적으로 추가하기보다, 하나의 섹션 또는 의미적으로 응집된 단락을 함께 입력한다.

---

## 5.3 Evidence 우선 원칙

추출 결과는 원문과 분리되지 않는다.

모든 Fact와 TaskMention은 반드시 다음을 포함한다.

```text
evidence_id
source_text
section_path
document_id
```

정규화 또는 병합 이후에도 원문 표현과 Evidence는 삭제하지 않는다.

---

# 6. STEP 2. LLM 1: Task-aware Open Extraction

## 6.1 LLM 1의 목적

LLM 1은 문서 청크에서 다음 세 가지를 한 번에 수행한다.

1. 의미 있는 자유형 삼중항 추출
2. 로컬 TaskMention 추출
3. 삼중항과 TaskMention 사이의 관련성 연결

LLM 1은 최종 TaskFrame을 생성하지 않는다.

---

## 6.2 LLM 1의 추출 원칙

LLM 1은 다음 순서를 따르도록 설계한다.

```text
LLM 1은 각 청크를 한 번 입력받아 자유형 삼중항과 로컬 TaskMention을 공동 추출하고, 동일한 출력 내에서 각 삼중항과 TaskMention 간의 관련성을 연결한다. 이때 TaskMention은 해당 청크에서 근거가 확인되는 로컬 업무 표현으로 한정하며, Task와 직접 관련되지 않거나 청크만으로 연결을 판단하기 어려운 삼중항은 삭제하지 않고 task_links를 빈 배열로 유지한다.
```

Task 중심 삼중항만 추출하도록 지시하지 않는다.

그 이유는 업무와 직접 관련되지 않더라도 문서 이해와 후속 온톨로지 확장에 필요한 사실이 존재할 수 있기 때문이다.

---

## 6.3 삼중항 추출 출력

```json
{
  "fact_id": "T01",
  "subject_raw": "제안 PM",
  "raw_predicate": "분석한다",
  "object_raw": "RFP 요구사항",
  "knowledge_function": "procedure",
  "qualifiers": [],
  "evidence": {
    "source_id": "E12",
    "exact_quote": "제안 PM은 RFP 요구사항을 분석한다."
  },
  "task_links": [
    {
      "task_mention_id": "M01",
      "relevance": "core"
    }
  ]
}
```

`knowledge_function`은 도메인 개념이 아니라 Fact의 문서 기능이며
`fact | definition | procedure | rule | decision | recommendation` 중 하나다.
조건·시간·예외 표현은 `condition | temporal | exception` qualifier로 분리한다.
모든 raw 슬롯과 qualifier는 `exact_quote`의 연속 substring이고,
`exact_quote`는 `source_id` 원문의 연속 substring이어야 한다.

초기 단계에서 다음과 같이 관계를 강제 변환하지 않는다.

```text
분석한다 → ANALYZES
확인한다 → REVIEWS
작성한다 → PRODUCES
```

이러한 정규화는 누적 데이터에 기반한 후속 단계에서 수행한다.

---

## 6.4 TaskMention 출력

```json
{
  "task_mention_id": "M01",
  "surface_form": "RFP 요구사항을 분석한다",
  "action_raw": "분석한다",
  "focal_object_raw": "RFP 요구사항",
  "intended_outcome_raw": "필수 요구조건과 평가기준을 파악한다",
  "evidence_ids": ["E12", "E13"],
  "section_path": "영업참여 검토 > RFP 분석"
}
```

`intended_outcome_raw`는 문서에 명시되어 있거나 청크 내에서 충분히 근거가 있는 경우에만 추출한다.

근거가 부족한 경우 빈 값으로 유지한다.

---

## 6.5 Triple–TaskMention 연결 유형

각 삼중항은 TaskMention과 다음 유형으로 연결한다.

| 유형 | 의미 |
|---|---|
| `core` | Task 수행 자체를 직접 표현 |
| `supporting` | Task의 목적, 조건, 산출물, 결과 등을 설명 |
| `context` | Task 이해에 필요한 주변 정보 |
| 연결 없음 | Task와 직접적인 관련이 없는 일반 Fact |

예시는 다음과 같다.

```text
제안 PM → 분석한다 → RFP 요구사항
→ RFP 요구사항 분석 / core

요구사항 분석 → 산출한다 → 요구사항 분석표
→ RFP 요구사항 분석 / supporting

RFP → 포함한다 → 평가 기준
→ RFP 요구사항 분석 / context
```

하나의 Fact는 여러 TaskMention에 연결될 수 있다.

---

## 6.6 LLM 1 전체 출력 예시

```json
{
  "task_mentions": [
    {
      "task_mention_id": "M01",
      "surface_form": "RFP 요구사항을 분석한다",
      "action_raw": "분석한다",
      "focal_object_raw": "RFP 요구사항",
      "intended_outcome_raw": "필수 요구조건을 파악한다",
      "evidence_ids": ["E12", "E13"]
    },
    {
      "task_mention_id": "M02",
      "surface_form": "제안전략을 수립한다",
      "action_raw": "수립한다",
      "focal_object_raw": "제안전략",
      "intended_outcome_raw": "수주 가능성을 높이는 대응방향을 결정한다",
      "evidence_ids": ["E14"]
    }
  ],
  "facts": [
    {
      "fact_id": "T01",
      "subject_raw": "제안 PM",
      "raw_predicate": "분석한다",
      "object_raw": "RFP 요구사항",
      "knowledge_function": "procedure",
      "qualifiers": [],
      "evidence": {
        "source_id": "E12",
        "exact_quote": "제안 PM은 RFP 요구사항을 분석한다."
      },
      "task_links": [
        {
          "task_mention_id": "M01",
          "relevance": "core"
        }
      ]
    },
    {
      "fact_id": "T02",
      "subject_raw": "요구사항 분석표",
      "raw_predicate": "활용된다",
      "object_raw": "제안전략 수립",
      "knowledge_function": "fact",
      "qualifiers": [],
      "evidence": {
        "source_id": "E14",
        "exact_quote": "요구사항 분석표는 제안전략 수립에 활용된다."
      },
      "task_links": [
        {
          "task_mention_id": "M01",
          "relevance": "supporting"
        },
        {
          "task_mention_id": "M02",
          "relevance": "core"
        }
      ]
    }
  ]
}
```

---

# 7. STEP 3. 보수적 Entity 정규화

## 7.1 목적

LLM 1에서 자유롭게 추출된 엔터티 표현 가운데 명백하게 동일한 표현만 우선 통합한다.

이 단계의 목적은 완전한 엔터티 정규화가 아니라 TaskMention 후보 생성과 검색의 안정성을 높이는 것이다.

---

## 7.2 자동 통합 대상

다음과 같은 표현은 알고리즘과 사전을 통해 보수적으로 통합할 수 있다.

- 띄어쓰기 차이
- 대소문자 차이
- 조사 또는 단순 어미 차이
- 영문 약어와 전체 명칭
- 괄호 병기 표현
- 문서에 명시된 동의 표현
- 기존에 검증된 용어사전 표현

예시:

```text
RFP
제안요청서
Request for Proposal
→ RFP
```

---

## 7.3 즉시 통합하지 않는 대상

다음과 같은 표현은 차이가 의미적일 수 있으므로 후보로만 유지한다.

```text
제안서
제안서 초안
제안서 최종본
기술제안서
가격제안서
```

또한 다음 행위 표현도 즉시 동의어로 통합하지 않는다.

```text
검토한다
분석한다
검증한다
평가한다
승인한다
```

유사하더라도 업무 목적과 프로세스 단계가 다를 수 있다.

---

# 8. STEP 4. TaskMention 후보 클러스터링

## 8.1 목적

TaskMention 후보 클러스터링은 최종 TaskFrame을 만드는 단계가 아니다.

LLM 2가 비교해야 할 후보 범위를 줄이는 **blocking 단계**이다.

이 단계에서는 정밀도보다 재현율을 우선한다.

잘못 같은 후보군에 포함된 Mention은 LLM 2가 분리할 수 있지만, 처음부터 서로 다른 후보군으로 나뉜 Mention은 비교 기회를 잃기 때문이다.

---

## 8.2 후보 생성 특징

TaskMention 간 유사도 계산에는 다음 정보를 활용한다.

| 특징 | 설명 |
|---|---|
| TaskMention 임베딩 | 전체 업무 표현의 의미 유사도 |
| Action 유사도 | 행위 표현의 유사성 |
| Focal object 유사도 | 중심 대상의 유사성 |
| Intended outcome 유사도 | 업무 목적 또는 기대 결과의 유사성 |
| 공유 Entity | 연결된 엔터티 중복 |
| 연결 Fact | TaskMention과 연결된 삼중항 문맥 |
| Section path | 문서 내 프로세스 위치 |
| 문서 내 거리 | 동일 섹션 또는 인접 섹션 여부 |

---

## 8.3 Action–Target의 역할

Action–Target은 최종 Task 동일성 기준이 아니다.

다음 용도로만 사용한다.

- 후보 검색
- blocking
- 유사 TaskMention 탐색
- 비교 대상 축소

최종 동일성 판단은 LLM 2가 다음 요소를 함께 고려하여 수행한다.

```text
업무 목적
핵심 행위
중심 대상
프로세스 단계
업무 범위
추상화 수준
연결된 Fact
```

---

# 9. STEP 5. LLM 2: Global TaskFrame Induction

## 9.1 LLM 2의 목적

LLM 2는 TaskMention 후보군과 연결 Fact를 바탕으로 문서 전역에서 TaskFrame을 구성한다.

LLM 2는 새로운 삼중항을 추출하는 모델이 아니다.

다음 판정을 수행한다.

1. 동일 TaskMention 병합
2. 잘못 묶인 TaskMention 분리
3. canonical task name 생성
4. TaskFrame 정의와 범위 생성
5. Task 간 동일·상하·관련·선후행 관계 판정
6. 경계가 애매한 Triple–Task 연결 재판정

---

## 9.2 TaskFrame 동일성 기준

동일 TaskFrame 여부는 다음 요소를 기준으로 판단한다.

| 요소 | 설명 |
|---|---|
| Intended outcome | 업무가 달성하려는 결과 |
| Action family | 검토, 분석, 작성, 승인 등의 핵심 행위군 |
| Focal object | 업무가 주로 다루는 대상 |
| Process scope | 어느 프로세스 단계의 업무인지 |
| Abstraction level | 상위 업무인지 세부 업무인지 |
| Related facts | 연결된 조건, 산출물, 의사결정, 결과 |

Actor, Input, Output은 일반적으로 업무의 속성으로 취급한다.

Actor나 산출물이 다르다는 이유만으로 TaskFrame을 자동 분리하지 않는다.

다만 그 차이가 업무 목적, 프로세스 단계 또는 업무 범위를 변화시키는 경우에는 별도 TaskFrame으로 분리한다.

---

## 9.3 TaskMention 병합 예시

```text
M01: RFP 요구사항 분석
M07: 고객 요구조건 검토
M15: 제안요청서 필수조건 확인
```

위 세 TaskMention이 동일한 업무 목적과 프로세스 범위를 가지는 경우 다음과 같이 병합한다.

```text
F01: RFP 요구사항 분석
```

---

## 9.4 TaskMention 분리 예시

```text
RFP 적합성 검토
제안서 품질 검토
계약조건 검토
```

`검토`라는 Action이 같더라도 대상, 목적, 프로세스 단계가 다르므로 별도 TaskFrame으로 분리할 수 있다.

---

## 9.5 Task 간 관계

LLM 2는 TaskFrame 간 다음 관계를 판정할 수 있다.

| 관계 | 의미 |
|---|---|
| `sameTask` | 동일한 업무 |
| `subTaskOf` | 상위·하위 업무 |
| `relatedTask` | 의미적으로 관련된 업무 |
| `precedes` | 선행 업무 |
| `follows` | 후행 업무 |
| `supports` | 다른 업무나 의사결정을 지원 |
| `distinctTask` | 유사하지만 별도 업무 |

예시:

```text
필수 요구사항 확인 -[:SUBTASK_OF]-> RFP 요구사항 분석
RFP 요구사항 분석 -[:SUPPORTS]-> Bid/No Bid 의사결정
RFP 요구사항 분석 -[:PRECEDES]-> 제안전략 수립
```

---

## 9.6 TaskFrame 출력 예시

```json
{
  "frame_id": "F01",
  "canonical_name": "RFP 요구사항 분석",
  "definition": "제안 참여 및 제안서 작성을 위해 RFP에 명시된 고객 요구사항, 필수조건 및 평가기준을 식별하고 분석하는 업무",
  "scope": "RFP 수신 이후 제안 참여 판단 및 제안전략 수립 이전 단계",
  "member_task_mentions": ["M01", "M07", "M15"],
  "task_relations": [
    {
      "relation": "supports",
      "target_frame_id": "F03"
    }
  ]
}
```

---

## 9.7 애매한 Triple 연결 판정

LLM 1에서 이미 명확한 TaskMention 연결이 생성된 Triple은 해당 연결을 우선 보존한다.
LLM 1에서 연결되지 않은 모든 Triple은 최종 TaskFrame 포괄성을 위해 LLM 2의
문맥 기반 연결 판정 대상으로 제공한다.

다음 조건에 해당하는 Triple은 LLM 2에서 특히 주의하여 판정한다.

- TaskMention 연결이 없지만 Task 관련 가능성이 높은 Triple
- 여러 TaskMention에 연결되었으나 관련성 유형이 불명확한 Triple
- 서로 다른 TaskFrame 후보에 동시에 연결된 Triple
- TaskFrame 경계를 결정하는 데 중요한 조건·결과 Fact
- 관계 표현이 지나치게 일반적이거나 모호한 Triple

LLM 2는 현재 Evidence, 현재 chunk, 앞·뒤 chunk, section path와 후보
TaskMention을 함께 보고 모든 미연결 Triple을 최소 하나 이상의 TaskMention에 연결한다.
업무 수행을 직접 나타내지 않는 정의·통계·배경 Fact는 가장 관련 있는 업무에
`context`로 연결한다. 하나의 Fact가 여러 업무에 관련되면 다중 연결을 허용한다.

이를 통해 LLM 2의 역할을 TaskFrame 경계 판정에 집중시킨다.

---

# 10. STEP 6. Triple–TaskFrame 연결 자동 전파

LLM 1은 다음 연결을 생성한다.

```text
Triple → TaskMention
```

LLM 2는 다음 연결을 생성한다.

```text
TaskMention → TaskFrame
```

따라서 최종 연결은 알고리즘으로 합성할 수 있다.

```text
Triple → TaskMention → TaskFrame
```

예시:

```text
T1 → M1
T2 → M1
T3 → M1, M2

M1, M5, M8 → F1
M2, M7 → F2
```

최종 결과:

```text
F1 → T1
F1 → T2
F1 → T3

F2 → T3
```

이 단계에는 추가 LLM이 필요하지 않다.

하나의 Fact가 여러 TaskFrame에 연결되는 것을 허용한다.

LLM 2 응답에서 누락된 Fact가 있으면 동일 Evidence, 동일 chunk, 동일 section,
Fact–TaskFrame 의미 유사도 순으로 보수적 fallback을 적용한다. 최종 KG에서는
모든 Fact가 최소 하나 이상의 TaskFrame에 연결되어야 하며, 미연결 Fact가 남으면
validation 실패로 처리한다.

---

# 11. STEP 7. Entity·Relation Canonicalization

## 11.1 기본 원칙

자유형 추출 결과를 즉시 하나의 고정 스키마에 맞추지 않는다.

먼저 원문 표현을 누적한 뒤, 반복적으로 등장하는 엔터티와 관계 표현을 클러스터링한다.

원문 표현은 삭제하지 않고 canonical form과 연결한다.

```text
제안요청서
RFP
Request for Proposal
  ↓
Canonical Entity: RFP
```

---

## 11.2 정규화 대상 분리

다음 정규화는 서로 분리하여 수행한다.

| 대상 | 목적 |
|---|---|
| Entity canonicalization | 동일 엔터티 표현 통합 |
| Relation canonicalization | 관계 표현의 동일·상하·관련 관계 정리 |
| Task canonicalization | TaskMention을 TaskFrame으로 통합 |
| Type induction | 반복되는 엔터티·관계 유형 발견 |

Task canonicalization은 LLM 2에서 수행한다.

Entity와 Relation canonicalization은 TaskFrame 문맥을 활용해 후속 알고리즘으로 수행한다.

---

## 11.3 클러스터링 입력 정보

관계 표현만 단독으로 임베딩하지 않는다.

다음 문맥을 함께 사용한다.

```text
subject 표현 및 타입 후보
predicate 원문 표현
object 표현 및 타입 후보
원문 문장
연결된 TaskFrame
section path
```

예시:

```text
[Subject: 제안 PM / Role 후보]
[Predicate: 검토한다]
[Object: RFP / DocumentArtifact 후보]
[TaskFrame: RFP 요구사항 분석]
```

이 문맥을 사용하면 동일한 `검토한다`라도 다음을 구분할 수 있다.

```text
RFP 검토
제안서 품질 검토
계약조건 검토
성과 검토
```

---

## 11.4 동의·상하·관련 관계 구분

유사한 표현을 모두 동의어로 합치지 않는다.

클러스터 내 표현 관계는 다음과 같이 구분한다.

```text
sameAs
broaderThan
narrowerThan
relatedTo
inverseOf
unresolved
```

예시:

```text
RFP = 제안요청서
```

```text
제안전략 수립
├─ 가격전략 수립
└─ 기술전략 수립
```

```text
검토한다
분석한다
검증한다
→ 관련 표현이지만 동일 관계로 즉시 병합하지 않음
```

불확실한 클러스터는 `unresolved` 상태로 유지한다.

---

# 12. STEP 8. 온톨로지 스키마 승격

## 12.1 승격 원칙

자유 추출된 모든 표현을 정식 온톨로지 클래스 또는 관계로 승격하지 않는다.

다음 조건을 만족하는 개념과 관계만 승격한다.

- 여러 문서 또는 여러 섹션에서 반복적으로 등장
- 의미가 일관됨
- canonicalization 결과가 안정적임
- TaskFrame 질의에 실질적으로 필요함
- domain과 range를 설명할 수 있음
- Evidence를 통해 의미를 검증할 수 있음

---

## 12.2 관계 승격 예시

자유형 관계 표현:

```text
작성한다
생성한다
수립한다
도출한다
```

이를 하나의 `produces`로 단순 통합하지 않고 문맥에 따라 구조화한다.

```text
createsArtifact
├─ writesDocument
├─ developsPlan
└─ derivesResult
```

충분한 사례와 문맥이 확보된 경우 정식 관계로 승격한다.

---

## 12.3 스키마 버전 관리

도메인 온톨로지는 데이터 누적에 따라 변화할 수 있으므로 버전 정보를 관리한다.

```text
schema_version
created_at
updated_at
source_documents
support_count
validation_status
```

기존 표현을 삭제하지 않고 새 canonical concept와 mapping을 추가하는 방식으로 확장한다.

---

# 13. STEP 9. NCS·APQC·PCF 매핑

## 13.1 매핑 시점

표준 업무 매핑은 문서 기반 TaskFrame을 먼저 확립한 후 수행한다.

표준체계를 TaskFrame 생성 전에 제시하면, 문서에 실제로 존재하는 업무가 NCS·APQC의 분류체계에 맞춰 과도하게 변형될 수 있다.

따라서 다음 순서를 따른다.

```text
문서 기반 TaskFrame 생성
  ↓
표준 업무 후보 검색
  ↓
매핑 검증
```

---

## 13.2 후보 검색

다음 정보를 이용해 표준 업무 Top-k 후보를 생성한다.

- TaskFrame canonical name
- TaskFrame definition
- Action family
- Focal object
- Input 및 Output
- 상위 Process
- 연결 Fact
- Task 추상화 수준

초기에는 embedding 및 구조 유사도를 사용한다.

---

## 13.3 매핑 관계

표준 업무와의 연결은 완전 동일 관계만 허용하지 않는다.

```text
exactMatch
closeMatch
broaderMatch
narrowerMatch
relatedMatch
unmapped
```

예시:

```text
RFP 요구사항 분석
  -[:CLOSE_MATCH]->
NCS 제안요청서 분석
```

매핑 결과는 TaskFrame 자체의 의미를 변경하지 않는다.

---

# 14. LLM 사용 위치

본 연구에서는 LLM 사용을 최대 두 번으로 제한한다.

| 단계 | LLM 사용 | 역할 |
|---|---:|---|
| 문서 구조화 | X | 규칙 기반 |
| Evidence 보존 | X | 규칙 기반 |
| 자유형 삼중항 추출 | LLM 1 | 개방형 의미 추출 |
| TaskMention 추출 | LLM 1 | 로컬 업무 표현 추출 |
| Triple–TaskMention 연결 | LLM 1 | 추출 시점의 업무 문맥 보존 |
| 보수적 Entity 정규화 | X | 사전·문자열·규칙 |
| TaskMention 후보 클러스터링 | X | 임베딩·그래프 알고리즘 |
| TaskFrame 구성 | LLM 2 | 전역 병합·분리 |
| Task 관계 판정 | LLM 2 | 동일·상하·관련·선후행 판정 |
| 애매한 Triple 재판정 | LLM 2 | 경계 사례만 처리 |
| Triple–TaskFrame 전파 | X | 그래프 경로 합성 |
| Entity·Relation 클러스터링 | X | 임베딩·규칙·누적 통계 |
| 온톨로지 승격 | X | 안정성 기준 및 검증 |
| NCS·APQC·PCF 후보 검색 | X | 임베딩 및 구조 유사도 |

---

# 15. 최종 지식그래프 구조

```text
(Document)
   └─[:HAS_SECTION]→(Section)
        └─[:HAS_EVIDENCE]→(EvidenceSpan)

(EvidenceSpan)
   ├─[:SUPPORTS]→(Fact)
   └─[:SUPPORTS]→(TaskMention)

(Fact)
   ├─[:HAS_SUBJECT]→(EntityMention)
   ├─[:HAS_PREDICATE]→(RelationExpression)
   └─[:HAS_OBJECT]→(EntityMention)

(Fact)
   └─[:RELATED_TO]→(TaskMention)

(TaskMention)
   └─[:MEMBER_OF]→(TaskFrame)

(TaskFrame)
   ├─[:HAS_CORE_FACT]→(Fact)
   ├─[:HAS_SUPPORTING_FACT]→(Fact)
   ├─[:HAS_CONTEXT_FACT]→(Fact)
   ├─[:SUBTASK_OF]→(TaskFrame)
   ├─[:PRECEDES]→(TaskFrame)
   ├─[:SUPPORTS]→(TaskFrame)
   └─[:MAPS_TO]→(StandardTask)

(EntityMention)
   └─[:CANONICALIZED_AS]→(CanonicalEntity)

(RelationExpression)
   └─[:CANONICALIZED_AS]→(CanonicalRelation)
```

---

# 16. 평가 방법

방법론의 각 단계를 분리하여 평가한다.

## 16.1 자유형 삼중항 추출 평가

- Subject 정확성
- Predicate 정확성
- Object 정확성
- Evidence grounding
- 의미 있는 Fact 재현율
- 중복 삼중항 비율
- hallucination 비율

---

## 16.2 TaskMention 추출 평가

- TaskMention precision
- TaskMention recall
- Task가 아닌 표현의 오탐률
- action 및 focal object 정확성
- Evidence 연결 정확성

---

## 16.3 Triple–TaskMention 연결 평가

- core 연결 정확성
- supporting 연결 정확성
- context 연결 정확성
- multi-task 연결 재현율
- Task와 무관한 Fact 보존율

---

## 16.4 TaskFrame 평가

- 동일 Task 병합 정확성
- 서로 다른 Task 분리 정확성
- 과병합률
- 과분할률
- singleton 비율
- canonical name 적절성
- definition 적절성
- TaskFrame 내 Fact 포괄률
- TaskFrame 내 관련 없는 Fact 비율

---

## 16.5 Canonicalization 평가

- 동일 엔터티 통합 정확성
- 서로 다른 엔터티 오병합률
- relation synonym 정확성
- 상하관계 정확성
- unresolved 비율
- 원문 표현 보존 여부

---

## 16.6 온톨로지 및 KG 평가

- domain/range 위반
- Evidence 없는 Fact 비율
- Evidence 없는 TaskFrame 비율
- 고립 노드 비율
- 중복 canonical node 비율
- 스키마 위반 수
- competency question 응답 가능률

---

## 16.7 최종 활용 평가

다음 Competency Question에 답할 수 있는지 평가한다.

1. 특정 프로세스에는 어떤 TaskFrame이 존재하는가?
2. 특정 TaskFrame과 관련된 모든 삼중항은 무엇인가?
3. 특정 업무를 수행하는 역할은 누구인가?
4. 특정 업무의 입력과 산출물은 무엇인가?
5. 특정 업무의 조건과 결과는 무엇인가?
6. 특정 의사결정 이전에 수행되는 업무는 무엇인가?
7. 특정 산출물은 어떤 업무에서 생성되고 어디에서 사용되는가?
8. 동일 업무가 문서에서 어떤 표현으로 나타나는가?
9. 특정 TaskFrame은 NCS·APQC·PCF의 어떤 업무와 대응되는가?
10. 각 답변을 뒷받침하는 원문 Evidence는 무엇인가?

---

# 17. 예상 위험과 대응 방안

## 17.1 TaskFrame이 지나치게 커지는 문제

### 위험

모든 Fact를 연결하면 TaskFrame이 섹션 전체와 유사해질 수 있다.

### 대응

다음 관련성 유형을 구분한다.

```text
core
supporting
context
unrelated
```

TaskFrame 검색 시 목적에 따라 core 또는 supporting Fact만 우선 사용한다.

---

## 17.2 Task 중심 추출로 일반 Fact가 누락되는 문제

### 위험

LLM 1이 Task에 집중하면서 Task와 직접 관련 없는 Fact를 누락할 수 있다.

### 대응

LLM 1 프롬프트에서 다음 순서를 강제한다.

```text
전체 Fact 추출
→ TaskMention 추출
→ Fact–TaskMention 연결
```

LLM 1에서는 Task와 무관한 Fact도 `task_links: []` 상태로 보존한다.
최종 단계에서는 주변 문맥을 이용하여 가장 관련 있는 TaskFrame에 `context`로 연결한다.

---

## 17.3 유사 표현을 동의어로 과도하게 병합하는 문제

### 위험

검토, 분석, 검증, 평가와 같은 표현이 단순 임베딩 유사도에 의해 하나로 합쳐질 수 있다.

### 대응

관계 표현만 단독 비교하지 않고 Subject, Object, TaskFrame, 원문 문맥을 함께 사용한다.

동의·상하·관련·미해결 상태를 구분한다.

---

## 17.4 TaskFrame 후보 과분할 문제

### 위험

Action–Target 차이를 지나치게 강조하면 거의 모든 TaskMention이 별도 Frame이 될 수 있다.

### 대응

Action–Target은 blocking에만 사용한다.

LLM 2에서는 목적, 프로세스 단계, 범위, 추상화 수준을 함께 고려한다.

---

## 17.5 과병합 문제

### 위험

표현이 유사한 서로 다른 업무가 하나로 합쳐질 수 있다.

### 대응

다음을 다르게 가지면 별도 Task로 분리한다.

- 목적
- 프로세스 단계
- 중심 대상
- 산출 결과
- 업무 범위
- 추상화 수준

---

## 17.6 표준체계 편향 문제

### 위험

NCS나 APQC를 먼저 제시하면 문서 기반 업무가 표준 업무에 맞춰 왜곡될 수 있다.

### 대응

TaskFrame을 문서 데이터만으로 먼저 생성한 후 표준체계와 독립적으로 매핑한다.

---

# 18. 최종 방법론 요약

본 연구는 수주·제안 문서에서 자유형 삼중항과 로컬 TaskMention을 공동 추출하고, 두 요소 사이의 연결을 추출 시점에 보존한다.

1차 LLM은 문서 청크에서 다음을 추출한다.

```text
자유형 삼중항
+ TaskMention
+ Triple–TaskMention 연결
```

이후 보수적 Entity 정규화와 알고리즘 기반 TaskMention 후보 클러스터링을 수행한다.

2차 LLM은 문서 전역의 후보군을 바탕으로 다음을 수행한다.

```text
TaskMention 병합·분리
+ TaskFrame 생성
+ Task 간 관계 판정
+ 애매한 Triple 연결 판정
```

최종적으로 다음 경로를 알고리즘으로 전파한다.

```text
Triple → TaskMention → TaskFrame
```

이를 통해 추가 LLM 호출 없이 각 TaskFrame을 해당 업무와 관련된 전체 지식 삼중항을 포괄하는 Task 중심 서브그래프로 구성한다.

엔터티와 관계 표현은 자유 추출 결과를 누적한 후 문맥 기반 클러스터링을 통해 정규화한다. 안정적으로 반복되는 개념과 관계만 도메인 온톨로지 스키마로 승격한다.

최종 접근은 다음과 같이 정리된다.

> **자유형 지식 추출의 정보 보존성, TaskFrame의 업무 중심 조직 능력, 최소 메타 스키마의 일관성, 데이터 기반 온톨로지 확장성을 결합한 Task-Centric Open KG 구축 방법론**
