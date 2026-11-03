# 검색 코드 설치 및 실행

## 구성

```text
Rag/
├── src/task_kg_rag/
│   ├── cli.py                 # 인덱스 생성·검색·선택적 답변 생성
│   ├── context.py             # 기존 과업 문맥 구성·보완 gate
│   ├── context_retrieval.py   # 기존 Dense40 경로 선택 및 CE 입력 구성
│   ├── legacy_aspects.py      # 기존 Aspect 검증·원 질문 fallback
│   ├── requirements.py        # 새 질문 span 기반 요구 검증
│   ├── extraction.py          # 질문만 사용하는 추출 및 1회 형식 수정
│   ├── spans.py               # 원문 span과 NFC 정규화
│   ├── hierarchy.py           # TH-KGR Global→계층→Local 및 후보 결합
│   └── backends.py            # 로컬 임베딩·CE·답변 서버 어댑터
├── assets/                    # 실험 KG의 실행용 축약 스냅샷·경로·설명
├── config/                    # 실제 추출 프롬프트·원본 코드 해시
├── scripts/                   # 그래프 내보내기·로컬 회귀 검증
└── tests/                     # 모델 없이 실행하는 동작 검증
```

기존 ContextBuilder, 경로 선택, Aspect 검증 모듈은 원본 그대로 옮겼습니다. `hierarchy.py`는 실제 v3 파일의 후보 생성 절차를 재사용 가능한 클래스로 옮겼습니다. `cli.py`와 `backends.py`는 이번 배포를 위한 경로·실행 어댑터입니다. 버전 대응은 `config/source_modules.json`에 기록합니다.

## 설치

저장소 루트에서 Python 3.11 이상을 사용합니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e './Rag[models]'
task-kg-rag --help
```

모델 없이 코드 검증만 하려면 `python -m pip install -e ./Rag`를 사용합니다. 의존성 범위는 설치 허용 범위이며 기존 실험 환경의 lockfile이 아닙니다.

GPU 사용은 기본적으로 꺼져 있습니다(`--device cpu`). GPU를 지정하려면 `CUDA_VISIBLE_DEVICES=0` 또는 `1`을 지정한 뒤 `--device cuda:0`을 사용합니다. 이 프로젝트의 공유 서버에서는 물리 GPU 0·1만 사용하며 2·3은 사용하지 않습니다. 명령이 서버를 자동 실행하거나 진행 중인 작업을 종료하지 않습니다.

## 필요한 모델

| 용도 | 모델 | 사용 방식 |
|---|---|---|
| Dense·경로·과업 설명 임베딩 | BAAI/bge-m3 | 로컬 체크포인트 경로 |
| CE | BAAI/bge-reranker-v2-m3 | 로컬 체크포인트 경로 |
| Aspect 추출 | Qwen3-8B | 사용자가 실행한 OpenAI 호환 API |
| 선택적 QA | Qwen3.5-27B | 사용자 실행 API와 로컬 tokenizer |

모델 다운로드·서버 실행·API 키는 포함하지 않습니다. 모델 로딩은 `local_files_only=True`입니다. 인증이 필요한 API는 환경변수 `RAG_API_KEY`를 사용하며 값을 코드에 기록하지 않습니다. GPU별 메모리 및 문맥 한도는 사용자가 실행한 서버에 따라 달라집니다.

## 1. 인덱스 생성

```bash
task-kg-rag build-index \
  --encoder /path/to/bge-m3 \
  --device cpu \
  --output Rag/artifacts/index_cpu
```

기본 입력은 `data/jaije_qa_dataset/corpus.json` 및 `Rag/assets/`입니다. 다른 자료는 `--corpus`, `--assets`로 지정합니다. 원문·과업 경로·과업 설명을 같은 인코더로 처리하고 ID 순서·해시를 기록합니다. 기존 1536토큰 설정을 명시적으로 사용하고 원문·경로의 잘림 수를 기록합니다. 질문의 임베딩 입력은 한도를 넘으면 실패합니다.

이 명령은 **새 인덱스**를 만듭니다. CPU/FP32 또는 다른 라이브러리 버전의 인덱스를 기존 GPU/FP16 실험의 완전히 동일한 벡터로 간주하지 않습니다. 검색 시 인덱스와 동일한 모델 경로·device를 요구합니다.

## 2. Task-KG Context CE 검색

아래 URL과 모델명은 예시이며 실제 실행 중인 서버에 맞춰 지정합니다.

```bash
task-kg-rag retrieve \
  --method task-context \
  --encoder /path/to/bge-m3 \
  --reranker /path/to/bge-reranker-v2-m3 \
  --index Rag/artifacts/index_cpu \
  --question 'Executive Summary는 누구를 위해 어떤 목적으로 작성하는 요약본인가요?' \
  --extractor-url http://127.0.0.1:8000/v1 \
  --extractor-model Qwen3-8B \
  --output Rag/outputs/task_context.json
```

원 질문 Dense40을 고정하고, 질문·Aspect와 정렬된 경로를 이용해 과업 문맥을 고릅니다. 원문 길이 ≤400자이며 100자 이하 또는 지정 지시 표현을 포함할 때만 문맥을 보완합니다. 문맥은 최대 500자이며 CE에는 원문 다음에 붙입니다. 512토큰 쌍 입력의 잘림 위치를 저장하고 상위 5개 원문 ID를 반환합니다.

기존 추출기의 최대 출력은 4개이며 단일 유효 요구는 원 질문으로 되돌립니다. 새 TH-KGR의 최대 3개 요구 추출기와 다릅니다. 원본 legacy 검증기는 query 전체가 질문의 부분 문자열인지까지 보장하지 않으므로 이를 강화된 의미 보존 검증으로 설명하지 않습니다.

## 3. TH-KGR 후보 검색

```bash
task-kg-rag retrieve \
  --method th-kgr \
  --encoder /path/to/bge-m3 \
  --index Rag/artifacts/index_cpu \
  --question 'Executive Summary는 누구를 위해 어떤 목적으로 작성하는 요약본인가요?' \
  --aspects Rag/examples/executive_summary_aspects.json \
  --output Rag/outputs/th_kgr_candidates.json
```

위 예제는 실제 저장된 질문-only 추출 문자열을 재사용합니다. 새 질문에는 `--aspects` 대신 `--extractor-url`·`--extractor-model`을 사용합니다. `--aspects` 파일에는 질문과 추출 문자열만 제공하며 참조 답변·Gold를 넣지 않습니다. 파일 재사용 시 질문 동일성과 출력 구조를 검사합니다.

Global Top2 과업 선택 → 타입이 명시된 계층 관계로 최대 4개 과업 → Local 원문 검색 → 원 질문 Top20 유지 및 round-robin 결합 → 40개 고유 원문 ID를 반환합니다. 중간 답을 다음 Aspect에 치환하는 기능은 구현하지 않았습니다.

`--reranker /path/to/bge-reranker-v2-m3`를 추가하면 원문 CE Top5까지 실행합니다. 이 신규 실행의 점수는 기존 메인 테이블에 자동 반영되지 않습니다. `--original-question`은 추출기를 쓰지 않는 명시적인 통제 조건이며 full Aspect 조건과 구분합니다.

## 4. 선택적 답변 생성

두 방법의 검색 명령에 다음을 추가하면 선택된 중심 원문만 답변기에 전달합니다.

```text
--reader-url http://127.0.0.1:8001/v1
--reader-model Qwen3.5-27B
--reader-tokenizer /path/to/Qwen3.5-27B
```

근거 본문 예산은 3,500토큰이며 실제 전달한 원문 ID를 별도로 기록합니다. 이 편의용 QA 어댑터의 프롬프트·근거 직렬화는 기존 고정 QA 평가 파이프라인과 동일하지 않습니다. 별도 검증 단계도 실행하지 않으므로, 이를 실행했다고 기존 메인 테이블이 재현된 것은 아닙니다.

## 출력과 시간

JSON에는 후보 ID, 최종 선택 ID(CE 실행 시), 문맥/계층 경로, 추출 결과, 단계별 소요 시간 및 답변(요청 시)이 저장됩니다. 기존 출력 파일은 덮어쓰지 않습니다.

`online_wall_seconds`는 모델 로딩 후 추출부터 요청한 마지막 단계까지의 벽시계 시간입니다. 후보 전용 실행은 답변 시간을 포함하지 않습니다. `verification_executed=false`이며 기존 표의 사용자 제공 Latency와 자동 대체하지 않습니다. API 실패나 잘린 생성 결과는 성공으로 처리하지 않습니다.

## 검증

```bash
python -m unittest discover -s Rag/tests -v
python data/jaije_qa_dataset/scripts/validate.py
```

원래 로컬 실험 캐시가 있는 경우 다음 회귀 검증도 가능합니다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
python Rag/scripts/verify_local_regression.py \
  --study-root /path/to/jaije_long_context_20261031_a100 \
  --tokenizer /path/to/bge-reranker-v2-m3 \
  --output /tmp/retrieval_regression.json
```

이번 배포에서는 기존 TH-KGR 개발 12문항·5조건의 **60개 후보 집합에서 ID와 순서가 모두 일치**했고, Task-KG Context CE의 단일/복수 Aspect 3문항·120개 후보에서 문맥과 선택 경로가 일치했습니다. 확인 기록은 [code_validation.json](../results/code_validation.json)에 있습니다. 이는 코드 정리 전후 비교이며, 신규 모델 추론이나 전체 450문항 성능 재실험을 뜻하지 않습니다.
