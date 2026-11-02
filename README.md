# Ontology_Rag

TaskFrame 기반 Ontology 파이프라인의 독립 소스 코드 패키지.

## GitHub 배포 범위

이 저장소에는 실행 코드, 설정 예시, 테스트, HTML 템플릿과 글꼴 라이선스만 포함한다.
아래의 데이터/결과 구조는 **로컬 패키지 기준**이며, 원문 Excel·PCF 데이터·승인 결과·
그래프 결과·생성 HTML·API 키·DB 비밀번호·모델 가중치는 Git에 포함하지 않는다.

따라서 clone만으로 기존 자이제 결과를 재현하거나 바로 실행할 수는 없다.
권한 있는 로컬 자료를 준비하고 `final_pipeline/config.yaml`의 입력·기준 업무·재사용 경로를
맞춘 뒤 실행한다. 원래 로컬 패키지에는 해당 자료가 그대로 보존돼 있다.
`verify_package.py`도 입력 자료가 필요하며, 로컬 배포용 해시 목록인
`package_manifest.json`을 복원하지 않은 경우에는 `--skip-hashes`를 사용한다.

## 로컬 독립 실행 패키지

이 폴더에는 TaskFrame 추출·기존 업무 직접 배정·PCF 연결·정제·Neo4j 적재 및
방법론/결과 HTML 생성에 필요한 프로젝트 코드를 모았다.
원래 `AX/final_pipeline`은 다른 프로젝트 폴더에 의존했지만,
이 복사본은 실행 경로가 모두 **Ontology 폴더 내부**를 가리킨다.
원래 AX 폴더의 코드·설정·결과를 읽거나 수정할 필요가 없다.

## 구성

```text
Ontology/
├── run.bash                      # 정식 실행 진입점
├── verify_package.py             # 로컬 의존성·승인 결과·해시 검증 (API/DB 호출 없음)
├── requirements.txt              # 전체 Python 의존성
├── config/                       # 비밀값 없는 환경설정 예시
├── final_pipeline/               # 기본 2단계 + 조건부 신규 업무 생성
├── taskframe_v4/                 # 원문 추출·Fact 연결·그래프 생성 라이브러리
├── layer_connect/                # BGE 임베딩·reranker 기반 PCF 연결
├── cleaning/                     # 정제·검증·선택적 Neo4j 적재
├── data/                         # 자이제 1·2·3 Excel, PCF 기준 데이터
└── final_outputs/                # 승인된 재사용 자료 및 HTML용 기존 결과 스냅샷
```

기존 내부 디렉터리 이름은 import 호환성을 위해 유지했다.
`final_pipeline` 폴더만 떼어내면 안 되며 **Ontology 폴더 전체**를 함께 이동한다.

## 준비와 실행

이 README가 있는 Ontology 폴더에서 실행한다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# 입력 파일·승인 자료·폴더 내부 경로 검증
python verify_package.py

# 계획만 출력: API/DB 호출 없음
bash run.bash --dry-run

# 전체 파일 파이프라인 실행 (DB 쓰기 없음)
bash run.bash

# 실제 DB에 적재할 때만 명시
bash run.bash --load-neo4j

# 포함된 결과 스냅샷으로 방법론·시각화 재생성
python final_pipeline/build_methodology_report.py
```

새 문서의 LLM 추출·배정에는 `config/taskframe.env.example`을 `config/taskframe.env`로
복사한 뒤 본인의 키를 입력해야 한다. DB 적재에는 `config/neo4j.env.example`을
`config/neo4j.env`로 복사하고 접속 정보를 입력한다. 기존 API 키와 DB 비밀번호는
보안상 가져오지 않았다. 실행 환경변수로 설정할 수도 있다.

**코드가 독립적이라는 것은 외부 서비스가 필요 없다는 뜻은 아니다.**
Python과 위 패키지, BGE 모델 가중치가 필요하다. 모델 가중치/가상환경은 포함하지 않았다.
PCF 작업자는 현재 실행 중인 Python을 사용하며 특정 conda 절대 경로에 의존하지 않는다.
최초 모델 다운로드에는 네트워크가 필요할 수 있다.
새 문서의 생성형 LLM은 DeepSeek, 선택적인 DB 적재는 Neo4j 서비스가 필요하다.

## 기존 결과 및 새 문서

- 기존 자이제1·2·3의 승인된 추출/배정 자료를 실제 파일로 복사했다. 원본을 향하는 심볼릭 링크가 아니다.
- 설정은 승인된 결과 재사용을 유지하므로 기존 문서에 대한 불필요한 LLM 재호출을 피한다.
- 기본 실행 결과는 이 폴더의 `final_outputs/v4_5_catalog_final`에 새로 생성된다.
- 신규 문서는 `final_pipeline/config.yaml`의 `documents`에 추가한다. 사용법은
  `final_pipeline/README.md`를 참고한다. 새 문서에는 `reuse_taskkg`/`approved_alignment`를 넣지 않는다.
- HTML의 12개 TaskFrame 결과는 이미 승인된 과거 결과이며, 새 직접 배정 방식의 재추출 결과가 아니다.
- HTML용 `catalog_refactor_check`는 조회에 필요한 최소 스냅샷이다. 원래 실행 전체의
  재개용 artifact/cache를 복사한 것은 아니다. 현재 실행 재개는 새 정식 출력에서 한다.
- 과거 JSON의 `source_file`, `inputs`, `source` 같은 출처 기록에는 이전 절대 경로가 남을 수 있다.
  이는 감사용 문자열이며 실행 입력 경로가 아니다. 해시 검증을 위해 원문·승인 자료를 변조하지 않았다.
- API 캐시, 실행 로그, 가상환경, 모델 다운로드 캐시는 복사하지 않았다.
- 로컬 문서와 원문 근거가 포함돼 있으므로 이 폴더를 공개 저장소에 그대로 올리지 않는다.

## 검증

```bash
python -m unittest discover -s final_pipeline/tests -v
(cd taskframe_v4 && python -m unittest discover -s tests -v)
python -m unittest discover -s cleaning/tests -v
python -m pytest layer_connect/tests -q
```

`verify_package.py`는 설정 경로의 패키지 내부 포함 여부, 원문 SHA-256,
승인 묶음의 digest·업무 ID, 원본 복사 자료의 파일별 SHA-256을 확인한다.
런타임 패키지 설치 여부는 별도로 표시한다. 실제 LLM/PCF 모델 추론·DB 접속은 수행하지 않는다.
이 패키지 생성 과정에서도 유료 API나 DB 쓰기를 실행하지 않았다.
