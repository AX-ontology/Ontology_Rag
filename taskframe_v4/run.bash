#!/usr/bin/env bash
# 오류 발생(-e), 미정의 변수 사용(-u), pipeline 중간 실패(-o pipefail) 시 즉시 종료한다.
set -euo pipefail

# 이 bash 파일이 위치한 절대 경로다. 어느 디렉터리에서 실행해도 파일을 찾게 해준다.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 처리할 원본 Excel 파일이다. 실행 시 INPUT_XLSX=/path/to/file.xlsx 로 변경할 수 있다.
INPUT_XLSX="${INPUT_XLSX:-$SCRIPT_DIR/../data/process_data/자이제_1.xlsx}"

# Excel에서 읽을 worksheet 이름이다.
SHEET="${SHEET:-clean_sentences}"

# llm: 실제 LLM 1·2를 호출한다. heuristic: 비용 없는 코드/KG smoke test용이다.
BACKEND="${BACKEND:-llm}"

# 처리할 Evidence 수다. 100은 앞의 100개만, 0은 전체 문서를 의미한다.
LIMIT="${LIMIT:-100}"

# LLM 2의 로컬 TaskFrame induction 한 호출에 넣을 최대 TaskMention 수다.
# 작게 하면 요청은 안전해지지만 호출 수가 늘고, 크게 하면 호출은 줄지만 prompt가 커진다.
LLM2_MAX_MENTIONS="${LLM2_MAX_MENTIONS:-60}"

# 애매한 Fact를 재판정할 때 현재 chunk의 앞·뒤로 각각 몇 개 chunk를 볼지 지정한다.
CONTEXT_CHUNK_RADIUS="${CONTEXT_CHUNK_RADIUS:-1}"

# LLM 1 한 호출에서 함께 추출할 chunk 수다.
# 크게 하면 호출 수는 줄지만 응답 JSON이 커질 수 있어 기본값 4를 사용한다.
EXTRACTION_BATCH_SIZE="${EXTRACTION_BATCH_SIZE:-4}"

# 단계별 JSONL/CSV, KG, 품질 보고서를 저장할 실행 결과 디렉터리다.
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/outputs/zaije_1_v4_5_1_limit${LIMIT}}"

# TASKFRAME_API_KEY, TASKFRAME_API_BASE, TASKFRAME_MODEL 등이 들어 있는 환경설정 파일이다.
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/../config/taskframe.env}"

ARGS=(
  # 실행할 Python 파이프라인 진입점이다.
  "$SCRIPT_DIR/run_pipeline.py"

  # Evidence를 생성할 원본 Excel 경로다.
  --input-xlsx "$INPUT_XLSX"

  # 원본 Excel에서 읽을 worksheet다.
  --sheet "$SHEET"

  # 실제 LLM 또는 heuristic smoke backend 중 하나를 선택한다.
  --backend "$BACKEND"

  # OpenAI-compatible API 설정을 읽을 .env 파일이다.
  --env-file "$ENV_FILE"

  # 모든 단계별 산출물을 기록할 경로다.
  --out-dir "$OUT_DIR"

  # 처리할 최대 Evidence 수다. 0이면 전체다.
  --limit "$LIMIT"

  # 계층형 LLM 2의 로컬 induction 호출당 최대 TaskMention 수다.
  --llm2-max-mentions "$LLM2_MAX_MENTIONS"

  # 애매한 Fact에 제공할 앞·뒤 chunk 문맥 반경이다.
  --context-chunk-radius "$CONTEXT_CHUNK_RADIUS"

  # LLM 1 호출 한 번에 묶을 chunk 수다.
  --extraction-batch-size "$EXTRACTION_BATCH_SIZE"
)

# RESUME=1이면 같은 v4 버전의 완료된 중간 산출물을 재사용한다.
# 코드 버전이 다르면 안전을 위해 pipeline이 resume을 거부한다.
if [[ "${RESUME:-0}" == "1" ]]; then
  ARGS+=(--resume)
fi

# 표준업무 JSON을 지정하면 TaskFrame 생성 후 NCS/APQC/PCF 등의 후보 매핑을 수행한다.
# 예: STANDARDS_JSON=examples/standard_tasks.sample.json bash run.bash
if [[ -n "${STANDARDS_JSON:-}" ]]; then
  ARGS+=(--standards-json "$STANDARDS_JSON")
fi

# 위에서 구성한 인자를 그대로 보존하여 Python 파이프라인을 실행한다.
python3 "${ARGS[@]}"

# GENERATE_HTML=1(기본)이면 KG 실행 결과를 사람이 검토할 수 있는 독립형 HTML로 만든다.
# HTML에는 방법론, 전체 삼중항, TaskFrame 내부 Fact와 Evidence, 품질 검토 큐가 포함된다.
if [[ "${GENERATE_HTML:-1}" == "1" ]]; then
  python3 "$SCRIPT_DIR/build_html_report.py" "$OUT_DIR"
fi
