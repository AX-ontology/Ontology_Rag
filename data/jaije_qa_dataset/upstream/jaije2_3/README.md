# 자이제 2·3 원문 기반 QA

`gpt-5.6-terra`로 각각 150문항을 생성했다.
자이제 2는 `종학 파트`, 자이제 3은 `clean_sentences`의 `사람 검수` 열을 사용했다.
빈 검수 셀은 제외했다.

| 데이터셋 | CSV | JSONL |
|---|---|---|
| 자이제 2 · 150문항 | [CSV](outputs/jaije2_terra_v1/qa_dataset.csv) | [JSONL](outputs/jaije2_terra_v1/qa_dataset.jsonl) |
| 자이제 3 · 150문항 | [CSV](outputs/jaije3_terra_v1/qa_dataset.csv) | [JSONL](outputs/jaije3_terra_v1/qa_dataset.jsonl) |

**[자이제 1·2·3 통합 450문항](../final_450/README.md)** ·
[생성 프롬프트·five-shot 요약](../README.md) ·
[시스템 프롬프트 원본](prompts/system_prompt.txt) ·
[five-shot 원본](prompts/few_shot_examples.json)

생성 코드와 프롬프트는 실행 당시 내용 그대로 보존했다.
원문 Excel·API 키·내부 요청 로그는 포함하지 않았다.
모든 문항은 사람 검수 전 초안이며, 검증 통계와 검수 메모는 각 출력 폴더에 있다.

재생성할 때는 `requirements.txt`를 설치하고 원문 Excel 3개를 부모 `jaije_qa_dataset/`에 준비한다.
`generate_document_qa.py --help`에서 입력·시트·출력 경로를 지정하고,
먼저 `--dry-run`으로 점검한다. 실제 생성에는 `OPENAI_API_KEY`가 필요하다.
`validate_document_outputs.py`는 원문 Excel과 전체 실행 로그를 필요로 하는 보존용 검증기다.
