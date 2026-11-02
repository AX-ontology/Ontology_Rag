# 자이제 1·2·3 통합 QA — 450문항

각 문서의 원문 기반 QA 150개씩을 합쳤다. 기존 질문·답변·근거·ID를 보존했으며,
일반 업무형 합성 60문항은 포함하지 않았다.

**[CSV 다운로드](qa_dataset.csv) · [JSONL 다운로드](qa_dataset.jsonl)**

사용 모델은 `gpt-5.6-terra`다. 생성 프롬프트와 실제 five-shot 5개는
[데이터셋 README](../README.md)에 정리했다.

450문항 모두 사람 검수 전 초안(`pending`)이며 그래프 Gold·평가 분할은 미확정이다.
근거 행은 `source_file` + `sheet_name` + `source_row_ids`로 찾는다.

[생성·통합 이력](manifest.json) · [중복·구성 검증](validation_report.json) ·
[Excel 원문 대조](source_validation_report.json) · [검수 메모](review_notes.json)

재통합: 저장소 루트에서 `python3 jaije_qa_dataset/build_final_dataset.py --output-dir /path/to/empty_output`을 실행한다.
