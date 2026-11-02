# 실험 설정 및 완료 결과

## 공통 조건

- 데이터: 『자주 이기는 제안서』 기반 원문 3,995행, 질문 450개.
- 답변기: Qwen3.5-27B BF16, tensor parallel 2, 최대 문맥 262,144토큰.
- 생성: seed 42, temperature 0, top-p 1. 최초 답변 최대 1,536토큰, 해당 비수정 검증 최대 1,024토큰.
- 검색 방법: 상위 5개 중심 원문, 근거 본문 예산 3,500토큰.
- FullDoc: 동일 답변기에 전체 3,995행을 입력. Top5 검색을 하지 않으므로 Recall@5는 적용하지 않음.
- LLM-only: 문서 없는 최초 답변. LLM + self-check: 동일 최초 답변을 문서 없이 점검·수정한 답변.
- 이 표는 이전 Qwen3-8B 답변기 결과와 다른 실행임.

## 메인 테이블

| Type | Method | Recall@5 | ROUGE-L | BLEU | BERTScore | Latency (s/query) |
|---|---|---|---|---|---|---|
| LLM-only | LLM-only | - | 0.2567 | 8.26 | 0.8965 | 0.015365 |
| LLM-only | LLM + self-check | - | 0.2616 | 8.37 | 0.8972 | 0.032970 |
| LLM w/full doc | FullDoc | - | 0.4142 | 16.02 | 0.9138 | 0.269186 |
| Text-only RAG | Dense | 52.06 | 0.4004 | 18.34 | 0.9146 | 0.029507 |
| Text-only RAG | BM25 | 52.96 | 0.4112 | 19.27 | 0.9122 | 0.028927 |
| KG-RAG | LightRAG (shared KG) | 39.87 | 0.3452 | 12.94 | 0.8988 | 0.035733 |
| KG-RAG | 1-hop KG-RAG | 38.28 | 0.3562 | 13.78 | 0.9038 | 0.031406 |
| KG-RAG | FlatKG | 38.28 | 0.3562 | 13.78 | 0.9038 | 0.031407 |
| Task-based KG-RAG | Task-KG Context CE | 71.96 | 0.4307 | 19.18 | 0.9167 | 0.034822 |

`-`는 적용 불가입니다. ROUGE-L·BERTScore는 0–1, Recall@5·BLEU는 0–100 척도입니다. BERTScore는 XLM-R-large layer 17, IDF 및 baseline rescaling 없이 계산했습니다. 512토큰 한도를 넘는 두 텍스트는 평가기에서 잘렸으며 다른 평가용 원문은 보존했습니다.

Task-KG Context CE는 표에 포함된 방법 중 Recall@5·ROUGE-L·BERTScore 최고값, BM25는 BLEU 최고값입니다. 전체 데이터는 개발 150문항과 이미 확인한 test 300문항을 포함합니다. TH-KGR의 점수는 이 표에 없습니다.

## 시간 값의 출처

Latency 열은 사용자가 제공한 쿼리당 ms 값을 1,000으로 나눈 s/query 값입니다. 예: 15.365ms → 0.015365s. `timing_provenance.json`에 입력 단위와 값을 보존했습니다. `ours` 입력은 현재 강조 행인 Task-KG Context CE에 대응합니다.

이 값은 이 문서 정리 과정에서 실행 로그로 독립 검증한 end-to-end 측정값이 아닙니다. 기존 QA 요청시간 합계와도 다른 자료입니다. 절대 시간 차이가 작다는 해석과 모든 단계의 overhead가 없다는 주장은 구분해야 합니다. 이전 QA 시간이나 새 TH-KGR 실행시간으로 대체하지 않습니다.

## 재현 범위

점수 CSV와 LaTeX 표는 로컬 최종 표의 스냅샷입니다. 미완료 RAGAS 결과는 포함하지 않습니다. FlatKG와 1-hop KG-RAG는 보관된 결과에서 동일한 점수를 보이며 독립적인 성능 향상 사례로 해석하지 않습니다.
