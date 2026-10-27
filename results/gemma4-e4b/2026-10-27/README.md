# Gemma E4B 실험 결과 · 2026-10-27

## 오늘 마감 상태

- POPE adversarial: 저자 원본 함수를 Gemma에 연결한 RUDDER·GrAInS를 포함해 6조건 × 3,000문항 생성·공식 채점 완료.
- MMVP: 이전 Gemma 적용 구현 5방법 × 300문항·150쌍 평가 완료. 원본 clone으로 다시 평가한 결과는 아니다.
- CHAIR: COCO val2014 고정 500장 × 6조건·최대512토큰. 생성된 캡션 552/3,000개를 보존하고 사용자 요청으로 실험 프로세스를 종료해 GPU0 메모리까지 해제했다. 최종 점수는 아직 없다.
- POPE random/popular 및 MMHal 추가 생성·공식 채점은 보류.

[내일 이어서 할 일](HANDOFF_KO.md) · [정지 상태 요약](run-state/PAUSED_SUMMARY.json) · [재개 안내](run-state/RESUME_NOTES_KO.md)

## POPE adversarial — 최신 원본 clone 실행

각 조건 3,000문항이며 표의 단위는 %다. GrAInS는 Base FP32와, 나머지 방법은 Base BF16과 비교한다.

| 방법 | Accuracy ↑ | F1 ↑ | Precision ↑ | Recall ↑ |
|---|---:|---:|---:|---:|
| Base BF16 | 83.83 | 83.13 | 86.91 | 79.67 |
| REVIS 기존 | 83.93 | 83.22 | 87.10 | 79.67 |
| VTI 기존 | 83.73 | 83.13 | 86.35 | 80.13 |
| RUDDER 원본 clone | 83.80 | 83.09 | 86.90 | 79.60 |
| Base FP32 | 83.90 | 83.23 | 86.82 | 79.93 |
| GrAInS 원본 clone | 84.00 | 83.77 | 84.98 | 82.60 |

GrAInS는 FP32 Base 대비 Acc +0.10%p, F1 +0.54%p다. 답이 달라진 83문항 중 43개 개선·40개 악화로 정답이 3개 늘었다. Precision은 −1.84%p, Recall은 +2.67%p다.
이미지 단위 paired bootstrap 10,000회에서 차이의 95% 구간은 Acc −0.50~+0.70%p, F1 −0.08~+1.19%p로 두 구간 모두 0을 포함한다. 뚜렷한 개선이 입증됐다고 해석하지 않는다.

[최신 CSV](pope-author-clone/metrics.csv) · [공식 보고서·신뢰구간](pope-author-clone/report.json) · [문항별 비교](pope-author-clone/paired-comparison.json) · [실행 프로토콜](pope-author-clone/protocol.json)

[원본 응답 18,000개·공식 채점 자료 ZIP](pope-author-clone/pope-complete-delivery.zip)에는 여섯 조건의 JSONL을 원래 형식으로 보존했다. 전달된 모든 응답을 독립 재채점해 공식 보고서와 일치함을 확인했다.

### 구현과 검증

- RUDDER 저자 고정 커밋 `bd149d8dbde6a9097dd02aea80f83d0bd8876761`, GrAInS `8ad78db3040fbb7b0aec9e544a3dcc7b9c050b5b`의 원본 핵심 함수를 실제 호출했다.
- 모델은 `google/gemma-4-E4B-it`, revision `ee0ef6023621cff504d758262d4e04895a5af4a2`. Gemma 입력·PLE·출력 반환형 연결은 별도 adapter이므로 저자 checkpoint와 환경 전체를 재현한 결과는 아니다.
- GrAInS는 겹치지 않는 개발 이미지 50장, 원본 IG20/top50/PCA layer27·alpha10·FP32다. 벡터·원본 추출물은 독립 재검증했다.
- REVIS·VTI는 이전 서버의 Gemma 적용 구현·벡터를 유지했다. 해당 방법까지 저자 원본 clone으로 재실행했다는 의미는 아니다.
- greedy 생성, seed42, POPE 최대128토큰, CHAIR 최대512토큰. 테스트와 개발 이미지 해시 중복0개.
- 실행 소스의 해시와 호환성 증거는 [source-hashes.json](pope-author-clone/source-hashes.json)과 [proofs](pope-author-clone/proofs/)에 보관했다. 전체 protocol의 소스·입력·캐시 해시는 원본 그대로 유지했다. 실행 소스 코드 본체는 이번 결과 업로드에서 제외했다.

## MMVP — 이전 Gemma 적용 구현

| 방법 | 문항 정답 / 300 | 문항 정확도 | 쌍 정답 / 150 | 쌍 정확도 |
|---|---:|---:|---:|---:|
| base | 212 | 70.67% | 68 | 45.33% |
| revis | 214 | 71.33% | 70 | 46.67% |
| rudder | 213 | 71.00% | 69 | 46.00% |
| grains | 216 | 72.00% | 72 | 48.00% |
| vti | 214 | 71.33% | 70 | 46.67% |

쌍의 두 문항을 모두 맞혀야 해당 쌍이 정답이다. 원래 MMVP 정답을 주 지표로 쓰며 알려진 4문항 수정의 민감도 분석은 기존 상세 보고서에 보존했다.
[MMVP CSV](mmvp/metrics.csv) · [상세 보고서](mmvp/report.json) · [원본 응답 1,500개](mmvp/responses/)

## 이전 결과와 진단 보존

이전 로컬 RUDDER·GrAInS 구현의 POPE 결과는 [pope-adversarial](pope-adversarial/)에 그대로 보존했다. 새 원본 clone 점수와 섞지 않는다.
[이전 요약](legacy/README_BEFORE_AUTHOR_CLONE.md) · [이전 POPE 진단](DIAGNOSIS_KO.md) · [RUDDER 원본 clone 진단](pope-author-clone/diagnostics/WHY_RUDDER_CLONE_FLAT.md)

GitHub에는 완료된 결과·식별 정보를 뺀 정지 상태 요약과 [CHAIR 부분 응답 백업 ZIP](run-state/chair-paused-delivery.zip)을 보관한다. 벡터·원본 추출물은 실험 서버에 보존돼 있다. 개인 서버 경로·프로세스 식별 정보·정확한 재개 명령을 포함한 상세 정지 원본은 로컬에 별도로 보관했다.

파일 무결성은 [manifest.json](manifest.json)의 SHA-256으로 확인할 수 있다.

출처: [POPE](https://github.com/RUCAIBox/POPE), [MMVP](https://github.com/tsb0601/MMVP), [CHAIR](https://github.com/LisaAnne/Hallucination).
