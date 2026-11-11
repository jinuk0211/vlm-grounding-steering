# Gemma E4B 전체 평가 — GrAInS FP16

POPE adversarial 각3,000문항, CHAIR 동일 고정500장. GrAInS는 새 FP16 응답이고 다른 네 방법은 기존 BF16 응답을 검증해 보존했다. Base FP32는 제외했다.

| 방법 | 정밀도 | POPE Acc ↑ | POPE F1 ↑ | CHAIR_s ↓ | CHAIR_i ↓ | 객체 recall ↑ |
|---|---|---:|---:|---:|---:|---:|
| base | bfloat16 | 83.83 | 83.13 | 36.20 | 7.57 | 64.97 |
| revis | bfloat16 | 83.93 | 83.22 | 26.80 | 6.19 | 59.74 |
| VTI 자체 구현 | bfloat16 | 83.73 | 83.13 | 33.60 | 7.10 | 64.59 |
| rudder_clone | bfloat16 | 83.80 | 83.09 | 35.20 | 7.32 | 65.03 |
| grains_clone | float16 | 84.00 | 83.76 | 27.40 | 6.17 | 60.74 |

수치는 %. GrAInS와 Base는 FP16/BF16 정밀도가 달라 방법 효과만 분리한 비교가 아니다.
GrAInS의 기존 원본 IG20·fit50·PCA27 벡터는 FP32로 추출했고 모델은 FP16으로 추론하며 steering 벡터의 원본 FP32 dtype를 유지했다. 추론 dtype 변경에 따른 새 POPE·CHAIR 응답만 사용했다.
RUDDER·GrAInS는 고정 원저자 함수와 Gemma 연결 adapter를 사용한다. REVIS·VTI는 선택한 자체 구현이며 원저자 전체 환경 재현을 뜻하지 않는다.
배치1·greedy·문항별 seed42·이미지448·POPE128/CHAIR512토큰, 공식 scorer 및10,000회 이미지 단위 paired bootstrap을 유지했다.
기존 FP32 결과와 원본 raw 파일은 역사 기록으로 보존하고 새 점수 집계에서 제외했다.
