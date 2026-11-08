# Gemma E4B 중간 평가

POPE adversarial은 각 3,000문항 완료. CHAIR는 BF16 네 방법에 공통으로 생성 완료된 동일 500장으로 공식 채점했다. BF16 네 방법의 고정500장 결과다. FP32 두 방법까지 포함한 전체 실험의 최종 결과는 아니다.

| 방법 | POPE Acc ↑ | POPE F1 ↑ | CHAIR_s ↓ | CHAIR_i ↓ | 객체 recall ↑ |
|---|---:|---:|---:|---:|---:|
| base | 83.83 | 83.13 | 36.20 | 7.57 | 64.97 |
| revis | 83.93 | 83.22 | 26.80 | 6.19 | 59.74 |
| VTI (local adaptation) | 83.73 | 83.13 | 33.60 | 7.10 | 64.59 |
| rudder_clone | 83.80 | 83.09 | 35.20 | 7.32 | 65.03 |
| base_fp32 | 83.90 | 83.23 | 대기 | 대기 | 대기 |
| grains_clone | 84.00 | 83.77 | 대기 | 대기 | 대기 |

수치는 %. CHAIR_s는 환각 객체가 한 개 이상 있는 캡션 비율, CHAIR_i는 생성한 객체 언급 중 환각 비율이다. 둘 다 낮을수록 좋다.

GrAInS와 Base FP32의 소수 공통 이미지 점수는 별도 report.json에만 보관했다. BF16 표와 표본 수가 달라 같은 CHAIR 열로 비교하지 않았다.

VTI는 현재 선택한 독립 구현이다. 원본과 개입 위치·정규화·PCA·mask 처리·prefill 범위가 다르므로 원본 VTI 성능으로 인용하면 안 된다. REVIS도 선택한 서버 구현이며 RUDDER는 고정 원저자 함수와 Gemma 연결 adapter를 사용한다.

진행 중인 생성·저장 파일과 실행 설정은 변경하지 않았다. 원본 CHAIR Python3 포트·COCO annotations와 검증된 scorer state를 사용했다. 공통 cohort 선택에는 점수나 정답을 사용하지 않았다.
