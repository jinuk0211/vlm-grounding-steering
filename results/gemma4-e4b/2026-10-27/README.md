# Gemma E4B 실험 결과 · 2026-10-27

**참고용 결과다. REVIS·RUDDER·GrAInS·VTI는 이 저장소의 Gemma 적용 구현이며, 논문 저자 원본의 성능 재현 결과가 아니다.**

| 평가 | 상태 | 방법당 평가량 |
|---|---|---:|
| POPE COCO adversarial | 완료·독립 검증 | 3,000문항 / 500장 |
| MMVP | 완료·독립 검증 | 300문항 / 150쌍 |
| CHAIR | 사용자 요청으로 일시 정지·미완료 | 고정 COCO val2014 500장 계획 |
| MMHal-Bench | 공식 채점 보류 | 추가 생성도 현재 보류 |

CHAIR와 MMHal의 최종 점수는 아직 없다. 이 업로드는 실험을 재개하지 않는다.

## POPE adversarial

| 방법 | 정확도 ↑ | F1 ↑ | Precision ↑ | Recall ↑ |
|---|---:|---:|---:|---:|
| base | 83.83% | 83.13% | 86.91% | 79.67% |
| revis | 83.93% | 83.22% | 87.10% | 79.67% |
| rudder | 83.77% | 83.06% | 86.84% | 79.60% |
| grains | 82.83% | 81.21% | 89.69% | 74.20% |
| vti | 83.73% | 83.13% | 86.35% | 80.13% |

REVIS의 정확도 차이는 Base 대비 +0.10%p이며 이미지 단위 paired bootstrap 10,000회의 95% 구간은 -0.20~0.43%p다. 전반적인 성능 개선을 확인했다고 해석하지 않는다.
GrAInS는 이 설정의 POPE에서 Base보다 정확도와 F1이 낮았다. 응답 형식 오류는 다섯 방법 모두 0개다.

[점수 CSV](pope-adversarial/metrics.csv) · [상세 보고서와 불확실성](pope-adversarial/report.json) · [프로토콜](pope-adversarial/protocol.json)

## MMVP

| 방법 | 문항 정확도 ↑ | 쌍 정확도 ↑ |
|---|---:|---:|
| base | 212/300 (70.67%) | 68/150 (45.33%) |
| revis | 214/300 (71.33%) | 70/150 (46.67%) |
| rudder | 213/300 (71.00%) | 69/150 (46.00%) |
| grains | 216/300 (72.00%) | 72/150 (48.00%) |
| vti | 214/300 (71.33%) | 70/150 (46.67%) |

쌍의 두 문항을 모두 맞혀야 해당 쌍을 정답으로 센다. 원래 MMVP 정답을 주 지표로 사용했고, 알려진 4문항의 정답 수정에 대한 민감도 분석은 CSV와 JSON에 별도 보관했다. 이 결과만으로 방법의 우월성을 확정할 수 없다.

[점수 CSV](mmvp/metrics.csv) · [상세 보고서](mmvp/report.json) · [문항별 원본 응답](mmvp/responses/) · [프로토콜](mmvp/protocol.json)

## 실행 조건과 구현 제한

- 모델: `google/gemma-4-E4B-it`, revision `ee0ef6023621cff504d758262d4e04895a5af4a2`, BF16, 이미지 크기 448, greedy 생성, seed 42.
- COCO train2017의 별도 개발 이미지 32장으로 벡터를 추출하고 REVIS는 추가 32장으로 보정했다. 테스트 이미지와 개발 이미지의 정확한 해시 중복은 0개다.
- 방법별 강도는 고정된 초기 설정이다. Gemma용 최적 하이퍼파라미터를 찾았다는 의미가 아니다. 테스트 정답으로 벡터·보정·강도를 선택하지 않았다.
- REVIS의 추출·보정·prefill 범위, VTI의 PCA·주입 방식 등은 저자 코드와 다르다. [구현 차이와 POPE 답변 변화](DIAGNOSIS_KO.md)를 함께 참고한다.
- GrAInS IG는 BF16의 비유한 gradient를 막기 위해 Gauss–Legendre 16단계를 사용했다. 실행 소스·입력·벡터 해시와 검증 내역은 상세 보고서에 보관했다.
- POPE는 공식 evaluate.py 결과와 독립 재채점이 일치했다. MMVP는 전체 응답을 독립 재채점했다.

## 저장 범위와 원본 위치

이 폴더에는 완료된 두 평가의 CSV·원본 JSON 보고서·설정·프로토콜, MMVP의 문항별 응답 1,500개를 보관한다. POPE의 문항별 응답 15,000개와 CHAIR의 중간 생성물은 서버 원본에 보존돼 있다.

- POPE: `/home/project-b/LJW/vlm-grounding-steering/outputs/pope-selected-20261027-161933/merged/`
- MMVP: `/home/project-b/LJW/vlm-grounding-steering/outputs/mmvp-full-igfix-20261027-152217/`
- CHAIR 중간 출력: `/home/project-b/LJW/vlm-grounding-steering/outputs/chair-selected-20261027-163104/`

원본 보고서의 실행 완료 상태는 각 벤치마크 범위에만 해당한다. 전체 벤치마크 계획의 완료를 뜻하지 않는다. 현재 GPU0 한 장에서 실행하던 CHAIR와 자동 실행은 사용자 요청으로 일시 정지 상태다. 이전 실행의 하드웨어 정보는 각 원본 보고서에 남아 있다.

파일 무결성은 [manifest.json](manifest.json)의 SHA-256으로 확인할 수 있다.

출처: [POPE](https://github.com/RUCAIBox/POPE), [MMVP](https://github.com/tsb0601/MMVP), [CHAIR](https://github.com/LisaAnne/Hallucination).
