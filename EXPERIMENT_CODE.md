# 실험 코드와 결과 모음

2026-10-29에 확보한 보고서 기준. 점수는 모두 %이며, ↓는 낮을수록 좋다. Qwen 전체 5개 방법의 최종 검증은 당시 진행 중이었다.

## 원저자 코드가 연결되는 곳

| 방법 | 실제 연결 파일 | 연결 내용 |
|---|---|---|
| RUDDER | [author_clone_bridge.py](scripts/author_clone_bridge.py#L40) | `load_authors()`가 원저자 `RUDDER/Qwen/qwen_methods.py`와 POPE mask 함수를 불러오고, `rudder_context()`가 모델에 연결 |
| GrAInS | [author_clone_bridge.py](scripts/author_clone_bridge.py#L40), [qwen_author_compat.py](scripts/qwen_author_compat.py) | 원저자 gradient, steering, token ablation을 불러와 Gemma 및 Qwen 입력·레이어에 연결 |
| VTI — Qwen | [run_qwen_original_vti.py](scripts/run_qwen_original_vti.py#L20) | 원저자 `vti_utils.utils`, `llm_layers`를 직접 import하고 Qwen의 입력과 텐서 형태를 연결 |
| VTI — Gemma 별도 실험 | [vti_author.py](scripts/vti_author.py), [vti_author_gemma.py](scripts/vti_author_gemma.py), [evaluate_vti_author_pope.py](scripts/evaluate_vti_author_pope.py) | 원저자 함수를 Gemma에 연결해 별도의 POPE 실험 실행 |
| REVIS — 5개 방법 비교 | [baselines.py](src/vlm_grounding/baselines.py) | 두 모델의 5개 방법 비교에서는 자체 구현을 사용 |
| REVIS/VTI — 별도 Gemma 연결 | [author_revis_vti_bridge.py](scripts/author_revis_vti_bridge.py) | 고정 원저자 함수에서 필요한 정의를 불러 Gemma에 연결하는 별도 브리지. 위 5개 방법 비교와 혼동하지 말 것 |

원저자 저장소는 이 브랜치에 복사되어 있지 않다. 실행 스크립트는 서버의 `upstream-author-20261026` 아래 원저자 checkout을 요구한다. 고정 버전: [RUDDER](https://github.com/Akko000/RUDDER-Residual-Update-Directed-DEcoding-Regulation-/tree/bd149d8dbde6a9097dd02aea80f83d0bd8876761), [GrAInS](https://github.com/duykhuongnguyen/GrAInS/tree/8ad78db3040fbb7b0aec9e544a3dcc7b9c050b5b), [VTI](https://github.com/shengliu66/VTI/tree/bb5dfee33a9e555f22af9faa7b3495c0ac3df69e), [REVIS](https://github.com/antgroup/Revis/tree/75f4d29d402d4c1c4dbb2c69e88e3aa50cd0c9cc).

## Gemma 4 E4B — 5개 방법 전체 평가

POPE adversarial 방법당 3,000문항, CHAIR 동일 500장. Greedy, batch 1, seed 42, 이미지 448px, POPE 최대 128·CHAIR 최대 512 토큰. 공식 채점 및 이미지 단위 paired bootstrap 10,000회. REVIS·VTI는 자체 구현, RUDDER·GrAInS는 원저자 함수 연결이다.

| 방법 | 정밀도 | POPE Acc ↑ | POPE F1 ↑ | CHAIR_s ↓ | CHAIR_i ↓ | 객체 recall ↑ |
|---|---|---:|---:|---:|---:|---:|
| Base | BF16 | 83.83 | 83.13 | 36.20 | 7.57 | 64.97 |
| REVIS | BF16 | 83.93 | 83.22 | 26.80 | 6.19 | 59.74 |
| VTI 자체 구현 | BF16 | 83.73 | 83.13 | 33.60 | 7.10 | 64.59 |
| RUDDER 원저자 함수 연결 | BF16 | 83.80 | 83.09 | 35.20 | 7.32 | 65.03 |
| GrAInS 원저자 함수 연결 | FP16 | 84.00 | 83.76 | 27.40 | 6.17 | 60.74 |

GrAInS 방향은 FP32로 추출하고 추론은 FP16으로 실행했다. BF16 Base와 정밀도가 달라 차이를 방법 효과만으로 해석할 수 없다. 실행 코드: [run_gemma_grains_fp16_v2.py](scripts/run_gemma_grains_fp16_v2.py), [grains_fp16_worker_v2.py](scripts/grains_fp16_worker_v2.py), [continue_gpu23_suite.py](scripts/continue_gpu23_suite.py).

## Gemma E4B — 원저자 VTI 별도 POPE 및 강도 실험

위의 5개 방법 표와 별도의 POPE adversarial 3,000문항 비교다. 70 demos, 50 trials, 99% patch mask, 방향 추출 시 visual alpha 0.9; 아래 후속 실험은 같은 벡터를 사용하고 추론 강도만 변경했다. BF16, greedy, 448px, seed 42.

| 방법 | Acc ↑ | F1 ↑ | Precision ↑ | Recall ↑ |
|---|---:|---:|---:|---:|
| Base | 83.8333 | 83.1304 | 86.9091 | 79.6667 |
| 자체 VTI | 83.7333 | 83.1259 | 86.3506 | 80.1333 |
| 원저자 VTI alpha 0.9 | 82.4000 | 82.3411 | 82.6174 | 82.0667 |
| 원저자 VTI alpha 0.45 | 83.6333 | 83.0397 | 86.1649 | 80.1333 |
| 원저자 VTI alpha 0.225 | 83.5000 | 82.9369 | 85.8672 | 80.2000 |

0.45와 0.225는 결과를 본 뒤 요청한 탐색적 후속 실험이다. 초기 원저자 VTI 보고의 F1 66.6667은 답변 앞뒤 공백을 제거하지 않아 생긴 파싱 오류다. 위 표는 모든 방법에 동일한 `strip()`을 적용한 정정 점수다. 원문 응답은 변경하지 않았고 재추론은 하지 않았다. 코드: [score_vti_trimmed.py](scripts/score_vti_trimmed.py), [run_vti_alpha045.py](scripts/run_vti_alpha045.py), [run_vti_alpha0225.py](scripts/run_vti_alpha0225.py).

## Qwen2.5-VL-7B — 확인된 부분 결과

Base·REVIS·RUDDER는 POPE 각 3,000문항 및 CHAIR 각 500장 공식 채점 완료. GrAInS POPE는 3,000문항 완료. 아래 빈칸은 이 문서 작성 시점에 확보한 최종 채점 보고가 없다는 뜻이며, 전체 5개 방법 결과가 아니다.

| 방법 | 정밀도 | POPE Acc ↑ | POPE F1 ↑ | CHAIR_s ↓ | CHAIR_i ↓ | 객체 recall ↑ |
|---|---|---:|---:|---:|---:|---:|
| Base | BF16 | 85.60 | 83.82 | 25.60 | 8.75 | 58.92 |
| REVIS 자체 구현 | BF16 | 85.60 | 83.82 | 26.00 | 8.87 | 59.42 |
| RUDDER 원저자 함수 연결 | BF16 | 85.57 | 83.79 | 23.20 | 7.57 | 60.30 |
| GrAInS 원저자 함수 연결 | FP16 | 85.43 | 83.74 | 미확인 | 미확인 | 미확인 |
| VTI 원저자 함수 연결 | FP16 | 진행 중 | 진행 중 | 진행 중 | 진행 중 | 진행 중 |

GrAInS와 원저자 VTI는 FP16 가중치와 FP32 attention 점수를 사용한다. Base는 BF16이므로 정밀도 차이가 비교에 영향을 줄 수 있다. 과거 자체 VTI 결과는 원저자 VTI 최종 비교에서 제외된다. 실행 코드: [run_qwen_five_profile_suite.py](scripts/run_qwen_five_profile_suite.py), [run_qwen_five_profile_worker.py](scripts/run_qwen_five_profile_worker.py), [run_qwen_original_vti.py](scripts/run_qwen_original_vti.py).

## 범위와 출처

점수 출처는 작업 폴더의 `HALLUCINATION_RESULTS/GEMMA_FIVE_PROFILE_20261029/COMPLETE/REPORT_KO.md`, `VTI_AUTHOR_POPE_20261029/COMPLETE/trimmed/REPORT_KO.md`, `VTI_ALPHA045_20261029/COMPLETE/REPORT_KO.md`, `VTI_ALPHA0225_20261029/COMPLETE/REPORT_KO.md`, `QWEN_FIVE_PROFILE_20261029/BF16_THREE_COMPLETE/REPORT_KO.md`, `QWEN_FIVE_PROFILE_20261029/GRAINS_POPE_COMPLETE/REPORT_KO.md`이다.

모델 가중치, 데이터, 원본 결과 및 일부 채점·운영 도구는 이 브랜치에 없다. 코드는 서버 경로, GPU UUID, 기존 프로토콜 및 소스 해시를 전제로 하므로 그대로 다른 컴퓨터에서 실행되는 패키지는 아니다. 이 문서는 확인된 주요 평가와 추가 VTI 강도 실험을 모은 스냅샷이며, 진행 중인 실험의 상태는 이후 달라질 수 있다.
