# Qwen2.5-VL-7B · POPE adversarial / CHAIR

각 방법 POPE 3,000문항·CHAIR 500장, 전체 17,500응답. 수치는 %.

| 방법 | 모델 정밀도 | POPE Acc ↑ | POPE F1 ↑ | CHAIR_s ↓ | CHAIR_i ↓ | 객체 recall ↑ |
|---|---|---:|---:|---:|---:|---:|
| Base | BF16 | 85.60 | 83.82 | 25.60 | 8.75 | 58.92 |
| REVIS 자체 구현 | BF16 | 85.60 | 83.82 | 26.00 | 8.87 | 59.42 |
| RUDDER 원본 함수 연결 | BF16 | 85.57 | 83.79 | 23.20 | 7.57 | 60.30 |
| GrAInS 원본 함수 연결 | FP16 | 85.43 | 83.74 | 29.40 | 8.45 | 68.43 |
| VTI 원저자 구현 연결 | FP16 | 82.00 | 79.25 | 6.20 | 4.51 | 29.43 |

VTI 응답 생성 GPU: POPE는 GPU0 2851·GPU1 149문항, CHAIR는 GPU0 133·GPU1 258·GPU2 55·GPU3 54장이다. GPU 종류: GPU0 NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, GPU1 NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, GPU2 NVIDIA RTX A5000, GPU3 NVIDIA RTX A5000. 가중치·수식·데이터·벡터·정밀도는 동일하지만 FP16 수치는 GPU 종류에 따라 미세하게 다를 수 있다. 각 row에 gpu_uuid를 기록했다.

원래 작업은 사용자 요청으로 중지되었고(user-stop-proof-20261029.json), 중지 시점의 출력은 바이트 단위로 보존해 검증했다. 남은 POPE 149문항과 CHAIR 258장은 GPU1에서 겹치지 않게 이어 생성했다.

VTI 자체 구현의 이전 점수는 제외했다. VTI는 고정 저자 commit의 PCA·방향 생성·MLP 개입·정규화 함수를 직접 호출하고 Qwen2.5 입출력과 ViT 배치 차원을 연결했다. 저자 데모 70개·seed42·마스크99%·각50회·rank1·시각/텍스트 강도0.9를 사용했다. 저자 저장소는 Qwen2.5 POPE/CHAIR 실행 경로를 제공하지 않으므로 논문 전체 환경 재현으로 표현하지 않는다.

GrAInS·VTI 모델 가중치와 attention 출력은 FP16, QK·mask·softmax 점수는 overflow 방지를 위해 FP32다. 다른 세 방법은 BF16이므로 방법 비교에는 정밀도 영향도 포함된다. REVIS는 자체 구현이다. RUDDER·GrAInS는 고정 저자 원본 함수를 호출한다.

공식 POPE evaluate.py 및 원본 CHAIR Python3 포트로 채점했다. 이미지 단위 paired bootstrap 10,000회의 95% 구간은 report.json에 있다.

VTI는 원시 fit hidden states를 모두 보존하지 않아 PCA 자체의 독립 재구성은 할 수 없다. 종료 코드를 직접 관측하지 못한 인계 작업은 그 한계를 execution_history에 표시했다.
