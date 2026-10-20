# VLM Grounding Steering

**GPU 실험을 준비하는 연구 구현입니다. 성능 개선이나 논문 재현 성공을 보고하는 저장소가 아닙니다.**

제안 방법 **EPHS (Evidence-Preserving Head Steering)**와 Base, REVIS, RUDDER,
GrAInS, VTI를 같은 모델·데이터·평가 코드에서 비교합니다. 기본 모델은
`Qwen/Qwen2.5-VL-7B-Instruct`, 두 번째 모델은 `llava-hf/llava-1.5-7b-hf`입니다.

## 연구 질문과 방법

주변 객체의 익숙한 조합 때문에 없는 객체를 있다고 판단하는 오류를 줄이면서,
실제 객체에 대한 정확한 정보를 보존할 수 있는가?

EPHS의 개발 단계에서는 COCO 원본, 특정 객체를 가린 이미지, 같은 모양으로 배경을
가린 대조 이미지를 사용합니다.

1. 원본의 attention head 출력을 객체를 가린 입력에 patch합니다.
2. 객체 판단의 회복 효과가 크고 배경 가림에 대한 효과는 작은 head를 선택합니다.
   원본 이미지별로 집계한 효과의 중앙값을 사용합니다.
3. 선택한 head의 작은 scaling 계수만 학습합니다. 기본 모델 가중치는 고정하고,
   yes/no 균형 학습과 원래 정답 분포를 보존하는 KL 항을 사용합니다.
4. 추론에는 이미지 하나만 필요합니다. 정답이나 원본 donor activation을 제공하지 않습니다.

기본 개입 위치는 마지막 prefill 토큰과 현재 decode 토큰입니다.
`positions="all"`은 별도 ablation입니다. 무작위 head와 활성 크기 기반 선택도
같은 gain 학습으로 비교할 수 있습니다.

이 가설의 신규성·효과는 미검증입니다. 인과적 head 선택, neuron scaling 자체는
HONES 등과 겹치므로, 실제 기여를 주장하려면 가까운 방법과 추가 비교해야 합니다.
자세한 설계와 제한은 [method.md](docs/method.md)에 있습니다.

## 구현 상태

| 항목 | 상태 |
|---|---|
| EPHS discovery / gain fitting / single-input inference | 구현, CPU 행동 테스트 |
| REVIS / RUDDER / GrAInS / VTI | 논문 기반 독립 구현, **local adaptation** |
| 실제 Qwen2.5-VL / LLaVA 아키텍처의 hook·cached decoding | 작은 무작위 모델로 CPU 검증 |
| COCO / POPE / Causal-HalBench / MME 변환 | 로컬 데이터 입력용 구현 |
| 공통 지표·원본 이미지 단위 bootstrap·CSV/JSON 비교표 | 구현 |
| 7B checkpoint의 실제 GPU 실행·원논문 점수 재현 | **미실행** |
| 제안 방법의 baseline 우월성 | **결과 없음** |

baseline 네 개는 저자 코드를 그대로 실행한 결과가 아닙니다. GrAInS의 fused-embedding
IG, VTI의 고정 visual grid, prefill 개입 범위 등 차이를
[baseline_implementation.md](docs/baseline_implementation.md)에 명시했습니다.
기록에는 `author_reproduction=false`를 남깁니다.

원 저장소는 `fetch-upstream` 명령으로 고정한 커밋을 별도 다운로드할 수 있습니다.
타인 코드·데이터·모델 가중치를 이 저장소에 복사해서 배포하지 않습니다.

## CPU에서 지금 확인하기

Python 3.11 또는 3.12를 사용합니다. 아래 설치는 새 가상환경에서 실행하세요.

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[test,eval]"
python -m vlm_grounding doctor
python -m pytest -q
ruff check src tests
```

테스트는 모델 가중치나 데이터셋을 다운로드하지 않습니다. 실제 HF 모델 클래스를
작은 무작위 설정으로 생성해 시각 입력, GQA head, prefill/decode, gradient와 hook
복원을 검사합니다. 합성 테스트 정답률은 연구 성능으로 보고하지 않습니다.

GPU 서버에서는 해당 CUDA에 맞는 PyTorch 2.8을 먼저 설치하고
`pip install -e ".[test,eval]"`를 실행합니다. 7B 실행은 `extract`, `discover`,
`fit`, `run` 명령을 호출할 때만 모델을 로드합니다.

## 데이터 준비

COCO 2014 이미지와 annotation, POPE JSON, Causal-HalBench 자료는 공식 배포처에서
별도로 받습니다. [데이터 안내](docs/data.md)에 스키마와 필요한 파일이 있습니다.
경로 예시는 실행 머신에 맞춰 바꿔야 합니다.

```bash
# 개발 이미지: discovery / fit / tune을 원본 이미지 ID로 분리
vlm-grounding prepare-coco \
  --annotations data/coco/annotations/instances_train2014.json \
  --image-root data/coco/train2014 --work-dir data/development \
  --discovery-count 200 --fit-count 200 --tune-count 200 \
  --output data/development.jsonl

# 자연 이미지 평가: train과 다른 val 이미지
vlm-grounding prepare-coco \
  --annotations data/coco/annotations/instances_val2014.json \
  --image-root data/coco/val2014 --work-dir data/evaluation \
  --discovery-count 0 --fit-count 0 --tune-count 0 --test-count 2000 \
  --output data/coco_test.jsonl

vlm-grounding prepare-pope \
  --input data/pope/coco_pope_adversarial.json \
  --image-root data/coco/val2014 --subset adversarial \
  --output data/pope_adversarial.jsonl

vlm-grounding prepare-causal \
  --input data/causal/qa.json --image-root data/causal/images \
  --output data/causal_test.jsonl

vlm-grounding verify-splits \
  --manifests data/development.jsonl data/coco_test.jsonl data/pope_adversarial.jsonl
```

POPE의 random/popular도 같은 명령으로 각각 변환합니다. 서로 다른 benchmark 이름이어도
COCO 원본 이미지가 같을 수 있습니다. 원본과 파생 이미지는 같은 `group_id`로 묶습니다.
동일 split 안에서 여러 질문에 같은 이미지를 사용하는 것은 허용합니다.

COCO builder의 chosen/rejected는 annotation에서 만든 **yes/no preference pair**입니다.
이를 사용하는 비교는 공통 VQA 개발 자료를 쓴 adaptation입니다. 원논문의 caption
preference 자료를 재현했다고 해석하지 마세요. 별도 실제 preference JSONL을 넣을 수도
있으며 필요한 필드는 `image,prompt,chosen,rejected,id,group_id,task,split`입니다.

## 제안 방법과 baseline 실행

```bash
vlm-grounding discover --config configs/qwen2_5_vl_7b.json \
  --manifest data/development.jsonl --output artifacts/selection.json

vlm-grounding fit --config configs/qwen2_5_vl_7b.json \
  --manifest data/development.jsonl --selection artifacts/selection.json \
  --output artifacts/ephs.json

vlm-grounding extract --method revis --config configs/qwen2_5_vl_7b.json \
  --manifest data/development.jsonl --calibration data/development.jsonl \
  --output artifacts/revis.pt

vlm-grounding extract --method grains --config configs/qwen2_5_vl_7b.json \
  --manifest data/development.jsonl --output artifacts/grains.pt

vlm-grounding extract --method vti --config configs/qwen2_5_vl_7b.json \
  --manifest data/development.jsonl --output artifacts/vti.pt

vlm-grounding run --method base --config configs/qwen2_5_vl_7b.json \
  --manifest data/coco_test.jsonl --output outputs/base.jsonl

vlm-grounding run --method ephs --artifact artifacts/ephs.json \
  --config configs/qwen2_5_vl_7b.json \
  --manifest data/coco_test.jsonl --output outputs/ephs.jsonl

vlm-grounding run --method rudder --config configs/qwen2_5_vl_7b.json \
  --manifest data/coco_test.jsonl --output outputs/rudder.jsonl
```

REVIS/GrAInS/VTI 추론에는 각각 `--artifact artifacts/<method>.pt`를 지정합니다.
RUDDER는 매 입력의 정상 prefill에서 방향을 만들기 때문에 offline 벡터가 필요 없습니다.
전체 비교는 [scripts/run_gpu.sh](scripts/run_gpu.sh)로도 실행할 수 있습니다.

처음에는 `discover/fit/extract`에 `--limit 4`를 붙여 실제 checkpoint에서 smoke test를
수행하세요. 테스트 split을 임의로 잘라 점수를 보고하지 않도록 `run --split test`에서는
`--limit`를 거부합니다. discovery에 유효한 인과 효과가 없으면 명시적으로 실패합니다.
이 경우 임의의 head로 바꿔 성공한 것처럼 진행하지 않습니다.

설정 파일의 layer·alpha·gain·표본 수는 **미튜닝 시작값**입니다. 모든 방법에 동일하게
정한 개발 예산으로 튜닝하고, 설정을 고정한 뒤 test를 실행해야 합니다.
기본 이미지는 448×448로 변환됩니다. Qwen의 VTI 위치별 방향을 고정하기 위한 공통
전처리이며 원논문 해상도와 다를 수 있습니다. LLaVA processor는 자체 입력 크기로
추가 변환합니다. 다른 해상도 실험은 별도 프로토콜로 기록합니다.

## 실제 CHAIR와 공통 비교표

```bash
vlm-grounding fetch-upstream --names chair causal --output-dir upstream
# CHAIR의 tokenizer/WordNet 자료는 GPU 없이 미리 준비 가능
python scripts/setup_eval.py --download-dir .cache/nltk_data
export NLTK_DATA="$PWD/.cache/nltk_data"
# Windows PowerShell: $env:NLTK_DATA = "$PWD\.cache\nltk_data"

vlm-grounding compare \
  --predictions outputs/base.jsonl outputs/ephs.jsonl outputs/rudder.jsonl \
  --chair-module upstream/chair/chair.py \
  --annotations data/coco/annotations \
  --output-dir outputs/comparison
```

CHAIR adapter는 원 CHAIR의 Python 3 포트를 사용합니다. 외부 scorer가 없으면
`chair_status=not_scored`이며 가짜 CHAIR 값을 만들지 않습니다. 패키지별
추가 의존성은 [데이터 안내](docs/data.md)를 확인하세요. Causal-HalBench 공식 지표는
`--causal-module upstream/causal/metric.py`로 연결합니다.

비교표에는 CHAIR_s/i, annotation 기준 객체 recall, 정확한 객체 언급 수,
출력 길이, yes/no 지표, invalid 비율, MME 점수, 이미지 단위 신뢰구간을 포함합니다.
모델·전처리·생성 예산이 다른 실행은 공통 비교에서 거부합니다.
외부 출력의 protocol metadata가 없으면 검증되지 않은 비교로 표시합니다.

Artifacts에는 checkpoint revision, 설정, 사용한 이미지 그룹, manifest hash를,
출력에는 실제 생성 토큰 수, 시간, 완료 상태를 저장합니다. 학습/추출 이미지와 test
이미지가 겹치거나 중간에 실패한 출력을 완성 결과로 비교하려고 하면 거부합니다.
이미지를 복사한 뒤 다른 ID를 붙이는 경우까지 탐지하는 content deduplication은 아닙니다.

## 실험을 확대할 조건

개발 평가에서 가장 강한 baseline과 비교해 **정확한 객체 언급을 유지하면서 환각이
감소하는지** 먼저 확인합니다. Random/magnitude head 선택, KL 제거, gain 크기,
마지막 토큰/전체 토큰 개입을 ablation으로 비교합니다. 그다음 세 discovery seed,
두 번째 모델, Causal-HalBench 및 MME로 확대합니다.

현재는 성능 수치·순위·accept 가능성에 대한 근거가 없습니다. 결과 표는 실제 GPU
실행 파일에서만 생성합니다. [검증 기록](VALIDATION.md)과
[baseline 출처·구현 차이](docs/baseline_implementation.md)를 함께 읽어주세요.
