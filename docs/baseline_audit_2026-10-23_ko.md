# Baseline 원본 대조 감사 — 2026-10-23

**판정: 현재 코드는 동작 검증을 거친 논문 기반 변형 구현이다. 네 baseline 모두
저자 공개 코드와 동등하지 않으며, 논문 성능을 재현한 완성본으로 사용할 근거는 없다.**

검토 기준은 원격 `main`의 `3d5359edc741449f715be6d5d7ea26c5e9510720`이다.
기존 로컬 checkout을 이 커밋으로 fast-forward한 뒤 원본 소스를 새로 다운로드했다.
이번 작업은 감사 스크립트·통합 테스트·문서를 추가했으며 baseline 실행 로직은 변경하지 않았다.

## 확인한 범위와 결과

| 확인 항목 | 결과 |
|---|---|
| 네 저자 저장소 재다운로드 및 커밋 검증 | 완료. 아래 고정 커밋이 조회 당시 각 원격 HEAD와도 일치 |
| 원본 코드와 추출·개입·보정 절차 비교 | 완료. 네 방법 모두 중요한 차이 발견 |
| 동일 합성 입력의 원본 함수/클래스와 로컬 구현 비교 | 7개 비교 항목에서 차이 검출 |
| 전체 CPU 테스트 | **86 passed, 0 skipped**, 21.37초 |
| 정적 검사 | `ruff check src tests scripts` 통과 |
| 실제 HF 모델 클래스의 추출→생성 | 작은 무작위 Qwen2.5, Qwen3, Gemma4, LLaVA에서 12개 추가 테스트 통과 |
| CHAIR / Causal-HalBench 원본 scorer | 기존 합성 fixture 통합 테스트 재통과 |
| pretrained checkpoint의 실제 이미지 추론 및 원본 출력 일치 | 미실행 |
| 논문 benchmark 성능 / EPHS 우월성 | 미검증 |

환경: Windows, Python 3.12.14, torch 2.8.0+cpu, torchvision 0.23.0+cpu,
Transformers 5.20.2. 실제 장치는 GTX 1650 4GB이며 이번 환경은 CPU wheel을 사용한다.
현재 설정처럼 전체 7B/8B 모델을 bfloat16으로 GPU에 올리는 실험은 이 VRAM에 맞지 않는다.
모델 가중치와 실제 benchmark 데이터는 다운로드하지 않았다.

추가 테스트는 tokenizer/processor 대신 고정 token·pixel tensor를 공급한다.
실제 HF 아키텍처에서 비영벡터 추출, alpha=0 복원, 비영 개입의 logit 변화,
cached generation, hook 복원, 가중치·gradient 불변을 확인한다.
LLaVA의 HF 내부 출력 수집 hook은 첫 forward 때 생기므로 먼저 정상 forward를 수행한
상태와 비교했다. 새 테스트 작성 중 확인한 이 동작은 로컬 hook 누수가 아니었다.

## 원본 커밋과 재검증 자료

| 방법 | 새로 받은 원본 |
|---|---|
| REVIS | [antgroup/Revis — 75f4d29](https://github.com/antgroup/Revis/tree/75f4d29d402d4c1c4dbb2c69e88e3aa50cd0c9cc) |
| RUDDER | [Akko000/RUDDER — bd149d8](https://github.com/Akko000/RUDDER-Residual-Update-Directed-DEcoding-Regulation-/tree/bd149d8dbde6a9097dd02aea80f83d0bd8876761) |
| GrAInS | [duykhuongnguyen/GrAInS — 8ad78db](https://github.com/duykhuongnguyen/GrAInS/tree/8ad78db3040fbb7b0aec9e544a3dcc7b9c050b5b) |
| VTI | [shengliu66/VTI — bb5dfee](https://github.com/shengliu66/VTI/tree/bb5dfee33a9e555f22af9faa7b3495c0ac3df69e) |

숫자와 파일 SHA-256은 [감사 결과 JSON](audits/baseline_parity_2026-10-23.json)에 저장했다.
[감사 스크립트](../scripts/audit_baseline_parity.py)는 clean pinned checkout에서 필요한
원본 함수/클래스의 AST를 수정 없이 실행한다. 모델·데이터 입출력은 작은 fixture로
대체한다. 원본 전체 실행이나 실제 모델의 재현 성공을 의미하지 않는다.
GrAInS objective 항목은 코드에서 확인한 식의 수치 예시이며 전체 원본 IG 실행은 아니다.

## 1. REVIS: 벡터 추출과 calibration이 모두 다르다

로컬 위치: `baselines.py:_revis_extract`, `revis_update`, `build_baseline`.

원본 `vector_calculator.py`는 다음 순서로 visual 방향을 만든다.

1. 언어 prior 차이 표본에 PCA를 적용하고 평균 L2 norm을 곱한다.
2. 각 visual 차이 표본에서 이 prior 방향의 성분을 제거한다.
3. 정제된 표본에 다시 PCA를 적용하고 평균 norm을 복원한다.

로컬 코드는 visual 차이와 prior 차이를 각각 **평균한 뒤 직교화**한다.
동일한 합성 표본으로 만든 두 방향의 cosine은 **0.4581**이었다.
이는 PCA 부호 하나의 차이로 설명되지 않는다.
[원본 벡터 계산](https://github.com/antgroup/Revis/blob/75f4d29d402d4c1c4dbb2c69e88e3aa50cd0c9cc/steering_utils/vector_calculator.py#L29)

Calibration 차이는 별도로 중요하다. 원본은 POPE 질문으로 실제 생성을 수행하고,
**첫 forward의 마지막 prompt 상태**에서 risk를 기록한다. 생성 결과와 GT를 비교해
TP/TN을 factual, FP를 hallucination으로 분류한다. 로컬은 chosen/rejected 답변을
teacher forcing으로 끝까지 넣고 **답변 끝 상태**의 risk를 모은다. 그 분포의
분위수로 정한 threshold를 생성 도중 상태에 적용하므로, 원본 threshold 보정과
동일하다고 볼 수 없다. 기본 분위수도 원본 CLI 90%, 로컬 80%이다.
[원본 calibration](https://github.com/antgroup/Revis/blob/75f4d29d402d4c1c4dbb2c69e88e3aa50cd0c9cc/layer_select.py#L73)

추론의 `risk = -cos(h, v)`와 threshold를 넘을 때 방향을 더하는 수식은
동일 조건에서 일치한다. 하지만 원본 hook은 prefill 전체 위치를 gate하고,
로컬은 마지막 위치만 수정한다. 두 토큰 예제에서 원본은 2개, 로컬은 1개 위치가
변했다. 이후 층의 prompt cache도 달라질 수 있다.
[원본 hook](https://github.com/antgroup/Revis/blob/75f4d29d402d4c1c4dbb2c69e88e3aa50cd0c9cc/steering_utils/llm_layers.py)

**해석:** 논문 부록 G에는 평균 차이와 Gram–Schmidt가 명시돼 있어 로컬 평균 방식을
단순한 오타로 취급해서는 안 된다. 논문 수식과 공개 구현 사이에도 차이가 있다.
저자 코드 재현을 목표로 하면 PCA·calibration·prefill 범위를 함께 맞춰야 한다.
[논문 부록 G](https://arxiv.org/html/2602.11824v2#A7)

## 2. RUDDER: 같은 Beta 수식에 서로 다른 방향과 상태를 넣는다

로컬 위치: `baselines.py:_rudder`, `beta_gate`; `backend.py:attn_updates`.

공개 Qwen 코드의 CARD는 self-attention 모듈 **출력 − 입력**을 pooling한다.
로컬은 실제 attention residual update 자체를 pooling한다. Qwen에서는 `o_proj` 출력,
Gemma4에서는 attention normalization 뒤 출력을 쓴다. 동일한 작은 attention fixture에서
두 CARD의 cosine은 **−0.8441**이었다. 단순한 alpha 조정으로 같아질 관계가 아니다.
[원본 CARD](https://github.com/Akko000/RUDDER-Residual-Update-Directed-DEcoding-Regulation-/blob/bd149d8dbde6a9097dd02aea80f83d0bd8876761/Qwen/qwen_methods.py#L190)

Beta gate 식은 동일 similarity·clamp 조건에서 일치했다. 그러나 원본은 attention
출력으로 similarity를 계산하고 로컬은 decoder block 입력을 사용한다.
따라서 같은 식을 구현했다는 사실만으로 실제 gate나 출력이 일치하지 않는다.
원본 Qwen POPE runner도 마지막 non-pad prefill 위치를 허용하므로, 모든 원본 RUDDER가
첫 토큰을 전혀 수정하지 않는다고 단정해서도 안 된다.
[원본 gate](https://github.com/Akko000/RUDDER-Residual-Update-Directed-DEcoding-Regulation-/blob/bd149d8dbde6a9097dd02aea80f83d0bd8876761/Qwen/qwen_methods.py#L255),
[POPE mask](https://github.com/Akko000/RUDDER-Residual-Update-Directed-DEcoding-Regulation-/blob/bd149d8dbde6a9097dd02aea80f83d0bd8876761/Qwen/qwen_pope.py#L376)

**해석:** 논문 v2 Eq.1은 pre-norm decoder의 residual update를 attention 출력 그
자체로 정의한다. 이 부분은 로컬이 논문 식을 따른다. 원본 코드의 subtraction을
공식 정답으로 가정해 기존 구현을 덮어쓰면 재현 기준이 오히려 불명확해진다.
논문 정의를 따르는 변형과 공개 코드 호환 실행을 명시적으로 구분해야 한다.
[논문 §3.2](https://arxiv.org/html/2511.10292v2#S3.SS2)

## 3. VTI: 개입 연산과 PCA가 실질적으로 다르다

로컬 위치: `baselines.py:_vti_extract`, `_masked_pixels`, `build_baseline`.

| 단계 | 공개 구현 | 현재 로컬 |
|---|---|---|
| 가림 값 | 이미지의 채널별 공간 평균 | processor 출력 공간의 0 |
| 텍스트 PCA | 층들을 펼쳐 공동 PCA 후 PC 성분에 평균을 더함 | 층별 독립 unit PC1 |
| 시각 PCA | 원본의 reshape로 묶은 특징에 PCA 후 평균을 더함 | 층·위치별 독립 PC1 |
| hook 위치 | MLP 출력 뒤, residual 합산 전 | decoder/vision block 전체 출력 뒤 |
| 업데이트 | 정규화한 입력과 방향을 `0.1 × alpha`로 섞고 기존 norm 복원 | `hidden + alpha × vector` |
| 텍스트 prefill | 모든 위치 | 마지막 위치 |

동일 텍스트 차이 표본의 방향 cosine은 **0.3500**이었다. 입력 norm이 5인 예제에서
원본 VTI는 약 5를 유지했고 로컬 마지막 토큰은 **5.3151**이 됐다.
원본 반환값의 fp16 반올림 때문에 norm은 정확히 5와 같지는 않다.
공개 예제의 mask ratio 0.99 / 50 trials와 로컬 기본값 0.1 / 8도 다르다.
[원본 VTI 연산과 hook](https://github.com/shengliu66/VTI/blob/bb5dfee33a9e555f22af9faa7b3495c0ac3df69e/vti_utils/llm_layers.py),
[mask 및 PCA](https://github.com/shengliu66/VTI/blob/bb5dfee33a9e555f22af9faa7b3495c0ac3df69e/vti_utils/icv_utils.py)

**해석:** 로컬의 corrupted−original visual contrast와 chosen−rejected text contrast는
논문 방향 정의를 따른다. 그러나 공개 코드와 비교할 때는 alpha만 조정해서 해결되지
않는다. 원본 연산, tensor 축, residual 경계, 입력 mask를 묶어서 대조해야 한다.
[논문 §4](https://arxiv.org/html/2410.15778v2#S4)

## 4. GrAInS: attribution·마스킹·steering 차이와 원본 결함을 함께 다뤄야 한다

로컬 위치: `baselines.py:_preference_attributions`, `_grains_extract`,
`sequence_log_probability`, `norm_preserving_add`.

공개 attribution 함수는 token embedding을 직접 얻고 `inputs_embeds`와
`attention_mask`만 모델에 전달한다. 이 forward에는 processor가 만든 `pixel_values`가
전달되지 않는다. 마지막 위치 logits에서 마지막 **입력** token의 확률을 읽으며,
답변까지 포함한 embedding에 attribution을 계산한다.
로컬은 이미 시각·텍스트가 합쳐진 첫 decoder 입력 중 prompt 부분에만 IG를 수행하고,
다음 토큰으로 shift한 **전체 completion log-probability 합의 차이**를 최적화한다.
[원본 attribution](https://github.com/duykhuongnguyen/GrAInS/blob/8ad78db3040fbb7b0aec9e544a3dcc7b9c050b5b/src/attribution/gradient/vlm_grad.py),
[원본 IG](https://github.com/duykhuongnguyen/GrAInS/blob/8ad78db3040fbb7b0aec9e544a3dcc7b9c050b5b/src/attribution/gradient/utils.py)

공개 ablation은 PAD/UNK token ID 교체이며 response별 상태 차이를 합쳐 PCA를 계산한다.
로컬은 fused prompt embedding을 0으로 바꾸고 prompt 마지막 상태의 차이를 수집한다.
공개 `add_steering_hook`에는 10배 계수와 `dim=1` norm이 있다. `[B,T,H]`에서
`dim=1`은 sequence 축이다. 단일 decode token `[3,4]`에 같은 부호의 방향을 더한
예제에서는 원본 hook의 변화가 **0**, 로컬 변화는 **0.2265**였다.
이 원본 동작은 구현 결함이 의심되는 부분이며, 그대로 복제했다고 과학적으로 올바른
baseline이 되는 것은 아니다.
[원본 vector와 hook](https://github.com/duykhuongnguyen/GrAInS/blob/8ad78db3040fbb7b0aec9e544a3dcc7b9c050b5b/src/utils/steering.py#L55)

**해석:** 논문은 preference log-probability, IG, 양/음 두 PCA 방향과 표현 크기 보존을
설명하며 현재 로컬 구현의 일부 선택은 이 설명에 더 직접적으로 대응한다.
다만 이 사실이 저자 공개 코드나 발표 성능 재현을 증명하지는 않는다.
Qwen3의 DeepStack과 Gemma4의 per-layer input은 첫 fused stream을 mask해도 남는다.
해당 모델에서는 모든 시각 경로를 제거한 attribution이라고 표현하면 안 된다.
[논문 §3](https://arxiv.org/html/2507.18043v2#S3), [모델별 경계](model_compatibility.md)

## 원본 전체 실행에도 남아 있는 장애

조회한 고정 커밋 기준으로 다음 파일/정의 누락을 확인했다.

- REVIS `main_all_visual_only.py`가 import하는 `utils.text_halu`, `utils.mmhal_bench` 파일이 없다.
- RUDDER `Qwen/qwen_chair.py`가 import하는 `methods_decodings_qwen.py` 파일이 없다.
- GrAInS `steering_generation_vlm.py`가 import하는 `generate_answer_dataset_vlm`,
  `steer_and_generate_dataset_vlm` 정의가 `utils/steering.py`에 없다.

이들은 소스 정적 검사로 확인한 배포 상태다. 원본 전체 환경을 설치해 모든 실행 오류를
열거한 결과는 아니다. 원본 의존성과 모듈 연결을 수리할 경우 patch와 커밋을 별도 기록하고,
방법론을 바꾸는 수정은 수학적 근거·ablation과 함께 명시해야 한다.

## 현재 코드 읽는 순서

1. **`backend.py:ModelAdapter`** — checkpoint마다 달라지는 모듈 경로를 연결한다.
   `prepare`는 이미지·질문을 토큰화하고 답변이 있으면 prompt labels를 −100으로 가린다.
   `forward`는 extraction용 cache 없는 실행, `generate`는 batch 1 greedy cached 실행이다.
2. **`cli.py:_extract`** — 개발 split의 preference pair를 읽고 baseline 방향을 추출한다.
   REVIS의 별도 tune calibration을 붙이고 checkpoint·설정·이미지 그룹·manifest hash를 저장한다.
3. **`baselines.py:extract_baseline`** — 방법별 extractor를 선택한다. 모델을 eval로 두고
   GrAInS에서 모델 가중치 gradient를 막는다. 완료·예외 후 원래 상태를 복원한다.
4. **`baselines.py:build_baseline`** — 추론 동안만 hook을 건다. Base는 hook 없이 실행한다.
   REVIS는 낮은 시각 정렬에 gate를 켜고, RUDDER는 입력마다 CARD를 만들어 정렬이 높을수록
   강하게 더한다. GrAInS는 양의 방향에서 음의 방향을 뺀 벡터를 더한 뒤 norm을 보존한다.
   VTI는 vision과 text 양쪽에 별도로 방향을 더한다. 현재 text 개입은 마지막 위치다.
5. **`cli.py:_run`** — 매 질문마다 새 hook context에서 한 번 생성하고 결과와 완료 상태를 기록한다.
   모델/전처리가 다른 artifact, 학습·test 이미지 그룹 중복 등을 검사한다.
6. **`evaluation.py` / `cli.py:_compare`** — 공통 데이터의 지표와 이미지 그룹 단위 paired bootstrap을 계산한다.
   미완료 run, 비교 조건 불일치 등을 거부한다. EPHS 고유 discovery·gain 학습은 `method.py`에 있다.

이 구조는 공통 실험 러너로 활용할 수 있다. 결과 이름에는 현재처럼
`paper-derived-local-adaptation`, `author_reproduction=false`를 유지해야 한다.

## 재실행

프로젝트와 eval/test 의존성이 설치된 별도 환경에서 실행한다.
PCA 원본 helper 검증용 `scikit-learn==1.9.1`은 이번 감사의 추가 의존성이다.

```powershell
python -m pip install scikit-learn==1.9.1
python -m vlm_grounding fetch-upstream --names revis rudder grains vti chair causal --output-dir upstream
python scripts/setup_eval.py --download-dir .cache/nltk_data
$env:NLTK_DATA = "$PWD\.cache\nltk_data"
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
python -m pytest -q -ra
python -m ruff check src tests scripts
python scripts/audit_baseline_parity.py --output outputs/baseline_parity.json
```

마지막 명령의 정상 종료는 감사가 실행됐다는 뜻이다. JSON의 `conclusion`과
`mismatched_probes`를 읽어야 한다. 현재 결과는 `not_equivalent`이다.

## 연구 결과로 사용하기 전 필요한 일

Qwen2.5처럼 공통 지원 모델부터 저자 코드 호환 실행 경로와 논문 기반 변형을 구별하고,
동일한 작은 실제 입력에 대해 token IDs, layer/position, 추출 벡터, gate, 첫 토큰 logits,
cached decode를 대조해야 한다. 이후 동일한 개발 예산으로 방법별 튜닝을 하고,
고정한 test 데이터에서 CHAIR/POPE/MMHal 등을 실행해야 한다.
현재 CPU 테스트와 합성 차이 실험만으로 어느 방법의 실제 성능이 더 좋은지는 알 수 없다.
