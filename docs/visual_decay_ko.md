# 캡션이 길어질수록 이미지 정보가 약해지는가: 분석 스크립트

Base CHAIR 캡션을 teacher forcing으로 다시 넣어 디코딩 위치별로 이미지 의존도를 잽니다. 기존 `visual_reliance.py`(H1–H3: attention 비중, 회색 이미지 대비 log p, Jacobian)와 겹치지 않는 세 가지를 추가합니다.

| 실험 | 측정 | 반증 조건 |
|---|---|---|
| E1 attention | 층별로 생성 위치 t의 query가 이미지 / 프롬프트 / 지금까지 쓴 캡션에 주는 attention 비중. pre-image 토큰(sink)은 따로 뺍니다. 이미지 토큰을 흡수 상태로 둔 rollout도 함께 잽니다. Gemma는 sliding window 때문에 이미지 키가 마스크에서 빠지는 비율도 기록합니다. | 이미지 비중이 위치에 따라 줄지 않으면 가설 기각 |
| E2 logit lens / 표현 | 같은 캡션을 실제 이미지와 회색 이미지로 넣고, 모든 층에서 logit lens log p 차이, ‖h_real − h_gray‖/‖h‖, cos(h_real, h_gray)를 잽니다. 25/50/75/100% 깊이 hidden state로 COCO 객체 존재를 맞히는 linear probe도 학습합니다(실제 이미지 − 회색 이미지 AUROC, 아직 언급하지 않은 객체만). | 후반부에도 probe의 이미지 기여(AUROC 차)가 유지되면 "정보 소멸"이 아님 |
| E3 인과 개입 | 회색 이미지 / 다른 이미지 / 위치 k(0, 32, 64, 128) 이후 query에서 이미지 키 차단(전체 층, 앞 절반 층, 뒤 절반 층). 토큰마다 KL, Δlog p, greedy 일치율을 잽니다. | 이미지 키를 차단한 KL ÷ 회색 이미지 KL이 1에 가까우면, 후반 토큰도 이미지를 직접 읽고 있다는 뜻 |

E3의 차단은 attention 가중치에만 적용합니다. 그래서 k > 0이면 k 이전 캡션 토큰이 이미 읽어 둔 이미지 정보는 KV cache 경로로 남습니다. "차단 KL ÷ 회색 KL"이 직접 경로의 몫이고, 나머지는 앞선 토큰을 거쳐 전달된 몫입니다.

## 실행

서버 `/home/project-b/LJW/vlm-grounding-steering`에서 실행합니다.

```bash
bash scripts/visual_decay_run.sh 1 qwen          # GPU 1, 캡션 0–500, 125개씩 shard
bash scripts/visual_decay_run.sh 1 gemma 0 300   # 고정 300장 subset
bash scripts/visual_decay_run.sh 1 qwen 0 500 125 --regenerate   # 캡션을 이 설정으로 직접 greedy 생성
```

- GPU 여유 메모리가 `NEED_GB`(기본 30GB)보다 적으면 shard를 시작하지 않고 멈춥니다.
- 출력은 `outputs/visual-decay-20261030/`에 생깁니다. 캡션별 `.pt`, `{model}-visual-decay-summary.json`, `visual_decay_report.html`(그래프와 표)입니다.
- 코드 경로만 확인할 때는 `--cpu-smoke`를 씁니다(float32, CPU). 이 옵션의 수치는 bf16 결과와 비교하면 안 됩니다.

## 해석 시 주의

- **Qwen의 Base 캡션은 순수 greedy가 아닙니다.** Qwen generation config의 `repetition_penalty: 1.05`가 `do_sample=False`에서도 적용됩니다. 그래서 raw argmax와 비교하면 재현율이 낮게 나옵니다. 수집 스크립트는 penalty를 반영한 재현율(`base_greedy_cfg`, status의 `greedy_reproduced_mean`)을 따로 기록합니다. 직접 생성한 캡션으로 CPU에서 확인한 값은 100%였고, 저장된 캡션은 GPU bf16 첫 11개 기준 93%입니다. CPU fp32에서 더 낮게(89%) 나온 건 정밀도 차이 때문입니다. 결과가 일부 캡션에 좌우되는지는 `visual_decay_analyze.py ... --min-repro 0.95`로 다시 집계해서 확인합니다. logit lens와 KL은 penalty를 적용하기 전의 raw logit 기준입니다.
- **Gemma 4 E4B는 42층 중 35층이 sliding window(512)입니다.** 이미지(256토큰)와 프롬프트, 캡션을 합친 길이가 512를 넘으면 그 층들은 이미지에 구조적으로 접근하지 못합니다. 캡션 약 244토큰 이후부터 해당됩니다. "이미지를 잊는다"는 결과가 나오면 학습된 현상인지 이 구조 때문인지 구분해서 써야 합니다. 리포트에 sliding 층과 full-attention 층이 따로 나옵니다.
- **Attention 비중 하락은 정보 미사용과 같은 말이 아닙니다.** 주장은 E3(인과 개입)와 E2 probe로 뒷받침해야 합니다.
- **Probe는 고정 cohort로 봐야 합니다.** 후반 위치는 긴 캡션에만 존재하므로, 짧은 캡션이 빠지면서 생기는 편향을 막으려고 cohort 결과를 따로 냅니다.
