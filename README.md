# TETHER / RBD — ablation · 튜닝 · steering 실험 코드와 결과 모음 (2026-10-30 ~ 2026-11-11)

이 브랜치(`ablations-code-results-20261111`)는 `main`과 히스토리를 공유하지 않는 orphan 브랜치입니다.
담긴 것:
- 논문 ablation
- 우리 방식(RBD → TETHER) 점수를 올리려고 dev/test 일부에서 돌린 작은 sweep·pick 실험
- 11-10~11의 steering 변형 실험 (anchor, no-anchor 루프, vast 박스)

각 실험의 **코드와 텍스트 결과**만 담았습니다.

| 폴더 | 원래 위치 |
|---|---|
| `server/` | project-b 서버 `/home/project-b/LJW/vlm-grounding-steering/` (경로 그대로) |
| `vast/steer/` | vast.ai 2080 Ti 박스 `/workspace/steer/` |

**뺀 것**
- 이미지 (COCO, blur 이미지)
- 모델 가중치
- hidden-state 텐서 (`.pt`, `.npy`): `rbd-20261030/collect`, `dvec`, `verify-probe`·`read-gap` 텐서 등
- 20MB가 넘는 파일
- baseline만 있는 run: THRONE 4000(`t4k_*`), `throne_full1000`, `gemma_throne1000`, `amber_full1004`, `*_vtiws` 등
- vast 박스의 venv와 이미지

vast 박스의 절대경로 symlink 19개는 Windows에서 풀리지 않아 빠졌습니다. 원래 무엇을 가리켰는지는 `vast/SYMLINKS.txt`에 적어 두었습니다.

공통 사항
- 모든 CHAIR 점수는 수정하지 않은 CHAIR scorer(`server/data/hallucination-suite/chair-scorer-state.json`)로 계산했습니다.
- set은 `chair_test500`(= `server/outputs/rbd2-20261102/sets/chair_test500.jsonl`)입니다.
- 비교는 paired bootstrap 95% CI입니다.

---

## 1. 논문 component ablation (CHAIR test 500)

| | Qwen2.5-VL | Gemma E4B |
|---|---|---|
| 코드 | `server/scripts/rbd2/abl_prep.py`, `abl_chain.sh` | `server/scripts/rbd2/gabl_prep.py`, `gabl_chain.sh` |
| 결과 (합본) | `server/outputs/rbd2-20261102/runs/chair_abl500_all/` | `.../gabl_chair500_all/` |
| 부분 run, full 대비 비교 | `chair_abl500/`, `chair_abl500_vsfull/` | `gabl_chair500/`, `gabl_chair500_vsfull/`, `gabl_repro/` |
| 설정 | L16, α2, λ1, π0.5 | L24, α1, λ1, π0.5 |

점수 표는 `server/outputs/steer-noanchor-20261110/RESULTS_SUMMARY.txt` 맨 위에 있습니다. 순서는 CHAIR_s / CHAIR_i / recall / tok입니다.

| Config | Qwen2.5-VL | Gemma E4B |
|---|---|---|
| Base | 30.4 / 8.40 / 69.8 / 189 | 34.6 / 7.00 / 64.7 / 370 |
| Full | 25.6 / 6.86 / 68.9 / 178 | 26.2 / 5.54 / 63.6 / 378 |
| − steering | 27.6 / 7.79 / 67.4 / 191 | 26.4 / 5.94 / 61.5 / 374 |
| − penalty | 28.2 / 7.47 / 68.9 / 177 | 27.4 / 5.50 / 63.9 / 377 |
| − belief | 29.6 / 8.19 / 69.3 / 175 | 34.6 / 7.30 / 65.1 / 375 |
| steering only | 31.6 / 8.26 / 69.8 / 175 | 35.0 / 7.58 / 65.7 / 373 |
| belief only | 27.0 / 7.26 / 68.4 / 191 | 26.2 / 5.83 / 61.8 / 372 |
| penalty only | 29.8 / 8.02 / 69.8 / 190 | 34.4 / 7.20 / 64.3 / 371 |
| const schedule | 27.0 / 7.72 / 68.3 / 167 | 26.4 / 5.22 / 62.0 / 384 |

---

## 2. 우리 방식 튜닝: dev/test 일부에서 돌린 작은 실험

디코더는 `server/scripts/rbd2/` 아래 `vdd_decode.py` 계열, `vdd3_decode.py`, `vdd7_decode.py`입니다. 실행 큐는 `server/outputs/rbd2-20261102/queues/`, 로그는 `server/outputs/rbd2-20261102/logs/`에 있습니다.

| 실험 | 날짜 | 결과 위치 (`server/outputs/...`) | 요약 |
|---|---|---|---|
| RBD pilot (round 1) | 10-30 | `rbd-20261030/dev`, `test100`, `logs` | dev 100: belief λ0.1이 Base보다 높음 (F1 69.09 vs 67.09). measured r(t)는 constant r보다 나빴고, steering만 단독으로 쓰면 오히려 나빠짐 |
| RBD2 CHAIR sweep (Qwen) | 11-01~02 | `rbd2-20261102/runs/chair_dev60`, `chair_test60`, `chair_test60_sweep`, `chair_test120(_all)`, `chair_test150(_all)`, `chair_test500(_all)`, `pilot_chair500_*` | test 60에서 고른 설정을 120/150/500장으로 넓힘. 120장에서 효과가 줄어듦 (base 30.0 vs 28.3) |
| Gemma CHAIR (ours) | 11-03~05 | `gchair_dev30_peek`, `gchair_dev60`, `gchair_smoke`, `gchair_t60`, `gchair_test500(_all,_tbl)` | pick s24a2bl1p05: 30.8 vs vdd Base 34.6. 이후 표에서는 s24a1bl1p05 사용 |
| Qwen3-VL CHAIR (ours, rbd5/rbd6) | 11-05 | `qchair_t60`, `qchair_test500(_all,_tbl)`, `qchair_interim_1758`, `qchair_report_*`, `rbd5_qwen3_all`, `rbd6_*`, `vdd5/` | gate/cfgate 변형 |
| CHAIR dev 100 (3모델) | 11-05~06 | `chair_dev100_{qwen,gemma,qwen3}(_all,_vdd)`, `chair_dev100/` | Qwen2.5 RBD 22.0 vs 32.0. Gemma RBD 25.0 vs 40/42 |
| AMBER 튜닝 | 11-03~04 | `amber_dev60`, `amber_test60(L)`, `amber_test150(_all)`, `amber_test500(_all)`, `amber_v3_*` (vdd3 gate/stuff/hedge), `amber_full1004_ours(_all)`, `amber_ours_partial` | 1004장 mainMed 5.0 / 63.4 / 27.8 / 1.3 vs Base 5.4 / 64.1 / 30.1 / 1.6. 유의하지 않음 |
| THRONE 튜닝 | 11-03~05 | `throne_test60`, `throne_test120(_all)`, `throne_test150(_all)`, `throne_test500(_all)`, `gemma_throne_dev60`, `throne_full1000_ours`, `throne_full500_2judge` | |
| THRONE colour-fix (v2.1) | 11-05 | `qwen_cf_t200(_all)`, `qwen_cf_test(_all)`, `gemma_cf_t200(_all)`, `gemma_cf_test(_all)`, `*_cf_pick.txt` | 앞 200장으로 pick → cf_a3bl2p1. 나머지 800장: Qwen F0.5_ALL 85.8 → 87.4 유의, Gemma는 유의하지 않음 |
| HAT v7 (ALOHa) | 11-05~09 | `hat385/`, `hat385_qwen`, `hat385_gemma`, `hat385_smoke_*`, `v7_smoke`, `hat_parse/` | pick v7_blur_a2bl2p1. 선택에 안 쓴 100–384에서 효과 없음 |
| smoke / repro | | `gemma_smoke`, `gchair_smoke`, `runs_dev_cloud/`, `main/` | |

초기 변형 아이디어(10-30, 텍스트 결과만)는 아래와 같습니다. 모두 selectivity test에서 실패했거나 보류됐습니다.

| 폴더 | 내용 |
|---|---|
| `server/outputs/verified-steer-20261030/` | 확인된 물체 logit steering. 보류 |
| `server/outputs/query-anchor-step0-20261030/` | query-anchor steering |
| `server/outputs/verify-probe-20261030/` | probe steering, 로그만 |
| `server/outputs/verified-decoding-20261030/` | 검증 디코딩. tau0 점수 포함 |

---

## 3. Steering 변형 실험 (11-10 ~ 11-11)

### 3a. vast 2080 Ti: Qwen steering screen + anchor (`vast/steer/`)
- 프로토콜: `vast/steer/PROTOCOL.md`
- 디코더: `vast/steer/vdd_steer.py`. 서버 최신본은 `vast/steer/vdd_steer_srv.py`
- 채점: `cscore.py`, `steer_score.py`, `steer_table.py`

| run 폴더 | 내용 |
|---|---|
| `runs/steer100`, `steer100b`, `steer100c` | round 1~3 screen, 20개 변형, rows 0–99 |
| `runs/confirm400`, `runs/final500` | full_an16_r03 500장: 24.4 / 6.45 / 67.0 / 162. −steering 대비 차이 유의하지 않음 |
| `runs/abl_new`, `smoke_abl_new` | anchor 설정의 −penalty / −belief, Qwen 500 |
| `runs/interim_e6d8`, `recheck_nosteer`, `steer_contrasts.json` | 다른 세션의 중간 확인 |
| `runs/gemma_abl` | Gemma: λ0.75·π0.35 Full vs w/o steering (100장), λ0.5 −penalty (진행 중 스냅샷) |

### 3b. 서버: anchor 계열 (다른 세션) — `server/outputs/`

| 폴더 | 내용 |
|---|---|
| `steer-anchor-20261110/gemma` | Gemma anchor ablation, λ1과 λ0.5. 11-10 22:2x에 중단돼 일부만 있음 |
| `steer-anchor-20261110/qwen3*` | Qwen3 anchor / rc 후보 |
| `steer-nobelief-20261110` | Qwen anchor −belief 500 |
| `steer-dev-20261110`, `steer-beliefdev-20261110` | dev 100 run |
| `steer-qwen3diag-20261110` | Qwen3 r(t) 곡선, 후보 진단 |
| `steer-gemma-search-20261111` | Gemma 탐색 준비 (hwcheck) |

해당 스크립트는 `server/scripts/steer_vast/`에 있습니다.

### 3c. 서버: no-anchor steering 루프 (`server/outputs/steer-noanchor-20261110/`)
- 시간순 기록: `PROTOCOL.md`
- 결과 정리: `RESULTS_SUMMARY.txt`
- run: `qwen/`, `gemma/`. vast에서 가져온 Gemma run은 `gemma_vast/`
- 실행·채점 스크립트: `pool.sh`, `queue.txt`(+ 백업), `na_score.py`, `na_vs.py`, `na_vs2.py`
- 디코더: `server/scripts/steer_vast/vdd_steer.py`. 이 루프에서 `--ssched early|gate`, `--steer restore`를 추가했고, 이전 버전은 `vdd_steer.py.bak-*`로 남겨 뒀습니다.

방식은 다음과 같습니다.
- belief를 약하게 둠 (λ0.5, π0.25)
- 같은 belief에서 steering만 뺀 run(`ns`)을 같은 GPU로 직접 돌려 기준으로 삼음
- 앞 100~200장으로 고르고, 나머지 rows로 확인

**결과**

| Gemma E4B, 500장 | CHAIR_s | CHAIR_i | recall | tok |
|---|---|---|---|---|
| ns | 34.0 | 6.89 | 63.2 | 370 |
| L24 α2 | 31.8 | 6.21 | 63.3 | 377 |
| L18 α2 | 32.6 | 6.71 | 65.2 | 423 |

- **L24 α2 (ns 대비):** CHAIR_s −2.2 [−6.0, +1.4], CHAIR_i −0.7 [−1.59, +0.18]. 고르는 데 안 쓴 rows 200–499에서는 +1.0.
- **L18 α2 (ns 대비):** recall +2.0 [+0.3, +3.5]로 유의.
- **나머지 변형 (rows 0–199):** α3, blur 대조, gate s(t), 2-layer, α_t(restore), sel 모두 유의하지 않음.
- **belief를 세게 했을 때:**
  - λ1 + L18 α2 (500장): 29.8. 저장된 −steering 26.4보다 나쁨.
  - vast의 λ0.75 + L24 α2 (100장): Full 35.0 vs w/o steering 29.0.
- **Qwen2.5-VL:**
  - L12 α6: 고른 rows에서는 −4였지만, 고르는 데 안 쓴 200장에서는 +0.5.
  - restore κ1 (500장): 29.4 vs 29.2.
  - restore κ1.5: 유의하게 나빠짐.
  - λ1 + L16 α3 (200장): 23.5 vs 저장된 −steering 28.0, TETHER 26.5. 다만 저장 결과는 다른 decode에서 나와서 판정 불가.

결론: d_t steering은 주로 recall과 CHAIR_i를 바꾸고, CHAIR_s 이득은 belief가 약할 때만 −2 안팎입니다.

---

## 다시 채점하는 법

서버 경로 기준이고, 로컬에서는 `server/`를 루트로 두면 됩니다.

```bash
source scripts/steer_vast/steer_env.sh; export PYTHONPATH=$PYTHONPATH:$R/scripts/rbd2
python scripts/steer_vast/steer_table.py chair_test500 data/hallucination-suite/chair-scorer-state.json NAME=RUN.jsonl ...
python outputs/steer-noanchor-20261110/na_vs.py gemma 0 500 L24a2 ns stored:base   # paired bootstrap
```

## 주의

- **dev 없이 test에서 고른 실험이 많습니다.** 각 실험에서 어느 rows로 고르고 어느 rows로 확인했는지는 해당 PROTOCOL/README와 로그에 적혀 있습니다.
- **GPU 종류가 섞여 있습니다** (A6000 / A5000 / 2080 Ti). 같은 설정이라도 캡션이 다르게 나오므로, 같은 GPU에서 나온 run끼리 비교해야 합니다.
- **`gemma_abl`은 진행 중 스냅샷입니다** (λ0.5 −penalty가 돌고 있음).
