# TETHER 논문 실험: 코드와 결과 (final)

`final.tex`(2026-11-11 16:46)의 표, 그림, 본문 수치를 만든 실험의 **코드와 결과만** 모은 브랜치입니다.
`paper-code-results-20261109-draft3`에서 이 원고에 나오는 실험만 남겼고, 그 뒤에 원고에 들어온 실험을 더했습니다.
추가한 실험은 Qwen2.5-VL TETHER 행, Gemma ablation 두 행, 그림 `fig:jacobian`입니다.
11-12에 THRONE 코드를 원고 캡션의 5,000장 판으로 바꿨습니다. 1,000장 실행에 나머지 4,000장을 돌리는 `t4k_*`를 더했고, 5,000장 실행은 아직 끝나지 않았습니다(아래 `tab:throne_full` 절).
이전 브랜치의 이력을 가져오지 않은 orphan 브랜치라서 커밋은 이 브랜치의 것만 있습니다.

**넣은 기준:** 원고에 숫자, 표, 그림으로 나오는 실험만 넣었습니다. 다음은 넣지 않았습니다.

- 설정을 고를 때 돌린 후보들
- 60·120·150장 단위의 중간 실행
- 원고에서 빠진 실험: HAT + ALOHa, AMBER 1,004장, 예전 Gemma ablation, 초기 탐색 표 `tab:dev2` 등

전체 목록은 맨 아래 「넣지 않은 것」에 있습니다.

- 서버(`/home/project-b/LJW/vlm-grounding-steering`)의 파일은 서버와 같은 경로에 있고, draft3 브랜치의 파일을 바이트 그대로 가져왔습니다.
  예외가 하나 있습니다. 큐 파일은 서버에서는 `outputs/rbd2-20261102/queues/`에 있지만 여기서는 `scripts/rbd2/queues/`에 있습니다.
- vast.ai 박스에서 돌린 실험은 따로 두었습니다.
  - `vast-tether-pilot/`: RTX 2080 Ti, Jacobian lens
  - `vast-qwen-3080/`: RTX 3080, Qwen2.5 TETHER 행. 이 박스에는 서버와 같은 경로로 코드를 복제했습니다.
- 결과 파일은 텍스트만 넣었습니다: 캡션(jsonl), 채점(json), THRONE 판정, 로그.
  모델 가중치, 이미지, steering 벡터(`.pt`), 활성값 덤프(`.npz`), 50 MB가 넘는 파일은 넣지 않았습니다.
- 원저자 코드와 채점기는 넣지 않았습니다. 고정한 버전은 맨 아래에 있습니다.
  스크립트는 서버 경로, `.venv`, HF 캐시, GPU를 전제로 하므로 다른 머신에서는 그대로 돌아가지 않습니다.
- 캡션에 나오는 이메일 주소는 COCO 사진의 워터마크를 모델이 옮겨 적은 것입니다.

**이름과 약칭**

- 방법 이름은 TETHER이지만 코드와 파일 이름에는 옛 이름 RBD가 남아 있습니다.
  첫 번째 변형은 `rbd_*`이고, TETHER는 `vdd_decode.py`, `vdd4_decode.py`, `steer_vast/vdd_steer.py`입니다.
- `R/` = `outputs/rbd2-20261102/`
- 설정 태그 `s{L}a{α}bl{λ}p{π}` = `--layer L --alpha α --lam λ --pen π` (`p05`는 0.5). 각 태그의 인자는 `scripts/rbd2/queues/*.txt`에 있습니다.
  - `cf_`: 색 단어를 뺀 V_obj
  - `L16a3b1`: `--steer resid --layers 16 --alpha 3 --lam 1 --pen 0.5`

## 3장, 부록 A: 감사

| 원고 | 코드 | 결과 |
|---|---|---|
| `tab:gap`, 그림 1 a1 (자기 검증) | `scripts/caption_selfcheck.py` → `caption_selfcheck_analyze.py` | `outputs/caption-selfcheck-20261030/{qwen,gemma}-base-analysis.json`, 객체별 `*-base-rows.jsonl` |
| 3.1절과 부록 A.4 POPE (0.940 / 0.916 vs probe 0.880 / 0.859, FN 13/379, 17/303) | `read_gap_collect.py`, `read_gap_collect_v2.py` → `read_gap_analyze*.py`, `read_gap_complement.py` | `outputs/read-gap-e0-20261030/analysis-*.json` |
| `tab:position`, 부록 A.6 (회귀 계수, 언급 순서, 중앙 위치) | `outputs/audit-recheck-20261105/audit_recheck.py`; 보조 `scripts/audit_inline/hall_share_by_position.sh` | `outputs/audit-recheck-20261105/{qwen,gemma}.json`, `inline/hall_share_by_position.out.txt` |
| `tab:findings`, `tab:decay`, 그림 1 a3, 부록 A.1 teacher forcing 재현율 (93.7 / 84.9 / 98.5%), A.2 (key blocking 0.81 / 0.72, logit-lens 기여 0.92 / 1.65, Gemma sliding window 35.5%), A.10 rollout | `scripts/visual_decay_run.sh` (`visual_decay_collect.py` → `visual_decay_analyze.py` → `visual_decay_plot.py`) | `outputs/visual-decay-20261030/*-visual-decay-summary.json` (`teacher_forced_greedy_reproduction`, `rollout_absorbing_image`, `window_limited_layers`), `visual_decay_report.html` |
| grad×input (`tab:findings`, `tab:objtok` 위, A.2의 0.70→0.18, A.4의 0.753 / 0.749 / 0.807 / 0.711) | `visual_reliance.py` → `reliance_analyze.py`, `reliance_analyze_v2.py` | `outputs/visual-reliance-20261030/qwen-reliance-summary.json` |
| `tab:objtok` 아래 (e_t, 전체·첫·반복 언급), 부록 A.7 공출현 (AUROC 0.73 / 0.68) | `outputs/audit-recheck-20261105/audit_recheck2.py` | `{qwen,gemma}-2.json` |
| 3.4절 순위 AUROC (e_t 0.55 / 0.51, log p 0.64 / 0.67), 부록 A.8 "Ranking intervals" | `audit_recheck.py`, `audit_recheck3.py`; lens 행은 `scripts/logit_lens.py` | `{qwen,gemma}.json`, `{qwen,gemma}-3.json`, `outputs/logit-lens-20261104/*-lens-summary.json` |
| 부록 A.3 attention (0.64 / 0.55, 모양 0.49–0.52, head L21–24) | `scripts/verify_probe_attn.py` | `outputs/query-anchor-step0-20261030/*-attn-summary.json` |
| `tab:probe`, 기준선 0.74 / 0.69 | `outputs/audit-recheck-20261105/probe_foldwise.py` (probe 학습 `verify_probe_collect.py` → `verify_probe_analyze.py`) | `{qwen,gemma}-probe-foldwise.json`, `outputs/verify-probe-20261030/*-probe-summary.json` |
| `tab:steer` 위 세 행, `tab:findings` 개입 행, 부록 A.5 선택성 차이와 CI, attention boost β = 3 | `scripts/verify_probe_e1.py` | `outputs/verify-probe-20261030/*-e1-summary.json` |
| `tab:lens_img` 위쪽 (AUROC, 같은 물체 안, 공출현 맞춤) | `scripts/logit_lens.py image/analyze`; 인라인 `audit_inline/lens_image_layers_qwen.sh`, `lens_gemma_image_caption.sh` | `outputs/logit-lens-20261104/{qwen,gemma}-lens-summary.json`, `outputs/audit-recheck-20261105/inline/*.out.txt` |
| `tab:lens_img` "Word in top-5 of a patch" (마지막 층, L21) | 인라인 `audit_inline/lens_word_top5.sh` (`logit_lens.py image`가 저장한 top-5에서 계산) | `inline/lens_word_top5.out.txt` |
| `tab:lens_cap`, 부록 A.8 고정 거리 차이와 CI | `scripts/logit_lens.py topk/topk-analyze` (`outputs/logit-lens-20261104/run_topk.sh`); 인라인 `audit_inline/lens_top100_distance_position.sh`; CI는 `audit_recheck3.py` | `*-topk-summary.json`, `inline/lens_top100_distance_position.out.txt`, `{qwen,gemma}-3.json` |
| 부록 A.8 "Writing step" (L26 −13.3, L27 −14.0, Gemma L34 −34.0, L38 −14.5) | 인라인 `audit_inline/lens_writing_step.sh` | `inline/lens_writing_step.out.txt` |
| 부록 A.8 hidden band top-1 (보조) | 인라인 `audit_inline/lens_hidden_band_top1.sh` | `inline/lens_hidden_band_top1.out.txt` |
| 부록 A.1 addCriterion 77개 제외 | `audit_recheck3.py`; 인라인 `audit_inline/addcriterion_exclusion.sh` | `{qwen,gemma}-3.json`, `inline/addcriterion_exclusion.out.txt` |
| 그림 1 a5, 부록 A.8 Jacobian-lens (Qwen3.5-4B, 100장) | `vast-tether-pilot/out/jlens_run.sh` → `pilot/jlens_probe.py`(`common.py`, `tests/test_jlens_probe.py`) → `pilot/jlens_summary.py`(`stats.py`); 층별 단어 분류 `pilot/jlens_layers.py` | `vast-tether-pilot/out/jlens/summary.json`, `layers.json` |
| `fig:jacobian` (부록 B.4에 있는 그림) | 캡션 생성과 전 위치 lens `out/jlens_pos_run.sh` → `pilot/jlens_positions.py`; 같은 캡션의 top-100 `out/jlens_pos100_run.sh` → `pilot/jlens_positions_top100.py`; 뷰어용 내보내기 `out/jl_export.py` → `vast-tether-pilot/viewer/` | `out/jlens_pos/objects.json`, `out/jlens_pos100/objects.json`, 그림의 이미지 4038 데이터 `viewer/data/4038.js` |

`scripts/audit_inline/`의 8개는 감사 세션이 11-05에 ssh로 서버에 보낸 명령입니다. 세션 기록에서 그대로 옮겼고, 로컬 ssh 래퍼 한 줄만 뺐습니다.
출력은 그때 화면에 찍힌 그대로입니다. import되는 보조 모듈도 들어 있습니다: `verified_decoding_v2.py`, `verified_decoding_v3.py`, `read_gap_analyze_v3.py`, `read_gap_collect_v2.py`, `open_vocab.py`.

`vast-tether-pilot/`는 박스의 `/workspace/tether-pilot` 가운데 이 두 실행에 필요한 부분입니다. 렌즈 `lens_qwen35_4b_final.pt`, `probe.npz`,
위치별 `.npz`(jlens_pos 507 MB, jlens_pos100 1.1 GB), COCO 이미지는 넣지 않았습니다. 뷰어(`viewer/index.html`)는 `fig:jacobian`의 이미지 4038만 열립니다.
`viewer/README.md`의 박스 주소는 지웠습니다.

## 4장, 부록 B: TETHER

| 원고 | 코드 | 결과 |
|---|---|---|
| r(t), `tab:schedule`, 4.2절 가중치 (96%, 35%, 24%) | `scripts/rbd_common.py`의 `r_curve`: `tab:decay`의 gray KL을 첫 구간 1로 정규화하고 구간 중심 사이를 선형 보간 | 감사와 같은 `*-visual-decay-summary.json` |
| V_obj (부록 B.3) | `scripts/rbd_collect.py`; 색 단어를 뺀 판 `scripts/rbd2/vdd4_setup.py` | `outputs/rbd-20261030/collect/{qwen,gemma}-vobj2.json`, `*-vobj2cf.json` |
| 디코더 (그림 `fig:rbd_algo`, 식 `eq:steer`·`eq:correction`) | `scripts/rbd2/vdd_decode.py`. V_obj 파일을 바꿀 수 있는 판은 `vdd4_decode.py`(THRONE, Qwen2.5 ablation)입니다. 단계별 steering 변형을 더한 판은 `scripts/steer_vast/vdd_steer.py`(Qwen2.5 본표 행, Gemma ablation)입니다. 이 판에서 식 `eq:steer`에 해당하는 것은 `--steer resid`이고, 같은 GPU에서 저장된 `vdd_decode.py` 행을 재현하는지 확인했습니다(Gemma 2행, Qwen3 2행) | — |
| 부록 B.4 첫 번째 변형, dev 결과 (λ ∈ {0.03, 0.05, 0.1, 0.3, 1}, measured vs constant r 46.7 vs 52.4, uniform, mode steering) | `make_probe_set.py` → `make_dev_chair.py`, `rbd_dvec.py`, `rbd_decode.py`(`rbd_common.py`), 채점 `score_dev.py` | `outputs/rbd-20261030/dev/` (표는 `score-qwen/dev-score.json`) |
| 부록 B.4 파일럿 (CHAIR 테스트 처음 100장 30.0→18.0, uniform 18.0, recall 58.2 vs 53.7; AMBER 1–100 Cover −5.2 ~ −7.6) | `rbd_decode.py`(λ 0.1), RTX 2080 Ti | `results/rbd_20261101/` |

## 5장: 실험

### `tab:hallucination_all_models` 왼쪽: CHAIR 500

| 행 | 코드 | 결과 |
|---|---|---|
| Qwen2.5 REVIS†, RUDDER (BF16 eager), GrAInS (FP16) | `run_qwen_five_profile_suite.py` → `run_qwen_five_profile_worker.py`, 채점 `audit_qwen_five_profile_suite.py`, 방향 `fit_qwen_retained.py`, `extract_qwen_author_grains.py`, 원저자 함수 연결 `author_clone_bridge.py`·`qwen_author_compat.py`; GrAInS `run_qwen_grains_extra_workers.py`, `run_qwen_fp16_stable_workers.py` | `outputs/qwen25-vl7b-fiveprofiles-20261029-1793247386/original-vti-gpu1-final/chair/*.chair.official.json`, `metrics.csv` |
| Qwen2.5 VTI (공백 정규화 데모) | `scripts/rbd2/vti_fix.py`, `vti_chair.sh` → `vti_ws_run.py` | `R/runs/chair_test500_vtiws/`, 데모 `R/vti_fix/` |
| Qwen2.5 Base (TETHER 코드 경로) | 세트 `pb_make500.py`; rows 0–149 `pb_worker.sh`·`pb_ext.sh`·`pb_pool.sh` → `pb_join150.py`, rows 150–499 `pb_run500*.sh` → `pb_join500.py` → `pb_score500.sh`/`score_sets.py` | `R/runs/chair_test500_all/score.json` |
| **Qwen2.5 TETHER `L16a3b1`** | `vast-qwen-3080/na_run.sh` → `scripts/steer_vast/vdd_steer.py` (`--steer resid --layers 16 --alpha 3 --lam 1 --pen 0.5`, sdpa, RTX 3080) | `vast-qwen-3080/na/qwen/L16a3b1-0-500.jsonl`; 채점은 아래 절 |
| Gemma Base, REVIS†, RUDDER, VTI† (BF16) | `run_author_clone_suite.py` → `continue_author_clone_suite.py` → `continue_gpu23_suite.py`, 채점 `audit_author_clone_suite.py`; REVIS와 VTI는 `src/vlm_grounding/baselines.py` | `outputs/author-clone-rerun-20261027-165446/bf16-full500-20261029-125744/bf16_common/*.chair.official.json` |
| Gemma GrAInS (FP16) | `extract_author_grains.py`, `run_gemma_grains_fp16_v2.py` → `grains_fp16_worker_v2.py`, 채점 `audit_five_profile_suite_v2.py` | `outputs/gemma-grains-fp16-v2-20261029-1793246507/chair/` |
| Gemma TETHER `s24a1bl1p05` | `scripts/rbd2/gchair3.sh` (+ `gchair_pool.sh`, `gchair_collect.py`, `gchair_pick.py`, `gchair_join.py`, `score_sets.py`) | Gemma 행 전부를 같은 이미지로 다시 채점한 `R/runs/gchair_test500_tbl/score.json`, 캡션 `R/runs/gchair_test500_all/` |
| Qwen3 Base, TETHER `s20a1bl1p05` | `qchair3.sh` + `qchair_pool.sh`, `qchair_collect.py` | `R/runs/qchair_test500_all/score.json` |
| Qwen3 RUDDER, REVIS†(자체), VTI | `qwen3_bench.py`, `qwen3_run_baselines.sh`, `run_vti_gpu23.sh`, 보고 `analysis/qchair_report.py` | `R/runs/qchair_test500_tbl/` |
| Qwen3 GrAInS, 공식 REVIS(dev 열에 사용) | `qbl_chain.sh`: `extract_qwen3_grains.py`(FP32) → `qwen3_bl.py`(FP16 eager); `qbl_revis.sh` → `revis_official_q3.py`; 보고 `analysis/qbl_report.py` | `R/runs/qchair_bl2/`, 표 `R/logs/qbl.log` |

### Qwen2.5 TETHER 행 (`L16a3b1`, RTX 3080)

vast.ai RTX 3080 박스에 서버와 같은 경로로 코드와 데이터를 복제해서 500장을 한 번에 돌렸습니다.
복제한 것은 `src`, `scripts`, chair_test500 세트와 이미지, gray448, V_obj, visual-decay 요약입니다.
환경은 torch 2.8.0+cu128, transformers 5.20.2(서버 `.venv`와 같은 버전), Qwen2.5-VL-7B-Instruct `cc59489`입니다.

- 실행: `vast-qwen-3080/na_run.sh L16a3b1 0 500 --steer resid --layers 16 --alpha 3 --lam 1 --pen 0.5` → `scripts/steer_vast/vdd_steer.py`.
  2026-11-11 06:53–08:17 UTC에 돌렸고, 전체 인자는 `L16a3b1-0-500-status.json`에 있습니다.
- 결과: `vast-qwen-3080/na/qwen/L16a3b1-0-500.jsonl`(500행), `-status.json`, `.log`, `vast-qwen-3080/logs/na.log`
- 평균 생성 토큰은 500장 171.4, 처음 212장 166.3입니다.
- **500장 CHAIR 채점은 아직 이 브랜치에 없습니다.** 표의 CHAIR 값은 212장 시점에 서버 채점기(`na_vs.py`)로 낸 것입니다.
  CHAIR 채점기 포트가 `nltk`와 `pattern`을 써서 서버에서 채점해야 합니다.
- 같은 설정의 rows 0–199를 서버 A6000에서 먼저 돌렸고(CHAIR_s 23.5), 그 결과를 보고 이 설정을 정했습니다. 이 0–199행 실행은 넣지 않았습니다.

### `tab:hallucination_all_models` 오른쪽: CHAIR dev 100

`scripts/rbd2/cdev_chain.sh` + `cdev_prep.py`는 Base, RUDDER, GrAInS, VTI, TETHER(각 백본의 표 설정)를 세 백본에서 돌리고, `score_chair100.py`로 채점합니다.
REVIS는 `after_v7.sh` 1단계(`hat_pool.sh`가 큐 `queues/chair_dev100_revis.txt`를 처리) 뒤 `cdev_prep.py score`로 채점했습니다. Qwen2.5와 Gemma는 자체 구현이고, Qwen3는 공식 코드입니다.

- 결과: `R/runs/chair_dev100_{qwen,qwen3,gemma}_all/score.json`
- 표 셀: `R/logs/cdev_table_{qwen,qwen3,gemma}.tex`
- 로그: `R/logs/cdev.log`
- `_vdd` 폴더: TETHER 코드 경로 Base를 기준으로 한 채점입니다(주석 처리된 "Base, TETHER code path" 행).

`hat_pool.sh`는 HAT 실험의 작업자 스크립트인데, dev REVIS 행도 이것으로 돌렸기 때문에 넣었습니다.

### `tab:ablation_study`

**Qwen2.5**

- 코드: `abl_chain.sh` + `abl_prep.py` (`vdd4_decode.py`, `queues/abl_chair.txt`). 전체 TETHER `s16a2bl1p05`에서 한 요소씩 뺀 변형입니다.
- 결과: `R/runs/chair_abl500_all/score.json`(vdd Base 대비 CI), `R/runs/chair_abl500_vsfull/score.json`(Full 대비 CI), `R/logs/abl_chair.log`
- 5.1절 Controls 문단이 정의한 belief only 27.0 / 7.3 / 68.4, penalty only 29.8 / 8.0 / 69.8, steering only 31.6 / 8.3 / 69.8, constant schedule 27.0 / 7.7 / 68.3도 같은 파일에 있습니다.
  표에는 이 네 행이 없습니다.

**Gemma**

원고의 Gemma 두 행은 서버의 no-anchor steering 탐색(`outputs/steer-noanchor-20261110/gemma/`)에서 나온 500장 결과입니다.

- Full = `L24a2`: `--lam 0.5 --pen 0.25 --steer resid --layers 24 --alpha 2` → 31.8 / 6.21 / 63.3 / 377
- −steering = `ns`: 같은 belief, α 0 → 34.0 / 6.89 / 63.2 / 370
- 디코더는 `scripts/steer_vast/vdd_steer.py`입니다. 이 브랜치의 파일은 11-11 15:11 수정본이고, 두 실행은 그보다 먼저 돌았습니다.
- **캡션과 실행·채점 스크립트(`pool.sh`, `queue.txt`, `na_score.py`, `na_vs.py`, `PROTOCOL.md`)는 서버에만 있어서 아직 이 브랜치에 없습니다.**

### `tab:throne_full`

원고 캡션의 5,000장은 THRONE의 COCO val2017 이미지 5,000장 전체입니다. 코드는 두 부분입니다.

- rows 0–999: 아래 표 위쪽의 1,000장 실행입니다. 지금 표의 숫자는 이 가운데 rows 200–999에서 나왔습니다.
- rows 1000–4999: 아래 표 마지막 두 줄의 `t4k_*`입니다. 1,000장 실행과 같은 디코더와 인자로 나머지 4,000장을 돌리고, rows 0–999의 판정과 합쳐 5,000장을 채점합니다.

| 단계 | 코드 | 결과 |
|---|---|---|
| 세트, 이미지 | `pb_make_throne_full.py`, `pb_get_throne_images.py`, `t4k_prep.py sets` | `R/sets/throne_full.jsonl`(5,000장 순서), `throne_full1000.jsonl`, 표의 800장 `throne_t800.jsonl`, 나머지 4,000장 `throne_r4000.jsonl` |
| Qwen2.5 Base, RUDDER, GrAInS, VTI‖(원래 데모), REVIS(Base와 1,000/1,000 같은 캡션이라 표에서 뺌) | `pb_bl_chain.sh` → `pb_bl_pool.sh` → `baseline_bench.py` → `pb_bl_join.py`/`pb_bl_full_join.py` → `pb_bl_full_score.sh` | `R/runs/throne_full1000/` (캡션, `eval/*/` 판정) |
| Gemma Base, REVIS†, RUDDER, GrAInS, VTI†, "TETHER, first try." `s24a1bl2p1` | `gemma_chain.sh`(선택 `gemma_pick.py`) → `gemma_pool.sh`, `gemma_bench.py`, `gemma_ours_fix.sh`, `gemma_split.sh`, `gemma_join.py`, 판정 `gemma_bl_judge.sh` | `R/runs/gemma_throne1000/` |
| TETHER `cf_a3bl2p1` (두 백본) | `vdd4_setup.py`(V_obj) → `vdd4_chain.sh` → `vdd4_prep.py`·`vdd4_pool.sh`·`vdd4_decode.py`, `vdd4_qwen_resume.sh`, 채점 `vdd4_score.py` | `R/runs/{qwen,gemma}_cf_test_all/`, 표 `R/logs/v4_{qwen,gemma}_t800_table.txt` (원고의 2-judge 열과 같음) |
| 판정 (Flan-T5 base, large, 임계값 6) | `pb_throne_judge.py`(`RBD2_JUDGES`), `score_throne.py`, `pb_throne_partial.py` | `R/diag/*_2judges/` (공식 채점기에 넣은 base·large 판정) |
| xl 열 (base, large, xl, 임계값 9) | `after_v7.sh` 2단계 `xl_pool.sh` → `xl_judge.py` → `xl_score.py` | `R/runs/*/eval/*/combined/*flan-t5-xl.json`, 표 `R/logs/throne_xl_table.txt` |
| ALL, CLS 지표 | `analysis/throne_all_metrics.sh`, `throne_official_scorer.sh`(공식 `throne_score_aqa.py`), `throne_classwise_ci.sh` | 표에 직접 |
| 5,000장: 나머지 4,000장의 디코딩과 판정 | `t4k_chain.sh` → `t4k_prep.py`(sets, queues, repro, join, status) → `t4k_pool.sh`(디코딩) → `t4k_jpool.sh` → `t4k_judge.py`(1,000장 블록마다 판정기 하나씩) | 큐 `scripts/rbd2/queues/t4k_{repro,decode,judge}.txt`. 캡션과 판정은 실행이 끝나지 않아서 아직 없습니다 |
| 5,000장 채점 | `t4k_score.py` (THRONE venv, CPU) | 위와 같음 |

`v4_qwen_t800_table.txt`와 `throne_xl_table.txt`에는 Qwen2.5의 첫 번째 설정 `s16a1bl2p1` 행이 있습니다. 원고에서 이 행은 주석 처리되어 있어서 캡션은 넣지 않았습니다. `t4k_*`는 이 설정도 4,000장에서 돌리고 채점합니다.

`t4k_*`의 내용:

- 실행 단위는 백본마다 7개 실행(Base, REVIS, RUDDER, GrAInS, VTI, 첫 번째 설정, TETHER `cf_a3bl2p1`) × 1,000장 블록 4개입니다. 블록 순서로 돌기 때문에 블록 하나가 끝나면 모든 방법을 그 블록까지 채점할 수 있습니다.
- 디코딩 전에 방법마다 저장된 A6000 캡션 2개를 다시 만들어 비교합니다(`t4k_prep.py repro`). 하나라도 다르면 모든 작업이 멈춥니다. 11-09 실행에서는 28개가 모두 같았습니다.
- `t4k_score.py`는 네 범위를 냅니다.
  - rows 200–999: 확인용이고, 지금 표와 같아야 합니다.
  - rows 200–N: 보고용이고, 설정 선택에 쓴 rows 0–199를 뺍니다.
  - rows 0–N
  - rows 1000–N
- N은 1,000 + 1,000 × (판정이 끝난 블록 수)이고, `--blocks K`로 정할 수도 있습니다.
- 2-judge 임계값 6(본 열)과 3-judge 임계값 9(xl 열)를 둘 다 내고, CI는 Base 대비 paired bootstrap 2,000회입니다.
- 11-09에 `--blocks 0`으로 `tab:throne_full`의 모든 숫자(본 열, Tok, xl)를 재현했습니다. 11-12 중간 채점의 rows 200–999 2-judge 행도 원고 표와 같습니다.
- Gemma 첫 번째 설정 `s24a1bl2p1`은 rows 4940–4999에서 골랐습니다. 이 60장은 5,000장 채점(rows 200–N)에 들어갑니다.
- 큐 파일은 `t4k_prep.py queues`가 만든 원본 그대로이고, 다시 만들어 같은 것을 확인했습니다. 서버의 지금 `t4k_decode.txt`·`t4k_judge.txt`는 고친 판입니다. 11-10에 사용자 지시로 우리 방법(첫 번째 설정, TETHER) 줄을 빼고 베이스라인을 먼저 돌리게 했습니다.
- 진행 상황(11-12 13:20 중간 채점 기준):
  - rows 1000–1999는 14개 실행 모두 디코딩과 base·large 판정이 끝났습니다. xl 판정은 5개 실행(Qwen REVIS·RUDDER, Gemma REVIS·GrAInS·VTI)에 남아 있습니다.
  - rows 2000–4999는 끝나지 않았습니다.
- rows 1000–1999의 일부는 vast 박스(RTX 3080, RTX 2080 Ti)에서 디코딩했습니다. 행마다 GPU가 기록되어 있습니다. 같은 설정이라도 GPU 종류가 다르면 greedy 캡션이 달라집니다.
- 블록 0에 GPU를 더 붙인 배치 스크립트는 넣지 않았습니다(`t4k_after_cdm.sh`, `t4k_b0_after.sh`, `t4k_jpool_q.sh`, vast 박스의 `t4k_v*.sh`, `judge_v.sh`). 같은 큐 단위를 다른 GPU에 나눌 뿐이고 디코더와 인자는 같습니다.

### VCD

기여 문단의 "and VCD"에 해당하는 실행입니다. VCD를 CHAIR 500에서 Qwen2.5와 Gemma로 돌렸고, Qwen3은 사용자 지시로 중단했습니다.
설정은 공식 VCD 식(`d6568ff`), α 1, β 0.1, noise step 500, greedy이고, `vdd_decode.py`와 같은 lockstep 경로를 씁니다.

- 코드: `scripts/rbd2/cdm_decode.py`(VCD, M3ID), `cdm_check.py`, `cdm_diag.py`, `cdm_pool.sh`, `cdm_chain.sh`, `cdm_join.py`
- **캡션과 채점 결과(`outputs/cdm-20261109/`)는 서버에만 있어서 아직 이 브랜치에 없습니다.**
  같이 돌린 M3ID는 원고에 없어서 넣지 않습니다.

## 부록 C, D

| 원고 | 코드 | 결과 |
|---|---|---|
| `tab:baseline_config` | `src/vlm_grounding/baselines.py`(REVIS, Gemma VTI), `scripts/author_clone_bridge.py`, `qwen_author_compat.py`, `scripts/rbd2/vti_fix.py` | — |
| REVIS: 처음 50장 중 46장, CHAIR 500장 중 420장, THRONE 전부가 Base와 같은 캡션 | — (파일 비교로 확인) | `outputs/qwen-vti-alpha-sweep-20261030/score-pilot-chair50/{base,revis}.captions.json`, `outputs/qwen25-vl7b-fiveprofiles-*/original-vti-gpu1-final/chair/{base,revis}.jsonl`, `R/runs/throne_full1000/{base,revis}-0-1000.jsonl` |
| `tab:vti_pilot` (VTI 강도, 처음 50장) | α 0.9: `prepare_original_vti.py`, `run_qwen_original_vti.py`, `qwen_vti_chair_*.py`; α 0.45·0.225: `qwen_vti_alpha_sweep.py` → `score_vti_alpha_sweep.py` | `outputs/qwen-vti-alpha-sweep-20261030/score-pilot-chair50/score.json` (α 0.9의 500장 캡션은 `original-vti-gpu1-final/chair/vti_author.jsonl`) |
| VTI 데모 줄바꿈 88/100 vs 19/100, logit lens L24–28 | `scripts/rbd2/vti_fix.py` | `R/vti_fix/`, `R/logs/vti_fix.log` |
| VTI dev 7점 grid ("recall을 Base −2 안에 둔 강도 없음") | `scripts/rbd2/vti_chair.sh` | `R/runs/chair_dev60_vtiws/`, `R/logs/vti_chair_dev.log`, 테스트 `R/logs/vti_chair_test.log` |
| `tab:cases`, 사례 1–4 | `scripts/rbd2/analysis/case_study_extract.sh` (같은 CHAIR scorer로 방법별 객체 추출); Yes-No margin은 `outputs/caption-selfcheck-20261030/qwen-base-rows.jsonl` | 위 CHAIR 500 캡션 (TETHER = `R/runs/chair_test500_all/s16a2bl1p05-0-500.jsonl`) |
| `fig:case_examples` (그림 3장) | — | 그림 파일과 이미지 ID가 이 PC에 없어서 출처를 확인하지 못했습니다 |

## 넣지 않은 것

- **원고에 없는 실험:** HAT + ALOHa(v7, 집계 수정 포함), AMBER 생성형 1,004장, 예전 Gemma CHAIR ablation(`gabl`),
  self-verification 디코딩, MMVP, MMHal-Bench, five-profile POPE, AMBER vdd3, Qwen3 탐색 `rbd5`·`rbd6`, Gemma CHAIR 재선택 `gchair2`, VTI-ws THRONE 100장,
  anchor steering(`full_an16_r03`)과 그 ablation, VCD와 같이 돌린 M3ID, HAT 파일럿, recall 분석, 진단 스크립트(`diag_*`, `pb_diag_*`)
- **초기 탐색과 설정 선택용 실행:**
  - 두 번째 변형의 초기 탐색(`tab:dev2`): 클라우드 박스 `scripts/rbd2/cloud_dev/`와 `R/runs/chair_dev60/`, `amber_dev60/`에 있었습니다.
  - CHAIR 테스트 60·120·150장 sweep, Gemma와 Qwen3의 rows 0–59 후보, THRONE rows 0–199 후보(`*_cf_t200_all`), Gemma THRONE dev60 후보
  - Gemma CHAIR의 첫 테스트 설정 `s24a2bl1p05`, no-anchor steering 탐색의 다른 후보
  - 이 실행들은 이전 브랜치 `paper-code-results-20261109-draft3`과 서버에 남아 있습니다.
- **원고에서 주석 처리된 행:** THRONE Qwen2.5 첫 번째 설정 `s16a1bl2p1`의 캡션과 판정(rows 0–999)
- **중간 산출물:** 부분 실행 폴더, 스모크 테스트, 원래 VTI 실행 폴더의 샤드·장애 기록. 원래 VTI 실행 폴더에서는 `protocol.json`, `demo-manifest.json`만 남겼습니다.
- **첫 번째 변형 dev의 기타 결과:** Gemma, dev 베이스라인, 부분 채점 폴더
- **코드:** 위 실험의 스크립트(`hat_*`, `pb_amber_*`, `gabl_*`, `vdd7_decode.py`, `cloud_dev/`, `sweep_decode.py`, `rbd_offline.py`, orange 분석 등)와 그 큐.
  `pb_worker.sh`의 sweep 모드는 빠진 `sweep_decode.py`를 부릅니다.

옛 베이스라인 파이프라인(Gemma 5방법, Qwen five-profile)은 실행할 때 서로의 소스 해시를 검사합니다. 그래서 그 검사 목록에 있는 스크립트는 모두 남겼습니다.

## 원저자 코드와 채점기 (넣지 않음)

| 이름 | 고정 버전 |
|---|---|
| RUDDER | Akko000/RUDDER-Residual-Update-Directed-DEcoding-Regulation- `bd149d8` |
| GrAInS | duykhuongnguyen/GrAInS `8ad78db` |
| VTI | shengliu66/VTI `bb5dfee` |
| REVIS (공식, Qwen3 dev 행) | antgroup/Revis `75f4d29` |
| VCD | DAMO-NLP-SG/VCD `d6568ff` |
| Jacobian lens | 박스의 `tools/jacobian-lens`, 렌즈 `JLensVL-lenses/lens_qwen35_4b_final.pt` |
| CHAIR | Python 3 포트 `4087a26` (서버 `upstream/chair/chair.py`, sha256 `98728cd2…`) |
| THRONE | amazon-science/THRONE `45badbb7` (one-click.sh 2–3단계) |
| AMBER (첫 번째 변형 파일럿만) | 공식 생성형 평가 (`inference.py`, spaCy `en_core_web_lg`) |

모델 revision: Qwen2.5-VL-7B-Instruct `cc59489`, Gemma 4 E4B-it `ee0ef60`, Qwen3-VL-8B-Instruct `0c351dd` (`configs/*.json`), Qwen3.5-4B(Jacobian lens 파일럿).
