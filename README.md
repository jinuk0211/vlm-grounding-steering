# 논문 코드와 결과 (2026-11-08)

논문 초안(TETHER, 2026-11-05)의 표와 본문 수치를 만든 **코드와 그 결과 파일**만 모은 브랜치입니다.
코드만 담았던 `paper-tables-code-20261104` 위에 이어서 만들었습니다. 논문에서 방법 이름은 TETHER이지만, 코드와 파일 이름에는 예전 이름인 RBD가 그대로 남아 있습니다
(첫 번째 변형 `rbd_*`, 두 번째 변형 `vdd*_decode.py`).

- 파일은 실험 서버(`/home/project-b/LJW/vlm-grounding-steering`)에서 2026-11-08에 그대로 가져왔고 경로도 서버와 같습니다.
  예외: 큐 파일은 서버의 `outputs/rbd2-20261102/queues/`에 있지만 여기서는 `scripts/rbd2/queues/`에 둡니다.
- 결과는 텍스트 파일만 넣었습니다: 캡션(jsonl), 채점 결과(json), THRONE 판정 파일, 로그. 모델 가중치, 이미지, steering 벡터(`.pt`),
  활성값 덤프(`.npz` 등)와 50 MB가 넘는 파일은 없습니다.
- 원저자 코드(RUDDER, GrAInS, VTI, REVIS 공식 코드, CHAIR·AMBER·THRONE·ALOHa 채점기)는 넣지 않았습니다. 고정 버전은 맨 아래에 있습니다.
- 스크립트는 서버 경로, `.venv`, HF 캐시, GPU UUID, 소스 해시 검사를 전제로 하므로 다른 머신에서 그대로 돌아가지 않습니다.
- 캡션에 나오는 이메일 주소는 COCO 사진의 워터마크를 모델이 그대로 옮겨 적은 것입니다.

경로 약칭: `R/` = `outputs/rbd2-20261102/`. 설정 태그 `s{L}a{α}bl{λ}p{π}` = `--layer L --alpha α --lam λ --pen π`
(`p05` = 0.5), `mainMed` = `s16a1bl1p05` + `--norm median`, `cf_` = 색 단어를 뺀 V_obj(THRONE 최종 설정). 태그별 인자는 `scripts/rbd2/queues/*.txt`에 있습니다.

## 감사 (Sec. Auditing the Generation–Verification Gap, 부록 Mechanistic Analysis)

| 표·본문 | 코드 | 결과 |
|---|---|---|
| `tab:gap` 생성–검증 격차 | `scripts/caption_selfcheck.py` → `caption_selfcheck_analyze.py` | `outputs/caption-selfcheck-20261030/{qwen,gemma}-base-analysis.json` (객체별 행 `*-base-rows.jsonl`) |
| POPE 본문 (margin 0.940 vs probe 0.880, FN 13/379, 17/303) | `read_gap_collect.py`, `read_gap_collect_v2.py` → `read_gap_analyze*.py`, `read_gap_complement.py` | `outputs/read-gap-e0-20261030/analysis-*.json` |
| `tab:decay`, `tab:findings` 위쪽(gray KL, attention, key blocking), `tab:schedule` | `visual_decay_run.sh` (`visual_decay_collect.py` → `visual_decay_analyze.py` → `visual_decay_plot.py`) | `outputs/visual-decay-20261030/*-visual-decay-summary.json` |
| grad×input 이미지 비중(`tab:findings`, `tab:objtok`), 쓸 때 0.753 vs 물을 때 0.749 | `visual_reliance.py` → `reliance_analyze.py`, `reliance_analyze_v2.py` | `outputs/visual-reliance-20261030/qwen-reliance-summary.json` |
| 언급 직전 이미지 attention (AUROC 0.64, 0.55) | `verify_probe_attn.py` | `outputs/query-anchor-step0-20261030/*-attn-summary.json` |
| `tab:steer`, `tab:findings` attention boost·probe/random shift, probe 학습 | `verify_probe_collect.py` → `verify_probe_analyze.py`, `verify_probe_e1.py` | `outputs/verify-probe-20261030/*-probe-summary.json`, `*-e1-summary.json` |
| `tab:lens_img`, `tab:lens_cap`, `tab:ranking`의 lens 행 | `logit_lens.py` | `outputs/logit-lens-20261104/{qwen,gemma}-lens-summary.json`, `*-topk-summary.json`, `*-lens-report.html` |
| CPU 재검토(11-05): `tab:position`, `tab:ranking`, `tab:objtok` 첫·반복 언급, `app:position`, `app:cooccur`, addCriterion 77개 제외 재확인 | `outputs/audit-recheck-20261105/audit_recheck.py`, `audit_recheck2.py`(첫·반복 언급, grad×input, 공출현, attention), `audit_recheck3.py`(거리×위치 lens, 77개 제외 E1·step 0) | 같은 폴더 `{qwen,gemma}.json`, `-2.json`, `-3.json` |
| `tab:probe` fold별 AUROC | `outputs/audit-recheck-20261105/probe_foldwise.py` | `{qwen,gemma}-probe-foldwise.json` |

보조 모듈 `verified_decoding_v2.py`, `verified_decoding_v3.py`(멘션 위치), `read_gap_analyze_v3.py`(AUROC), `read_gap_collect_v2.py`(모델 로더),
`open_vocab.py`(WordNet 명사 필터)는 위 스크립트가 import해서 들어 있습니다. self-verification 디코딩 실험 자체는 논문에 없습니다.

## 첫 번째 변형 (확률 belief): `tab:dev`, `sec:offline`, `sec:dev_pilot`

| 표·본문 | 코드 | 결과 |
|---|---|---|
| `tab:dev` (dev 100장, Qwen) | 세트 `make_probe_set.py` → `make_dev_chair.py`, 모드 방향 `rbd_dvec.py`, 디코딩 `rbd_decode.py`(`rbd_common.py`), 채점 `score_dev.py` | `outputs/rbd-20261030/dev/` |
| `sec:offline` (log p 0.667·0.694, 첫 변형 0.55·0.52) | `rbd_collect.py` → `rbd_offline.py` | `outputs/rbd-20261030/collect/{qwen,gemma}-offline.json` |
| V_obj | `rbd_collect.py` | `outputs/rbd-20261030/collect/*-vobj2.json` (색 단어 제외판 `*-vobj2cf.json`은 `scripts/rbd2/vdd4_setup.py`) |
| r(t) (`tab:schedule`) | `rbd_common.r_curve`: visual-decay 요약의 gray-KL 곡선을 첫 구간 1로 정규화 | — |

## 두 번째 변형 (시각 증거, TETHER) 개발 세트: `tab:dev2`

- 디코더 `scripts/rbd2/vdd_decode.py`. 서버 버전은 클라우드 버전에 `VDD_GRAY`, `--norm median`, `--bpos`를 더한 것이고,
  기본값에서는 클라우드 버전과 같은 계산입니다. THRONE 최종 설정과 ablation은 V_obj 파일을 바꿀 수 있는 `vdd4_decode.py`를 씁니다.
- dev60은 클라우드 박스에서 돌렸습니다: `scripts/rbd2/cloud_dev/` (`prep.py`, `prep2.py` → `worker.sh`·`vrun.sh`·`chain.sh`, 큐 `cloud_dev/queues/`).
- 채점 `scripts/rbd2/score_sets.py` (CHAIR: 캐시된 원본 scorer, AMBER: `score_amber100.py`의 공식 매칭), paired bootstrap 2,000회.
- 결과: `R/runs/chair_dev60/`, `R/runs/amber_dev60/`, `R/runs_dev_cloud/amber_dev60/`. VTI 강도 dev: `R/runs/chair_dev60_vtiws/` (`vti_chair.sh`).

## CHAIR 500: `tab:hallucination_all_models` 왼쪽

| 행 | 코드 | 결과 |
|---|---|---|
| Qwen2.5 Base·REVIS†·RUDDER (eager BF16), GrAInS (FP16) | `run_qwen_five_profile_suite.py` → `run_qwen_five_profile_worker.py`, 채점 `audit_qwen_five_profile_suite.py`, 방향 `fit_qwen_retained.py`, `extract_qwen_author_grains.py`, 원저자 함수 연결 `author_clone_bridge.py`·`qwen_author_compat.py`; GrAInS `run_qwen_grains_extra_workers.py`, `run_qwen_fp16_stable_workers.py`; 집계 `qwen_vti_gpu1.py` 계열 | `outputs/qwen25-vl7b-fiveprofiles-20261029-1793247386/original-vti-gpu1-final/chair/*.chair.official.json`, `report.json`, `metrics.csv` |
| Qwen2.5 VTI¶ (공백 정규화 데모) | `scripts/rbd2/vti_fix.py`, `vti_chair.sh` → `vti_ws_run.py` | `R/runs/chair_test500_vtiws/`, 데모 `R/vti_fix/` |
| Qwen2.5 Base (RBD code path)·RBD `s16a2bl1p05` | 세트 `pb_make500.py`; 0–149행 `pb_worker.sh`·`pb_ext.sh`·`pb_pool.sh` → `pb_join150.py`, 150–499행 `pb_run500*.sh` → `pb_join500.py` → `pb_score500.sh`/`score_sets.py` | `R/runs/chair_test500_all/` (`score.json`). 설정을 비교한 첫 150장: `R/runs/chair_test60_sweep/`, `R/runs/chair_test150_all/` |
| Gemma Base·REVIS†·RUDDER·VTI† (BF16) | `run_author_clone_suite.py` → `continue_author_clone_suite.py` → `continue_gpu23_suite.py`, 채점 `audit_author_clone_suite.py`; REVIS·VTI 자체 구현 `src/vlm_grounding/baselines.py` | `outputs/author-clone-rerun-20261027-165446/bf16-full500-20261029-125744/bf16_common/*.chair.official.json` |
| Gemma GrAInS (FP16) | `extract_author_grains.py`, `run_gemma_grains_fp16_v2.py` → `grains_fp16_worker_v2.py`, 채점 `audit_five_profile_suite_v2.py` | `outputs/gemma-grains-fp16-v2-20261029-1793246507/chair/` |
| Gemma Base (RBD code path)·RBD `s24a1bl1p05` | `scripts/rbd2/gchair3.sh` (테스트 0–59행에서 8개 후보 → `gchair_pick.py` 규칙 → 60–499행) + `gchair_pool.sh`, `gchair_collect.py`, `gchair_join.py`, `score_sets.py` | `R/runs/gchair_test500_all/`, 표의 베이스라인 행과 같은 이미지 비교 `R/runs/gchair_test500_tbl/`, 선택 `R/runs/gchair_t60/` |
| Qwen3 Base (RBD code path)·RBD `s20a1bl1p05` | `qchair3.sh` (0–59행 9개 후보 → 자동 선택 → 60–499행) + `qchair_pool.sh`, `qchair_collect.py` | `R/runs/qchair_test500_all/`, 선택 `R/runs/qchair_t60/` |
| Qwen3 RUDDER·REVIS†(자체)·VTI | `qwen3_bench.py`, `qwen3_run_baselines.sh`, `run_vti_gpu23.sh`, 보고 `analysis/qchair_report.py` | `R/runs/qchair_test500_tbl/` |
| Qwen3 GrAInS, 공식 REVIS | `qbl_chain.sh`: `extract_qwen3_grains.py`(FP32) → `qwen3_bl.py`(FP16 eager); `qbl_revis.sh` → `revis_official_q3.py`; 보고 `analysis/qbl_report.py` | `R/runs/qchair_bl2/`, 표 `R/logs/qbl.log` |

## CHAIR dev 100 (COCO train2014): `tab:hallucination_all_models` 오른쪽

`scripts/rbd2/cdev_chain.sh` + `cdev_prep.py`: Base, RUDDER, GrAInS, VTI, RBD(각 백본의 표 설정)를 Qwen2.5·Qwen3·Gemma에서 실행, 채점 `score_chair100.py`.
결과 `R/runs/chair_dev100_{qwen,qwen3,gemma}_all/score.json`, 표 셀 `R/logs/cdev_table_{qwen,qwen3,gemma}.tex`, 로그 `R/logs/cdev.log`.

## AMBER 생성형 1,004장

| 행 | 코드 | 결과 |
|---|---|---|
| 베이스라인 (bench Base, REVIS, RUDDER, GrAInS, VTI-ws) | `pb_amber_bl.sh` → `pb_amber_pool.sh` → `baseline_bench.py` (`queues/amber_full.txt`), 채점 `amber_full_score.sh` | `R/runs/amber_full1004_all/` |
| Base (RBD code path)·RBD `mainMed` | 0–499행은 500장 실행, 나머지 `pb_amber_ours1004.sh` (`queues/amber_ours1004.txt`) → `score_sets.py` | `R/runs/amber_full1004_ours_all/`, 설정 비교 150장 `R/runs/amber_test150_all/` |

## HAT + ALOHa (385장)

`hat_setup.sh`(ALOHa 공식 코드 환경) → `hat_gpu3.sh`/`hat_chain.sh` (재현 확인 → Qwen → Gemma), 작업자 `hat_pool.sh`·`hat_join.sh`,
세트·큐·선택 규칙 `hat_prep.py`, 객체 파싱 `hat_parse.py`(공식 GPT 파서 대신 Qwen3-VL-8B + 공식 프롬프트), 채점 `hat_aloha.py`(공식 `ALOHa.__call__`, MPNet, DETR), `hat_common.py`, `hat_smoke.sh`.
선택은 0–59행, 보고는 60–384행입니다. Gemma는 선택 규칙을 통과한 RBD 후보가 없어 RBD를 끝까지 돌리지 않았습니다.
결과 `R/runs/hat385_{qwen,gemma}/`, 점수 `R/hat385/score-{qwen,gemma}-*.json`, 선택 `R/hat385/pick-qwen.txt`, 파싱 결과 `R/hat_parse/`, 표 `R/logs/hat385.log`.

## Ablation (Qwen2.5 CHAIR 500)

`abl_chain.sh` + `abl_prep.py` (`vdd4_decode.py`, 큐 `queues/abl_chair.txt`): 전체 RBD `s16a2bl1p05`에서 한 요소씩 빼거나 하나만 남긴 7개 변형.
결과 `R/runs/chair_abl500_all/score.json`(Base 대비), `R/runs/chair_abl500_vsfull/score.json`(전체 RBD 대비), 로그 `R/logs/abl_chair.log`.

## THRONE: `tab:throne`, `tab:throne_full`

| 단계 | 코드 | 결과 |
|---|---|---|
| 세트·이미지 | `pb_make_throne_full.py`, `pb_get_throne_images.py` | `R/sets/throne_full1000.jsonl`, `throne_t200.jsonl`, `throne_t800.jsonl` |
| Qwen 베이스라인 1,000장 | `pb_bl_chain.sh` → `pb_bl_pool.sh` → `baseline_bench.py` → `pb_bl_join.py`/`pb_bl_full_join.py` → `pb_bl_full_score.sh` | `R/runs/throne_full1000/` (캡션, `eval/*/` 판정) |
| Qwen 첫 번째 설정 `s16a1bl2p1` | 0–499행 `throne_test500`, 500–999행 `pb_full1000_ours.sh` → `pb_join_full1000_ours.py` → `pb_full1000_finish.sh` | `R/runs/throne_test500_all/`, `R/runs/throne_full1000_ours/`, 선택에 쓴 150장 `R/runs/throne_test150_all/` |
| Gemma 베이스라인·첫 번째 설정 `s24a1bl2p1` | `gemma_chain.sh`(dev 선택 `gemma_pick.py`) → `gemma_pool.sh`, `gemma_bench.py`, `gemma_ours_fix.sh`, `gemma_split.sh`, `gemma_join.py`, 판정 `gemma_bl_judge.sh` | `R/runs/gemma_throne1000/`, dev `R/runs/gemma_throne_dev60/` |
| 최종 설정 `cf_a3bl2p1` (색 단어 제외 V_obj) | `vdd4_setup.py`(V_obj) → `vdd4_chain.sh` → `vdd4_prep.py`·`vdd4_pool.sh`·`vdd4_decode.py`, `vdd4_qwen_resume.sh`, 채점 `vdd4_score.py`. 테스트 0–199행에서 선택, 200–999행에서 보고 | 선택 `R/runs/{qwen,gemma}_cf_t200_all/`, 1,000장 `R/runs/{qwen,gemma}_cf_test_all/`, 선택 결과 `R/runs/*_cf_pick.txt`, 표 `R/logs/v4_*_table.txt` |
| 판정 (Flan-T5 base·large, 임계값 6) | `pb_throne_judge.py`(`RBD2_JUDGES`), `score_throne.py`, `pb_throne_partial.py` | 표에 쓴 2-judge 파일 `R/diag/*_2judges/` |
| ALL·CLS 지표, CI, orange 분석 | `analysis/throne_all_metrics.sh`, `throne_official_scorer.sh`(공식 `throne_score_aqa.py`), `throne_classwise_ci.sh`, `throne_orange.sh`, `throne_without_orange.sh`, `throne_breakdown.sh` | (인라인 실행, 출력은 표에 직접) |

`analysis/`의 THRONE 스크립트는 원래 세션에서 인라인으로 서버에 보내 실행한 것을 세션 기록에서 복원했습니다.

## 부록

| 표 | 코드 | 결과 |
|---|---|---|
| `tab:vti_pilot` (Qwen 원저자 VTI 강도, 첫 50장) | α=0.9: `prepare_original_vti.py`, `run_qwen_original_vti.py`, `qwen_vti_chair_gpu23.py` 계열; α=0.45·0.225: `qwen_vti_alpha_sweep.py` → `score_vti_alpha_sweep.py` | `outputs/qwen-original-vti-1793260357/`, `outputs/qwen-vti-alpha-sweep-20261030/score-pilot-chair50/` |
| `tab:cases` | `scripts/rbd2/analysis/case_study_extract.sh` (같은 CHAIR scorer로 방법별 객체 추출) | 위 CHAIR 500 캡션 |
| `tab:baseline_config` | `src/vlm_grounding/baselines.py`, `author_clone_bridge.py`, `qwen_author_compat.py`, `rbd2/vti_fix.py`, `revis_official_q3.py` | — |

공통: 세트 `R/sets/`, 로그 `R/logs/`(논문에 없는 실험의 로그는 뺐음), 큐 `scripts/rbd2/queues/`.

## 넣지 않은 것

- 논문에 없는 실험: self-verification 디코딩, MMVP, MMHal-Bench(표가 모두 `--`), five-profile POPE 결과, AMBER vdd3 변형, Qwen3 탐색 rbd5·rbd6,
  Gemma CHAIR 재선택 `gchair2.sh`, VTI-ws THRONE 100장, 진단 스크립트(`diag_*`, `pb_diag_*`), recall 분석, HAT 파일럿.
- 끝나지 않은 실험: Gemma CHAIR ablation (`gabl_*`). 2026-11-06 09:31에 작업자가 모두 멈춰 3,440행 중 1,841행만 있습니다.
- 중간 산출물: 부분 실행 폴더(`*_parts`, 행 구간별 폴더), 60·120장 중간 채점, 스모크 테스트 결과.

옛 베이스라인 파이프라인(Gemma 5방법, Qwen five-profile)은 실행할 때 서로의 소스 해시를 검사하므로, 그 검사 목록에 있는 파일은 모두 포함했습니다.

## 원저자 코드와 채점기 (포함하지 않음)

| 이름 | 고정 버전 |
|---|---|
| RUDDER | Akko000/RUDDER-Residual-Update-Directed-DEcoding-Regulation- `bd149d8` |
| GrAInS | duykhuongnguyen/GrAInS `8ad78db` |
| VTI | shengliu66/VTI `bb5dfee` |
| REVIS (공식, Qwen3 행) | antgroup/Revis `75f4d29` |
| CHAIR | Python 3 포트, `4087a26` (서버 `upstream/chair/chair.py`) |
| THRONE | amazon-science/THRONE `45badbb7` (one-click.sh 2–3단계) |
| AMBER | 공식 생성형 평가 (`inference.py`, spaCy `en_core_web_lg`) |
| ALOHa | DavidMChan/aloha `7da6b90` |

모델 revision: Qwen2.5-VL-7B-Instruct `cc59489`, Gemma 4 E4B-it `ee0ef60`, Qwen3-VL-8B-Instruct (`configs/*.json`).
