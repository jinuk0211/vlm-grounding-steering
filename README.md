# 논문 코드와 결과: 초안 3 (2026-11-09)

`CVPR_VL_model_steering (3).pdf`(2026-11-09 18:13)의 표, 그림, 본문 수치를 만든 **코드와 결과 파일**을 모은 브랜치입니다.
`paper-code-results-20261109` 위에 다음을 더했습니다.

- 초안 3의 표·그림 번호로 다시 정리한 대응표 (아래), 그리고 논문 수치를 결과 파일과 맞춰 본 결과
- THRONE 5,000장 실행 코드 `scripts/rbd2/t4k_*` (11-09 17:31 시작, 진행 중)
- 그림 1 a5 Jacobian-lens 파일럿의 코드와 결과 `vast-tether-pilot/` (vast.ai 박스에서 가져옴)
- 서버에 파일로 남지 않고 세션에서 인라인으로 실행한 감사(audit) 분석 8개 `scripts/audit_inline/` (출력 `outputs/audit-recheck-20261105/inline/`)
- 부록 B.4 첫 번째 변형 파일럿(CHAIR 테스트 앞 100장, AMBER 1–100) `results/rbd_20261101/` (`rbd-round2-20261101` 브랜치 `1d2d676`에서 복사)

초안 3에 없는 실험(HAT + ALOHa, AMBER 1,004장, Gemma ablation)도 지우지 않고 남겼습니다. 맨 아래 절에 따로 적었습니다.

논문의 방법 이름은 TETHER이지만 코드와 파일 이름에는 예전 이름 RBD가 남아 있습니다
(첫 번째 변형 `rbd_*`, 두 번째 변형 = TETHER `vdd*_decode.py`).

- 서버(`/home/project-b/LJW/vlm-grounding-steering`) 파일은 2026-11-08, 11-09에 그대로 가져왔고 경로도 서버와 같습니다.
  예외: 큐 파일은 서버의 `outputs/rbd2-20261102/queues/`에 있지만 여기서는 `scripts/rbd2/queues/`에 둡니다.
  `t4k_*`는 11-09 17:3x에 서버에 올린 로컬 사본입니다(이번에는 서버에 접속하지 않았습니다).
- 결과는 텍스트 파일만 넣었습니다: 캡션(jsonl), 채점 결과(json), THRONE 판정 파일, 로그. 모델 가중치, 이미지, steering 벡터(`.pt`),
  활성값 덤프(`.npz` 등)와 50 MB가 넘는 파일은 없습니다.
- 원저자 코드(RUDDER, GrAInS, VTI, REVIS 공식 코드, Jacobian lens, CHAIR·AMBER·THRONE·ALOHa 채점기)는 넣지 않았습니다. 고정 버전은 맨 아래에 있습니다.
- 스크립트는 서버 경로, `.venv`, HF 캐시, GPU UUID, 소스 해시 검사를 전제로 하므로 다른 머신에서 그대로 돌아가지 않습니다.
- 캡션에 나오는 이메일 주소는 COCO 사진의 워터마크를 모델이 그대로 옮겨 적은 것입니다.

경로 약칭: `R/` = `outputs/rbd2-20261102/`. 설정 태그 `s{L}a{α}bl{λ}p{π}` = `--layer L --alpha α --lam λ --pen π`
(`p05` = 0.5), `cf_` = 색 단어를 뺀 V_obj(THRONE 최종 설정). 태그별 인자는 `scripts/rbd2/queues/*.txt`에 있습니다.

## 논문과 맞춰 본 결과

표 1, 5, 6, 7, B.2, C.2, 4.2절의 가중치(96%, 35%, 24%)와 5.3·5.4절 본문의 차이값을 결과 파일과 한 칸씩 비교했습니다.
아래 항목을 빼면 모두 일치합니다.

### 논문을 고쳐야 할 곳

| # | 위치 | 논문 | 결과 파일 | 출처 |
|---|---|---|---|---|
| 1 | 표 5, 초록, 서론 | Qwen3-VL TETHER CHAIRs 37.0 ("49.2 to 37.0") | **41.8** (CHAIRi 7.44, recall 75.2, Tok 431은 맞음). 37.0은 class orange를 뺀 값 | `R/runs/qchair_test500_all/score.json`, `R/logs/qbl.log` |
| 2 | 표 5, 5.3절 | Qwen3-VL RUDDER 47.5 / 9.14 / 76.6 / 396 | **47.8 / 9.14 / 76.7 / 396**. 47.5는 440장 중간값. 본문 "-1.7 to +3.0"은 -1.4 to +3.0 | `R/logs/qbl.log` |
| 3 | 표 5, 부록 C | Qwen3-VL REVIS: 테스트 52.2, dev 48.0 | 테스트 52.2는 자체 구현(`baselines.py`), dev 48.0은 공식 코드(`qwen3_bl.py revis_official`). 공식 코드 테스트 값은 50.2 / 9.87 / 76.9 / 389 | `R/logs/qbl.log`, `R/logs/cdev_table_qwen3.tex` |
| 4 | 표 5 | Qwen2.5-VL Base Tok 205 | Base 행의 30.4 / 8.40 / 69.8은 TETHER 코드 경로 Base이고, 이 Base의 Tok는 **189**. 205는 BF16 eager 표 Base(25.6 / 8.75 / 58.9)의 값 | `R/runs/chair_test500_all/score.json`, `outputs/qwen25-vl7b-fiveprofiles-*/original-vti-gpu1-final/metrics.csv` |
| 5 | 표 5 | "Qwen3.5-VL-8B" | 실행한 모델은 Qwen3-VL-8B-Instruct (`0c351dd`) | `configs/` |
| 6 | 표 7, 5.1절 | THRONE "5,000 images" | 표 숫자는 rows 200–999의 **800장**. rows 0–199는 최종 설정 선택에 썼고, rows 1000–4999는 `t4k` 실행 중 | `R/logs/v4_*_t800_table.txt`, `R/logs/throne_xl_table.txt` |
| 7 | 5.1절, 부록 B.2 | bootstrap 10,000회 | 모든 CI는 **2,000회**(seed 0) | `score_sets.py`, `score_chair100.py`, `vdd4_score.py`, `xl_score.py`, `score_throne.py`, `t4k_score.py` |
| 8 | 표 6, 4.3·5.4·5.5절 | "ablation에 penalty only, steering only가 없다", "w/o decay schedule in Table 6", "penalty-only row of Table 6" | 네 행 모두 실행했고 표에만 없음: belief only 27.0 / 7.26 / 68.4, penalty only 29.8 / 8.02 / 69.8, steering only 31.6 / 8.26 / 69.8, constant schedule 27.0 / 7.72 / 68.3 | `R/runs/chair_abl500_vsfull/score.json`, `R/logs/abl_chair.log` |
| 9 | 4.6절 | "layer와 강도는 Qwen2.5-VL 개발 이미지에서 정함" | 표의 설정은 테스트 행에서 골랐음: Qwen2.5 CHAIR `s16a2bl1p05` = CHAIR 테스트 0–149행, Gemma `s24a1bl1p05`와 Qwen3 `s20a1bl1p05` = 0–59행, THRONE `cf_a3bl2p1` = THRONE 0–199행 | `R/runs/chair_test60_sweep/`, `chair_test150_all/`, `gchair_t60/`, `qchair_t60/`, `*_cf_t200_all/` |
| 10 | 4.3·4.6절 | "schedule은 백본마다 한 번 측정" | Qwen3-VL은 Qwen2.5-VL의 r(t)를 씀: `rbd_common.r_curve`가 읽는 서버의 `qwen3-visual-decay-summary.json`이 `qwen-visual-decay-summary.json`의 symlink (11-05 확인) | `scripts/rbd_common.py`, 서버 `outputs/visual-decay-20261030/` |
| 11 | 표 7을 5,000장으로 바꿀 때 | — | Gemma THRONE 첫 번째 설정 `s24a1bl2p1`은 throne_full 4940–4999행(`gemma_throne_dev60`)에서 골랐고, 이 행은 `t4k` 범위 안에 있음 | `scripts/rbd2/gemma_chain.sh`, `R/runs/gemma_throne_dev60/` |
| 12 | 표 5 Qwen2.5 | REVIS·RUDDER 행 | BF16 eager 실행이라 캡션 일부에 ` addCriterion` 반복이 있음(REVIS 43, RUDDER 74, 같은 경로 Base 77 / 500; GrAInS 0). 비교 대상 Base(30.4)에는 없음 | `outputs/qwen25-vl7b-fiveprofiles-*/original-vti-gpu1-final/chair/*.jsonl` |

### 출처를 찾지 못한 수치

- **그림 1 a5**: 0.963은 `vast-tether-pilot/out/jlens/summary.json`(이미지 토큰, L24–25)에 있습니다. "Caption, early 0.79", "Caption, late 0.73",
  "+2.4 layers"는 박스의 결과 파일과 로그 어디에도 없습니다(논문 부록 A.8에도 TODO).
- **부록 B.5** "Base 10.4 s per image": 맞는 파일이 없습니다. CHAIR dev 100의 이미지당 시간은 bench Base 5.5 s, TETHER 코드 경로 Base(회색 행 포함) 10.3 s입니다
  (`R/runs/chair_dev100_qwen_all/{base,vddbase}-0-100.jsonl`의 `elapsed_seconds`). 10.3 s에는 이미 두 번째 행이 들어 있으므로 "roughly doubles"와 겹칩니다.

### 아직 결과가 없는 항목 (논문에도 pending)

표 A.4의 TETHER 두 행, 표 A.7(초반 증거), 5.5절 재감사 표, penalty-only 강도 sweep(π ∈ {0.25, 0.5, 1, 2}). 코드도 아직 없습니다.

## 3장과 부록 A: 감사

| 표·그림·본문 | 코드 | 결과 |
|---|---|---|
| 표 1, 그림 1 a1 (자기 검증) | `scripts/caption_selfcheck.py` → `caption_selfcheck_analyze.py`; 마지막 행(위치 맞춤 AUROC 0.932, 0.862)은 `audit_recheck.py` | `outputs/caption-selfcheck-20261030/{qwen,gemma}-base-analysis.json`, 객체별 행 `*-base-rows.jsonl`; `audit-recheck-20261105/{qwen,gemma}.json`의 `ranking/question_margin/position` |
| 3.1절, 부록 A.4 POPE (0.940 / 0.916 vs probe 0.880 / 0.859, FN 13/379, 17/303) | `read_gap_collect.py`, `read_gap_collect_v2.py` → `read_gap_analyze*.py`, `read_gap_complement.py` | `outputs/read-gap-e0-20261030/analysis-*.json` |
| 표 2 (위치별 새 물체), 부록 A.6 | `outputs/audit-recheck-20261105/audit_recheck.py`; 보조 `scripts/audit_inline/hall_share_by_position.sh` | `outputs/audit-recheck-20261105/{qwen,gemma}.json` |
| 표 3, 표 A.1, 그림 1 a3, 부록 A.2 (gray KL, attention, key blocking 0.81·0.72, logit-lens 기여 0.92·1.65, Gemma sliding window 35.5%), 부록 A.10 rollout | `visual_decay_run.sh` (`visual_decay_collect.py` → `visual_decay_analyze.py` → `visual_decay_plot.py`) | `outputs/visual-decay-20261030/*-visual-decay-summary.json` (`rollout_absorbing_image`, `window_limited_layers`) |
| grad×input (표 3, 표 A.2 위, 부록 A.4의 0.753 vs 0.749, A.2의 0.70→0.18) | `visual_reliance.py` → `reliance_analyze.py`, `reliance_analyze_v2.py` | `outputs/visual-reliance-20261030/qwen-reliance-summary.json` |
| 표 A.2 아래 (e_t 전체·첫·반복 언급), 부록 A.7 공출현 | `outputs/audit-recheck-20261105/audit_recheck2.py` | `{qwen,gemma}-2.json` |
| 그림 1 a2, 표 4 (위치·log p 맞춤 AUROC), 부록 A.8 순위 구간 | `audit_recheck.py`, `audit_recheck3.py`, lens 행은 `scripts/logit_lens.py` | `{qwen,gemma}.json`, `{qwen,gemma}-3.json`, `outputs/logit-lens-20261104/*-lens-summary.json` |
| 부록 A.3 attention (0.64 / 0.55, 모양 0.49–0.52, head 21–24) | `verify_probe_attn.py` | `outputs/query-anchor-step0-20261030/*-attn-summary.json` |
| 표 A.3 probe, 기준선 0.74 / 0.69 | `outputs/audit-recheck-20261105/probe_foldwise.py` (probe 학습은 `verify_probe_collect.py` → `verify_probe_analyze.py`) | `{qwen,gemma}-probe-foldwise.json`, `outputs/verify-probe-20261030/*-probe-summary.json` |
| 표 3·표 A.4 개입, 부록 A.5 선택성 차이 | `verify_probe_e1.py` | `outputs/verify-probe-20261030/*-e1-summary.json` |
| 표 A.5 위쪽 (이미지 토큰 lens AUROC, 같은 물체 안, 공출현 맞춤) | `scripts/logit_lens.py image/analyze`; 인라인 `audit_inline/lens_image_layers_qwen.sh`, `lens_gemma_image_caption.sh` | `outputs/logit-lens-20261104/{qwen,gemma}-lens-summary.json`, `inline/*.out.txt` |
| 표 A.5 "Word in top-5 of a patch" (마지막 층, L21) | 인라인 `audit_inline/lens_word_top5.sh` (`logit_lens.py image`가 저장한 L7·14·21·28 top-5에서 계산) | `outputs/audit-recheck-20261105/inline/lens_word_top5.out.txt` |
| 표 A.6 (top-100, 거리 × 위치), 부록 A.8 고정 거리 차이 | `scripts/logit_lens.py topk/topk-analyze` (`run_topk.sh`); 인라인 `audit_inline/lens_top100_distance_position.sh`; CI는 `audit_recheck3.py` | `*-topk-summary.json`, `inline/lens_top100_distance_position.out.txt`, `{qwen,gemma}-3.json` |
| 부록 A.8 "Writing step" (L26 −13.3, L27 −14.0, Gemma L34 −34.0, L38 −14.5) | 인라인 `audit_inline/lens_writing_step.sh` | `inline/lens_writing_step.out.txt` |
| 부록 A.8 hidden band top-1 (보조) | 인라인 `audit_inline/lens_hidden_band_top1.sh` | `inline/lens_hidden_band_top1.out.txt` |
| 부록 A.1 addCriterion 77개 제외 | `audit_recheck3.py`; 인라인 `audit_inline/addcriterion_exclusion.sh` | `{qwen,gemma}-3.json`, `inline/addcriterion_exclusion.out.txt` |

보조 모듈 `verified_decoding_v2.py`, `verified_decoding_v3.py`(멘션 위치), `read_gap_analyze_v3.py`(AUROC), `read_gap_collect_v2.py`(모델 로더),
`open_vocab.py`(WordNet 명사 필터)는 위 스크립트가 import해서 들어 있습니다.

`scripts/audit_inline/`의 8개는 감사 세션(12a8f468)이 11-05에 ssh로 서버에 보낸 명령을 세션 기록에서 그대로 옮긴 것입니다.
로컬 ssh 래퍼 한 줄만 뺐고, 출력은 그때 화면에 찍힌 그대로입니다.

### 그림 1 a5: Jacobian-lens 파일럿 (Qwen3.5-4B, vast.ai RTX 2080 Ti)

`vast-tether-pilot/` = 박스의 `/workspace/tether-pilot` 중 이 파일럿에 필요한 부분입니다.
`out/jlens_run.sh` → `pilot/jlens_probe.py`(`common.py`, `tests/test_jlens_probe.py`) → `pilot/jlens_summary.py`(`stats.py`) → `out/jlens/summary.json`.
층별 단어 분류 `pilot/jlens_layers.py` → `out/jlens/layers.json`. 캡션 전 위치 lens `out/jlens_pos_run.sh` → `pilot/jlens_positions.py`
(위치별 `.npz`는 507 MB라 빼고 `out/jlens_pos/objects.json`만 넣음). 렌즈는 `models/JLensVL-lenses/lens_qwen35_4b_final.pt`(넣지 않음),
`probe.npz`(58 MB)도 뺐습니다. 100장, 2026-11-08 05:46 UTC 완료. 300장 실행은 250장에서 멈췄습니다(`out/jlens_run_300_stopped.log`).

## 4장과 부록 B: TETHER

| 내용 | 코드 | 결과 |
|---|---|---|
| r(t), 표 B.2, 4.2절 가중치 | `scripts/rbd_common.py`의 `r_curve`: 표 A.1 gray KL을 첫 구간 1로 정규화, 구간 중심 사이 선형 보간 | 표 A.1과 같은 요약 파일 |
| V_obj (부록 B.3) | `rbd_collect.py` | `outputs/rbd-20261030/collect/*-vobj2.json`, 색 단어 제외판 `*-vobj2cf.json`은 `scripts/rbd2/vdd4_setup.py` |
| 디코더 (그림 2) | `scripts/rbd2/vdd_decode.py`; V_obj 파일을 바꿀 수 있는 판 `vdd4_decode.py` (THRONE 최종 설정, ablation) | — |
| 초기 탐색 (AMBER 101–160, CHAIR dev 0–59) | 클라우드 박스 `scripts/rbd2/cloud_dev/` (`prep.py`, `prep2.py` → `worker.sh`·`vrun.sh`·`chain.sh`) | `R/runs/chair_dev60/`, `R/runs/amber_dev60/`, `R/runs_dev_cloud/amber_dev60/` |
| 5.2절, 표 A.7 앞 두 행 (log p 0.667·0.694, 첫 변형 0.55·0.52) | `rbd_collect.py` → `rbd_offline.py` | `outputs/rbd-20261030/collect/{qwen,gemma}-offline.json` |
| 부록 B.4 dev 결과 (λ, measured vs constant r) | `make_probe_set.py` → `make_dev_chair.py`, `rbd_dvec.py`, `rbd_decode.py`(`rbd_common.py`), `score_dev.py` | `outputs/rbd-20261030/dev/` |
| 부록 B.4 파일럿 (CHAIR 테스트 앞 100장 30.0→18.0, uniform 18.0, recall 58.2 vs 53.7; AMBER 1–100 Cover −5.2~−7.6) | `rbd_decode.py`(λ 0.1), 클라우드 박스 2080 Ti | `results/rbd_20261101/round1_test_pilot/score-{chair-test100,amber100}.json`, 표는 `results/rbd_20261101/README.md` 1절 |

## 표 5: CHAIR 500 (왼쪽)

| 행 | 코드 | 결과 |
|---|---|---|
| Qwen2.5 REVIS†·RUDDER (BF16 eager), GrAInS (FP16) | `run_qwen_five_profile_suite.py` → `run_qwen_five_profile_worker.py`, 채점 `audit_qwen_five_profile_suite.py`, 방향 `fit_qwen_retained.py`, `extract_qwen_author_grains.py`, 원저자 함수 연결 `author_clone_bridge.py`·`qwen_author_compat.py`; GrAInS `run_qwen_grains_extra_workers.py`, `run_qwen_fp16_stable_workers.py` | `outputs/qwen25-vl7b-fiveprofiles-20261029-1793247386/original-vti-gpu1-final/chair/*.chair.official.json`, `metrics.csv` |
| Qwen2.5 VTI (공백 정규화 데모) | `scripts/rbd2/vti_fix.py`, `vti_chair.sh` → `vti_ws_run.py` | `R/runs/chair_test500_vtiws/`, 데모 `R/vti_fix/` |
| Qwen2.5 Base (TETHER 코드 경로)·TETHER `s16a2bl1p05` | 세트 `pb_make500.py`; 0–149행 `pb_worker.sh`·`pb_ext.sh`·`pb_pool.sh` → `pb_join150.py`, 150–499행 `pb_run500*.sh` → `pb_join500.py` → `pb_score500.sh`/`score_sets.py` | `R/runs/chair_test500_all/score.json`. 설정 선택(0–149행): `R/runs/chair_test60_sweep/`, `R/runs/chair_test150_all/` |
| Gemma Base·REVIS†·RUDDER·VTI† (BF16) | `run_author_clone_suite.py` → `continue_author_clone_suite.py` → `continue_gpu23_suite.py`, 채점 `audit_author_clone_suite.py`; REVIS·VTI 자체 구현 `src/vlm_grounding/baselines.py` | `outputs/author-clone-rerun-20261027-165446/bf16-full500-20261029-125744/bf16_common/*.chair.official.json` |
| Gemma GrAInS (FP16) | `extract_author_grains.py`, `run_gemma_grains_fp16_v2.py` → `grains_fp16_worker_v2.py`, 채점 `audit_five_profile_suite_v2.py` | `outputs/gemma-grains-fp16-v2-20261029-1793246507/chair/` |
| Gemma TETHER `s24a1bl1p05` | `scripts/rbd2/gchair3.sh` (0–59행 8개 후보 → `gchair_pick.py` → 60–499행) + `gchair_pool.sh`, `gchair_collect.py`, `gchair_join.py`, `score_sets.py` | 표의 모든 Gemma 행을 같은 이미지로 채점한 `R/runs/gchair_test500_tbl/score.json`, 선택 `R/runs/gchair_t60/` |
| Qwen3 Base·TETHER `s20a1bl1p05` | `qchair3.sh` (0–59행 9개 후보 → 자동 선택 → 60–499행) + `qchair_pool.sh`, `qchair_collect.py` | `R/runs/qchair_test500_all/score.json`, 선택 `R/runs/qchair_t60/` |
| Qwen3 RUDDER·REVIS(자체)·VTI | `qwen3_bench.py`, `qwen3_run_baselines.sh`, `run_vti_gpu23.sh`, 보고 `analysis/qchair_report.py` | `R/runs/qchair_test500_tbl/` |
| Qwen3 GrAInS, 공식 REVIS, 최종 표 | `qbl_chain.sh`: `extract_qwen3_grains.py`(FP32) → `qwen3_bl.py`(FP16 eager); `qbl_revis.sh` → `revis_official_q3.py`; 보고 `analysis/qbl_report.py` | `R/runs/qchair_bl2/`, 표 `R/logs/qbl.log` (Qwen3 행 전부) |

## 표 5: CHAIR dev 100 (오른쪽)

`scripts/rbd2/cdev_chain.sh` + `cdev_prep.py`: Base, RUDDER, GrAInS, VTI, TETHER(각 백본의 표 설정)를 세 백본에서 실행, 채점 `score_chair100.py`.
REVIS는 `after_v7.sh` 1단계(`hat_pool.sh`, 큐 `queues/chair_dev100_revis.txt`) → `cdev_prep.py score`. Qwen2.5·Gemma는 자체 구현, Qwen3는 공식 코드.
결과 `R/runs/chair_dev100_{qwen,qwen3,gemma}_all/score.json`, TETHER와 Qwen2.5 VTI-ws의 별표는 TETHER 코드 경로 Base 대비 `R/runs/chair_dev100_*_vdd/score.json`.
표 셀 `R/logs/cdev_table_{qwen,qwen3,gemma}.tex`(논문 표와 모든 칸이 같음), 로그 `R/logs/cdev.log`.
Qwen2.5 REVIS는 THRONE에서 같은 GPU 종류의 Base와 1,000/1,000 같은 캡션이었고, dev에서는 Base(A5000)와 REVIS(A6000)의 GPU 수치 차이로 96/100이 다릅니다.

## 표 6: Ablation (Qwen2.5 CHAIR 500)

`abl_chain.sh` + `abl_prep.py` (`vdd4_decode.py`, 큐 `queues/abl_chair.txt`): 전체 TETHER `s16a2bl1p05`에서 한 요소씩 빼거나 하나만 남긴 7개 변형.
결과 `R/runs/chair_abl500_vsfull/score.json`(전체 TETHER 대비 CI, 표 6의 CI), `R/runs/chair_abl500_all/score.json`(Base 대비), 로그 `R/logs/abl_chair.log`.
표 6에 없는 네 변형(belief only, penalty only, steering only, constant schedule)도 같은 파일에 있습니다(위 8번).

## 표 7: THRONE

| 단계 | 코드 | 결과 |
|---|---|---|
| 세트·이미지 | `pb_make_throne_full.py`, `pb_get_throne_images.py` | `R/sets/throne_full1000.jsonl`, `throne_t200.jsonl`, `throne_t800.jsonl` |
| Qwen 베이스라인 | `pb_bl_chain.sh` → `pb_bl_pool.sh` → `baseline_bench.py` → `pb_bl_join.py`/`pb_bl_full_join.py` → `pb_bl_full_score.sh` | `R/runs/throne_full1000/` (캡션, `eval/*/` 판정) |
| Qwen 첫 번째 설정 `s16a1bl2p1` | 0–499행 `throne_test500`, 500–999행 `pb_full1000_ours.sh` → `pb_join_full1000_ours.py` → `pb_full1000_finish.sh` | `R/runs/throne_test500_all/`, `R/runs/throne_full1000_ours/`, 선택 150장 `R/runs/throne_test150_all/` |
| Gemma 베이스라인·첫 번째 설정 `s24a1bl2p1` | `gemma_chain.sh`(선택 `gemma_pick.py`, throne_full 4940–4999행) → `gemma_pool.sh`, `gemma_bench.py`, `gemma_ours_fix.sh`, `gemma_split.sh`, `gemma_join.py`, 판정 `gemma_bl_judge.sh` | `R/runs/gemma_throne1000/`, 선택 `R/runs/gemma_throne_dev60/` |
| TETHER `cf_a3bl2p1` | `vdd4_setup.py`(V_obj) → `vdd4_chain.sh` → `vdd4_prep.py`·`vdd4_pool.sh`·`vdd4_decode.py`, `vdd4_qwen_resume.sh`, 채점 `vdd4_score.py`. 0–199행에서 선택, 200–999행 보고 | 선택 `R/runs/{qwen,gemma}_cf_t200_all/`, `R/runs/*_cf_pick.txt`, 1,000장 `R/runs/{qwen,gemma}_cf_test_all/`, 표 `R/logs/v4_{qwen,gemma}_t800_table.txt` (표 7의 2-judge 열과 모든 칸이 같음) |
| 판정 (Flan-T5 base·large, 임계값 6) | `pb_throne_judge.py`(`RBD2_JUDGES`), `score_throne.py`, `pb_throne_partial.py` | `R/diag/*_2judges/` |
| xl 열 (base·large·xl, 임계값 9) | `after_v7.sh` 2단계 `xl_pool.sh` → `xl_judge.py` → `xl_score.py` (rows 200–999) | `R/runs/*/eval/*/combined/*flan-t5-xl.json`, 표 `R/logs/throne_xl_table.txt` (표 7의 xl 열과 같음) |
| ALL·CLS 지표, orange 분석 | `analysis/throne_all_metrics.sh`, `throne_official_scorer.sh`(공식 `throne_score_aqa.py`), `throne_classwise_ci.sh`, `throne_orange.sh`, `throne_without_orange.sh`, `throne_breakdown.sh` | 인라인 실행, 출력은 표에 직접 |
| **5,000장으로 확장 (진행 중)** | `t4k_chain.sh`: `t4k_prep.py`(세트 `throne_r4000` = throne_full 1000–4999행, 큐) → `t4k_pool.sh`(디코딩, 먼저 저장된 A6000 행 재현 확인) → `t4k_jpool.sh`·`t4k_judge.py`(블록별 판정) → `t4k_score.py` | 서버 `R/logs/t4k.log`(T5K_DONE), `R/logs/t4k_table.txt`. 아직 결과 없음 |

`t4k_score.py`는 rows 200–999(현재 표 확인용), 200–N(보고), 0–N, 1000–N을 출력합니다. `--blocks 0`으로 현재 표 7의 모든 숫자를 재현했습니다.
GPU 0, 5(A6000 두 장), 디코딩 약 374 GPU-시간 + 판정 약 126 시간으로 추정해 11-09 기준 4–5일 걸립니다.

## 부록 C, D

| 표 | 코드 | 결과 |
|---|---|---|
| 표 C.1 설정 | `src/vlm_grounding/baselines.py`(REVIS, Gemma VTI), `author_clone_bridge.py`, `qwen_author_compat.py`, `rbd2/vti_fix.py` | — |
| 표 C.2 (VTI 강도, 앞 50장) | α=0.9: `prepare_original_vti.py`, `run_qwen_original_vti.py`, `qwen_vti_chair_gpu23.py` 계열; α=0.45·0.225: `qwen_vti_alpha_sweep.py` → `score_vti_alpha_sweep.py` | `outputs/qwen-original-vti-1793260357/`, `outputs/qwen-vti-alpha-sweep-20261030/score-pilot-chair50/score.json` |
| 부록 C의 VTI 데모 줄바꿈 88/100 vs 19/100, 7점 grid | `scripts/rbd2/vti_fix.py`, `vti_chair.sh` | `R/vti_fix/`, `R/runs/chair_dev60_vtiws/` |
| 표 D.1, 사례 1–4 | `scripts/rbd2/analysis/case_study_extract.sh` (같은 CHAIR scorer로 방법별 객체 추출), Yes-No margin은 `caption-selfcheck-20261030/qwen-base-rows.jsonl` | 위 CHAIR 500 캡션 |

## 초안 3에 없는 실험 (브랜치에는 남김)

| 실험 | 코드 | 결과 |
|---|---|---|
| HAT + ALOHa 385장 | `hat_setup.sh`, `hat_gpu3.sh`/`hat_chain.sh`, `hat_pool.sh`·`hat_join.sh`, `hat_prep.py`, `hat_parse.py`, `hat_aloha.py`, `hat_common.py`, `hat_smoke.sh` | `R/runs/hat385_{qwen,gemma}/`, `R/hat385/`, `R/hat_parse/`, `R/logs/hat385.log` |
| HAT 두 번째 선택 v7 | `vdd7_decode.py`, `hat_v7_chain.sh`, `hat_v7_prep.py` | `R/hat385/score-qwen-v7-*.json`, `R/logs/hat_v7.log` |
| ALOHa 집계 버그 수정 | `hat_aloha.py`(`HAT_ALOHA_FIX=1`), `R/hat_fix.sh` | `R/hat385/scorefix-*.json`, `R/logs/hat_aloha_fixed_table.tex` |
| AMBER 생성형 1,004장 | `pb_amber_bl.sh` → `pb_amber_pool.sh` → `baseline_bench.py`, `amber_full_score.sh`; `pb_amber_ours1004.sh` | `R/runs/amber_full1004_all/`, `R/runs/amber_full1004_ours_all/`, `R/runs/amber_test150_all/` |
| Gemma CHAIR 500 ablation | `gabl_chain.sh` + `gabl_prep.py`, 이어서 `after_v7.sh` 3단계 | `R/runs/gabl_chair500_{all,vsfull}/`, `R/logs/gabl_chair_table.tex` |

`hat_pool.sh`는 CHAIR dev 100의 REVIS 행(표 5)에도 쓰였습니다.

## 넣지 않은 것

- 논문에 없는 실험: self-verification 디코딩, MMVP, MMHal-Bench, five-profile POPE 결과, AMBER vdd3 변형, Qwen3 탐색 rbd5·rbd6,
  Gemma CHAIR 재선택 `gchair2.sh`, VTI-ws THRONE 100장, 진단 스크립트(`diag_*`, `pb_diag_*`), recall 분석, HAT 파일럿.
- M3ID·VCD CHAIR 500: 다른 세션이 11-09에 코드를 준비했지만 아직 실행 전입니다.
- Jacobian lens의 11-08 추가 실행(`jlens_gray`, `jlens_pos100`): 요약 파일이 없고 논문에 쓰이지 않았습니다.
- 중간 산출물: 부분 실행 폴더(`*_parts`, 행 구간별 폴더), 60·120장 중간 채점과 그 xl 판정, 스모크 테스트 결과, 판정 전 백업 `combined.bak-1108`, v7 블러 이미지 캐시 `R/cf/`.

옛 베이스라인 파이프라인(Gemma 5방법, Qwen five-profile)은 실행할 때 서로의 소스 해시를 검사하므로, 그 검사 목록에 있는 파일은 모두 포함했습니다.

## 원저자 코드와 채점기 (포함하지 않음)

| 이름 | 고정 버전 |
|---|---|
| RUDDER | Akko000/RUDDER-Residual-Update-Directed-DEcoding-Regulation- `bd149d8` |
| GrAInS | duykhuongnguyen/GrAInS `8ad78db` |
| VTI | shengliu66/VTI `bb5dfee` |
| REVIS (공식, Qwen3 행) | antgroup/Revis `75f4d29` |
| Jacobian lens | 박스의 `tools/jacobian-lens`, 렌즈 `JLensVL-lenses/lens_qwen35_4b_final.pt` |
| CHAIR | Python 3 포트, `4087a26` (서버 `upstream/chair/chair.py`) |
| THRONE | amazon-science/THRONE `45badbb7` (one-click.sh 2–3단계) |
| AMBER | 공식 생성형 평가 (`inference.py`, spaCy `en_core_web_lg`) |
| ALOHa | DavidMChan/aloha `7da6b90` |

모델 revision: Qwen2.5-VL-7B-Instruct `cc59489`, Gemma 4 E4B-it `ee0ef60`, Qwen3-VL-8B-Instruct `0c351dd` (`configs/*.json`), Qwen3.5-4B(파일럿).
