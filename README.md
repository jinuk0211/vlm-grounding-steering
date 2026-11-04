# 논문 표의 기반 실험 코드 (2026-11-04)

논문 초안(`prev_2026-11-03_updated.tex`)의 **표에 실린 숫자**를 만든 실험 코드만 모은 브랜치입니다.
파일은 실험 서버(`/home/project-b/LJW/vlm-grounding-steering`)에서 결과를 만든 그 상태로 가져왔고, 경로도 서버와 같습니다
(`scripts/`, `scripts/rbd2/`, `src/vlm_grounding/`). 아래 "서버 결과" 경로는 모두 서버 저장소 기준입니다.

- 원저자 코드(RUDDER, GrAInS, VTI, CHAIR·POPE·AMBER·THRONE 채점기)는 넣지 않았습니다. 고정 버전은 맨 아래에 있습니다.
- 모델 가중치, 데이터, 캡션·채점 결과도 들어 있지 않습니다.
- 스크립트는 서버 경로, `.venv`, HF 캐시, GPU UUID, 소스 해시 검사를 전제로 하므로 다른 머신에서 그대로 돌아가지 않습니다.

## 표 → 코드

### 감사(audit): `tab:gap`, `tab:findings`, `tab:decay`, `tab:objtok`, `tab:probe`, `tab:steer`

| 표 | 진입 스크립트 | 서버 결과 |
|---|---|---|
| `tab:gap` 생성–검증 격차 (자기 캡션의 객체를 물었을 때 부정 비율, AUROC) | `scripts/caption_selfcheck.py` → `scripts/caption_selfcheck_analyze.py` | `outputs/caption-selfcheck-20261030/{qwen,gemma}-base-analysis.json` |
| `tab:decay`, `tab:findings` F1, `tab:objtok` gray Δlog p | `scripts/visual_decay_run.sh` (`visual_decay_collect.py` → `visual_decay_analyze.py` → `visual_decay_plot.py`) | `outputs/visual-decay-20261030/*-visual-decay-summary.json` |
| `tab:objtok`·`tab:findings` F2 grad×input 이미지 비중 (Qwen 200 캡션) | `scripts/visual_reliance.py` → `scripts/reliance_analyze_v2.py` | `outputs/visual-reliance-20261030/qwen-reliance-summary.json` |
| `tab:probe`, `tab:findings` F3 부정 예측 probe (real / gray) | `scripts/verify_probe_collect.py` → `scripts/verify_probe_analyze.py` | `outputs/verify-probe-20261030/*-probe-summary.json` |
| `tab:steer`, `tab:findings` F2 attention boost·F4 probe/random shift | `scripts/verify_probe_e1.py` | `outputs/verify-probe-20261030/*-e1-summary.json` |

`verified_decoding_v2.py`, `verified_decoding_v3.py`(멘션 토큰 위치), `read_gap_analyze_v3.py`(AUROC), `read_gap_collect_v2.py`(모델 로더),
`open_vocab.py`(WordNet 명사 필터)는 위 스크립트가 import하는 보조 모듈이라 포함했습니다. self-verification 디코딩 실험 자체는 논문에 없습니다.

### RBD 첫 번째 변형(확률 belief): `tab:dev`, `tab:schedule`

| 표 | 코드 |
|---|---|
| `tab:dev` (dev 100장, Qwen) | 데이터: `make_probe_set.py` → `make_dev_chair.py`. 모드 방향: `rbd_dvec.py`. 디코딩: `rbd_decode.py` (`rbd_common.py`). 채점: `score_dev.py`. 결과 `outputs/rbd-20261030/dev/` |
| `tab:schedule` r(t) | `rbd_common.r_curve`: visual-decay 요약의 gray-KL 곡선을 첫 구간 1로 정규화 |

### RBD 두 번째 변형(시각 증거 belief): `tab:dev2`

- 디코더: `scripts/rbd2/vdd_decode.py`. 서버 버전은 클라우드 버전에 `VDD_GRAY` 환경변수, `--norm median`, `--bpos`를 더한 것이고,
  기본값(`--norm raw`, `--bpos` 없음)에서는 클라우드 버전과 같은 계산입니다.
- dev60 실행은 클라우드 박스(`/workspace`)에서 했습니다: `scripts/rbd2/cloud_dev/`
  (`prep.py`, `prep2.py` 데이터·세트 → `worker.sh`/`vrun.sh`/`chain.sh` 큐 실행, 설정은 `cloud_dev/queues/q0,q0b,q1,q1b.txt`).
- 채점: `scripts/rbd2/score_sets.py` (CHAIR: 캐시된 원본 scorer 상태, AMBER: `score_amber100.py`의 공식 매칭), paired bootstrap 2,000회.
- AMBER용 median 정규화 dev: `scripts/rbd2/queues/dev_q2.txt`, `dev_q3.txt`.

설정 태그: `s{L}a{α}bl{λ}p{π}` = `--layer L --alpha α --lam λ --pen π` (예: `s16a2bl1p05` = `--layer 16 --alpha 2 --lam 1 --pen 0.5`),
`mainMed` = `s16a1bl1p05` + `--norm median`. 모든 큐 파일(`scripts/rbd2/queues/*.txt`)에 태그별 인자가 그대로 있습니다.

### 본 결과: `tab:hallucination_all_models` (CHAIR 500 / AMBER 1,004)

| 행 | 코드 |
|---|---|
| Qwen Base·REVIS·RUDDER (CHAIR, eager BF16) | `run_qwen_five_profile_suite.py` → `run_qwen_five_profile_worker.py`, 채점 `audit_qwen_five_profile_suite.py`. 방향 적합 `fit_qwen_retained.py`, GrAInS 추출 `extract_qwen_author_grains.py`, 원저자 함수 연결 `author_clone_bridge.py`·`qwen_author_compat.py` |
| Qwen GrAInS (CHAIR, FP16) | `run_qwen_grains_extra_workers.py`, `run_qwen_fp16_stable_workers.py` (`qwen_fp16_stable_stage.py`, `qwen_fp16_attention_guard.py`), 최종 집계 `qwen_vti_gpu1.py` 계열(`launch_/resume_/audit_..._completion/export_qwen_vti_gpu1.py`) |
| Qwen VTI¶ (공백 정규화 데모) | 데모 정규화·방향 재추출 `scripts/rbd2/vti_fix.py`, CHAIR 실행 `vti_chair.sh` → `vti_ws_run.py`; AMBER는 `pb_amber_bl.sh` |
| Qwen "Base, RBD code path"·RBD (CHAIR) | 세트 `pb_make500.py`; 0–59행 `pb_worker.sh`(`queues/q0d,q1d,sweep_q0,sweep_q1.txt`; sweep 행은 `sweep_decode.py`), 60–119행 `pb_make120.py`+`pb_ext.sh`(`queues/ext_*.txt`), 120–149행 `pb_make150.py`+`pb_pool.sh`(`queues/pool150.txt`) → `pb_join150.py`; 150–499행 `pb_run500*.sh`+`pb_pool.sh`(`queues/pool500.txt`) → `pb_join500.py` → `pb_score500.sh`/`score_sets.py` |
| AMBER 1,004 베이스라인 | `pb_amber_bl.sh` → `pb_amber_pool.sh` → `baseline_bench.py` (`queues/amber_full.txt`), 채점 `amber_full_score.sh` |
| AMBER 1,004 RBD·자체 Base | 0–499행은 위 500장 실행(`mainMed`), 나머지 `pb_amber_ours1004.sh` (`queues/amber_ours1004.txt`) → `score_sets.py` |
| Gemma Base·REVIS†·RUDDER·VTI† (CHAIR) | Gemma 5방법 실행: `run_author_clone_suite.py` → `continue_author_clone_suite.py` → `continue_gpu23_suite.py`(+ `continue_gpu23_entry/worker.py`, `gpu23_device_adapter.py`), 채점 `audit_author_clone_suite.py`. 데이터·채점 캐시 `prepare_hallucination_suite.py`, `build_chair_cache.py`; REVIS·VTI 자체 구현은 `src/vlm_grounding/baselines.py` |
| Gemma GrAInS (CHAIR, FP16) | `extract_author_grains.py`, `run_gemma_grains_fp16_v2.py` → `grains_fp16_worker_v2.py`, 채점 `audit_five_profile_suite_v2.py` |
| Gemma "Base, RBD code path"·RBD (CHAIR) | `scripts/rbd2/gchair_chain.sh`(30장 dev 확인) → `gchair_finish.sh` (s24a2bl1p05 고정) → `gchair_pool.sh` → `gchair_join.py` → `score_sets.py` |

### THRONE: `tab:throne`, `tab:throne_full`

| 단계 | 코드 |
|---|---|
| 세트·이미지 | `pb_make_throne_full.py`, `pb_get_throne_images.py`, 클라우드 `cloud_dev/prep3.py`(throne_test60) |
| Qwen 베이스라인 1,000장 | `pb_bl_chain.sh` → `pb_bl_pool.sh` → `baseline_bench.py` (`queues/bl_throne_full.txt`) → `pb_bl_join.py`/`pb_bl_full_join.py` → `pb_bl_full_score.sh` |
| Qwen RBD 1,000장 | 0–499행 = 위 `throne_test500` 실행, 500–999행 `pb_full1000_ours.sh` (`queues/full1000_ours.txt`) → `pb_join_full1000_ours.py` → `pb_full1000_finish.sh` |
| Gemma 전체 | `gemma_chain.sh`(dev 선택 `gemma_pick.py`, `queues/gemma_dev.txt`) → `gemma_pool.sh` → `gemma_bench.py`(베이스라인, `queues/gemma_bl.txt`), `gemma_ours_fix.sh`(`queues/gemma_ours.txt`), `gemma_split.sh`, `gemma_join.py`, 판정 `gemma_bl_judge.sh` |
| 판정·채점 | `pb_throne_judge.py`(flan-t5 base·large, `RBD2_JUDGES`), `score_throne.py`, `pb_throne_partial.py` |
| 표의 ALL·CLS 지표와 CI | `scripts/rbd2/analysis/throne_all_metrics.sh`(ALL, 판정자별), `throne_official_scorer.sh`(공식 `throne_score_aqa.py`, 임계값 4–6), `throne_classwise_ci.sh`(CLS + paired bootstrap) |
| 표 캡션의 orange 분석 | `analysis/throne_orange.sh`, `throne_without_orange.sh`, `throne_breakdown.sh` |

`analysis/`의 스크립트는 원래 세션에서 인라인으로 서버에 보내 실행한 것을 세션 기록에서 그대로 복원한 것입니다.

### 부록

| 표 | 코드 |
|---|---|
| `tab:vti_pilot` (Qwen 원저자 VTI 강도, 첫 50장) | α=0.9: `prepare_original_vti.py`, `run_qwen_original_vti.py`, `qwen_vti_chair_gpu23.py` 계열; α=0.45/0.225: `qwen_vti_alpha_sweep.py` → `score_vti_alpha_sweep.py` (`outputs/qwen-vti-alpha-sweep-20261030/score-pilot-chair50`) |
| `tab:cases` | `scripts/rbd2/analysis/case_study_extract.sh` (같은 CHAIR scorer로 방법별 객체 추출) |
| `tab:baseline_config` | 위 베이스라인 코드 (`src/vlm_grounding/baselines.py`, `author_clone_bridge.py`, `qwen_author_compat.py`, `rbd2/vti_fix.py`) |

## 넣지 않은 것

- 표가 비어 있거나 본문에만 나오는 결과의 코드: MMHal-Bench(표 전부 `--`), POPE read-gap probe(`read_gap_*` 중 import되지 않는 것),
  attention 모양 분석(`verify_probe_attn.py`), 오프라인 판별(`rbd_offline.py`, `rbd_collect.py`), 60·120장 중간 채점(`pb_score.sh`, `pb_finish.sh`, `pb_finish120.sh`).
- 논문에 없는 실험: self-verification 디코딩(`verified_decoding.py`, `verify_probe_steer.py`, `score_verified*.py`), logit lens(`logit_lens.py`),
  AMBER vdd3(`vdd3_*`), Gemma CHAIR 재선택(`gchair2.sh`, `gchair3.sh`), VTI-ws THRONE 100(`vti_ws100.sh`), 진단(`pb_diag_*`).
- 운영 도구 중 실행 경로에 없는 것(관찰·프리플라이트 일부, 서버 셋업 스크립트).

옛 베이스라인 파이프라인(Gemma 5방법, Qwen five-profile)은 실행 시 서로의 소스 해시를 검사하므로, 그 검사 목록에 있는 파일은 모두 포함했습니다.

## 원저자 코드와 채점기 (포함하지 않음)

| 이름 | 고정 버전 |
|---|---|
| RUDDER | Akko000/RUDDER-Residual-Update-Directed-DEcoding-Regulation- `bd149d8` |
| GrAInS | duykhuongnguyen/GrAInS `8ad78db` |
| VTI | shengliu66/VTI `bb5dfee` |
| CHAIR | Python 3 포트, `4087a26` (서버 `upstream/chair/chair.py`) |
| THRONE | amazon-science/THRONE `45badbb7` (one-click.sh 2–3단계) |
| AMBER | 공식 생성형 평가 (`inference.py`, spaCy `en_core_web_lg`) |

모델 revision: Qwen2.5-VL-7B-Instruct `cc59489`, Gemma 4 E4B-it `ee0ef60` (`configs/*.json`).
