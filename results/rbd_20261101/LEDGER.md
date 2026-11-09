# Round 2 ledger (2026-11-01): visual-evidence belief + steering, Qwen2.5-VL-7B, 60-image sets

Goal (user): late in the caption the image's contribution fades; put back belief and steering in later tokens and
get good scores. Theme fixed; method free.

## Protocol
- Selection only on dev: chair_dev60 (COCO train2014 dev, first 60) + amber_dev60 (AMBER 101-160).
- Test sets: chair_test60 (first 60 CHAIR test) + amber_test60 (AMBER 1-60). Every look at a test set is logged
  below with the configs evaluated.
- Base for every set = vdd_decode.py with alpha = lam = pen = 0 (same batch-2 path, sdpa, bf16, 2080 Ti).

## Setup changes vs round 1
- attn sdpa instead of eager: bf16 + eager writes ' addCriterion' into 12-18% of captions (round 1 and A5000 alike);
  sdpa 0/12 vs eager 5/12 on the first 12 CHAIR test images (check_glitch.py). Round-1 numbers carry this defect.
- batch-2 lockstep decoding (real + 448x448 gray); decode-step linears row-split (bf16 2-row matmul 4.8x slower on
  Turing).

## Dev selection (chair_dev60 + amber_dev60, all configs in round2_dev/runs/*/score.json)
Rule: lowest CHAIR_s(dev) + AMBER Hal(dev) change with CHAIR recall drop <= 1 and AMBER Cover drop <= 2.
Chosen: L16 alpha 1 + belief lam 1 + penalty 0.5 (tag s16a1bl1p05 on dev, 'main' on test):
CHAIR_s -13.3, recall +1.0; AMBER CHAIR -2.1, Hal -6.7, Cover -0.9 (vs Base).
Excluded by the rule: s16a1bl2p1 (largest hallucination drop, AMBER Cover -5.3).

## Test-set looks
1. 2026-11-01 ~17:00 UTC, queued once (q0d.txt, q1d.txt): chair_test60, amber_test60, throne_test60, amber_test60L
   x {base, main, noSteer (alpha 0), noBelief (lam 0), constSched (s, w = 1)}. Main fixed on dev before this look;
   the other four are the Base and ablations of that one config, not alternatives to choose from.
2. 2026-11-01 17:30 UTC, part of look 1: chair_test60 base and main scored (score_sets.py). base CHAIR_s 38.3,
   CHAIR_i 12.9, recall 68.4, F1 76.7; main 30.0, 11.6, 63.6, 74.0; dCHAIR_s [-20.0, +3.3], drecall [-10.0, +0.5].
   On dev main kept recall (+1.0); on test it fell 4.8.
3. 2026-11-01 17:37 UTC, user request: chair_test60 sweep of every round-2 dev config plus bpos_s16a1 and bpos_s16a3
   (chair_test60_sweep/). chair_test60 is no longer held out for any config picked from this sweep.

## Dropped
Object HalBench: its official scorer extracts objects with gpt-3.5-turbo-0613 (retired); the rule-based mode is
CHAIR's COCO-80 matching under other prompts, so it adds little over CHAIR.
