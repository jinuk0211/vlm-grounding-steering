# Steering-variant screen: protocol (fixed 2026-11-10, before any run)

## Question
Belief and penalty are kept exactly as in TETHER (`--lam 1 --pen 0.5`, decay schedules, raw belief, gray control).
Only the steering step changes. Does any steering variant lower CHAIR_s beyond belief + penalty alone (`nosteer`)
without losing recall?

## Model and data
- Qwen2.5-VL-7B-Instruct `cc59489`, bf16, sdpa, greedy, 512 new tokens, prompt "Please describe this image in detail."
- Screening: `chair_test100` = rows 0-99 of the paper's `chair_test500` (same ids, prompts and image sha256).
- Confirmation: rows 100-499 of `chair_test500` (400 images not used for selection).

## Conditions (screen)
| tag | steering | hypothesis it tests |
|---|---|---|
| full | residual difference real-gray, L16, alpha 2 (the paper) | reference |
| nosteer | none (belief + penalty only) | reference (the -steer ablation row) |
| r_l8 | residual, L8, alpha 2 | earlier layer |
| r_l22 | residual, L22, alpha 2 | later layer (image attention peak) |
| r_multi | residual, L8/12/16/20/24, alpha 0.5 each | spread over depth |
| r_norm | residual difference rescaled to 0.5 of |h|, L16 | the difference shrinks along the caption (too weak late) |
| r_anchor | mean difference of caption steps 1-16, re-applied later, L16 | visual evidence is strongest early |
| r_dose4 | residual, L16, alpha 4 | dose-response: is the direction or the strength the problem |
| attn_mid | attention-update difference, L12-L20 | site: what the layer reads from the image, not the cumulative stream |
| pai_mid | image-key attention logits + log(1 + s(t)), L8-L20 | mechanism: read the image more (PAI-style) |
| blur16 | residual real-blur (Gaussian r=12 at 448 px), L16, alpha 2 | contrast: keep scene gist, remove object detail |
| sel24 | belief-guided direction toward objects above threshold, L24, rho 0.1 | selectivity: steer by evidence, not by image |

## Analysis (fixed)
- Unmodified CHAIR scorer state; paired bootstrap 2,000 over images (seed 0).
- Primary: CHAIR_s(variant) - CHAIR_s(nosteer). Secondary: recall difference, CHAIR_i.
- Selection rule: among the 10 variants with recall >= nosteer recall - 1.0, the lowest CHAIR_s (ties: CHAIR_i).
  Carried to confirmation only if its CHAIR_s is below `full`'s on the screen.
- 100 images detect only large differences (a CHAIR_s CI on 100 images is roughly +-8 points), so the screen picks a
  candidate and does not test it. Significance is claimed only from the confirmation rows 100-499, with the picked
  variant, `nosteer` and `full` all decoded on the same hardware.
## Round 1 result (2026-11-10)
No variant passed the rule (eligible attn_mid, blur16, r_l8; picked r_l8, CHAIR_s not below full). Lower CHAIR_s came only
with lower recall (r_norm 23.0 / recall 65.5, r_anchor 24.0 / 62.5); late-layer and attention-output steering raised CHAIR_s.

## Round 2 (fixed 2026-11-10, before any round-2 run)
Ten variants built from round 1: fixed-relative-strength steering (norm: rho 0.3 / 0.8 at L16, 0.5 at L12), anchored
direction with fixed strength (anchornorm), belief-guided steering (sel: stronger rho 0.3; median-normalised belief; push
away from objects below threshold only; at L20), and combinations of the paper's steering or norm steering with
median-belief sel at L24. Same references (round-1 full and nosteer), same selection rule applied to these 10 only, same
confirmation on rows 100-499. Twenty variants have now been screened on the same 100 images, so a screen-level CI that
excludes 0 is not evidence by itself; only the confirmation run counts.

- All 12 screening rows are reported, whatever they show. If no variant is carried forward, the result is "no steering
  variant lowered CHAIR_s beyond belief + penalty", and steering's reported role stays recall preservation.

## Post-hoc confirmation of an16_r05 (deviation; written 2026-11-10 04:50 UTC, before any rows 100-499 run)
Deviation: an16_r05 FAILED the pre-registered recall criterion on the screen (recall 62.5, needed >= 65.8). It is confirmed
anyway because it had the lowest screen CHAIR_s of the 20 variants (17.0 vs nosteer 27.0, CI [-18,-3]; vs full CI [-15,-2])
and, against nosteer captions cut to the same per-image length (lenctl.py), still CHAIR_s 17.0 vs 24.0, CI [-14.0,+0.0],
at equal recall (62.5 vs 61.3). It was picked after seeing the screen, as the best of 20, so the screen numbers are inflated
(winner's curse); only rows 100-499 can support a claim.
- Runs on rows 100-499, same box: nosteer, an16_r05, full (unchanged args; skipped if the screen-2 confirmation already made them).
- Primary: CHAIR_s(an16_r05) - CHAIR_s(nosteer), paired bootstrap 2,000 (seed 0). Called significant only if the CI excludes 0.
- Always reported with it: recall difference (a recall loss is a cost, not hidden), CHAIR_i, caption length, and the
  length-matched control (an16_r05 vs nosteer cut to its length). If an16_r05 beats nosteer but not the length-matched
  nosteer, the result is "shorter captions", not "better grounding".
- Also reported: full vs nosteer on rows 100-499 and pooled with rows 0-99 (500 images, this box).

## Round 3: an16_r05 developed toward no recall loss (fixed 2026-11-10 ~04:55 UTC, before any round-3 run)
Goal (from the author): keep recall; a smaller CHAIR_s reduction is acceptable.
- Six variants on rows 0-99 (queue_qwen100_r3.txt), round-1 full / nosteer as references:
  weaker anchored push (an16_r03, an16_r04), longer anchor window K=32 (an16_r04_k32, an16_r05_k32), and the paper's
  steering plus a small anchored push at the same layer (full_an16_r02, full_an16_r03). In the combined runs the second
  hook sees the first hook's output; the first adds alpha s(t) d_t, parallel to d_t, so the second hook's anchor has the
  same per-step directions (reweighted by 1 + alpha s(t)).
- Rule: eligible if recall >= nosteer recall - 1.0 (65.8); pick the lowest CHAIR_s (ties CHAIR_i); carried to
  confirmation if its CHAIR_s is below nosteer's (27.0). Relaxed from "below full" because the goal is a recall-neutral
  reduction, not the largest one.
- Confirmation on rows 100-499 (replaces the posthoc.sh order): nosteer, round-3 pick, full, an16_r05, same args.
  Two selected variants are tested on the same 400 images, so each is called significant only if its Bonferroni 97.5% CI
  for CHAIR_s vs nosteer excludes 0 (95% CIs also reported). "No recall loss" is claimed only if the 95% CI of the recall
  difference vs nosteer has its lower bound above -2.0 points. Length-matched control reported for both.

## Final run (author's decision, 2026-11-10 ~05:10 UTC, before any round-3 run; supersedes round 3 and the round-2 confirmation)
The author chose to skip the round-3 screen and the 400-image confirmations and to run one recall-preserving variant of
an16_r05 on all 500 test rows, compared with the paper's existing 500-row runs.
- Variant, fixed before any data on it: full_an16_r03 = the paper's steering (resid L16 alpha 2, kept because it preserves
  recall) + anchored fixed-strength steering at L16 with rho 0.3 (an16_r05 used 0.5), K 16. Belief and penalty unchanged.
- Comparisons: vs the paper's abl_nosteer (primary), rbd_full and base (all 500 rows, from the paper's server), and vs
  same-box nosteer (rows 0-99 existing + rows 100-499 run after the main run). Box vs paper-server captions differ (only
  2/100 nosteer captions identical on rows 0-99), so the same-box comparison is the hardware check.
- Rows 0-99 informed the choice of the an16 family, so rows 100-499 are reported separately as well.
- One test: significant if the 95% CI of CHAIR_s vs nosteer excludes 0. "No recall loss" only if the recall difference's
  95% CI lower bound is above -2.0. Length-matched control reported.
- Round 2's last variant (n16_selm24) is scored and reported, but its rule-based confirmation is not run.
