# RBD round 2 (2026-11-01): RBD on held-out sets, then visual-evidence decoding

Status: in progress. Dev selection is done and the test runs are queued (see [LEDGER.md](LEDGER.md)). Everything
here is Qwen2.5-VL-7B on 60- or 100-image sets; 95% paired-bootstrap CIs over images (2000, seed 0) are given where
they matter, and differences of a few points are inside them. Code: [scripts/rbd2/](../../scripts/rbd2/).

## 1. RBD on held-out sets (round 1)

`scripts/rbd_decode.py` unchanged, lambda 0.1 (chosen on dev), bf16 on RTX 2080 Ti, Base re-run under the same
conditions. These runs use the repo's eager attention and carry the defect in section 2.

CHAIR test, first 100 images:

| run | CHAIR_s | CHAIR_i | recall | F1 | dCHAIR_s CI | drecall CI |
|---|---:|---:|---:|---:|---|---|
| Base | 30.0 | 11.32 | 57.3 | 69.63 | | |
| RBD, measured r | 18.0 | 8.95 | 53.7 | 67.52 | [-19, -5] | [-8.4, +1.0] |
| RBD, constant r | 23.0 | 8.86 | 57.0 | 70.15 | [-14, 0] | [-3.8, +3.3] |
| uniform control | 18.0 | 8.01 | 58.2 | 71.32 | [-20, -4] | [-3.3, +5.3] |

AMBER generative, images 1-100:

| run | CHAIR | Cover | Hal | Cog | dCover CI |
|---|---:|---:|---:|---:|---|
| Base | 5.7 | 56.4 | 27.0 | 1.4 | |
| RBD, measured r | 5.3 | 51.2 | 27.0 | 0.8 | [-8.6, -1.8] |
| RBD, constant r | 5.5 | 50.7 | 29.0 | 0.8 | [-9.0, -2.4] |
| uniform control | 6.4 | 48.8 | 28.0 | 1.0 | [-11.3, -4.0] |

- The uniform control (same average push, no object-specific belief) matches RBD on CHAIR_s and has higher recall.
  RBD minus uniform: CHAIR_s [-6, +6], recall [-10.6, +0.9].
- On AMBER no variant lowers hallucination and all lose coverage.
- Per object ([diagnostics/round1_*](diagnostics/)): against Base, RBD lost 26 correct COCO objects and gained 14;
  uniform lost 17 and gained 20. On AMBER the lost objects were late background nouns (sky, forest, sea, cloud):
  52 of 53 were in V_obj, 5 of 53 in the COCO-80 list, median position 0.69 of the Base caption. CHAIR does not
  count most of them, which is why the loss shows on AMBER only.

## 2. Setup defect: bf16 with eager attention

With the repo's `attn_implementation: eager`, bf16 Qwen2.5-VL writes ' addCriterion' into captions: 12-18% of the
round-1 captions, and also the earlier A5000 runs (33/234 AMBER, 4/50 CHAIR test Base captions). sdpa removes it:
0/12 vs 5/12 on the first 12 CHAIR test images, and no 512-token runaways
([round2_dev/logs/glitch-*.log](round2_dev/logs/)). A float32 vision tower does not (2/12). All round-2 runs use
sdpa. Round-1 numbers above, and earlier results made with eager bf16, carry this defect.

## 3. Visual-evidence decoding (round 2)

`scripts/rbd2/vdd_decode.py`. Each caption is decoded as a batch of two in lockstep: the real image and the 448x448
gray image of visual_decay E3, with the gray row always fed the real row's token. Every step then gives
log p_real and log p_gray, and every layer h_real and h_gray, without an extra pass over the caption.
r(t) is the measured gray-KL reliance curve (1 at the start, about 0.02 at 512 tokens).

- belief: A(o) += r(t) * (log p_real(o) - log p_gray(o)) over V_obj (visual evidence, the language prior cancels);
  logits(o) += lam * (1 - r(t)) * clip(A(o) / sum r) for objects not yet written
- steering: at layer L, h_real += alpha * (1 - r(t)) * (h_real - h_gray) on decode steps
- penalty: logits(o) -= pen * (1 - r(t)) for objects not yet written (with lam, a visual-evidence threshold)

Engineering: on Turing a bf16 matmul with 2 rows is 4.8x slower than with 1 (2.87 vs 0.60 ms), so decode-step
linears run row by row. Base is the same code path with alpha = lam = pen = 0.

Dev results, chair_dev60 (COCO train2014 dev, first 60) and amber_dev60 (AMBER 101-160):

| config | L | alpha | lam | pen | CHAIR_s | CHAIR_i | recall | AMBER CHAIR | Cover | Hal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | | 0 | 0 | 0 | 38.3 | 8.4 | 65.3 | 4.8 | 64.0 | 26.7 |
| s8a2 | 8 | 2 | 0 | 0 | 41.7 | 9.0 | 65.3 | 4.2 | 64.7 | 26.7 |
| s16a1 | 16 | 1 | 0 | 0 | 35.0 | 7.3 | 67.3 | 3.7 | 64.4 | 25.0 |
| s16a3 | 16 | 3 | 0 | 0 | 25.0 | 5.8 | 68.4 | 5.4 | 66.2 | 31.7 |
| s24a2 | 24 | 2 | 0 | 0 | 43.3 | 9.3 | 69.4 | 4.6 | 67.5 | 26.7 |
| bl1 | | 0 | 1 | 0 | 31.7 | 6.0 | 66.8 | 3.3 | 62.1 | 18.3 |
| pen1 | | 0 | 0 | 1 | 28.3 | 6.5 | 65.8 | 3.6 | 63.7 | 23.3 |
| s16a1bl1 | 16 | 1 | 1 | 0 | 25.0 | 5.9 | 67.3 | 4.0 | 63.7 | 26.7 |
| s16a2bl1 | 16 | 2 | 1 | 0 | 30.0 | 7.2 | 66.8 | 3.1 | 64.4 | 21.7 |
| s16a3bl1 | 16 | 3 | 1 | 0 | 30.0 | 6.5 | 67.9 | 3.1 | 62.8 | 18.3 |
| s20a2bl1 | 20 | 2 | 1 | 0 | 28.3 | 6.3 | 62.8 | 3.7 | 63.7 | 21.7 |
| **s16a1bl1p05** | 16 | 1 | 1 | 0.5 | 25.0 | 6.0 | 66.3 | 2.7 | 63.1 | 20.0 |
| s16a2bl1p05 | 16 | 2 | 1 | 0.5 | 28.3 | 6.6 | 65.3 | 2.9 | 62.5 | 20.0 |
| s16a1bl2p1 | 16 | 1 | 2 | 1 | 20.0 | 4.9 | 64.3 | 2.1 | 58.7 | 13.3 |

CIs per config are in [round2_dev/runs/*/score.json](round2_dev/runs/). Significant against Base at 60 images:
s16a3 CHAIR_s [-25.0, -1.7]; s16a1bl1p05 AMBER CHAIR [-4.2, -0.1]; s16a1bl2p1 CHAIR_s [-30.0, -6.7], AMBER CHAIR,
Hal, Cog lower and Cover lower [-8.5, -2.2]; bl1 AMBER Cog lower.

- Steering raises recall and coverage. Layer 16 works, layer 8 does nothing, layer 24 raises coverage without
  lowering hallucination, in line with the visual_decay E2 profile of ||h_real - h_gray|| by layer.
- The visual-evidence belief lowers hallucination on both dev sets, unlike the log-p belief of round 1.
- The penalty on top acts as an evidence threshold; the strongest setting (lam 2, pen 1) costs AMBER coverage.
- Selected on dev (rule in LEDGER.md): s16a1bl1p05, 'main' on the test sets.

Why AMBER moves less ([diagnostics/round2_amber_dev60_*](diagnostics/)): of the 22 hallucinated mentions in the
amber_dev60 Base captions, 6 are scorer artifacts (negation "no visible trees", "a carpet of apples", "a sun
design", "glass container", and a surfboard and a basketball net that are in the images but not in the
annotations, checked by eye), 6 are hedged guesses ("possibly a phone", "tracks or paths"), 6 are background
nouns not checked, and 4 are visual errors (one checked: "the sky" in an image without sky). AMBER captions are
about 100 tokens and their hallucinations sit at tokens 30-100 (median about 75); CHAIR captions are about 200
tokens and half of their hallucinations come after token 100, where the decay is larger.

## 4. Test sets (queued, not yet run)

| set | images | prompt | role |
|---|---|---|---|
| chair_test60 | first 60 CHAIR test (COCO val2014) | "Please describe this image in detail." | main |
| throne_test60 | first 60 sorted COCO val2017 ids | "Describe this image in detail." (THRONE) | main |
| amber_test60 | AMBER 1-60 | "Describe this image." (official) | standard report |
| amber_test60L | AMBER 1-60 | "Please describe this image in detail." | analysis (not the AMBER protocol) |

Runs: base, main, noSteer (alpha 0), noBelief (lam 0), constSched (both schedules 1 instead of 1 - r(t)).
THRONE: amazon-science/THRONE at 45badbb7, steps 2-3 of one-click.sh unchanged (flan-t5 base/large/xl at pinned
revisions, M = 3), micro P/R/F1/F0.5 at voting threshold 9; our captions replace step 1.
Object HalBench was dropped: its official scorer extracts objects with gpt-3.5-turbo-0613, which is retired, and its
rule-based mode is CHAIR's COCO-80 matching under other prompts.

## 5. Files

- `round1_test_pilot/`: rows and status of the four round-1 runs on CHAIR test 100 and AMBER 100, scores,
  dtype checks (fp16 is garbage on this model; bf16 and fp32 logs).
- `round2_dev/runs/`: rows, status and score.json for every dev config; `check/` is the batch-2 Base equivalence
  check. `round2_dev/queues/` are the exact run lists, `round2_dev/logs/` the worker logs and the attention check.
- `sets/`: every manifest used (image paths are the server's).
- `diagnostics/`: per-object breakdowns and hallucination positions printed by the diag_* scripts.
- `scripts/rbd2/` ran from /workspace/rbd_pilot on the server; paths inside are the server's. Entry points:
  `vdd_decode.py` (round-2 decoder), `worker.sh`/`vrun.sh`/`chain.sh` (queues), `score_sets.py` (CHAIR/AMBER),
  `score_throne.py` (THRONE), `queue.sh` + `score_chair100.py` + `score_amber100.py` (round 1),
  `prep*.py`, `bench_setup.sh`, `fetchx.py`, `pdl.py`, `wheels.py`, `install.sh` (data, models, environments).

## 6. Added on 2026-11-02 (after 5d8ec3e)

The rows and score.json files behind these numbers are on the server (`/workspace/vdd/`), which was unreachable when
this was committed; the numbers are copied from the scorer output.

AMBER dev with the long prompt (amber_dev60L: AMBER 101-160, "Please describe this image in detail."; analysis
only, not the AMBER protocol):

| config | CHAIR | Cover | Hal | Cog | tokens |
|---|---:|---:|---:|---:|---:|
| Base | 5.4 | 72.9 | 38.3 | 3.5 | 153 |
| bl1 | 5.4 | 68.8 | 38.3 | 4.2 | 156 |
| s16a1bl1 | 4.1 | 68.5 | 33.3 | 2.8 | 153 |
| s16a3 | 5.4 | 71.6 | 43.3 | 4.2 | 150 |

Only the Cover drops of bl1 [-6.9, -1.6] and s16a1bl1 [-7.5, -1.6] exclude zero. Longer captions did not make the
belief lower Hal (bl1 38.3, the same as Base).

chair_test60, base and main (the first runs of the queued test look):

| run | CHAIR_s | CHAIR_i | recall | F1 | tokens |
|---|---:|---:|---:|---:|---:|
| base | 38.3 | 12.9 | 68.4 | 76.7 | 173 |
| main | 30.0 | 11.6 | 63.6 | 74.0 | 177 |

dCHAIR_s [-20.0, +3.3], drecall [-10.0, +0.5]. On dev, main kept recall (+1.0); on test, recall fell 4.8, the same
direction as round 1 on CHAIR test 100 (CHAIR_s -12, recall -3.6).

No results here for the rest of the test queue (amber_test60, throne_test60, amber_test60L, ablations) or for the
chair_test60 sweep ([chair_test60_sweep/README.md](chair_test60_sweep/README.md); `scripts/rbd2/sweep.sh`,
`scripts/rbd2/sweep_decode.py` with `--bpos`).
