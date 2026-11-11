# First variant (probability belief): held-out pilot (2026-11-01)

Section 1 of the original `results/rbd_20261101/README.md` (branch `rbd-round2-20261101`, `1d2d676`). Only the sentence that pointed to
section 2 now says what that section was. The rest of that file covered the round-2 development search, which is not part
of the paper and is not on this branch.

## 1. RBD on held-out sets (round 1)

`scripts/rbd_decode.py` unchanged, lambda 0.1 (chosen on dev), bf16 on RTX 2080 Ti, Base re-run under the same
conditions. These runs use the repo's eager attention, which writes ' addCriterion' into some bf16 captions (section 2 of the original file).

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

