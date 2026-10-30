# RBD (reliability-weighted belief decoding) pilot, 2026-10-30

Code: `scripts/rbd_*.py`. Pilot only; nothing here is a final result.

## Offline belief AUROC (CHAIR test captions, first mentions, `rbd_offline.py`)
Hallucinated vs correct first mention. Belief over V_obj beats the token's own log p only through position:
position-matched AUROC drops (Qwen logp 0.667 vs belief 0.55; Gemma 0.694 vs 0.52), and measured r(t) is no
better than constant r. The pre-registered target (>= 0.80, measured r > constant r) was not met.

## Dev decoding (Qwen2.5-VL-7B, COCO train2014 dev 100, official CHAIR scorer, `rbd_decode.py`)
| config | CHAIR_s | CHAIR_i | recall | object F1 |
|---|---:|---:|---:|---:|
| Base (same code path) | 24.0 | 8.58 | 53.0 | 67.09 |
| belief lambda 0.1, measured r | 18.0 | 5.62 | 54.5 | 69.09 |
| belief lambda 0.3, measured r | 14.0 | 4.77 | 46.7 | 62.67 |
| belief lambda 1.0, measured r | 13.0 | 5.16 | 43.7 | 59.84 |
| belief lambda 0.3, constant r | 15.0 | 3.71 | 52.4 | 67.86 |
| uniform control lambda 0.3 | 15.0 | 3.11 | 47.0 | 63.30 |
| steering only (alpha 1, layer 16) | 27.0 | 7.88 | 51.5 | 66.06 |

100 images: differences of a few points are within noise. Still running when this was written: uniform and
constant-r controls at lambda 0.1, lambda 0.05 and 0.03, Gemma Base and lambda 0.3.
