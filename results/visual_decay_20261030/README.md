# Visual decay along the caption, 2026-10-30

Base CHAIR captions (500 per model), teacher-forced. Code: `scripts/visual_decay_*.py`, notes: `docs/visual_decay_ko.md`.

| file | content |
|---|---|
| `qwen-visual-decay-summary.json`, `gemma-visual-decay-summary.json` | per-bin summaries of E1 (attention), E2 (logit lens, representations), E3 (interventions) |
| `visual_decay_report.html` | charts and tables for both models |

**Do not use the `probes` / `image_tokens_ceiling` fields in these files, nor the probe charts in the report.**
That run pooled cross-validation scores across folds, which biases AUROC below 0.5 for rare classes
(an input with no information scored 0.38). The code has since been fixed (fold-wise AUROC), but the
corrected probes were never run, so there are no valid probe results. Everything else (E1, E3, the logit-lens
and hidden-state-change parts of E2) is unaffected.

The Qwen captions are the stored Base captions, which match greedy decoding with the model's
`repetition_penalty=1.05` on 93.7% of tokens (85% by raw argmax); Gemma's match on 98.5%.
