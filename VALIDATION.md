# Validation

Status date: 2026-10-20 (Asia/Seoul).

This repository implements a research hypothesis. No pretrained 7B checkpoint
has been downloaded or evaluated during implementation. No GPU benchmark
scores, superiority claims, or author reproduction claims are available.

## Completed CPU checks

Local final check: **57 passed, no skips**, with Python 3.12.11,
PyTorch 2.8.0+cpu and Transformers 4.57.1. Ruff passed for `src`, `tests`,
and `scripts`. This includes fixture executions of the unmodified pinned CHAIR
port and Causal-HalBench scorer; it does not include real benchmark images.

- Pure tensor transformations and actual differentiable extraction for the four
  local adapted baselines.
- Causal head selection, random/magnitude controls, bounded gradient fitting,
  preservation KL, frozen base weights and restoration after exceptions.
- Actual Hugging Face Qwen2.5-VL and LLaVA classes with tiny random configurations:
  image forward passes, GQA query-head splitting, all four baseline hooks,
  EPHS hooks, cached generation and restoration of baseline behavior.
- Manifest preparation, segmentation/background controls, train/test image-group
  isolation, invalid-answer handling, paired bootstrap and scorer adapters.
- CLI artifact/config/revision binding, completion/coverage checks, safe tensor
  artifact loading and comparison protocol checks.
- Independent Python code review. No remaining critical/high-severity finding
  in the reviewed prototype at that stage.

Run `python -m pytest -q` and `ruff check src tests` for the current exact test
count. The latest final implementation verification is recorded in the Git
commit and GitHub Actions run; passing tests do not validate research efficacy.

## Required before scientific claims

1. Real-checkpoint smoke tests using the selected GPU, image processor and
   image/token budgets; inspect nonzero direction norms and intervention rates.
2. Upstream parity checks for each baseline, or report every architecture/data
   adaptation explicitly. Do not copy published scores into a common table.
3. Equal development/tuning budgets, including EPHS's extra corruption/labels,
   and held-out natural and edited images.
4. Multiple discovery samples/seeds, random and magnitude head controls,
   preservation-loss and intervention-position ablations.
5. Actual CHAIR/POPE/Causal-HalBench/MME evaluation and failure review.

Checkpoint weights, datasets, generated captions, tensors and credentials are
excluded from Git. External upstream sources are pinned and opt-in.
