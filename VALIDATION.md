# Validation

Status date: 2026-10-23 (Asia/Seoul).

This repository implements a research hypothesis. No pretrained checkpoint
has been downloaded or evaluated during implementation. No GPU benchmark
scores, superiority claims, or author reproduction claims are available.

## Completed CPU checks

Local final check: **74 passed, no skips**, with Python 3.12.11,
PyTorch 2.8.0+cpu and Transformers 5.20.2. Ruff passed for `src`, `tests`,
and `scripts`. This includes fixture executions of the unmodified pinned CHAIR
port and Causal-HalBench scorer; it does not include real benchmark images.

- Pure tensor transformations and actual differentiable extraction for the four
  local adapted baselines.
- Causal head selection, random/magnitude controls, bounded gradient fitting,
  preservation KL, frozen base weights and restoration after exceptions.
- Actual Hugging Face Qwen2.5-VL, Qwen3-VL, Gemma 4 and LLaVA classes with tiny random configurations:
  image forward passes, GQA query-head splitting, all four baseline hooks,
  EPHS hooks, cached generation and restoration of baseline behavior.
- Gemma's RUDDER residual addition is checked after attention normalization.
  A separate tiny four-layer Gemma fixture tests EPHS at every layer with
  mixed attention head widths, shared KV states and per-layer inputs.
- The pinned Qwen3-VL-8B processor and tokenizer were loaded without weights:
  local 448px image processing worked and `Yes`/`No` were each one token.
- The pinned Gemma 4 E4B processor and tokenizer were loaded without weights:
  local 448px image processing returned 2,520 padded patch rows, of which
  2,304 were valid and mapped to 256 image tokens; `Yes`/`No` were each one
  token and supervised labels excluded the prompt.
- Manifest preparation, segmentation/background controls, train/test image-group
  isolation, invalid-answer handling, paired bootstrap and scorer adapters.
- CLI artifact/config/revision binding, completion/coverage checks, safe tensor
  artifact loading and comparison protocol checks.
- MMHal-Bench 96-question template conversion, REVIS/RUDDER response export,
  raw judge-output alignment and paired score comparison on synthetic fixtures.
  No real MMHal responses were generated or judged.
- Independent Python code review. No remaining critical/high-severity finding
  in the reviewed prototype at that stage.

Run `python -m pytest -q` and `ruff check src tests` for the current exact test
count. Passing tests do not validate research efficacy.

## GitHub verification status

The initial implementation was pushed to the private repository
`jinuk0211/vlm-grounding-steering` at commit
`596c0dd5cf2d666335723ca3485e443d08d30cd7`. Repository visibility and remote
upload were confirmed.

The initial push and one explicit workflow dispatch both ended with
`startup_failure` before creating any jobs. GitHub's API returned no check runs
or execution logs, so the underlying cause has not been established. These
are **not passing CI runs**; the 74-test result above is the local CPU result.
The workflow remains available for a later run.

- [Initial push run](https://github.com/jinuk0211/vlm-grounding-steering/actions/runs/39420075754)
- [Explicit dispatch run](https://github.com/jinuk0211/vlm-grounding-steering/actions/runs/39420507174)

## Required before scientific claims

1. Real-checkpoint smoke tests using the selected GPU, image processor and
   image/token budgets; inspect nonzero direction norms and intervention rates.
2. Upstream parity checks for each baseline, or report every architecture/data
   adaptation explicitly. Do not copy published scores into a common table.
3. Equal development/tuning budgets, including EPHS's extra corruption/labels,
   and held-out natural and edited images.
4. Multiple discovery samples/seeds, random and magnitude head controls,
   preservation-loss and intervention-position ablations.
5. Actual CHAIR/POPE/Causal-HalBench/MME/MMHal-Bench evaluation and failure review.

The [three-model audit](docs/model_compatibility.md) records substantive
upstream-code differences. In particular, current VTI and GrAInS adaptations
do not establish author-code parity, despite passing local mechanics tests.

Checkpoint weights, datasets, generated captions, tensors and credentials are
excluded from Git. External upstream sources are pinned and opt-in.
