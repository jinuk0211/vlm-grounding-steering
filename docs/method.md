# EPHS: Evidence-Preserving Head Steering

EPHS is a provisional experimental method. It has not been demonstrated to
improve a real VLM, and the name does not assert novelty. Head-level causal
selection is closely related to HONES; visual steering overlaps with REVIS,
RUDDER, VTI, GrAInS, SHARP, and other contemporary methods.

## Research question

Can a small set of attention-head gains reduce false object-existence answers
without losing correct visual information, compared with stronger residual-space
steering baselines under identical evaluation conditions?

The primary claim to test is a better hallucination–information-preservation
tradeoff. A reduction in caption length or an increased tendency to answer “no”
is not sufficient evidence.

## Stage 1: causal discovery on development images

Each positive object-existence example supplies a clean image, an image with the
target object masked, and an image with a comparable background/control region
masked. The question must be unchanged. Mask construction, area, image encoding,
and artifacts must be controlled; this package consumes prepared paired images
and does not manufacture a causal benchmark by itself.

For a clean-correct example where object masking reduces the yes-minus-no answer
margin, EPHS records each candidate attention head at the last prompt token. It
hooks the **input to the attention output projection** (`o_proj`), before head
outputs are mixed. The input is split using the number of **query heads**. Grouped
query attention's KV-head count must not be used for this split.

EPHS replaces one masked-input head activation with its clean counterpart and
measures recovery of the answer margin. It repeats that replacement on the
background-control input. A head's score is object-mask recovery minus the
absolute control-input change, normalized by the clean/object-mask margin gap
with a fixed denominator floor. Scores are averaged within image groups, then
aggregated by their median across groups to reduce duplicate-question and outlier
effects. The top `top_k` heads are selected.

This patching is an **oracle diagnostic**. A successful patch does not establish
that rescaling the same head will work. Clean donors, paired images, and ground
truth are used only in development discovery, never at deployed inference.

Defaults: `top_k=8`, `control_penalty=1.0`, `normalization_floor=1.0`,
`min_margin_drop=0.05`. Choose candidate `layers`, candidate `heads`, and the
number of development examples before viewing held-out evaluation. Enumerating
every head requires two extra forward passes per candidate per retained example.
Use a preregistered limited layer/head candidate set for the pilot.

`selection_mode=random` samples from the same candidate pool with a fixed seed.
`selection_mode=magnitude` selects the largest clean activation RMS. Both use the
same paired data and candidate pool as the causal selection, so subsequent gain
fitting budgets can remain identical. These modes currently still compute the
causal diagnostics, so their calibration runtime is not an optimized baseline.

## Stage 2: fit only bounded head gains

The base VLM remains frozen. EPHS learns one scalar per selected head. A bounded
transformation restricts each head multiplier to `1 ± max_gain` (`0.25` by
default); heads cannot change sign or be zeroed. Both positive and negative
object-existence examples are required. Training samples the two labels equally,
cycling the smaller class rather than silently optimizing an unbalanced set.

The objective combines a binary yes/no margin loss, KL preservation of the full
next-token distribution on examples the unmodified model answered correctly, and
a small squared-gain penalty. Reference distributions come from the unmodified
model. `preserve=false` can explicitly exclude edited/corrupted development cases
from the KL term, but must be decided by the data protocol rather than outcomes.

Defaults: `epochs=2`, `lr=0.03`, `max_gain=0.25`, `kl_weight=1.0`,
`l2_weight=0.01`, `seed=0`, `positions="last"`. `max_steps`, if supplied, must be even to preserve label
balance. These defaults are starting configurations, not validated hyperparameters.
The target first-token yes/no definition comes from the adapter and must be shared
by discovery and fitting. It is not a full answer-sequence likelihood.

Preserving this prompt-level next-token distribution does **not** guarantee
caption quality or preservation of multi-token behavior. Those claims require
separate held-out caption and capability evaluations.

## Stage 3: single-input deployment

Only the fitted head indices, scalar gains, and position setting are loaded. By
default, each forward pass scales the selected heads at the last token position:
the final prompt token during prefill and the current generated token during
cached decoding. Earlier prefill positions remain unchanged. This aligns the
intervention positions with the common baseline comparison protocol.

`positions="all"` is an explicit ablation that scales every prefill position.
Fitting and deployment must use the same position setting; it is saved in the
artifact, and a conflicting deployment override is rejected. There is
no test-time donor, ground-truth answer, object mask, per-input backward pass,
additional model, or feature dictionary.

Hooks are scoped to a context manager and removed on normal exit or exceptions.
Currently batch size is restricted to one. Calling `steer` with `strength=0`
provides an identity intervention check; any strength that exceeds the fitted
artifact's gain bound is rejected.

## API and artifacts

```python
selection = discover(adapter, discovery_records, discovery_config)
artifact = fit(adapter, fit_records, selection, fit_config)
with steer(adapter, artifact):
    output = adapter.forward(batch)  # Or the model's normal generation call.
```

The adapter supplies `model`, `layers`, `attn_projections`, `num_heads`, `device`,
`prepare(image_path, prompt, answer=None)`, `forward(batch)`, and
`answer_margin(batch)`. `forward` returns `.logits` or `{"logits": tensor}` with
shape `[1, sequence, vocabulary]`. `answer_margin` must return a differentiable
scalar and execute the hooked model. Batch-one inputs must end at the actual
answer prediction position; right padding is unsupported.

Discovery records contain `id`, `group_id`, `image`, `object_masked_image`,
`control_image`, `prompt`, and `label="yes"`. Fitting records contain `id`,
`group_id`, `image`, `prompt`, and a `yes`/`no` label. Artifacts are JSON-safe;
fitted `selected_heads` entries contain a `delta`, where the multiplier is
`1 + delta`. Architecture dimensions are checked when loading an artifact.
The experiment runner should additionally record the exact checkpoint, revision,
processor configuration, dataset/image manifests, and package versions.

## Comparison and falsification

Compare the unmodified model, REVIS, RUDDER, GrAInS, and VTI under common image
processing, prompts, decoding settings, evaluator, and tuning data budgets.
HONES is a close method baseline if the head-localization claim is developed.
Include random-head and activation-magnitude selection with identical gain
fitting, and a zero-gain check. A layer/residual scaling control can distinguish
local selection benefits from a generic change in activation magnitude.

Report POPE precision, recall, F1, accuracy, yes ratio, and unparsed outputs.
Report COCO CHAIR_s and CHAIR_i alongside correct object mentions, object recall,
and actual caption length. Report MME or another held-out capability measure.
The CHAIR taxonomy does not certify correctness of all attributes or relations.

Split **image IDs**, not merely question rows or benchmark names: COCO-derived
POPE and CHAIR may overlap. Discovery, fitting, tuning, and final evaluation must
have disjoint image groups. Use paired bootstrap over images, not correlated
question rows. Causal-HalBench should remain a held-out diagnostic if used; its
edited examples must not be silently reused as training donors.

Stop or revise the method if patching finds no robust causal signal, gains fail
to improve held-out development behavior, or the strongest reproduced baseline
dominates at matched object recall and generation budget. No GPU results,
novelty claim, or expected acceptance rate is implied by this implementation.
