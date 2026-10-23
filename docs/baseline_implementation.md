# Baseline implementation status

These are independently written, paper-derived implementations. They are **not
the authors' repositories, not verified reproductions, and not measured GPU
results**. No upstream implementation code is vendored. Their shared runner
supports batch size 1, decoder caching, and one generation per hook context.
Each artifact records the source, extraction IDs, configuration, and adaptation
status. CPU tests validate mathematical properties and hook behavior, not the
published scores.

See the [three-model audit](model_compatibility.md) for specific upstream-code
departures, including VTI normalization/PCA and REVIS prefill-cache behavior.

| Method | Implemented core | Source |
|---|---|---|
| REVIS | Image/no-image visual contrast; blind rejected/neutral prior contrast; Gram–Schmidt purification; deepest separable layer; factual-risk quantile; low-alignment gate | [Sections 4.1–4.3, Eqs. 3–7](https://arxiv.org/html/2602.11824v2) |
| RUDDER-Beta | Normal-prefill attention-output pooling; unit CARD vector; increasing Beta gate; addition after attention | [Sections 3.2–3.4, Eqs. 1–5](https://arxiv.org/html/2511.10292v2) |
| VTI | Random patch corruption averaging minus original vision states; chosen-minus-rejected text states; PC1 directions; vision and text injection | [Section 4, Eqs. 1–4](https://arxiv.org/html/2410.15778v2) |
| GrAInS | Contrastive completion log likelihood IG; positive/negative prompt-token masking; last-token layer differences; two PC1s; norm-preserving steering | [Sections 3.1–3.3, Eqs. 1–5](https://arxiv.org/html/2507.18043v2) |

## Shared conventions and departures that must be reported

Decoder steering changes the final prefill position and the single current
decode position. Thus the first answer token can be affected without editing
the earlier prompt cache. This answer-prediction-position convention is common
across our local implementations; it is not a claim that upstream prompt-cache
intervention conventions are identical. Vision steering changes all vision
positions. Tuple-returning and tensor-returning blocks are supported. Hooks are
removed on exceptions. Model weights and gradients are not updated.

PCA is centered across examples and uses a deterministic sign aligned with the
mean contrast. A constant contrast has no defined principal component, so its
normalized mean is used and a zero contrast stays zero. This convention must be
checked against each authors' exact PCA preprocessing before claiming
reproduction. The final token is the final token emitted by the adapter's
teacher-forcing template; verify that it is the intended EOS/end-of-turn token.

Defaults are runnable starting configurations, not tuned paper hyperparameters.
Tune all methods with the same development budget, keep extraction/development
images out of the test set, and never compare these scores directly with scores
from another paper's table. Extraction stores CPU tensors; load artifacts only
from trusted sources, with `torch.load(..., weights_only=True)`.

## REVIS

Extraction requires `image`, `prompt`, `chosen`, and `rejected` for every row.
`neutral` may specify an explicit no-image uncertainty response. Otherwise a
fixed template is used, a departure from the paper's randomized refusal pool.
Rows with `role: "calibration"` form a separate factual/hallucinated calibration
set. Their image paths must not overlap vector-extraction rows. With no such
rows, an explicitly supplied `layer` and `threshold` are required and marked as
external calibration. The risk threshold uses `risk_quantile`, a fraction in
[0, 1], default 0.8. No suitable separating layer raises an error instead of
silently selecting the last layer. The vector retains its original magnitude.

## RUDDER

No offline direction is needed. CARD extraction and steering use the attention
residual update: `o_proj` output for Qwen/LLaVA, and
`post_attention_layernorm` output for Gemma 4. Gemma normalizes the projected
attention output before adding it to the residual, so steering `o_proj` output
would define a different intervention. A block pre-hook captures
the unnormalized residual input for the gate. During normal prefill, CARD pools
all image and prompt positions, using norm weighting by default or `pooling:
"mean"`. It is reused on cached decode steps. Higher cosine alignment gives a
larger gate; this is a trust gate, not a reversed hallucination-risk gate.

`steer_first_token: true` changes the last prefill attention update after CARD
extraction. Set false for a strict decode-only variant; its first output token
is then unmodified, which matters for one-token POPE responses. Report this
choice. `alpha`, `gate_k`, `gate_c`, `gate_min`, `gate_max`, and layer selection
need held-out tuning. This module does not silently perform a CHAIR-based
hyperparameter search or claim the paper's recall constraint has been met.

## VTI

Masks replace a sampled spatial patch with zero **in processor output space**.
Qwen's flattened patch rows, Gemma 4's batched patch rows, and LLaVA's
`[1,C,H,W]` tensors are supported. Gemma 4 receives patches in [0,1], so
zero is a black patch; this differs from zero in Qwen's normalized space.
The original vision hidden states are subtracted from the mean corrupted states,
not the reverse. Text contrasts use the same real image with both captions.

Default `vision_pooling: "position"` preserves position-specific visual
directions. It requires the same vision token grid throughout extraction and
evaluation. With variable-resolution Qwen images, preprocess fixed-size images
or explicitly request `vision_pooling: "mean"`. The latter pools positions and
broadcasts one vector per vision layer: **report it as a pooled VTI adaptation**,
not as the original positional implementation. Random masks use an isolated
seeded CPU generator. Input processing, masking value, number of masks, and PCA
sign handling can differ from original VTI and require parity checks.

## GrAInS

The first decoder-block input contains the already merged visual and textual
embeddings. A pre-hook integrates from a zero prompt embedding to that fused
prompt using trapezoidal IG. The objective is the sum of teacher-forced chosen
response log probabilities minus the sum for the rejected response. Only prompt
tokens are attributed; response embeddings remain fixed. Labels must mask
prompt positions with -100 and align with the expanded multimodal sequence.
Prompt tokenization and embeddings must match between preference responses.

The `top_k` strictly positive and strictly negative token indices are separately
zeroed at the same fused-embedding boundary. Hidden-state differences from the
unmasked prompt produce separate positive and negative PCA directions. Their
difference is added during inference with original-norm preservation. IG is
performed only offline, not on each generated token.

This is a **zero-baseline fused-embedding adaptation**, not pixel-space IG.
It includes visual and textual prompt embeddings but does not backpropagate
through the vision encoder to input pixels. No external attribution API,
reference model, extra test-time preference labels, or generated fake vectors
are used. Extraction needs real preference data and may be expensive even though
inference is cheap. Eager attention and gradient-enabled forwards are required.

This boundary does not intercept every input-dependent path on the new models.
Qwen3-VL retains its later DeepStack visual injections. Gemma 4 retains the
per-layer inputs computed before the first decoder block. Those auxiliary inputs
are held fixed during IG and token masking. The resulting attribution is
conditional on these paths, not complete removal of image/text evidence. Treat
these runs as architecture-specific adaptations; full-path attribution requires
an explicit new definition and validation before claiming GrAInS parity.

## Before reporting a comparison

Run a small real-model smoke test, inspect nonzero extracted vectors and gate
activation rates, and validate first-token effects. Compare equivalent prompts,
image resolution, token budgets, decoding, scoring, and development budgets.
Check each implementation against upstream outputs on a shared small input set
where licensing permits. Until that work is completed, describe all four as
local adapted implementations and keep `author_reproduction: false`.
