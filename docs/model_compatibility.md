# Three-model comparison audit

Audited on 2026-10-23 against the local code and Transformers 5.20.2.

The three backbones can support a within-model experiment, but the current
implementation does **not** establish faithful reproduction of the four
published baselines or equal intervention effects across architectures.
All baseline runs remain local adaptations (`author_reproduction: false`).

## Architecture and intervention boundaries

| Model | EPHS head intervention | Additional considerations |
|---|---|---|
| Qwen2.5-VL-7B | Query-head outputs before decoder `o_proj` | Starting model for baseline parity checks; its vision features enter the initial decoder stream. |
| Qwen3-VL-8B | Same computational boundary | DeepStack injects additional visual features after early decoder blocks. Masking only the first block input leaves those paths intact. |
| Gemma 4 E4B-it | Same boundary, using the actual per-layer projection width | Sliding/full attention, differing head widths, shared KV states and per-layer inputs require separate validation. Attention output is normalized before the residual addition. |

EPHS uses query heads, not KV heads, and computes head width from each actual
`o_proj` input. RUDDER uses a distinct adapter boundary: Gemma's
`post_attention_layernorm` output and Qwen's `o_proj` output. A CPU regression
test checks that Gemma's last-position residual receives the intended CARD
addition after normalization, while earlier positions remain unchanged.

GrAInS currently attributes and masks the first decoder-block input only.
Qwen3 DeepStack inputs and Gemma per-layer inputs stay fixed. This is conditional
attribution to one stream, not removal of all input evidence. These variants
need an explicit protocol before being used to support a published GrAInS
comparison. VTI also changes vision features upstream of Qwen3 DeepStack, so its
effect need not match a model with a single visual-feature injection.

Sources: [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct),
[Gemma 4 documentation](https://huggingface.co/docs/transformers/model_doc/gemma4).
The audit also inspected the installed Transformers implementation at the
pinned package version, rather than assuming identical module names imply
identical residual boundaries.

## Original baselines and concrete departures

| Method | Backbones reported in the cited paper version | Local implementation departure |
|---|---|---|
| REVIS | Qwen2.5-VL, Qwen3-VL-8B, LLaVA and InternVL variants | The inspected author hook gates all prefill positions; our hook changes the last position. This changes the prompt cache intervention. |
| RUDDER | LLaVA-1.5, Idefics2, InstructBLIP, Qwen2.5-VL | The inspected Qwen code derives CARD from attention-output minus attention-input and gates on attention output. Our paper-derived version pools the residual update and gates on the block input. |
| GrAInS | LLaVA-1.6-7B, Qwen2.5-VL-7B, Gemma3-12B | Our fused-embedding IG and zero masking differ from the inspected public token-embedding attribution and PAD/UNK replacement. Gemma3 support does not establish Gemma4 parity. |
| VTI | LLaVA-1.5, InstructBLIP, Qwen-VL | The inspected author code injects after the MLP with norm preservation, image-mean masking and PCA over flattened layer features. Our implementation uses block-output addition, processor-space zero masking and layerwise PCA. These are substantive differences. |

Paper versions: [REVIS](https://arxiv.org/html/2602.11824v2),
[RUDDER](https://arxiv.org/html/2511.10292v2),
[GrAInS](https://arxiv.org/html/2507.18043v2),
[VTI](https://arxiv.org/html/2410.15778v2).

Inspected upstream code:
- [REVIS prefill hook, pinned commit](https://github.com/antgroup/Revis/blob/75f4d29d402d4c1c4dbb2c69e88e3aa50cd0c9cc/steering_utils/llm_layers.py)
- [RUDDER Qwen hooks, pinned commit](https://github.com/Akko000/RUDDER-Residual-Update-Directed-DEcoding-Regulation-/blob/bd149d8dbde6a9097dd02aea80f83d0bd8876761/Qwen/qwen_methods.py)
- [GrAInS attribution](https://github.com/duykhuongnguyen/GrAInS/blob/main/src/attribution/gradient/vlm_grad.py)
- [VTI hooks](https://github.com/shengliu66/VTI/blob/main/vti_utils/llm_layers.py)
  and [masking/PCA](https://github.com/shengliu66/VTI/blob/main/vti_utils/icv_utils.py)

The latter links point to mutable branches as inspected on the audit date;
pin their commits before a reproduction run. Differences found by inspection
are not an end-to-end evaluation of upstream correctness. A local win over
these adaptations alone does not demonstrate a win over the published methods.

## Comparison protocol still required

1. First validate baseline parity on Qwen2.5, then extend to Qwen3 and Gemma4.
   For each checkpoint, compare unmodified HF inference with the Base adapter,
   verify zero-strength restoration and run extraction, fitting, prefill and
   cached decoding on real inputs. Tiny random model hooks are insufficient.
2. Tune every method separately per model using the same held-out development
   examples and a declared search budget. Re-extract directions and re-fit gains;
   never transfer an artifact just because its dimensions fit.
3. Compare methods **within each model** using the same test examples, image
   processing, prompts, decoding and scorer. Report improvement relative to that
   model's Base alongside absolute scores and paired confidence intervals.
4. Record actual image-token counts, effective resolution, response length and
   truncation. A common 448px input and 128 output tokens do not equalize token
   budgets across different processors and tokenizers.
5. Treat configured layer lists and strengths as untuned starting points.
   Qwen's fixed layer indices and Gemma's full-attention-only candidate set are
   not matched by relative depth or candidate-head count. Declare the policy
   and include layer/type sensitivity checks before drawing architecture claims.

No pretrained checkpoint benchmarks have been run. A separate tiny four-layer
Gemma fixture covers mixed global/local head widths, shared KV states and
per-layer inputs, including steering and cached decoding at every layer. This
does not test the full E4B configuration or learned checkpoint. Real-checkpoint
smoke tests remain prerequisites for a research result.
