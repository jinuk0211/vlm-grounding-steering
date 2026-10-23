# Data preparation and common evaluation

All functions consume local, separately downloaded datasets. No images, annotations,
model weights, credentials, or upstream pickle caches are included in this project.
Builders run on CPU with Pillow and NumPy; COCO RLE masks additionally require
`pycocotools`. Dataset use remains subject to the original dataset licenses.

## Manifest contract

One JSONL object per question or caption:

```json
{"id":"pope:adversarial:0","group_id":"coco:1171","image":"/data/val2014/COCO_val2014_000000001171.jpg","prompt":"Is there a bird in the image?","label":"yes","task":"pope","split":"test","metadata":{"subset":"adversarial"}}
```

- `id` is globally unique across all input manifests.
- `group_id` identifies the **original image**, including all edited derivatives.
  COCO `...000000001171.jpg` and Causal-HalBench
  `...000000001171_001.jpg` both map to `coco:1171`.
- `task` is `pope`, `caption`, `causal`, `mme`, or `mmhal`. Caption and MMHal labels are omitted.
- `split` is `discovery`, `fit`, `tune`, or `test`. Split names describe research
  roles, not COCO's train/val names.
- Images may have multiple questions or tasks in the **same** split. Reusing any
  original image group across splits is an error.
- Relative paths resolve relative to the manifest file. All builders emit absolute
  paths. Move manifests and datasets together, or regenerate on the GPU host.

`read_manifest(path)` validates required fields, IDs, files and group isolation.
`validate_disjoint([discovery, fit, tune, test])` accepts record lists or manifest
paths. It also rejects a resolved image path assigned to inconsistent groups/splits.
These checks rely on correct original-image IDs; they are not perceptual duplicate
detection. Keep original IDs when adding new datasets.

## Builders

```python
from vlm_grounding.data import (
    prepare_coco, prepare_pope, prepare_causal, prepare_mme, prepare_mmhal,
    read_manifest, validate_disjoint, write_manifest,
)

pope = prepare_pope("/data/pope/coco_pope_adversarial.json",
                    "/data/val2014", subset="adversarial")
causal = prepare_causal("/data/causal/qa.json", "/data/causal/images")
mme = prepare_mme("/data/MME_Benchmark_release_version")
mmhal = prepare_mmhal("/data/mmhal/response_template.json", "/data/mmhal/images")
development = prepare_coco(
    "/data/annotations/instances_train2014.json", "/data/train2014",
    "data/derived", seed=42, discovery_count=200, fit_count=200,
    tune_count=200, test_count=0,
    exclude_group_ids={r["group_id"] for r in pope + causal},
)
test_captions = prepare_coco(
    "/data/annotations/instances_val2014.json", "/data/val2014",
    "data/derived_test", discovery_count=0, fit_count=0, tune_count=0,
    test_count=2000,
)
validate_disjoint([development, test_captions, pope, causal, mme])
for split in ("discovery", "fit", "tune"):
    write_manifest([r for r in development if r["split"] == split],
                   f"data/{split}.jsonl")
```

Counts are **image counts**, not row counts. The default COCO split builder is
deterministic after sorting image IDs and applying its seeded shuffle. Use one
combined builder call for development splits; never invoke separate calls with
the same seed and pretend the resulting subsets are disjoint.

`prepare_pope` supports the official JSONL schema
`question_id, image, text, label` and the equivalent JSON array. Pass distinct
`subset` names for random/popular/adversarial files. All three remain separate
evaluation cohorts.

`prepare_causal` expects the original array/JSONL containing
`id, image_name, question, answer, type, tag`; it preserves each original QA entry
in `metadata.causal_qa` for the official scorer. Set `image_root` to the directory
that makes the supplied image names resolve.

`prepare_mme` reads category directories with `images/` and
`questions_answers_YN/`, or categories containing images and text side by side.
Question files contain `question<TAB>yes/no`. Preserve the official categories.

`prepare_mmhal` reads the ordered official 96-question `response_template.json`
and local images from the official `test_data.zip`. It preserves the original
question index, image ID, question type, image contents and reference answer in
metadata. Only the question and local image are supplied to model inference;
reference answers are for external judging. `export-mmhal` requires all 96
predictions in original order and writes the official response JSON schema.
`compare --mmhal-evaluation METHOD RESPONSES JUDGE_OUTPUT` consumes the raw JSON
list saved by the official `eval_gpt4.py` for each method. It verifies the
judged responses match the model outputs before computing mean rating (0–6),
hallucination rate (rating 0–2), and question-type means. Ambiguous or missing
ratings fail instead of silently becoming zero. The code cannot verify that a
supplied judgement file used the official prompt or the same judge model for
all methods; retain the judge model/version and scorer script in experiment
provenance. When raw judge responses declare their model, comparison rejects
different model labels across methods and records the value. The official script uses a legacy OpenAI SDK call, so check its
runtime and selected judge model before running paid evaluations.

## Diagnostic image construction

Each discovery row has `image`, `object_masked_image`, `control_image`, one
positive yes/no question and its original `label="yes"`. The label describes the
**original** image; the masked image is a diagnostic input, not another labeled
training example.

The builder unions all COCO segmentation instances of the selected category,
so another annotated instance of the same queried category is not left visible.
It fills that segmentation with RGB `(128,128,128)`. It searches for a translation
of the identical mask shape wholly inside background according to **all annotated
COCO objects**, and applies the same fill there for the control image. Target and
control have exactly the same number of altered pixels. If no such control exists,
the image is skipped; insufficient eligible images cause an explicit error.

Polygons are rasterized with Pillow, **not approximated by bounding boxes**.
Pillow boundary rasterization can differ by edge pixels from COCO's official mask
API. RLE uses `pycocotools`; its absence is an error, not a silent box fallback.
The selected mask covers 0.5–30% of the image, and smaller feasible category masks
are considered first. This biases the diagnostic sample and must be reported.

These are gray occlusions, not natural counterfactuals or inpainting. Background
control excludes annotated objects, but cannot exclude unannotated objects or
semantic background evidence. Location, context and edit artifacts can still
confound attribution. Report this diagnostic alongside unedited natural-image
evaluation and Causal-HalBench; never claim that oracle patching proves deployed
head scaling works.

Fit and tune each contain one annotation-present and one annotation-absent object
question per image, plus a caption prompt. Both questions include `chosen` and
`rejected` strings (`Yes.`/`No.`), explicitly marked as annotation-derived,
task-adapted preference pairs. These are **not** the original papers' caption
preference datasets. Annotation-absent does not prove visual absence outside the
COCO taxonomy/annotation coverage. Filter `task == "pope"` for preference fitting.
Test construction emits caption prompts only.

## Prediction and scoring contract

Keep manifest fields and add `prediction` (decoded text), `method`, and preferably
`elapsed_seconds` and `generated_tokens`. Timing should exclude model loading,
include the method's required inference work, and use the same synchronization/
warmup policy. `match_predictions(manifest, outputs)` merges external ID-only
outputs and rejects missing, extra, duplicate, or changed-input predictions.

`evaluate_predictions(records)` requires a single task and method.
`compare_methods(all_outputs, baseline="base", bootstrap_samples=1000, seed=42)`
produces:

- metrics by method and task/subset;
- paired metric differences with percentile 95% intervals;
- bootstrap provenance, resampling **original image groups**, not questions.

All methods must have exactly the same IDs, prompts, images, labels and splits.
POPE subsets are separate. Missing outputs never silently shrink the evaluation
set. Repeated bootstrap copies use new identifiers while preserving group
membership within each copy.
Generated outputs carry a `protocol_fingerprint` for the shared checkpoint,
revision and actual generation settings. Comparisons reject different nonempty
fingerprints. Imported external outputs without fingerprints can be evaluated,
but the report marks `protocol_verified=false`; supply the full generation
provenance before treating such results as a controlled comparison.

The binary parser accepts leading `Yes`/`No`, optionally `Answer:`, followed by
ordinary punctuation or whitespace. It never searches for a word anywhere in a
long answer. Unparseable responses remain incorrect in the accuracy denominator;
invalid positive responses count as misses for recall/F1. Precision is undefined
(`null`) if there are no positive predictions. Report invalid rate and yes ratio
with accuracy/precision/recall/F1; this is a documented common parser, not a claim
to reproduce every upstream parser.

MME reports category `100*(accuracy + both-questions-correct accuracy)` and their
sum, using the same strict parser. Exactly two answers per category/image are
required. The output is named with its parser protocol; use the official scorer
as an additional reproduction check when claiming original-paper MME results.

## CHAIR: real external scorer, no approximate substitution

The [original CHAIR implementation](https://github.com/LisaAnne/Hallucination)
uses Python 2 and additional caption-evaluation artifacts. This project calls the
[Maxlinn Python 3 port](https://github.com/Maxlinn/CHAIR-metric-standalone) through an
**explicit local module path**. The port retains the original lexical object
mapping and adds recall. The port's README does not warrant exact numerical
equivalence with every original setup; report its pinned revision in a paper.
The adapter records a SHA-256 of the actual scorer source.

```python
from vlm_grounding.evaluation import run_official_chair, evaluate_predictions

scored = run_official_chair(
    caption_predictions,
    "external/CHAIR-metric-standalone/chair.py",
    "/data/annotations",
    output_path="runs/base/chair_sentences.json",
)
metrics = evaluate_predictions(scored)
```

The genuine [PatternLite 3.6](https://pypi.org/project/PatternLite/) distribution
provides the original `pattern.en` module with fewer unrelated dependencies.
This exact dependency and the unmodified pinned CHAIR port were CPU-tested on
Python 3.12 using tiny COCO-format fixtures. PatternLite's published compatibility
list only covers Python 3.6–3.8; our smoke test is narrower than a complete
compatibility guarantee. Install `PatternLite==3.6`, `nltk` and `tqdm`; do not
substitute the unrelated similarly named `pattern` package.

PatternLite imports WordNet even though CHAIR only uses its singularizer. Set up
all resources before an offline GPU run:

```bash
python -m pip install -e ".[eval]"
python scripts/setup_eval.py
```

Use the active virtual environment's Python. The helper downloads to
`.cache/nltk_data` (override with `--download-dir`), safely extracts the official
WordNet archive and tests the real tokenizer/singularizer. It prints the exact
`NLTK_DATA` setting for Linux/macOS and PowerShell; apply that setting before
evaluation, including when resuming in another shell. It validates all archive
member paths before extraction.

Explicit WordNet extraction is necessary because the NLTK downloader leaves
this corpus zipped, but PatternLite looks for the unpacked directory and
otherwise attempts its own download. Do not modify the upstream scorer or replace
its tokenizer/singularizer.

The annotation directory must contain all four official files:
`captions_train2014.json`, `captions_val2014.json`,
`instances_train2014.json`, `instances_val2014.json`.
The adapter rebuilds annotations from these JSONs and does **not** deserialize
downloaded pickle caches. Each scored image must exist in annotations.

Alternatively `attach_chair_scores(predictions, path_to_output_json)` consumes
the scorer's `sentences` output. It requires exact image coverage and an identical
caption string, then uses the scorer's normalized generated/ground-truth objects
to compute micro CHAIR_i, sentence CHAIR_s, object recall and correct distinct
objects. Recall is within CHAIR's COCO annotation/caption-derived vocabulary,
not all aspects of caption informativeness.

Without these outputs, caption evaluation explicitly reports
`chair_status="not_scored"` and length only. A missing scorer is never replaced
with regex matches or a zero hallucination score. Zero object mentions make
CHAIR_i undefined; the upstream port may raise a division-by-zero error, which the
adapter surfaces. No hallucination claim should be made from an unscored run.

## Causal-HalBench official metrics

`run_official_causal(records, "external/Causal-HalBench/metric.py")` imports the
[official script](https://github.com/zhexu-ustc/Causal-HalBench/blob/main/metric.py),
exports preserved QA metadata and parsed responses, and calls its `evaluate`.
Invalid answers are kept as an impossible label, hence incorrect. Always first
call `match_predictions(full_manifest, outputs)` to prove complete coverage.
The scorer source hash and record count are returned.

The checked official script names the absent-object difference `nac`, whereas
the README calls it AAC. Preserve the script's output names; do not invent a
renaming/formula. The official script includes CAC, NAC, CHR and paired CPR.
Report these alongside absolute yes/no accuracy from the common evaluator.
No official Causal-HalBench scores are fabricated when its scorer is unavailable.

## Validation scope

CPU tests cover real polygon-versus-box behavior, mask/control area and overlap,
deterministic splitting, balanced pair labels, missing images, duplicate IDs,
cross-split derivative leakage, incomplete output coverage, invalid-answer
denominators, image-group bootstrap and CHAIR output matching/aggregation.
An interface-fixture test runs without upstream code. Two optional integration
tests execute the **actual pinned CHAIR port and Causal-HalBench scorer** on tiny
annotation/response fixtures when their fetched sources/dependencies are present.
Both passed under Python 3.12 on CPU. They skip in offline CI if external sources
or NLTK resources have not been installed. This establishes actual scorer API and
dependency execution, not full COCO scoring parity, model performance, or GPU
execution.
