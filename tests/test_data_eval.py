"""Behavioral checks for leakage prevention, denominator safety, and real mask use."""

import json
import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from vlm_grounding.data import (
    image_group_id,
    prepare_coco,
    prepare_mmhal,
    prepare_pope,
    read_manifest,
    validate_disjoint,
    write_manifest,
)
from vlm_grounding.evaluation import (
    attach_chair_scores,
    attach_mmhal_scores,
    compare_methods,
    evaluate_predictions,
    match_predictions,
    mmhal_responses,
    parse_yes_no,
    run_official_chair,
    run_official_causal,
)


class DataEvalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / "COCO_val2014_000000001171.jpg"
        Image.new("RGB", (32, 32), (255, 255, 255)).save(self.image)

    def record(self, identifier="q1", **changes):
        return {
            "id": identifier, "group_id": "coco:1171", "image": str(self.image),
            "prompt": "Is there a bird?", "task": "pope", "split": "test",
            "label": "yes", "prediction": "Yes.", "method": "base",
            **changes,
        }

    def test_repeated_questions_allowed_but_original_edit_split_leak_rejected(self):
        first = self.record()
        second = self.record("q2", prompt="Is there a cat?", label="no")
        self.assertEqual(validate_disjoint([first, second]), {"test": 2})
        edited = self.root / "COCO_val2014_000000001171_001.jpg"
        Image.new("RGB", (32, 32)).save(edited)
        self.assertEqual(image_group_id(self.image.name), image_group_id(edited.name))
        second.update(image=str(edited), split="fit")
        with self.assertRaisesRegex(ValueError, "Image-group leakage"):
            validate_disjoint([first, second])

    def test_duplicate_ids_missing_files_and_file_aliases_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_disjoint([self.record(), self.record()])
        with self.assertRaises(FileNotFoundError):
            validate_disjoint([self.record(image=str(self.root / "missing.jpg"))])
        with self.assertRaisesRegex(ValueError, "different groups"):
            validate_disjoint([self.record(), self.record("q2", group_id="wrong", split="fit")])

    def test_pope_round_trip_and_local_path_boundary(self):
        source = self.root / "pope_adversarial.json"
        source.write_text(json.dumps([
            {"question_id": 0, "image": self.image.name, "text": "Bird?", "label": "yes"},
            {"question_id": 1, "image": self.image.name, "text": "Cat?", "label": "no"},
        ]), encoding="utf-8")
        records = prepare_pope(source, self.root)
        output = self.root / "manifest.jsonl"
        write_manifest(records, output)
        self.assertEqual(read_manifest(output), records)
        malicious = json.loads(source.read_text())[0]
        malicious["image"] = "../outside.jpg"
        source.write_text(json.dumps([malicious]), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "escapes"):
            prepare_pope(source, self.root)

    def test_mmhal_official_template_round_trip_and_judge_alignment(self):
        template = self.root / "response_template.json"
        rows = [{
            "question_type": ("attribute", "adversarial")[index % 2],
            "question_topic": "outdoor", "image_id": "fixture-image",
            "image_src": f"https://example.org/{self.image.name}",
            "image_content": ["Bird"], "question": f"Question {index}?",
            "gt_answer": "A bird.", "model_answer": "",
        } for index in range(96)]
        template.write_text(json.dumps(rows), encoding="utf-8")
        manifest = prepare_mmhal(template, self.root)
        self.assertEqual(len(manifest), 96)
        self.assertEqual(manifest[0]["image"], str(self.image))
        self.assertNotIn("label", manifest[0])
        manifest_path = self.root / "mmhal.jsonl"
        write_manifest(manifest, manifest_path)
        self.assertEqual(read_manifest(manifest_path), manifest)

        predictions = [{**row, "prediction": "A bird.", "method": "revis"}
                       for row in manifest]
        response_path = self.root / "responses.json"
        response_path.write_text(json.dumps(mmhal_responses(predictions, template)),
                                 encoding="utf-8")
        judgement_path = self.root / "judgements.json"
        judgement_path.write_text(json.dumps([
            {"choices": [{"message": {"content": f"Explanation. Rating: {index % 7}"}}]}
            for index in range(96)
        ]), encoding="utf-8")
        scored = attach_mmhal_scores(predictions, response_path, judgement_path)
        metrics = evaluate_predictions(scored)
        self.assertEqual(metrics["mmhal_status"], "scored")
        self.assertAlmostEqual(metrics["mmhal_average_score"], sum(i % 7 for i in range(96)) / 96)
        self.assertAlmostEqual(metrics["mmhal_hallucination_rate"],
                               sum(i % 7 < 3 for i in range(96)) / 96)
        self.assertEqual(set(metrics["mmhal_type_scores"]), {"attribute", "adversarial"})
        self.assertEqual(evaluate_predictions(predictions)["mmhal_status"], "not_scored")
        responses = json.loads(response_path.read_text(encoding="utf-8"))
        responses[0]["model_answer"] = "a different model"
        response_path.write_text(json.dumps(responses), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "differs"):
            attach_mmhal_scores(predictions, response_path, judgement_path)
        responses[0]["model_answer"] = "A bird."
        response_path.write_text(json.dumps(responses), encoding="utf-8")
        judgments = json.loads(judgement_path.read_text(encoding="utf-8"))
        judgments[0]["choices"][0]["message"]["content"] = "No rating available"
        judgement_path.write_text(json.dumps(judgments), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "exactly one rating"):
            attach_mmhal_scores(predictions, response_path, judgement_path)

    def test_coco_real_polygon_mask_matched_background_and_balanced_labels(self):
        image_records, annotations = [], []
        for index in range(5):
            path = self.root / f"COCO_train2014_{index:012d}.jpg"
            Image.new("RGB", (32, 32), "white").save(path)
            image_records.append({"id": index, "file_name": path.name, "width": 32, "height": 32})
            annotations.append({
                "id": index, "image_id": index, "category_id": 1,
                "segmentation": [[2, 2, 10, 2, 2, 10]], "bbox": [2, 2, 8, 8],
            })
        source = self.root / "instances.json"
        source.write_text(json.dumps({
            "images": image_records, "annotations": annotations,
            "categories": [{"id": 1, "name": "bird"}, {"id": 2, "name": "cat"}],
        }), encoding="utf-8")
        first = prepare_coco(
            source, self.root, self.root / "derived", seed=7,
            discovery_count=1, fit_count=1, tune_count=1, test_count=1,
        )
        second = prepare_coco(
            source, self.root, self.root / "derived", seed=7,
            discovery_count=1, fit_count=1, tune_count=1, test_count=1,
        )
        self.assertEqual(first, second)
        discovery = first[0]
        masked = np.asarray(Image.open(discovery["object_masked_image"]))
        control = np.asarray(Image.open(discovery["control_image"]))
        target_pixels = (masked == 128).all(axis=2)
        control_pixels = (control == 128).all(axis=2)
        self.assertEqual(int(target_pixels.sum()), int(control_pixels.sum()))
        self.assertFalse((target_pixels & control_pixels).any())
        self.assertFalse(target_pixels[9, 9])  # bbox interior outside the triangle
        self.assertTrue(target_pixels[3, 3])
        for split in ("fit", "tune"):
            pairs = [r for r in first if r["split"] == split and r["task"] == "pope"]
            self.assertEqual({r["label"] for r in pairs}, {"yes", "no"})
            self.assertTrue(all(r["chosen"] != r["rejected"] for r in pairs))
        validate_disjoint(first)

    def test_invalid_answers_are_not_deleted_or_treated_as_negative(self):
        rows = [
            self.record("a", prediction="I cannot tell."),
            self.record("b", prediction="No.", label="no"),
            self.record("c", prediction="Yes.", label="yes"),
            self.record("d", prediction="unclear, yes maybe", label="no"),
        ]
        metrics = evaluate_predictions(rows)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["negative_accuracy"], 0.5)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["invalid_count"], 2)
        self.assertIsNone(parse_yes_no("yes/no"))
        self.assertIsNone(parse_yes_no("Nobody is visible"))
        self.assertEqual(parse_yes_no(" Answer: No, there is not."), "no")

    def test_missing_outputs_or_changed_inputs_cannot_win_comparison(self):
        base = [self.record("a"), self.record("b")]
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            match_predictions(base, [{"id": "a", "prediction": "yes", "method": "ours"}])
        with self.assertRaisesRegex(ValueError, "IDs differ"):
            compare_methods(base + [self.record("a", method="ours")], bootstrap_samples=2)
        bad = [self.record("a", method="ours", label="no"), self.record("b", method="ours")]
        with self.assertRaisesRegex(ValueError, "unmatched label"):
            compare_methods(base + bad, bootstrap_samples=2)

    def test_paired_bootstrap_uses_original_image_groups(self):
        base = [
            self.record("a", prediction="No."),
            self.record("b", prediction="No."),
            self.record("c", group_id="coco:2"),
        ]
        other = [{**row, "method": "ours", "prediction": "Yes."} for row in base]
        report = compare_methods(base + other, bootstrap_samples=40, seed=3)
        delta = next(row for row in report["paired_deltas"] if row["metric"] == "accuracy")
        self.assertEqual(delta["image_groups"], 2)
        self.assertAlmostEqual(delta["delta"], 2 / 3)
        self.assertEqual(delta["valid_bootstrap_samples"], 40)
        self.assertEqual(report, compare_methods(base + other, bootstrap_samples=40, seed=3))

    def test_protocol_mismatch_rejected_and_missing_protocol_not_verified(self):
        base = self.record(protocol_fingerprint="same")
        other = self.record(method="ours", protocol_fingerprint="different")
        with self.assertRaisesRegex(ValueError, "protocol fingerprints"):
            compare_methods([base, other], bootstrap_samples=0)
        del other["protocol_fingerprint"]
        report = compare_methods([base, other], bootstrap_samples=0)
        self.assertFalse(report["protocol_verified"])
        other["protocol_fingerprint"] = "same"
        self.assertTrue(compare_methods([base, other], bootstrap_samples=0)["protocol_verified"])

    def test_chair_aggregates_mentions_and_preserves_correct_object_count(self):
        predictions = [
            self.record("a", task="caption", prediction="bird cat", group_id="coco:1"),
            self.record("b", task="caption", prediction="bird bird bird", group_id="coco:2"),
        ]
        output = {"sentences": [
            {"image_id": 1, "caption": "bird cat", "mscoco_generated_words": ["bird", "cat"],
             "mscoco_gt_words": ["bird", "dog"], "mscoco_hallucinated_words": [["cat", "cat"]]},
            {"image_id": 2, "caption": "bird bird bird", "mscoco_generated_words": ["bird"] * 3,
             "mscoco_gt_words": ["bird"], "mscoco_hallucinated_words": []},
        ]}
        rows = attach_chair_scores(predictions, output)
        scores = evaluate_predictions(rows)
        self.assertEqual(scores["CHAIR_s"], 0.5)
        self.assertEqual(scores["CHAIR_i"], 0.2)  # micro count, not mean sentence ratio
        self.assertAlmostEqual(scores["object_recall"], 2 / 3)
        self.assertEqual(scores["mean_correct_distinct_objects"], 1)
        output["sentences"][0]["caption"] = "different text"
        with self.assertRaisesRegex(ValueError, "differs"):
            attach_chair_scores(predictions, output)

    def test_caption_unscored_is_explicit_not_zero_hallucination(self):
        result = evaluate_predictions([self.record(task="caption")])
        self.assertEqual(result["chair_status"], "not_scored")
        self.assertNotIn("CHAIR_i", result)

    def test_chair_adapter_calls_external_port_api_and_records_hash(self):
        annotation_dir = self.root / "annotations"
        annotation_dir.mkdir()
        for kind in ("instances", "captions"):
            for split in ("train", "val"):
                (annotation_dir / f"{kind}_{split}2014.json").write_text("{}")
        module = self.root / "fixture_scorer.py"
        module.write_text(
            "import json\n"
            "class CHAIR:\n"
            "    def __init__(self, path): self.imid_to_objects = {1171: {'bird'}}\n"
            "    def compute_chair(self, path, image_id_key, caption_key):\n"
            "        rows = json.load(open(path))\n"
            "        return {'sentences': [{'image_id': r[image_id_key], 'caption': r[caption_key],\n"
            "            'mscoco_generated_words': ['bird'], 'mscoco_gt_words': ['bird'],\n"
            "            'mscoco_hallucinated_words': []} for r in rows]}\n"
        )
        rows = run_official_chair(
            [self.record(task="caption", prediction="A bird.")], module, annotation_dir
        )
        self.assertEqual(len(rows[0]["chair"]["source"]["sha256"]), 64)
        self.assertEqual(evaluate_predictions(rows)["CHAIR_i"], 0)

    def test_mme_requires_complete_question_pairs(self):
        one = self.record(task="mme", metadata={"category": "existence"})
        with self.assertRaisesRegex(ValueError, "exactly two"):
            evaluate_predictions([one])
        second = {**one, "id": "q2", "label": "no", "prediction": "No."}
        self.assertEqual(evaluate_predictions([one, second])["mme_score_sum"], 200)

    def test_actual_pinned_chair_port_on_tiny_coco_annotations(self):
        """Optional real upstream integration, not a replacement scorer fixture."""
        scorer = Path(__file__).resolve().parents[1] / "upstream" / "chair" / "chair.py"
        if not scorer.is_file() or importlib.util.find_spec("pattern") is None:
            self.skipTest("Fetch pinned CHAIR and install eval dependencies to run integration")
        import nltk
        for resource in (
            "corpora/wordnet", "corpora/wordnet_ic", "corpora/sentiwordnet",
            "tokenizers/punkt_tab/english",
        ):
            try:
                nltk.data.find(resource)
            except LookupError:
                self.skipTest(f"Install NLTK resource {resource} to run upstream integration")
        annotations = self.root / "actual_chair_annotations"
        annotations.mkdir()
        categories = [{"id": 1, "name": "bird"}, {"id": 2, "name": "cat"}]
        for split, image_id, category_id, caption in (
            ("train", 1, 1, "A bird."), ("val", 2, 2, "Two cats."),
        ):
            common = {
                "info": {}, "licenses": [], "images": [{"id": image_id}],
                "categories": categories,
            }
            (annotations / f"instances_{split}2014.json").write_text(json.dumps({
                **common,
                "annotations": [{"image_id": image_id, "category_id": category_id}],
            }), encoding="utf-8")
            (annotations / f"captions_{split}2014.json").write_text(json.dumps({
                **common, "annotations": [{"image_id": image_id, "caption": caption}],
            }), encoding="utf-8")
        predictions = [
            self.record("a", task="caption", group_id="coco:1", prediction="A bird and a dog."),
            self.record("b", task="caption", group_id="coco:2", prediction="Two cats."),
        ]
        rows = run_official_chair(predictions, scorer, annotations)
        metrics = evaluate_predictions(rows)
        self.assertEqual(metrics["CHAIR_s"], 0.5)
        self.assertAlmostEqual(metrics["CHAIR_i"], 1 / 3)
        self.assertEqual(metrics["object_recall"], 1)
        self.assertEqual(metrics["mean_correct_distinct_objects"], 1)

    def test_actual_pinned_causal_scorer_keeps_invalid_answers_in_denominator(self):
        scorer = Path(__file__).resolve().parents[1] / "upstream" / "causal" / "metric.py"
        if not scorer.is_file():
            self.skipTest("Fetch pinned Causal-HalBench scorer to run integration")
        specifications = [
            ("target", "origin", "yes", "Yes."),
            ("absent", "origin", "no", "No."),
            ("target", "inpainted", "yes", "No."),
            ("absent", "inpainted", "no", "Yes."),
            ("inpaint", "inpainted", "yes", "I cannot tell."),
        ]
        records = []
        for index, (kind, tag, label, prediction) in enumerate(specifications):
            name = self.image.name if tag == "origin" else self.image.stem + "_001.jpg"
            records.append(self.record(
                f"causal:{index}", task="causal", label=label, prediction=prediction,
                metadata={"causal_qa": {
                    "id": index, "image_name": name, "type": kind, "tag": tag,
                    "answer": label, "question": "Is it present?",
                }},
            ))
        metrics = run_official_causal(records, scorer)
        self.assertEqual(metrics["cac"], 1)
        self.assertEqual(metrics["nac"], -1)
        self.assertEqual(metrics["chr"], 1)
        self.assertEqual(metrics["coverage"], 5)
        self.assertEqual(metrics["cpr_target"], 0)
        self.assertEqual(metrics["cpr_absent"], 0)

    def test_eval_setup_rejects_zip_traversal_before_extracting_any_member(self):
        helper = Path(__file__).resolve().parents[1] / "scripts" / "setup_eval.py"
        spec = importlib.util.spec_from_file_location("setup_eval_test", helper)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        archive = self.root / "wordnet.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("wordnet/good.txt", "valid")
            handle.writestr("wordnet/../../outside.txt", "invalid")
        destination = self.root / "corpora"
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            module.extract_wordnet(archive, destination)
        self.assertFalse((destination / "wordnet" / "good.txt").exists())
        self.assertFalse((self.root / "outside.txt").exists())
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("wordnet/good.txt", "valid")
        module.extract_wordnet(archive, destination)
        self.assertEqual((destination / "wordnet" / "good.txt").read_text(), "valid")


if __name__ == "__main__":
    unittest.main()
