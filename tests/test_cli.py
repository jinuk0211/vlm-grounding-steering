import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from vlm_grounding.cli import (
    _model_signature, _prediction_rows, _select, _validate_artifact, build_parser, main,
)


class MockAdapter:
    def __init__(self):
        self.model = SimpleNamespace(config=SimpleNamespace(_commit_hash="test-commit"))
        self.last_generated_tokens = 1

    def prepare(self, image, prompt):
        return {"image": image, "prompt": prompt}

    def generate(self, batch, max_new_tokens):
        return "Yes."


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.image = self.root / "image.png"
        self.image.write_bytes(b"test-only")
        self.config = {"model": {"checkpoint": "tiny-test", "device": "cpu",
                                 "family": "llava", "dtype": "float32"},
                       "generation": {"max_new_tokens": 4}}
        self.config_path = self.root / "config.json"
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        self.record = {"id": "question:1", "group_id": "image:1", "image": str(self.image),
                       "prompt": "Is a cat present?", "label": "yes", "task": "pope",
                       "split": "test"}
        self.manifest = self.root / "test.jsonl"
        self.manifest.write_text(json.dumps(self.record) + "\n", encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def command(self, extra):
        return ["run", "--config", str(self.config_path), "--manifest", str(self.manifest),
                "--output", str(self.root / "predictions.jsonl"), *extra]

    def test_parser_exposes_required_pipeline(self):
        parser = build_parser()
        args = parser.parse_args(["compare", "--predictions", "a.jsonl", "b.jsonl",
                                  "--output-dir", "report"])
        self.assertEqual(args.predictions, ["a.jsonl", "b.jsonl"])
        self.assertEqual(args.baseline, "base")

    def test_base_run_without_gpu_and_complete_provenance(self):
        with patch("vlm_grounding.cli._load_adapter", return_value=MockAdapter()) as loader, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(self.command(["--method", "base"])), 0)
        loader.assert_called_once_with(self.config["model"])
        path = self.root / "predictions.jsonl"
        rows = _prediction_rows(str(path))
        self.assertEqual(rows[0]["prediction"], "Yes.")
        self.assertEqual(rows[0]["generated_tokens"], 1)
        metadata = json.loads(Path(str(path) + ".metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["status"], "complete")
        self.assertEqual(metadata["checkpoint_revision"], "test-commit")
        self.assertEqual(metadata["record_ids"], ["question:1"])
        self.assertIsNone(metadata["artifact_sha256"])
        self.assertEqual(rows[0]["protocol_fingerprint"], metadata["protocol_fingerprint"])
        self.assertFalse(metadata["evaluation_protocol"]["generation"]["do_sample"])

    def test_prevents_overwrite_before_model_load(self):
        (self.root / "predictions.jsonl").write_text("existing", encoding="utf-8")
        with patch("vlm_grounding.cli._load_adapter") as loader, \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(self.command(["--method", "base"])), 2)
        loader.assert_not_called()

    def test_refuses_partial_test_run_before_model_load(self):
        with patch("vlm_grounding.cli._load_adapter") as loader, \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(self.command(["--method", "base", "--limit", "1"])), 2)
        loader.assert_not_called()

    def test_rejects_artifact_training_group_leakage_and_model_mismatch(self):
        artifact = {"_provenance": {"model_config": self.config["model"],
                                     "training_group_ids": ["image:1"]}}
        with self.assertRaisesRegex(ValueError, "overlap"):
            _validate_artifact(artifact, self.config["model"], [self.record])
        with self.assertRaisesRegex(ValueError, "model config"):
            _validate_artifact(artifact, {**self.config["model"], "checkpoint": "different"})
        self.assertEqual(_model_signature(self.config["model"]),
                         _model_signature({**self.config["model"], "device": "cuda"}))

    def test_rejects_resolved_revision_drift(self):
        artifact = {"_provenance": {"model_config": self.config["model"],
                                     "checkpoint_revision": "different-commit"}}
        with self.assertRaisesRegex(ValueError, "revision"):
            _validate_artifact(artifact, self.config["model"], adapter=MockAdapter())

    def test_failed_generation_is_marked_and_not_comparable(self):
        adapter = MockAdapter()
        adapter.generate = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("failed"))
        with patch("vlm_grounding.cli._load_adapter", return_value=adapter), \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(self.command(["--method", "base"])), 2)
        with self.assertRaisesRegex(ValueError, "did not complete"):
            _prediction_rows(str(self.root / "predictions.jsonl"))

    def test_select_filters_fit_preferences_without_test_leakage(self):
        records = [{**self.record, "id": "fit", "split": "fit", "chosen": "yes", "rejected": "no"},
                   self.record]
        self.assertEqual([row["id"] for row in _select(records, "fit", None, preferences=True)],
                         ["fit"])

    def test_compare_two_complete_prediction_files(self):
        paths = []
        for method in ("base", "ephs"):
            path = self.root / f"{method}.jsonl"
            path.write_text(json.dumps({**self.record, "prediction": "Yes.", "method": method}) + "\n",
                            encoding="utf-8")
            paths.append(str(path))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["compare", "--predictions", *paths,
                                   "--output-dir", str(self.root / "report"),
                                   "--bootstrap-samples", "10"]), 0)
        report = json.loads((self.root / "report" / "comparison.json").read_text(encoding="utf-8"))
        self.assertEqual(set(report["methods"]), {"base", "ephs"})
        self.assertFalse(report["all_protocols_verified"])
        self.assertTrue((self.root / "report" / "metrics.csv").is_file())

    def test_compare_rejects_different_generation_budgets(self):
        paths = []
        for method, tokens in (("base", 64), ("ephs", 128)):
            path = self.root / f"{method}.jsonl"
            path.write_text(json.dumps({**self.record, "prediction": "Yes.", "method": method}) + "\n",
                            encoding="utf-8")
            metadata = {"status": "complete", "record_ids": [self.record["id"]],
                        "config": {**self.config, "generation": {"max_new_tokens": tokens}}}
            Path(str(path) + ".metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            paths.append(str(path))
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["compare", "--predictions", *paths,
                                   "--output-dir", str(self.root / "report")]), 2)
        self.assertFalse((self.root / "report" / "comparison.json").exists())

    def test_mmhal_revis_rudder_export_and_scored_compare(self):
        template = self.root / "response_template.json"
        template.write_text(json.dumps([{
            "question_type": ("attribute", "adversarial")[i % 2],
            "question_topic": "outdoor", "image_id": "one-image",
            "image_src": f"https://example.org/{self.image.name}",
            "image_content": ["Cat"], "question": f"Question {i}?",
            "gt_answer": "A cat.", "model_answer": "",
        } for i in range(96)]), encoding="utf-8")
        mmhal_manifest = self.root / "mmhal.jsonl"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["prepare-mmhal", "--input", str(template),
                                   "--image-root", str(self.root),
                                   "--output", str(mmhal_manifest)]), 0)
        rows = [json.loads(line) for line in mmhal_manifest.read_text(encoding="utf-8").splitlines()]
        smoke = self.root / "mmhal_smoke.jsonl"
        with patch("vlm_grounding.cli._load_adapter", return_value=MockAdapter()), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["run", "--method", "base", "--task", "mmhal",
                                   "--config", str(self.config_path),
                                   "--manifest", str(mmhal_manifest),
                                   "--output", str(smoke)]), 0)
        self.assertEqual(len(_prediction_rows(str(smoke))), 96)
        files, judge_args = [], []
        for method, rating in (("base", 3), ("revis", 5), ("rudder", 2)):
            predictions = self.root / f"{method}.jsonl"
            predictions.write_text("".join(json.dumps({**row, "prediction": method,
                                                        "method": method}) + "\n"
                                           for row in rows), encoding="utf-8")
            responses = self.root / f"{method}_responses.json"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["export-mmhal", "--predictions", str(predictions),
                                       "--template", str(template),
                                       "--output", str(responses)]), 0)
            self.assertEqual(json.loads(responses.read_text(encoding="utf-8"))[0]["model_answer"],
                             method)
            judged = self.root / f"{method}_judge.json"
            judged.write_text(json.dumps([{"model": "fixture-gpt-4", "choices": [{"message": {
                "content": f"Explanation. Rating: {rating}"}}]} for _ in range(96)]),
                encoding="utf-8")
            files.append(str(predictions))
            judge_args.extend(["--mmhal-evaluation", method, str(responses), str(judged)])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["compare", "--predictions", *files,
                                   "--output-dir", str(self.root / "mmhal_report"),
                                   "--bootstrap-samples", "3", *judge_args]), 0)
        report = json.loads((self.root / "mmhal_report" / "comparison.json")
                            .read_text(encoding="utf-8"))
        self.assertEqual(report["methods"]["revis"]["mmhal"]["mmhal_average_score"], 5)
        self.assertEqual(report["methods"]["rudder"]["mmhal"]["mmhal_hallucination_rate"], 1)
        self.assertTrue(report["mmhal_judge_model_verified"])
        deltas = {(row["method"], row["metric"]): row["delta"]
                  for row in report["paired_deltas"]}
        self.assertEqual(deltas[("revis", "mmhal_average_score")], 2)
        self.assertEqual(deltas[("rudder", "mmhal_average_score")], -1)
        rudder_judge = self.root / "rudder_judge.json"
        changed = json.loads(rudder_judge.read_text(encoding="utf-8"))
        for entry in changed:
            entry["model"] = "different-judge"
        rudder_judge.write_text(json.dumps(changed), encoding="utf-8")
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["compare", "--predictions", *files,
                                   "--output-dir", str(self.root / "mismatched_judge"),
                                   "--bootstrap-samples", "0", *judge_args]), 2)
        self.assertFalse((self.root / "mismatched_judge" / "comparison.json").exists())


if __name__ == "__main__":
    unittest.main()
