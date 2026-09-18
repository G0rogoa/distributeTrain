import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from train_perf.result_schema import validate_manifest
from train_perf.summarize import summarize


class SchemaTest(unittest.TestCase):
    def test_success(self):
        validate_manifest({"schema_version": 1, "run_id": "x", "created_at_utc": "now", "experiment_type": "environment",
                           "status": "success", "config": {}, "raw_logs": []})

    def test_failure_needs_reason(self):
        value = {"schema_version": 1, "run_id": "x", "created_at_utc": "now", "experiment_type": "train",
                 "status": "failed", "config": {}, "raw_logs": []}
        with self.assertRaisesRegex(ValueError, "failure_reason"):
            validate_manifest(value)

    def test_failed_runs_are_not_summarized(self):
        with TemporaryDirectory() as directory:
            run = Path(directory) / "failed"
            run.mkdir()
            (run / "manifest.json").write_text(
                '{"status":"failed","experiment_type":"single_gpu_train","config":{}}', encoding="utf-8")
            rows = summarize(directory, str(Path(directory) / "summary.csv"))
            self.assertEqual(rows, [])

    def test_training_model_mapping_is_groupable(self):
        with TemporaryDirectory() as directory:
            run = Path(directory) / "success"
            run.mkdir()
            manifest = {
                "status": "success", "experiment_type": "single_gpu_train",
                "config": {"model_config": {"hidden_size": 16}, "sequence_length": 8,
                           "micro_batch_size": 1, "precision": "bf16_autocast", "gpu_uuid": "GPU-x"},
            }
            (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (run / "metrics.jsonl").write_text('{"input_tokens_per_s": 10}\n', encoding="utf-8")
            rows = summarize(directory, str(Path(directory) / "summary.csv"))
            self.assertEqual(rows[0]["median"], 10.0)

    def test_nccl_preflight_is_not_summarized(self):
        with TemporaryDirectory() as directory:
            run = Path(directory) / "preflight"
            run.mkdir()
            manifest = {"status": "success", "experiment_type": "nccl_all_reduce",
                        "config": {"preflight": True}}
            (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (run / "metrics.jsonl").write_text('{"busbw_gbps": 1}\n', encoding="utf-8")
            rows = summarize(directory, str(Path(directory) / "summary.csv"))
            self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
