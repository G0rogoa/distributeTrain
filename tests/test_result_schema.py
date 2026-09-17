import unittest
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


if __name__ == "__main__":
    unittest.main()
