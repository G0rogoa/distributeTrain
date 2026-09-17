import json
import os
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from train_perf.nccl import execute


class NcclRunnerTest(unittest.TestCase):
    def test_runner_writes_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "all_reduce_perf"
            binary.write_text(textwrap.dedent("""\
                #!/usr/bin/env python3
                import sys
                if "--help" in sys.argv:
                    print("--iters --warmup_iters --check --datatype")
                else:
                    print("1024 256 float sum -1 8.50 0.12 0.18 0 7.90 0.13 0.19 0")
                """), encoding="utf-8")
            binary.chmod(0o755)
            config = {
                "binary": str(binary), "gpu_sets": [[2, 3]], "types": ["float"],
                "min_bytes": "1K", "max_bytes": "1M", "preflight_max_bytes": "1M",
                "step_factor": 2, "warmup_iterations": 2, "iterations": 3,
                "repeats": 1, "timeout_seconds": 10,
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            results = root / "results"
            with patch("train_perf.nccl.query_gpu_uuids", return_value=["GPU-two", "GPU-three"]):
                self.assertTrue(execute(str(config_path), str(results), preflight_only=True))
            manifests = list(results.glob("*/manifest.json"))
            self.assertEqual(len(manifests), 1)
            manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "success")
            self.assertEqual(manifest["config"]["gpu_uuids"], ["GPU-two", "GPU-three"])
            metrics = (manifests[0].parent / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(metrics), 2)


if __name__ == "__main__":
    unittest.main()
