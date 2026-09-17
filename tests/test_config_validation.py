import unittest

from train_perf.nccl import validate_config as validate_nccl
from train_perf.train import validate_config as validate_train


class ConfigValidationTest(unittest.TestCase):
    def test_nccl_rejects_single_gpu(self):
        config = {"binary": "x", "gpu_sets": [[1]], "types": ["float"], "min_bytes": "1K",
                  "max_bytes": "1M", "step_factor": 2, "warmup_iterations": 1, "iterations": 1,
                  "repeats": 1, "timeout_seconds": 10}
        with self.assertRaisesRegex(ValueError, "two unique"):
            validate_nccl(config)

    def test_training_rejects_silent_feature_change(self):
        config = {"model_config": "x", "mode": "benchmark", "sequence_lengths": [8],
                  "micro_batch_sizes": [1], "warmup_steps": 1, "measurement_steps": 1,
                  "repeats": 1, "seed": 1, "learning_rate": 1e-3, "betas": [0.9, 0.95],
                  "eps": 1e-8, "weight_decay": 0.1, "gradient_accumulation": 1,
                  "precision": "bf16_autocast", "attention_backend": "sdpa",
                  "activation_checkpointing": True, "torch_compile": False, "allow_tf32": True}
        with self.assertRaisesRegex(ValueError, "must remain disabled"):
            validate_train(config)


if __name__ == "__main__":
    unittest.main()
