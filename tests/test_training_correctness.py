import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),
                     "torch and transformers are not installed")
class TrainingCorrectnessTest(unittest.TestCase):
    def test_llama_applies_internal_label_shift_and_updates(self):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
        model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                                             num_hidden_layers=1, num_attention_heads=2,
                                             num_key_value_heads=2, max_position_embeddings=16))
        tokens = torch.randint(0, 32, (1, 8))
        before = model.model.embed_tokens.weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss = model(input_ids=tokens, labels=tokens).loss
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        optimizer.step()
        self.assertFalse(torch.equal(before, model.model.embed_tokens.weight))


if __name__ == "__main__":
    unittest.main()
