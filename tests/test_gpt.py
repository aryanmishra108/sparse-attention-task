import torch

from attention.block_sparse import BlockSparseConfig
from gpt.model import GPTConfig, TinyGPT


def test_gpt_forward_all_attention_modes():
    torch.manual_seed(10)
    for pattern in ['dense', 'sliding_window', 'block_sparse']:
        cfg = GPTConfig(
            vocab_size=31,
            block_size=24,
            n_layer=2,
            n_head=4,
            n_embd=64,
            attention_pattern=pattern,
            window_size=8,
            block_sparse=BlockSparseConfig(
                block_size=8,
                local_blocks=1,
                random_blocks=1,
                global_tokens=4,
                seed=3,
            ),
        )
        model = TinyGPT(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
        logits, loss = model(x, x)
        assert logits.shape == (2, cfg.block_size, cfg.vocab_size)
        assert loss.ndim == 0
        assert torch.isfinite(logits).all()
        assert torch.isfinite(loss)
