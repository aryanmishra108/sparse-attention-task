import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention.block_sparse import BlockSparseConfig, block_sparse_attention
from attention.dense import dense_attention
from attention.sliding_window import sliding_window_attention
from attention.masks import causal_mask


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    attention_pattern: str = 'dense'
    window_size: int = 32
    block_sparse: BlockSparseConfig = BlockSparseConfig()


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig, layer_idx: int):
        super().__init__()
        if cfg.n_embd % cfg.n_head != 0:
            raise ValueError('n_embd must be divisible by n_head')

        self.cfg = cfg
        self.layer_idx = layer_idx
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        b, t, c = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.cfg.n_head, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.cfg.attention_pattern == 'dense':
            mask = causal_mask(t, x.device)
            y = dense_attention(q, k, v, mask)
        elif self.cfg.attention_pattern == 'sliding_window':
            y = sliding_window_attention(q, k, v, self.cfg.window_size)
        elif self.cfg.attention_pattern == 'block_sparse':
            base = self.cfg.block_sparse
            cfg = BlockSparseConfig(
                block_size=base.block_size,
                local_blocks=base.local_blocks,
                random_blocks=base.random_blocks,
                global_tokens=base.global_tokens,
                seed=base.seed + self.layer_idx,
            )
            y = block_sparse_attention(q, k, v, cfg)
        else:
            raise ValueError(f'unknown attention pattern: {self.cfg.attention_pattern}')

        y = y.transpose(1, 2).contiguous().view(b, t, c)
        y = self.resid_dropout(self.proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = self.fc(x)
        x = F.gelu(x)
        x = self.proj(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig, layer_idx: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg, layer_idx)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position_embedding = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)
        # GPT-style weight tying keeps the model compact and gives a useful
        # parameter-count check in the README.
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        if t > self.cfg.block_size:
            raise ValueError(f'sequence length {t} > block_size {self.cfg.block_size}')

        pos = torch.arange(t, device=idx.device)
        x = self.drop(self.token_embedding(idx) + self.position_embedding(pos)[None, :, :])
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx
