import math
from dataclasses import dataclass
from functools import lru_cache

import torch


@dataclass(frozen=True)
class BlockSparseConfig:
    block_size: int = 32
    local_blocks: int = 1
    random_blocks: int = 1
    global_tokens: int = 4
    seed: int = 17


def _block_pattern(num_blocks: int, cfg: BlockSparseConfig) -> list[set[int]]:
    """Choose key blocks for each query block before applying causality."""
    if num_blocks <= 0:
        return []

    generator = torch.Generator(device='cpu')
    generator.manual_seed(cfg.seed)
    result: list[set[int]] = []

    for qb in range(num_blocks):
        allowed = set()

        # Local neighborhood.
        lo = max(0, qb - cfg.local_blocks)
        hi = qb + cfg.local_blocks
        allowed.update(range(lo, min(num_blocks, hi + 1)))

        # Global block is deliberately deterministic. This gives every block
        # a cheap long-range route without needing a global dense matrix.
        allowed.add(0)

        # Deterministic random blocks. We exclude blocks already in the local
        # set so the random connections are actually useful.
        candidates = [x for x in range(num_blocks) if x not in allowed]
        if candidates and cfg.random_blocks:
            perm = torch.randperm(len(candidates), generator=generator).tolist()
            for idx in perm[: min(cfg.random_blocks, len(candidates))]:
                allowed.add(candidates[idx])

        result.append(allowed)

    return result


def block_sparse_mask(
    seq_len: int,
    cfg: BlockSparseConfig,
    device=None,
) -> torch.Tensor:
    """Return the exact [N, N] boolean mask implied by the block pattern.

    This dense mask is mainly for visualization and the correctness oracle.
    The actual attention kernel below gathers only the selected positions.
    """
    if seq_len <= 0:
        raise ValueError('seq_len must be positive')
    if cfg.block_size <= 0 or cfg.local_blocks < 0 or cfg.random_blocks < 0:
        raise ValueError('invalid block-sparse configuration')

    nblocks = math.ceil(seq_len / cfg.block_size)
    pattern = _block_pattern(nblocks, cfg)
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

    global_positions = set(range(min(cfg.global_tokens, seq_len)))

    for qb, key_blocks in enumerate(pattern):
        q0 = qb * cfg.block_size
        q1 = min(seq_len, q0 + cfg.block_size)
        for kb in key_blocks:
            k0 = kb * cfg.block_size
            k1 = min(seq_len, k0 + cfg.block_size)
            if k0 >= seq_len:
                continue
            mask[q0:q1, k0:k1] = True

    if global_positions:
        g = torch.tensor(sorted(global_positions), dtype=torch.long, device=device)
        qpos = torch.arange(seq_len, device=device)[:, None]
        mask[:, g] = g[None, :] <= qpos

    # GPT-style causal masking is always enforced.
    mask &= torch.tril(torch.ones_like(mask))
    return mask


@lru_cache(maxsize=32)
def _allowed_indices_cpu(seq_len: int, cfg: BlockSparseConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a padded index table [N, Kmax] plus a validity mask."""
    nblocks = math.ceil(seq_len / cfg.block_size)
    pattern = _block_pattern(nblocks, cfg)
    global_positions = set(range(min(cfg.global_tokens, seq_len)))

    rows: list[list[int]] = []
    max_len = 0
    for i in range(seq_len):
        qb = i // cfg.block_size
        allowed = set(global_positions)
        for kb in pattern[qb]:
            k0 = kb * cfg.block_size
            k1 = min(seq_len, k0 + cfg.block_size)
            allowed.update(range(k0, min(k1, i + 1)))
        allowed = sorted(x for x in allowed if x <= i)
        rows.append(allowed)
        max_len = max(max_len, len(allowed))

    indices = torch.zeros(seq_len, max_len, dtype=torch.long)
    valid = torch.zeros(seq_len, max_len, dtype=torch.bool)
    for i, row in enumerate(rows):
        if row:
            r = torch.tensor(row, dtype=torch.long)
            indices[i, : len(row)] = r
            valid[i, : len(row)] = True

    return indices, valid


def _allowed_indices(seq_len: int, cfg: BlockSparseConfig, device):
    indices, valid = _allowed_indices_cpu(seq_len, cfg)
    return indices.to(device), valid.to(device)


def block_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cfg: BlockSparseConfig,
) -> torch.Tensor:
    """Local + global + random causal block-sparse attention."""
    if q.ndim != 4 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError('q, k and v must have the same [B, H, N, D] shape')

    b, h, n, d = q.shape
    indices, valid = _allowed_indices(n, cfg, q.device)
    width = indices.size(1)

    k_local = k.index_select(2, indices.reshape(-1)).reshape(b, h, n, width, d)
    v_local = v.index_select(2, indices.reshape(-1)).reshape(b, h, n, width, d)

    scores = (q.unsqueeze(-2) * k_local).sum(dim=-1) / math.sqrt(d)
    scores = scores.masked_fill(~valid[None, None, :, :], float('-inf'))
    probs = torch.softmax(scores, dim=-1)
    probs = torch.where(valid[None, None, :, :], probs, torch.zeros_like(probs))
    return (probs.unsqueeze(-1) * v_local).sum(dim=-2)
