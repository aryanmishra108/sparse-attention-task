import math
import torch


def safe_softmax(scores: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Softmax that returns zeros instead of NaN for a fully-masked row."""
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
        valid = mask.any(dim=-1, keepdim=True)
    else:
        valid = torch.ones_like(scores[..., :1], dtype=torch.bool)

    probs = torch.softmax(scores, dim=-1)
    return torch.where(valid, probs, torch.zeros_like(probs))


def dense_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Manual scaled dot-product attention.

    q: [B, H, Nq, D]
    k: [B, H, Nk, D]
    v: [B, H, Nk, D]
    mask: broadcastable to [B, H, Nq, Nk], True means allowed.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError('q, k and v must be [B, H, N, D] tensors')
    if q.shape[:2] != k.shape[:2] or k.shape != v.shape:
        raise ValueError('incompatible q/k/v shapes')
    if q.size(-1) != k.size(-1):
        raise ValueError('q and k need the same head dimension')

    d = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
    probs = safe_softmax(scores, mask)
    return torch.matmul(probs, v)
