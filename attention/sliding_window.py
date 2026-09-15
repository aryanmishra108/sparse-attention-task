import torch


def sliding_window_mask(seq_len: int, window_size: int, device=None) -> torch.Tensor:
    """Causal mask where token i sees only the last window_size tokens."""
    if seq_len <= 0:
        raise ValueError('seq_len must be positive')
    if window_size <= 0:
        raise ValueError('window_size must be positive')

    pos = torch.arange(seq_len, device=device)
    rows = pos[:, None]
    cols = pos[None, :]
    return (cols <= rows) & (cols >= rows - window_size + 1)


def sliding_window_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    """Actually sparse sliding-window attention.

    The implementation gathers only the keys/values in each query's local
    window, so its score tensor is O(N * window_size), rather than O(N^2).
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError('q, k and v must be [B, H, N, D] tensors')
    b, h, n, d = q.shape
    if k.shape != (b, h, n, d) or v.shape != (b, h, n, d):
        raise ValueError('q, k and v must have the same shape')
    if window_size <= 0:
        raise ValueError('window_size must be positive')

    width = min(window_size, n)
    # For query i, candidate key index = i - (width - 1) ... i.
    # Negative positions are clamped and then rejected by valid.
    offsets = torch.arange(width - 1, -1, -1, device=q.device)
    query_pos = torch.arange(n, device=q.device)[:, None]
    indices = query_pos - offsets[None, :]
    valid = indices >= 0
    safe_indices = indices.clamp_min(0).long()

    k_local = k.index_select(2, safe_indices.reshape(-1)).reshape(b, h, n, width, d)
    v_local = v.index_select(2, safe_indices.reshape(-1)).reshape(b, h, n, width, d)

    scores = (q.unsqueeze(-2) * k_local).sum(dim=-1) / (d ** 0.5)
    scores = scores.masked_fill(~valid[None, None, :, :], float('-inf'))
    probs = torch.softmax(scores, dim=-1)
    probs = torch.where(valid[None, None, :, :], probs, torch.zeros_like(probs))

    return (probs.unsqueeze(-1) * v_local).sum(dim=-2)
