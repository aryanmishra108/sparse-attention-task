import torch


def causal_mask(seq_len: int, device=None) -> torch.Tensor:
    if seq_len <= 0:
        raise ValueError('seq_len must be positive')
    positions = torch.arange(seq_len, device=device)
    return positions.unsqueeze(1) >= positions.unsqueeze(0)
