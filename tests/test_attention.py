import math

import torch

from attention.block_sparse import BlockSparseConfig, block_sparse_attention, block_sparse_mask
from attention.dense import dense_attention, safe_softmax
from attention.masks import causal_mask
from attention.sliding_window import sliding_window_attention, sliding_window_mask


def slow_reference(q, k, v, mask=None):
    bsz, heads, nq, d = q.shape
    nk = k.size(-2)
    out = torch.zeros_like(q)
    for b in range(bsz):
        for h in range(heads):
            for i in range(nq):
                logits = []
                values = []
                for j in range(nk):
                    if mask is None or bool(mask[i, j]):
                        logits.append(torch.dot(q[b, h, i], k[b, h, j]) / math.sqrt(d))
                        values.append(v[b, h, j])
                if logits:
                    logits = torch.stack(logits)
                    values = torch.stack(values)
                    out[b, h, i] = torch.softmax(logits, dim=0) @ values
    return out


def test_causal_mask():
    expected = torch.tensor([
        [True, False, False, False],
        [True, True, False, False],
        [True, True, True, False],
        [True, True, True, True],
    ])
    assert torch.equal(causal_mask(4).cpu(), expected)


def test_dense_matches_slow_reference():
    torch.manual_seed(0)
    q = torch.randn(2, 3, 5, 7)
    k = torch.randn(2, 3, 5, 7)
    v = torch.randn(2, 3, 5, 7)
    mask = causal_mask(5)
    actual = dense_attention(q, k, v, mask)
    expected = slow_reference(q, k, v, mask)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_fully_masked_row_does_not_nan():
    scores = torch.tensor([[[1.0, 2.0], [float('-inf'), float('-inf')]]])
    mask = torch.tensor([[[True, True], [False, False]]])
    probs = safe_softmax(scores, mask)
    assert torch.isfinite(probs).all()
    torch.testing.assert_close(probs[0, 1], torch.zeros(2), rtol=0, atol=0)
