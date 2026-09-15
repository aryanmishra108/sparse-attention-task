import math

import torch

from attention.block_sparse import BlockSparseConfig, block_sparse_attention, block_sparse_mask
from attention.dense import dense_attention, safe_softmax
from attention.sliding_window import sliding_window_attention
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


def test_sliding_window_mask():
    from attention.sliding_window import sliding_window_mask
    expected = torch.tensor([
        [1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1],
    ], dtype=torch.bool)
    assert torch.equal(sliding_window_mask(6, 3).cpu(), expected)


def test_sliding_window_matches_masked_dense():
    from attention.sliding_window import sliding_window_attention, sliding_window_mask
    torch.manual_seed(1)
    q = torch.randn(2, 2, 9, 6)
    k = torch.randn(2, 2, 9, 6)
    v = torch.randn(2, 2, 9, 6)
    mask = sliding_window_mask(9, 4)
    expected = dense_attention(q, k, v, mask)
    actual = sliding_window_attention(q, k, v, 4)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_sliding_window_is_finite():
    torch.manual_seed(2)
    q = torch.randn(1, 1, 8, 4)
    k = torch.randn(1, 1, 8, 4)
    v = torch.randn(1, 1, 8, 4)
    out = sliding_window_attention(q, k, v, 3)
    assert torch.isfinite(out).all()


def test_block_sparse_mask_is_causal_and_nonempty():
    from attention.block_sparse import BlockSparseConfig, block_sparse_mask
    cfg = BlockSparseConfig(block_size=3, local_blocks=1, random_blocks=1, global_tokens=2, seed=7)
    mask = block_sparse_mask(11, cfg)
    assert mask.shape == (11, 11)
    assert torch.equal(mask, mask.tril())
    assert mask.any(dim=-1).all()


def test_block_sparse_matches_masked_dense():
    from attention.block_sparse import BlockSparseConfig, block_sparse_attention, block_sparse_mask
    torch.manual_seed(3)
    q = torch.randn(2, 2, 17, 5)
    k = torch.randn(2, 2, 17, 5)
    v = torch.randn(2, 2, 17, 5)
    cfg = BlockSparseConfig(block_size=4, local_blocks=1, random_blocks=1, global_tokens=2, seed=11)
    mask = block_sparse_mask(17, cfg)
    expected = dense_attention(q, k, v, mask)
    actual = block_sparse_attention(q, k, v, cfg)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_block_sparse_random_pattern_is_reproducible():
    from attention.block_sparse import BlockSparseConfig, block_sparse_mask
    cfg = BlockSparseConfig(block_size=4, local_blocks=1, random_blocks=2, global_tokens=2, seed=123)
    assert torch.equal(block_sparse_mask(32, cfg), block_sparse_mask(32, cfg))


def test_sparse_outputs_are_finite():
    from attention.block_sparse import BlockSparseConfig, block_sparse_attention
    torch.manual_seed(4)
    q = torch.randn(1, 1, 23, 8)
    k = torch.randn(1, 1, 23, 8)
    v = torch.randn(1, 1, 23, 8)
    sw = sliding_window_attention(q, k, v, 5)
    bs = block_sparse_attention(q, k, v, BlockSparseConfig(block_size=4))
    assert torch.isfinite(sw).all()
    assert torch.isfinite(bs).all()
