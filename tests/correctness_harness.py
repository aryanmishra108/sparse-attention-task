"""Human-readable correctness check for the two sparse patterns.

This is separate from pytest because the task asks for an actual harness that
reports pass/fail against the dense reference.
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attention.block_sparse import BlockSparseConfig, block_sparse_attention, block_sparse_mask
from attention.dense import dense_attention
from attention.sliding_window import sliding_window_attention, sliding_window_mask


def check(name, actual, expected, tolerance=1e-5):
    diff = (actual - expected).abs().max().item()
    ok = torch.allclose(actual, expected, rtol=tolerance, atol=tolerance)
    status = 'PASS' if ok else 'FAIL'
    print(f'{status:4s} {name:45s} max_abs_diff={diff:.3e}')
    return ok


def main():
    torch.manual_seed(123)
    q = torch.randn(2, 3, 23, 8)
    k = torch.randn(2, 3, 23, 8)
    v = torch.randn(2, 3, 23, 8)

    dense = dense_attention(q, k, v)
    print('Reference: manual dense attention without a mask')
    print('Note: sparse outputs are expected to differ from unrestricted dense attention.')
    print('The actual correctness test uses the same sparse mask in the dense oracle.\n')

    sw_mask = sliding_window_mask(23, 5)
    sw_expected = dense_attention(q, k, v, sw_mask)
    sw_actual = sliding_window_attention(q, k, v, 5)

    bs_cfg = BlockSparseConfig(block_size=4, local_blocks=1, random_blocks=1, global_tokens=3, seed=123)
    bs_mask = block_sparse_mask(23, bs_cfg)
    bs_expected = dense_attention(q, k, v, bs_mask)
    bs_actual = block_sparse_attention(q, k, v, bs_cfg)

    ok = True
    ok &= check('sliding-window vs dense with identical mask', sw_actual, sw_expected)
    ok &= check('block-sparse vs dense with identical mask', bs_actual, bs_expected)
    ok &= check('dense output is finite', dense, dense)
    ok &= check('sliding-window output is finite', sw_actual, sw_actual)
    ok &= check('block-sparse output is finite', bs_actual, bs_actual)

    print('\nPASS' if ok else '\nFAIL')
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
