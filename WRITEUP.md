# Task 1 — Sparse Attention from Scratch

## 1. What I built

The goal of this task was to implement attention from scratch, then reduce the number of query-key interactions without hiding the work inside a built-in attention kernel.

I first implemented a manual dense attention reference. Given `q`, `k`, and `v`, it computes `QK^T`, scales by `sqrt(d)`, applies a boolean mask, takes the softmax, and multiplies by `V`. I deliberately did not use PyTorch's `scaled_dot_product_attention` because the task asks for the attention calculation to be implemented by hand.

I then implemented two sparse patterns. The first is causal sliding-window attention. With window size `w`, query position `i` can attend only to positions `i-w+1 ... i`. The second is a block-sparse pattern combining local blocks, a small global token set, and deterministic random blocks. The random selection is seeded so that the sparsity pattern is reproducible, and causal masking is applied so a query never attends to a future token.

The block-sparse design is similar in spirit to the local/global/random structure used by BigBird: keep nearby interactions, add a small number of long-range routes, and avoid constructing the full `N x N` score matrix.

## 2. Correctness and NaN handling

A sparse implementation is not supposed to produce the same output as unrestricted dense attention. A sliding window intentionally removes some keys. Therefore, the correctness test uses the manual dense implementation with the **same sparse mask** as the oracle:

```text
same Q, K, V
      |
      +--> manual dense attention + sparse mask --> reference
      |
      +--> sparse implementation                --> candidate
```

The correctness harness passed for both sparse implementations. The maximum absolute differences were:

| Comparison | Maximum absolute error |
|---|---:|
| Sliding-window vs dense with identical mask | `3.576e-07` |
| Block-sparse vs dense with identical mask | `2.980e-07` |

Both are well below the `1e-5` tolerance used by the tests. The harness also confirmed that dense, sliding-window, and block-sparse outputs remain finite.

There is a separate edge case when a query has no legal keys. After masking, the score row becomes `[-inf, -inf, ...]`; a normal softmax evaluates to `0/0` and returns NaN. I handle this explicitly in `safe_softmax()`: a fully masked row gets an all-zero probability vector and therefore an all-zero output. This can arise naturally when causal masking and sparse structure meet at boundaries, so it is part of the normal implementation rather than only a synthetic test.

## 3. Benchmark setup and results

The benchmark measures forward-pass wall-clock time and peak process memory for sequence lengths 512, 1024, 2048, 4096, and 8192. The same attention dimensions and input tensors are used for the three implementations within a run. The benchmark was run on the local Windows CPU environment; the benchmark script records the hardware information in its output/results.

Measured results were:

### Runtime

| Sequence length | Dense (s) | Sliding-window (s) | Block-sparse (s) |
|---:|---:|---:|---:|
| 512 | 0.0057 | 0.0167 | 0.0270 |
| 1024 | 0.0073 | 0.0283 | 0.0565 |
| 2048 | 0.0384 | 0.0569 | 0.1536 |
| 4096 | 0.1720 | 0.1316 | 0.3477 |
| 8192 | 1.0093 | 0.3366 | 0.6827 |

### Peak memory

| Sequence length | Dense (MB) | Sliding-window (MB) | Block-sparse (MB) |
|---:|---:|---:|---:|
| 512 | 234.6 | 258.3 | 325.4 |
| 1024 | 267.5 | 328.4 | 430.8 |
| 2048 | 392.4 | 434.0 | 638.6 |
| 4096 | 904.3 | 643.6 | 1028.9 |
| 8192 | 2865.2 | 1080.0 | 1405.2 |

The crossover is the main result. At short sequence lengths, sparse implementations can be slower because indexing, gathering, and block bookkeeping add overhead. By `N=4096`, sliding-window attention is already faster than dense attention, and at `N=8192` it takes `0.3366 s` compared with `1.0093 s` for dense attention. That is about a 3.0x speedup in this configuration.

The memory trend is even clearer at long sequence lengths. At `N=8192`, dense attention uses about `2865 MB`, compared with `1080 MB` for sliding-window and `1405 MB` for block-sparse. These numbers include the Python/PyTorch process baseline, so I treat them as relative measurements rather than as a pure accounting of attention tensors.

The benchmark is intentionally forward-only, as requested. The important scaling idea is that a masked dense implementation still materializes an `N x N` score matrix, whereas the sparse implementations construct only the allowed interactions.

## 4. What each pattern loses

Sliding-window attention preserves local information well. For a character-level language model, that includes spelling, punctuation, local syntax, and short-range patterns. This helps explain why a fairly small window can still work reasonably well on TinyShakespeare.

The main loss is direct long-range access. A token near the end of a sequence cannot directly inspect a distant early token. Information has to travel through intermediate positions and layers. With only two Transformer layers, there are relatively few opportunities for that information to propagate.

The block-sparse pattern tries to recover some of this connectivity. Local blocks preserve the nearby information, global tokens provide a small number of paths across the entire sequence, and random blocks add extra long-range shortcuts. Global tokens matter disproportionately because they connect otherwise distant regions with only a small increase in the number of allowed interactions. They do not recreate every dense pair, however, so they are still a bottleneck rather than a perfect replacement.

The extra structure also has a cost. The block-sparse benchmark was consistently heavier than the simple sliding window because it retains more connections and has more indexing/bookkeeping work.

## 5. Results

### Attention benchmark

Benchmarked sequence lengths 512, 1024, 2048, 4096, and 8192.

At N=8192:

| Pattern | Time | Peak memory |
|---|---:|---:|
| Dense | 1.0093 s | 2865.2 MB |
| Sliding-window | 0.3366 s | 1080.0 MB |
| Block-sparse | 0.6827 s | 1405.2 MB |

### TinyShakespeare

Final validation loss at step 1999:

| Pattern | Validation loss |
|---|---:|
| Dense | 1.6475 |
| Sliding-window | 1.6044 |
| Block-sparse | 1.6585 |


The exact values came from the `quality_results.json` files produced by the three runs. There is a useful consistency check here: the same ordering appears in all three runs. Sliding-window has the lowest validation loss, block-sparse is very close to dense, and dense is between them in two of the three runs.

I do not interpret the lower sliding-window loss as proof that sparse attention is inherently better than dense attention. This is a small character-level model, with only two Transformer layers and a fixed training budget. A more restrictive attention pattern can sometimes act like a useful inductive bias or simply change optimization enough to move the measured loss. The result is best read as evidence that the sparse patterns did **not** cause a quality collapse in this experiment.

The training-time numbers show a different tradeoff from the standalone attention benchmark. In the GPT implementation, the sparse variants include additional indexing/gather overhead, so end-to-end training was not automatically faster.
## 6. Takeaways

Sparsity helps only when the implementation actually avoids the work for the missing connections. Applying a sparse mask to a dense `N x N` score matrix preserves the quadratic cost, even if most entries are `-inf`.

Sliding-window attention gives the largest savings and the simplest pattern, but it has the weakest direct access to long-range information. Block-sparse attention gives some of that connectivity back through global and random links, at the cost of more computation and more complicated indexing.

On the 8192-token CPU benchmark, sliding-window attention was about 3x faster and used substantially less memory than dense attention. In the TinyShakespeare experiment, neither sparse pattern caused the expected large loss increase: sliding-window achieved the best validation loss of the three, while block-sparse stayed close to dense. That combination suggests that the practical value of sparse attention is a tradeoff between interaction coverage, implementation overhead, and the amount of context the task actually needs.

## 7. Limitations

There are several limitations to this experiment. The benchmark is a local CPU measurement rather than a production GPU sparse-attention kernel benchmark. The GPT is deliberately small, so the quality results should not be generalized to large language models. The sparse implementations are also not equivalent to a highly optimized library kernel: Python/PyTorch indexing overhead can dominate at smaller sequence lengths.
